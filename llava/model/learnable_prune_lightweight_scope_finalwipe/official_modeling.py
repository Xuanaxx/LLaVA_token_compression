#!/usr/bin/env python3
# coding: utf-8
"""Official LLaVA lightweight learnable-prune + SeededResidualSCOPE + final-wipe inference."""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaForCausalLM, apply_rotary_pos_emb

try:
    from transformers.models.llama.modeling_llama import create_causal_mask as llama_create_causal_mask
except ImportError:
    llama_create_causal_mask = None

from learnable_pruner_lightweight.predictor import LearnablePrunePredictor
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import get_anyres_image_grid_shape
from llava.model.llava_arch import unpad_image
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM, LlavaLlamaModel


DEFAULT_CHECKPOINT = "/data1/chenzixuan/train_output/official_llava_learnable_prune_lightweight_top64_layer18_sample0.2"
LEARNABLE_TOPK = 64
SCOPE_TARGET_COUNT = 137
MID_PRUNING_LAYER_IDX = 12
MID_TARGET_COUNT = 32
MID_ATTN_ANCHOR = "query"
FINAL_WIPE_LAYER_IDX = 25
ENABLE_FINALWIPE = True
ENABLE_PREDICTOR = True
_SCOPE_GRAPH_CACHE: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()
_CAUSAL_BOOL_MASK_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}
_VISUAL_COORDINATE_CACHE: Dict[Tuple[Any, ...], torch.Tensor] = {}


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return int(default)
    return int(value)


def _env_int_list(name: str, default: List[int]) -> List[int]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return [int(x) for x in default]
    parsed = [int(x) for x in re.findall(r"-?\d+", value)]
    return parsed or [int(x) for x in default]


def _enable_finalwipe(default: Optional[bool] = None) -> bool:
    base = ENABLE_FINALWIPE if default is None else bool(default)
    return _env_flag("ENABLE_FINALWIPE", base)


def _enable_predictor() -> bool:
    return _env_flag("ENABLE_PREDICTOR", ENABLE_PREDICTOR)


def _predictor_only() -> bool:
    """Select predictor TopK once, then keep those visual tokens through every layer."""
    return _env_flag("LEARNABLE_PRUNE_PREDICTOR_ONLY", False)


def _mid_target_count(default: Optional[int] = None) -> int:
    base = MID_TARGET_COUNT if default is None else int(default)
    return _env_int("MID_TARGET_COUNT", base)


def _learnable_topk(default: Optional[int] = None) -> int:
    base = LEARNABLE_TOPK if default is None else int(default)
    return _env_int("LEARNABLE_TOPK", base)


def _scope_target_count(default: Optional[int] = None) -> int:
    base = SCOPE_TARGET_COUNT if default is None else int(default)
    return _env_int("SCOPE_TARGET_COUNT", base)


def _mid_pruning_layer_idx(default: Optional[int] = None) -> int:
    base = MID_PRUNING_LAYER_IDX if default is None else int(default)
    return _env_int("MID_PRUNING_LAYER_IDX", base)


def _final_wipe_layer_idx() -> int:
    return _env_int("FINAL_WIPE_LAYER_IDX", FINAL_WIPE_LAYER_IDX)


def _final_wipe_layer_idxs(default: Optional[List[int]] = None) -> List[int]:
    base = [FINAL_WIPE_LAYER_IDX] if default is None else [int(x) for x in default]
    return _env_int_list("FINAL_WIPE_LAYER_IDX", base)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_key_value_heads * n_rep, seq_len, head_dim)


def _image_size_tuple(image_sizes: Any, image_idx: int) -> Tuple[int, int]:
    if image_sizes is None:
        raise ValueError("image_sizes is required for LLaVA-NeXT anyres spatial merging.")
    image_size = image_sizes[image_idx]
    if torch.is_tensor(image_size):
        image_size = image_size.detach().cpu().tolist()
    if len(image_size) != 2:
        raise ValueError(f"Expected image_sizes[{image_idx}] to contain (width, height), got: {image_size}")
    return int(image_size[0]), int(image_size[1])


def _normalized_grid_coordinates(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    append_newline: bool = False,
) -> torch.Tensor:
    cache_key = (
        int(height), int(width), bool(append_newline), device.type, device.index, dtype
    )
    cached = _VISUAL_COORDINATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    y = torch.linspace(0.0, 1.0, max(1, int(height)), device=device, dtype=torch.float32)
    x = torch.linspace(0.0, 1.0, max(1, int(width)), device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack((xx, yy), dim=-1)
    if append_newline:
        newline = torch.stack((torch.full_like(y, 1.05), y), dim=-1)[:, None, :]
        coords = torch.cat((coords, newline), dim=1)
    coords = coords.reshape(-1, 2).to(dtype=dtype)
    _VISUAL_COORDINATE_CACHE[cache_key] = coords
    return coords


def _cached_causal_bool_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Return a broadcastable SDPA keep-mask; True means the key is visible."""
    key = (int(device.index or 0), int(seq_len))
    mask = _CAUSAL_BOOL_MASK_CACHE.get(key)
    if mask is None or mask.device != device:
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril_()
        _CAUSAL_BOOL_MASK_CACHE[key] = mask
    return mask


def _seeded_scope_greedy_core(
    visual_feature_vectors: torch.Tensor,
    seed_relative: torch.Tensor,
    target_keep: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed-shape exact SCOPE core, suitable for CUDA graph capture."""
    num_tokens = int(visual_feature_vectors.shape[1])
    norm_vectors = visual_feature_vectors / visual_feature_vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cosine_simi = torch.bmm(norm_vectors, norm_vectors.transpose(1, 2))
    sim = cosine_simi[0]
    selected = torch.zeros(num_tokens, dtype=torch.bool, device=visual_feature_vectors.device)
    selected.scatter_(0, seed_relative, True)
    cur_max = sim.index_select(0, seed_relative).max(dim=0).values
    for _ in range(max(0, int(target_keep) - int(seed_relative.numel()))):
        gains = (sim - cur_max.unsqueeze(0)).clamp_min(0.0).sum(dim=1)
        gains = gains.masked_fill(selected, float("-inf"))
        best_idx = gains.argmax().view(1)
        selected.scatter_(0, best_idx, True)
        cur_max = torch.maximum(cur_max, sim.index_select(0, best_idx)[0])
    # Unlike nonzero(), topk has a static output shape required by CUDA graphs.
    keep = torch.topk(selected.to(torch.uint8), k=int(target_keep)).indices.sort().values.unsqueeze(0)
    return keep, cosine_simi


class _SeededScopeCudaGraph:
    """Replay the exact sequential greedy kernels without Python launch gaps."""

    def __init__(self, visual: torch.Tensor, seeds: torch.Tensor, target_keep: int):
        self.static_visual = torch.empty_like(visual)
        self.static_seeds = torch.empty_like(seeds)
        self.static_visual.copy_(visual)
        self.static_seeds.copy_(seeds)
        self.target_keep = int(target_keep)

        # Warm allocator/library state before capture. This is created during
        # benchmark warmup and therefore excluded from measured samples.
        _seeded_scope_greedy_core(self.static_visual, self.static_seeds, self.target_keep)
        torch.cuda.synchronize(visual.device)
        self.graph = torch.cuda.CUDAGraph()
        # DataLoader pin-memory workers may allocate host buffers while the
        # training thread captures this graph. Thread-local capture mode keeps
        # those unrelated worker allocations from invalidating the capture.
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self.keep, self.cosine = _seeded_scope_greedy_core(
                self.static_visual, self.static_seeds, self.target_keep
            )

    def __call__(self, visual: torch.Tensor, seeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.static_visual.copy_(visual)
        self.static_seeds.copy_(seeds)
        self.graph.replay()
        return self.keep, self.cosine


@torch.no_grad()
def SeededResidualSCOPE(
    visual_feature_vectors: torch.Tensor,
    seed_relative: torch.Tensor,
    target_keep: int,
    spatial_bonus: float = 0.0,
    use_cuda_graph: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if visual_feature_vectors.dim() != 3 or visual_feature_vectors.shape[0] != 1:
        raise ValueError("SeededResidualSCOPE expects visual_feature_vectors with shape [1, N, D].")

    device = visual_feature_vectors.device
    dtype = visual_feature_vectors.dtype
    num_tokens = int(visual_feature_vectors.shape[1])
    target_keep = max(0, min(int(target_keep), num_tokens))

    if target_keep <= 0 or num_tokens <= 0:
        empty = torch.empty((1, 0), dtype=torch.long, device=device)
        cosine_simi = visual_feature_vectors.new_zeros((1, num_tokens, num_tokens))
        return empty, cosine_simi

    seed_relative = seed_relative.to(device=device, dtype=torch.long).flatten()
    if seed_relative.numel() > 0:
        seed_relative = seed_relative[(seed_relative >= 0) & (seed_relative < num_tokens)]
        seed_relative = torch.unique(seed_relative, sorted=False)

    scope_graph_cache_size = max(0, _env_int("LEARNABLE_PRUNE_SCOPE_GRAPH_CACHE_SIZE", 4))
    if (
        use_cuda_graph
        and scope_graph_cache_size > 0
        and visual_feature_vectors.is_cuda
        and seed_relative.numel() > 0
        and spatial_bonus <= 0.0
    ):
        key = (
            visual_feature_vectors.device.index,
            tuple(visual_feature_vectors.shape),
            visual_feature_vectors.dtype,
            int(seed_relative.numel()),
            int(target_keep),
        )
        cached = _SCOPE_GRAPH_CACHE.get(key)
        if cached is None:
            try:
                cached = _SeededScopeCudaGraph(visual_feature_vectors, seed_relative, target_keep)
            except Exception as exc:
                # Some older CUDA/PyTorch combinations cannot capture dynamic
                # tensor indexing. Cache the failure and retain the eager path.
                cached = exc
                if _env_flag("LEARNABLE_PRUNE_PROFILE_INTERNAL", False):
                    print(f"[learnable_prune_scope_graph] capture failed: {exc!r}", flush=True)
            _SCOPE_GRAPH_CACHE[key] = cached
            while len(_SCOPE_GRAPH_CACHE) > scope_graph_cache_size:
                _SCOPE_GRAPH_CACHE.popitem(last=False)
        else:
            _SCOPE_GRAPH_CACHE.move_to_end(key)
        if isinstance(cached, _SeededScopeCudaGraph):
            return cached(visual_feature_vectors, seed_relative)

    norm_vectors = visual_feature_vectors / visual_feature_vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cosine_simi = torch.bmm(norm_vectors, norm_vectors.transpose(1, 2))
    if seed_relative.numel() >= target_keep:
        return seed_relative[:target_keep].sort().values.unsqueeze(0), cosine_simi

    sim = cosine_simi[0]
    selected = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    if seed_relative.numel() > 0:
        selected[seed_relative] = True
        cur_max = sim.index_select(0, seed_relative).max(dim=0).values
    else:
        cur_max = torch.zeros(num_tokens, dtype=dtype, device=device)

    spatial_dist = None
    if spatial_bonus > 0.0:
        side = int(num_tokens**0.5)
        if side * side == num_tokens:
            yy, xx = torch.meshgrid(
                torch.arange(side, device=device),
                torch.arange(side, device=device),
                indexing="ij",
            )
            coords = torch.stack([yy.flatten(), xx.flatten()], dim=-1).float()
            spatial_dist = torch.cdist(coords, coords, p=2)
            spatial_dist = spatial_dist / spatial_dist.max().clamp(min=1e-6)

    # The number of iterations is known from tensor shapes. Avoid synchronizing
    # the GPU on selected.sum().item() before every greedy update.
    for _ in range(max(0, target_keep - int(seed_relative.numel()))):
        unselected = ~selected
        gains = (sim - cur_max.unsqueeze(0)).clamp_min(0.0).sum(dim=1)
        if spatial_dist is not None and bool(selected.any().item()):
            selected_idx = selected.nonzero(as_tuple=False).flatten()
            min_dist_to_selected = spatial_dist.index_select(1, selected_idx).min(dim=1).values
            gains = gains + float(spatial_bonus) * min_dist_to_selected.to(dtype=gains.dtype)
        gains = gains.masked_fill(~unselected, float("-inf"))
        best_idx = gains.argmax()
        selected[best_idx] = True
        cur_max = torch.maximum(cur_max, sim[best_idx])

    return selected.nonzero(as_tuple=False).flatten().sort().values.unsqueeze(0), cosine_simi


class LlavaLearnablePruneLightweightScopeFinalwipeModel(LlavaLlamaModel):
    config_class = LlavaConfig


class LlavaLearnablePruneLightweightScopeFinalwipeForCausalLM(LlavaLlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlavaLlamaForCausalLM, self).__init__(config)
        self.model = LlavaLearnablePruneLightweightScopeFinalwipeModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._scope_finalwipe_applied = False
        self._scope_finalwipe_next_position_id = None
        self.post_init()

    def get_model(self):
        return self.model

    def _device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        param = next(self.parameters())
        return param.device, param.dtype

    def _ensure_learnable_prune_loaded(self) -> None:
        if hasattr(self, "learnable_prune_predictor"):
            return
        checkpoint = os.environ.get("LEARNABLE_PRUNE_CHECKPOINT", DEFAULT_CHECKPOINT)
        self.load_learnable_prune_checkpoint(checkpoint)

    def load_learnable_prune_checkpoint(self, checkpoint_path: str = DEFAULT_CHECKPOINT) -> None:
        checkpoint = Path(checkpoint_path).expanduser()
        if not (checkpoint / "learnable_prune_config.pt").is_file():
            checkpoint_dirs = sorted(
                (path for path in checkpoint.glob("checkpoint-*") if path.is_dir()),
                key=lambda path: int(path.name.rsplit("-", 1)[-1]) if path.name.rsplit("-", 1)[-1].isdigit() else -1,
            )
            if checkpoint_dirs:
                checkpoint = checkpoint_dirs[-1]
        config = torch.load(checkpoint / "learnable_prune_config.pt", map_location="cpu")
        budget_profiles = [dict(profile) for profile in config.get("budget_profiles", [])]
        requested_budget_value = os.environ.get(
            "LEARNABLE_PRUNE_AVG_TOKEN_BUDGET",
            os.environ.get("AVG_TOKEN_BUDGET"),
        )
        requested_budget = (
            int(requested_budget_value)
            if requested_budget_value is not None and requested_budget_value.strip() != ""
            else int(config.get("default_avg_token_budget", -1))
        )
        selected_profile: Dict[str, Any] = {}
        if budget_profiles:
            named_budget_profiles = [
                profile for profile in budget_profiles if int(profile.get("avg_token_budget", -1)) >= 0
            ]
            matching = [
                profile
                for profile in named_budget_profiles
                if int(profile.get("avg_token_budget", -1)) == requested_budget
            ]
            if requested_budget_value is not None and named_budget_profiles and not matching:
                supported = sorted(int(profile["avg_token_budget"]) for profile in named_budget_profiles)
                raise ValueError(
                    f"Requested average token budget {requested_budget} is not in checkpoint profiles {supported}."
                )
            selected_profile = matching[0] if matching else budget_profiles[0]
        predictor = LearnablePrunePredictor(
            input_size=int(config["predictor_input_size"]),
            hidden_size=int(config["predictor_hidden_size"]),
            rank=int(config.get("predictor_rank", 128)),
            num_heads=int(config.get("predictor_num_heads", 1)),
            rank_mlp_ratio=int(config.get("predictor_rank_mlp_ratio", 2)),
            use_visual_position=bool(config.get("predictor_use_visual_position", True)),
            use_text_position=bool(config.get("predictor_use_text_position", True)),
        )
        predictor.load_state_dict(torch.load(checkpoint / "predictor.pt", map_location="cpu"))
        device, _ = self._device_dtype()
        predictor.to(device=device, dtype=torch.float32)
        predictor.eval()
        for param in predictor.parameters():
            param.requires_grad = False
        self.learnable_prune_predictor = predictor
        self.learnable_prune_config = dict(config)
        self.learnable_prune_budget_profiles = budget_profiles
        self.learnable_prune_active_budget_profile = dict(selected_profile)
        self.learnable_prune_keep_k = _learnable_topk(
            int(selected_profile.get("keep_k", config.get("keep_k", LEARNABLE_TOPK)))
        )
        self.learnable_prune_scope_target_count = int(
            selected_profile.get("scope_target_count", config.get("scope_target_count", SCOPE_TARGET_COUNT))
        )
        self.learnable_prune_mid_pruning_layer_idx = int(
            selected_profile.get(
                "mid_pruning_layer_idx",
                config.get("mid_pruning_layer_idx", MID_PRUNING_LAYER_IDX),
            )
        )
        self.learnable_prune_mid_target_count = int(
            selected_profile.get("mid_target_count", config.get("mid_target_count", MID_TARGET_COUNT))
        )
        self.learnable_prune_mid_attn_anchor = str(config.get("mid_attn_anchor", MID_ATTN_ANCHOR))
        self.learnable_prune_final_wipe_layer_idx = int(
            selected_profile.get(
                "final_wipe_layer_idx",
                config.get("final_wipe_layer_idx", FINAL_WIPE_LAYER_IDX),
            )
        )
        self.learnable_prune_enable_final_wipe = bool(
            selected_profile.get(
                "enable_final_wipe",
                config.get("enable_final_wipe", ENABLE_FINALWIPE),
            )
        )
        self.learnable_prune_checkpoint = str(checkpoint)
        self.learnable_prune_stats: List[Dict[str, Any]] = []

    def reset_learnable_prune_stats(self) -> None:
        self.learnable_prune_stats = []

    def get_learnable_prune_stats(self) -> List[Dict[str, Any]]:
        return list(getattr(self, "learnable_prune_stats", []))

    @staticmethod
    def _is_initial_prefill_cache(past_key_values: Optional[Any]) -> bool:
        if past_key_values is None:
            return True
        if hasattr(past_key_values, "get_seq_length"):
            return int(past_key_values.get_seq_length()) == 0
        return False

    @staticmethod
    def _set_cache_seen_tokens(cache: object, seen_tokens: int) -> None:
        for attr in ("seen_tokens", "_seen_tokens", "past_seen_tokens", "num_tokens", "_num_tokens"):
            if hasattr(cache, attr):
                try:
                    setattr(cache, attr, int(seen_tokens))
                except Exception:
                    pass

    def _encode_and_merge_image_features(
        self, images: Any, image_sizes: Optional[Any] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Match LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal image merging."""

        if isinstance(images, (list, tuple)) or (torch.is_tensor(images) and images.ndim == 5):
            if isinstance(images, (list, tuple)):
                image_batches = [image.unsqueeze(0) if image.ndim == 3 else image for image in images]
            else:
                image_batches = [image for image in images]

            concat_images = torch.cat([image for image in image_batches], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in image_batches]
            image_features = torch.split(image_features, split_sizes, dim=0)

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            if mm_patch_merge_type == "flat":
                merged = [feature.flatten(0, 1) for feature in image_features]
                side = int(self.get_vision_tower().num_patches_per_side)
                coordinates = [
                    _normalized_grid_coordinates(side, side, device=flat.device, dtype=flat.dtype).repeat(
                        feature.shape[0], 1
                    )
                    for feature, flat in zip(image_features, merged)
                ]
                return merged, coordinates
            if not mm_patch_merge_type.startswith("spatial"):
                raise ValueError(f"Unexpected mm_patch_merge_type: {mm_patch_merge_type}")

            merged_features: List[torch.Tensor] = []
            merged_coordinates: List[torch.Tensor] = []
            vision_tower = self.get_vision_tower()
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    patch_image_feature = image_feature[1:]
                    height = width = vision_tower.num_patches_per_side
                    if height * width != base_image_feature.shape[0]:
                        raise ValueError(
                            f"Unexpected base image feature length {base_image_feature.shape[0]} for {height}x{width} patches."
                        )
                    if image_aspect_ratio != "anyres":
                        raise NotImplementedError(
                            f"Spatial patch merging currently expects image_aspect_ratio='anyres', got {image_aspect_ratio!r}."
                        )
                    image_size = _image_size_tuple(image_sizes, image_idx)
                    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                        image_size,
                        self.config.image_grid_pinpoints,
                        vision_tower.config.image_size,
                    )
                    patch_image_feature = patch_image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    if "unpad" in mm_patch_merge_type:
                        patch_image_feature = patch_image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        patch_image_feature = patch_image_feature.flatten(1, 2).flatten(2, 3)
                        patch_image_feature = unpad_image(patch_image_feature, image_size)
                        patch_height, patch_width = patch_image_feature.shape[-2:]
                        image_newline = self.model.image_newline.to(
                            device=patch_image_feature.device,
                            dtype=patch_image_feature.dtype,
                        )
                        patch_image_feature = torch.cat(
                            (
                                patch_image_feature,
                                image_newline[:, None, None].expand(*patch_image_feature.shape[:-1], 1),
                            ),
                            dim=-1,
                        )
                        patch_image_feature = patch_image_feature.flatten(1, 2).transpose(0, 1)
                        patch_coordinates = _normalized_grid_coordinates(
                            patch_height,
                            patch_width,
                            device=patch_image_feature.device,
                            dtype=patch_image_feature.dtype,
                            append_newline=True,
                        )
                    else:
                        patch_image_feature = patch_image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        patch_image_feature = patch_image_feature.flatten(0, 3)
                        patch_coordinates = _normalized_grid_coordinates(
                            num_patch_height * height,
                            num_patch_width * width,
                            device=patch_image_feature.device,
                            dtype=patch_image_feature.dtype,
                        )
                    image_feature = torch.cat((base_image_feature, patch_image_feature), dim=0)
                    base_coordinates = _normalized_grid_coordinates(
                        height, width, device=image_feature.device, dtype=image_feature.dtype
                    )
                    image_coordinates = torch.cat((base_coordinates, patch_coordinates), dim=0)
                else:
                    image_feature = image_feature[0]
                    height = width = vision_tower.num_patches_per_side
                    if "unpad" in mm_patch_merge_type:
                        image_newline = self.model.image_newline.to(
                            device=image_feature.device,
                            dtype=image_feature.dtype,
                        )
                        image_feature = torch.cat((image_feature, image_newline[None]), dim=0)
                        image_coordinates = _normalized_grid_coordinates(
                            height, width, device=image_feature.device, dtype=image_feature.dtype
                        )
                        image_coordinates = torch.cat(
                            (image_coordinates, image_coordinates.new_tensor([[1.05, 1.0]])), dim=0
                        )
                    else:
                        image_coordinates = _normalized_grid_coordinates(
                            height, width, device=image_feature.device, dtype=image_feature.dtype
                        )
                merged_features.append(image_feature)
                merged_coordinates.append(image_coordinates)
            return merged_features, merged_coordinates

        image_features = self.encode_images(images)
        if torch.is_tensor(image_features):
            merged = [feature for feature in image_features]
        else:
            merged = list(image_features)
        coordinates = []
        for feature in merged:
            side = int(round(feature.shape[0] ** 0.5))
            if side * side == feature.shape[0]:
                coordinates.append(
                    _normalized_grid_coordinates(side, side, device=feature.device, dtype=feature.dtype)
                )
            else:
                x = torch.linspace(0.0, 1.0, feature.shape[0], device=feature.device, dtype=feature.dtype)
                coordinates.append(torch.stack((x, torch.zeros_like(x)), dim=-1))
        return merged, coordinates

    def _embed_multimodal_for_generation(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        image_sizes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model_device, _ = self._device_dtype()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        image_features, image_coordinates = self._encode_and_merge_image_features(images, image_sizes=image_sizes)

        compact_input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        new_input_embeds: List[torch.Tensor] = []
        new_input_ids: List[torch.Tensor] = []
        new_visual_coordinates: List[torch.Tensor] = []
        cur_image_idx = 0
        embed_tokens = self.get_model().embed_tokens

        for cur_input_ids in compact_input_ids:
            num_images = int((cur_input_ids == IMAGE_TOKEN_INDEX).sum().item())
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds = embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds, cur_image_features[0:0].to(cur_input_embeds.dtype)], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_input_ids.append(cur_input_ids)
                new_visual_coordinates.append(cur_input_embeds.new_zeros((cur_input_embeds.shape[0], 2)))
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = [
                cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                for i in range(len(image_token_indices) - 1)
            ]
            split_sizes = [x.shape[0] for x in cur_input_ids_noim]
            text_embeds = embed_tokens(torch.cat(cur_input_ids_noim)) if sum(split_sizes) > 0 else None
            text_embeds_split = torch.split(text_embeds, split_sizes, dim=0) if text_embeds is not None else [
                self.get_input_embeddings().weight.new_zeros((0, self.config.hidden_size))
                for _ in split_sizes
            ]

            cur_new_embeds = []
            cur_new_ids = []
            cur_new_coordinates = []
            for i in range(num_images + 1):
                cur_new_embeds.append(text_embeds_split[i])
                cur_new_ids.append(cur_input_ids_noim[i])
                cur_new_coordinates.append(text_embeds_split[i].new_zeros((text_embeds_split[i].shape[0], 2)))
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx].to(
                        device=text_embeds_split[i].device,
                        dtype=text_embeds_split[i].dtype,
                    )
                    cur_image_idx += 1
                    cur_image_coordinates = image_coordinates[cur_image_idx - 1].to(
                        device=text_embeds_split[i].device, dtype=text_embeds_split[i].dtype
                    )
                    visual_len = cur_image_features.shape[0]
                    cur_new_embeds.append(cur_image_features)
                    cur_new_ids.append(torch.full((visual_len,), IMAGE_TOKEN_INDEX, device=cur_input_ids.device, dtype=cur_input_ids.dtype))
                    cur_new_coordinates.append(cur_image_coordinates)

            new_input_embeds.append(torch.cat([x.to(model_device) for x in cur_new_embeds], dim=0))
            new_input_ids.append(torch.cat(cur_new_ids, dim=0))
            new_visual_coordinates.append(torch.cat(cur_new_coordinates, dim=0).to(model_device))

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        pad_token_id = int(getattr(self.config, "pad_token_id", 0) or 0)
        padded_embeds = []
        padded_ids = torch.full((batch_size, max_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        padded_attention = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        padded_position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        padded_visual_coordinates = new_input_embeds[0].new_zeros((batch_size, max_len, 2))

        for i, (cur_embeds, cur_ids) in enumerate(zip(new_input_embeds, new_input_ids)):
            cur_len = cur_embeds.shape[0]
            padded_embeds.append(
                torch.cat(
                    [
                        cur_embeds,
                        torch.zeros((max_len - cur_len, cur_embeds.shape[1]), dtype=cur_embeds.dtype, device=cur_embeds.device),
                    ],
                    dim=0,
                )
            )
            if cur_len > 0:
                padded_ids[i, :cur_len] = cur_ids
                padded_attention[i, :cur_len] = True
                padded_position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                padded_visual_coordinates[i, :cur_len] = new_visual_coordinates[i]

        return (
            torch.stack(padded_embeds, dim=0),
            padded_ids,
            padded_attention,
            padded_position_ids,
            padded_visual_coordinates,
        )

    @torch.no_grad()
    def _learnable_scope_prune_prefill(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        visual_coordinates: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.shape[0] != 1:
            raise ValueError("learnable-prune scope-finalwipe generation currently expects batch_size=1")
        enable_predictor = _enable_predictor()
        predictor_only = _predictor_only()
        if predictor_only and not enable_predictor:
            raise ValueError("LEARNABLE_PRUNE_PREDICTOR_ONLY=1 requires ENABLE_PREDICTOR=1")
        if enable_predictor:
            self._ensure_learnable_prune_loaded()
        elif not hasattr(self, "learnable_prune_stats"):
            self.learnable_prune_stats = []

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device=input_ids.device)
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(attention_mask.eq(0), 0)
        else:
            position_ids = position_ids.to(device=input_ids.device)

        valid_mask = attention_mask.bool()
        visual_token_mask = input_ids.eq(IMAGE_TOKEN_INDEX) & valid_mask
        visual_positions = visual_token_mask[0].nonzero(as_tuple=False).flatten()
        if visual_positions.numel() == 0:
            valid_positions = attention_mask[0].bool().nonzero(as_tuple=False).flatten()
            return inputs_embeds, input_ids[:, valid_positions], attention_mask, position_ids, valid_positions.unsqueeze(0)

        text_positions = (valid_mask & ~visual_token_mask)[0].nonzero(as_tuple=False).flatten()
        internal_profile = _env_flag("LEARNABLE_PRUNE_PROFILE_INTERNAL", False) and inputs_embeds.is_cuda
        predictor_start = torch.cuda.Event(enable_timing=True) if internal_profile else None
        predictor_end = torch.cuda.Event(enable_timing=True) if internal_profile else None
        scope_end = torch.cuda.Event(enable_timing=True) if internal_profile else None
        if predictor_start is not None:
            predictor_start.record()
        if enable_predictor:
            predictor = self.learnable_prune_predictor
            predictor_dtype = next(predictor.parameters()).dtype
            predictor_inputs = inputs_embeds.to(dtype=predictor_dtype)
            scores = predictor.score_visual_text_by_positions(
                predictor_inputs,
                visual_positions=visual_positions,
                text_positions=text_positions,
                visual_coordinates=(
                    visual_coordinates.index_select(1, visual_positions)
                    if visual_coordinates is not None
                    and bool(self.learnable_prune_config.get("predictor_use_true_visual_coordinates", False))
                    else None
                ),
            )[0]
            learnable_topk = int(getattr(self, "learnable_prune_keep_k", LEARNABLE_TOPK))
            top_indices, top_valid = predictor.budget_topk_indices(
                scores.unsqueeze(0), learnable_topk
            )
            top_relative = top_indices[0, top_valid[0]]
            topk = int(top_relative.numel())
        else:
            scores = inputs_embeds.new_empty((visual_positions.numel(),), dtype=torch.float32)
            topk = 0
            top_relative = torch.empty((0,), dtype=torch.long, device=visual_positions.device)
        if predictor_end is not None:
            predictor_end.record()

        if predictor_only:
            # The ablation applies the predictor budget directly: there is no
            # SCOPE diversity fill, even when the checkpoint's saved scope
            # target is larger than the requested TopK.
            target_keep = topk
            visual_keep_relative = top_relative.sort().values
        else:
            target_keep = min(
                _scope_target_count(getattr(self, "learnable_prune_scope_target_count", SCOPE_TARGET_COUNT)),
                int(visual_positions.numel()),
            )
        if not predictor_only and target_keep > topk:
            visual_source = predictor_inputs if enable_predictor else inputs_embeds
            visual_embeds = visual_source.index_select(1, visual_positions)
            seeded_scope_rank, _ = SeededResidualSCOPE(
                visual_feature_vectors=visual_embeds.to(dtype=torch.float32),
                seed_relative=top_relative,
                target_keep=target_keep,
                spatial_bonus=0.0,
            )
            visual_keep_relative = seeded_scope_rank[0]
        elif not predictor_only:
            visual_keep_relative = top_relative[:target_keep].sort().values
        if scope_end is not None:
            scope_end.record()
            torch.cuda.synchronize(inputs_embeds.device)
            selection_label = "topk_pack" if predictor_only else "scope_fill"
            print(
                "[learnable_prune_scope_profile] "
                f"predictor={predictor_start.elapsed_time(predictor_end):.3f}ms "
                f"{selection_label}={predictor_end.elapsed_time(scope_end):.3f}ms",
                flush=True,
            )

        visual_keep = visual_positions.index_select(0, visual_keep_relative)
        keep_mask = valid_mask[0].clone()
        keep_mask[visual_positions] = False
        keep_mask[visual_keep] = True
        keep_positions = keep_mask.nonzero(as_tuple=False).flatten()
        pruned_embeds = inputs_embeds[:, keep_positions, :]
        pruned_input_ids = input_ids[:, keep_positions]
        pruned_attention_mask = attention_mask.new_ones((1, keep_positions.numel()))
        pruned_position_ids = position_ids[:, keep_positions]
        self.learnable_prune_stats.append(
            {
                "original_tokens": int(input_ids.shape[1]),
                "original_visual_tokens": int(visual_positions.numel()),
                "enable_predictor": bool(enable_predictor),
                "learnable_topk_visual_tokens": int(topk),
                "kept_visual_tokens": int(visual_keep.numel()),
                "initial_target_visual_tokens": int(target_keep),
                "scope_target_visual_tokens": 0 if predictor_only else int(target_keep),
                "predictor_only": bool(predictor_only),
                "scope_enabled": not predictor_only,
                "avg_token_budget_profile": int(
                    getattr(self, "learnable_prune_active_budget_profile", {}).get(
                        "avg_token_budget", -1
                    )
                ),
                "diversity_fill_method": (
                    "none_predictor_topk_only" if predictor_only else "seeded_residual_scope"
                ),
                "pruned_tokens": int(input_ids.shape[1] - keep_positions.numel()),
                "checkpoint": getattr(self, "learnable_prune_checkpoint", None),
            }
        )
        return pruned_embeds, pruned_input_ids, pruned_attention_mask, pruned_position_ids, keep_positions.unsqueeze(0)

    def _build_mid_query_indices(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_positions: torch.Tensor,
        attn_anchor: str,
    ) -> torch.Tensor:
        valid_text_positions = (attention_mask[0].bool() & input_ids[0].ne(IMAGE_TOKEN_INDEX)).nonzero(as_tuple=False).flatten()
        if valid_text_positions.numel() == 0:
            return attention_mask[0].bool().nonzero(as_tuple=False).flatten()[-1:]
        if attn_anchor == "query" and visual_positions.numel() > 0:
            query_positions = valid_text_positions[valid_text_positions > visual_positions.max()]
            if query_positions.numel() > 0:
                return query_positions
        return valid_text_positions[-1:]

    def _get_visual_token_attention_scores(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        scoring_layer_idx: int,
        visual_positions: torch.Tensor,
        query_indices: torch.Tensor,
    ) -> torch.Tensor:
        layers = self.model.layers
        scoring_layer = layers[scoring_layer_idx]
        self_attn = scoring_layer.self_attn
        bsz, seq_len, hidden_size = hidden_states.shape
        if bsz != 1:
            raise ValueError("Importance scoring expects a single sample.")
        if visual_positions.numel() == 0 or query_indices.numel() == 0:
            return hidden_states.new_zeros((0, visual_positions.numel()))

        layer_device = next(scoring_layer.parameters()).device
        hidden_states = hidden_states.to(device=layer_device)
        attention_mask = attention_mask.to(device=layer_device)
        position_ids = position_ids.to(device=layer_device)
        visual_positions = visual_positions.to(dtype=torch.long, device=layer_device)
        query_indices = query_indices.to(dtype=torch.long, device=layer_device)
        hidden_normed = scoring_layer.input_layernorm(hidden_states)
        num_heads = getattr(self_attn.config, "num_attention_heads", None)
        if num_heads is None:
            num_heads = getattr(self_attn, "num_heads")
        num_heads = int(num_heads)
        num_kv_heads = int(getattr(self_attn.config, "num_key_value_heads", num_heads))
        head_dim = int(self_attn.head_dim)
        scaling = float(getattr(self_attn, "scaling", head_dim ** -0.5))

        query_states = self_attn.q_proj(hidden_normed).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = self_attn.k_proj(hidden_normed).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = self_attn.v_proj(hidden_normed).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        cos, sin = self.model.rotary_emb(hidden_states, position_ids)
        scoring_device = query_states.device
        key_states = key_states.to(device=scoring_device)
        value_states = value_states.to(device=scoring_device)
        cos = cos.to(device=scoring_device)
        sin = sin.to(device=scoring_device)
        attention_mask = attention_mask.to(device=scoring_device)
        visual_positions = visual_positions.to(device=scoring_device)
        q_idx = query_indices.to(device=scoring_device)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = _repeat_kv(key_states, num_heads // num_kv_heads)
        value_states = _repeat_kv(value_states, num_heads // num_kv_heads)

        key_positions = torch.arange(seq_len, device=scoring_device).view(1, 1, 1, seq_len)
        key_valid = attention_mask[:, None, None, :].bool()
        v_visual = value_states.index_select(2, visual_positions)[0]
        q_states = query_states.index_select(2, q_idx)
        attn_scores = torch.matmul(q_states, key_states.transpose(2, 3)) * scaling
        causal = key_positions <= q_idx.view(1, 1, -1, 1)
        attn_scores = attn_scores.float().masked_fill(~(causal & key_valid), torch.finfo(torch.float32).min)
        attn_probs = torch.softmax(attn_scores, dim=-1).to(dtype=q_states.dtype)
        z_heads = torch.matmul(attn_probs, value_states)[0].permute(1, 0, 2).contiguous()

        alpha_visual = attn_probs.index_select(-1, visual_positions)[0].permute(1, 0, 2).contiguous()
        beta = alpha_visual / (1.0 - alpha_visual).clamp(min=1e-6)
        delta_z = beta.unsqueeze(-1) * (z_heads.unsqueeze(2) - v_visual.unsqueeze(0))
        delta_z_cat = delta_z.permute(0, 2, 1, 3).contiguous().view(-1, num_heads * head_dim)
        delta_y = self_attn.o_proj(delta_z_cat).view(q_idx.numel(), -1, hidden_size)
        return delta_y.norm(dim=-1)

    def _prune_by_mid_attention_scores(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        importance_scores: torch.Tensor,
        target_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = input_ids.to(device=hidden_states.device)
        attention_mask = attention_mask.to(device=hidden_states.device)
        position_ids = position_ids.to(device=hidden_states.device)
        visual_positions = input_ids[0].eq(IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
        target_count = min(max(int(target_count), 0), int(visual_positions.numel()))
        if visual_positions.numel() > target_count:
            keep_mask = input_ids[0].ne(IMAGE_TOKEN_INDEX)
            if target_count > 0:
                top_relative = torch.topk(importance_scores, k=target_count).indices.sort().values
                visual_keep = visual_positions.index_select(0, top_relative)
                keep_mask[visual_keep] = True
            keep_positions = keep_mask.nonzero(as_tuple=False).flatten()
        else:
            keep_positions = attention_mask[0].bool().nonzero(as_tuple=False).flatten()
        return (
            hidden_states.index_select(1, keep_positions),
            input_ids.index_select(1, keep_positions),
            attention_mask.index_select(1, keep_positions),
            position_ids.index_select(1, keep_positions),
            keep_positions.unsqueeze(0),
        )

    def _wipe_visual_tokens(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = input_ids.to(device=hidden_states.device)
        attention_mask = attention_mask.to(device=hidden_states.device)
        position_ids = position_ids.to(device=hidden_states.device)
        visual_mask = input_ids[0].eq(IMAGE_TOKEN_INDEX)
        keep_positions = (~visual_mask).nonzero(as_tuple=False).flatten()
        return (
            hidden_states.index_select(1, keep_positions),
            input_ids.index_select(1, keep_positions),
            attention_mask.index_select(1, keep_positions),
            position_ids.index_select(1, keep_positions),
            keep_positions.unsqueeze(0),
        )

    def _prepare_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        cache_position: Optional[torch.Tensor] = None,
    ):
        if attention_mask is None:
            return None
        attention_mask = attention_mask.to(device=inputs_embeds.device)
        if position_ids is not None:
            position_ids = position_ids.to(device=inputs_embeds.device)
        if cache_position is not None:
            cache_position = cache_position.to(device=inputs_embeds.device)
        if llama_create_causal_mask is not None and position_ids is not None and cache_position is not None:
            return llama_create_causal_mask(
                config=self.model.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        if hasattr(self.model, "_prepare_decoder_attention_mask"):
            past_len = 0
            if isinstance(past_key_values, Cache):
                try:
                    past_len = int(past_key_values.get_seq_length())
                except Exception:
                    past_len = 0
            elif past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
                past_len = int(past_key_values[0][0].shape[-2])
            return self.model._prepare_decoder_attention_mask(
                attention_mask,
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                inputs_embeds,
                past_len,
            )
        return attention_mask

    def _run_layer(
        self,
        layer,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_value=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if position_embeddings is None and hasattr(self.model, "rotary_emb"):
            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        if (
            _env_flag("LEARNABLE_PRUNE_DIRECT_SDPA", True)
            and str(getattr(self.model.config, "_attn_implementation", "eager")) == "sdpa"
            and not output_attentions
        ):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            self_attn = layer.self_attn
            bsz, query_len, _ = normed.shape
            head_dim = int(self_attn.head_dim)
            num_heads = int(self_attn.config.num_attention_heads)
            num_kv_heads = int(self_attn.config.num_key_value_heads)
            query_states = self_attn.q_proj(normed).view(bsz, query_len, num_heads, head_dim).transpose(1, 2)
            key_states = self_attn.k_proj(normed).view(bsz, query_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = self_attn.v_proj(normed).view(bsz, query_len, num_kv_heads, head_dim).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self_attn.layer_idx, cache_kwargs
                )
            key_states = _repeat_kv(key_states, num_heads // num_kv_heads)
            value_states = _repeat_kv(value_states, num_heads // num_kv_heads)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=query_len > 1,
                scale=float(self_attn.scaling),
            )
            attn_output = attn_output.transpose(1, 2).reshape(bsz, query_len, -1).contiguous()
            hidden_states = residual + self_attn.o_proj(attn_output)
            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))
            return hidden_states, None, None
        outputs = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            cache_position=cache_position,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
        )
        if torch.is_tensor(outputs):
            return outputs, None, None
        next_hidden = outputs[0]
        attn = outputs[1] if output_attentions and len(outputs) > 1 else None
        present = outputs[-1] if use_cache else None
        return next_hidden, attn, present

    def _manual_decode(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_values=None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = past_key_values if isinstance(past_key_values, Cache) else ([] if use_cache else None)
        if cache_position is None and position_ids is not None and position_ids.shape[0] == 1:
            cache_position = position_ids.squeeze(0)
        causal_mask = None if hidden_states.shape[1] == 1 else self._prepare_mask(
            attention_mask,
            hidden_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.model.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_past = past_key_values if isinstance(past_key_values, Cache) else (
                past_key_values[layer_idx] if past_key_values is not None else None
            )
            hidden_states, attn, present = self._run_layer(
                layer,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_value=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if use_cache:
                if isinstance(next_cache, list):
                    next_cache.append(present)
            if output_attentions:
                all_attns = all_attns + (attn,)

        hidden_states = self.model.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return hidden_states, (tuple(next_cache) if isinstance(next_cache, list) else next_cache), all_hidden_states, all_attns

    def _prefill_with_finalwipe(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        visual_coordinates: Optional[torch.Tensor],
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        attn_anchor: Optional[str] = None,
        mid_target_count: Optional[int] = None,
    ):
        profile_internal = _env_flag("LEARNABLE_PRUNE_PROFILE_INTERNAL", False)
        profile_events: List[Tuple[str, torch.cuda.Event]] = []
        if profile_internal and inputs_embeds.is_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            profile_events.append(("start", event))

        def mark_profile(name: str) -> None:
            if profile_internal and inputs_embeds.is_cuda:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                profile_events.append((name, event))

        inputs_embeds, scoped_input_ids, attention_mask, position_ids, scoped_keep_positions = self._learnable_scope_prune_prefill(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_coordinates=visual_coordinates,
        )
        predictor_only = _predictor_only()
        mark_profile("predictor_topk" if predictor_only else "scope")

        if predictor_only:
            # Predictor-only ablation: the selected TopK is the complete and
            # only compression stage. Run the compact sequence through every
            # decoder layer without SCOPE, mid-layer scoring/pruning, or a
            # final visual-token wipe.
            cache_position = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long
            )
            initial_cache = DynamicCache(config=self.model.config) if use_cache else None
            hidden_states, next_cache, all_hidden_states, all_attns = self._manual_decode(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=initial_cache,
                cache_position=cache_position,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            mark_profile("all_layers")

            kept_visual_tokens = int(
                scoped_input_ids[0].eq(IMAGE_TOKEN_INDEX).sum().item()
            )
            if getattr(self, "learnable_prune_stats", None):
                self.learnable_prune_stats[-1].update(
                    {
                        "pruning_mode": "predictor_topk_only",
                        "scope_enabled": False,
                        "mid_layer_prune_enabled": False,
                        "enable_finalwipe": False,
                        "avg_visual_tokens_budget": float(kept_visual_tokens),
                        "mid_pruned_visual_tokens": 0,
                        "final_wiped_visual_tokens": 0,
                        "final_sequence_tokens": int(attention_mask.shape[1]),
                    }
                )
                if os.environ.get("LEARNABLE_PRUNE_DEBUG", "").strip().lower() in {
                    "1", "true", "yes", "on"
                }:
                    print(
                        "[learnable_prune_lightweight_predictor_only] "
                        f"stats={self.learnable_prune_stats[-1]}",
                        flush=True,
                    )

            if next_cache is not None:
                self._set_cache_seen_tokens(next_cache, hidden_states.size(1))
            self._scope_finalwipe_applied = True
            self._scope_finalwipe_next_position_id = int(input_ids.shape[1])
            if len(profile_events) > 1:
                torch.cuda.synchronize(inputs_embeds.device)
                segments = {
                    profile_events[i][0]: profile_events[i - 1][1].elapsed_time(profile_events[i][1])
                    for i in range(1, len(profile_events))
                }
                print(f"[learnable_prune_profile] {segments}", flush=True)
            return (
                hidden_states,
                next_cache,
                all_hidden_states,
                all_attns,
                scoped_input_ids,
                attention_mask,
                scoped_keep_positions,
            )

        layers = self.model.layers
        enable_finalwipe = _enable_finalwipe(
            getattr(self, "learnable_prune_enable_final_wipe", ENABLE_FINALWIPE)
        )
        final_wipe_layer_idxs = _final_wipe_layer_idxs(
            [getattr(self, "learnable_prune_final_wipe_layer_idx", FINAL_WIPE_LAYER_IDX)]
        )
        wipe_layer_idx = min(min(final_wipe_layer_idxs), len(layers)) if enable_finalwipe else len(layers)
        mid_pruning_layer_idx = min(
            _mid_pruning_layer_idx(
                getattr(self, "learnable_prune_mid_pruning_layer_idx", MID_PRUNING_LAYER_IDX)
            ),
            wipe_layer_idx,
        )
        mid_scoring_layer_idx = min(mid_pruning_layer_idx, max(len(layers) - 1, 0))
        mid_attn_anchor = str(
            getattr(self, "learnable_prune_mid_attn_anchor", MID_ATTN_ANCHOR)
            if attn_anchor is None
            else attn_anchor
        ).strip().lower()
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = DynamicCache(config=self.model.config) if use_cache else None
        # Scope/mid/final pruning below always packs batch=1 into a dense,
        # padding-free sequence. SDPA/FA can express the identical triangular
        # mask through is_causal=True when attention_mask=None, avoiding three
        # materialized 4D masks and enabling the fused causal kernel.
        attn_impl = str(getattr(self.model.config, "_attn_implementation", "eager"))
        use_implicit_causal = (
            _env_flag("LEARNABLE_PRUNE_IMPLICIT_CAUSAL", False)
            and attn_impl in {"sdpa", "flash_attention_2"}
        )
        use_cached_bool_mask = (
            _env_flag("LEARNABLE_PRUNE_CACHED_BOOL_MASK", False)
            and attn_impl == "sdpa"
        )

        pre_cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
        pre_mask = (
            None if use_implicit_causal
            else _cached_causal_bool_mask(hidden_states.shape[1], hidden_states.device) if use_cached_bool_mask
            else self._prepare_mask(attention_mask, hidden_states, position_ids=position_ids, cache_position=pre_cache_position)
        )
        pre_position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        for layer_idx in range(mid_pruning_layer_idx):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states, attn, present = self._run_layer(
                layers[layer_idx],
                hidden_states,
                pre_mask,
                position_ids,
                past_key_value=next_cache,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=pre_cache_position,
                position_embeddings=pre_position_embeddings,
            )
            if use_cache:
                if isinstance(next_cache, list):
                    next_cache.append(present)
            if output_attentions:
                all_attns = all_attns + (attn,)
        mark_profile("early_layers")

        mid_visual_positions = scoped_input_ids[0].eq(IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
        mid_query_indices = self._build_mid_query_indices(
            input_ids=scoped_input_ids,
            attention_mask=attention_mask,
            visual_positions=mid_visual_positions,
            attn_anchor=mid_attn_anchor,
        )
        if mid_visual_positions.numel() > 0:
            mid_importance_scores = self._get_visual_token_attention_scores(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                scoring_layer_idx=mid_scoring_layer_idx,
                visual_positions=mid_visual_positions,
                query_indices=mid_query_indices,
            ).mean(dim=0)
        else:
            mid_importance_scores = hidden_states.new_zeros((0,))
        mark_profile("mid_score")
        mid_target_default = (
            getattr(self, "learnable_prune_mid_target_count", MID_TARGET_COUNT)
            if mid_target_count is None
            else int(mid_target_count)
        )
        mid_target_count_value = _mid_target_count(mid_target_default)
        hidden_states, mid_input_ids, mid_attention_mask, mid_position_ids, mid_scoped_keep_positions = self._prune_by_mid_attention_scores(
            hidden_states=hidden_states,
            input_ids=scoped_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            importance_scores=mid_importance_scores,
            target_count=mid_target_count_value,
        )
        scoped_keep_positions = scoped_keep_positions.to(device=mid_scoped_keep_positions.device)
        mid_keep_positions_in_original = scoped_keep_positions.index_select(1, mid_scoped_keep_positions[0])
        mid_pruned_visual_tokens = int(attention_mask.shape[1] - mid_attention_mask.shape[1])
        mid_cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
        mid_mask = (
            None if use_implicit_causal
            else _cached_causal_bool_mask(hidden_states.shape[1], hidden_states.device) if use_cached_bool_mask
            else self._prepare_mask(mid_attention_mask, hidden_states, position_ids=mid_position_ids, cache_position=mid_cache_position)
        )
        mid_position_embeddings = self.model.rotary_emb(hidden_states, mid_position_ids)
        for layer_idx in range(mid_pruning_layer_idx, wipe_layer_idx):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states, attn, present = self._run_layer(
                layers[layer_idx],
                hidden_states,
                mid_mask,
                mid_position_ids,
                past_key_value=next_cache,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=mid_cache_position,
                position_embeddings=mid_position_embeddings,
            )
            if use_cache:
                if isinstance(next_cache, list):
                    next_cache.append(present)
            if output_attentions:
                all_attns = all_attns + (attn,)
        mark_profile("mid_layers")

        if enable_finalwipe:
            hidden_states, final_input_ids, final_attention_mask, final_position_ids, wiped_keep_positions = self._wipe_visual_tokens(
                hidden_states=hidden_states,
                input_ids=mid_input_ids,
                attention_mask=mid_attention_mask,
                position_ids=mid_position_ids,
            )
            mid_keep_positions_in_original = mid_keep_positions_in_original.to(device=wiped_keep_positions.device)
            final_keep_positions = mid_keep_positions_in_original.index_select(1, wiped_keep_positions[0])
            final_wiped_visual_tokens = int(mid_attention_mask.shape[1] - final_attention_mask.shape[1])
        else:
            final_input_ids = mid_input_ids
            final_attention_mask = mid_attention_mask
            final_position_ids = mid_position_ids
            final_keep_positions = mid_keep_positions_in_original
            final_wiped_visual_tokens = 0
        if getattr(self, "learnable_prune_stats", None):
            stats = self.learnable_prune_stats[-1]
            scope_visual_tokens = int(stats.get("scope_target_visual_tokens", 0))
            mid_visual_tokens = int(min(max(int(mid_target_count_value), 0), int(mid_visual_positions.numel())))
            avg_visual_tokens_budget = (
                mid_pruning_layer_idx * scope_visual_tokens
                + (wipe_layer_idx - mid_pruning_layer_idx) * mid_visual_tokens
            ) / max(len(layers), 1)
            self.learnable_prune_stats[-1].update(
                {
                    "enable_finalwipe": bool(enable_finalwipe),
                    "mid_pruning_layer_idx": int(mid_pruning_layer_idx),
                    "mid_scoring_layer_idx": int(mid_scoring_layer_idx),
                    "mid_attn_anchor": mid_attn_anchor,
                    "mid_query_count": int(mid_query_indices.numel()),
                    "mid_target_visual_tokens": int(mid_visual_tokens),
                    "avg_visual_tokens_budget": float(avg_visual_tokens_budget),
                    "mid_pruned_visual_tokens": int(mid_pruned_visual_tokens),
                    "mid_sequence_tokens": int(mid_attention_mask.shape[1]),
                    "final_wipe_layer_idx": int(wipe_layer_idx),
                    "final_wipe_layer_idxs": [int(x) for x in final_wipe_layer_idxs],
                    "final_wiped_visual_tokens": int(final_wiped_visual_tokens),
                    "final_sequence_tokens": int(final_attention_mask.shape[1]),
                }
            )
            if os.environ.get("LEARNABLE_PRUNE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
                print(f"[learnable_prune_lightweight_scope_finalwipe] stats={self.learnable_prune_stats[-1]}", flush=True)

        post_cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
        post_mask = (
            None if use_implicit_causal
            else _cached_causal_bool_mask(hidden_states.shape[1], hidden_states.device) if use_cached_bool_mask
            else self._prepare_mask(final_attention_mask, hidden_states, position_ids=final_position_ids, cache_position=post_cache_position)
        )
        post_position_embeddings = self.model.rotary_emb(hidden_states, final_position_ids)
        for layer_idx in range(wipe_layer_idx, len(layers)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states, attn, present = self._run_layer(
                layers[layer_idx],
                hidden_states,
                post_mask,
                final_position_ids,
                past_key_value=next_cache,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=post_cache_position,
                position_embeddings=post_position_embeddings,
            )
            if use_cache:
                if isinstance(next_cache, list):
                    next_cache.append(present)
            if output_attentions:
                all_attns = all_attns + (attn,)
        mark_profile("late_layers")

        hidden_states = self.model.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if next_cache is not None:
            self._set_cache_seen_tokens(next_cache, hidden_states.size(1))
        self._scope_finalwipe_applied = True
        self._scope_finalwipe_next_position_id = int(input_ids.shape[1])
        if len(profile_events) > 1:
            torch.cuda.synchronize(inputs_embeds.device)
            segments = {
                profile_events[i][0]: profile_events[i - 1][1].elapsed_time(profile_events[i][1])
                for i in range(1, len(profile_events))
            }
            print(f"[learnable_prune_profile] {segments}", flush=True)
        return (
            hidden_states,
            (tuple(next_cache) if isinstance(next_cache, list) else next_cache),
            all_hidden_states,
            all_attns,
            final_input_ids,
            final_attention_mask,
            final_keep_positions,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        learnable_prune_input_ids: Optional[torch.LongTensor] = None,
        learnable_prune_visual_coordinates: Optional[torch.Tensor] = None,
        learnable_prune: bool = True,
        attn_anchor: str = MID_ATTN_ANCHOR,
        mid_target_count: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if labels is not None or images is not None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                images=images,
                image_sizes=image_sizes,
                return_dict=return_dict,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        is_initial_prefill_cache = self._is_initial_prefill_cache(past_key_values)
        if is_initial_prefill_cache:
            self._scope_finalwipe_applied = False
            self._scope_finalwipe_next_position_id = None

        should_prune = (
            learnable_prune
            and not self.training
            and learnable_prune_input_ids is not None
            and inputs_embeds.shape[0] == 1
            and inputs_embeds.shape[1] > 1
            and is_initial_prefill_cache
        )

        pruned_input_ids = None
        pruned_attention_mask = None
        pruned_keep_positions = None
        if should_prune:
            (
                hidden_states,
                past_key_values,
                all_hidden_states,
                all_attns,
                pruned_input_ids,
                pruned_attention_mask,
                pruned_keep_positions,
            ) = self._prefill_with_finalwipe(
                input_ids=learnable_prune_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                visual_coordinates=learnable_prune_visual_coordinates,
                use_cache=bool(use_cache),
                output_attentions=bool(output_attentions),
                output_hidden_states=bool(output_hidden_states),
                attn_anchor=attn_anchor,
                mid_target_count=mid_target_count,
            )
        elif getattr(self, "_scope_finalwipe_applied", False) and past_key_values is not None:
            current_seq_len = inputs_embeds.shape[1]
            if position_ids is None:
                if attention_mask is not None:
                    compact_position_ids = attention_mask.long().cumsum(dim=-1) - 1
                    compact_position_ids = compact_position_ids.masked_fill(attention_mask.eq(0), 1)
                    position_ids = compact_position_ids[:, -current_seq_len:]
                elif hasattr(past_key_values, "get_seq_length"):
                    past_seq_len = int(past_key_values.get_seq_length())
                    position_ids = torch.arange(
                        past_seq_len,
                        past_seq_len + current_seq_len,
                        device=inputs_embeds.device,
                        dtype=torch.long,
                    ).unsqueeze(0)

            if position_ids is None:
                position_ids = torch.arange(
                    current_seq_len,
                    device=inputs_embeds.device,
                    dtype=torch.long,
                ).unsqueeze(0)

            decode_cache_position = position_ids.squeeze(0)
            next_position_id = getattr(self, "_scope_finalwipe_next_position_id", None)
            if next_position_id is not None:
                position_ids = torch.arange(
                    int(next_position_id),
                    int(next_position_id) + current_seq_len,
                    device=inputs_embeds.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                self._scope_finalwipe_next_position_id = int(next_position_id) + current_seq_len
            elif cache_position is not None:
                position_ids = cache_position.unsqueeze(0)
            hidden_states, past_key_values, all_hidden_states, all_attns = self._manual_decode(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=decode_cache_position,
                use_cache=bool(use_cache),
                output_attentions=bool(output_attentions),
                output_hidden_states=bool(output_hidden_states),
            )
        else:
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            all_hidden_states = outputs.hidden_states
            all_attns = outputs.attentions

        if isinstance(logits_to_keep, int):
            logits_hidden_states = hidden_states[:, -logits_to_keep:, :] if logits_to_keep > 0 else hidden_states
        else:
            logits_hidden_states = hidden_states[:, logits_to_keep, :]
        logits = self.lm_head(logits_hidden_states)
        loss = None
        if not return_dict:
            output = (logits, past_key_values, all_hidden_states, all_attns)
            return (loss,) + output if loss is not None else output

        causal_output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )
        if pruned_attention_mask is not None:
            causal_output.pruned_input_ids = pruned_input_ids
            causal_output.pruned_attention_mask = pruned_attention_mask
            causal_output.pruned_keep_positions = pruned_keep_positions
        return causal_output

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, **kwargs)
        pruned_attention_mask = getattr(outputs, "pruned_attention_mask", None)
        if pruned_attention_mask is not None:
            model_kwargs["_pruning_done"] = True
            model_kwargs["learnable_prune_input_ids"] = None
            model_kwargs["learnable_prune_visual_coordinates"] = None
            model_kwargs["attention_mask"] = torch.cat(
                [pruned_attention_mask, pruned_attention_mask.new_ones((pruned_attention_mask.shape[0], 1))],
                dim=-1,
            )
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        learnable_prune_input_ids=None,
        learnable_prune_visual_coordinates=None,
        attn_anchor=None,
        mid_target_count=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if self._is_initial_prefill_cache(past_key_values) and learnable_prune_input_ids is not None:
            model_inputs["learnable_prune_input_ids"] = learnable_prune_input_ids
            model_inputs["learnable_prune_visual_coordinates"] = learnable_prune_visual_coordinates
            if attn_anchor is not None:
                model_inputs["attn_anchor"] = attn_anchor
            if mid_target_count is not None:
                model_inputs["mid_target_count"] = mid_target_count
        return model_inputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None and inputs is not None and inputs.shape[1] > 1:
            (
                inputs_embeds,
                expanded_input_ids,
                attention_mask,
                position_ids,
                visual_coordinates,
            ) = self._embed_multimodal_for_generation(
                inputs,
                images,
                attention_mask,
                position_ids,
                image_sizes=image_sizes,
            )
            return LlamaForCausalLM.generate(
                self,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                learnable_prune_input_ids=expanded_input_ids,
                learnable_prune_visual_coordinates=visual_coordinates,
                **kwargs,
            )

        return super().generate(
            inputs=inputs,
            images=images,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )


LlavaForConditionalGeneration = LlavaLearnablePruneLightweightScopeFinalwipeForCausalLM
