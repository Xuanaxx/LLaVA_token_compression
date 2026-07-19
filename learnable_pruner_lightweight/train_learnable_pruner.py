#!/usr/bin/env python3
# coding: utf-8
"""Stage-aligned JS-divergence training for a lightweight LLaVA visual-token pruner.

This trainer implements a single integrated objective:

    L = w_js * JS(full, soft-pruned)
      + w_rank * TopK precision structured hinge
      + w_ce * CE(labels, soft-pruned)

The selector teacher is computed from one fixed decoder layer, controlled by
``--teacher_layer`` and defaulting to layer 18.  This version intentionally does
not run the dynamic candidate-layer search used by the original trainer.

By default the student branch mirrors inference in one segmented decoder pass:
predictor TopK seeds are expanded by SeededResidualSCOPE before layer 0, visual
tokens are pruned again from mid-layer attention importance, and all remaining
visual tokens are removed at the final-wipe layer.  Physical packing keeps the
student activations small.  A fixed-budget straight-through gate on the retained
visual embeddings supplies an end-to-end surrogate gradient to the predictor.

Multiple deployment budgets can share one predictor: a deterministic shuffled
cycle activates one complete pruning profile per microbatch on every DDP rank.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import logging
import math
import os
import random
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import tokenizers
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PretrainedConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers import Trainer, TrainingArguments
from packaging import version
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from .predictor import LearnablePrunePredictor  # type: ignore  # noqa: E402
except ImportError:
    from predictor import LearnablePrunePredictor  # noqa: E402


from llava import conversation as conversation_lib  # noqa: E402
from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX  # noqa: E402
from llava.mm_utils import get_anyres_image_grid_shape, process_anyres_image, tokenizer_image_token  # noqa: E402
from llava.model.llava_arch import unpad_image  # noqa: E402
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM  # noqa: E402
from llava.model.learnable_prune_lightweight_scope_finalwipe.official_modeling import (  # noqa: E402
    SeededResidualSCOPE as _inference_seeded_residual_scope,
)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")
_VISUAL_COORDINATE_CACHE: Dict[Tuple[Any, ...], torch.Tensor] = {}


def validate_distributed_cuda_mapping() -> None:
    """Fail before model loading when local ranks would alias visible GPUs."""

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if local_world_size <= 1:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible_devices = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if visible_devices and len(set(visible_devices)) != len(visible_devices):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES contains duplicate entries, which would map multiple DDP ranks "
            f"to one GPU: {visible_devices}."
        )

    visible_count = torch.cuda.device_count()
    if visible_count < local_world_size:
        raise RuntimeError(
            "Distributed launch requested more local processes than visible CUDA devices: "
            f"LOCAL_WORLD_SIZE={local_world_size}, torch.cuda.device_count()={visible_count}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}. "
            "Accelerate maps local ranks modulo the visible-device count, which would duplicate GPUs."
        )
    if not 0 <= local_rank < local_world_size:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside [0, {local_world_size - 1}]."
        )


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def parse_bool_flag(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def parse_budget_profiles(value: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Parse semicolon-separated multi-budget profiles.

    Each entry is
    ``avg:topk:scope:mid_layer:mid_target:final_layer[:enable_finalwipe]``.
    Example: ``160:272:340:12:80:25;320:544:680:12:160:25``.
    """

    if value is None or str(value).strip() == "":
        return None
    profiles: List[Dict[str, Any]] = []
    for raw_profile in str(value).split(";"):
        parts = [part.strip() for part in raw_profile.split(":")]
        if len(parts) not in {6, 7}:
            raise argparse.ArgumentTypeError(
                "Each budget profile must contain 6 or 7 colon-separated values: "
                "avg:topk:scope:mid_layer:mid_target:final_layer[:enable_finalwipe]. "
                f"Got {raw_profile!r}."
            )
        try:
            avg, topk, scope, mid_layer, mid_target, final_layer = map(int, parts[:6])
            enable_finalwipe = parse_bool_flag(parts[6]) if len(parts) == 7 else True
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"Invalid budget profile {raw_profile!r}: {exc}") from exc
        profiles.append(
            {
                "avg_token_budget": avg,
                "keep_k": topk,
                "scope_target_count": scope,
                "mid_pruning_layer_idx": mid_layer,
                "mid_target_count": mid_target,
                "final_wipe_layer_idx": final_layer,
                "enable_final_wipe": bool(enable_finalwipe),
            }
        )
    if len({profile["avg_token_budget"] for profile in profiles}) != len(profiles):
        raise argparse.ArgumentTypeError("avg_token_budget values must be unique within --budget_profiles.")
    return profiles


def _canonical_budget_profiles(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not profiles:
        raise ValueError("At least one budget profile is required.")
    canonical = []
    required = {
        "avg_token_budget",
        "keep_k",
        "scope_target_count",
        "mid_pruning_layer_idx",
        "mid_target_count",
        "final_wipe_layer_idx",
    }
    for profile in profiles:
        missing = required - set(profile)
        if missing:
            raise ValueError(f"Budget profile is missing required fields: {sorted(missing)}")
        canonical.append(
            {
                "avg_token_budget": int(profile["avg_token_budget"]),
                "keep_k": int(profile["keep_k"]),
                "scope_target_count": int(profile["scope_target_count"]),
                "mid_pruning_layer_idx": int(profile["mid_pruning_layer_idx"]),
                "mid_target_count": int(profile["mid_target_count"]),
                "final_wipe_layer_idx": int(profile["final_wipe_layer_idx"]),
                "enable_final_wipe": bool(profile.get("enable_final_wipe", True)),
            }
        )
    if len({profile["avg_token_budget"] for profile in canonical}) != len(canonical):
        raise ValueError("Budget profile avg_token_budget values must be unique.")
    if any(profile["avg_token_budget"] < 0 for profile in canonical):
        raise ValueError("Budget profile avg_token_budget values must be non-negative.")
    return canonical


class IndexedDataset(Dataset):
    """Attach a stable integer sample_id to each training example."""

    def __init__(self, dataset: Dataset, sample_indices: Optional[List[int]] = None):
        self.dataset = dataset
        self.sample_indices = list(sample_indices) if sample_indices is not None else None

    def __len__(self) -> int:
        return len(self.sample_indices) if self.sample_indices is not None else len(self.dataset)

    def __getitem__(self, index: int):
        source_index = int(self.sample_indices[index]) if self.sample_indices is not None else int(index)
        item = self.dataset[source_index]
        if not isinstance(item, dict):
            raise TypeError("IndexedDataset expects the wrapped dataset to return dict examples.")
        item = dict(item)
        item["_sample_id"] = source_index
        return item


class SimpleDataArguments:
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        image_processor,
        image_aspect_ratio: str = "pad",
        image_grid_pinpoints=None,
    ):
        self.data_path = data_path
        self.image_folder = image_folder
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
        self.image_grid_pinpoints = image_grid_pinpoints
        self.is_multimodal = True
        self.mm_use_im_start_end = False


def preprocess_multimodal(sources: List[List[Dict[str, str]]], data_args: SimpleDataArguments):
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN)
    return sources


def preprocess_v1(sources, tokenizer, has_image: bool = False) -> Dict[str, torch.Tensor]:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            target[:] = IGNORE_INDEX
            print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources, tokenizer, has_image: bool = False) -> Dict[str, torch.Tensor]:
    if not conversation_lib.default_conversation.version.startswith("v1"):
        raise ValueError("This official learnable-pruner trainer currently expects the v1 LLaVA conversation template.")
    return preprocess_v1(sources, tokenizer, has_image=has_image)


def _expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def resolve_image_path(image_root: str, image_field: str) -> str:
    if os.path.isabs(image_field) and os.path.exists(image_field):
        return image_field
    candidate = os.path.join(image_root, image_field)
    if os.path.exists(candidate):
        return candidate
    filename = os.path.basename(image_field)
    candidate = os.path.join(image_root, filename)
    if os.path.exists(candidate):
        return candidate
    known_subdirs = [
        "coco/train2017",
        "gqa/images",
        "ocr_vqa/images",
        "textvqa/train_images",
        "vg/VG_100K",
        "vg/VG_100K_2",
        "VG_100K",
        "VG_100K_2",
    ]
    for subdir in known_subdirs:
        candidate = os.path.join(image_root, subdir, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(image_root, image_field)


def resolve_vision_tower_path(vision_tower: Optional[str]) -> Optional[str]:
    if not vision_tower or os.path.exists(vision_tower):
        return vision_tower
    env_path = os.environ.get("LLAVA_VISION_TOWER")
    if env_path and os.path.exists(env_path):
        return env_path
    cache_candidates = [
        Path.home() / ".cache/huggingface/hub",
        Path.home() / ".cache/huggingface/transformers",
        Path("/data1/zx/.cache/huggingface/transformers"),
        Path("/data1/zx/.cache/huggingface/hub"),
    ]
    repo_dir_name = f"models--{vision_tower.replace('/', '--')}"
    for cache_root in cache_candidates:
        base = cache_root / repo_dir_name / "snapshots"
        if not base.is_dir():
            continue
        snapshots = sorted(path for path in base.iterdir() if (path / "config.json").exists())
        if snapshots:
            return str(snapshots[-1])
    return vision_tower


class LazySupervisedDataset(Dataset):
    def __init__(self, tokenizer, data_path: str, data_args: SimpleDataArguments):
        super().__init__()
        with open(data_path, "r", encoding="utf-8") as f:
            self.list_data_dict = json.load(f)
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_path = resolve_image_path(self.data_args.image_folder, image_file)
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                crop_size = self.data_args.image_processor.crop_size
                h = crop_size["height"] if isinstance(crop_size, dict) else crop_size
                w = crop_size["width"] if isinstance(crop_size, dict) else crop_size
                image = Image.new("RGB", (w, h), (0, 0, 0))
            processor = self.data_args.image_processor
            image_size = tuple(int(x) for x in image.size)
            if self.data_args.image_aspect_ratio == "pad":
                image = _expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            elif self.data_args.image_aspect_ratio == "anyres":
                if self.data_args.image_grid_pinpoints is None:
                    raise ValueError("image_grid_pinpoints is required when --image_aspect_ratio anyres.")
                image = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints)
            else:
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            text_sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            text_sources = copy.deepcopy([e["conversations"] for e in sources])
            crop_size = self.data_args.image_processor.crop_size
            h = crop_size["height"] if isinstance(crop_size, dict) else crop_size
            w = crop_size["width"] if isinstance(crop_size, dict) else crop_size
            image = torch.zeros(3, h, w)
            image_size = (int(w), int(h))

        data_dict = preprocess(text_sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        data_dict["image"] = image
        data_dict["image_size"] = image_size
        return data_dict


class DataCollatorForSupervisedDataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        sample_ids = [int(instance.get("_sample_id", idx)) for idx, instance in enumerate(instances)]
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = [instance["image"] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch["images"] = torch.stack(images)
        else:
            batch["images"] = images
        if all("image_size" in instance for instance in instances):
            batch["image_sizes"] = [tuple(int(x) for x in instance["image_size"]) for instance in instances]
        batch["sample_ids"] = torch.tensor(sample_ids, dtype=torch.long)
        return batch


def _cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _default_checkpoint_contexts():
    """Checkpoint contexts for masks that do not require gradients."""

    return nullcontext(), nullcontext()


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_key_value_heads * n_rep, seq_len, head_dim)


def _make_4d_causal_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    *,
    boolean: bool = False,
) -> torch.Tensor:
    """Build an additive causal mask for a physically packed student batch."""

    _, seq_len = attention_mask.shape
    device = attention_mask.device
    if boolean:
        # PyTorch SDPA interprets True as visible. A bool mask uses half the
        # storage of BF16 (one quarter of FP32) and avoids fill/add kernels.
        causal_keep = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).tril_()
        return causal_keep[None, None, :, :] & attention_mask[:, None, None, :].bool()
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((seq_len, seq_len), min_dtype, device=device, dtype=dtype).triu_(diagonal=1)
    key_bias = torch.zeros_like(attention_mask, dtype=dtype)
    key_bias = key_bias.masked_fill(~attention_mask.bool(), min_dtype)
    # Broadcasting avoids materializing a BxSxS boolean `allowed` tensor and a
    # second dense zero tensor before producing the additive attention mask.
    return causal_mask[None, None, :, :] + key_bias[:, None, None, :]


def _image_size_tuple(image_sizes: Any, image_idx: int) -> Tuple[int, int]:
    if image_sizes is None:
        raise ValueError("image_sizes is required for LLaVA-NeXT anyres spatial merging.")
    image_size = image_sizes[image_idx]
    if torch.is_tensor(image_size):
        image_size = image_size.detach().cpu().tolist()
    elif isinstance(image_size, np.ndarray):
        image_size = image_size.tolist()
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
    """Return row-major normalized (x, y) coordinates matching merged image features."""
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
        # Newline embeddings are not image patches. A small out-of-range x
        # coordinate distinguishes them while preserving their row coordinate.
        newline = torch.stack((torch.full_like(y, 1.05), y), dim=-1)[:, None, :]
        coords = torch.cat((coords, newline), dim=1)
    coords = coords.reshape(-1, 2).to(dtype=dtype)
    _VISUAL_COORDINATE_CACHE[cache_key] = coords
    return coords


@torch.no_grad()
def _seeded_residual_scope_indices(
    visual_feature_vectors: torch.Tensor,
    seed_relative: torch.Tensor,
    target_keep: int,
) -> torch.Tensor:
    """Match inference SeededResidualSCOPE for one unpadded visual sequence.

    The similarity matrix is intentionally row-local and transient.  This avoids
    padding an anyres batch to ``B x Nmax x Nmax`` and bounds peak memory by the
    largest single sample, exactly as the inference implementation does.
    """

    if visual_feature_vectors.dim() != 2:
        raise ValueError("visual_feature_vectors must have shape [N, D].")
    selected, _ = _inference_seeded_residual_scope(
        visual_feature_vectors.unsqueeze(0).float(),
        seed_relative,
        int(target_keep),
        spatial_bonus=0.0,
        # The shared implementation uses thread-local CUDA capture, so pin-memory
        # DataLoader workers can keep running. Fixed-grid batches then replay the
        # complete greedy loop as one graph instead of issuing hundreds of tiny
        # Python-dispatched kernels. CPU and unsupported CUDA paths stay eager.
        use_cuda_graph=True,
    )
    return selected[0]


class _BudgetedSigmoidTopK(torch.autograd.Function):
    """Sigmoid relaxation with an exact TopK budget in the forward value.

    Given logits x and integer k, this returns soft gates m satisfying
    sum_i m_i ~= k by solving for lambda in

        m_i = sigmoid((x_i - lambda) / temperature).

    The backward pass uses the implicit derivative of the equality-constrained
    solution, so gradients stay on the fixed-budget manifold instead of behaving
    like independent sigmoid gates.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
        keep_k: int,
        temperature: float,
        max_iter: int,
    ):
        if logits.dim() != 2 or valid_mask.shape != logits.shape:
            raise ValueError("_BudgetedSigmoidTopK expects matching [B, N] logits and masks.")

        valid = valid_mask.bool()
        counts = valid.sum(dim=-1)
        target = counts.clamp(max=int(keep_k))
        keep_k = int(keep_k)
        temperature = max(float(temperature), 1e-6)
        max_iter = int(max_iter)

        x = logits.float()
        free_rows = (target > 0) & (target < counts)
        out = valid.to(dtype=x.dtype) * (target >= counts).unsqueeze(-1).to(dtype=x.dtype)

        # Wide but finite brackets.  These make sigmoid mass almost n / 0 at the
        # endpoints, while avoiding +/-inf in lower-precision training.
        pos_inf = torch.tensor(torch.inf, device=x.device, dtype=x.dtype)
        neg_inf = torch.tensor(-torch.inf, device=x.device, dtype=x.dtype)
        lo = x.masked_fill(~valid, pos_inf).amin(dim=-1) - 50.0 * temperature - 1.0
        hi = x.masked_fill(~valid, neg_inf).amax(dim=-1) + 50.0 * temperature + 1.0
        lo = torch.where(free_rows, lo, torch.zeros_like(lo))
        hi = torch.where(free_rows, hi, torch.zeros_like(hi))
        target_float = target.to(dtype=x.dtype)
        for _ in range(max(8, max_iter)):
            mid = (lo + hi) * 0.5
            mass = (torch.sigmoid((x - mid[:, None]) / temperature) * valid).sum(dim=-1)
            # mass decreases as lambda increases.
            move_lo = (mass > target_float) & free_rows
            lo = torch.where(move_lo, mid, lo)
            hi = torch.where(free_rows & ~move_lo, mid, hi)

        lam = (lo + hi) * 0.5
        relaxed = torch.sigmoid((x - lam[:, None]) / temperature) * valid
        out = torch.where(free_rows[:, None], relaxed, out)
        ctx.save_for_backward(out, valid, free_rows)
        ctx.temperature = temperature
        return out.to(dtype=logits.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        soft, valid, free_rows = ctx.saved_tensors
        temperature = max(float(ctx.temperature), 1e-6)
        grad = grad_output.float()
        soft = soft.float()
        slope = soft * (1.0 - soft) * valid
        denom = slope.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # Implicit differentiation of sum_i sigmoid((x_i-lambda)/tau)=k:
        # dL/dx_j = a_j/tau * (g_j - sum_i g_i a_i / sum_i a_i),
        # where a_i = m_i(1-m_i).
        weighted_mean_grad = (grad * slope).sum(dim=-1, keepdim=True) / denom
        grad_logits = (slope / temperature) * (grad - weighted_mean_grad)
        grad_logits = grad_logits * free_rows[:, None]
        return grad_logits.to(dtype=grad_output.dtype), None, None, None, None


class LearnablePruneWrapper(nn.Module):
    def __init__(
        self,
        llava_model: nn.Module,
        predictor: LearnablePrunePredictor,
        budget_profiles: List[Dict[str, Any]],
        teacher_layer: int = 18,
        rank_loss_weight: float = 1.0,
        js_loss_weight: float = 1.0,
        ce_loss_weight: float = 1.0,
        enable_scale: bool = True,
        enable_rss: bool = False,
        topk_hinge_margin: float = 1.0,
        kd_temperature: float = 1.0,
        budgeted_soft_topk_iters: int = 32,
        budget_schedule_seed: int = 42,
        teacher_target_log_dir: Optional[str] = None,
        teacher_target_log_to_console: bool = False,
    ):
        super().__init__()
        self.llava = llava_model
        self.predictor = predictor
        self.teacher_layer = int(teacher_layer)
        self.rank_loss_weight = float(rank_loss_weight)
        self.js_loss_weight = float(js_loss_weight)
        self.ce_loss_weight = float(ce_loss_weight)
        self.enable_scale = bool(enable_scale)
        self.enable_rss = bool(enable_rss)
        self.topk_hinge_margin = float(topk_hinge_margin)
        self.kd_temperature = float(kd_temperature)
        self.budgeted_soft_topk_iters = int(budgeted_soft_topk_iters)
        self.budget_profiles = _canonical_budget_profiles(budget_profiles)
        self.budget_schedule_seed = int(budget_schedule_seed)
        self._active_budget_profile_index = 0
        self._budget_schedule_cached_cycle = -1
        self._budget_schedule_cached_order: Optional[torch.Tensor] = None
        self._apply_budget_profile(0)
        self._training_progress = 0.0
        self._last_soft_budget_error: Optional[torch.Tensor] = None

        self.teacher_target_log_to_console = bool(teacher_target_log_to_console)
        self._teacher_target_global_step = -1
        self._teacher_target_log_count = 0
        self._teacher_target_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        self._teacher_target_logger: Optional[logging.Logger] = None
        if teacher_target_log_dir:
            os.makedirs(teacher_target_log_dir, exist_ok=True)
            log_path = os.path.join(teacher_target_log_dir, f"teacher_targets_rank{self._teacher_target_rank}.jsonl")
            logger = logging.getLogger(f"learnable_prune.teacher_targets.rank{self._teacher_target_rank}.{id(self)}")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            logger.handlers.clear()
            handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            self._teacher_target_logger = logger

        if self.topk_hinge_margin <= 0.0:
            raise ValueError("topk_hinge_margin must be positive")
        if self.budgeted_soft_topk_iters <= 0:
            raise ValueError("budgeted_soft_topk_iters must be positive")
        num_layers = len(getattr(self.llava.model, "layers", []))
        self._decoder_layer_forward_parameters = (
            set(inspect.signature(self.llava.model.layers[0].forward).parameters)
            if num_layers > 0
            else set()
        )
        for profile in self.budget_profiles:
            if min(profile["keep_k"], profile["scope_target_count"], profile["mid_target_count"]) < 0:
                raise ValueError(f"Budget counts must be non-negative, got: {profile}")
            if profile["keep_k"] > profile["scope_target_count"]:
                raise ValueError(
                    "Predictor keep_k cannot exceed the SCOPE target count; "
                    f"got profile={profile}."
                )
            if profile["mid_target_count"] > profile["scope_target_count"]:
                raise ValueError(
                    "The mid-stage target cannot exceed the SCOPE target count; "
                    f"got profile={profile}."
                )
            if not (
                0
                <= profile["mid_pruning_layer_idx"]
                <= profile["final_wipe_layer_idx"]
                <= num_layers
            ):
                raise ValueError(
                    "Three-stage layer indices must satisfy 0 <= mid <= final <= num_layers; "
                    f"got profile={profile}, num_layers={num_layers}."
                )
            final_target = 0 if profile["enable_final_wipe"] else profile["mid_target_count"]
            computed_average = (
                profile["mid_pruning_layer_idx"] * profile["scope_target_count"]
                + (
                    profile["final_wipe_layer_idx"]
                    - profile["mid_pruning_layer_idx"]
                )
                * profile["mid_target_count"]
                + (num_layers - profile["final_wipe_layer_idx"]) * final_target
            ) / float(max(num_layers, 1))
            if abs(computed_average - profile["avg_token_budget"]) > 1.0:
                raise ValueError(
                    "Budget profile label does not match its layer-wise average for this model: "
                    f"declared={profile['avg_token_budget']}, computed={computed_average:.3f}, "
                    f"num_layers={num_layers}, profile={profile}."
                )

        for param in self.llava.parameters():
            param.requires_grad = False
        self.llava.eval()
        # DDP's initialization broadcast otherwise includes every frozen LLaVA
        # parameter even though its gradient reducer correctly tracks only the
        # predictor. Ignoring frozen parameters removes a full ~13-26 GiB model
        # broadcast per rank and its transient coalescing buffers. All ranks load
        # those immutable weights from the same checkpoint before DDP wrapping.
        frozen_parameter_names = [
            name for name, parameter in self.named_parameters() if not parameter.requires_grad
        ]
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            self, frozen_parameter_names
        )

    def _apply_budget_profile(self, profile_index: int) -> None:
        profile_index = int(profile_index)
        if not 0 <= profile_index < len(self.budget_profiles):
            raise IndexError(
                f"Budget profile index {profile_index} is outside [0, {len(self.budget_profiles) - 1}]."
            )
        profile = self.budget_profiles[profile_index]
        self._active_budget_profile_index = profile_index
        # Existing pruning helpers intentionally read these scalar attributes;
        # switching them once before forward keeps their hot paths branch-free.
        self.keep_k = int(profile["keep_k"])
        self.scope_target_count = int(profile["scope_target_count"])
        self.mid_pruning_layer_idx = int(profile["mid_pruning_layer_idx"])
        self.mid_target_count = int(profile["mid_target_count"])
        self.final_wipe_layer_idx = int(profile["final_wipe_layer_idx"])
        self.enable_final_wipe = bool(profile["enable_final_wipe"])

    @property
    def active_budget_profile(self) -> Dict[str, Any]:
        return dict(self.budget_profiles[self._active_budget_profile_index])

    def budget_profile_index_for_micro_step(self, micro_step: int) -> int:
        """Balanced shuffled-cycle schedule, deterministic and identical on every DDP rank."""

        profile_count = len(self.budget_profiles)
        if profile_count <= 1:
            return 0
        cycle, offset = divmod(max(0, int(micro_step)), profile_count)
        if cycle != self._budget_schedule_cached_cycle or self._budget_schedule_cached_order is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.budget_schedule_seed + cycle)
            self._budget_schedule_cached_order = torch.randperm(profile_count, generator=generator)
            self._budget_schedule_cached_cycle = cycle
        return int(self._budget_schedule_cached_order[offset].item())

    def set_budget_micro_step(self, micro_step: int) -> Dict[str, Any]:
        self._apply_budget_profile(self.budget_profile_index_for_micro_step(micro_step))
        return self.active_budget_profile

    def _budget_output_metrics(
        self,
        reference: torch.Tensor,
        *,
        total_loss: torch.Tensor,
        rank_loss: torch.Tensor,
        js_loss: torch.Tensor,
        ce_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        profile = self.active_budget_profile
        zero = reference.detach() * 0.0
        avg_budget = int(profile["avg_token_budget"])
        budget_label = str(avg_budget)
        return {
            "active_budget_profile_index": zero + float(self._active_budget_profile_index),
            "active_avg_token_budget": zero + float(avg_budget),
            "active_keep_k": zero + float(profile["keep_k"]),
            "active_scope_target_count": zero + float(profile["scope_target_count"]),
            "active_mid_target_count": zero + float(profile["mid_target_count"]),
            f"budget_{budget_label}_total_loss": total_loss.detach(),
            f"budget_{budget_label}_rank_loss": rank_loss.detach(),
            f"budget_{budget_label}_js_loss": js_loss.detach(),
            f"budget_{budget_label}_ce_loss": ce_loss.detach(),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.llava.eval()
        return self

    @property
    def is_gradient_checkpointing(self) -> bool:
        value = getattr(self.llava, "is_gradient_checkpointing", False)
        return bool(value() if callable(value) else value)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {
                "use_reentrant": False,
                "context_fn": _default_checkpoint_contexts,
            }
        if hasattr(self.llava, "gradient_checkpointing_enable"):
            try:
                self.llava.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            except TypeError:
                self.llava.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        if hasattr(self.llava, "gradient_checkpointing_disable"):
            self.llava.gradient_checkpointing_disable()

    def set_training_progress(self, progress: float) -> None:
        self._training_progress = float(min(max(progress, 0.0), 1.0))

    def set_training_log_step(self, global_step: int) -> None:
        self._teacher_target_global_step = int(global_step)

    def _loss_weight_scale(self, start: float, end: float) -> float:
        if not self.enable_scale:
            return 1.0
        progress = float(min(max(self._training_progress, 0.0), 1.0))
        return float(start + (end - start) * progress)

    def _current_rank_loss_weight(self) -> float:
        return self.rank_loss_weight * self._loss_weight_scale(2.0, 0.2)

    def _current_js_loss_weight(self) -> float:
        return self.js_loss_weight * self._loss_weight_scale(0.2, 2.0)

    def _current_ce_loss_weight(self) -> float:
        return self.ce_loss_weight * self._loss_weight_scale(0.2, 2.0)

    @contextmanager
    def _temporary_decoder_checkpoint_training(self):
        language_model = getattr(self.llava, "model", None)
        layers = list(getattr(language_model, "layers", []) or [])
        if not any(bool(getattr(layer, "gradient_checkpointing", False)) for layer in layers):
            yield
            return

        originals = [(layer, bool(layer.training)) for layer in layers]
        try:
            for layer, _ in originals:
                # Trigger HF's GradientCheckpointingLayer without changing
                # dropout/eval state of frozen child modules.
                layer.training = True
            yield
        finally:
            for layer, original in originals:
                layer.training = original

    def _full_forward_with_teacher_qkv(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[Any, Dict[str, torch.Tensor], int]:
        language_model = self.llava.model
        num_layers = len(language_model.layers)
        if not 0 <= self.teacher_layer < num_layers:
            raise ValueError(
                f"--teacher_layer must be in [0, {num_layers - 1}] for this model; got {self.teacher_layer}."
            )
        teacher_layer = self.teacher_layer
        teacher_qkv: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_projection_hook(name: str):
            def hook(_module, _args, output):
                teacher_qkv[name] = output.detach()
            return hook

        self_attn = language_model.layers[teacher_layer].self_attn
        hooks.extend(
            (
                self_attn.q_proj.register_forward_hook(make_projection_hook("query")),
                self_attn.k_proj.register_forward_hook(make_projection_hook("key")),
                self_attn.v_proj.register_forward_hook(make_projection_hook("value")),
            )
        )
        try:
            full_outputs = self.llava.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            for handle in hooks:
                handle.remove()

        if set(teacher_qkv) != {"query", "key", "value"}:
            raise RuntimeError(f"Failed to capture teacher Q/K/V projections for layer: {teacher_layer}")
        return full_outputs, teacher_qkv, teacher_layer

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
            image_features = self.llava.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in image_batches]
            image_features = torch.split(image_features, split_sizes, dim=0)

            mm_patch_merge_type = getattr(self.llava.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.llava.config, "image_aspect_ratio", "square")
            if mm_patch_merge_type == "flat":
                merged = [feature.flatten(0, 1) for feature in image_features]
                coordinates = []
                side = int(self.llava.get_vision_tower().num_patches_per_side)
                for feature, flat_feature in zip(image_features, merged):
                    grid = _normalized_grid_coordinates(
                        side, side, device=flat_feature.device, dtype=flat_feature.dtype
                    )
                    coordinates.append(grid.repeat(feature.shape[0], 1))
                return merged, coordinates
            if not mm_patch_merge_type.startswith("spatial"):
                raise ValueError(f"Unexpected mm_patch_merge_type: {mm_patch_merge_type}")

            merged_features: List[torch.Tensor] = []
            merged_coordinates: List[torch.Tensor] = []
            vision_tower = self.llava.get_vision_tower()
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
                        self.llava.config.image_grid_pinpoints,
                        vision_tower.config.image_size,
                    )
                    patch_image_feature = patch_image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    if "unpad" in mm_patch_merge_type:
                        patch_image_feature = patch_image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        patch_image_feature = patch_image_feature.flatten(1, 2).flatten(2, 3)
                        patch_image_feature = unpad_image(patch_image_feature, image_size)
                        patch_height, patch_width = patch_image_feature.shape[-2:]
                        image_newline = self.llava.model.image_newline.to(
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
                        image_newline = self.llava.model.image_newline.to(
                            device=image_feature.device,
                            dtype=image_feature.dtype,
                        )
                        image_feature = torch.cat((image_feature, image_newline[None]), dim=0)
                        image_coordinates = _normalized_grid_coordinates(
                            height,
                            width,
                            device=image_feature.device,
                            dtype=image_feature.dtype,
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

        image_features = self.llava.encode_images(images)
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

    def _build_multimodal_inputs(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        image_sizes: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand official LLaVA IMAGE_TOKEN_INDEX placeholders to image embeddings.

        This mirrors LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal,
        but also returns pseudo input ids with one IMAGE_TOKEN_INDEX per visual
        embedding so the HF learnable-prune objective can reuse the same span
        and mask logic.
        """

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        image_features, image_coordinates = self._encode_and_merge_image_features(images, image_sizes=image_sizes)

        compact_input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        compact_labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        embed_tokens = self.llava.get_model().embed_tokens
        safe_input_ids = input_ids.masked_fill(input_ids.eq(IMAGE_TOKEN_INDEX), 0)
        input_embeds = embed_tokens(safe_input_ids)
        compact_input_embeds = [
            cur_embeds[cur_attention_mask]
            for cur_embeds, cur_attention_mask in zip(input_embeds, attention_mask)
        ]

        new_input_embeds: List[torch.Tensor] = []
        new_input_ids: List[torch.Tensor] = []
        new_labels: List[torch.Tensor] = []
        new_visual_coordinates: List[torch.Tensor] = []
        cur_image_idx = 0
        image_token_value = int(IMAGE_TOKEN_INDEX)

        for batch_idx, cur_input_ids in enumerate(compact_input_ids):
            num_images = int((cur_input_ids == IMAGE_TOKEN_INDEX).sum().item())
            cur_labels = compact_labels[batch_idx]
            cur_input_embeds = compact_input_embeds[batch_idx]
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds = torch.cat([cur_input_embeds, cur_image_features[0:0].to(cur_input_embeds.dtype)], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_input_ids.append(cur_input_ids)
                new_labels.append(cur_labels)
                new_visual_coordinates.append(cur_input_embeds.new_zeros((cur_input_embeds.shape[0], 2)))
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_input_embeds_noim = []
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_input_embeds_noim.append(cur_input_embeds[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])

            cur_new_embeds = []
            cur_new_ids = []
            cur_new_labels = []
            cur_new_coordinates = []
            for i in range(num_images + 1):
                cur_new_embeds.append(cur_input_embeds_noim[i])
                cur_new_ids.append(cur_input_ids_noim[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_coordinates.append(cur_input_embeds_noim[i].new_zeros((cur_input_embeds_noim[i].shape[0], 2)))
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx].to(
                        device=cur_input_embeds.device,
                        dtype=cur_input_embeds.dtype,
                    )
                    cur_image_idx += 1
                    cur_image_coordinates = image_coordinates[cur_image_idx - 1].to(
                        device=cur_input_embeds.device, dtype=cur_input_embeds.dtype
                    )
                    visual_len = cur_image_features.shape[0]
                    cur_new_embeds.append(cur_image_features)
                    cur_new_ids.append(torch.full((visual_len,), image_token_value, device=cur_input_ids.device, dtype=cur_input_ids.dtype))
                    cur_new_labels.append(torch.full((visual_len,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_coordinates.append(cur_image_coordinates)

            new_input_embeds.append(torch.cat(cur_new_embeds, dim=0))
            new_input_ids.append(torch.cat(cur_new_ids, dim=0))
            new_labels.append(torch.cat(cur_new_labels, dim=0))
            new_visual_coordinates.append(torch.cat(cur_new_coordinates, dim=0))

        tokenizer_model_max_length = getattr(self.llava.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_input_ids = [x[:tokenizer_model_max_length] for x in new_input_ids]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_visual_coordinates = [x[:tokenizer_model_max_length] for x in new_visual_coordinates]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        pad_token_id = int(getattr(self.llava.config, "pad_token_id", 0) or 0)
        hidden_size = new_input_embeds[0].shape[-1]
        padded_embeds = new_input_embeds[0].new_zeros((batch_size, max_len, hidden_size))
        padded_ids = torch.full((batch_size, max_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        padded_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
        padded_attention = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        padded_position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        padded_visual_coordinates = padded_embeds.new_zeros((batch_size, max_len, 2))
        position_template = torch.arange(max_len, dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_embeds, cur_ids, cur_labels) in enumerate(zip(new_input_embeds, new_input_ids, new_labels)):
            cur_len = cur_embeds.shape[0]
            if getattr(self.llava.config, "tokenizer_padding_side", "right") == "left":
                if cur_len > 0:
                    padded_embeds[i, -cur_len:] = cur_embeds
                    padded_ids[i, -cur_len:] = cur_ids
                    padded_labels[i, -cur_len:] = cur_labels
                    padded_attention[i, -cur_len:] = True
                    padded_position_ids[i, -cur_len:] = position_template[:cur_len]
                    padded_visual_coordinates[i, -cur_len:] = new_visual_coordinates[i]
            else:
                if cur_len > 0:
                    padded_embeds[i, :cur_len] = cur_embeds
                    padded_ids[i, :cur_len] = cur_ids
                    padded_labels[i, :cur_len] = cur_labels
                    padded_attention[i, :cur_len] = True
                    padded_position_ids[i, :cur_len] = position_template[:cur_len]
                    padded_visual_coordinates[i, :cur_len] = new_visual_coordinates[i]

        return (
            padded_embeds,
            padded_ids,
            padded_attention,
            padded_position_ids,
            padded_labels,
            padded_visual_coordinates,
        )

    @staticmethod
    def _pack_mask_positions(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack true positions of a [B, S] mask into [B, max_count]."""

        mask = mask.bool()
        counts = mask.sum(dim=-1)
        max_count = int(counts.max().item()) if counts.numel() else 0
        positions = torch.zeros((mask.shape[0], max_count), device=mask.device, dtype=torch.long)
        valid = torch.arange(max_count, device=mask.device)[None, :] < counts[:, None]
        if max_count:
            batch_idx, seq_idx = mask.nonzero(as_tuple=True)
            slot_idx = mask.long().cumsum(dim=-1)[batch_idx, seq_idx] - 1
            positions[batch_idx, slot_idx] = seq_idx
        return positions, valid

    def _get_visual_token_attention_scores(
        self,
        projected_qkv: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        scoring_layer_idx: int,
        visual_positions: torch.Tensor,
        visual_valid: torch.Tensor,
        query_positions: torch.Tensor,
        query_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Leave-one-visual-token-out scores without anyres padding GEMMs.

        Q/K/V and RoPE are still computed once for the complete teacher batch.
        The expensive query-key and query-visual interactions are evaluated per
        row at their true lengths: LLaVA-NeXT batches can differ by thousands of
        visual/query tokens, so padding those matrices to both batch maxima does
        substantially more work than a tiny batch-row loop.
        """

        language_model = self.llava.model
        scoring_layer = language_model.layers[scoring_layer_idx]
        self_attn = scoring_layer.self_attn
        query_projected = projected_qkv["query"]
        key_projected = projected_qkv["key"]
        value_projected = projected_qkv["value"]
        bsz, seq_len, hidden_size = query_projected.shape
        if visual_positions.shape[1] == 0 or query_positions.shape[1] == 0:
            return query_projected.new_zeros((bsz, visual_positions.shape[1]), dtype=torch.float32)

        num_heads = getattr(self_attn.config, "num_attention_heads", None)
        if num_heads is None:
            num_heads = getattr(self_attn, "num_heads")
        num_heads = int(num_heads)
        num_kv_heads = int(getattr(self_attn.config, "num_key_value_heads", num_heads))
        head_dim = int(self_attn.head_dim)
        scaling = float(getattr(self_attn, "scaling", head_dim ** -0.5))

        query_states = query_projected.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_projected.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_projected.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        cos, sin = language_model.rotary_emb(query_projected, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = _repeat_kv(key_states, num_heads // num_kv_heads)
        value_states = _repeat_kv(value_states, num_heads // num_kv_heads)

        visual_positions = visual_positions.to(dtype=torch.long, device=query_projected.device)
        visual_valid = visual_valid.to(device=query_projected.device)
        query_positions = query_positions.to(dtype=torch.long, device=query_projected.device)
        query_valid = query_valid.to(device=query_projected.device)
        score_sum = query_projected.new_zeros((bsz, visual_positions.shape[1]), dtype=torch.float32)

        # One device-to-host synchronization supplies all small row sizes.
        # Uniform/near-uniform batches take a vectorized path; highly ragged
        # anyres batches retain the row path to avoid padding quadratic work.
        row_sizes = torch.stack(
            (attention_mask.sum(dim=-1), visual_valid.sum(dim=-1), query_valid.sum(dim=-1)),
            dim=-1,
        ).to(device="cpu", dtype=torch.long).tolist()
        true_attention_work = sum(key_count * query_count for key_count, _, query_count in row_sizes)
        true_delta_work = sum(visual_count * query_count for _, visual_count, query_count in row_sizes)
        padded_attention_work = bsz * seq_len * query_positions.shape[1]
        padded_delta_work = bsz * visual_positions.shape[1] * query_positions.shape[1]
        padding_ratio = max(
            padded_attention_work / float(max(true_attention_work, 1)),
            padded_delta_work / float(max(true_delta_work, 1)),
        )
        max_padding_ratio = float(os.environ.get("LEARNABLE_PRUNE_BATCHED_SCORE_MAX_PADDING", "1.5"))
        if padding_ratio <= max_padding_ratio:
            return self._get_visual_token_attention_scores_batched(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                self_attn=self_attn,
                scaling=scaling,
                attention_mask=attention_mask,
                visual_positions=visual_positions,
                visual_valid=visual_valid,
                query_positions=query_positions,
                query_valid=query_valid,
                hidden_size=hidden_size,
            )

        left_padding = getattr(self.llava.config, "tokenizer_padding_side", "right") == "left"
        for batch_idx, (key_count, visual_count, query_count) in enumerate(row_sizes):
            if visual_count == 0 or query_count == 0:
                continue

            # _build_multimodal_inputs always emits one contiguous valid span.
            # Slicing it avoids materializing nonzero indices and copying K/V
            # with index_select while retaining left/right-padding semantics.
            key_start = seq_len - key_count if left_padding else 0
            key_end = key_start + key_count
            row_key = key_states[batch_idx, :, key_start:key_end, :]
            row_value = value_states[batch_idx, :, key_start:key_end, :]
            row_visual_positions = visual_positions[batch_idx, :visual_count]
            row_query_positions = query_positions[batch_idx, :query_count]
            row_key_positions = torch.arange(key_start, key_end, device=query_projected.device)
            visual_key_columns = row_visual_positions - key_start
            row_v_visual = row_value.index_select(1, visual_key_columns).permute(1, 0, 2).contiguous()
            row_score_sum = score_sum[batch_idx, :visual_count]

            # Eight queries keeps the temporary [Q, Nv, hidden] delta tensor
            # bounded while providing enough parallel work for the projection.
            for chunk_start in range(0, query_count, 8):
                q_pos = row_query_positions[chunk_start : chunk_start + 8]
                chunk_len = q_pos.shape[0]
                q_states = query_states[batch_idx].index_select(1, q_pos)
                attn_scores = torch.matmul(q_states, row_key.transpose(1, 2)) * scaling
                causal = row_key_positions[None, None, :] <= q_pos[None, :, None]
                attn_scores = attn_scores.float().masked_fill(~causal, torch.finfo(torch.float32).min)
                attn_probs = F.softmax(attn_scores, dim=-1).to(dtype=q_states.dtype)
                z_heads = torch.matmul(attn_probs, row_value).permute(1, 0, 2).contiguous()

                # Visual tokens are guaranteed to be valid keys. Convert their
                # sequence positions to columns in the compact key matrix.
                alpha_visual = attn_probs.index_select(-1, visual_key_columns).permute(1, 2, 0).contiguous()
                beta = alpha_visual / (1.0 - alpha_visual).clamp(min=1e-6)
                delta_z = beta.unsqueeze(-1) * (z_heads.unsqueeze(1) - row_v_visual.unsqueeze(0))
                delta_y = self_attn.o_proj(delta_z.flatten(0, 1).flatten(1, 2)).view(
                    chunk_len, visual_count, hidden_size
                )
                row_score_sum.add_(delta_y.float().norm(dim=-1).sum(dim=0))

            row_score_sum.div_(float(query_count))
        return score_sum

    @staticmethod
    def _get_visual_token_attention_scores_batched(
        *,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        self_attn: nn.Module,
        scaling: float,
        attention_mask: torch.Tensor,
        visual_positions: torch.Tensor,
        visual_valid: torch.Tensor,
        query_positions: torch.Tensor,
        query_valid: torch.Tensor,
        hidden_size: int,
    ) -> torch.Tensor:
        """Vectorized leave-one-out scoring for low-padding batches.

        Query chunks bound the transient ``[B, Q, Nv, hidden]`` perturbation.
        This collapses the former batch-row loop into larger GEMMs for fixed-grid
        LLaVA batches while preserving the ragged fallback for LLaVA-NeXT.
        """

        bsz, num_heads, _, head_dim = query_states.shape
        query_count = query_positions.shape[1]
        visual_count = visual_positions.shape[1]
        if query_count == 0 or visual_count == 0:
            return query_states.new_zeros((bsz, visual_count), dtype=torch.float32)

        q_index = query_positions[:, None, :, None].expand(bsz, num_heads, query_count, head_dim)
        packed_query = query_states.gather(dim=2, index=q_index)
        v_index = visual_positions[:, None, :, None].expand(bsz, num_heads, visual_count, head_dim)
        visual_value = value_states.gather(dim=2, index=v_index).permute(0, 2, 1, 3).contiguous()
        key_transposed = key_states.transpose(2, 3)
        key_positions = torch.arange(key_states.shape[2], device=key_states.device)

        # Keep both delta_z and o_proj(delta_z) near this budget at peak. The
        # minimum chunk is one query, so unusually large anyres rows still make
        # progress without allocating a full Qmax perturbation tensor.
        temporary_mib = max(16, int(os.environ.get("LEARNABLE_PRUNE_SCORE_TEMP_MIB", "128")))
        target_elements = (temporary_mib << 20) // max(query_states.element_size(), 1)
        elements_per_query = max(1, bsz * visual_count * hidden_size)
        query_chunk = max(1, min(query_count, target_elements // elements_per_query))
        score_sum = query_states.new_zeros((bsz, visual_count), dtype=torch.float32)

        for chunk_start in range(0, query_count, query_chunk):
            chunk_end = min(chunk_start + query_chunk, query_count)
            q_pos = query_positions[:, chunk_start:chunk_end]
            q_valid = query_valid[:, chunk_start:chunk_end]
            q_states = packed_query[:, :, chunk_start:chunk_end, :]
            attn_scores = torch.matmul(q_states, key_transposed) * scaling
            allowed = (
                attention_mask[:, None, None, :].bool()
                & q_valid[:, None, :, None]
                & (key_positions[None, None, None, :] <= q_pos[:, None, :, None])
            )
            attn_probs = F.softmax(
                attn_scores.float().masked_fill(~allowed, torch.finfo(torch.float32).min),
                dim=-1,
            ).to(dtype=q_states.dtype)
            z_heads = torch.matmul(attn_probs, value_states).permute(0, 2, 1, 3).contiguous()

            alpha_index = visual_positions[:, None, None, :].expand(
                bsz, num_heads, chunk_end - chunk_start, visual_count
            )
            alpha_visual = attn_probs.gather(dim=-1, index=alpha_index).permute(0, 2, 3, 1).contiguous()
            beta = alpha_visual / (1.0 - alpha_visual).clamp(min=1e-6)
            delta_z = beta.unsqueeze(-1) * (z_heads.unsqueeze(2) - visual_value.unsqueeze(1))
            delta_y = self_attn.o_proj(delta_z.flatten(0, 2).flatten(1, 2)).view(
                bsz, chunk_end - chunk_start, visual_count, hidden_size
            )
            chunk_scores = delta_y.float().norm(dim=-1)
            chunk_scores = chunk_scores * q_valid[:, :, None] * visual_valid[:, None, :]
            score_sum.add_(chunk_scores.sum(dim=1))

        return score_sum / query_valid.sum(dim=-1, keepdim=True).clamp_min(1).float()

    def _teacher_targets(
        self,
        teacher_qkv: Dict[str, torch.Tensor],
        teacher_layer: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = input_ids.shape[0]
        device = input_ids.device
        image_token_id = int(IMAGE_TOKEN_INDEX)
        visual_mask = input_ids.eq(image_token_id) & attention_mask.bool()
        visual_positions, valid_masks = self._pack_mask_positions(visual_mask)
        if visual_positions.shape[1] == 0:
            return torch.zeros_like(valid_masks, dtype=torch.float32), valid_masks

        sequence_positions = torch.arange(input_ids.shape[1], device=device)[None, :]
        visual_end = torch.where(
            visual_mask,
            sequence_positions,
            torch.full_like(sequence_positions, -1),
        ).amax(dim=-1) + 1
        # Teacher queries must be available at inference. Exclude supervised
        # assistant/answer tokens and score only prompt text after the image.
        query_mask = (
            attention_mask.bool()
            & labels.eq(IGNORE_INDEX)
            & ~visual_mask
            & (sequence_positions >= visual_end[:, None])
        )
        query_positions, query_valid = self._pack_mask_positions(query_mask)
        # Do not manufacture arbitrary TopK labels from an all-zero teacher in
        # the unusual case where no inference-visible text follows the image.
        valid_masks = valid_masks & query_valid.any(dim=-1, keepdim=True)

        scores = self._get_visual_token_attention_scores(
            projected_qkv=teacher_qkv,
            attention_mask=attention_mask,
            position_ids=position_ids,
            scoring_layer_idx=teacher_layer,
            visual_positions=visual_positions,
            visual_valid=valid_masks,
            query_positions=query_positions,
            query_valid=query_valid,
        ).float()

        visual_counts = valid_masks.sum(dim=-1)
        visual_starts = visual_positions[:, 0]
        if self._teacher_target_logger is not None or self.teacher_target_log_to_console:
            for b in range(bsz):
                start = int(visual_starts[b].item())
                visual_count = int(visual_counts[b].item())
                sample_id = int(sample_ids[b].item()) if sample_ids is not None else None
                self._log_teacher_target_selection(
                    sample_id=sample_id,
                    local_batch_index=b,
                    visual_start=start,
                    visual_end=start + visual_count,
                    query_count=int(query_valid[b].sum().item()),
                    teacher_layer=teacher_layer,
                )
        if self.enable_rss and inputs_embeds is not None:
            # RSS depends on a variable-sized Nv x Nv similarity matrix. It is
            # opt-in and remains row-wise to avoid padding anyres batches to the
            # largest quadratic matrix.
            for b in range(bsz):
                start = int(visual_starts[b].item())
                visual_count = int(valid_masks[b].sum().item())
                visual_embeds = inputs_embeds[b : b + 1, start : start + visual_count, :].detach().float()
                scores[b, :visual_count] = self._apply_rss_algorithm(
                    visual_embeds,
                    scores[b, :visual_count],
                )
        return scores, valid_masks

    def _apply_rss_algorithm(
        self,
        visual_embeds: torch.Tensor,
        attention_scores: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        visual_norm = F.normalize(visual_embeds, p=2, dim=-1)
        sim_matrix = F.relu(torch.matmul(visual_norm, visual_norm.transpose(1, 2)))
        if threshold > 0.0:
            sim_matrix = sim_matrix.masked_fill(sim_matrix < threshold, 0.0)

        score_row = attention_scores.view(1, 1, -1)
        score_col = attention_scores.view(1, -1, 1)
        dominance_mask = (score_row > score_col).to(dtype=sim_matrix.dtype)
        max_sim_from_superiors = (sim_matrix * dominance_mask).max(dim=2).values.squeeze(0)
        suppression_factor = (1.0 - max_sim_from_superiors).clamp(min=0.0).pow(beta)
        return attention_scores * suppression_factor.to(device=attention_scores.device, dtype=attention_scores.dtype)

    def _log_teacher_target_selection(
        self,
        *,
        sample_id: Optional[int],
        local_batch_index: int,
        visual_start: int,
        visual_end: int,
        query_count: int,
        teacher_layer: int,
    ) -> None:
        if self._teacher_target_logger is None and not self.teacher_target_log_to_console:
            return

        record = {
            "global_step": int(self._teacher_target_global_step),
            "log_index": int(self._teacher_target_log_count),
            "rank": int(self._teacher_target_rank),
            "sample_id": sample_id,
            "local_batch_index": int(local_batch_index),
            "visual_span": [int(visual_start), int(visual_end)],
            "visual_token_count": int(visual_end - visual_start),
            "query_count": int(query_count),
            "teacher_layer": int(teacher_layer),
            "selected_layers": [int(teacher_layer)],
            "pooled_layer_count": 1,
        }
        message = json.dumps(record, ensure_ascii=False)
        if self._teacher_target_logger is not None:
            self._teacher_target_logger.info(message)
        if self.teacher_target_log_to_console and self._teacher_target_rank == 0:
            print(f"[teacher-target] {message}", flush=True)
        self._teacher_target_log_count += 1

    def _budgeted_soft_topk_gate(
        self,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return _BudgetedSigmoidTopK.apply(
            logits,
            valid_mask,
            int(self.keep_k),
            1.0,
            int(self.budgeted_soft_topk_iters),
        )

    def _retention_probabilities(
        self,
        predictor_logits: torch.Tensor,
        visual_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return hard-ST visual retention gates for physical three-stage pruning."""
        valid = visual_valid.bool()
        soft = self._budgeted_soft_topk_gate(predictor_logits, valid)
        valid_counts = valid.sum(dim=-1)
        keep_counts = valid_counts.clamp(max=self.keep_k)
        self._last_soft_budget_error = (
            soft.detach().float().sum(dim=-1) - keep_counts.float()
        ).abs().mean()

        hard = self.predictor.budget_topk_mask(
            predictor_logits, self.keep_k, valid
        ).to(dtype=predictor_logits.dtype)
        gate = hard + soft - soft.detach()
        return torch.where(valid, gate, torch.ones_like(gate))

    def _pack_by_keep_mask(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Physically pack variable-length rows and right-pad only to the batch maximum."""

        keep_mask = keep_mask.bool() & attention_mask.bool()
        counts = keep_mask.sum(dim=-1)
        min_count, max_count = torch.aminmax(counts)
        min_count_host, max_count_host = torch.stack((min_count, max_count)).to(device="cpu").tolist()
        if min_count_host == 0:
            raise ValueError("Three-stage pruning cannot remove every token from a sequence.")
        max_count_host = int(max_count_host)

        batch_idx, seq_idx = keep_mask.nonzero(as_tuple=True)
        row_offsets = counts.cumsum(dim=0) - counts
        slot_idx = torch.arange(seq_idx.numel(), device=seq_idx.device) - row_offsets[batch_idx]
        packed_source_positions = torch.zeros(
            (keep_mask.shape[0], max_count_host), device=keep_mask.device, dtype=torch.long
        )
        packed_source_positions[batch_idx, slot_idx] = seq_idx
        packed_attention = (
            torch.arange(max_count_host, device=attention_mask.device)[None, :] < counts[:, None]
        )

        hidden_index = packed_source_positions.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        packed_hidden = hidden_states.gather(dim=1, index=hidden_index)
        packed_hidden.masked_fill_(~packed_attention.unsqueeze(-1), 0)
        packed_ids = input_ids.gather(dim=1, index=packed_source_positions)
        packed_positions = position_ids.gather(dim=1, index=packed_source_positions)
        packed_labels = labels.gather(dim=1, index=packed_source_positions)
        pad_token_id = int(getattr(self.llava.config, "pad_token_id", 0) or 0)
        packed_ids.masked_fill_(~packed_attention, pad_token_id)
        packed_positions.masked_fill_(~packed_attention, 0)
        packed_labels.masked_fill_(~packed_attention, IGNORE_INDEX)
        packed_attention = packed_attention.to(dtype=attention_mask.dtype)
        return packed_hidden, packed_ids, packed_attention, packed_positions, packed_labels

    def _run_decoder_layer_range(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        start_layer: int,
        end_layer: int,
    ) -> torch.Tensor:
        """Run a contiguous decoder segment without a second model forward."""

        if start_layer >= end_layer:
            return hidden_states
        language_model = self.llava.model
        use_boolean_mask = str(getattr(language_model.config, "_attn_implementation", "eager")) == "sdpa"
        causal_mask = _make_4d_causal_mask(
            attention_mask, dtype=hidden_states.dtype, boolean=use_boolean_mask
        )
        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
        position_embeddings = language_model.rotary_emb(hidden_states, position_ids)

        for layer_idx in range(int(start_layer), int(end_layer)):
            layer = language_model.layers[layer_idx]
            parameters = self._decoder_layer_forward_parameters
            kwargs: Dict[str, Any] = {
                "attention_mask": causal_mask,
                "position_ids": position_ids,
            }
            if "use_cache" in parameters:
                kwargs["use_cache"] = False
            if "output_attentions" in parameters:
                kwargs["output_attentions"] = False
            if "cache_position" in parameters:
                kwargs["cache_position"] = cache_position
            if "position_embeddings" in parameters:
                kwargs["position_embeddings"] = position_embeddings
            layer_outputs = layer(hidden_states, **kwargs)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, (tuple, list)) else layer_outputs
        return hidden_states

    def _scope_stage_keep_mask(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        predictor_logits: torch.Tensor,
        visual_positions: torch.Tensor,
        visual_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Select predictor seeds and fill to the inference SCOPE budget."""

        keep_mask = attention_mask.bool() & input_ids.ne(int(IMAGE_TOKEN_INDEX))
        visual_counts = visual_valid.sum(dim=-1).to(device="cpu", dtype=torch.long).tolist()
        with torch.no_grad():
            detached_logits = predictor_logits.detach().float()
            seed_indices, seed_valid = self.predictor.budget_topk_indices(
                detached_logits, self.keep_k, visual_valid
            )
            for batch_idx, visual_count in enumerate(visual_counts):
                if visual_count <= 0:
                    continue
                row_positions = visual_positions[batch_idx, :visual_count]
                seeds = seed_indices[batch_idx, seed_valid[batch_idx]]
                target_count = min(self.scope_target_count, visual_count)
                selected_relative = _seeded_residual_scope_indices(
                    inputs_embeds[batch_idx].index_select(0, row_positions),
                    seeds,
                    target_count,
                )
                selected_positions = row_positions.index_select(0, selected_relative)
                keep_mask[batch_idx, selected_positions] = True
        return keep_mask

    def _mid_stage_attention_scores(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        scoring_layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute inference-equivalent mid-stage scores from prompt-visible queries."""

        visual_mask = input_ids.eq(int(IMAGE_TOKEN_INDEX)) & attention_mask.bool()
        visual_positions, visual_valid = self._pack_mask_positions(visual_mask)
        if visual_positions.shape[1] == 0:
            return hidden_states.new_zeros(visual_valid.shape, dtype=torch.float32), visual_positions, visual_valid

        sequence_positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        visual_end = torch.where(
            visual_mask,
            sequence_positions,
            torch.full_like(sequence_positions, -1),
        ).amax(dim=-1)
        prompt_text_mask = (
            attention_mask.bool()
            & input_ids.ne(int(IMAGE_TOKEN_INDEX))
            & labels.eq(IGNORE_INDEX)
        )
        query_mask = prompt_text_mask & (sequence_positions > visual_end[:, None])
        # Match inference's last-text fallback without allowing supervised answer
        # tokens to leak into normal training-time importance estimates.
        any_text_mask = attention_mask.bool() & input_ids.ne(int(IMAGE_TOKEN_INDEX))
        fallback_mask = torch.where(
            prompt_text_mask.any(dim=-1, keepdim=True),
            prompt_text_mask,
            any_text_mask,
        )
        fallback_positions = torch.where(
            fallback_mask,
            sequence_positions,
            torch.full_like(sequence_positions, -1),
        ).amax(dim=-1)
        fallback_rows = ~query_mask.any(dim=-1) & fallback_positions.ge(0)
        fallback_batch = fallback_rows.nonzero(as_tuple=False).flatten()
        query_mask[fallback_batch, fallback_positions[fallback_batch]] = True
        query_positions, query_valid = self._pack_mask_positions(query_mask)

        scoring_layer = self.llava.model.layers[int(scoring_layer_idx)]
        with torch.no_grad():
            normed = scoring_layer.input_layernorm(hidden_states.detach())
            self_attn = scoring_layer.self_attn
            projected_qkv = {
                "query": self_attn.q_proj(normed),
                "key": self_attn.k_proj(normed),
                "value": self_attn.v_proj(normed),
            }
            scores = self._get_visual_token_attention_scores(
                projected_qkv=projected_qkv,
                attention_mask=attention_mask,
                position_ids=position_ids,
                scoring_layer_idx=int(scoring_layer_idx),
                visual_positions=visual_positions,
                visual_valid=visual_valid,
                query_positions=query_positions,
                query_valid=query_valid,
            )
        return scores, visual_positions, visual_valid

    def _mid_stage_keep_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        importance_scores: torch.Tensor,
        visual_positions: torch.Tensor,
        visual_valid: torch.Tensor,
    ) -> torch.Tensor:
        keep_mask = attention_mask.bool() & input_ids.ne(int(IMAGE_TOKEN_INDEX))
        with torch.no_grad():
            selected_relative, selected_valid = self.predictor.budget_topk_indices(
                importance_scores.float(), self.mid_target_count, visual_valid
            )
            if selected_relative.shape[1] > 0:
                selected_positions = visual_positions.gather(dim=1, index=selected_relative)
                selected_batch = torch.arange(keep_mask.shape[0], device=keep_mask.device)[:, None]
                selected_batch = selected_batch.expand_as(selected_positions)
                keep_mask[
                    selected_batch[selected_valid], selected_positions[selected_valid]
                ] = True
        return keep_mask

    def _three_stage_student_forward(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
        predictor_logits: torch.Tensor,
        retention_probs: torch.Tensor,
        visual_positions: torch.Tensor,
        visual_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Run scope -> mid-attention -> final-wipe in one compact student pass."""

        scope_keep_mask = self._scope_stage_keep_mask(
            inputs_embeds,
            input_ids,
            attention_mask,
            predictor_logits,
            visual_positions,
            visual_valid,
        )
        # Forward value is exactly one for every visual token.  Its derivative is
        # the fixed-budget soft TopK derivative, so hard physical packing remains
        # inference-faithful while JS/CE still optimize the predictor end to end.
        visual_st_gate = torch.ones_like(retention_probs) + (
            retention_probs - retention_probs.detach()
        )
        sequence_gate = inputs_embeds.new_ones(attention_mask.shape)
        visual_mask = input_ids.eq(int(IMAGE_TOKEN_INDEX)) & attention_mask.bool()
        sequence_gate[visual_mask] = visual_st_gate[visual_valid].to(dtype=sequence_gate.dtype)
        gated_inputs = inputs_embeds * sequence_gate.unsqueeze(-1)
        hidden_states, stage_ids, stage_attention, stage_positions, stage_labels = self._pack_by_keep_mask(
            gated_inputs,
            input_ids,
            attention_mask,
            position_ids,
            labels,
            scope_keep_mask,
        )
        scope_visual_counts = (
            stage_ids.eq(int(IMAGE_TOKEN_INDEX)) & stage_attention.bool()
        ).sum(dim=-1)

        num_layers = len(self.llava.model.layers)
        mid_layer = min(self.mid_pruning_layer_idx, num_layers)
        wipe_layer = min(self.final_wipe_layer_idx, num_layers) if self.enable_final_wipe else num_layers
        hidden_states = self._run_decoder_layer_range(
            hidden_states, stage_attention, stage_positions, 0, mid_layer
        )

        scoring_layer_idx = min(mid_layer, max(num_layers - 1, 0))
        mid_scores, mid_visual_positions, mid_visual_valid = self._mid_stage_attention_scores(
            hidden_states,
            stage_ids,
            stage_attention,
            stage_positions,
            stage_labels,
            scoring_layer_idx,
        )
        mid_keep_mask = self._mid_stage_keep_mask(
            stage_ids,
            stage_attention,
            mid_scores,
            mid_visual_positions,
            mid_visual_valid,
        )
        hidden_states, stage_ids, stage_attention, stage_positions, stage_labels = self._pack_by_keep_mask(
            hidden_states,
            stage_ids,
            stage_attention,
            stage_positions,
            stage_labels,
            mid_keep_mask,
        )
        mid_visual_counts = (
            stage_ids.eq(int(IMAGE_TOKEN_INDEX)) & stage_attention.bool()
        ).sum(dim=-1)
        hidden_states = self._run_decoder_layer_range(
            hidden_states, stage_attention, stage_positions, mid_layer, wipe_layer
        )

        if self.enable_final_wipe:
            final_keep_mask = stage_attention.bool() & stage_ids.ne(int(IMAGE_TOKEN_INDEX))
            hidden_states, stage_ids, stage_attention, stage_positions, stage_labels = self._pack_by_keep_mask(
                hidden_states,
                stage_ids,
                stage_attention,
                stage_positions,
                stage_labels,
                final_keep_mask,
            )
        final_visual_counts = (
            stage_ids.eq(int(IMAGE_TOKEN_INDEX)) & stage_attention.bool()
        ).sum(dim=-1)
        hidden_states = self._run_decoder_layer_range(
            hidden_states, stage_attention, stage_positions, wipe_layer, num_layers
        )
        hidden_states = self.llava.model.norm(hidden_states)

        avg_visual_budget = (
            float(mid_layer) * scope_visual_counts.float()
            + float(wipe_layer - mid_layer) * mid_visual_counts.float()
            + float(num_layers - wipe_layer) * final_visual_counts.float()
        ) / float(max(num_layers, 1))
        stats = {
            "scope_visual_tokens": scope_visual_counts.float().mean(),
            "mid_visual_tokens": mid_visual_counts.float().mean(),
            "final_visual_tokens": final_visual_counts.float().mean(),
            "avg_visual_tokens_budget": avg_visual_budget.mean(),
            "student_sequence_tokens": stage_attention.sum(dim=-1).float().mean(),
        }
        return hidden_states, stage_attention, stage_labels, stats

    def _selector_loss(
        self,
        predictor_logits: torch.Tensor,
        teacher_scores: torch.Tensor,
        teacher_valid: torch.Tensor,
        keep_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keep_k = self.keep_k if keep_k is None else int(keep_k)
        valid = teacher_valid.bool()
        counts = valid.sum(dim=-1)
        metric_rows = counts > 1
        metric_denom = metric_rows.sum().clamp_min(1).float()
        pred = predictor_logits.float()
        teacher = teacher_scores.float()
        pos_inf = torch.tensor(torch.inf, device=pred.device, dtype=pred.dtype)
        neg_inf = torch.tensor(-torch.inf, device=pred.device, dtype=pred.dtype)

        teacher_ranks = torch.argsort(
            torch.argsort(teacher.masked_fill(~valid, pos_inf), dim=-1), dim=-1
        ).float()
        pred_ranks = torch.argsort(
            torch.argsort(pred.masked_fill(~valid, pos_inf), dim=-1), dim=-1
        ).float()
        rank_mean = (counts.float() - 1.0).clamp_min(0.0)[:, None] * 0.5
        teacher_ranks = (teacher_ranks - rank_mean) * valid
        pred_ranks = (pred_ranks - rank_mean) * valid
        spearman_per_row = (teacher_ranks * pred_ranks).sum(dim=-1) / (
            teacher_ranks.square().sum(dim=-1).sqrt()
            * pred_ranks.square().sum(dim=-1).sqrt()
        ).clamp_min(1e-6)
        selector_spearman = (spearman_per_row * metric_rows).sum() / metric_denom

        topk_count = min(keep_k, pred.shape[1])
        if topk_count == 0:
            zero = pred.sum() * 0.0
            return zero, selector_spearman, zero, zero
        teacher_top_idx = torch.topk(teacher.masked_fill(~valid, neg_inf), k=topk_count, dim=-1).indices
        pred_top_idx = torch.topk(pred.masked_fill(~valid, neg_inf), k=topk_count, dim=-1).indices
        teacher_top = torch.zeros_like(valid).scatter(1, teacher_top_idx, True) & valid
        pred_top = torch.zeros_like(valid).scatter(1, pred_top_idx, True) & valid
        keep_counts = counts.clamp(max=keep_k).clamp_min(1)
        overlap_per_row = (teacher_top & pred_top).sum(dim=-1).float() / keep_counts.float()
        selector_topk_overlap = (overlap_per_row * metric_rows).sum() / metric_denom

        boundary_rows = counts > keep_k
        boundary_denom = boundary_rows.sum().clamp_min(1).float()
        negatives = valid & ~teacher_top
        margin_scores = (pred + self.topk_hinge_margin * negatives).masked_fill(~valid, neg_inf)
        violating = torch.topk(margin_scores, k=topk_count, dim=-1).values.sum(dim=-1)
        positive = pred.gather(1, teacher_top_idx).sum(dim=-1)
        row_rank_loss = ((violating - positive) / float(topk_count)).clamp_min(0.0)
        rank_loss = (row_rank_loss * boundary_rows).sum() / metric_denom

        hard_neg_idx = torch.topk(pred.masked_fill(~negatives, neg_inf), k=topk_count, dim=-1).indices
        pos_logits = pred.gather(1, teacher_top_idx)
        neg_logits = pred.gather(1, hard_neg_idx)
        neg_counts = (counts - keep_k).clamp(min=0, max=topk_count)
        neg_slots = torch.arange(topk_count, device=pred.device)[None, :] < neg_counts[:, None]
        comparisons = (pos_logits[:, :, None] > neg_logits[:, None, :]) & neg_slots[:, None, :]
        comparison_count = (topk_count * neg_counts).clamp_min(1).float()
        boundary_per_row = comparisons.sum(dim=(1, 2)).float() / comparison_count
        selector_boundary_accuracy = (boundary_per_row * boundary_rows).sum() / boundary_denom
        return rank_loss, selector_spearman, selector_topk_overlap, selector_boundary_accuracy

    def _answer_token_mask(
        self,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        valid = attention_mask.bool()
        shifted_labels: Optional[torch.Tensor] = None
        if labels is not None and labels.shape == attention_mask.shape:
            shifted_labels = labels[:, 1:]
            valid = valid[:, :-1] & shifted_labels.ne(IGNORE_INDEX)
        return valid, shifted_labels

    def _teacher_answer_logits(
        self,
        full_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        valid, shifted_labels = self._answer_token_mask(attention_mask, labels)
        teacher_hidden = full_hidden_states[:, :-1, :] if shifted_labels is not None else full_hidden_states
        return self.llava.lm_head(teacher_hidden[valid]).detach()

    def _js_ce_losses(
        self,
        teacher_logits: torch.Tensor,
        pruned_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute JS divergence and CE from one shared student LM-head GEMM."""

        student_hidden = pruned_hidden_states
        valid, shifted_labels = self._answer_token_mask(attention_mask, labels)
        if shifted_labels is not None:
            student_hidden = student_hidden[:, :-1, :]
        if teacher_logits.shape[0] == 0:
            zero = student_hidden.sum() * 0.0
            return zero, zero

        student_logits = self.llava.lm_head(student_hidden[valid])
        temperature = max(self.kd_temperature, 1e-6)
        teacher_logp = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
        student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
        mixture_logp = torch.logaddexp(teacher_logp, student_logp) - math.log(2.0)
        js_per_token = 0.5 * (
            F.kl_div(mixture_logp, teacher_logp, log_target=True, reduction="none").sum(dim=-1)
            + F.kl_div(mixture_logp, student_logp, log_target=True, reduction="none").sum(dim=-1)
        )
        js_loss = js_per_token.mean() * (temperature * temperature)

        if shifted_labels is None:
            ce_loss = student_logits.sum() * 0.0
        elif temperature == 1.0:
            # At T=1 student_logp is exactly the log-softmax required by CE.
            # Reusing it avoids a second full-vocabulary fp32 log-softmax pass.
            ce_loss = F.nll_loss(
                student_logp,
                shifted_labels[valid].to(dtype=torch.long),
            )
        else:
            ce_loss = F.cross_entropy(
                student_logits.float(),
                shifted_labels[valid].to(dtype=torch.long),
            )
        return js_loss, ce_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_sizes: Optional[Any] = None,
        position_ids: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        image_tensor = images if images is not None else pixel_values
        if image_tensor is None:
            raise ValueError("Official LLaVA learnable-prune training requires images or pixel_values.")
        (
            inputs_embeds,
            expanded_input_ids,
            expanded_attention_mask,
            original_position_ids,
            expanded_labels,
            expanded_visual_coordinates,
        ) = self._build_multimodal_inputs(
            input_ids=input_ids,
            images=image_tensor,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            image_sizes=image_sizes,
        )

        with torch.no_grad():
            full_outputs, teacher_qkv, teacher_layer = self._full_forward_with_teacher_qkv(
                inputs_embeds=inputs_embeds,
                attention_mask=expanded_attention_mask,
                position_ids=original_position_ids,
            )
            teacher_scores, teacher_valid = self._teacher_targets(
                teacher_qkv,
                teacher_layer,
                expanded_input_ids,
                expanded_attention_mask,
                expanded_labels,
                original_position_ids,
                inputs_embeds=inputs_embeds,
                sample_ids=sample_ids,
            )
            teacher_logits = self._teacher_answer_logits(
                full_outputs.last_hidden_state,
                expanded_attention_mask,
                expanded_labels,
            )
            del teacher_qkv
            del full_outputs

        max_visual = teacher_valid.shape[1]
        if max_visual == 0:
            zero = sum(param.sum() for param in self.predictor.parameters()) * 0.0
            return {
                "loss": zero,
                "total_loss": zero.detach(),
                "logits": inputs_embeds.new_zeros((expanded_input_ids.shape[0], 0)),
                "rank_loss": zero.detach(),
                "selector_spearman": zero.detach(),
                "selector_topk_overlap": zero.detach(),
                "selector_boundary_accuracy": zero.detach(),
                "js_loss": zero.detach(),
                "ce_loss": zero.detach(),
                "retention_mean": zero.detach(),
                "retention_top_mean": zero.detach(),
                "retention_bottom_mean": zero.detach(),
                "soft_budget_error": zero.detach(),
                "weighted_rank_loss": zero.detach(),
                "weighted_js_loss": zero.detach(),
                "weighted_ce_loss": zero.detach(),
                "loss_weight_progress": zero.detach() + self._training_progress,
                "current_rank_loss_weight": zero.detach() + self._current_rank_loss_weight(),
                "current_js_loss_weight": zero.detach() + self._current_js_loss_weight(),
                "current_ce_loss_weight": zero.detach() + self._current_ce_loss_weight(),
                "scope_visual_tokens": zero.detach(),
                "mid_visual_tokens": zero.detach(),
                "final_visual_tokens": zero.detach(),
                "avg_visual_tokens_budget": zero.detach(),
                "student_sequence_tokens": expanded_attention_mask.sum(dim=-1).float().mean(),
                **self._budget_output_metrics(
                    zero,
                    total_loss=zero,
                    rank_loss=zero,
                    js_loss=zero,
                    ce_loss=zero,
                ),
            }

        visual_token_mask = expanded_input_ids.eq(int(IMAGE_TOKEN_INDEX)) & expanded_attention_mask.bool()
        visual_positions, visual_valid = self._pack_mask_positions(visual_token_mask)
        prompt_text_mask = (
            expanded_attention_mask.bool()
            & expanded_labels.eq(IGNORE_INDEX)
            & ~visual_token_mask
        )
        predictor_logits = self.predictor(
            inputs_embeds.detach(),
            visual_token_mask=visual_token_mask,
            text_token_mask=prompt_text_mask,
            attention_mask=expanded_attention_mask,
            visual_coordinates=expanded_visual_coordinates,
        )

        retention_probs = self._retention_probabilities(predictor_logits, visual_valid)
        selector_by_keep_k = {
            keep_k: self._selector_loss(
                predictor_logits=predictor_logits,
                teacher_scores=teacher_scores,
                teacher_valid=teacher_valid,
                keep_k=keep_k,
            )
            for keep_k in sorted({int(profile["keep_k"]) for profile in self.budget_profiles})
        }
        (
            _active_rank_loss,
            selector_spearman,
            selector_topk_overlap,
            selector_boundary_accuracy,
        ) = selector_by_keep_k[self.keep_k]
        # Ranking boundaries for every configured predictor TopK are cheap and
        # shared every step; only the expensive staged JS/CE branch is sampled.
        rank_loss = torch.stack([values[0] for values in selector_by_keep_k.values()]).mean()

        with self._temporary_decoder_checkpoint_training():
            (
                pruned_hidden_states,
                student_attention_mask,
                student_labels,
                stage_stats,
            ) = self._three_stage_student_forward(
                inputs_embeds=inputs_embeds,
                input_ids=expanded_input_ids,
                attention_mask=expanded_attention_mask,
                position_ids=original_position_ids,
                labels=expanded_labels,
                predictor_logits=predictor_logits,
                retention_probs=retention_probs,
                visual_positions=visual_positions,
                visual_valid=visual_valid,
            )
        js_loss, ce_loss = self._js_ce_losses(
            teacher_logits,
            pruned_hidden_states,
            student_attention_mask,
            labels=student_labels,
        )

        weighted_rank_loss = self._current_rank_loss_weight() * rank_loss
        weighted_js_loss = self._current_js_loss_weight() * js_loss
        weighted_ce_loss = self._current_ce_loss_weight() * ce_loss
        loss = weighted_rank_loss + weighted_js_loss + weighted_ce_loss

        finite_loss = torch.isfinite(loss)
        if loss.is_cuda:
            # Queue the check without synchronizing the training stream every
            # microbatch. A failing assertion is surfaced at the next natural
            # synchronization point (DDP/logging/backward).
            torch._assert_async(finite_loss, "Non-finite learnable-prune loss")
        elif not bool(finite_loss.item()):
            details = {
                "rank_loss": float(rank_loss.detach().float().item()),
                "js_loss": float(js_loss.detach().float().item()),
                "ce_loss": float(ce_loss.detach().float().item()),
            }
            raise FloatingPointError(f"Non-finite learnable-prune loss: {details}")

        detached_probs = retention_probs.detach().float()
        valid_probs = detached_probs[teacher_valid]
        retention_mean = valid_probs.mean() if valid_probs.numel() > 0 else loss.detach() * 0.0
        valid_counts = teacher_valid.sum(dim=-1)
        topk_count = min(self.keep_k, predictor_logits.shape[1])
        masked_pred = predictor_logits.detach().float().masked_fill(~teacher_valid, -torch.inf)
        top_idx = torch.topk(masked_pred, k=topk_count, dim=-1).indices
        top_mask = torch.zeros_like(teacher_valid).scatter(1, top_idx, True) & teacher_valid
        metric_rows = valid_counts > 1
        metric_denom = metric_rows.sum().clamp_min(1).float()
        keep_counts = valid_counts.clamp(max=self.keep_k).clamp_min(1)
        top_per_row = (detached_probs * top_mask).sum(dim=-1) / keep_counts.float()
        retention_top_mean = (top_per_row * metric_rows).sum() / metric_denom
        bottom_mask = teacher_valid & ~top_mask
        bottom_rows = valid_counts > self.keep_k
        bottom_counts = bottom_mask.sum(dim=-1).clamp_min(1)
        bottom_per_row = (detached_probs * bottom_mask).sum(dim=-1) / bottom_counts.float()
        retention_bottom_mean = (
            (bottom_per_row * bottom_rows).sum() / bottom_rows.sum().clamp_min(1).float()
        )

        return {
            "loss": loss,
            "total_loss": loss.detach(),
            "logits": predictor_logits,
            "rank_loss": rank_loss.detach(),
            "selector_spearman": selector_spearman.detach(),
            "selector_topk_overlap": selector_topk_overlap.detach(),
            "selector_boundary_accuracy": selector_boundary_accuracy.detach(),
            "js_loss": js_loss.detach(),
            "ce_loss": ce_loss.detach(),
            "retention_mean": retention_mean.detach(),
            "retention_top_mean": retention_top_mean.detach(),
            "retention_bottom_mean": retention_bottom_mean.detach(),
            "soft_budget_error": (self._last_soft_budget_error.detach() if self._last_soft_budget_error is not None else loss.detach() * 0.0),
            "weighted_rank_loss": weighted_rank_loss.detach(),
            "weighted_js_loss": weighted_js_loss.detach(),
            "weighted_ce_loss": weighted_ce_loss.detach(),
            "loss_weight_progress": loss.detach() * 0.0 + self._training_progress,
            "current_rank_loss_weight": loss.detach() * 0.0 + self._current_rank_loss_weight(),
            "current_js_loss_weight": loss.detach() * 0.0 + self._current_js_loss_weight(),
            "current_ce_loss_weight": loss.detach() * 0.0 + self._current_ce_loss_weight(),
            **{key: value.detach() for key, value in stage_stats.items()},
            **self._budget_output_metrics(
                loss,
                total_loss=loss,
                rank_loss=rank_loss,
                js_loss=js_loss,
                ce_loss=ce_loss,
            ),
        }


class LearnablePruneTrainer(Trainer):
    loss_metric_keys = (
        "total_loss",
        "rank_loss",
        "selector_spearman",
        "selector_topk_overlap",
        "selector_boundary_accuracy",
        "js_loss",
        "ce_loss",
        "retention_mean",
        "retention_top_mean",
        "retention_bottom_mean",
        "soft_budget_error",
        "weighted_rank_loss",
        "weighted_js_loss",
        "weighted_ce_loss",
        "loss_weight_progress",
        "current_rank_loss_weight",
        "current_js_loss_weight",
        "current_ce_loss_weight",
        "scope_visual_tokens",
        "mid_visual_tokens",
        "final_visual_tokens",
        "avg_visual_tokens_budget",
        "student_sequence_tokens",
        "active_budget_profile_index",
        "active_avg_token_budget",
        "active_keep_k",
        "active_scope_target_count",
        "active_mid_target_count",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self._pending_loss_metrics: List[Dict[str, torch.Tensor]] = []
        self._scale_total_steps_cache: Optional[int] = None
        self._budget_micro_step: Optional[int] = None

    def _metric_scalars(self, values: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Reduce every pending scalar with one cross-rank collective."""

        if not values:
            return {}
        keys = sorted(values)
        packed = torch.stack([values[key].detach().float().reshape(()) for key in keys]).unsqueeze(0)
        if hasattr(self, "accelerator") and self.accelerator is not None:
            packed = self.accelerator.gather_for_metrics(packed)
        packed = packed.reshape(-1, len(keys)).mean(dim=0).to(device="cpu")
        return {key: float(value) for key, value in zip(keys, packed.tolist())}

    def _current_train_progress(self) -> float:
        total_steps = self._infer_total_train_steps()
        if total_steps <= 1:
            return 0.0
        step = int(getattr(self.state, "global_step", 0) or 0)
        return float(min(max(step / float(max(1, total_steps - 1)), 0.0), 1.0))

    def _infer_total_train_steps(self) -> int:
        for value in (getattr(self.state, "max_steps", 0), getattr(self.args, "max_steps", 0)):
            value = int(value or 0)
            if value > 0:
                self._scale_total_steps_cache = value
                return value

        if self._scale_total_steps_cache is not None:
            return self._scale_total_steps_cache

        try:
            dataloader_len = len(self.get_train_dataloader())
        except Exception:
            dataloader_len = 0
        if dataloader_len <= 0:
            return 0

        grad_accum = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1) or 1))
        updates_per_epoch = max(1, math.ceil(dataloader_len / grad_accum))
        num_epochs = max(float(getattr(self.args, "num_train_epochs", 1.0) or 1.0), 0.0)
        self._scale_total_steps_cache = max(1, math.ceil(updates_per_epoch * num_epochs))
        return self._scale_total_steps_cache

    @staticmethod
    def _maybe_set_training_progress(model: nn.Module, progress: float) -> None:
        candidates = [model, getattr(model, "module", None)]
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "set_training_progress"):
                candidate.set_training_progress(progress)
                return

    @staticmethod
    def _maybe_set_training_log_step(model: nn.Module, global_step: int) -> None:
        candidates = [model, getattr(model, "module", None)]
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "set_training_log_step"):
                candidate.set_training_log_step(global_step)
                return

    @staticmethod
    def _maybe_set_budget_micro_step(model: nn.Module, micro_step: int) -> None:
        candidates = [model, getattr(model, "module", None)]
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "set_budget_micro_step"):
                candidate.set_budget_micro_step(micro_step)
                return

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self._budget_micro_step is None:
            global_step = int(getattr(self.state, "global_step", 0) or 0)
            grad_accum = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1) or 1))
            self._budget_micro_step = global_step * grad_accum
        self._maybe_set_budget_micro_step(model, self._budget_micro_step)
        self._budget_micro_step += 1
        self._maybe_set_training_progress(model, self._current_train_progress())
        self._maybe_set_training_log_step(model, int(getattr(self.state, "global_step", 0) or 0))
        outputs = model(**inputs)
        loss = outputs["loss"]
        self._pending_loss_metrics.append(
            {
                key: value.detach().float().reshape(1)
                for key, value in outputs.items()
                if torch.is_tensor(value) and (key in self.loss_metric_keys or key.startswith("budget_"))
            }
        )
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if self._pending_loss_metrics:
            keys = sorted({key for metrics in self._pending_loss_metrics for key in metrics})
            local_means: Dict[str, torch.Tensor] = {}
            for key in keys:
                values = [metrics[key] for metrics in self._pending_loss_metrics if key in metrics]
                if values:
                    local_means[key] = torch.stack(values, dim=0).mean()
            if torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
                local_means["cuda_peak_memory_gib"] = torch.tensor(
                    torch.cuda.max_memory_allocated(device) / float(1 << 30), device=device
                )
                local_means["cuda_reserved_memory_gib"] = torch.tensor(
                    torch.cuda.memory_reserved(device) / float(1 << 30), device=device
                )
            logs.update(self._metric_scalars(local_means))
            self._pending_loss_metrics.clear()
        super().log(logs, start_time=start_time)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if not self.is_world_process_zero():
            return
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
        if hasattr(model, "module"):
            model = model.module
        torch.save(model.predictor.state_dict(), os.path.join(output_dir, "predictor.pt"))
        default_profile = model.budget_profiles[0]
        config = {
            "training_variant": "three_stage_scope_mid_finalwipe_fixed_layer_js_packed_hard_st"
            + ("_multibudget" if len(model.budget_profiles) > 1 else "")
            + "_tclm_multihead_rank_swiglu",
            "keep_k": int(default_profile["keep_k"]),
            "teacher_layer": model.teacher_layer,
            "rank_loss_weight": model.rank_loss_weight,
            "js_loss_weight": model.js_loss_weight,
            "ce_loss_weight": model.ce_loss_weight,
            "enable_scale": model.enable_scale,
            "enable_rss": model.enable_rss,
            "topk_hinge_margin": model.topk_hinge_margin,
            "teacher_layer_quality": "fixed_layer",
            "teacher_layer_pooling": "fixed_single_layer",
            "kd_temperature": model.kd_temperature,
            "attention_gate_mode": "packed_embedding_hard_st",
            "budgeted_soft_topk_iters": model.budgeted_soft_topk_iters,
            "training_pruning_mode": "three_stage",
            "scope_target_count": int(default_profile["scope_target_count"]),
            "mid_pruning_layer_idx": int(default_profile["mid_pruning_layer_idx"]),
            "mid_target_count": int(default_profile["mid_target_count"]),
            "mid_attn_anchor": "query",
            "final_wipe_layer_idx": int(default_profile["final_wipe_layer_idx"]),
            "enable_final_wipe": bool(default_profile["enable_final_wipe"]),
            "budget_profiles": [dict(profile) for profile in model.budget_profiles],
            "default_avg_token_budget": int(default_profile["avg_token_budget"]),
            "budget_schedule": "balanced_shuffled_microbatch",
            "budget_schedule_seed": model.budget_schedule_seed,
            "teacher_target_logging": model._teacher_target_logger is not None or model.teacher_target_log_to_console,
            "predictor_input_context": "pre_llm_visual_tokens_plus_unlabeled_prompt_tokens",
            "distillation_loss": "jensen_shannon_full_vocabulary",
            "distillation_mask_type": "physical_scope_mid_finalwipe_with_packed_embedding_st",
            "predictor_input_size": model.predictor.input_size,
            "predictor_hidden_size": model.predictor.hidden_size,
            "predictor_rank": getattr(model.predictor, "rank", None),
            "predictor_num_heads": getattr(model.predictor, "num_heads", None),
            "predictor_rank_mlp_ratio": getattr(model.predictor, "rank_mlp_ratio", None),
            "predictor_use_visual_position": getattr(model.predictor, "use_visual_position", None),
            "predictor_use_true_visual_coordinates": True,
            "predictor_attention_implementation": "fused_sdpa",
            "predictor_use_text_position": getattr(model.predictor, "use_text_position", None),
        }
        torch.save(config, os.path.join(output_dir, "learnable_prune_config.pt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stage-aligned lightweight learnable visual-token pruner")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--vision_tower", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="Randomly sample this fraction of the training set after max_samples is applied. Use 0.1 for 10%%.",
    )
    parser.add_argument("--teacher_layer", type=int, default=18)
    parser.add_argument("--predictor_hidden_size", type=int, default=512)
    parser.add_argument("--predictor_rank", type=int, default=256)
    parser.add_argument("--predictor_num_heads", type=int, default=4)
    parser.add_argument("--predictor_rank_mlp_ratio", type=int, default=2)
    parser.add_argument("--predictor_use_visual_position", type=parse_bool_flag, default=True)
    parser.add_argument("--predictor_use_text_position", type=parse_bool_flag, default=True)

    parser.add_argument("--rank_loss_weight", type=float, default=0.5)
    parser.add_argument(
        "--js_loss_weight",
        "--kl_loss_weight",
        dest="js_loss_weight",
        type=float,
        default=6.0,
        help="JS-divergence loss weight. --kl_loss_weight remains a deprecated compatibility alias.",
    )
    parser.add_argument("--ce_loss_weight", type=float, default=0.5)
    parser.add_argument("--enable_scale", type=parse_bool_flag, default=True)
    parser.add_argument("--enable_rss", type=parse_bool_flag, default=False)
    parser.add_argument("--topk_hinge_margin", type=float, default=1.0)
    parser.add_argument("--kd_temperature", type=float, default=1.0)
    parser.add_argument("--budgeted_soft_topk_iters", type=int, default=16)
    parser.add_argument(
        "--budget_profiles",
        type=parse_budget_profiles,
        required=True,
        help=(
            "Semicolon-separated avg:topk:scope:mid_layer:mid_target:final_layer profiles. "
            "One profile is activated for every microbatch using a balanced shuffled cycle."
        ),
    )
    parser.add_argument(
        "--budget_schedule_seed",
        type=int,
        default=None,
        help="Seed for the DDP-identical balanced shuffled budget cycle; defaults to --seed.",
    )
    parser.add_argument("--teacher_target_log_dir", type=str, default=None)
    parser.add_argument("--teacher_target_log_to_console", type=parse_bool_flag, default=False)

    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--gradient_checkpointing", type=parse_bool_flag, default=True)
    parser.add_argument(
        "--ddp_find_unused_parameters",
        type=parse_bool_flag,
        default=False,
        help="Passed to TrainingArguments. Keep false unless debugging dynamic unused-parameter errors.",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_distributed_cuda_mapping()
    seed_everything(args.seed)
    if not (0.0 < args.sample_rate <= 1.0):
        raise ValueError("--sample_rate must be in (0, 1].")
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    if args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    config_dict, _ = PretrainedConfig.get_config_dict(args.model_name_or_path)
    config = LlavaConfig(**config_dict)
    vision_tower_name = args.vision_tower or getattr(config, "mm_vision_tower", None)
    resolved_vision_tower = resolve_vision_tower_path(vision_tower_name)
    if resolved_vision_tower is not None:
        config.mm_vision_tower = resolved_vision_tower

    llava = LlavaLlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    llava.config.use_cache = False
    llava.config.tokenizer_padding_side = tokenizer.padding_side
    llava.config.tokenizer_model_max_length = tokenizer.model_max_length
    llava.config.image_aspect_ratio = args.image_aspect_ratio
    llava.config._attn_implementation = args.attn_implementation
    vision_tower = llava.get_vision_tower()
    if vision_tower is not None:
        vision_tower.load_model()
        vision_tower.to(dtype=dtype)
    if args.gradient_checkpointing and hasattr(llava, "gradient_checkpointing_enable"):
        gradient_checkpointing_kwargs = {
            "use_reentrant": False,
            "context_fn": _default_checkpoint_contexts,
        }
        try:
            llava.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        except TypeError:
            llava.gradient_checkpointing_enable()

    hidden_size = int(llava.config.hidden_size)
    predictor = LearnablePrunePredictor(
        input_size=hidden_size,
        hidden_size=args.predictor_hidden_size,
        rank=args.predictor_rank,
        num_heads=args.predictor_num_heads,
        rank_mlp_ratio=args.predictor_rank_mlp_ratio,
        use_visual_position=args.predictor_use_visual_position,
        use_text_position=args.predictor_use_text_position,
    )
    model = LearnablePruneWrapper(
        llava_model=llava,
        predictor=predictor,
        budget_profiles=args.budget_profiles,
        teacher_layer=args.teacher_layer,
        rank_loss_weight=args.rank_loss_weight,
        js_loss_weight=args.js_loss_weight,
        ce_loss_weight=args.ce_loss_weight,
        enable_scale=args.enable_scale,
        enable_rss=args.enable_rss,
        topk_hinge_margin=args.topk_hinge_margin,
        kd_temperature=args.kd_temperature,
        budgeted_soft_topk_iters=args.budgeted_soft_topk_iters,
        budget_schedule_seed=(args.seed if args.budget_schedule_seed is None else args.budget_schedule_seed),
        teacher_target_log_dir=args.teacher_target_log_dir,
        teacher_target_log_to_console=args.teacher_target_log_to_console,
    )

    data_path = args.data_path or os.path.join(args.data_dir, "llava_v1_5_mix665k.json")
    if not os.path.exists(data_path):
        alternatives = sorted(Path(args.data_dir).glob("llava_v1_5_mix665k*.json"))
        if not alternatives:
            raise FileNotFoundError(f"Missing llava_v1_5_mix665k.json in {args.data_dir}")
        data_path = str(alternatives[0])
    image_folder = args.image_folder or os.path.join(args.data_dir, "images")
    data_args = SimpleDataArguments(
        data_path=data_path,
        image_folder=image_folder,
        image_processor=vision_tower.image_processor,
        image_aspect_ratio=args.image_aspect_ratio,
        image_grid_pinpoints=getattr(llava.config, "image_grid_pinpoints", None),
    )
    dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    if args.max_samples is not None:
        dataset.list_data_dict = dataset.list_data_dict[: args.max_samples]
    if args.sample_rate < 1.0:
        sample_count = max(1, int(len(dataset) * args.sample_rate))
        generator = torch.Generator().manual_seed(args.seed)
        sample_indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
        dataset = IndexedDataset(dataset, sample_indices)
        print(f"Using sample_rate={args.sample_rate:g}: sampled {sample_count} training examples.")
    else:
        dataset = IndexedDataset(dataset)

    wandb_available = importlib.util.find_spec("wandb") is not None
    wandb_disabled = str(os.environ.get("WANDB_MODE", "")).lower() == "disabled"
    report_to = "wandb" if args.wandb_project and wandb_available and not wandb_disabled else "none"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        ddp_broadcast_buffers=False,
        report_to=report_to,
        run_name=args.run_name,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        dataloader_prefetch_factor=(args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None),
    )
    trainer = LearnablePruneTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
        if vision_tower is not None and getattr(vision_tower, "image_processor", None) is not None:
            vision_tower.image_processor.save_pretrained(args.output_dir)
    _cleanup_distributed()


if __name__ == "__main__":
    main()
