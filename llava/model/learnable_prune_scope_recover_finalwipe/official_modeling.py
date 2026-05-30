#!/usr/bin/env python3
# coding: utf-8
"""Official LLaVA learnable-prune + SCOPE recover + final-wipe inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaForCausalLM

try:
    from transformers.models.llama.modeling_llama import create_causal_mask as llama_create_causal_mask
except ImportError:
    llama_create_causal_mask = None

from learnable_pruner.predictor import LearnablePrunePredictor
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM, LlavaLlamaModel


DEFAULT_CHECKPOINT = "/data1/chenzixuan/train_output/official_llava_learnable_prune_precision_at_k_hinge_top64_layers16_24_top1_iqr_sample0.2"
LEARNABLE_TOPK = 64
MERGE_TARGET_COUNT = 4
SCOPE_TARGET_COUNT = 103
RECOVER_LAYER_IDX = 12
FINAL_WIPE_LAYER_IDX = 24
ENABLE_FINALWIPE = True


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value.strip())


def _enable_finalwipe() -> bool:
    return _env_flag("ENABLE_FINALWIPE", ENABLE_FINALWIPE)


def _scope_target_count() -> int:
    return _env_int("SCOPE_TARGET_COUNT", SCOPE_TARGET_COUNT)


def _recover_layer_idx() -> int:
    return _env_int("RECOVER_LAYER_IDX", RECOVER_LAYER_IDX)


def _final_wipe_layer_idx() -> int:
    return _env_int("FINAL_WIPE_LAYER_IDX", FINAL_WIPE_LAYER_IDX)


@torch.no_grad()
def SeededResidualSCOPE(
    visual_feature_vectors: torch.Tensor,
    seed_relative: torch.Tensor,
    target_keep: int,
    spatial_bonus: float = 0.0,
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

    while int(selected.sum().item()) < target_keep:
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


class LlavaLearnablePruneScopeRecoverFinalwipeModel(LlavaLlamaModel):
    config_class = LlavaConfig


class LlavaLearnablePruneScopeRecoverFinalwipeForCausalLM(LlavaLlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlavaLlamaForCausalLM, self).__init__(config)
        self.model = LlavaLearnablePruneScopeRecoverFinalwipeModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._scope_recover_finalwipe_applied = False
        self._scope_recover_finalwipe_next_position_id = None
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
        predictor = LearnablePrunePredictor(
            input_size=int(config["predictor_input_size"]),
            hidden_size=int(config["predictor_hidden_size"]),
            num_heads=int(config["predictor_heads"]),
            mlp_ratio=int(config.get("predictor_mlp_ratio", 2)),
            num_layers=int(config.get("predictor_layers", 4)),
            use_final_full_attention=bool(config.get("predictor_use_final_full_attention", True)),
        )
        predictor.load_state_dict(torch.load(checkpoint / "predictor.pt", map_location="cpu"))
        device, _ = self._device_dtype()
        predictor.to(device=device, dtype=torch.float32)
        predictor.eval()
        for param in predictor.parameters():
            param.requires_grad = False
        self.learnable_prune_predictor = predictor
        self.learnable_prune_config = dict(config)
        self.learnable_prune_keep_k = int(config.get("keep_k", LEARNABLE_TOPK))
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

    @staticmethod
    def _cluster_supplement_tokens_for_recover(
        visual_embeds: torch.Tensor,
        supplement_relative: torch.Tensor,
        merge_target_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if supplement_relative.numel() == 0:
            empty_visual = visual_embeds[:, :0, :]
            empty_idx = supplement_relative.new_empty((0,))
            return empty_visual, empty_idx, empty_idx

        supplement_visual = visual_embeds.index_select(1, supplement_relative)
        supplement_count = int(supplement_visual.shape[1])
        merge_count = min(max(int(merge_target_count), 1), supplement_count)
        if supplement_count <= merge_count:
            child_to_parent = torch.arange(supplement_count, dtype=torch.long, device=visual_embeds.device)
            return supplement_visual, supplement_relative, child_to_parent

        normed = F.normalize(supplement_visual[0].float(), p=2, dim=-1)
        sim = torch.matmul(normed, normed.transpose(0, 1))
        centers = [int(sim.mean(dim=1).argmax().item())]
        while len(centers) < merge_count:
            selected = torch.tensor(centers, dtype=torch.long, device=visual_embeds.device)
            max_sim_to_selected = sim.index_select(1, selected).max(dim=1).values
            max_sim_to_selected[selected] = float("inf")
            centers.append(int(max_sim_to_selected.argmin().item()))

        center_local = torch.tensor(centers, dtype=torch.long, device=visual_embeds.device)
        center_sim = sim.index_select(1, center_local)
        child_to_parent = center_sim.argmax(dim=1).to(dtype=torch.long)
        child_to_parent[center_local] = torch.arange(merge_count, dtype=torch.long, device=visual_embeds.device)

        pooled = []
        for cluster_idx, center_idx in enumerate(centers):
            member_mask = child_to_parent.eq(cluster_idx)
            member_indices = member_mask.nonzero(as_tuple=False).flatten()
            if member_indices.numel() == 0:
                member_indices = center_local.new_tensor([center_idx])
                child_to_parent[member_indices] = cluster_idx
            weights = torch.softmax(sim[member_indices, center_idx], dim=0).to(dtype=supplement_visual.dtype)
            pooled.append((supplement_visual[:, member_indices, :] * weights.view(1, -1, 1)).sum(dim=1, keepdim=True))

        return torch.cat(pooled, dim=1), supplement_relative.index_select(0, center_local), child_to_parent

    def _embed_multimodal_for_generation(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        image_sizes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model_device, _ = self._device_dtype()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        image_features = self.encode_images(images)
        if torch.is_tensor(image_features):
            image_features = [feat for feat in image_features]
        else:
            image_features = list(image_features)

        compact_input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        new_input_embeds: List[torch.Tensor] = []
        new_input_ids: List[torch.Tensor] = []
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
            for i in range(num_images + 1):
                cur_new_embeds.append(text_embeds_split[i])
                cur_new_ids.append(cur_input_ids_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx].to(
                        device=text_embeds_split[i].device,
                        dtype=text_embeds_split[i].dtype,
                    )
                    cur_image_idx += 1
                    visual_len = cur_image_features.shape[0]
                    cur_new_embeds.append(cur_image_features)
                    cur_new_ids.append(torch.full((visual_len,), IMAGE_TOKEN_INDEX, device=cur_input_ids.device, dtype=cur_input_ids.dtype))

            new_input_embeds.append(torch.cat([x.to(model_device) for x in cur_new_embeds], dim=0))
            new_input_ids.append(torch.cat(cur_new_ids, dim=0))

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        pad_token_id = int(getattr(self.config, "pad_token_id", 0) or 0)
        padded_embeds = []
        padded_ids = torch.full((batch_size, max_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        padded_attention = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        padded_position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

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

        return torch.stack(padded_embeds, dim=0), padded_ids, padded_attention, padded_position_ids

    @torch.no_grad()
    def _learnable_scope_prune_prefill(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if input_ids.shape[0] != 1:
            raise ValueError("learnable-prune scope-recover-finalwipe generation currently expects batch_size=1")
        self._ensure_learnable_prune_loaded()

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
            return inputs_embeds, input_ids[:, valid_positions], attention_mask, position_ids, None

        predictor = self.learnable_prune_predictor
        predictor_dtype = next(predictor.parameters()).dtype
        text_positions = (valid_mask & ~visual_token_mask)[0].nonzero(as_tuple=False).flatten()
        scores = predictor.score_visual_text_by_positions(
            inputs_embeds.to(dtype=predictor_dtype),
            visual_positions=visual_positions,
            text_positions=text_positions,
        )[0]
        learnable_topk = int(getattr(self, "learnable_prune_keep_k", LEARNABLE_TOPK))
        topk = min(learnable_topk, int(scores.numel()))
        top_relative = torch.topk(scores, k=topk).indices.sort().values if topk > 0 else scores.new_empty((0,), dtype=torch.long)

        target_keep = min(_scope_target_count(), int(scores.numel()))
        visual_embeds = inputs_embeds.index_select(1, visual_positions)
        if target_keep > topk:
            seeded_scope_rank, _ = SeededResidualSCOPE(
                visual_feature_vectors=visual_embeds.to(dtype=torch.float32),
                seed_relative=top_relative,
                target_keep=target_keep,
                spatial_bonus=0.0,
            )
            scope_relative = seeded_scope_rank[0]
        else:
            scope_relative = top_relative[:target_keep].sort().values

        in_top = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
        if top_relative.numel() > 0:
            in_top[top_relative] = True
        supplement_relative = scope_relative[~in_top.index_select(0, scope_relative)]

        top_visual = visual_embeds.index_select(1, top_relative) if top_relative.numel() > 0 else visual_embeds[:, :0, :]
        supplement_visual = (
            visual_embeds.index_select(1, supplement_relative)
            if supplement_relative.numel() > 0
            else visual_embeds[:, :0, :]
        )
        (
            supplement_parent_visual,
            supplement_center_relative,
            supplement_child_to_parent,
        ) = self._cluster_supplement_tokens_for_recover(
            visual_embeds=visual_embeds,
            supplement_relative=supplement_relative,
            merge_target_count=MERGE_TARGET_COUNT,
        )

        parent_visual = torch.cat([top_visual, supplement_parent_visual], dim=1)
        child_hidden_at_merge = torch.cat([top_visual, supplement_visual], dim=1)
        top_child_to_parent = torch.arange(top_visual.shape[1], dtype=torch.long, device=input_ids.device)
        child_to_parent = torch.cat([top_child_to_parent, top_visual.shape[1] + supplement_child_to_parent], dim=0)
        calibration_child_indices = top_child_to_parent

        image_start_idx = int(visual_positions[0].item())
        image_end_idx = int(visual_positions[-1].item()) + 1
        valid_positions = attention_mask[0].bool().nonzero(as_tuple=False).flatten()
        pre_positions = valid_positions[valid_positions < image_start_idx]
        post_positions = valid_positions[valid_positions >= image_end_idx]
        compact_visual_relative = torch.cat([top_relative, supplement_center_relative], dim=0)
        scope_visual_relative = torch.cat([top_relative, supplement_relative], dim=0)
        compact_visual_positions = visual_positions.index_select(0, compact_visual_relative)
        scope_visual_positions = visual_positions.index_select(0, scope_visual_relative)

        compact_keep_positions = torch.cat([pre_positions, compact_visual_positions, post_positions], dim=0)
        scope_keep_positions = torch.cat([pre_positions, scope_visual_positions, post_positions], dim=0)
        compact_embeds = torch.cat(
            [
                inputs_embeds.index_select(1, pre_positions),
                parent_visual,
                inputs_embeds.index_select(1, post_positions),
            ],
            dim=1,
        )
        compact_input_ids = torch.cat(
            [
                input_ids.index_select(1, pre_positions),
                input_ids.new_full((1, parent_visual.shape[1]), IMAGE_TOKEN_INDEX),
                input_ids.index_select(1, post_positions),
            ],
            dim=1,
        )
        compact_attention_mask = attention_mask.new_ones((1, compact_embeds.shape[1]))
        compact_position_ids = torch.cat(
            [
                position_ids.index_select(1, pre_positions),
                position_ids.index_select(1, compact_visual_positions),
                position_ids.index_select(1, post_positions),
            ],
            dim=1,
        )

        recovered_input_ids = torch.cat(
            [
                input_ids.index_select(1, pre_positions),
                input_ids.new_full((1, child_hidden_at_merge.shape[1]), IMAGE_TOKEN_INDEX),
                input_ids.index_select(1, post_positions),
            ],
            dim=1,
        )
        recovered_position_ids = torch.cat(
            [
                position_ids.index_select(1, pre_positions),
                position_ids.index_select(1, scope_visual_positions),
                position_ids.index_select(1, post_positions),
            ],
            dim=1,
        )
        recovered_attention_mask = attention_mask.new_ones((1, recovered_input_ids.shape[1]))
        recover_state = {
            "image_start_idx": torch.tensor(image_start_idx, dtype=torch.long, device=input_ids.device),
            "compact_image_end_idx": torch.tensor(image_start_idx + parent_visual.shape[1], dtype=torch.long, device=input_ids.device),
            "recovered_image_end_idx": torch.tensor(image_start_idx + child_hidden_at_merge.shape[1], dtype=torch.long, device=input_ids.device),
            "parent_hidden_at_merge": parent_visual.detach(),
            "child_hidden_at_merge": child_hidden_at_merge.detach(),
            "child_to_parent": child_to_parent.detach(),
            "calibration_child_indices": calibration_child_indices.detach(),
            "recovered_input_ids": recovered_input_ids,
            "recovered_attention_mask": recovered_attention_mask,
            "recovered_position_ids": recovered_position_ids,
            "recovered_keep_positions": scope_keep_positions.unsqueeze(0),
            "compact_keep_positions": compact_keep_positions.unsqueeze(0),
        }
        self.learnable_prune_stats.append(
            {
                "original_tokens": int(input_ids.shape[1]),
                "original_visual_tokens": int(visual_positions.numel()),
                "learnable_topk_visual_tokens": int(topk),
                "scope_visual_tokens": int(scope_relative.numel()),
                "supplement_visual_tokens": int(supplement_relative.numel()),
                "merged_supplement_visual_tokens": int(supplement_parent_visual.shape[1]),
                "kept_visual_tokens_before_recover": int(parent_visual.shape[1]),
                "scope_target_visual_tokens": int(target_keep),
                "diversity_fill_method": "seeded_residual_scope",
                "supplement_merge_method": "similarity_center_softmax_pool",
                "pruned_tokens": int(input_ids.shape[1] - compact_keep_positions.numel()),
                "checkpoint": getattr(self, "learnable_prune_checkpoint", None),
            }
        )
        return compact_embeds, compact_input_ids, compact_attention_mask, compact_position_ids, recover_state

    def _estimate_recover_per_head_gamma(
        self,
        current_visual: torch.Tensor,
        parent_hidden_at_merge: torch.Tensor,
        child_to_parent: torch.LongTensor,
        child_hidden_at_merge: torch.Tensor,
        calibration_child_indices: Optional[torch.LongTensor] = None,
        min_calib_groups: int = 4,
    ) -> Optional[torch.Tensor]:
        del parent_hidden_at_merge
        if current_visual.size(0) != 1 or child_hidden_at_merge.size(0) != 1:
            return None
        hidden_size = int(child_hidden_at_merge.size(-1))
        num_heads = int(getattr(self.config, "num_attention_heads", 1))
        if num_heads <= 0 or hidden_size % num_heads != 0:
            return None
        if calibration_child_indices is None or calibration_child_indices.numel() < 4 * min_calib_groups:
            return None

        calibration_child_indices = calibration_child_indices.to(device=child_hidden_at_merge.device, dtype=torch.long)
        calibration_child_indices = calibration_child_indices[
            (calibration_child_indices >= 0) & (calibration_child_indices < child_hidden_at_merge.size(1))
        ]
        group_count = int(calibration_child_indices.numel()) // 4
        if group_count < min_calib_groups:
            return None
        calibration_child_indices = calibration_child_indices[: group_count * 4].view(group_count, 4)
        calibration_parent_indices = child_to_parent.index_select(0, calibration_child_indices.reshape(-1)).view(group_count, 4)

        shallow_children = child_hidden_at_merge[:, calibration_child_indices.reshape(-1), :].view(group_count, 4, hidden_size)
        deep_children = current_visual[:, calibration_parent_indices.reshape(-1), :].view(group_count, 4, hidden_size)

        shallow_norm = F.normalize(shallow_children.float(), p=2, dim=-1)
        sim_matrix = torch.matmul(shallow_norm, shallow_norm.transpose(1, 2))
        weights = torch.softmax(sim_matrix.mean(dim=-1), dim=1).to(dtype=child_hidden_at_merge.dtype)

        shallow_center = (shallow_children * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
        deep_center = (deep_children * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
        head_dim = hidden_size // num_heads
        shallow_dev = (shallow_children - shallow_center).float().view(-1, num_heads, head_dim)
        deep_dev = (deep_children - deep_center).float().view(-1, num_heads, head_dim)

        denominator = shallow_dev.square().sum(dim=(0, 2)).clamp(min=1e-6)
        numerator = (deep_dev * shallow_dev).sum(dim=(0, 2))
        gamma = torch.nan_to_num(numerator / denominator, nan=1.0, posinf=1.0, neginf=1.0)
        gamma = gamma.clamp(min=0.5, max=1.5)
        return gamma.to(device=child_hidden_at_merge.device, dtype=child_hidden_at_merge.dtype)

    def _recover_quadtree_children_from_residual(
        self,
        hidden_states: torch.Tensor,
        image_start_idx: int,
        image_end_idx: int,
        parent_hidden_at_merge: torch.Tensor,
        child_to_parent: torch.LongTensor,
        child_hidden_at_merge: torch.Tensor,
        calibration_child_indices: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        current_visual = hidden_states[:, image_start_idx:image_end_idx, :]
        parent_current = current_visual.index_select(1, child_to_parent)
        parent_at_merge = parent_hidden_at_merge.index_select(1, child_to_parent)
        child_deviation = child_hidden_at_merge - parent_at_merge

        gamma = self._estimate_recover_per_head_gamma(
            current_visual=current_visual,
            parent_hidden_at_merge=parent_hidden_at_merge,
            child_to_parent=child_to_parent,
            child_hidden_at_merge=child_hidden_at_merge,
            calibration_child_indices=calibration_child_indices,
        )
        if gamma is not None:
            hidden_size = int(child_deviation.size(-1))
            num_heads = int(getattr(self.config, "num_attention_heads", 1))
            if num_heads > 0 and hidden_size % num_heads == 0:
                head_dim = hidden_size // num_heads
                child_deviation = (
                    child_deviation.view(child_deviation.size(0), child_deviation.size(1), num_heads, head_dim)
                    * gamma.view(1, 1, num_heads, 1)
                ).view_as(child_deviation)

        recovered_visual = parent_current + child_deviation
        return torch.cat(
            [
                hidden_states[:, :image_start_idx, :],
                recovered_visual,
                hidden_states[:, image_end_idx:, :],
            ],
            dim=1,
        )

    def _wipe_visual_tokens(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    ):
        position_embeddings = None
        if hasattr(self.model, "rotary_emb"):
            try:
                position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
            except TypeError:
                position_embeddings = None
        new_kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_value,
            "cache_position": cache_position,
            "output_attentions": output_attentions,
            "use_cache": use_cache,
        }
        if position_embeddings is not None:
            new_kwargs["position_embeddings"] = position_embeddings
        old_kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_value": past_key_value,
            "output_attentions": output_attentions,
            "use_cache": use_cache,
        }
        try:
            outputs = layer(hidden_states, **new_kwargs)
        except TypeError:
            try:
                compat_kwargs = dict(old_kwargs)
                compat_kwargs["cache_position"] = cache_position
                if position_embeddings is not None:
                    compat_kwargs["position_embeddings"] = position_embeddings
                outputs = layer(hidden_states, **compat_kwargs)
            except TypeError:
                outputs = layer(hidden_states, **old_kwargs)
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
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
    ):
        inputs_embeds, scoped_input_ids, attention_mask, position_ids, recover_state = self._learnable_scope_prune_prefill(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        layers = self.model.layers
        enable_finalwipe = _enable_finalwipe()
        recover_layer_idx = min(_recover_layer_idx(), len(layers))
        wipe_layer_idx = min(_final_wipe_layer_idx(), len(layers)) if enable_finalwipe else len(layers)
        wipe_layer_idx = max(wipe_layer_idx, recover_layer_idx)
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = DynamicCache(config=self.model.config) if use_cache else None

        def _run_layer_span(
            start_idx: int,
            end_idx: int,
            curr_hidden_states: torch.Tensor,
            curr_attention_mask: torch.Tensor,
            curr_position_ids: torch.Tensor,
        ) -> torch.Tensor:
            nonlocal all_hidden_states, all_attns, next_cache
            if start_idx >= end_idx:
                return curr_hidden_states
            curr_cache_position = torch.arange(curr_hidden_states.shape[1], device=curr_hidden_states.device, dtype=torch.long)
            curr_mask = self._prepare_mask(
                curr_attention_mask,
                curr_hidden_states,
                position_ids=curr_position_ids,
                cache_position=curr_cache_position,
            )
            for layer_idx in range(start_idx, end_idx):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (curr_hidden_states,)
                curr_hidden_states, attn, present = self._run_layer(
                    layers[layer_idx],
                    curr_hidden_states,
                    curr_mask,
                    curr_position_ids,
                    past_key_value=next_cache,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cache_position=curr_cache_position,
                )
                if use_cache:
                    if isinstance(next_cache, list):
                        next_cache.append(present)
                if output_attentions:
                    all_attns = all_attns + (attn,)
            return curr_hidden_states

        hidden_states = _run_layer_span(0, recover_layer_idx, hidden_states, attention_mask, position_ids)

        if recover_state is not None:
            image_start_idx = int(recover_state["image_start_idx"].item())
            compact_image_end_idx = int(recover_state["compact_image_end_idx"].item())
            hidden_states = self._recover_quadtree_children_from_residual(
                hidden_states=hidden_states,
                image_start_idx=image_start_idx,
                image_end_idx=compact_image_end_idx,
                parent_hidden_at_merge=recover_state["parent_hidden_at_merge"],
                child_to_parent=recover_state["child_to_parent"],
                child_hidden_at_merge=recover_state["child_hidden_at_merge"],
                calibration_child_indices=recover_state["calibration_child_indices"],
            )
            scoped_input_ids = recover_state["recovered_input_ids"]
            attention_mask = recover_state["recovered_attention_mask"]
            position_ids = recover_state["recovered_position_ids"]
            scoped_keep_positions = recover_state["recovered_keep_positions"]
        else:
            scoped_keep_positions = torch.arange(scoped_input_ids.shape[1], device=scoped_input_ids.device).unsqueeze(0)

        hidden_states = _run_layer_span(recover_layer_idx, wipe_layer_idx, hidden_states, attention_mask, position_ids)

        if enable_finalwipe:
            hidden_states, final_input_ids, final_attention_mask, final_position_ids, wiped_keep_positions = self._wipe_visual_tokens(
                hidden_states=hidden_states,
                input_ids=scoped_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            final_keep_positions = scoped_keep_positions.index_select(1, wiped_keep_positions[0])
            final_wiped_visual_tokens = int(attention_mask.shape[1] - final_attention_mask.shape[1])
        else:
            final_input_ids = scoped_input_ids
            final_attention_mask = attention_mask
            final_position_ids = position_ids
            final_keep_positions = scoped_keep_positions
            final_wiped_visual_tokens = 0

        if getattr(self, "learnable_prune_stats", None):
            self.learnable_prune_stats[-1].update(
                {
                    "recover_layer_idx": int(recover_layer_idx),
                    "recovered_visual_tokens": int(scoped_input_ids[0].eq(IMAGE_TOKEN_INDEX).sum().item()),
                    "enable_finalwipe": bool(enable_finalwipe),
                    "final_wipe_layer_idx": int(wipe_layer_idx),
                    "final_wiped_visual_tokens": int(final_wiped_visual_tokens),
                    "final_sequence_tokens": int(final_attention_mask.shape[1]),
                }
            )
            if os.environ.get("LEARNABLE_PRUNE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
                print(f"[learnable_prune_scope_recover_finalwipe] stats={self.learnable_prune_stats[-1]}", flush=True)

        hidden_states = _run_layer_span(wipe_layer_idx, len(layers), hidden_states, final_attention_mask, final_position_ids)

        hidden_states = self.model.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if next_cache is not None:
            self._set_cache_seen_tokens(next_cache, hidden_states.size(1))
        self._scope_recover_finalwipe_applied = True
        self._scope_recover_finalwipe_next_position_id = int(input_ids.shape[1])
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
        learnable_prune: bool = True,
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
            self._scope_recover_finalwipe_applied = False
            self._scope_recover_finalwipe_next_position_id = None

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
                use_cache=bool(use_cache),
                output_attentions=bool(output_attentions),
                output_hidden_states=bool(output_hidden_states),
            )
        elif getattr(self, "_scope_recover_finalwipe_applied", False) and past_key_values is not None:
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
            next_position_id = getattr(self, "_scope_recover_finalwipe_next_position_id", None)
            if next_position_id is not None:
                position_ids = torch.arange(
                    int(next_position_id),
                    int(next_position_id) + current_seq_len,
                    device=inputs_embeds.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                self._scope_recover_finalwipe_next_position_id = int(next_position_id) + current_seq_len
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
            inputs_embeds, expanded_input_ids, attention_mask, position_ids = self._embed_multimodal_for_generation(
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


LlavaForConditionalGeneration = LlavaLearnablePruneScopeRecoverFinalwipeForCausalLM
