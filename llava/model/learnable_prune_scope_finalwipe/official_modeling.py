#!/usr/bin/env python3
# coding: utf-8
"""Official LLaVA learnable-prune + SeededResidualSCOPE + final-wipe inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
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


DEFAULT_CHECKPOINT = "/data1/chenzixuan/train_output/official_llava_learnable_prune_precision_at_k_hinge_top64_layers16_24_iqr_top3"
LEARNABLE_TOPK = 64
SCOPE_TARGET_COUNT = 82
FINAL_WIPE_LAYER_IDX = 25


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


class LlavaLearnablePruneScopeFinalwipeModel(LlavaLlamaModel):
    config_class = LlavaConfig


class LlavaLearnablePruneScopeFinalwipeForCausalLM(LlavaLlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlavaLlamaForCausalLM, self).__init__(config)
        self.model = LlavaLearnablePruneScopeFinalwipeModel(config)
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
        predictor = LearnablePrunePredictor(
            input_size=int(config["predictor_input_size"]),
            hidden_size=int(config["predictor_hidden_size"]),
            num_heads=int(config["predictor_heads"]),
            mlp_ratio=int(config.get("predictor_mlp_ratio", 2)),
            num_layers=int(config.get("predictor_layers", 4)),
            use_final_full_attention=bool(config.get("predictor_use_final_full_attention", True)),
        )
        predictor.load_state_dict(torch.load(checkpoint / "predictor.pt", map_location="cpu"))
        device, dtype = self._device_dtype()
        predictor.to(device=device, dtype=dtype)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.shape[0] != 1:
            raise ValueError("learnable-prune scope-finalwipe generation currently expects batch_size=1")
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
            return inputs_embeds, input_ids[:, valid_positions], attention_mask, position_ids, valid_positions.unsqueeze(0)

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
        top_relative = torch.topk(scores, k=topk).indices if topk > 0 else scores.new_empty((0,), dtype=torch.long)

        target_keep = min(SCOPE_TARGET_COUNT, int(scores.numel()))
        if target_keep > topk:
            visual_embeds = inputs_embeds.index_select(1, visual_positions)
            seeded_scope_rank, _ = SeededResidualSCOPE(
                visual_feature_vectors=visual_embeds.to(dtype=torch.float32),
                seed_relative=top_relative,
                target_keep=target_keep,
                spatial_bonus=0.0,
            )
            visual_keep_relative = seeded_scope_rank[0]
        else:
            visual_keep_relative = top_relative[:target_keep].sort().values

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
                "learnable_topk_visual_tokens": int(topk),
                "kept_visual_tokens": int(visual_keep.numel()),
                "scope_target_visual_tokens": int(target_keep),
                "diversity_fill_method": "seeded_residual_scope",
                "pruned_tokens": int(input_ids.shape[1] - keep_positions.numel()),
                "checkpoint": getattr(self, "learnable_prune_checkpoint", None),
            }
        )
        return pruned_embeds, pruned_input_ids, pruned_attention_mask, pruned_position_ids, keep_positions.unsqueeze(0)

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
        inputs_embeds, scoped_input_ids, attention_mask, position_ids, scoped_keep_positions = self._learnable_scope_prune_prefill(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        layers = self.model.layers
        wipe_layer_idx = min(FINAL_WIPE_LAYER_IDX, len(layers))
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = DynamicCache(config=self.model.config) if use_cache else None

        pre_cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
        pre_mask = self._prepare_mask(
            attention_mask,
            hidden_states,
            position_ids=position_ids,
            cache_position=pre_cache_position,
        )
        for layer_idx in range(wipe_layer_idx):
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
            )
            if use_cache:
                if isinstance(next_cache, list):
                    next_cache.append(present)
            if output_attentions:
                all_attns = all_attns + (attn,)

        hidden_states, wiped_input_ids, wiped_attention_mask, wiped_position_ids, wiped_keep_positions = self._wipe_visual_tokens(
            hidden_states=hidden_states,
            input_ids=scoped_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        final_keep_positions = scoped_keep_positions.index_select(1, wiped_keep_positions[0])
        if getattr(self, "learnable_prune_stats", None):
            self.learnable_prune_stats[-1].update(
                {
                    "final_wipe_layer_idx": int(wipe_layer_idx),
                    "final_wiped_visual_tokens": int(attention_mask.shape[1] - wiped_attention_mask.shape[1]),
                    "final_sequence_tokens": int(wiped_attention_mask.shape[1]),
                }
            )

        post_cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
        post_mask = self._prepare_mask(
            wiped_attention_mask,
            hidden_states,
            position_ids=wiped_position_ids,
            cache_position=post_cache_position,
        )
        for layer_idx in range(wipe_layer_idx, len(layers)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states, attn, present = self._run_layer(
                layers[layer_idx],
                hidden_states,
                post_mask,
                wiped_position_ids,
                past_key_value=next_cache,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=post_cache_position,
            )
            if use_cache:
                if isinstance(next_cache, list):
                    next_cache.append(present)
            if output_attentions:
                all_attns = all_attns + (attn,)

        hidden_states = self.model.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        self._scope_finalwipe_applied = True
        self._scope_finalwipe_next_position_id = int(input_ids.shape[1])
        return (
            hidden_states,
            (tuple(next_cache) if isinstance(next_cache, list) else next_cache),
            all_hidden_states,
            all_attns,
            wiped_input_ids,
            wiped_attention_mask,
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
        learnable_prune_input_ids: Optional[torch.LongTensor] = None,
        learnable_prune: bool = True,
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

        if past_key_values is None:
            self._scope_finalwipe_applied = False
            self._scope_finalwipe_next_position_id = None

        should_prune = (
            learnable_prune
            and not self.training
            and learnable_prune_input_ids is not None
            and inputs_embeds.shape[0] == 1
            and inputs_embeds.shape[1] > 1
            and past_key_values is None
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
        elif getattr(self, "_scope_finalwipe_applied", False) and past_key_values is not None:
            current_seq_len = inputs_embeds.shape[1]
            cache_position = None
            if position_ids is not None and position_ids.shape[0] == 1:
                cache_position = position_ids.squeeze(0)
            elif attention_mask is not None:
                compact_position_ids = attention_mask.long().cumsum(dim=-1) - 1
                compact_position_ids = compact_position_ids.masked_fill(attention_mask.eq(0), 0)
                cache_position = compact_position_ids[:, -current_seq_len:].squeeze(0)
            next_position_id = getattr(self, "_scope_finalwipe_next_position_id", None)
            if next_position_id is not None:
                position_ids = torch.arange(
                    int(next_position_id),
                    int(next_position_id) + current_seq_len,
                    device=inputs_embeds.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                self._scope_finalwipe_next_position_id = int(next_position_id) + current_seq_len
            elif position_ids is None:
                position_ids = torch.arange(current_seq_len, device=inputs_embeds.device, dtype=torch.long).unsqueeze(0)
            hidden_states, past_key_values, all_hidden_states, all_attns = self._manual_decode(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
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
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            all_hidden_states = outputs.hidden_states
            all_attns = outputs.attentions

        logits = self.lm_head(hidden_states)
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
        if past_key_values is None and learnable_prune_input_ids is not None:
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


LlavaForConditionalGeneration = LlavaLearnablePruneScopeFinalwipeForCausalLM
