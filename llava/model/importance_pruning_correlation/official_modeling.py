#!/usr/bin/env python3
# coding: utf-8
"""Official LLaVA top-k visual-token importance pruning for layer sweeps."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaForCausalLM, apply_rotary_pos_emb

try:
    from transformers.models.llama.modeling_llama import create_causal_mask as llama_create_causal_mask
except ImportError:
    llama_create_causal_mask = None

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM, LlavaLlamaModel


WORKDIR = Path(os.environ.get("WORKDIR", "/data1/chenzixuan/MLLM_Token_Compression_Workdir"))
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

try:
    from src.importance_pruning_correlation.llava_top64_importance_model import (
        _aggregate_metric_buffer,
        _compute_distribution_metrics,
        _compute_kl_metrics,
    )
except Exception as exc:  # pragma: no cover - surfaced at runtime with a clearer message.
    _IMPORTANCE_METRIC_IMPORT_ERROR = exc
    _aggregate_metric_buffer = None
    _compute_distribution_metrics = None
    _compute_kl_metrics = None
else:
    _IMPORTANCE_METRIC_IMPORT_ERROR = None


def _as_bool(value: Union[bool, str, int, None]) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return bool(value)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_key_value_heads * n_rep, seq_len, head_dim)


class LlavaImportancePruningCorrelationModel(LlavaLlamaModel):
    config_class = LlavaConfig


class LlavaImportancePruningCorrelationForCausalLM(LlavaLlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlavaLlamaForCausalLM, self).__init__(config)
        self.model = LlavaImportancePruningCorrelationModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._importance_pruning_applied = False
        self._importance_next_position_id = None
        self.post_init()

    def get_model(self):
        return self.model

    def _device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        param = next(self.parameters())
        return param.device, param.dtype

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

        compact_input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
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
            if sum(split_sizes) > 0:
                text_embeds = embed_tokens(torch.cat(cur_input_ids_noim))
                text_embeds_split = torch.split(text_embeds, split_sizes, dim=0)
            else:
                text_embeds_split = [
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
                    cur_new_ids.append(
                        torch.full((visual_len,), IMAGE_TOKEN_INDEX, device=cur_input_ids.device, dtype=cur_input_ids.dtype)
                    )

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
                        torch.zeros(
                            (max_len - cur_len, cur_embeds.shape[1]),
                            dtype=cur_embeds.dtype,
                            device=cur_embeds.device,
                        ),
                    ],
                    dim=0,
                )
            )
            if cur_len > 0:
                padded_ids[i, :cur_len] = cur_ids
                padded_attention[i, :cur_len] = True
                padded_position_ids[i, :cur_len] = torch.arange(
                    0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                )

        return torch.stack(padded_embeds, dim=0), padded_ids, padded_attention, padded_position_ids

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
    ):
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=hidden_states.device)
        position_ids = position_ids.to(device=hidden_states.device)
        if cache_position is not None:
            cache_position = cache_position.to(device=hidden_states.device)
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
        next_cache = DynamicCache(config=self.model.config) if use_cache and past_key_values is None else (
            past_key_values if isinstance(past_key_values, Cache) else ([] if use_cache else None)
        )
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
            if isinstance(next_cache, Cache):
                layer_past = next_cache
            else:
                layer_past = past_key_values[layer_idx] if past_key_values is not None else None
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
            if use_cache and isinstance(next_cache, list):
                next_cache.append(present)
            if output_attentions:
                all_attns = all_attns + (attn,)

        hidden_states = self.model.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return hidden_states, (tuple(next_cache) if isinstance(next_cache, list) else next_cache), all_hidden_states, all_attns

    def _build_query_indices(
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
        if attn_anchor == "last":
            return valid_text_positions[-1:]
        raise ValueError(f"Unsupported attn_anchor={attn_anchor!r}. Expected 'query' or 'last'.")

    def _visual_token_scores(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        scoring_layer_idx: int,
        visual_positions: torch.Tensor,
        query_indices: torch.Tensor,
        prune_ref: str,
    ) -> torch.Tensor:
        scoring_layer = self.model.layers[scoring_layer_idx]
        self_attn = scoring_layer.self_attn
        bsz, seq_len, hidden_size = hidden_states.shape
        if bsz != 1:
            raise ValueError("Importance scoring expects batch_size=1.")
        if visual_positions.numel() == 0 or query_indices.numel() == 0:
            return hidden_states.new_zeros((visual_positions.numel(),))

        attention_mask = attention_mask.to(device=hidden_states.device)
        position_ids = position_ids.to(device=hidden_states.device)
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
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = _repeat_kv(key_states, num_heads // num_kv_heads)
        value_states = _repeat_kv(value_states, num_heads // num_kv_heads)

        q_idx = query_indices.to(dtype=torch.long, device=hidden_states.device)
        visual_positions = visual_positions.to(dtype=torch.long, device=hidden_states.device)
        key_positions = torch.arange(seq_len, device=hidden_states.device).view(1, 1, 1, seq_len)
        key_valid = attention_mask[:, None, None, :].bool()
        v_visual = value_states.index_select(2, visual_positions)[0]
        score_chunks: List[torch.Tensor] = []
        attn_chunks: List[torch.Tensor] = []
        for q_chunk in q_idx.split(8):
            q_states = query_states.index_select(2, q_chunk)
            attn_scores = torch.matmul(q_states, key_states.transpose(2, 3)) * scaling
            causal = key_positions <= q_chunk.view(1, 1, -1, 1)
            attn_scores = attn_scores.float().masked_fill(~(causal & key_valid), torch.finfo(torch.float32).min)
            attn_probs = torch.softmax(attn_scores, dim=-1).to(dtype=q_states.dtype)
            visual_attn = attn_probs.index_select(-1, visual_positions)[0].permute(1, 0, 2).contiguous()
            attn_chunks.append(visual_attn.mean(dim=1))
            if prune_ref == "attention":
                continue

            z_heads = torch.matmul(attn_probs, value_states)[0].permute(1, 0, 2).contiguous()
            beta = visual_attn / (1.0 - visual_attn).clamp(min=1e-6)
            delta_z = beta.unsqueeze(-1) * (z_heads.unsqueeze(2) - v_visual.unsqueeze(0))
            delta_z_cat = delta_z.permute(0, 2, 1, 3).contiguous().view(-1, num_heads * head_dim)
            delta_y = self_attn.o_proj(delta_z_cat).view(q_chunk.numel(), -1, hidden_size)
            score_chunks.append(delta_y.norm(dim=-1))

        if prune_ref == "attention":
            return torch.cat(attn_chunks, dim=0).mean(dim=0)
        if prune_ref != "importance":
            raise ValueError(f"Unsupported prune_ref={prune_ref!r}. Expected 'importance' or 'attention'.")
        return torch.cat(score_chunks, dim=0).mean(dim=0)

    def _forward_to_layer_input(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        scoring_layer_idx: int,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
        causal_mask = self._prepare_mask(
            attention_mask,
            hidden_states,
            position_ids=position_ids,
            cache_position=cache_position,
        )
        for layer in self.model.layers[:scoring_layer_idx]:
            hidden_states, _, _ = self._run_layer(
                layer,
                hidden_states,
                causal_mask,
                position_ids,
                use_cache=False,
                output_attentions=False,
                cache_position=cache_position,
            )
        return hidden_states

    def _topk_keep_positions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        scores: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        valid_positions = attention_mask[0].bool().nonzero(as_tuple=False).flatten()
        visual_positions = (input_ids[0].eq(IMAGE_TOKEN_INDEX) & attention_mask[0].bool()).nonzero(as_tuple=False).flatten()
        target_count = min(max(int(target_count), 1), int(visual_positions.numel()))
        if visual_positions.numel() <= target_count:
            return valid_positions
        top_relative = torch.topk(scores, k=target_count).indices.sort().values
        visual_keep = visual_positions.index_select(0, top_relative.to(visual_positions.device))
        keep_mask = input_ids[0].ne(IMAGE_TOKEN_INDEX) & attention_mask[0].bool()
        keep_mask[visual_keep] = True
        return keep_mask.nonzero(as_tuple=False).flatten()

    def _prefill_with_importance_pruning(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        scoring_layer_idx: int,
        visual_token_target_count: int,
        prune_ref: str,
        attn_anchor: str,
        limit_data_num: Optional[int],
        metrics_save_path: Optional[str],
        save_kl_div: bool,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
    ):
        if input_ids.shape[0] != 1:
            raise ValueError("importance-pruning correlation generation currently expects batch_size=1")
        if _IMPORTANCE_METRIC_IMPORT_ERROR is not None and metrics_save_path is not None:
            raise RuntimeError(
                f"Could not import importance metric utilities from {WORKDIR}: {_IMPORTANCE_METRIC_IMPORT_ERROR}"
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device=input_ids.device)
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(attention_mask.eq(0), 0)
        else:
            position_ids = position_ids.to(device=input_ids.device)

        visual_positions = (input_ids[0].eq(IMAGE_TOKEN_INDEX) & attention_mask[0].bool()).nonzero(as_tuple=False).flatten()
        if visual_positions.numel() == 0:
            return self._manual_decode(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            ) + (input_ids, attention_mask, None)

        scoring_layer_idx = int(scoring_layer_idx)
        if not 0 <= scoring_layer_idx < len(self.model.layers):
            raise ValueError(f"Invalid scoring_layer_idx={scoring_layer_idx}")
        query_indices = self._build_query_indices(input_ids, attention_mask, visual_positions, attn_anchor)
        layer_input = self._forward_to_layer_input(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            scoring_layer_idx=scoring_layer_idx,
        )
        scores = self._visual_token_scores(
            hidden_states=layer_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            scoring_layer_idx=scoring_layer_idx,
            visual_positions=visual_positions,
            query_indices=query_indices,
            prune_ref=prune_ref,
        )

        if metrics_save_path is not None:
            if not hasattr(self, "_importance_metric_buffer"):
                self._importance_metric_buffer = []
            sample_index = len(self._importance_metric_buffer)
            metric_record = _compute_distribution_metrics(scores, top_k=visual_token_target_count)
            metric_record["sample_index"] = int(sample_index)
            metric_record["query_count"] = int(query_indices.numel())
            metric_record["visual_token_count"] = int(visual_positions.numel())
            self._importance_metric_buffer.append(metric_record)
            current_len = len(self._importance_metric_buffer)
            if limit_data_num is None or current_len % 50 == 0 or current_len >= int(limit_data_num):
                _aggregate_metric_buffer(self._importance_metric_buffer, scoring_layer_idx, metrics_save_path)

        keep_positions = self._topk_keep_positions(input_ids, attention_mask, scores, visual_token_target_count)
        pruned_embeds = inputs_embeds.index_select(1, keep_positions)
        pruned_input_ids = input_ids.index_select(1, keep_positions)
        pruned_attention_mask = attention_mask.new_ones((1, keep_positions.numel()))
        pruned_position_ids = position_ids.index_select(1, keep_positions)
        pruned_cache_position = torch.arange(keep_positions.numel(), device=inputs_embeds.device, dtype=torch.long)

        hidden_states, past_key_values, all_hidden_states, all_attns = self._manual_decode(
            inputs_embeds=pruned_embeds,
            attention_mask=pruned_attention_mask,
            position_ids=pruned_position_ids,
            cache_position=pruned_cache_position,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        if past_key_values is not None:
            self._set_cache_seen_tokens(past_key_values, hidden_states.size(1))
        self._importance_pruning_applied = True
        self._importance_next_position_id = int(position_ids.max().item()) + 1 if position_ids.numel() > 0 else int(input_ids.shape[1])

        if save_kl_div and metrics_save_path is not None and getattr(self, "_importance_metric_buffer", None):
            with torch.no_grad():
                baseline_hidden, _, _, _ = self._manual_decode(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                baseline_hidden = baseline_hidden.index_select(1, keep_positions.to(baseline_hidden.device))
                self._importance_metric_buffer[-1].update(
                    _compute_kl_metrics(self.lm_head(baseline_hidden), self.lm_head(hidden_states))
                )
                current_len = len(self._importance_metric_buffer)
                if limit_data_num is None or current_len % 50 == 0 or current_len >= int(limit_data_num):
                    _aggregate_metric_buffer(self._importance_metric_buffer, scoring_layer_idx, metrics_save_path)

        return hidden_states, past_key_values, all_hidden_states, all_attns, pruned_input_ids, pruned_attention_mask, keep_positions.unsqueeze(0)

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
        importance_pruning_input_ids: Optional[torch.LongTensor] = None,
        scoring_layer_idx: Optional[int] = None,
        visual_token_target_count: int = 64,
        prune_ref: str = "importance",
        attn_anchor: str = "query",
        limit_data_num: Optional[int] = None,
        metrics_save_path: Optional[str] = None,
        save_kl_div: Union[bool, str, int] = False,
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
            self._importance_pruning_applied = False
            self._importance_next_position_id = None

        pruned_input_ids = None
        pruned_attention_mask = None
        pruned_keep_positions = None
        should_prune = (
            not self.training
            and scoring_layer_idx is not None
            and importance_pruning_input_ids is not None
            and inputs_embeds.shape[0] == 1
            and inputs_embeds.shape[1] > 1
            and is_initial_prefill_cache
        )
        if should_prune:
            (
                hidden_states,
                past_key_values,
                all_hidden_states,
                all_attns,
                pruned_input_ids,
                pruned_attention_mask,
                pruned_keep_positions,
            ) = self._prefill_with_importance_pruning(
                input_ids=importance_pruning_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                scoring_layer_idx=int(scoring_layer_idx),
                visual_token_target_count=int(visual_token_target_count),
                prune_ref=str(prune_ref).strip().lower(),
                attn_anchor=str(attn_anchor).strip().lower(),
                limit_data_num=limit_data_num,
                metrics_save_path=metrics_save_path,
                save_kl_div=_as_bool(save_kl_div),
                use_cache=bool(use_cache),
                output_attentions=bool(output_attentions),
                output_hidden_states=bool(output_hidden_states),
            )
        elif getattr(self, "_importance_pruning_applied", False) and past_key_values is not None:
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
                position_ids = torch.arange(current_seq_len, device=inputs_embeds.device, dtype=torch.long).unsqueeze(0)

            decode_cache_position = position_ids.squeeze(0)
            next_position_id = getattr(self, "_importance_next_position_id", None)
            if next_position_id is not None:
                position_ids = torch.arange(
                    int(next_position_id),
                    int(next_position_id) + current_seq_len,
                    device=inputs_embeds.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                self._importance_next_position_id = int(next_position_id) + current_seq_len
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
            causal_output.selected_scoring_layer_idx = int(scoring_layer_idx)
        return causal_output

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, **kwargs):
        pruned_attention_mask = getattr(outputs, "pruned_attention_mask", None)
        if (
            pruned_attention_mask is not None
            and model_kwargs.get("use_cache", True)
            and "cache_position" not in model_kwargs
        ):
            previous_attention_mask = model_kwargs.get("attention_mask", pruned_attention_mask)
            previous_seq_len = int(previous_attention_mask.shape[-1])
            model_kwargs["cache_position"] = pruned_attention_mask.new_tensor([max(previous_seq_len - 1, 0)], dtype=torch.long)
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, **kwargs)
        if pruned_attention_mask is not None:
            model_kwargs["_pruning_done"] = True
            model_kwargs["importance_pruning_input_ids"] = None
            model_kwargs["attention_mask"] = torch.cat(
                [pruned_attention_mask, pruned_attention_mask.new_ones((pruned_attention_mask.shape[0], 1))],
                dim=-1,
            )
            if model_kwargs.get("use_cache", True):
                pruned_seq_len = int(pruned_attention_mask.shape[1])
                cache_position = model_kwargs.get("cache_position", None)
                if cache_position is None:
                    model_kwargs["cache_position"] = pruned_attention_mask.new_tensor([pruned_seq_len], dtype=torch.long)
                else:
                    model_kwargs["cache_position"] = cache_position.new_tensor([pruned_seq_len])
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        importance_pruning_input_ids=None,
        scoring_layer_idx=None,
        visual_token_target_count=None,
        prune_ref=None,
        attn_anchor=None,
        limit_data_num=None,
        metrics_save_path=None,
        save_kl_div=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if self._is_initial_prefill_cache(past_key_values) and importance_pruning_input_ids is not None:
            model_inputs["importance_pruning_input_ids"] = importance_pruning_input_ids
            if scoring_layer_idx is not None:
                model_inputs["scoring_layer_idx"] = scoring_layer_idx
            if visual_token_target_count is not None:
                model_inputs["visual_token_target_count"] = visual_token_target_count
            if prune_ref is not None:
                model_inputs["prune_ref"] = prune_ref
            if attn_anchor is not None:
                model_inputs["attn_anchor"] = attn_anchor
            if limit_data_num is not None:
                model_inputs["limit_data_num"] = limit_data_num
            if metrics_save_path is not None:
                model_inputs["metrics_save_path"] = metrics_save_path
            if save_kl_div is not None:
                model_inputs["save_kl_div"] = save_kl_div
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
                importance_pruning_input_ids=expanded_input_ids,
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


LlavaForConditionalGeneration = LlavaImportancePruningCorrelationForCausalLM
