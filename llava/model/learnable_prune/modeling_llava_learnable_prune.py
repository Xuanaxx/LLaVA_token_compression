#!/usr/bin/env python3
# coding: utf-8
"""LLaVA inference wrapper for the learnable-prune predictor checkpoint."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers import LlavaForConditionalGeneration as HFLlavaForConditionalGeneration
from transformers.models.llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaModel as HFLlavaModel,
    LlavaModelOutputWithPast,
    LlavaPreTrainedModel,
)


DEFAULT_CHECKPOINT = "/data2/chenzixuan/train_output/llava_learnable_prune_budgeted_CE_KL_topk_hardening_top64_iqr_top3/checkpoint-1000"


def _first_visual_span(input_ids: torch.Tensor, image_token_id: int) -> Optional[Tuple[int, int]]:
    positions = (input_ids == image_token_id).nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return None
    start = int(positions[0].item())
    end = start
    while end < input_ids.numel() and int(input_ids[end].item()) == image_token_id:
        end += 1
    return start, end


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: int = 2):
        super().__init__()
        inner = hidden_size * mlp_ratio
        self.gate_proj = nn.Linear(hidden_size, inner, bias=False)
        self.up_proj = nn.Linear(hidden_size, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LinearAttentionBlock(nn.Module):
    """Bidirectional linear self-attention over visual tokens."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_norm = RMSNorm(hidden_size)
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        h = self.attn_norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        bsz, seq_len, _ = q.shape
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        if token_mask is not None:
            valid = token_mask[:, None, :, None].to(dtype=k.dtype)
            k = k * valid
            v = v * valid
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        k_sum = k.sum(dim=2)
        denom = torch.einsum("bhnd,bhd->bhn", q, k_sum).clamp(min=1e-6)
        attn = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, denom.reciprocal())
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        x = residual + self.o_proj(attn)
        x = x + self.mlp(self.mlp_norm(x))
        if token_mask is not None:
            x = x * token_mask.to(dtype=x.dtype).unsqueeze(-1)
        return x


class CrossAttentionBlock(nn.Module):
    """Non-causal cross-attention from visual-token queries to prompt text."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.query_norm = RMSNorm(hidden_size)
        self.context_norm = RMSNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, mlp_ratio=mlp_ratio)

    def forward(
        self,
        visual_states: torch.Tensor,
        context_states: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        visual_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.query_norm(visual_states)
        context = self.context_norm(context_states)
        key_padding_mask = None
        if context_mask is not None:
            key_padding_mask = ~context_mask.bool()
        attn_out, _ = self.attn(query, context, context, key_padding_mask=key_padding_mask, need_weights=False)
        visual_states = visual_states + torch.sigmoid(self.gate(query)) * attn_out
        visual_states = visual_states + self.mlp(self.mlp_norm(visual_states))
        if visual_mask is not None:
            visual_states = visual_states * visual_mask.to(dtype=visual_states.dtype).unsqueeze(-1)
        return visual_states


class GatedFullAttentionBlock(nn.Module):
    """A final full-attention mixer over visual tokens."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.attn_norm(x)
        key_padding_mask = None
        safe_token_mask = token_mask
        if token_mask is not None:
            safe_token_mask = token_mask.bool().clone()
            empty_rows = ~safe_token_mask.any(dim=1)
            if bool(empty_rows.any().item()) and safe_token_mask.shape[1] > 0:
                safe_token_mask[empty_rows, 0] = True
            key_padding_mask = ~safe_token_mask
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + torch.sigmoid(self.gate(h)) * attn_out
        x = x + self.mlp(self.mlp_norm(x))
        if token_mask is not None:
            x = x * token_mask.to(dtype=x.dtype).unsqueeze(-1)
        return x


class PromptConditionedVisualBlock(nn.Module):
    """Visual self-modeling followed by prompt cross-attention."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.visual_self = LinearAttentionBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
        self.prompt_cross = CrossAttentionBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)

    def forward(
        self,
        visual_states: torch.Tensor,
        visual_mask: torch.Tensor,
        text_states: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        visual_states = self.visual_self(visual_states, visual_mask)
        visual_states = self.prompt_cross(visual_states, text_states, text_mask, visual_mask)
        return visual_states


class LearnablePrunePredictor(nn.Module):
    """Prompt-conditioned visual-token scorer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_heads: int = 8,
        mlp_ratio: int = 2,
        num_layers: int = 4,
        use_final_full_attention: bool = True,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.mlp_ratio = int(mlp_ratio)
        self.num_layers = int(num_layers)
        self.use_final_full_attention = bool(use_final_full_attention)
        self.in_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.visual_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.layers = nn.ModuleList(
            [PromptConditionedVisualBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(self.num_layers)]
        )
        self.global_text_gate = nn.Linear(hidden_size, hidden_size)
        self.final_visual_mixer = (
            GatedFullAttentionBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            if self.use_final_full_attention
            else None
        )
        self.norm = RMSNorm(hidden_size)
        self.score = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.visual_type_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.text_type_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.score.weight, mean=0.0, std=1.0 / math.sqrt(self.score.in_features))
        nn.init.zeros_(self.score.bias)

    def _gather_token_states(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, hidden_size = hidden_states.shape
        token_mask = token_mask.bool()
        token_lengths = token_mask.long().sum(dim=1)
        max_tokens = int(token_lengths.max().item()) if token_lengths.numel() > 0 else 0
        token_states = hidden_states.new_zeros((bsz, max_tokens, hidden_size))
        valid_mask = torch.zeros((bsz, max_tokens), device=hidden_states.device, dtype=torch.bool)
        for b in range(bsz):
            length = int(token_lengths[b].item())
            if length > 0:
                token_states[b, :length] = hidden_states[b, token_mask[b]]
                valid_mask[b, :length] = True
        return token_states, valid_mask

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        multimodal_hidden_states: torch.Tensor,
        visual_token_mask: torch.Tensor | None = None,
        text_token_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = multimodal_hidden_states.new_ones(multimodal_hidden_states.shape[:2], dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if visual_token_mask is None:
            visual_token_mask = attention_mask
        else:
            visual_token_mask = visual_token_mask.bool() & attention_mask
        if text_token_mask is None:
            text_token_mask = attention_mask & ~visual_token_mask
        else:
            text_token_mask = text_token_mask.bool() & attention_mask & ~visual_token_mask

        context = self.in_proj(multimodal_hidden_states)
        visual, visual_valid = self._gather_token_states(context, visual_token_mask)
        if visual.shape[1] == 0:
            return context.new_zeros((context.shape[0], 0))
        text, text_valid = self._gather_token_states(context, text_token_mask)
        if text.shape[1] == 0:
            text = context.new_zeros((context.shape[0], 1, context.shape[-1]))
            text_valid = torch.ones((context.shape[0], 1), device=context.device, dtype=torch.bool)
        else:
            empty_text = ~text_valid.any(dim=1)
            if bool(empty_text.any().item()):
                text_valid[empty_text, 0] = True

        visual = visual + self.visual_type_embed.to(dtype=visual.dtype)
        text = text + self.text_type_embed.to(dtype=text.dtype)

        for layer in self.layers:
            visual = layer(visual, visual_valid, text, text_valid)

        global_text = self._masked_mean(text, text_valid).unsqueeze(1)
        visual = visual + torch.sigmoid(self.global_text_gate(visual)) * global_text
        if self.final_visual_mixer is not None:
            visual = self.final_visual_mixer(visual, visual_valid)

        logits = self.score(self.norm(visual)).squeeze(-1)
        return logits.masked_fill(~visual_valid, 0.0)

    def score_visual_text_by_positions(
        self,
        multimodal_hidden_states: torch.Tensor,
        visual_positions: torch.Tensor,
        text_positions: torch.Tensor,
    ) -> torch.Tensor:
        context = self.in_proj(multimodal_hidden_states)
        visual = context.index_select(1, visual_positions)
        text = context.index_select(1, text_positions)
        visual_valid = torch.ones(visual.shape[:2], device=visual.device, dtype=torch.bool)
        if text.shape[1] == 0:
            text = context.new_zeros((context.shape[0], 1, context.shape[-1]))
        text_valid = torch.ones(text.shape[:2], device=text.device, dtype=torch.bool)

        visual = visual + self.visual_type_embed.to(dtype=visual.dtype)
        text = text + self.text_type_embed.to(dtype=text.dtype)

        for layer in self.layers:
            visual = layer(visual, visual_valid, text, text_valid)

        global_text = self._masked_mean(text, text_valid).unsqueeze(1)
        visual = visual + torch.sigmoid(self.global_text_gate(visual)) * global_text
        if self.final_visual_mixer is not None:
            visual = self.final_visual_mixer(visual, visual_valid)

        logits = self.score(self.norm(visual)).squeeze(-1)
        return logits.masked_fill(~visual_valid, 0.0)


class LlavaModel(HFLlavaModel):
    """HF LLaVA model with learnable visual-token pruning during prefill."""

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
        predictor.to(device=self.device, dtype=self.dtype)
        predictor.eval()
        for param in predictor.parameters():
            param.requires_grad = False
        self.learnable_prune_predictor = predictor
        self.learnable_prune_config = dict(config)
        self.learnable_prune_checkpoint = str(checkpoint)
        self.learnable_prune_stats: List[Dict[str, Any]] = []

    def reset_learnable_prune_stats(self) -> None:
        self.learnable_prune_stats = []

    def get_learnable_prune_stats(self) -> List[Dict[str, Any]]:
        return list(getattr(self, "learnable_prune_stats", []))

    @torch.no_grad()
    def _learnable_prune_prefill(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        visual_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.shape[0] != 1:
            raise ValueError("learnable-prune generation currently expects batch_size=1")
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
        if visual_positions is None:
            visual_token_mask = input_ids.eq(int(self.config.image_token_id)) & valid_mask
            visual_positions = visual_token_mask[0].nonzero(as_tuple=False).flatten()
        else:
            visual_positions = visual_positions.to(device=input_ids.device, dtype=torch.long)
            visual_token_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            visual_token_mask[:, visual_positions] = True
            visual_token_mask &= valid_mask
            visual_positions = visual_token_mask[0].nonzero(as_tuple=False).flatten()

        if visual_positions.numel() == 0:
            valid_positions = attention_mask[0].bool().nonzero(as_tuple=False).flatten()
            return inputs_embeds, input_ids[:, valid_positions], attention_mask, position_ids, valid_positions.unsqueeze(0)

        config = self.learnable_prune_config
        predictor = self.learnable_prune_predictor
        predictor_dtype = next(predictor.parameters()).dtype
        text_token_mask = valid_mask & ~visual_token_mask
        text_positions = text_token_mask[0].nonzero(as_tuple=False).flatten()
        scores = predictor.score_visual_text_by_positions(
            inputs_embeds.to(dtype=predictor_dtype),
            visual_positions=visual_positions,
            text_positions=text_positions,
        )[0]
        keep_k = min(int(config.get("keep_k", 64)), int(scores.numel()))
        visual_keep = visual_positions[torch.topk(scores, k=keep_k).indices.sort().values]
        keep_mask = valid_mask[0].clone()
        keep_mask[visual_positions] = False
        keep_mask[visual_keep] = True
        keep_positions = keep_mask.nonzero(as_tuple=False).flatten()
        original_tokens = int(input_ids.shape[1])
        pruned_embeds = inputs_embeds[:, keep_positions, :]
        pruned_input_ids = input_ids[:, keep_positions]
        pruned_attention_mask = attention_mask.new_ones((1, keep_positions.numel()))
        pruned_position_ids = position_ids[:, keep_positions]
        self.learnable_prune_stats.append(
            {
                "original_tokens": original_tokens,
                "original_visual_tokens": int(visual_positions.numel()),
                "kept_visual_tokens": int(keep_k),
                "pruned_tokens": int(original_tokens - keep_positions.numel()),
                "checkpoint": getattr(self, "learnable_prune_checkpoint", None),
            }
        )
        return pruned_embeds, pruned_input_ids, pruned_attention_mask, pruned_position_ids, keep_positions.unsqueeze(0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        learnable_prune: bool = True,
        **kwargs,
    ) -> Union[tuple, LlavaModelOutputWithPast]:
        vision_feature_layer = vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        is_initial_prefill_cache = past_key_values is None or (
            hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0
        )
        if is_initial_prefill_cache:
            self._learnable_prune_applied = False
            self._learnable_prune_next_position_id = None

        prefill_prune_candidate = (
            learnable_prune
            and not self.training
            and pixel_values is not None
            and input_ids is not None
            and inputs_embeds.shape[1] > 1
            and is_initial_prefill_cache
        )
        visual_positions = (
            input_ids[0].eq(int(self.config.image_token_id)).nonzero(as_tuple=False).flatten()
            if prefill_prune_candidate
            else None
        )

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )
            image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            if visual_positions is not None and image_features.shape[0] == visual_positions.numel():
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[:, visual_positions, :] = image_features.unsqueeze(0)
            else:
                special_image_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_features
                )
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        should_prune = prefill_prune_candidate

        pruned_attention_mask = None
        pruned_input_ids = None
        pruned_keep_positions = None
        if should_prune:
            (
                inputs_embeds,
                pruned_input_ids,
                attention_mask,
                position_ids,
                pruned_keep_positions,
            ) = self._learnable_prune_prefill(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                visual_positions=visual_positions,
            )
            pruned_attention_mask = attention_mask
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long)
            self._learnable_prune_applied = True
            self._learnable_prune_next_position_id = int(input_ids.shape[1])
        elif (
            getattr(self, "_learnable_prune_applied", False)
            and past_key_values is not None
            and pixel_values is None
            and inputs_embeds is not None
        ):
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

            if position_ids is not None:
                cache_position = position_ids.squeeze(0)
                next_position_id = getattr(self, "_learnable_prune_next_position_id", None)
                if next_position_id is not None:
                    position_ids = torch.arange(
                        int(next_position_id),
                        int(next_position_id) + current_seq_len,
                        device=inputs_embeds.device,
                        dtype=torch.long,
                    ).unsqueeze(0)
                    self._learnable_prune_next_position_id = int(next_position_id) + current_seq_len

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        model_output = LlavaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )
        if pruned_attention_mask is not None:
            model_output.pruned_input_ids = pruned_input_ids
            model_output.pruned_attention_mask = pruned_attention_mask
            model_output.pruned_keep_positions = pruned_keep_positions
        return model_output


class LlavaForConditionalGeneration(HFLlavaForConditionalGeneration):
    """HF LLaVA with learnable visual-token pruning during generation prefill."""

    def __init__(self, config):
        LlavaPreTrainedModel.__init__(self, config)
        self.model = LlavaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def load_learnable_prune_checkpoint(self, checkpoint_path: str = DEFAULT_CHECKPOINT) -> None:
        self.model.load_learnable_prune_checkpoint(checkpoint_path)

    def reset_learnable_prune_stats(self) -> None:
        self.model.reset_learnable_prune_stats()

    def get_learnable_prune_stats(self) -> List[Dict[str, Any]]:
        return self.model.get_learnable_prune_stats()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: Optional[torch.Tensor] = None,
        learnable_prune: bool = True,
        **kwargs,
    ) -> Union[tuple, LlavaCausalLMOutputWithPast]:
        vision_feature_layer = vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            cache_position=cache_position,
            image_sizes=image_sizes,
            learnable_prune=bool(learnable_prune and labels is None),
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        causal_output = LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )
        pruned_attention_mask = getattr(outputs, "pruned_attention_mask", None)
        if pruned_attention_mask is not None:
            causal_output.pruned_input_ids = getattr(outputs, "pruned_input_ids", None)
            causal_output.pruned_attention_mask = pruned_attention_mask
            causal_output.pruned_keep_positions = getattr(outputs, "pruned_keep_positions", None)
        return causal_output

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, **kwargs)
        pruned_input_ids = getattr(outputs, "pruned_input_ids", None)
        pruned_attention_mask = getattr(outputs, "pruned_attention_mask", None)
        if pruned_input_ids is not None or pruned_attention_mask is not None:
            model_kwargs["_pruning_done"] = True
            if pruned_attention_mask is not None:
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
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs
