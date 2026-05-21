"""Prompt-conditioned predictor for one-stage Dynamic-KL visual-token pruning.

The predictor is intentionally kept independent from the frozen LVLM.  It consumes
only inference-time-visible states: visual token embeddings and unlabeled prompt
text embeddings.  It returns one scalar logit per visual token; inference can
hard-keep TopK visual tokens from these logits.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


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
    """Bidirectional linear self-attention over visual tokens.

    This is used before each text cross-attention layer to let visual tokens build
    local/global visual context without paying full O(Nv^2) cost in every layer.
    """

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
    """A final full-attention mixer over visual tokens.

    We keep this as a shallow final refinement layer only, so the predictor remains
    cheaper than passing all visual tokens through multiple LVLM layers.
    """

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
            # nn.MultiheadAttention produces NaNs when every key in a row is masked.
            # Give fully-empty rows one dummy key, then zero them again afterwards.
            safe_token_mask = token_mask.bool().clone()
            empty_rows = ~safe_token_mask.any(dim=1)
            if safe_token_mask.shape[1] > 0:
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
    """Prompt-conditioned visual-token scorer.

    Args:
        input_size: LVLM hidden size of the input embeddings.
        hidden_size: Predictor width.
        num_heads: Predictor attention heads.
        mlp_ratio: SwiGLU expansion ratio.
        num_layers: Number of alternating visual-self/prompt-cross blocks.
        use_final_full_attention: Whether to add one final O(Nv^2) visual mixer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_heads: int = 8,
        mlp_ratio: int = 2,
        num_layers: int = 4,
        use_final_full_attention: bool = False,
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
        if max_tokens > 0:
            batch_idx, seq_idx = token_mask.nonzero(as_tuple=True)
            slot_idx = token_mask.long().cumsum(dim=1)[batch_idx, seq_idx] - 1
            token_states[batch_idx, slot_idx] = hidden_states[batch_idx, seq_idx]
            valid_mask[batch_idx, slot_idx] = True
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
            if text_valid.shape[1] > 0:
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
