# coding=utf-8
"""Attention and cache helpers for the v35 hybrid LLaVA model."""

from typing import Any, Optional

import torch
from torch import nn

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

try:
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb as _llama_apply_rotary_pos_emb,
        eager_attention_forward as _llama_eager_attention_forward,
        repeat_kv as _llama_repeat_kv,
    )
except Exception:
    _llama_apply_rotary_pos_emb = None
    _llama_eager_attention_forward = None
    _llama_repeat_kv = None

try:
    from transformers.models.llama.modeling_llama import create_causal_mask as _llama_create_causal_mask
except Exception:
    _llama_create_causal_mask = None


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    if _llama_apply_rotary_pos_emb is not None:
        return _llama_apply_rotary_pos_emb(q, k, cos, sin)
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0)
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
    if sin.dim() == 3:
        sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if _llama_repeat_kv is not None:
        return _llama_repeat_kv(hidden_states, n_rep)
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)


def _resolve_attn_meta(attn: nn.Module) -> tuple[int, int, int, int, float]:
    config = getattr(attn, "config", None)
    num_heads = int(getattr(attn, "num_heads", getattr(config, "num_attention_heads")))
    num_kv_heads = int(getattr(attn, "num_key_value_heads", getattr(config, "num_key_value_heads")))
    if hasattr(attn, "head_dim"):
        head_dim = int(attn.head_dim)
    else:
        hidden_size = int(getattr(config, "hidden_size"))
        head_dim = hidden_size // max(1, num_heads)
    num_kv_groups = int(getattr(attn, "num_key_value_groups", num_heads // max(1, num_kv_heads)))
    scaling = float(getattr(attn, "scaling", head_dim ** -0.5))
    return num_heads, num_kv_heads, head_dim, num_kv_groups, scaling


def _get_attention_interface(attn: nn.Module):
    eager_impl = _llama_eager_attention_forward
    impl = getattr(getattr(attn, "config", None), "_attn_implementation", "eager")
    if impl != "eager":
        return ALL_ATTENTION_FUNCTIONS.get(impl, eager_impl)
    return eager_impl


def _create_llama_causal_mask(
    *,
    config,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.LongTensor,
    past_key_values: Optional[Any],
    position_ids: torch.LongTensor,
):
    try:
        return _llama_create_causal_mask(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
    except TypeError as exc:
        if "inputs_embeds" not in str(exc):
            raise
        return _llama_create_causal_mask(
            config=config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )


def _supports_sparse_decode(attn: nn.Module) -> bool:
    return all(hasattr(attn, name) for name in ["q_proj", "k_proj", "v_proj", "o_proj"])


def _append_dynamic_kv_cache(
    past_key_values: Optional[Any],
    layer_idx: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    if past_key_values is None or not hasattr(past_key_values, "layers"):
        return None
    layers = getattr(past_key_values, "layers", None)
    if layers is None or layer_idx < 0 or layer_idx >= len(layers):
        return None
    cache_layer = layers[layer_idx]
    if getattr(cache_layer, "is_sliding", False) or not all(hasattr(cache_layer, name) for name in ["keys", "values"]):
        return None
    if not hasattr(cache_layer, "is_initialized") or not hasattr(cache_layer, "lazy_initialization"):
        return None

    if not cache_layer.is_initialized:
        cache_layer.lazy_initialization(key_states)
    cache_layer.keys = torch.cat([cache_layer.keys, key_states], dim=-2)
    cache_layer.values = torch.cat([cache_layer.values, value_states], dim=-2)
    return cache_layer.keys, cache_layer.values


def _project_kv_with_cache(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    past_key_values: Optional[Any],
    cache_position: Optional[torch.LongTensor],
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Project K/V for all rows and update the cache without computing attention."""
    if position_embeddings is None or not _supports_sparse_decode(attn):
        return None

    bsz, q_len, _ = hidden_states.shape
    _, num_kv_heads, head_dim, _, _ = _resolve_attn_meta(attn)
    cos, sin = position_embeddings

    key_states = attn.k_proj(hidden_states)
    value_states = attn.v_proj(hidden_states)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    _, key_states = _apply_rotary_pos_emb(key_states, key_states, cos, sin)

    if past_key_values is not None and hasattr(attn, "layer_idx"):
        direct_update = _append_dynamic_kv_cache(past_key_values, int(attn.layer_idx), key_states, value_states)
        if direct_update is not None:
            return direct_update
    if past_key_values is not None and hasattr(past_key_values, "update") and hasattr(attn, "layer_idx"):
        cache_kwargs = {"cache_position": cache_position}
        try:
            cache_kwargs.update({"sin": sin, "cos": cos})
            key_states, value_states = past_key_values.update(key_states, value_states, attn.layer_idx, cache_kwargs)
        except TypeError:
            key_states, value_states = past_key_values.update(key_states, value_states, attn.layer_idx)

    return key_states, value_states




def _project_query_subset(
    attn: nn.Module,
    hidden_subset: torch.Tensor,
    cos_q: torch.Tensor,
    sin_q: torch.Tensor,
) -> torch.Tensor:
    bsz, q_len, _ = hidden_subset.shape
    num_heads, _, head_dim, _, _ = _resolve_attn_meta(attn)
    query_states = attn.q_proj(hidden_subset).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    query_states, _ = _apply_rotary_pos_emb(query_states, query_states, cos_q, sin_q)
    return query_states


def _slice_decode_attention_mask(
    attention_mask: Optional[torch.Tensor],
    active_idx: torch.Tensor,
    q_len: int,
    kv_len: int,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.dim() == 2:
        mask = attention_mask.index_select(0, active_idx)[:, None, None, :kv_len]
        return (1.0 - mask.to(dtype=dtype)) * torch.finfo(dtype).min
    if attention_mask.dim() == 4:
        return attention_mask.index_select(0, active_idx)[..., -q_len:, :kv_len]
    return None


def _run_attention_interface(
    attn: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    force_eager: bool = False,
):
    _, _, _, _, scaling = _resolve_attn_meta(attn)
    interface = _llama_eager_attention_forward if force_eager else _get_attention_interface(attn)
    if interface is None:
        raise RuntimeError("No usable attention interface found for FlexiExit sparse decode.")
    return interface(
        attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not attn.training else getattr(attn, "attention_dropout", 0.0),
        scaling=scaling,
    )


def _sparse_decode_attention(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Any],
    cache_position: Optional[torch.LongTensor],
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
    active_rows: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Decode-only sparse attention.

    K/V are projected and cached for every row so later tokens can still attend to
    this token. Q/attention/O projection are computed only for active rows.
    """
    if position_embeddings is None or not _supports_sparse_decode(attn):
        return None

    bsz, q_len, hidden_size = hidden_states.shape
    if q_len != 1:
        return None

    if active_rows.dtype != torch.bool:
        active_rows = active_rows.to(torch.bool)
    active_idx = active_rows.nonzero(as_tuple=False).flatten()

    kv_states = _project_kv_with_cache(attn, hidden_states, past_key_values, cache_position, position_embeddings)
    if kv_states is None:
        return None
    key_states, value_states = kv_states

    attn_out = hidden_states.new_zeros((bsz, q_len, hidden_size))
    if active_idx.numel() == 0:
        return attn_out

    cos, sin = position_embeddings
    hidden_subset = hidden_states.index_select(0, active_idx)
    if cos.dim() >= 3:
        if cos.shape[0] == 1:
            cos_q = cos.expand(active_idx.numel(), -1, -1)
            sin_q = sin.expand(active_idx.numel(), -1, -1)
        else:
            cos_q = cos.index_select(0, active_idx)
            sin_q = sin.index_select(0, active_idx)
    else:
        cos_q, sin_q = cos, sin

    query_states = _project_query_subset(attn, hidden_subset, cos_q, sin_q)
    key_active = key_states.index_select(0, active_idx)
    value_active = value_states.index_select(0, active_idx)
    mask_active = _slice_decode_attention_mask(
        attention_mask=attention_mask,
        active_idx=active_idx,
        q_len=q_len,
        kv_len=key_active.shape[-2],
        dtype=query_states.dtype,
    )

    attn_output, _ = _run_attention_interface(attn, query_states, key_active, value_active, mask_active)
    attn_output = attn_output.reshape(active_idx.numel(), q_len, hidden_size).contiguous()
    attn_output = attn.o_proj(attn_output)
    attn_out.index_copy_(0, active_idx, attn_output)
    return attn_out
