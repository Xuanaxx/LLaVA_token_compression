# coding=utf-8
"""Explicit FlexiExit layer-loop helpers for the v35 hybrid LLaVA model."""

from typing import Any, Optional

import torch
from torch import nn

from .modeling_llava_v35_flexiexit_hybrid_attention import (
    _project_kv_with_cache,
    _sparse_decode_attention,
)
from .modeling_llava_v35_flexiexit_hybrid_flexiexit import (
    _FlexiExitState,
    _resolve_flexiexit_tau,
    _router_logit_threshold,
)


def _selective_ffn(layer: nn.Module, hs_ffn_in: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """Compute MLP on active tokens and adapter on exited tokens."""
    hidden_size = hs_ffn_in.shape[-1]
    flat = hs_ffn_in.reshape(-1, hidden_size)
    active_flat = active_mask.reshape(-1).to(torch.bool)

    n_total = flat.shape[0]
    n_active = int(active_flat.sum().item())

    if n_active == n_total:
        return layer.mlp(hs_ffn_in)
    if n_active == 0:
        return layer.flexiexit_adapter(hs_ffn_in)

    active_idx = active_flat.nonzero(as_tuple=False).flatten()
    exited_idx = (~active_flat).nonzero(as_tuple=False).flatten()

    out = torch.zeros_like(flat)
    out.index_copy_(0, active_idx, layer.mlp(flat.index_select(0, active_idx)).to(out.dtype))
    out.index_copy_(0, exited_idx, layer.flexiexit_adapter(flat.index_select(0, exited_idx)).to(out.dtype))
    return out.view_as(hs_ffn_in)




def _route_flexiexit_gate(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    route_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_logit = hidden_states.new_zeros(hidden_states.shape[:2])
    gate = hidden_states.new_zeros(hidden_states.shape[:2])
    route_count = int(route_mask.sum().item())
    if route_count == route_mask.numel():
        gate_logit = layer.flexiexit_router(hidden_states).squeeze(-1)
        gate = torch.sigmoid(gate_logit)
    elif route_count > 0:
        hidden_size = hidden_states.shape[-1]
        flat_states = hidden_states.reshape(-1, hidden_size)
        route_idx = route_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        routed = flat_states.index_select(0, route_idx).view(1, route_count, hidden_size)
        routed_gate_logit = layer.flexiexit_router(routed).view(-1)
        routed_gate = torch.sigmoid(routed_gate_logit)
        gate_logit_flat = gate_logit.reshape(-1)
        gate_logit_flat.index_copy_(0, route_idx, routed_gate_logit.to(gate_logit_flat.dtype))
        gate_flat = gate.reshape(-1)
        gate_flat.index_copy_(0, route_idx, routed_gate.to(gate_flat.dtype))
        gate_logit = gate_logit_flat.view_as(gate_logit)
        gate = gate_flat.view_as(gate)
    return gate_logit, gate


def _flexiexit_train_layer_forward(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    state: _FlexiExitState,
    *,
    layer_idx: int,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor],
    past_key_values: Optional[Any],
    use_cache: bool,
    cache_position: Optional[torch.LongTensor],
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    state.ensure_shape(hidden_states)
    alive = state.alive_mask(hidden_states)
    eligible = state.current_token_mask
    if eligible is None:
        eligible = torch.ones_like(alive)
    else:
        eligible = eligible.to(device=hidden_states.device, dtype=torch.bool)
        if eligible.shape != alive.shape:
            raise ValueError(
                f"FlexiExit token mask shape mismatch: got {tuple(eligible.shape)}, expected {tuple(alive.shape)}."
            )
    eligible_alive = alive & eligible

    tau, tau_logit = _resolve_flexiexit_tau(
        layer,
        state.tau,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    gate_logit, gate = _route_flexiexit_gate(layer, hidden_states, eligible_alive)
    continue_mask = eligible_alive & (gate > tau)
    new_exit_mask = eligible_alive & (~continue_mask)
    model_path_mask = (~eligible) | continue_mask

    if isinstance(getattr(layer, "flexiexit_tau", None), nn.Module):
        continue_prob = torch.sigmoid(gate_logit - tau_logit)
    else:
        continue_prob = gate
    model_path_prob = torch.where(
        eligible_alive,
        continue_prob,
        torch.where(eligible, torch.zeros_like(gate), torch.ones_like(gate)),
    )
    continue_st = (model_path_mask.to(gate.dtype) - model_path_prob).detach() + model_path_prob
    cost_gate = gate * eligible_alive.to(gate.dtype)

    state.record(
        layer_idx=layer_idx,
        gate=gate,
        continue_mask=continue_mask,
        new_exit_mask=new_exit_mask,
        cost_gate=cost_gate,
    )

    residual = hidden_states
    hs_attn_in = layer.input_layernorm(hidden_states)
    attn_outputs = layer.self_attn(
        hidden_states=hs_attn_in,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        output_attentions=False,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + attn_outputs[0] * continue_st.unsqueeze(-1)

    residual = hidden_states
    hs_ffn_in = layer.post_attention_layernorm(hidden_states)
    mlp_out = layer.mlp(hs_ffn_in)
    adapter_out = layer.flexiexit_adapter(hs_ffn_in)
    p = model_path_prob.to(dtype=mlp_out.dtype)
    ffn_out = continue_st.unsqueeze(-1) * (p.unsqueeze(-1) * mlp_out) + (
        1.0 - continue_st
    ).unsqueeze(-1) * ((1.0 - p).unsqueeze(-1) * adapter_out)
    return residual + ffn_out


def _flexiexit_decode_layer_forward(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    exited: bool,
    *,
    tau: float,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor],
    past_key_values: Optional[Any],
    use_cache: bool,
    cache_position: Optional[torch.LongTensor],
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, bool, bool]:
    """FlexiExit decode path used by the bs=1 language-model layer loop.

    Returns (hidden_states, exited, token_continue). ``token_continue`` is False
    both for a newly exited token and for layers after an earlier exit.
    """
    if exited:
        token_continue = False
        exit_gain = None
    else:
        tau_module = getattr(layer, "flexiexit_tau", None)
        if isinstance(tau_module, nn.Module):
            tau_logit_param = getattr(tau_module, "logit", None)
            if tau_logit_param is None:
                raise AttributeError("FlexiExit tau module must expose a `logit` parameter.")
            tau_logit = tau_logit_param.to(device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            tau_value = float(tau_module if tau_module is not None else tau)
            tau_logit = _router_logit_threshold(tau_value)
        gate_logit = layer.flexiexit_router(hidden_states)
        gate = torch.sigmoid(gate_logit)
        token_continue = bool((gate_logit[0, 0, 0] > tau_logit).item())
        exited = not token_continue
        if token_continue:
            continue_gain = gate
            exit_gain = None
        else:
            exit_gain = 1.0 - gate

    residual = hidden_states
    hs_attn_in = layer.input_layernorm(hidden_states)

    if token_continue:
        attn_outputs = layer.self_attn(
            hidden_states=hs_attn_in,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + attn_outputs[0]
        residual = hidden_states
        hs_ffn_in = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(hs_ffn_in)
        return residual + continue_gain.to(device=mlp_out.device, dtype=mlp_out.dtype) * mlp_out, exited, bool(token_continue)

    if _project_kv_with_cache(layer.self_attn, hs_attn_in, past_key_values, cache_position, position_embeddings) is None:
        attn_outputs = layer.self_attn(
            hidden_states=hs_attn_in,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + attn_outputs[0] * 0.0

    residual = hidden_states
    hs_ffn_in = layer.post_attention_layernorm(hidden_states)
    adapter_out = layer.flexiexit_adapter(hs_ffn_in)
    if exit_gain is None:
        return residual + adapter_out, exited, token_continue
    return residual + exit_gain.to(device=adapter_out.device, dtype=adapter_out.dtype) * adapter_out, exited, token_continue


def _flexiexit_decode_exited_layers_forward(
    layers: list[nn.Module],
    hidden_states: torch.Tensor,
    *,
    start_idx: int = 0,
    position_ids: Optional[torch.LongTensor],
    past_key_values: Optional[Any],
    cache_position: Optional[torch.LongTensor],
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Run the already-exited decode suffix without router/attention dispatch."""
    for layer_idx in range(int(start_idx), len(layers)):
        layer = layers[layer_idx]
        residual = hidden_states
        hs_attn_in = layer.input_layernorm(hidden_states)
        if _project_kv_with_cache(layer.self_attn, hs_attn_in, past_key_values, cache_position, position_embeddings) is None:
            attn_outputs = layer.self_attn(
                hidden_states=hs_attn_in,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = residual + attn_outputs[0] * 0.0
        else:
            hidden_states = residual

        residual = hidden_states
        hs_ffn_in = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.flexiexit_adapter(hs_ffn_in)
    return hidden_states


def _flexiexit_prefill_layer_forward(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    exit_mask: torch.Tensor,
    *,
    tau: float,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.LongTensor],
    past_key_values: Optional[Any],
    use_cache: bool,
    cache_position: Optional[torch.LongTensor],
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Non-training prefill FlexiExit path used by the language-model layer loop."""
    alive = ~exit_mask
    route_count = int(alive.sum().item())
    tau_prob, _ = _resolve_flexiexit_tau(
        layer,
        tau,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    gate_logit = hidden_states.new_zeros(hidden_states.shape[:2])
    gate = hidden_states.new_zeros(hidden_states.shape[:2])
    if route_count == alive.numel():
        gate_logit = layer.flexiexit_router(hidden_states).squeeze(-1)
        gate = torch.sigmoid(gate_logit)
    elif route_count > 0:
        hidden_size = hidden_states.shape[-1]
        flat_states = hidden_states.reshape(-1, hidden_size)
        route_idx = alive.reshape(-1).nonzero(as_tuple=False).flatten()
        routed = flat_states.index_select(0, route_idx).view(1, route_count, hidden_size)
        routed_gate_logit = layer.flexiexit_router(routed).view(-1)
        routed_gate = torch.sigmoid(routed_gate_logit)
        gate_logit_flat = gate_logit.reshape(-1)
        gate_logit_flat.index_copy_(0, route_idx, routed_gate_logit.to(gate_logit_flat.dtype))
        gate_flat = gate.reshape(-1)
        gate_flat.index_copy_(0, route_idx, routed_gate.to(gate_flat.dtype))
        gate_logit = gate_logit_flat.view_as(gate_logit)
        gate = gate_flat.view_as(gate)

    continue_mask = alive & (gate > tau_prob)
    new_exit_mask = alive & (~continue_mask)

    residual = hidden_states
    hs_attn_in = layer.input_layernorm(hidden_states)
    need_cache = bool(use_cache or past_key_values is not None)
    all_continue = bool(continue_mask.all().item())
    none_continue = bool((~continue_mask).all().item())

    if all_continue:
        attn_outputs = layer.self_attn(
            hidden_states=hs_attn_in,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attn_out = attn_outputs[0]
    elif none_continue and not need_cache:
        attn_out = torch.zeros_like(hs_attn_in)
    else:
        attn_outputs = layer.self_attn(
            hidden_states=hs_attn_in,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attn_out = attn_outputs[0] * continue_mask.to(attn_outputs[0].dtype).unsqueeze(-1)

    hidden_states = residual + attn_out
    residual = hidden_states
    hs_ffn_in = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + _selective_ffn(layer, hs_ffn_in, continue_mask)
    return hidden_states, exit_mask | new_exit_mask, continue_mask, gate
