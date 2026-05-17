# coding=utf-8
"""FlexiExit modules and routing state for the v35 hybrid LLaVA model."""

from typing import Any, Optional

import math
import torch
from torch import nn

try:
    from ..llama.modeling_llama import LlamaRMSNorm as _FlexiDepthRMSNorm
except Exception:
    class _FlexiDepthRMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            input_dtype = x.dtype
            x_float = x.float()
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            x_norm = x_float * torch.rsqrt(variance + self.eps)
            return (self.weight.float() * x_norm).to(input_dtype)


class _FlexiExitRouter(nn.Module):
    """Bottleneck MLP that predicts one continue score per token."""

    def __init__(self, hidden_size: int, bottleneck_size: int):
        super().__init__()
        bottleneck_size = max(1, int(bottleneck_size))
        self.down = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.act = nn.Tanh()
        self.norm = _FlexiDepthRMSNorm(bottleneck_size)
        self.up = nn.Linear(bottleneck_size, hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, 1, bias=False)
        self._fused_up_out_weight: Optional[torch.Tensor] = None

    def _get_fused_up_out_weight(self) -> torch.Tensor:
        fused = self._fused_up_out_weight
        ref = self.up.weight
        if fused is None or fused.device != ref.device or fused.dtype != ref.dtype:
            fused = torch.matmul(self.out.weight.to(ref.dtype), ref)
            self._fused_up_out_weight = fused
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.act(x)
        x = self.norm(x)
        if not self.training and not torch.is_grad_enabled():
            return torch.matmul(x, self._get_fused_up_out_weight().transpose(0, 1))
        x = self.up(x)
        return self.out(x)


class _FlexiExitAdapter(nn.Module):
    """Small SwiGLU-style adapter used after a token exits."""

    def __init__(self, hidden_size: int, adapter_hidden_size: int):
        super().__init__()
        adapter_hidden_size = max(1, int(adapter_hidden_size))
        self.gate_proj = nn.Linear(hidden_size, adapter_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, adapter_hidden_size, bias=False)
        self.down_proj = nn.Linear(adapter_hidden_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self._fused_gate_up_weight: Optional[torch.Tensor] = None

    def _get_fused_gate_up_weight(self) -> torch.Tensor:
        fused = self._fused_gate_up_weight
        ref = self.gate_proj.weight
        if fused is None or fused.device != ref.device or fused.dtype != ref.dtype:
            fused = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0).to(dtype=ref.dtype)
            self._fused_gate_up_weight = fused
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and not torch.is_grad_enabled():
            gate, up = torch.matmul(x, self._get_fused_gate_up_weight().transpose(0, 1)).chunk(2, dim=-1)
            return self.down_proj(self.act_fn(gate) * up)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class flexiexit_tau(nn.Module):
    """Learnable per-layer exit threshold parameterized in logit space."""

    def __init__(self, init_tau: float):
        super().__init__()
        init_tau = min(max(float(init_tau), 1e-6), 1.0 - 1e-6)
        self.logit = nn.Parameter(torch.tensor(math.log(init_tau / (1.0 - init_tau)), dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)


class _FlexiExitState:
    """Per-forward-pass state shared by all patched decoder layers."""

    def __init__(
        self,
        *,
        tau: float,
        track_stats: bool = False,
        is_training: Optional[bool] = None,
    ):
        self.tau = float(tau)
        self.track_stats = bool(track_stats)
        self.is_training = is_training

        self.exit_mask: Optional[torch.Tensor] = None
        self.exit_layer: Optional[torch.Tensor] = None
        self.active_depth: Optional[torch.Tensor] = None
        self._all_gate_sum: Optional[torch.Tensor] = None
        self.current_token_mask: Optional[torch.Tensor] = None
        self.single_token_exited = False

    def ensure_shape(self, reference: torch.Tensor) -> None:
        shape = reference.shape[:2]
        device = reference.device
        if self.exit_mask is None or self.exit_mask.shape != shape:
            self.exit_mask = torch.zeros(shape, dtype=torch.bool, device=device)
            self.exit_layer = torch.full(shape, -1, dtype=torch.long, device=device)
            self.active_depth = torch.zeros(shape, dtype=torch.float32, device=device)
            self._all_gate_sum = None

    def alive_mask(self, reference: torch.Tensor) -> torch.Tensor:
        self.ensure_shape(reference)
        return ~self.exit_mask

    def record(
        self,
        *,
        layer_idx: int,
        gate: torch.Tensor,
        continue_mask: torch.Tensor,
        new_exit_mask: torch.Tensor,
        cost_gate: Optional[torch.Tensor] = None,
    ) -> None:
        self.ensure_shape(gate)

        if self.track_stats:
            if cost_gate is None:
                cost_gate = gate
            self._all_gate_sum = cost_gate if self._all_gate_sum is None else (self._all_gate_sum + cost_gate)

            if self.active_depth is not None:
                self.active_depth = self.active_depth + continue_mask.detach().to(torch.float32)

        if self.exit_mask is not None and self.exit_layer is not None:
            first_exit = new_exit_mask & (~self.exit_mask)
            self.exit_layer = torch.where(
                first_exit,
                torch.full_like(self.exit_layer, int(layer_idx)),
                self.exit_layer,
            )
            self.exit_mask = self.exit_mask | new_exit_mask

    def finalize(self, token_mask: Optional[torch.Tensor] = None, sum_type: str = "square"):
        depth_map = self.active_depth
        exit_mask = self.exit_mask
        exit_layer = self.exit_layer

        ref = self._all_gate_sum

        token_costs = None
        if self.track_stats and ref is not None:
            gate_device = ref.device
            gate_dtype = ref.dtype
            if token_mask is not None:
                cost_mask = token_mask.to(device=gate_device, dtype=gate_dtype)
            else:
                cost_mask = torch.ones_like(ref)

            def compute_cost(g_sum: Optional[torch.Tensor]):
                if g_sum is None:
                    return 0.0
                cost = g_sum if sum_type == "l1" else g_sum.square()
                return cost * cost_mask

            token_costs = {"all": compute_cost(self._all_gate_sum)}

        if token_mask is not None:
            mask_bool = token_mask.to(dtype=torch.bool, device=token_mask.device)
            if depth_map is not None and depth_map.shape == mask_bool.shape:
                depth_map = depth_map * mask_bool.to(depth_map.dtype)
            if exit_mask is not None and exit_mask.shape == mask_bool.shape:
                exit_mask = exit_mask & mask_bool
            if exit_layer is not None and exit_layer.shape == mask_bool.shape:
                exit_layer = torch.where(mask_bool, exit_layer, torch.full_like(exit_layer, -1))

        return depth_map, exit_mask, exit_layer, None, token_costs


def _is_llama_like_decoder(model: nn.Module) -> bool:
    layers = getattr(model, "layers", None)
    if layers is None or not isinstance(layers, (nn.ModuleList, list)) or len(layers) == 0:
        return False
    layer0 = layers[0]
    return all(hasattr(layer0, name) for name in ["self_attn", "mlp", "input_layernorm", "post_attention_layernorm"])


def _parse_flexiexit_layers(raw_layers: Optional[Any], total_layers: int) -> list[int]:
    """Parse config-friendly layer specs into sorted unique decoder-layer indices.

    Supported forms:
    - None: use the last half of the decoder layers.
    - [16, 17, 18]: explicit layer indices.
    - [[16, 31]] or [(16, 31)]: inclusive intervals.
    - [[16, 23], [28, 31]]: multiple inclusive intervals.
    """
    if raw_layers is None:
        start = total_layers // 2
        return list(range(start, total_layers))

    if isinstance(raw_layers, str):
        raw_layers = raw_layers.strip()
        if not raw_layers:
            return []
        return _parse_flexiexit_layers([int(x) for x in raw_layers.split(",")], total_layers)

    indices: set[int] = set()
    for item in raw_layers:
        if isinstance(item, int):
            indices.add(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = int(item[0]), int(item[1])
            if start > end:
                start, end = end, start
            indices.update(range(start, end + 1))
        else:
            raise ValueError(f"Unsupported flexiexit_layers item: {item!r}")

    return sorted(i for i in indices if 0 <= i < total_layers)




def _router_logit_threshold(tau: float) -> float:
    if tau <= 0.0:
        return float("-inf")
    if tau >= 1.0:
        return float("inf")
    return math.log(tau / (1.0 - tau))


def _resolve_flexiexit_tau(
    layer: nn.Module,
    default_tau: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    tau_module = getattr(layer, "flexiexit_tau", None)
    if isinstance(tau_module, nn.Module):
        tau_logit_param = getattr(tau_module, "logit", None)
        if tau_logit_param is None:
            raise AttributeError("FlexiExit tau module must expose a `logit` parameter.")
        tau_logit = tau_logit_param.to(device=device, dtype=dtype)
        tau_prob = torch.sigmoid(tau_logit)
        return tau_prob, tau_logit

    tau_value = min(max(float(tau_module if tau_module is not None else default_tau), 1e-6), 1.0 - 1e-6)
    tau_prob = torch.tensor(tau_value, device=device, dtype=dtype)
    tau_logit = torch.tensor(math.log(tau_value / (1.0 - tau_value)), device=device, dtype=dtype)
    return tau_prob, tau_logit
