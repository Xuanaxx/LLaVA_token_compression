# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hybrid LLaVA model: v35 visual-token compression for prefill + FlexiExit for decode/training.

Design goals:
- Prefill/generation can use the v35 three-stage quadtree/recover/RSS top-k visual-token compression path.
- Cached single-token decode uses the FlexiExit router/adapter path.
- Training keeps the sequence batch-shaped and trains only FlexiExit modules with an adaptive token-level skip loss.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import math
import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers import AutoModel
from transformers.models.llava.configuration_llava import LlavaConfig

logger = logging.get_logger(f"transformers.{__name__}")



# =============================================================================
# FlexiExit: compact single-file implementation
# =============================================================================


# =============================================================================
# Split helper modules
# =============================================================================


def _load_hybrid_sibling(module_stem: str):
    module_name = f"{__package__}.{module_stem}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = Path(__file__).with_name(f"{module_stem}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load hybrid helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_flexiexit_helpers = _load_hybrid_sibling("modeling_llava_v35_flexiexit_hybrid_flexiexit")
_attention_helpers = _load_hybrid_sibling("modeling_llava_v35_flexiexit_hybrid_attention")
_decode_helpers = _load_hybrid_sibling("modeling_llava_v35_flexiexit_hybrid_decode")

_FlexiExitRouter = _flexiexit_helpers._FlexiExitRouter
_FlexiExitAdapter = _flexiexit_helpers._FlexiExitAdapter
flexiexit_tau = _flexiexit_helpers.flexiexit_tau
_FlexiExitState = _flexiexit_helpers._FlexiExitState
_is_llama_like_decoder = _flexiexit_helpers._is_llama_like_decoder
_parse_flexiexit_layers = _flexiexit_helpers._parse_flexiexit_layers

_create_llama_causal_mask = _attention_helpers._create_llama_causal_mask
_llama_create_causal_mask = _attention_helpers._llama_create_causal_mask
_apply_rotary_pos_emb = _attention_helpers._apply_rotary_pos_emb
_repeat_kv = _attention_helpers._repeat_kv

_flexiexit_decode_layer_forward = _decode_helpers._flexiexit_decode_layer_forward
_flexiexit_decode_exited_layers_forward = _decode_helpers._flexiexit_decode_exited_layers_forward
_flexiexit_prefill_layer_forward = _decode_helpers._flexiexit_prefill_layer_forward
_flexiexit_train_layer_forward = _decode_helpers._flexiexit_train_layer_forward


# =============================================================================
# LLaVA outputs
# =============================================================================


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava outputs, with hidden states and attentions.
    """
)
class LlavaModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: Optional[torch.FloatTensor] = None
    pruned_input_ids: Optional[torch.LongTensor] = None
    pruned_attention_mask: Optional[torch.Tensor] = None
    pruned_keep_positions: Optional[torch.LongTensor] = None

    # FlexiExit diagnostics.  All are optional and only populated when
    # flexiexit=True and the corresponding runtime flags request tracking.
    flexiexit_depth_map: Optional[torch.FloatTensor] = None
    flexiexit_exit_mask: Optional[torch.Tensor] = None
    flexiexit_exit_layer: Optional[torch.LongTensor] = None
    flexiexit_gates: Optional[tuple[torch.FloatTensor, ...]] = None
    flexi_token_costs: Optional[Dict[str, torch.FloatTensor]] = None
    # Decode-only diagnostics. Populated when record_flexiexit_stats=True.
    flexiexit_decode_exit_layer: Optional[torch.LongTensor] = None
    flexiexit_decode_active_depth: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava causal language model (or autoregressive) outputs.
    """
)
class LlavaCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None

    flexiexit_depth_map: Optional[torch.FloatTensor] = None
    flexiexit_exit_mask: Optional[torch.Tensor] = None
    flexiexit_exit_layer: Optional[torch.LongTensor] = None
    flexiexit_gates: Optional[tuple[torch.FloatTensor, ...]] = None
    flexi_token_costs: Optional[Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]] = None
    flexiexit_decode_exit_layer: Optional[torch.LongTensor] = None
    flexiexit_decode_active_depth: Optional[torch.FloatTensor] = None
    full_loss_map: Optional[torch.FloatTensor] = None
    pruned_input_ids: Optional[torch.LongTensor] = None
    pruned_attention_mask: Optional[torch.Tensor] = None
    pruned_keep_positions: Optional[torch.LongTensor] = None


# =============================================================================
# LLaVA model
# =============================================================================


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@auto_docstring
class LlavaPreTrainedModel(PreTrainedModel):
    config: LlavaConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"

    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_flex_attn = True
    _supports_attention_backend = True


@auto_docstring(
    custom_intro="""
    The Llava model which consists of a vision backbone and a language model, without a language modeling head.
    """
)
class LlavaModel(LlavaPreTrainedModel):
    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}

    def __init__(self, config: LlavaConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config)

        # v35 prefill/generation state. These are used only by the visual-token
        # compression path and by the subsequent cached decode step.
        self.reassign_position_ids = bool(getattr(config, "reassign_position_ids", False))
        self._pruning_applied = False
        self._next_position_id = None
        self._prefill_cache_position_cache = {}
        self._prefill_causal_mask_cache = {}
        self._recover_pair_cache = {}
        self._warned_v35_batch_fallback = False

        self._flexiexit_enabled = False
        self._flexiexit_tau = float(getattr(config, "flexiexit_tau", 0.5))
        self._enable_learnable_tau = bool(getattr(config, "enable_learnable_tau", True))
        self._flexiexit_router_ratio = int(getattr(config, "flexiexit_router_ratio", 16))
        self._flexiexit_adapter_ratio = int(getattr(config, "flexiexit_adapter_ratio", 16))
        self._flexiexit_layers: list[int] = []

        # Decode-time FlexiExit statistics. These are intentionally kept outside
        # the HF generation outputs because GenerationMixin drops arbitrary
        # per-step ModelOutput fields. Test/eval scripts can read them through
        # get_flexiexit_stats() after generate().
        self._record_flexiexit_stats = bool(getattr(config, "record_flexiexit_stats", False))
        self._flexiexit_decode_records: List[Dict[str, Any]] = []
        self._flexiexit_decode_step = 0
        self._last_flexiexit_decode_exit_layer: Optional[torch.Tensor] = None
        self._last_flexiexit_decode_active_depth: Optional[torch.Tensor] = None

        if bool(getattr(config, "flexiexit", False) or getattr(config.text_config, "flexiexit", False)):
            self.enable_flexiexit(
                layers=getattr(config, "flexiexit_layers", None),
            )

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    # -------------------------------------------------------------------------
    # v35 visual-token compression utilities and prefill implementation
    # -------------------------------------------------------------------------

    def _compute_visual_token_importance_scores(
        self,
        scoring_layer,
        hidden_states: Optional[torch.Tensor],
        hidden_normed: torch.Tensor,
        attn_output: torch.Tensor,
        attn_weights: torch.Tensor,
        image_start_idx: int,
        image_end_idx: int,
        norm_weight: float = 0.5,
        combine_type: str = "mul",
        importance_mode: str = "attn",
        ffn_propagation: str = "jvp",
        query_index: int = -1,
    ) -> torch.Tensor:
        self_attn = scoring_layer.self_attn
        bsz, seq_len, hidden_size = hidden_normed.shape
        num_heads = self_attn.config.num_attention_heads
        num_kv_heads = self_attn.config.num_key_value_heads
        head_dim = self_attn.head_dim

        v_proj_out = self_attn.v_proj(hidden_normed)
        v_states = v_proj_out.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        if num_kv_heads != num_heads:
            n_rep = num_heads // num_kv_heads
            v_states = (
                v_states.unsqueeze(2)
                .expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
                .reshape(bsz, num_heads, seq_len, head_dim)
            )

        attn_last = attn_weights[:, :, query_index, :]
        z_heads = torch.einsum("bhs,bhsd->bhd", attn_last, v_states)[0]
        v_visual = v_states[0, :, image_start_idx:image_end_idx, :]
        alpha_visual = attn_last[0, :, image_start_idx:image_end_idx]
        beta = alpha_visual / (1.0 - alpha_visual).clamp(min=1e-6)
        delta_z = beta.unsqueeze(-1) * (z_heads.unsqueeze(1) - v_visual)
        delta_z_cat = delta_z.permute(1, 0, 2).contiguous().view(-1, num_heads * head_dim)
        delta_y = self_attn.o_proj(delta_z_cat)

        if importance_mode == "attn":
            x_vec = attn_output[0, query_index, :]
            delta_x = delta_y
        elif importance_mode == "ffn":
            if hidden_states is None or bsz != 1:
                raise ValueError("importance_mode='ffn' requires batch_size=1 hidden_states.")

            residual_vec = hidden_states[0, query_index, :]
            attn_vec = attn_output[0, query_index, :]
            u_vec = residual_vec + attn_vec
            post_ln = scoring_layer.post_attention_layernorm
            mlp = scoring_layer.mlp
            z_base = post_ln(u_vec.view(1, 1, -1)).view(-1)
            ffn_base = mlp(z_base.view(1, 1, -1)).view(-1)
            x_vec = u_vec + ffn_base
            delta_u = delta_y

            if ffn_propagation == "exact":
                u_prime = u_vec.unsqueeze(0) + delta_u
                z_prime = post_ln(u_prime.view(-1, 1, hidden_size)).view(-1, hidden_size)
                ffn_prime = mlp(z_prime.view(-1, 1, hidden_size)).view(-1, hidden_size)
                delta_x = u_prime + ffn_prime - x_vec.unsqueeze(0)
            elif ffn_propagation == "jvp":
                w = getattr(post_ln, "weight", None)
                eps = getattr(post_ln, "eps", getattr(post_ln, "variance_epsilon", 1e-6))
                if w is None:
                    raise AttributeError("post_attention_layernorm has no weight.")

                d = u_vec.numel()
                r2 = u_vec.pow(2).mean() + eps
                r = torch.sqrt(r2)
                dot = (delta_u * u_vec.unsqueeze(0)).sum(dim=-1, keepdim=True)
                proj = u_vec.unsqueeze(0) * (dot / (d * r2))
                delta_z = (w / r).unsqueeze(0) * (delta_u - proj)

                gate_base = mlp.gate_proj(z_base)
                up_base = mlp.up_proj(z_base)
                act_base = F.silu(gate_base)
                sig = torch.sigmoid(gate_base)
                silu_prime = sig + gate_base * sig * (1.0 - sig)
                delta_gate = mlp.gate_proj(delta_z)
                delta_up = mlp.up_proj(delta_z)
                delta_act = delta_gate * silu_prime.unsqueeze(0)
                delta_m = delta_act * up_base.unsqueeze(0) + act_base.unsqueeze(0) * delta_up
                delta_ffn = mlp.down_proj(delta_m)
                delta_x = delta_u + delta_ffn
            else:
                raise ValueError(f"Unsupported ffn_propagation: {ffn_propagation}")
        else:
            raise ValueError(f"Unsupported importance_mode: {importance_mode}")

        x_norm = x_vec.norm(p=2).clamp(min=1e-6)
        delta_x_norm = delta_x.norm(dim=-1)
        x_dot_delta = torch.matmul(delta_x, x_vec)
        new_norm_sq = x_norm**2 + 2.0 * x_dot_delta + delta_x_norm**2
        new_norm = torch.sqrt(new_norm_sq.clamp(min=1e-12))
        cos_theta = (x_norm**2 + x_dot_delta) / (x_norm * new_norm + 1e-6)
        cos_theta = cos_theta.clamp(min=-1.0, max=1.0)
        angle_loss = 1.0 - cos_theta

        if combine_type == "add":
            return norm_weight * delta_x_norm + (1.0 - norm_weight) * angle_loss
        if combine_type == "mul":
            return (delta_x_norm.clamp(min=1e-12) ** norm_weight) * (
                angle_loss.clamp(min=1e-12) ** (1.0 - norm_weight)
            )
        raise ValueError(f"Unsupported combine_type: {combine_type}")

    def _compute_visual_token_importance_scores_batch(
        self,
        scoring_layer,
        hidden_states: Optional[torch.Tensor],
        hidden_normed: torch.Tensor,
        attn_output: torch.Tensor,
        attn_weights: torch.Tensor,
        image_start_idx: int,
        image_end_idx: int,
        query_indices: torch.Tensor,
        norm_weight: float = 0.5,
        combine_type: str = "mul",
        importance_mode: str = "attn",
        ffn_propagation: str = "jvp",
    ) -> torch.Tensor:
        self_attn = scoring_layer.self_attn
        bsz, seq_len, hidden_size = hidden_normed.shape
        if bsz != 1:
            raise ValueError("Current pruning/importance implementation assumes batch_size=1.")

        q_idx = query_indices.to(dtype=torch.long, device=hidden_normed.device)
        num_heads = self_attn.config.num_attention_heads
        num_kv_heads = self_attn.config.num_key_value_heads
        head_dim = self_attn.head_dim

        v_proj_out = self_attn.v_proj(hidden_normed)
        v_states = v_proj_out.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        if num_kv_heads != num_heads:
            n_rep = num_heads // num_kv_heads
            v_states = (
                v_states.unsqueeze(2)
                .expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
                .reshape(bsz, num_heads, seq_len, head_dim)
            )

        attn_last = attn_weights[:, :, q_idx, :]
        attn_last_b0 = attn_last[0].permute(1, 0, 2).contiguous()
        v_states_b0 = v_states[0]
        z_heads = torch.einsum("qhs,hsd->qhd", attn_last_b0, v_states_b0)

        v_visual = v_states_b0[:, image_start_idx:image_end_idx, :]
        alpha_visual = attn_last_b0[:, :, image_start_idx:image_end_idx]
        beta = alpha_visual / (1.0 - alpha_visual).clamp(min=1e-6)
        delta_z = beta.unsqueeze(-1) * (z_heads.unsqueeze(2) - v_visual.unsqueeze(0))
        delta_z_cat = delta_z.permute(0, 2, 1, 3).contiguous().view(-1, num_heads * head_dim)
        delta_y = self_attn.o_proj(delta_z_cat).view(q_idx.numel(), -1, hidden_size)

        if importance_mode == "attn":
            x_vec = attn_output[0, q_idx, :]
            delta_x = delta_y
        elif importance_mode == "ffn":
            post_ln = scoring_layer.post_attention_layernorm
            mlp = scoring_layer.mlp
            residual_vec = hidden_states[0, q_idx, :]
            attn_vec = attn_output[0, q_idx, :]
            u_vec = residual_vec + attn_vec
            z_base = post_ln(u_vec.view(-1, 1, hidden_size)).view(-1, hidden_size)
            ffn_base = mlp(z_base.view(-1, 1, hidden_size)).view(-1, hidden_size)
            x_vec = u_vec + ffn_base
            delta_u = delta_y

            if ffn_propagation == "exact":
                u_prime = u_vec.unsqueeze(1) + delta_u
                z_prime = post_ln(u_prime.view(-1, 1, hidden_size)).view(-1, hidden_size)
                ffn_prime = mlp(z_prime.view(-1, 1, hidden_size)).view(-1, hidden_size)
                delta_x = u_prime + ffn_prime.view(q_idx.numel(), -1, hidden_size) - x_vec.unsqueeze(1)
            elif ffn_propagation == "jvp":
                w = getattr(post_ln, "weight", None)
                eps = getattr(post_ln, "eps", getattr(post_ln, "variance_epsilon", 1e-6))
                if w is None:
                    raise AttributeError("post_attention_layernorm has no weight.")
                d = u_vec.shape[-1]
                r2 = u_vec.pow(2).mean(dim=-1) + eps
                r = torch.sqrt(r2)
                dot = (delta_u * u_vec.unsqueeze(1)).sum(dim=-1, keepdim=True)
                proj = u_vec.unsqueeze(1) * (dot / (d * r2.view(-1, 1, 1)))
                delta_z = (w / r.view(-1, 1)).unsqueeze(1) * (delta_u - proj)

                gate_base = mlp.gate_proj(z_base)
                up_base = mlp.up_proj(z_base)
                act_base = F.silu(gate_base)
                sig = torch.sigmoid(gate_base)
                silu_prime = sig + gate_base * sig * (1.0 - sig)
                delta_gate = mlp.gate_proj(delta_z.reshape(-1, hidden_size)).view(q_idx.numel(), -1, gate_base.shape[-1])
                delta_up = mlp.up_proj(delta_z.reshape(-1, hidden_size)).view(q_idx.numel(), -1, up_base.shape[-1])
                delta_act = delta_gate * silu_prime.unsqueeze(1)
                delta_m = delta_act * up_base.unsqueeze(1) + act_base.unsqueeze(1) * delta_up
                delta_ffn = mlp.down_proj(delta_m.reshape(-1, gate_base.shape[-1])).view(q_idx.numel(), -1, hidden_size)
                delta_x = delta_u + delta_ffn
            else:
                raise ValueError(f"Unsupported ffn_propagation: {ffn_propagation}")
        else:
            raise ValueError(f"Unsupported importance_mode: {importance_mode}")

        x_norm = x_vec.norm(p=2, dim=-1).clamp(min=1e-6)
        delta_x_norm = delta_x.norm(dim=-1)
        x_dot_delta = (delta_x * x_vec.unsqueeze(1)).sum(dim=-1)
        new_norm_sq = x_norm.unsqueeze(1) ** 2 + 2.0 * x_dot_delta + delta_x_norm**2
        new_norm = torch.sqrt(new_norm_sq.clamp(min=1e-12))
        cos_theta = (x_norm.unsqueeze(1) ** 2 + x_dot_delta) / (x_norm.unsqueeze(1) * new_norm + 1e-6)
        cos_theta = cos_theta.clamp(min=-1.0, max=1.0)
        angle_loss = 1.0 - cos_theta

        if combine_type == "add":
            return norm_weight * delta_x_norm + (1.0 - norm_weight) * angle_loss
        if combine_type == "mul":
            return (delta_x_norm.clamp(min=1e-12) ** norm_weight) * (
                angle_loss.clamp(min=1e-12) ** (1.0 - norm_weight)
            )
        raise ValueError(f"Unsupported combine_type: {combine_type}")

    def _get_visual_token_attention_scores(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        scoring_layer_idx: int,
        image_start_idx: int,
        image_end_idx: int,
        importance_mode: str = "attn",
        ffn_propagation: str = "jvp",
        attn_anchor: str = "last",
        pool_type: str = "avg",
        query_indices: Optional[torch.Tensor] = None,
    ):
        layers = self.language_model.layers
        rotary_emb = self.language_model.rotary_emb
        pos_emb = rotary_emb(hidden_states, position_ids)
        scoring_layer = layers[scoring_layer_idx]
        hidden_normed = scoring_layer.input_layernorm(hidden_states)

        if attn_anchor != "query":
            return self._get_visual_token_attention_scores_last_query(
                scoring_layer=scoring_layer,
                hidden_normed=hidden_normed,
                position_embeddings=pos_emb,
                image_start_idx=image_start_idx,
                image_end_idx=image_end_idx,
            )

        causal_mask = _create_llama_causal_mask(
            config=self.language_model.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        attn_output, attn_weights = scoring_layer.self_attn(
            hidden_states=hidden_normed,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=pos_emb,
            output_attentions=True,
        )

        if attn_anchor == "query" and query_indices is not None and query_indices.numel() > 0:
            attn_last = attn_weights[:, :, query_indices, :]
            attn_last_b0 = attn_last[0].permute(1, 0, 2).contiguous()
            avg_attn_full = attn_last_b0.mean(dim=1)
            visual_attn = avg_attn_full[:, image_start_idx:image_end_idx]
            attention_quality_q = (visual_attn.sum(dim=-1) / avg_attn_full.sum(dim=-1)).clamp(min=0.0, max=1.0)
            importance_scores_q = self._compute_visual_token_importance_scores_batch(
                scoring_layer=scoring_layer,
                hidden_states=hidden_states,
                hidden_normed=hidden_normed,
                attn_output=attn_output,
                attn_weights=attn_weights,
                image_start_idx=image_start_idx,
                image_end_idx=image_end_idx,
                query_indices=query_indices,
                norm_weight=1.0,
                combine_type="add",
                importance_mode=importance_mode,
                ffn_propagation=ffn_propagation,
            )
            if pool_type == "avg":
                return visual_attn.mean(dim=0), importance_scores_q.mean(dim=0), attention_quality_q.mean()
            return visual_attn.max(dim=0).values, importance_scores_q.max(dim=0).values, attention_quality_q.max()

        attn_last = attn_weights[:, :, -1, :]
        avg_attn_full = attn_last[0].mean(dim=0)
        visual_attn = avg_attn_full[image_start_idx:image_end_idx]
        attention_quality = (visual_attn.sum() / avg_attn_full.sum()).clamp(min=0.0, max=1.0)
        importance_scores = self._compute_visual_token_importance_scores(
            scoring_layer=scoring_layer,
            hidden_states=hidden_states,
            hidden_normed=hidden_normed,
            attn_output=attn_output,
            attn_weights=attn_weights,
            image_start_idx=image_start_idx,
            image_end_idx=image_end_idx,
            norm_weight=1.0,
            combine_type="add",
            importance_mode=importance_mode,
            ffn_propagation=ffn_propagation,
            query_index=-1,
        )
        return visual_attn, importance_scores, attention_quality

    def _get_visual_token_attention_scores_last_query(
        self,
        scoring_layer,
        hidden_normed: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        image_start_idx: int,
        image_end_idx: int,
    ):
        self_attn = scoring_layer.self_attn
        bsz, seq_len, _ = hidden_normed.shape
        num_heads = self_attn.config.num_attention_heads
        num_kv_heads = self_attn.config.num_key_value_heads
        head_dim = self_attn.head_dim

        query_states = self_attn.q_proj(hidden_normed[:, -1:, :]).view(bsz, 1, num_heads, head_dim).transpose(1, 2)
        key_states = self_attn.k_proj(hidden_normed).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = self_attn.v_proj(hidden_normed).view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, _ = _apply_rotary_pos_emb(query_states, query_states, cos[:, -1:, :], sin[:, -1:, :])
        _, key_states = _apply_rotary_pos_emb(key_states, key_states, cos, sin)
        key_states = _repeat_kv(key_states, self_attn.num_key_value_groups)
        value_states = _repeat_kv(value_states, self_attn.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self_attn.scaling
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        z_heads = torch.matmul(attn_weights, value_states)[0, :, 0, :]

        alpha_visual = attn_weights[0, :, 0, image_start_idx:image_end_idx]
        v_visual = value_states[0, :, image_start_idx:image_end_idx, :]
        beta = alpha_visual / (1.0 - alpha_visual).clamp(min=1e-6)
        delta_z = beta.unsqueeze(-1) * (z_heads.unsqueeze(1) - v_visual)
        delta_z_cat = delta_z.permute(1, 0, 2).contiguous().view(-1, num_heads * head_dim)
        delta_y = self_attn.o_proj(delta_z_cat)
        return None, delta_y.norm(dim=-1), None

    def _compute_spatial_distance_matrix(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        side = int(num_tokens ** 0.5)
        if side * side != num_tokens:
            return torch.ones((num_tokens, num_tokens), device=device)
        y_coords = torch.arange(side, device=device).repeat_interleave(side).float()
        x_coords = torch.arange(side, device=device).repeat(side).float()
        y_coords /= max(side - 1, 1)
        x_coords /= max(side - 1, 1)
        coords = torch.stack([y_coords, x_coords], dim=1)
        return torch.cdist(coords, coords, p=2)

    def _set_cache_seen_tokens(self, cache: object, seen_tokens: int) -> None:
        for attr in ("seen_tokens", "_seen_tokens", "past_seen_tokens", "num_tokens", "_num_tokens"):
            if hasattr(cache, attr):
                try:
                    setattr(cache, attr, int(seen_tokens))
                except Exception:
                    pass

    def _index_select_kv_seq_dim(
        self,
        kv: torch.Tensor,
        index: torch.LongTensor,
        seq_len_hint: Optional[int] = None,
    ) -> torch.Tensor:
        if kv is None or (not torch.is_tensor(kv)) or kv.dim() < 2:
            return kv

        seq_dim = None
        if kv.dim() == 4:
            if seq_len_hint is not None:
                if kv.size(2) == seq_len_hint:
                    seq_dim = 2
                elif kv.size(1) == seq_len_hint:
                    seq_dim = 1
                elif kv.size(-2) == seq_len_hint:
                    seq_dim = -2
            if seq_dim is None:
                seq_dim = 2
        elif kv.dim() == 3:
            if seq_len_hint is not None:
                if kv.size(1) == seq_len_hint:
                    seq_dim = 1
                elif kv.size(-2) == seq_len_hint:
                    seq_dim = -2
            if seq_dim is None:
                seq_dim = 1
        else:
            seq_dim = -2

        return kv.index_select(seq_dim, index.to(device=kv.device, dtype=torch.long))

    def _prune_cache_layers_by_positions(
        self,
        past_key_values: Optional[Cache],
        layer_start: int,
        layer_end: int,
        keep_positions: torch.LongTensor,
        seq_len_hint: Optional[int] = None,
    ) -> Optional[Cache]:
        if (
            past_key_values is None
            or not isinstance(past_key_values, Cache)
            or not hasattr(past_key_values, "layers")
            or layer_start >= layer_end
        ):
            return past_key_values

        layer_start = max(int(layer_start), 0)
        layer_end = min(int(layer_end), len(past_key_values.layers))
        keep_positions = keep_positions.to(dtype=torch.long)

        for layer_idx in range(layer_start, layer_end):
            layer = past_key_values.layers[layer_idx]
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if torch.is_tensor(keys) and keys.numel() > 0:
                layer.keys = self._index_select_kv_seq_dim(keys, keep_positions, seq_len_hint)
                if hasattr(layer, "is_initialized"):
                    layer.is_initialized = True
            if torch.is_tensor(values) and values.numel() > 0:
                layer.values = self._index_select_kv_seq_dim(values, keep_positions, seq_len_hint)
                if hasattr(layer, "is_initialized"):
                    layer.is_initialized = True
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = int(keep_positions.numel())

        return past_key_values

    def _map_original_keep_positions_to_compressed_cache_positions(
        self,
        keep_positions: torch.LongTensor,
        image_start_idx: int,
        image_end_idx: int,
        child_to_parent: torch.LongTensor,
        compressed_visual_count: int,
    ) -> torch.LongTensor:
        mapped_positions = keep_positions.clone()
        visual_mask = (keep_positions >= image_start_idx) & (keep_positions < image_end_idx)
        post_visual_mask = keep_positions >= image_end_idx

        if visual_mask.any():
            visual_child_indices = keep_positions[visual_mask] - image_start_idx
            parent_indices = child_to_parent.to(device=keep_positions.device, dtype=torch.long).index_select(
                0, visual_child_indices
            )
            mapped_positions[visual_mask] = image_start_idx + parent_indices
        if post_visual_mask.any():
            mapped_positions[post_visual_mask] = (
                keep_positions[post_visual_mask] - (image_end_idx - image_start_idx) + int(compressed_visual_count)
            )

        return mapped_positions

    def _get_prefill_cache_position(self, seq_len: int, device: torch.device) -> torch.LongTensor:
        key = (device.type, device.index, int(seq_len))
        cache_position = self._prefill_cache_position_cache.get(key)
        if cache_position is None:
            cache_position = torch.arange(seq_len, device=device, dtype=torch.long)
            self._prefill_cache_position_cache[key] = cache_position
        return cache_position

    def _create_prefill_causal_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
        is_full_attention_mask: bool = False,
    ):
        if is_full_attention_mask and self.language_model.config._attn_implementation == "eager":
            key = (hidden_states.device.type, hidden_states.device.index, hidden_states.dtype, int(hidden_states.size(1)))
            causal_mask = self._prefill_causal_mask_cache.get(key)
            if causal_mask is None:
                seq_len = hidden_states.size(1)
                min_dtype = torch.finfo(hidden_states.dtype).min
                causal_mask = torch.full(
                    (1, 1, seq_len, seq_len),
                    min_dtype,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                causal_mask = torch.triu(causal_mask, diagonal=1)
                self._prefill_causal_mask_cache[key] = causal_mask
            return causal_mask

        if is_full_attention_mask and self.language_model.config._attn_implementation == "sdpa":
            return None

        return _create_llama_causal_mask(
            config=self.language_model.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

    def _get_cache_layer_seq_len(self, cache: Optional[Cache], layer_idx: int) -> int:
        if not isinstance(cache, Cache) or not hasattr(cache, "layers") or layer_idx >= len(cache.layers):
            return 0
        layer = cache.layers[layer_idx]
        keys = getattr(layer, "keys", None)
        if torch.is_tensor(keys) and keys.numel() > 0:
            return int(keys.shape[-2])
        cumulative_length = getattr(layer, "cumulative_length", None)
        if cumulative_length is not None:
            try:
                return int(cumulative_length)
            except Exception:
                return 0
        return 0

    def _create_layer_specific_causal_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
        position_ids: Optional[torch.Tensor],
        layer_idx: int,
    ):
        if not isinstance(past_key_values, Cache):
            return _create_llama_causal_mask(
                config=self.language_model.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        class _SingleLayerCacheMaskView:
            def __init__(self, cache: Cache, target_layer_idx: int):
                self.cache = cache
                self.target_layer_idx = int(target_layer_idx)
                layer_is_sliding = False
                layer_is_compileable = False
                if hasattr(cache, "layers") and self.target_layer_idx < len(cache.layers):
                    layer = cache.layers[self.target_layer_idx]
                    layer_is_sliding = bool(getattr(layer, "is_sliding", False))
                    layer_is_compileable = bool(getattr(layer, "is_compileable", False))
                self.is_sliding = [layer_is_sliding]
                self.is_compileable = layer_is_compileable

            def get_mask_sizes(self, local_cache_position: torch.Tensor, layer_idx: int = 0):
                return self.cache.get_mask_sizes(local_cache_position, self.target_layer_idx)

        return _create_llama_causal_mask(
            config=self.language_model.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=_SingleLayerCacheMaskView(past_key_values, layer_idx),
            position_ids=position_ids,
        )

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
        return attention_scores * suppression_factor

    def _compute_quadtree_redundancy_scores(
        self,
        similarity_values: torch.Tensor,
        variance_values: torch.Tensor,
    ) -> torch.Tensor:
        if similarity_values.numel() == 0:
            return similarity_values.new_zeros((0,))

        sim_values = similarity_values.float()
        var_values = variance_values.float()
        sim_q25 = torch.quantile(sim_values, 0.25)
        sim_q75 = torch.quantile(sim_values, 0.75)
        var_q25 = torch.quantile(var_values, 0.25)
        var_q75 = torch.quantile(var_values, 0.75)
        sim_scale = (sim_q75 - sim_q25).clamp(min=1e-4)
        var_scale = (var_q75 - var_q25).clamp(min=1e-4)
        sim_center = torch.quantile(sim_values, 0.5)
        var_center = torch.quantile(var_values, 0.5)

        sim_score = (sim_values - sim_center) / sim_scale
        var_score = (var_center - var_values) / var_scale
        return sim_score + 0.75 * var_score

    def _merge_visual_tokens_quadtree_exact_target(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        image_start_idx: int,
        image_end_idx: int,
        target_count: int,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.LongTensor],
        Optional[torch.Tensor],
        int,
        int,
        torch.LongTensor,
        torch.Tensor,
        torch.LongTensor,
        List[int],
        int,
    ]:
        """Merge the most redundant 2x2 quadtree sibling groups to hit an exact visual target.

        Returns the compressed sequence plus a child_to_parent map so that the merged visual
        tokens can later be expanded back to the original visual-token grid.
        """
        visual_embeds = inputs_embeds[:, image_start_idx:image_end_idx, :]
        num_visual = image_end_idx - image_start_idx
        device = inputs_embeds.device
        resolved_target_count = max(1, min(int(target_count), num_visual))

        def _identity_merge():
            keep_indices = torch.arange(num_visual, device=device, dtype=torch.long)
            child_to_parent = keep_indices.clone()
            return (
                inputs_embeds,
                input_ids,
                attention_mask,
                image_start_idx,
                image_end_idx,
                keep_indices,
                visual_embeds,
                child_to_parent,
                [num_visual],
                resolved_target_count,
            )

        side = int(num_visual ** 0.5)
        if num_visual <= 1 or side * side != num_visual or side % 2 != 0 or resolved_target_count >= num_visual:
            return _identity_merge()

        # Each selected 2x2 group reduces token count by 3, so this exact budget gives
        # 576 -> 288 when target_count=288.
        merge_budget = min((side // 2) * (side // 2), max(0, int(round((num_visual - resolved_target_count) / 3.0))))
        if merge_budget <= 0:
            return _identity_merge()

        hidden_size = visual_embeds.shape[-1]
        visual_grid = visual_embeds[0].view(side, side, hidden_size)
        group_embeds = (
            visual_grid.view(side // 2, 2, side // 2, 2, hidden_size)
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, 4, hidden_size)
        )
        group_indices = (
            torch.arange(num_visual, device=device, dtype=torch.long)
            .view(side, side)
            .view(side // 2, 2, side // 2, 2)
            .permute(0, 2, 1, 3)
            .reshape(-1, 4)
        )

        group_norm = F.normalize(group_embeds.float(), p=2, dim=-1)
        sim_matrix = torch.matmul(group_norm, group_norm.transpose(1, 2))
        similarity_tensor = (sim_matrix.sum(dim=(1, 2)) - 4.0) / 12.0
        centroid = group_norm.mean(dim=1, keepdim=True)
        variance_tensor = ((group_norm - centroid) ** 2).sum(dim=-1).mean(dim=1)
        centrality = sim_matrix.mean(dim=-1)

        redundancy_scores = self._compute_quadtree_redundancy_scores(similarity_tensor, variance_tensor)
        selected_candidate_indices = torch.topk(redundancy_scores, k=merge_budget, largest=True).indices
        selected_mask = torch.zeros(group_embeds.shape[0], device=device, dtype=torch.bool)
        selected_mask[selected_candidate_indices] = True

        merge_weights = torch.softmax(centrality, dim=1).to(group_embeds.dtype)
        merged_group_embeds = (group_embeds * merge_weights.unsqueeze(-1)).sum(dim=1)
        representative_local_indices = torch.argmax(centrality, dim=1)
        representative_group_indices = group_indices.gather(1, representative_local_indices.unsqueeze(1)).squeeze(1)

        local_offsets = torch.arange(4, device=device, dtype=torch.long)
        group_lengths = torch.where(
            selected_mask,
            torch.ones_like(selected_mask, dtype=torch.long),
            torch.full_like(selected_mask, 4, dtype=torch.long),
        )
        group_starts = torch.cumsum(group_lengths, dim=0) - group_lengths
        slot_mask = (~selected_mask).unsqueeze(1) | (local_offsets == 0).unsqueeze(0)

        expanded_visual = visual_embeds[0].index_select(0, group_indices.reshape(-1)).view(group_indices.shape[0], 4, hidden_size)
        expanded_visual = torch.where(
            selected_mask.view(-1, 1, 1),
            merged_group_embeds.unsqueeze(1).expand_as(expanded_visual),
            expanded_visual,
        )
        representative_grid = torch.where(
            selected_mask.unsqueeze(1),
            representative_group_indices.unsqueeze(1).expand_as(group_indices),
            group_indices,
        )

        valid_slots = slot_mask.reshape(-1)
        valid_indices = valid_slots.nonzero(as_tuple=False).squeeze(1)
        merged_visual = expanded_visual.reshape(-1, hidden_size).index_select(0, valid_indices).unsqueeze(0)
        representative_indices_tensor = representative_grid.reshape(-1).index_select(0, valid_indices)

        child_parent_values = torch.where(
            selected_mask.unsqueeze(1),
            group_starts.unsqueeze(1).expand_as(group_indices),
            group_starts.unsqueeze(1) + local_offsets.unsqueeze(0),
        )
        child_to_parent_tensor = torch.empty(num_visual, device=device, dtype=torch.long)
        child_to_parent_tensor.scatter_(0, group_indices.reshape(-1), child_parent_values.reshape(-1))

        # Keep the compressed visual tokens in original-raster order. Update child_to_parent
        # to point into the sorted parent sequence.
        sort_order = torch.argsort(representative_indices_tensor)
        inverse_sort_order = torch.empty_like(sort_order)
        inverse_sort_order[sort_order] = torch.arange(sort_order.numel(), device=device, dtype=torch.long)
        child_to_parent_tensor = inverse_sort_order.index_select(0, child_to_parent_tensor)
        representative_indices_tensor = representative_indices_tensor.index_select(0, sort_order)
        merged_visual = merged_visual.index_select(1, sort_order)

        pre_visual = inputs_embeds[:, :image_start_idx, :]
        post_visual = inputs_embeds[:, image_end_idx:, :]
        merged_inputs_embeds = torch.cat([pre_visual, merged_visual, post_visual], dim=1)

        merged_input_ids = input_ids
        if input_ids is not None:
            pre_ids = input_ids[:, :image_start_idx]
            visual_ids = input_ids[:, image_start_idx:image_end_idx]
            post_ids = input_ids[:, image_end_idx:]
            merged_input_ids = torch.cat([pre_ids, visual_ids[:, representative_indices_tensor], post_ids], dim=1)

        merged_attention_mask = attention_mask
        if attention_mask is not None:
            pre_mask = attention_mask[:, :image_start_idx]
            post_mask = attention_mask[:, image_end_idx:]
            visual_mask = attention_mask.new_ones((attention_mask.shape[0], merged_visual.shape[1]))
            merged_attention_mask = torch.cat([pre_mask, visual_mask, post_mask], dim=1)

        token_counts = [num_visual, merged_visual.shape[1]]
        return (
            merged_inputs_embeds,
            merged_input_ids,
            merged_attention_mask,
            image_start_idx,
            image_start_idx + merged_visual.shape[1],
            representative_indices_tensor,
            merged_visual,
            child_to_parent_tensor,
            token_counts,
            resolved_target_count,
        )

    def _estimate_recover_per_head_gamma(
        self,
        current_visual: torch.Tensor,
        parent_hidden_at_merge: torch.Tensor,
        child_to_parent: torch.LongTensor,
        child_hidden_at_merge: torch.Tensor,
        min_calib_groups: int = 4,
        one_sided: bool = False,
        recover_use_identity_gate: bool = False,
        recover_use_sample_count_shrinkage: bool = False,
    ) -> Optional[torch.Tensor]:
        num_children = int(child_hidden_at_merge.size(1))
        num_parents = int(parent_hidden_at_merge.size(1))
        hidden_size = int(child_hidden_at_merge.size(-1))
        if num_children <= 1 or num_parents <= 0:
            return None
        if current_visual.size(0) != 1 or child_hidden_at_merge.size(0) != 1:
            return None

        num_heads = int(getattr(self.language_model.config, "num_attention_heads", 1))
        if num_heads <= 0 or hidden_size % num_heads != 0:
            return None
        head_dim = hidden_size // num_heads
        device = child_hidden_at_merge.device

        side = int(num_children ** 0.5)
        if side * side != num_children or side % 2 != 0:
            return None

        group_indices = (
            torch.arange(num_children, device=device, dtype=torch.long)
            .view(side, side)
            .view(side // 2, 2, side // 2, 2)
            .permute(0, 2, 1, 3)
            .reshape(-1, 4)
        )

        parent_counts = torch.bincount(child_to_parent, minlength=num_parents)
        retained_child_mask = parent_counts.index_select(0, child_to_parent) == 1
        calib_group_mask = retained_child_mask.index_select(0, group_indices.reshape(-1)).view(-1, 4).all(dim=1)
        calib_group_indices = calib_group_mask.nonzero(as_tuple=False).squeeze(1)
        num_calib_groups = int(calib_group_indices.numel())
        if num_calib_groups < min_calib_groups:
            return None

        selected_group_indices = group_indices.index_select(0, calib_group_indices)
        selected_parent_indices = child_to_parent.index_select(0, selected_group_indices.reshape(-1)).view(-1, 4)

        shallow_children = child_hidden_at_merge[:, selected_group_indices.reshape(-1), :].view(
            num_calib_groups, 4, hidden_size
        )
        deep_children = current_visual[:, selected_parent_indices.reshape(-1), :].view(
            num_calib_groups, 4, hidden_size
        )

        shallow_norm = F.normalize(shallow_children.float(), p=2, dim=-1)
        sim_matrix = torch.matmul(shallow_norm, shallow_norm.transpose(1, 2))
        centrality = sim_matrix.mean(dim=-1)
        weights = torch.softmax(centrality, dim=1).to(dtype=child_hidden_at_merge.dtype)

        shallow_center = (shallow_children * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
        deep_center = (deep_children * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
        shallow_dev = (shallow_children - shallow_center).float().view(-1, num_heads, head_dim)
        deep_dev = (deep_children - deep_center).float().view(-1, num_heads, head_dim)

        denominator = shallow_dev.square().sum(dim=(0, 2)).clamp(min=1e-6)
        numerator = (deep_dev * shallow_dev).sum(dim=(0, 2))
        gamma_star = numerator / denominator
        gamma_star = torch.nan_to_num(gamma_star, nan=1.0, posinf=1.0, neginf=1.0)
        gamma_star = gamma_star.clamp(min=0.5, max=1.5)

        sample_count = int(shallow_dev.size(0))
        if not recover_use_identity_gate and not recover_use_sample_count_shrinkage:
            gamma = gamma_star
            if one_sided:
                gamma = gamma.clamp(min=1.0)
            gamma = torch.where(torch.isfinite(gamma), gamma, torch.ones_like(gamma))
            gamma = gamma.clamp(min=1.0 if one_sided else 0.5, max=1.5)
            logger.info(
                "Calibrated per-head gamma for deviation transport: groups=%s samples=%s "
                "identity_gate=False sample_shrinkage=False fast_path=True "
                "gamma_mean=%.6f gamma_min=%.6f gamma_max=%.6f",
                num_calib_groups,
                sample_count,
                float(gamma.mean().item()),
                float(gamma.min().item()),
                float(gamma.max().item()),
            )
            return gamma.to(device=device, dtype=child_hidden_at_merge.dtype)

        if recover_use_identity_gate:
            gamma_view = gamma_star.view(1, num_heads, 1)
            error_gamma = (deep_dev - gamma_view * shallow_dev).square().sum(dim=(0, 2))
            error_identity = (deep_dev - shallow_dev).square().sum(dim=(0, 2))
            raw_gain = ((error_identity - error_gamma) / error_identity.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
            gain = raw_gain
            raw_gain_mean = float(raw_gain.mean().item())
            gain_mean = raw_gain_mean
        else:
            gain = 1.0
            raw_gain_mean = float("nan")
            gain_mean = 1.0

        rho = float(sample_count) / (float(sample_count) + 64.0) if recover_use_sample_count_shrinkage else 1.0
        eta = gain * rho

        delta = gamma_star - 1.0
        if one_sided:
            delta = delta.clamp(min=0.0)
        gamma = 1.0 + eta * delta
        gamma = torch.where(torch.isfinite(gamma), gamma, torch.ones_like(gamma))
        gamma = gamma.clamp(min=1.0 if one_sided else 0.5, max=1.5)

        logger.info(
            "Calibrated per-head gamma for deviation transport: groups=%s samples=%s "
            "identity_gate=%s sample_shrinkage=%s rho=%.6f raw_gain_mean=%.6f "
            "gain_mean=%.6f gamma_star_mean=%.6f gamma_mean=%.6f gamma_min=%.6f gamma_max=%.6f",
            num_calib_groups,
            sample_count,
            recover_use_identity_gate,
            recover_use_sample_count_shrinkage,
            float(rho),
            raw_gain_mean,
            gain_mean,
            float(gamma_star.mean().item()),
            float(gamma.mean().item()),
            float(gamma.min().item()),
            float(gamma.max().item()),
        )
        return gamma.to(device=device, dtype=child_hidden_at_merge.dtype)

    def _recover_quadtree_children_from_residual(
        self,
        hidden_states: torch.Tensor,
        image_start_idx: int,
        image_end_idx: int,
        parent_hidden_at_merge: torch.Tensor,
        child_to_parent: torch.LongTensor,
        child_hidden_at_merge: torch.Tensor,
    ) -> torch.Tensor:
        """Recover original children with mean-preserving per-head deviation transport.

        For child i in parent group g:
            h_hat_i = m_g(recover) + Gamma * (h_i(merge) - m_g(merge)).
        If Gamma cannot be estimated, Gamma=I, which reduces to residual/offset recovery.
        """
        current_visual = hidden_states[:, image_start_idx:image_end_idx, :]
        parent_current = current_visual.index_select(1, child_to_parent)
        parent_at_merge = parent_hidden_at_merge.index_select(1, child_to_parent)
        child_deviation = child_hidden_at_merge - parent_at_merge

        gamma = self._estimate_recover_per_head_gamma(
            current_visual=current_visual,
            parent_hidden_at_merge=parent_hidden_at_merge,
            child_to_parent=child_to_parent,
            child_hidden_at_merge=child_hidden_at_merge,
        )

        if gamma is not None:
            hidden_size = int(child_deviation.size(-1))
            num_heads = int(getattr(self.language_model.config, "num_attention_heads", 1))
            if num_heads > 0 and hidden_size % num_heads == 0:
                head_dim = hidden_size // num_heads
                transported_deviation = (
                    child_deviation.view(child_deviation.size(0), child_deviation.size(1), num_heads, head_dim)
                    * gamma.view(1, 1, num_heads, 1)
                ).view_as(child_deviation)
            else:
                transported_deviation = child_deviation
        else:
            transported_deviation = child_deviation

        recovered_visual = parent_current + transported_deviation
        return torch.cat(
            [
                hidden_states[:, :image_start_idx, :],
                recovered_visual,
                hidden_states[:, image_end_idx:, :],
            ],
            dim=1,
        )

    def _build_nearest_selected_visual_parent_map(
        self,
        num_children: int,
        selected_child_indices: torch.LongTensor,
        device: torch.device,
    ) -> torch.LongTensor:
        """Map every original visual token to the nearest retained visual token.

        The returned values are parent indices in the retained-token order, not original
        token ids. Retained tokens map to themselves because their spatial distance is 0.
        """
        if selected_child_indices.numel() <= 0:
            raise ValueError("selected_child_indices must contain at least one token.")

        selected_child_indices = selected_child_indices.to(device=device, dtype=torch.long)
        side = int(num_children ** 0.5)
        child_indices = torch.arange(num_children, device=device, dtype=torch.long)

        if side * side == num_children:
            child_y = torch.div(child_indices, side, rounding_mode="floor").float()
            child_x = (child_indices % side).float()
            selected_y = torch.div(selected_child_indices, side, rounding_mode="floor").float()
            selected_x = (selected_child_indices % side).float()
            distances = (child_y[:, None] - selected_y[None, :]).square() + (
                child_x[:, None] - selected_x[None, :]
            ).square()
        else:
            distances = (child_indices[:, None] - selected_child_indices[None, :]).abs().float()

        child_to_parent = distances.argmin(dim=1).to(dtype=torch.long)
        retained_child_positions = torch.empty(num_children, device=device, dtype=torch.long)
        retained_child_positions.fill_(-1)
        retained_child_positions.scatter_(
            0,
            selected_child_indices,
            torch.arange(selected_child_indices.numel(), device=device, dtype=torch.long),
        )
        retained_mask = retained_child_positions >= 0
        child_to_parent = torch.where(retained_mask, retained_child_positions, child_to_parent)
        return child_to_parent

    def _recover_selected_children_from_residual(
        self,
        hidden_states: torch.Tensor,
        image_start_idx: int,
        image_end_idx: int,
        parent_hidden_at_prune: torch.Tensor,
        child_to_parent: torch.LongTensor,
        child_hidden_at_prune: torch.Tensor,
    ) -> torch.Tensor:
        """Recover a full visual-token grid from a selected-token sequence.

        For every original child token i assigned to retained parent p(i), recover with
            h_i^L ~= h_{p(i)}^L + (h_i^P - h_{p(i)}^P),
        where P is the layer at which the parent tokens were selected and L is the
        current final-prune layer.
        """
        current_visual = hidden_states[:, image_start_idx:image_end_idx, :]
        if current_visual.size(1) != parent_hidden_at_prune.size(1):
            raise ValueError(
                "The current visual span must match parent_hidden_at_prune for residual recover."
            )
        if child_to_parent.numel() != child_hidden_at_prune.size(1):
            raise ValueError(
                "child_to_parent length must match the full child_hidden_at_prune visual span."
            )

        child_to_parent = child_to_parent.to(device=hidden_states.device, dtype=torch.long)
        parent_current = current_visual.index_select(1, child_to_parent)
        parent_at_prune = parent_hidden_at_prune.index_select(1, child_to_parent)
        recovered_visual = parent_current + (child_hidden_at_prune - parent_at_prune)
        return torch.cat(
            [
                hidden_states[:, :image_start_idx, :],
                recovered_visual,
                hidden_states[:, image_end_idx:, :],
            ],
            dim=1,
        )

    def _topk_visual_tokens_by_scores(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        attention_scores: torch.Tensor,
        image_start_idx: int,
        image_end_idx: int,
        target_count: int,
        apply_rss: bool = False,
        rss_beta: float = 1.0,
        rss_threshold: float = 0.5,
        rss_visual_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.LongTensor],
        Optional[torch.Tensor],
        int,
        int,
        torch.LongTensor,
        torch.LongTensor,
    ]:
        num_visual = image_end_idx - image_start_idx
        keep_k = max(1, min(int(target_count), num_visual))
        selection_scores = attention_scores.float()
        if apply_rss:
            # RSS uses a similarity space that is explicitly passed in by the caller.
            # For recover-based pruning this should be the original pre-compression
            # visual embedding space, while the scores come from the recovered hidden states.
            rss_source_embeds = inputs_embeds if rss_visual_embeds is None else rss_visual_embeds
            visual_embeds_for_rss = rss_source_embeds[:, image_start_idx:image_end_idx, :]
            if visual_embeds_for_rss.size(1) != num_visual:
                raise ValueError(
                    "rss_visual_embeds must have the same visual-token span as attention_scores."
                )
            selection_scores = self._apply_rss_algorithm(
                visual_embeds_for_rss,
                selection_scores,
                beta=rss_beta,
                threshold=rss_threshold,
            )
        keep_indices = torch.topk(selection_scores, k=keep_k, largest=True).indices.sort().values
        device = inputs_embeds.device

        pre_visual = inputs_embeds[:, :image_start_idx, :]
        visual_embeds = inputs_embeds[:, image_start_idx:image_end_idx, :]
        post_visual = inputs_embeds[:, image_end_idx:, :]
        pruned_embeds = torch.cat([pre_visual, visual_embeds[:, keep_indices, :], post_visual], dim=1)

        pruned_input_ids = input_ids
        if input_ids is not None:
            pre_ids = input_ids[:, :image_start_idx]
            visual_ids = input_ids[:, image_start_idx:image_end_idx]
            post_ids = input_ids[:, image_end_idx:]
            pruned_input_ids = torch.cat([pre_ids, visual_ids[:, keep_indices], post_ids], dim=1)

        pruned_attention_mask = attention_mask
        if attention_mask is not None:
            pre_mask = attention_mask[:, :image_start_idx]
            post_mask = attention_mask[:, image_end_idx:]
            visual_mask = attention_mask.new_ones((attention_mask.shape[0], keep_k))
            pruned_attention_mask = torch.cat([pre_mask, visual_mask, post_mask], dim=1)

        keep_positions = torch.cat(
            [
                torch.arange(image_start_idx, device=device, dtype=torch.long),
                keep_indices + image_start_idx,
                torch.arange(image_end_idx, inputs_embeds.size(1), device=device, dtype=torch.long),
            ]
        )
        return (
            pruned_embeds,
            pruned_input_ids,
            pruned_attention_mask,
            image_start_idx,
            image_start_idx + keep_k,
            keep_indices,
            keep_positions,
        )


    def _v35_visual_compression_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        scoring_layer_idx: Optional[int] = None,
        stage1_merge_layer_idx: Optional[int] = None,
        recover_layer_idx: Optional[int] = None,
        final_prune_layer_idx: Optional[int] = None,
        stage1_target_count: Optional[int] = None,
        recover_topk_target_count: Optional[int] = None,
        attn_anchor: str = "last",
        pool_type: str = "avg",
        visual_token_target_count: Optional[int] = None,
        rss_beta: float = 1.0,
        rss_threshold: float = 0.5,
        flexiexit_token_mask: Optional[torch.Tensor] = None,
        flexiexit_state: Optional[_FlexiExitState] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LlavaModelOutputWithPast]:

        vision_feature_layer = vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        vision_feature_select_strategy = (
            vision_feature_select_strategy if vision_feature_select_strategy is not None else self.config.vision_feature_select_strategy
        )

        use_cache = kwargs.pop("use_cache", False)
        output_hidden_states = kwargs.get("output_hidden_states", False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        is_initial_prefill_cache = past_key_values is None or (
            hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0
        )

        if is_initial_prefill_cache:
            self._pruning_applied = False
            self._next_position_id = None

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )
            image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

        pruning_enabled = scoring_layer_idx is not None
        if not pruning_enabled:
            if pixel_values is not None:
                special_image_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_features
                )
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

            outputs = self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                use_cache=use_cache,
                **kwargs,
            )

            return LlavaModelOutputWithPast(
                last_hidden_state=outputs.last_hidden_state,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                image_hidden_states=image_features if pixel_values is not None else None,
            )

        layers = self.language_model.layers
        norm = self.language_model.norm
        rotary_emb = self.language_model.rotary_emb
        num_layers = len(layers)

        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("Visual token pruning requires input_ids to locate image placeholders.")

            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            original_full_inputs_embeds = inputs_embeds

            if inputs_embeds.shape[0] != 1:
                raise ValueError("v35 visual-token compression currently supports batch_size=1; hybrid training falls back to the batch-safe FlexiExit path.")

            placeholder_mask = input_ids == self.config.image_token_id
            placeholder_indices = placeholder_mask.nonzero(as_tuple=False)[:, 1]
            if len(placeholder_indices) == 0:
                raise ValueError("No image placeholder tokens found in input_ids.")

            image_start_idx = int(placeholder_indices.min().item())
            image_end_idx = int(placeholder_indices.max().item()) + 1
            num_visual_tokens = image_end_idx - image_start_idx

            device = inputs_embeds.device
            if attention_mask is None:
                attention_mask = torch.ones((1, inputs_embeds.size(1)), device=device, dtype=torch.bool)
                attention_mask_is_full = True
            elif attention_mask.dtype != torch.bool or attention_mask.device != device:
                attention_mask = attention_mask.to(device=device, dtype=torch.bool)
                attention_mask_is_full = bool(attention_mask.all().item())
            else:
                attention_mask_is_full = bool(attention_mask.all().item())

            original_seq_length = inputs_embeds.size(1)
            if position_ids is not None:
                original_position_ids = position_ids.to(device=device, dtype=torch.long)
            else:
                original_position_ids = torch.arange(original_seq_length, device=device, dtype=torch.long).unsqueeze(0)
            original_token_positions = torch.arange(original_seq_length, device=device, dtype=torch.long).unsqueeze(0)
            if flexiexit_token_mask is not None:
                flexiexit_token_mask = flexiexit_token_mask.to(device=device, dtype=torch.bool)

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache(config=self.language_model.config)

            # v35 defaults from the provided implementation, now exposed as runtime knobs.
            # Defaults: layer-0 quadtree merge 576->288, layer-6 recover+RSS top-k 288,
            # layer-18 recover+RSS top-k final_target_count (32 by default).
            stage1_merge_layer_idx = min(
                max(int(stage1_merge_layer_idx if stage1_merge_layer_idx is not None else 0), 0),
                num_layers - 1,
            )
            recover_layer_idx = min(
                max(int(recover_layer_idx if recover_layer_idx is not None else 6), stage1_merge_layer_idx),
                num_layers - 1,
            )
            final_prune_layer_idx = min(
                max(
                    int(final_prune_layer_idx if final_prune_layer_idx is not None else 18),
                    recover_layer_idx,
                ),
                num_layers - 1,
            )
            stage1_target_count = min(
                max(int(stage1_target_count if stage1_target_count is not None else 288), 1),
                num_visual_tokens,
            )
            recover_topk_target_count = min(
                max(int(recover_topk_target_count if recover_topk_target_count is not None else 288), 1),
                num_visual_tokens,
            )
            final_target_count = min(
                max(int(visual_token_target_count if visual_token_target_count is not None else 32), 1),
                num_visual_tokens,
            )

            all_hidden_states = () if not output_hidden_states else (inputs_embeds,)
            active_flexiexit_state = flexiexit_state if self._flexiexit_enabled and self._flexiexit_layers else None

            def _build_query_indices(
                seq_len: int,
                img_end: int,
                curr_attention_mask: Optional[torch.Tensor],
            ) -> Optional[torch.LongTensor]:
                if attn_anchor != "query":
                    return None
                valid_mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
                if img_end < seq_len:
                    valid_mask[img_end:] = True
                if curr_attention_mask is not None:
                    valid_mask = valid_mask & curr_attention_mask[0].bool()
                return valid_mask.nonzero(as_tuple=False).squeeze(-1)

            def _set_prefill_flexiexit_token_mask(
                current_positions: torch.LongTensor,
                curr_attention_mask: Optional[torch.Tensor],
            ) -> None:
                if active_flexiexit_state is None:
                    return
                if flexiexit_token_mask is None:
                    active_flexiexit_state.current_token_mask = None
                    return
                valid_positions = current_positions.ge(0) & current_positions.lt(flexiexit_token_mask.shape[1])
                safe_positions = current_positions.clamp(min=0, max=max(flexiexit_token_mask.shape[1] - 1, 0))
                current_token_mask = flexiexit_token_mask.gather(1, safe_positions) & valid_positions
                if curr_attention_mask is not None and curr_attention_mask.shape == current_token_mask.shape:
                    current_token_mask = current_token_mask & curr_attention_mask.to(device=device, dtype=torch.bool)
                active_flexiexit_state.current_token_mask = current_token_mask

            def _run_prefill_layers(
                start_layer_idx: int,
                end_layer_idx: int,
                curr_hidden_states: torch.Tensor,
                curr_attention_mask: torch.Tensor,
                curr_position_ids: torch.LongTensor,
                current_positions: torch.LongTensor,
            ) -> torch.Tensor:
                nonlocal all_hidden_states
                if start_layer_idx >= end_layer_idx:
                    return curr_hidden_states
                _set_prefill_flexiexit_token_mask(current_positions, curr_attention_mask)
                curr_cache_position = self._get_prefill_cache_position(curr_hidden_states.size(1), device)
                curr_causal_mask = self._create_prefill_causal_mask(
                    hidden_states=curr_hidden_states,
                    attention_mask=curr_attention_mask,
                    cache_position=curr_cache_position,
                    position_ids=curr_position_ids,
                    is_full_attention_mask=attention_mask_is_full,
                )
                curr_pos_emb = rotary_emb(curr_hidden_states, curr_position_ids)
                flexiexit_layers = set(self._flexiexit_layers)
                for layer_idx in range(start_layer_idx, end_layer_idx):
                    layer = layers[layer_idx]
                    if active_flexiexit_state is not None and layer_idx in flexiexit_layers:
                        curr_hidden_states = _flexiexit_train_layer_forward(
                            layer,
                            curr_hidden_states,
                            active_flexiexit_state,
                            layer_idx=layer_idx,
                            attention_mask=curr_causal_mask,
                            position_ids=curr_position_ids,
                            past_key_values=past_key_values,
                            use_cache=use_cache,
                            cache_position=curr_cache_position,
                            position_embeddings=curr_pos_emb,
                            kwargs={},
                        )
                    else:
                        curr_hidden_states = layer(
                            curr_hidden_states,
                            attention_mask=curr_causal_mask,
                            position_ids=curr_position_ids,
                            past_key_values=past_key_values,
                            use_cache=use_cache,
                            cache_position=curr_cache_position,
                            position_embeddings=curr_pos_emb,
                            output_attentions=False,
                        )
                        if isinstance(curr_hidden_states, tuple):
                            curr_hidden_states = curr_hidden_states[0]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (curr_hidden_states,)
                return curr_hidden_states

            hidden_states = _run_prefill_layers(
                0,
                stage1_merge_layer_idx,
                inputs_embeds,
                attention_mask,
                original_position_ids,
                original_token_positions,
            )
            child_hidden_at_merge = hidden_states[:, image_start_idx:image_end_idx, :]

            (
                stage1_embeds,
                stage1_input_ids,
                stage1_attention_mask,
                stage1_image_start,
                stage1_image_end,
                stage1_keep_indices,
                stage1_parent_hidden_at_merge,
                stage1_child_to_parent,
                quadtree_token_counts,
                resolved_quadtree_target_count,
            ) = self._merge_visual_tokens_quadtree_exact_target(
                inputs_embeds=hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_start_idx=image_start_idx,
                image_end_idx=image_end_idx,
                target_count=stage1_target_count,
            )
            stage1_visual_tokens = stage1_image_end - stage1_image_start
            stage1_visual_position_ids = original_position_ids[:, image_start_idx + stage1_keep_indices]
            stage1_visual_token_positions = original_token_positions[:, image_start_idx + stage1_keep_indices]
            stage1_position_ids = torch.cat(
                [
                    original_position_ids[:, :image_start_idx],
                    stage1_visual_position_ids,
                    original_position_ids[:, image_end_idx:],
                ],
                dim=1,
            )
            stage1_token_positions = torch.cat(
                [
                    original_token_positions[:, :image_start_idx],
                    stage1_visual_token_positions,
                    original_token_positions[:, image_end_idx:],
                ],
                dim=1,
            )
            logger.info(
                "Layer-%s quadtree merge: %s -> %s visual tokens | target=%s | levels=%s",
                stage1_merge_layer_idx,
                num_visual_tokens,
                stage1_visual_tokens,
                resolved_quadtree_target_count,
                quadtree_token_counts,
            )

            hidden_states = _run_prefill_layers(
                stage1_merge_layer_idx,
                recover_layer_idx,
                stage1_embeds,
                stage1_attention_mask,
                stage1_position_ids,
                stage1_token_positions,
            )

            recovered_hidden_states = self._recover_quadtree_children_from_residual(
                hidden_states=hidden_states,
                image_start_idx=stage1_image_start,
                image_end_idx=stage1_image_end,
                parent_hidden_at_merge=stage1_parent_hidden_at_merge,
                child_to_parent=stage1_child_to_parent,
                child_hidden_at_merge=child_hidden_at_merge,
            )

            recover_cache_position = self._get_prefill_cache_position(recovered_hidden_states.size(1), device)
            recover_query_indices = _build_query_indices(
                recovered_hidden_states.size(1),
                image_end_idx,
                attention_mask,
            )
            _, recover_importance_scores, _ = self._get_visual_token_attention_scores(
                hidden_states=recovered_hidden_states,
                attention_mask=attention_mask,
                position_ids=original_position_ids,
                cache_position=recover_cache_position,
                scoring_layer_idx=recover_layer_idx,
                image_start_idx=image_start_idx,
                image_end_idx=image_end_idx,
                importance_mode="attn",
                ffn_propagation="jvp",
                attn_anchor=attn_anchor,
                pool_type=pool_type,
                query_indices=recover_query_indices,
            )
            (
                top288_embeds,
                top288_input_ids,
                top288_attention_mask,
                top288_image_start,
                top288_image_end,
                top288_keep_indices,
                top288_keep_positions,
            ) = self._topk_visual_tokens_by_scores(
                inputs_embeds=recovered_hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                attention_scores=recover_importance_scores,
                image_start_idx=image_start_idx,
                image_end_idx=image_end_idx,
                target_count=recover_topk_target_count,
                apply_rss=True,
                rss_beta=rss_beta,
                rss_threshold=rss_threshold,
                rss_visual_embeds=original_full_inputs_embeds,
            )
            top288_position_ids = original_position_ids.index_select(1, top288_keep_positions)
            top288_token_positions = original_token_positions.index_select(1, top288_keep_positions)
            full_visual_hidden_at_recover = recovered_hidden_states[:, image_start_idx:image_end_idx, :]
            top288_parent_hidden_at_recover = full_visual_hidden_at_recover.index_select(1, top288_keep_indices)
            top288_child_to_parent = self._build_nearest_selected_visual_parent_map(
                num_children=num_visual_tokens,
                selected_child_indices=top288_keep_indices,
                device=device,
            )
            logger.info(
                "Layer-%s residual recover + RSS top-k truncate: %s -> %s -> %s visual tokens",
                recover_layer_idx,
                stage1_visual_tokens,
                num_visual_tokens,
                top288_image_end - top288_image_start,
            )

            hidden_states = _run_prefill_layers(
                recover_layer_idx,
                final_prune_layer_idx,
                top288_embeds,
                top288_attention_mask,
                top288_position_ids,
                top288_token_positions,
            )

            final_recovered_hidden_states = self._recover_selected_children_from_residual(
                hidden_states=hidden_states,
                image_start_idx=top288_image_start,
                image_end_idx=top288_image_end,
                parent_hidden_at_prune=top288_parent_hidden_at_recover,
                child_to_parent=top288_child_to_parent,
                child_hidden_at_prune=full_visual_hidden_at_recover,
            )
            final_recover_cache_position = self._get_prefill_cache_position(
                final_recovered_hidden_states.size(1), device
            )
            final_recover_query_indices = _build_query_indices(
                final_recovered_hidden_states.size(1),
                image_end_idx,
                attention_mask,
            )
            _, final_recover_importance_scores, _ = self._get_visual_token_attention_scores(
                hidden_states=final_recovered_hidden_states,
                attention_mask=attention_mask,
                position_ids=original_position_ids,
                cache_position=final_recover_cache_position,
                scoring_layer_idx=final_prune_layer_idx,
                image_start_idx=image_start_idx,
                image_end_idx=image_end_idx,
                importance_mode="attn",
                ffn_propagation="jvp",
                attn_anchor=attn_anchor,
                pool_type=pool_type,
                query_indices=final_recover_query_indices,
            )
            scoring_layer_idx = final_prune_layer_idx

            (
                pruned_embeds,
                pruned_input_ids,
                pruned_attention_mask,
                new_img_start,
                new_img_end,
                final_keep_indices,
                final_keep_positions,
            ) = self._topk_visual_tokens_by_scores(
                inputs_embeds=final_recovered_hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                attention_scores=final_recover_importance_scores,
                image_start_idx=image_start_idx,
                image_end_idx=image_end_idx,
                target_count=final_target_count,
                apply_rss=True,
                rss_beta=rss_beta,
                rss_threshold=rss_threshold,
                rss_visual_embeds=original_full_inputs_embeds,
            )

            pruned_seq_len = pruned_embeds.size(1)
            pruned_cache_position = self._get_prefill_cache_position(pruned_seq_len, device)
            kept_position_ids = original_position_ids.index_select(1, final_keep_positions)
            kept_token_positions = original_token_positions.index_select(1, final_keep_positions)
            if self.reassign_position_ids:
                pruned_position_ids = pruned_cache_position.unsqueeze(0)
            else:
                pruned_position_ids = kept_position_ids
            self._pruning_applied = True
            self._next_position_id = int(original_position_ids.max().item()) + 1
            logger.info(
                "Layer-%s residual recover + RSS top-k prune: %s -> %s -> %s visual tokens",
                scoring_layer_idx,
                top288_image_end - top288_image_start,
                num_visual_tokens,
                new_img_end - new_img_start,
            )

            if use_cache and past_key_values is not None:
                stage1_non_visual_cache_positions = torch.cat(
                    [
                        torch.arange(stage1_image_start, device=device, dtype=torch.long),
                        torch.arange(stage1_image_end, stage1_embeds.size(1), device=device, dtype=torch.long),
                    ],
                    dim=0,
                )
                # self._prune_cache_layers_by_positions(
                #     past_key_values,
                #     0,
                #     stage1_merge_layer_idx,
                #     final_keep_positions,
                #     original_seq_length,
                # )
                self._prune_cache_layers_by_positions(
                    past_key_values,
                    stage1_merge_layer_idx,
                    recover_layer_idx,
                    stage1_non_visual_cache_positions,
                    stage1_embeds.size(1),
                )

            hidden_states = _run_prefill_layers(
                scoring_layer_idx,
                num_layers,
                pruned_embeds,
                pruned_attention_mask,
                pruned_position_ids,
                kept_token_positions,
            )

            hidden_states = norm(hidden_states)
            if past_key_values is not None:
                self._set_cache_seen_tokens(past_key_values, pruned_seq_len)

            return LlavaModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                hidden_states=all_hidden_states if output_hidden_states else None,
                attentions=None,
                image_hidden_states=image_features,
                pruned_input_ids=pruned_input_ids,
                pruned_attention_mask=pruned_attention_mask,
                pruned_keep_positions=final_keep_positions.unsqueeze(0),
            )
        else:
            # Decode阶段
            if flexiexit_state is not None:
                outputs = self._language_model_train_flexiexit_path(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    cache_position=cache_position,
                    use_cache=use_cache,
                    flexiexit_state=flexiexit_state,
                    flexiexit_token_mask=flexiexit_token_mask,
                    output_hidden_states=output_hidden_states,
                    kwargs=kwargs,
                )
                if outputs is not None:
                    return LlavaModelOutputWithPast(
                        last_hidden_state=outputs.last_hidden_state,
                        past_key_values=outputs.past_key_values,
                        hidden_states=outputs.hidden_states,
                        attentions=outputs.attentions,
                        image_hidden_states=None,
                    )

            if not use_cache or past_key_values is None:
                outputs = self.language_model(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    cache_position=cache_position,
                    use_cache=use_cache,
                    **kwargs,
                )
                return LlavaModelOutputWithPast(
                    last_hidden_state=outputs.last_hidden_state,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                    image_hidden_states=None,
                )
            hidden_states = inputs_embeds
            _, current_seq_len, _ = hidden_states.shape
            device = hidden_states.device
            if (
                not self.reassign_position_ids
                and self._pruning_applied
                and self._next_position_id is not None
                and hasattr(past_key_values, "get_seq_length")
                and past_key_values.get_seq_length() > 0
            ):
                cache_position = position_ids.squeeze(0)
                position_ids = torch.arange(
                    self._next_position_id,
                    self._next_position_id + current_seq_len,
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(0)
                self._next_position_id += current_seq_len
            else:
                cache_position = position_ids.squeeze(0)

            all_hidden_states = () if not output_hidden_states else (hidden_states,)
            use_layer_specific_cache_mask = isinstance(past_key_values, Cache) and self._pruning_applied
            default_causal_mask = None
            if not use_layer_specific_cache_mask:
                default_causal_mask = _create_llama_causal_mask(
                    config=self.language_model.config,
                    inputs_embeds=hidden_states,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
            pos_emb = self.language_model.rotary_emb(hidden_states, position_ids)

            for layer_idx in range(num_layers):
                layer_cache_position = cache_position
                if use_layer_specific_cache_mask:
                    if current_seq_len == 1:
                        # Single-token decode has no future keys. DynamicCache appends per layer,
                        # so no explicit layer-specific causal mask is needed in the speed path.
                        layer_causal_mask = None
                    else:
                        layer_cache_seq_len = self._get_cache_layer_seq_len(past_key_values, layer_idx)
                        layer_cache_position = torch.arange(
                            layer_cache_seq_len,
                            layer_cache_seq_len + current_seq_len,
                            device=device,
                            dtype=torch.long,
                        )
                        layer_attention_mask = torch.ones(
                            (1, layer_cache_seq_len + current_seq_len), device=device, dtype=torch.long
                        )
                        layer_causal_mask = self._create_layer_specific_causal_mask(
                            hidden_states=hidden_states,
                            attention_mask=layer_attention_mask,
                            cache_position=layer_cache_position,
                            past_key_values=past_key_values,
                            position_ids=position_ids,
                            layer_idx=layer_idx,
                        )
                else:
                    layer_causal_mask = default_causal_mask
                hidden_states = layers[layer_idx](
                    hidden_states,
                    attention_mask=layer_causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=layer_cache_position,
                    position_embeddings=pos_emb,
                    output_attentions=False,
                )
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = norm(hidden_states)

            return LlavaModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                hidden_states=all_hidden_states if output_hidden_states else None,
                attentions=None,
                image_hidden_states=None,
            )

    def _v35_visual_compression_batch_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        scoring_layer_idx: Optional[int] = None,
        stage1_merge_layer_idx: Optional[int] = None,
        recover_layer_idx: Optional[int] = None,
        final_prune_layer_idx: Optional[int] = None,
        stage1_target_count: Optional[int] = None,
        recover_topk_target_count: Optional[int] = None,
        attn_anchor: str = "last",
        pool_type: str = "avg",
        visual_token_target_count: Optional[int] = None,
        rss_beta: float = 1.0,
        rss_threshold: float = 0.5,
        flexiexit: bool = False,
        flexiexit_tau: float = 0.5,
        flexiexit_track_stats: bool = False,
        flexiexit_sum_type: str = "square",
        flexiexit_is_training: bool = False,
        flexiexit_token_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> LlavaModelOutputWithPast:
        if input_ids is None:
            return self._v35_visual_compression_forward(
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
                scoring_layer_idx=scoring_layer_idx,
                stage1_merge_layer_idx=stage1_merge_layer_idx,
                recover_layer_idx=recover_layer_idx,
                final_prune_layer_idx=final_prune_layer_idx,
                stage1_target_count=stage1_target_count,
                recover_topk_target_count=recover_topk_target_count,
                attn_anchor=attn_anchor,
                pool_type=pool_type,
                visual_token_target_count=visual_token_target_count,
                rss_beta=rss_beta,
                rss_threshold=rss_threshold,
                flexiexit_token_mask=flexiexit_token_mask,
                **kwargs,
            )

        batch_size = int(input_ids.shape[0])
        pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is None and hasattr(self.config, "text_config"):
            pad_token_id = getattr(self.config.text_config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        def _slice_batch_value(value: Any, batch_idx: int) -> Any:
            if torch.is_tensor(value) and value.dim() > 0 and int(value.shape[0]) == batch_size:
                return value[batch_idx : batch_idx + 1]
            return value

        def _pad_sequence_tensor(tensor: torch.Tensor, target_len: int, pad_value: float = 0.0) -> torch.Tensor:
            if int(tensor.shape[1]) == target_len:
                return tensor
            pad_shape = (tensor.shape[0], target_len - int(tensor.shape[1]), *tensor.shape[2:])
            pad = tensor.new_full(pad_shape, pad_value)
            return torch.cat([tensor, pad], dim=1)

        sample_outputs: List[LlavaModelOutputWithPast] = []
        depth_maps: List[Optional[torch.Tensor]] = []
        exit_masks: List[Optional[torch.Tensor]] = []
        exit_layers: List[Optional[torch.Tensor]] = []
        token_costs: List[Optional[Dict[str, torch.Tensor]]] = []
        image_hidden_states: List[torch.Tensor] = []

        image_token_id = int(self.config.image_token_id)
        for batch_idx in range(batch_size):
            sample_input_ids = input_ids[batch_idx : batch_idx + 1]
            sample_attention_mask = _slice_batch_value(attention_mask, batch_idx)
            sample_position_ids = _slice_batch_value(position_ids, batch_idx)
            sample_cache_position = _slice_batch_value(cache_position, batch_idx)
            sample_inputs_embeds = _slice_batch_value(inputs_embeds, batch_idx)
            sample_image_sizes = _slice_batch_value(image_sizes, batch_idx)
            sample_flexiexit_token_mask = _slice_batch_value(flexiexit_token_mask, batch_idx)
            sample_kwargs = {key: _slice_batch_value(value, batch_idx) for key, value in kwargs.items()}

            has_image_tokens = bool((sample_input_ids == image_token_id).any().item())
            sample_pixel_values = None
            if pixel_values is not None and has_image_tokens:
                sample_pixel_values = _slice_batch_value(pixel_values, batch_idx)

            sample_state = None
            if flexiexit and bool(flexiexit_is_training):
                sample_state = _FlexiExitState(
                    tau=flexiexit_tau,
                    track_stats=flexiexit_track_stats,
                    is_training=True,
                )

            output = self._v35_visual_compression_forward(
                input_ids=sample_input_ids,
                pixel_values=sample_pixel_values,
                attention_mask=sample_attention_mask,
                position_ids=sample_position_ids,
                past_key_values=None if batch_size > 1 else past_key_values,
                inputs_embeds=sample_inputs_embeds,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                cache_position=sample_cache_position,
                image_sizes=sample_image_sizes,
                scoring_layer_idx=scoring_layer_idx,
                stage1_merge_layer_idx=stage1_merge_layer_idx,
                recover_layer_idx=recover_layer_idx,
                final_prune_layer_idx=final_prune_layer_idx,
                stage1_target_count=stage1_target_count,
                recover_topk_target_count=recover_topk_target_count,
                attn_anchor=attn_anchor,
                pool_type=pool_type,
                visual_token_target_count=visual_token_target_count,
                rss_beta=rss_beta,
                rss_threshold=rss_threshold,
                flexiexit_token_mask=sample_flexiexit_token_mask,
                flexiexit_state=sample_state,
                **sample_kwargs,
            )

            if sample_state is not None and flexiexit_track_stats:
                token_mask = output.pruned_attention_mask
                if token_mask is None:
                    token_mask = sample_attention_mask
                depth_map, exit_mask, exit_layer, _, costs = sample_state.finalize(
                    token_mask=token_mask,
                    sum_type=flexiexit_sum_type,
                )
            else:
                depth_map = exit_mask = exit_layer = costs = None

            if output.image_hidden_states is not None:
                image_hidden_states.append(output.image_hidden_states)

            sample_outputs.append(output)
            depth_maps.append(depth_map)
            exit_masks.append(exit_mask)
            exit_layers.append(exit_layer)
            token_costs.append(costs)

        max_len = max(int(output.last_hidden_state.shape[1]) for output in sample_outputs)
        hidden_states = torch.cat(
            [_pad_sequence_tensor(output.last_hidden_state, max_len, 0.0) for output in sample_outputs],
            dim=0,
        )

        pruned_input_ids = []
        pruned_attention_masks = []
        pruned_keep_positions = []
        for batch_idx, output in enumerate(sample_outputs):
            seq_len = int(output.last_hidden_state.shape[1])
            sample_ids = output.pruned_input_ids
            if sample_ids is None:
                sample_ids = input_ids[batch_idx : batch_idx + 1, :seq_len]
            sample_mask = output.pruned_attention_mask
            if sample_mask is None:
                if attention_mask is None:
                    sample_mask = torch.ones((1, seq_len), device=hidden_states.device, dtype=torch.long)
                else:
                    sample_mask = attention_mask[batch_idx : batch_idx + 1, :seq_len]
            sample_keep = output.pruned_keep_positions
            if sample_keep is None:
                sample_keep = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0)

            pruned_input_ids.append(_pad_sequence_tensor(sample_ids, max_len, int(pad_token_id)))
            pruned_attention_masks.append(_pad_sequence_tensor(sample_mask, max_len, 0))
            pruned_keep_positions.append(_pad_sequence_tensor(sample_keep, max_len, -1))

        def _pad_optional_tensors(
            tensors: List[Optional[torch.Tensor]],
            pad_value: float,
        ) -> Optional[torch.Tensor]:
            if not any(tensor is not None for tensor in tensors):
                return None
            padded = []
            for batch_idx, tensor in enumerate(tensors):
                if tensor is None:
                    tensor = hidden_states.new_full(
                        (1, int(sample_outputs[batch_idx].last_hidden_state.shape[1])),
                        pad_value,
                    )
                    if isinstance(pad_value, bool):
                        tensor = tensor.to(dtype=torch.bool)
                    elif isinstance(pad_value, int):
                        tensor = tensor.to(dtype=torch.long)
                padded.append(_pad_sequence_tensor(tensor, max_len, pad_value))
            return torch.cat(padded, dim=0)

        combined_costs = None
        if any(cost is not None for cost in token_costs):
            keys = set()
            for cost in token_costs:
                if cost is not None:
                    keys.update(cost.keys())
            combined_costs = {}
            for key in keys:
                values = []
                for batch_idx, cost in enumerate(token_costs):
                    if cost is not None and key in cost:
                        value = cost[key]
                    else:
                        value = hidden_states.new_zeros((1, int(sample_outputs[batch_idx].last_hidden_state.shape[1])))
                    values.append(_pad_sequence_tensor(value, max_len, 0.0))
                combined_costs[key] = torch.cat(values, dim=0)

        return LlavaModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=sample_outputs[0].past_key_values if batch_size == 1 else None,
            hidden_states=sample_outputs[0].hidden_states if batch_size == 1 else None,
            attentions=None,
            image_hidden_states=torch.cat(image_hidden_states, dim=0) if image_hidden_states else None,
            pruned_input_ids=torch.cat(pruned_input_ids, dim=0),
            pruned_attention_mask=torch.cat(pruned_attention_masks, dim=0),
            pruned_keep_positions=torch.cat(pruned_keep_positions, dim=0),
            flexiexit_depth_map=_pad_optional_tensors(depth_maps, 0.0),
            flexiexit_exit_mask=_pad_optional_tensors(exit_masks, False),
            flexiexit_exit_layer=_pad_optional_tensors(exit_layers, -1),
            flexi_token_costs=combined_costs,
        )


    def reset_flexiexit_stats(self) -> None:
        """Clear cached decode-time FlexiExit statistics."""
        self._flexiexit_decode_records = []
        self._flexiexit_decode_step = 0
        self._last_flexiexit_decode_exit_layer = None
        self._last_flexiexit_decode_active_depth = None

    def get_flexiexit_stats(self) -> List[Dict[str, Any]]:
        """Return a copy of cached decode-time FlexiExit statistics.

        Each record corresponds to one cached single-token decode forward.
        ``exit_layer`` is the decoder layer index where the token first exited;
        it is -1 when the token continued through every configured FlexiExit layer.
        """
        return list(self._flexiexit_decode_records)

    def _append_flexiexit_decode_record(
        self,
        *,
        decode_input_ids: Optional[torch.LongTensor],
        position_ids: Optional[torch.LongTensor],
        exit_layer: int,
        active_depth: int,
        num_flexiexit_layers: int,
    ) -> None:
        token_id = None
        if decode_input_ids is not None and torch.is_tensor(decode_input_ids) and decode_input_ids.numel() > 0:
            token_id = int(decode_input_ids.reshape(-1)[-1].detach().cpu().item())

        position_id = None
        if position_ids is not None and torch.is_tensor(position_ids) and position_ids.numel() > 0:
            position_id = int(position_ids.reshape(-1)[-1].detach().cpu().item())

        self._flexiexit_decode_records.append(
            {
                "step": int(self._flexiexit_decode_step),
                "input_id": token_id,
                "position_id": position_id,
                "exit_layer": int(exit_layer),
                "active_depth": int(active_depth),
                "num_flexiexit_layers": int(num_flexiexit_layers),
                "exited": bool(int(exit_layer) >= 0),
            }
        )
        self._flexiexit_decode_step += 1


    # -------------------------------------------------------------------------
    # FlexiExit public API
    # -------------------------------------------------------------------------

    def enable_flexiexit(
        self,
        *,
        layers: Optional[Any] = None,
        tau: Optional[float] = None,
        router_ratio: Optional[int] = None,
        adapter_ratio: Optional[int] = None,
        enable_learnable_tau: Optional[bool] = None,
    ) -> None:
        """Attach FlexiExit modules to selected LLaMA-like decoder layers.

        Minimal config surface:
        - flexiexit=True
        - flexiexit_layers=[[16, 31]] or [16, 17, ...]
        - flexiexit_tau=0.5
        """
        if not _is_llama_like_decoder(self.language_model):
            raise ValueError("FlexiExit currently supports LLaMA-like text backbones with a `.layers` stack.")

        if tau is not None:
            self._flexiexit_tau = float(tau)
        if enable_learnable_tau is not None:
            self._enable_learnable_tau = bool(enable_learnable_tau)
        router_ratio = int(router_ratio or self._flexiexit_router_ratio)
        adapter_ratio = int(adapter_ratio or self._flexiexit_adapter_ratio)

        decoder_layers = self.language_model.layers
        total_layers = len(decoder_layers)
        target_layers = _parse_flexiexit_layers(layers, total_layers)

        hidden_size = int(getattr(self.language_model.config, "hidden_size"))
        intermediate_size = int(getattr(self.language_model.config, "intermediate_size", hidden_size * 4))
        router_hidden = max(1, hidden_size // router_ratio)
        adapter_hidden = max(1, intermediate_size // adapter_ratio)

        for layer_idx in target_layers:
            layer = decoder_layers[layer_idx]
            layer.flexiexit_layer_idx = int(layer_idx)

            if self._enable_learnable_tau:
                if not isinstance(getattr(layer, "flexiexit_tau", None), nn.Module):
                    layer.flexiexit_tau = flexiexit_tau(self._flexiexit_tau)
            else:
                if isinstance(getattr(layer, "flexiexit_tau", None), nn.Module):
                    delattr(layer, "flexiexit_tau")
                layer.flexiexit_tau = float(self._flexiexit_tau)

            if not hasattr(layer, "flexiexit_router"):
                layer.flexiexit_router = _FlexiExitRouter(hidden_size, router_hidden)
            if not hasattr(layer, "flexiexit_adapter"):
                layer.flexiexit_adapter = _FlexiExitAdapter(hidden_size, adapter_hidden)

        self._flexiexit_enabled = True
        self._flexiexit_layers = target_layers

    def disable_flexiexit(self) -> None:
        self._flexiexit_enabled = False
        self._flexiexit_layers = []

    def _set_flexiexit_tau(self, tau: float) -> None:
        self._flexiexit_tau = float(tau)
        if not _is_llama_like_decoder(self.language_model):
            return
        target_layers = self._flexiexit_layers or range(len(self.language_model.layers))
        for layer_idx in target_layers:
            layer = self.language_model.layers[int(layer_idx)]
            tau_module = getattr(layer, "flexiexit_tau", None)
            if isinstance(tau_module, nn.Module):
                tau_value = min(max(float(tau), 1e-6), 1.0 - 1e-6)
                tau_logit = math.log(tau_value / (1.0 - tau_value))
                with torch.no_grad():
                    tau_module.logit.fill_(tau_logit)
            elif hasattr(layer, "flexiexit_tau"):
                layer.flexiexit_tau = float(tau)

    def _language_model_decode_fast_path(
        self,
        *,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Cache],
        inputs_embeds: torch.FloatTensor,
        cache_position: Optional[torch.LongTensor],
        use_cache: bool,
        flexiexit: bool = False,
        flexiexit_tau: float = 0.5,
        record_flexiexit_stats: bool = False,
        decode_input_ids: Optional[torch.LongTensor] = None,
    ) -> Optional[BaseModelOutputWithPast]:
        """Single-token cached decode path with FlexiExit and v35 position bookkeeping."""
        if (
            self.training
            or not use_cache
            or past_key_values is None
            or inputs_embeds.shape[1] != 1
            or inputs_embeds.shape[0] != 1
            or not _is_llama_like_decoder(self.language_model)
        ):
            return None

        current_seq_len = inputs_embeds.shape[1]
        device = inputs_embeds.device
        if position_ids is None:
            if cache_position is not None:
                position_ids = cache_position.view(1, -1)
            elif attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1)[:, -1:] - 1
            else:
                return None
        if cache_position is None:
            cache_position = position_ids.squeeze(0)

        if (
            not self.reassign_position_ids
            and self._pruning_applied
            and self._next_position_id is not None
            and hasattr(past_key_values, "get_seq_length")
            and past_key_values.get_seq_length() > 0
        ):
            cache_position = position_ids.squeeze(0)
            position_ids = torch.arange(
                int(self._next_position_id),
                int(self._next_position_id) + current_seq_len,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)
            self._next_position_id += current_seq_len

        hidden_states = inputs_embeds
        pos_emb = self.language_model.rotary_emb(hidden_states, position_ids)
        flexiexit_layers = set(self._flexiexit_layers) if flexiexit and self._flexiexit_enabled else set()
        decoder_layers = self.language_model.layers
        flexiexit_suffix_start = len(decoder_layers)
        while flexiexit_suffix_start > 0 and (flexiexit_suffix_start - 1) in flexiexit_layers:
            flexiexit_suffix_start -= 1
        exited = False
        decode_exit_layer = -1
        active_depth = 0
        for layer_idx, layer in enumerate(decoder_layers):
            if layer_idx in flexiexit_layers:
                was_exited = bool(exited)
                hidden_states, exited, token_continue = _flexiexit_decode_layer_forward(
                    layer,
                    hidden_states,
                    exited,
                    tau=flexiexit_tau,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=pos_emb,
                )
                if not was_exited:
                    if bool(token_continue):
                        active_depth += 1
                    elif decode_exit_layer < 0:
                        decode_exit_layer = int(layer_idx)
                        if layer_idx >= flexiexit_suffix_start:
                            hidden_states = _flexiexit_decode_exited_layers_forward(
                                decoder_layers,
                                hidden_states,
                                start_idx=layer_idx + 1,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                cache_position=cache_position,
                                position_embeddings=pos_emb,
                            )
                            break
                continue

            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=pos_emb,
                output_attentions=False,
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        hidden_states = self.language_model.norm(hidden_states)

        if record_flexiexit_stats and not self.training:
            self._last_flexiexit_decode_exit_layer = torch.tensor(
                [[decode_exit_layer]], device=hidden_states.device, dtype=torch.long
            )
            self._last_flexiexit_decode_active_depth = torch.tensor(
                [[active_depth]], device=hidden_states.device, dtype=torch.float32
            )
            self._append_flexiexit_decode_record(
                decode_input_ids=decode_input_ids,
                position_ids=position_ids,
                exit_layer=decode_exit_layer,
                active_depth=active_depth,
                num_flexiexit_layers=len(flexiexit_layers),
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def _language_model_prefill_flexiexit_path(
        self,
        *,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Cache],
        inputs_embeds: torch.FloatTensor,
        cache_position: Optional[torch.LongTensor],
        use_cache: bool,
        flexiexit_tau: float,
        flexiexit_track_stats: bool,
        flexiexit_sum_type: str,
        kwargs: dict[str, Any],
    ) -> Optional[tuple[BaseModelOutputWithPast, Dict[str, Optional[torch.Tensor]]]]:
        if (
            self.training
            or inputs_embeds.shape[1] <= 1
            or not _is_llama_like_decoder(self.language_model)
            or _llama_create_causal_mask is None
            or not self._flexiexit_enabled
        ):
            return None

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.language_model.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _create_llama_causal_mask(
            config=self.language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.language_model.rotary_emb(hidden_states, position_ids=position_ids)
        flexiexit_layers = set(self._flexiexit_layers)
        exit_mask = torch.zeros(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        active_depth = torch.zeros(hidden_states.shape[:2], dtype=torch.float32, device=hidden_states.device)
        exit_layer = torch.full(hidden_states.shape[:2], -1, dtype=torch.long, device=hidden_states.device)
        all_gate_sum = None
        layer_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"use_cache", "output_attentions", "output_hidden_states"}
        }

        num_hidden_layers = int(getattr(self.language_model.config, "num_hidden_layers", len(self.language_model.layers)))
        for layer_idx, layer in enumerate(self.language_model.layers[:num_hidden_layers]):
            if layer_idx in flexiexit_layers:
                prev_exit_mask = exit_mask
                hidden_states, exit_mask, continue_mask, gate = _flexiexit_prefill_layer_forward(
                    layer,
                    hidden_states,
                    exit_mask,
                    tau=flexiexit_tau,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    kwargs=layer_kwargs,
                )
                if flexiexit_track_stats:
                    active_depth = active_depth + continue_mask.detach().to(torch.float32)
                    first_exit = exit_mask & (~prev_exit_mask)
                    exit_layer = torch.where(first_exit, torch.full_like(exit_layer, int(layer_idx)), exit_layer)
                    cost_gate = gate * (~prev_exit_mask).to(gate.dtype)
                    all_gate_sum = cost_gate if all_gate_sum is None else all_gate_sum + cost_gate
                continue

            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_attentions=False,
                **layer_kwargs,
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        hidden_states = self.language_model.norm(hidden_states)
        diagnostics: Dict[str, Optional[torch.Tensor]] = {
            "flexiexit_depth_map": None,
            "flexiexit_exit_mask": None,
            "flexiexit_exit_layer": None,
            "flexi_token_costs": None,
        }
        if flexiexit_track_stats:
            token_mask = attention_mask
            if token_mask is not None:
                mask_bool = token_mask.to(dtype=torch.bool, device=token_mask.device)
                if active_depth.shape == mask_bool.shape:
                    active_depth = active_depth * mask_bool.to(active_depth.dtype)
                if exit_mask.shape == mask_bool.shape:
                    exit_mask = exit_mask & mask_bool
                if exit_layer.shape == mask_bool.shape:
                    exit_layer = torch.where(mask_bool, exit_layer, torch.full_like(exit_layer, -1))

            token_costs = None
            if all_gate_sum is not None:
                if token_mask is not None:
                    cost_mask = token_mask.to(device=all_gate_sum.device, dtype=all_gate_sum.dtype)
                else:
                    cost_mask = torch.ones_like(all_gate_sum)
                cost = all_gate_sum if flexiexit_sum_type == "l1" else all_gate_sum.square()
                token_costs = {"all": cost * cost_mask}

            diagnostics = {
                "flexiexit_depth_map": active_depth,
                "flexiexit_exit_mask": exit_mask,
                "flexiexit_exit_layer": exit_layer,
                "flexi_token_costs": token_costs,
            }

        return (
            BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                hidden_states=None,
                attentions=None,
            ),
            diagnostics,
        )

    def _language_model_train_flexiexit_path(
        self,
        *,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Cache],
        inputs_embeds: torch.FloatTensor,
        cache_position: Optional[torch.LongTensor],
        use_cache: bool,
        flexiexit_state: _FlexiExitState,
        flexiexit_token_mask: Optional[torch.Tensor],
        output_hidden_states: bool,
        kwargs: dict[str, Any],
    ) -> Optional[BaseModelOutputWithPast]:
        if (
            not _is_llama_like_decoder(self.language_model)
            or _llama_create_causal_mask is None
            or not self._flexiexit_enabled
        ):
            return None

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.language_model.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _create_llama_causal_mask(
            config=self.language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        position_embeddings = self.language_model.rotary_emb(hidden_states, position_ids=position_ids)
        flexiexit_layers = set(self._flexiexit_layers)
        layer_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"use_cache", "output_attentions", "output_hidden_states"}
        }
        flexiexit_state.current_token_mask = (
            flexiexit_token_mask.to(device=hidden_states.device, dtype=torch.bool)
            if flexiexit_token_mask is not None
            else None
        )

        num_hidden_layers = int(getattr(self.language_model.config, "num_hidden_layers", len(self.language_model.layers)))
        for layer_idx, layer in enumerate(self.language_model.layers[:num_hidden_layers]):
            if layer_idx in flexiexit_layers:
                hidden_states = _flexiexit_train_layer_forward(
                    layer,
                    hidden_states,
                    flexiexit_state,
                    layer_idx=layer_idx,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    kwargs=layer_kwargs,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    output_attentions=False,
                    **layer_kwargs,
                )
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.language_model.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    # -------------------------------------------------------------------------
    # Vision and multimodal embedding utilities
    # -------------------------------------------------------------------------

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        **kwargs,
    ):
        vision_feature_layer = vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)

        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self.multi_modal_projector(selected_image_feature)

        if "image_sizes" in kwargs:
            split_sizes = [
                (height // self.vision_tower.patch_size) * (width // self.vision_tower.patch_size)
                for height, width in kwargs["image_sizes"]
            ]
            image_features = torch.split(image_features.squeeze(0), split_sizes)
        else:
            image_features = list(image_features)
        return image_features

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ):
        if input_ids is None:
            image_token_embed = self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = (inputs_embeds == image_token_embed).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        scoring_layer_idx: Optional[int] = None,
        stage1_merge_layer_idx: Optional[int] = None,
        recover_layer_idx: Optional[int] = None,
        final_prune_layer_idx: Optional[int] = None,
        stage1_target_count: Optional[int] = None,
        recover_topk_target_count: Optional[int] = None,
        attn_anchor: str = "last",
        pool_type: str = "avg",
        visual_token_target_count: Optional[int] = 64,
        rss_beta: float = 1.0,
        rss_threshold: float = 0.5,
        record_flexiexit_stats: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
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
            self._pruning_applied = False
            self._next_position_id = None

        use_cache = bool(kwargs.get("use_cache", False))
        output_attentions = bool(kwargs.get("output_attentions", False))
        output_hidden_states = bool(kwargs.get("output_hidden_states", False))
        if record_flexiexit_stats is None:
            record_flexiexit_stats = bool(getattr(self.config, "record_flexiexit_stats", self._record_flexiexit_stats))
        else:
            record_flexiexit_stats = bool(record_flexiexit_stats)
        record_flexiexit_stats = bool(record_flexiexit_stats and not self.training)
        self._record_flexiexit_stats = record_flexiexit_stats
        if record_flexiexit_stats and past_key_values is None:
            self.reset_flexiexit_stats()

        flexiexit_arg = kwargs.pop("flexiexit", None)
        if flexiexit_arg is None:
            flexiexit = bool(getattr(self.config, "flexiexit", False) or getattr(self.config.text_config, "flexiexit", False))
        else:
            flexiexit = bool(flexiexit_arg)
        flexiexit_tau = float(kwargs.pop("flexiexit_tau", getattr(self.config, "flexiexit_tau", self._flexiexit_tau)))
        flexiexit_need_loss_map = bool(kwargs.pop("flexiexit_need_loss_map", False))
        flexiexit_track_stats = bool(kwargs.pop("flexiexit_track_stats", flexiexit_need_loss_map))
        flexiexit_sum_type = str(kwargs.pop("flexiexit_sum_type", getattr(self.config, "flexiexit_sum_type", "square")))
        flexiexit_is_training = kwargs.pop("is_training", self.training)
        flexiexit_token_mask = kwargs.pop("flexiexit_token_mask", None)
        kwargs.pop("visual_token_keep_ratio", None)
        kwargs.pop("visual_token_min_keep", None)
        kwargs.pop("add_type", None)
        enable_prefill_flexiexit = bool(
            kwargs.pop("enable_prefill_flexiexit", getattr(self.config, "enable_prefill_flexiexit", False))
        )

        enable_v35_prefill_compression = bool(
            kwargs.pop(
                "enable_v35_prefill_compression",
                getattr(self.config, "enable_v35_prefill_compression", True),
            )
        )
        is_v35_prefill = (
            enable_v35_prefill_compression
            and pixel_values is not None
            and input_ids is not None
            and is_initial_prefill_cache
            and scoring_layer_idx is not None
        )
        if is_v35_prefill:
            v35_can_run = input_ids.shape[0] == 1 or self.training
            if v35_can_run:
                run_training_prefill_flexiexit = flexiexit and bool(flexiexit_is_training)
                if run_training_prefill_flexiexit:
                    if not self._flexiexit_enabled:
                        self.enable_flexiexit(
                            layers=getattr(self.config, "flexiexit_layers", None),
                        )
                    if float(flexiexit_tau) != float(self._flexiexit_tau):
                        self._set_flexiexit_tau(flexiexit_tau)
                return self._v35_visual_compression_batch_forward(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=None,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    cache_position=cache_position,
                    image_sizes=image_sizes,
                    scoring_layer_idx=scoring_layer_idx,
                    stage1_merge_layer_idx=stage1_merge_layer_idx,
                    recover_layer_idx=recover_layer_idx,
                    final_prune_layer_idx=final_prune_layer_idx,
                    stage1_target_count=stage1_target_count,
                    recover_topk_target_count=recover_topk_target_count,
                    attn_anchor=attn_anchor,
                    pool_type=pool_type,
                    visual_token_target_count=visual_token_target_count,
                    rss_beta=rss_beta,
                    rss_threshold=rss_threshold,
                    flexiexit=run_training_prefill_flexiexit,
                    flexiexit_tau=flexiexit_tau,
                    flexiexit_track_stats=flexiexit_track_stats,
                    flexiexit_sum_type=flexiexit_sum_type,
                    flexiexit_is_training=bool(flexiexit_is_training),
                    flexiexit_token_mask=flexiexit_token_mask,
                    **kwargs,
                )
            if not self._warned_v35_batch_fallback:
                logger.warning(
                    "v35 visual-token compression currently runs only for batch_size=1 prefill; "
                    "falling back to the batch-safe non-compressed prefill path for this batch."
                )
                self._warned_v35_batch_fallback = True

        image_features = None
        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("Need input_ids to detect image tokens when pixel_values is provided.")

            image_token_id = self.config.image_token_id
            has_image = (input_ids == image_token_id).any(dim=1)

            if has_image.any():
                pixel_values_sub = pixel_values[has_image]
                image_sizes_sub = image_sizes[has_image] if image_sizes is not None else None

                image_features = self.get_image_features(
                    pixel_values=pixel_values_sub,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    image_sizes=image_sizes_sub,
                )
                image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

                sub_input_ids = input_ids[has_image]
                sub_inputs_embeds = inputs_embeds[has_image]
                special_image_mask = self.get_placeholder_mask(
                    sub_input_ids,
                    inputs_embeds=sub_inputs_embeds,
                    image_features=image_features,
                )
                sub_inputs_embeds = sub_inputs_embeds.masked_scatter(special_image_mask, image_features)
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[has_image] = sub_inputs_embeds

        is_decode_step = (not self.training) and inputs_embeds.shape[1] == 1 and past_key_values is not None and use_cache
        is_prefill_step = (not self.training) and inputs_embeds.shape[1] > 1 and is_initial_prefill_cache
        run_prefill_flexiexit = flexiexit and enable_prefill_flexiexit and is_prefill_step
        run_flexiexit = flexiexit and (is_decode_step or bool(flexiexit_is_training) or run_prefill_flexiexit)

        flexiexit_state = None
        flexiexit_depth_map = None
        flexiexit_exit_mask = None
        flexiexit_exit_layer = None
        flexiexit_gates = None
        flexi_token_costs = None
        if run_flexiexit:
            if not self._flexiexit_enabled:
                self.enable_flexiexit(
                    layers=getattr(self.config, "flexiexit_layers", None),
                )
            if float(flexiexit_tau) != float(self._flexiexit_tau):
                self._set_flexiexit_tau(flexiexit_tau)
            if (flexiexit_is_training or not is_decode_step) and not run_prefill_flexiexit:
                flexiexit_state = _FlexiExitState(
                    tau=flexiexit_tau,
                    track_stats=flexiexit_track_stats,
                    is_training=bool(flexiexit_is_training),
                )

        outputs = None
        if is_decode_step and not output_attentions and not output_hidden_states:
            outputs = self._language_model_decode_fast_path(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                use_cache=use_cache,
                flexiexit=run_flexiexit,
                flexiexit_tau=flexiexit_tau,
                record_flexiexit_stats=record_flexiexit_stats,
                decode_input_ids=input_ids,
            )
        elif run_prefill_flexiexit and not output_attentions and not output_hidden_states:
            prefill_outputs = self._language_model_prefill_flexiexit_path(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                use_cache=use_cache,
                flexiexit_tau=flexiexit_tau,
                flexiexit_track_stats=flexiexit_track_stats,
                flexiexit_sum_type=flexiexit_sum_type,
                kwargs=kwargs,
            )
            if prefill_outputs is not None:
                outputs, prefill_diagnostics = prefill_outputs
                flexiexit_depth_map = prefill_diagnostics["flexiexit_depth_map"]
                flexiexit_exit_mask = prefill_diagnostics["flexiexit_exit_mask"]
                flexiexit_exit_layer = prefill_diagnostics["flexiexit_exit_layer"]
                flexi_token_costs = prefill_diagnostics["flexi_token_costs"]
        elif flexiexit_state is not None and not output_attentions:
            outputs = self._language_model_train_flexiexit_path(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                use_cache=use_cache,
                flexiexit_state=flexiexit_state,
                flexiexit_token_mask=flexiexit_token_mask,
                output_hidden_states=output_hidden_states,
                kwargs=kwargs,
            )
        else:
            outputs = self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                **kwargs,
            )

        if outputs is None:
            outputs = self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                **kwargs,
            )

        if flexiexit_state is not None:
            if flexiexit_state.track_stats:
                (
                    flexiexit_depth_map,
                    flexiexit_exit_mask,
                    flexiexit_exit_layer,
                flexiexit_gates,
                flexi_token_costs,
            ) = flexiexit_state.finalize(token_mask=attention_mask, sum_type=flexiexit_sum_type)

        return LlavaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
            pruned_input_ids=getattr(outputs, "pruned_input_ids", None),
            pruned_attention_mask=getattr(outputs, "pruned_attention_mask", None),
            flexiexit_depth_map=flexiexit_depth_map,
            flexiexit_exit_mask=flexiexit_exit_mask,
            flexiexit_exit_layer=flexiexit_exit_layer,
            flexiexit_gates=flexiexit_gates,
            flexi_token_costs=flexi_token_costs,
            flexiexit_decode_exit_layer=self._last_flexiexit_decode_exit_layer,
            flexiexit_decode_active_depth=self._last_flexiexit_decode_active_depth,
        )


@auto_docstring(
    custom_intro="""
    The LLAVA model which consists of a vision backbone and a language model.
    """
)
class LlavaForConditionalGeneration(LlavaPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlavaConfig):
        super().__init__(config)
        self.model = LlavaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        **kwargs,
    ):
        return self.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            **kwargs,
        )

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

    def reset_flexiexit_stats(self) -> None:
        """Clear cached non-training decode-time FlexiExit statistics."""
        self.model.reset_flexiexit_stats()

    def get_flexiexit_stats(self) -> List[Dict[str, Any]]:
        """Return cached non-training decode-time FlexiExit statistics.

        Each record corresponds to one cached single-token decode forward.
        ``exit_layer`` is the decoder layer index where the token first exited;
        it is -1 when the token continued through all configured FlexiExit layers.
        """
        return self.model.get_flexiexit_stats()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: Optional[torch.Tensor] = None,
        scoring_layer_idx: Optional[int] = 18,
        stage1_merge_layer_idx: Optional[int] = None,
        recover_layer_idx: Optional[int] = None,
        final_prune_layer_idx: Optional[int] = None,
        stage1_target_count: Optional[int] = None,
        recover_topk_target_count: Optional[int] = None,
        attn_anchor: str = "last",
        pool_type: str = "avg",
        visual_token_target_count: Optional[int] = 32,
        rss_beta: float = 1.0,
        rss_threshold: float = 0.5,
        record_flexiexit_stats: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LlavaCausalLMOutputWithPast]:
        vision_feature_layer = vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        flexiexit_need_loss_map = bool(kwargs.pop("flexiexit_need_loss_map", labels is not None))
        kwargs.pop("visual_token_keep_ratio", None)
        kwargs.pop("visual_token_min_keep", None)
        kwargs.pop("add_type", None)

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
            scoring_layer_idx=scoring_layer_idx,
            stage1_merge_layer_idx=stage1_merge_layer_idx,
            recover_layer_idx=recover_layer_idx,
            final_prune_layer_idx=final_prune_layer_idx,
            stage1_target_count=stage1_target_count,
            recover_topk_target_count=recover_topk_target_count,
            attn_anchor=attn_anchor,
            pool_type=pool_type,
            visual_token_target_count=visual_token_target_count,
            rss_beta=rss_beta,
            rss_threshold=rss_threshold,
            record_flexiexit_stats=record_flexiexit_stats,
            flexiexit_need_loss_map=flexiexit_need_loss_map,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        full_loss_map = None
        if labels is not None:
            loss_labels = labels
            pruned_keep_positions = getattr(outputs, "pruned_keep_positions", None)
            if pruned_keep_positions is not None:
                keep_positions = pruned_keep_positions.to(device=labels.device, dtype=torch.long)
                valid_positions = keep_positions.ge(0) & keep_positions.lt(labels.shape[1])
                safe_positions = keep_positions.clamp(min=0, max=max(labels.shape[1] - 1, 0))
                gathered_labels = labels.gather(1, safe_positions)
                loss_labels = torch.where(
                    valid_positions,
                    gathered_labels,
                    torch.full_like(gathered_labels, -100),
                )

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = loss_labels[..., 1:].contiguous()
            flat_shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_shift_labels = shift_labels.view(-1)
            loss_fct_none = nn.CrossEntropyLoss(reduction="none")
            flat_ce = loss_fct_none(flat_shift_logits, flat_shift_labels)

            valid_flat = flat_shift_labels.ne(-100)
            if valid_flat.any():
                loss = flat_ce[valid_flat].mean()
            else:
                loss = flat_ce.sum() * 0.0

            flexi_token_costs = getattr(outputs, "flexi_token_costs", None)
            if flexiexit_need_loss_map and flexi_token_costs is not None:
                bsz, seq_len_minus_1 = shift_labels.shape
                ce_loss_per_token = flat_ce.view(bsz, seq_len_minus_1)
                ce_loss_detached = ce_loss_per_token.detach()

                flexi_ref_tensor = flexi_token_costs
                if isinstance(flexi_token_costs, dict):
                    flexi_ref_tensor = flexi_token_costs.get("all", next(iter(flexi_token_costs.values())))

                full_loss_map = torch.zeros_like(flexi_ref_tensor)
                seq_minus_1_map = full_loss_map[:, :-1]
                if seq_minus_1_map.shape == ce_loss_detached.shape:
                    valid_mask = shift_labels.ne(-100)
                    seq_minus_1_map.copy_(torch.where(valid_mask, ce_loss_detached, torch.zeros_like(ce_loss_detached)))

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            flexiexit_depth_map=getattr(outputs, "flexiexit_depth_map", None),
            flexiexit_exit_mask=getattr(outputs, "flexiexit_exit_mask", None),
            flexiexit_exit_layer=getattr(outputs, "flexiexit_exit_layer", None),
            flexiexit_gates=getattr(outputs, "flexiexit_gates", None),
            flexi_token_costs=getattr(outputs, "flexi_token_costs", None),
            flexiexit_decode_exit_layer=getattr(outputs, "flexiexit_decode_exit_layer", None),
            flexiexit_decode_active_depth=getattr(outputs, "flexiexit_decode_active_depth", None),
            full_loss_map=full_loss_map,
            pruned_input_ids=getattr(outputs, "pruned_input_ids", None),
            pruned_attention_mask=getattr(outputs, "pruned_attention_mask", None),
            pruned_keep_positions=getattr(outputs, "pruned_keep_positions", None),
        )

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, **kwargs)
        pruned_attention_mask = getattr(outputs, "pruned_attention_mask", None)
        if pruned_attention_mask is not None:
            model_kwargs["attention_mask"] = torch.cat(
                [pruned_attention_mask, pruned_attention_mask.new_ones((pruned_attention_mask.shape[0], 1))], dim=-1
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

        is_initial_generation_step = past_key_values is None or (
            cache_position is not None and cache_position.numel() != 1
        )
        if is_initial_generation_step:
            model_inputs["pixel_values"] = pixel_values

        if "record_flexiexit_stats" in kwargs:
            model_inputs["record_flexiexit_stats"] = kwargs["record_flexiexit_stats"]

        return model_inputs


__all__ = ["LlavaForConditionalGeneration", "LlavaPreTrainedModel", "LlavaModel"]
