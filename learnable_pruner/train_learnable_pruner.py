#!/usr/bin/env python3
# coding: utf-8
"""One-stage Dynamic-KL training for a learnable LLaVA visual-token pruner.

This trainer implements a single integrated objective:

    L = w_kl * KL(full || soft-pruned)
      + w_rank * TopK precision structured hinge
      + w_ce * CE(labels, soft-pruned)

The soft-pruned branch follows the CoViPAL-style idea of adding log-retention
biases to visual-token keys in the causal attention mask, but uses the frozen
full model's output distribution as the main preservation target.  The retention
gate is a hard-ST TopK gate: the forward path is exact hard TopK with zero
retention for non-kept visual tokens, while the backward path uses a strictly
budgeted soft TopK relaxation.  At inference only the predictor logits are
needed; visual tokens are hard-pruned by TopK.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import os
import random
import sys
from contextlib import contextmanager
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
from llava.mm_utils import tokenizer_image_token  # noqa: E402
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM  # noqa: E402


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


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
    def __init__(self, data_path: str, image_folder: str, image_processor, image_aspect_ratio: str = "pad"):
        self.data_path = data_path
        self.image_folder = image_folder
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
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
            if self.data_args.image_aspect_ratio == "pad":
                image = _expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            text_sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            text_sources = copy.deepcopy([e["conversations"] for e in sources])
            crop_size = self.data_args.image_processor.crop_size
            h = crop_size["height"] if isinstance(crop_size, dict) else crop_size
            w = crop_size["width"] if isinstance(crop_size, dict) else crop_size
            image = torch.zeros(3, h, w)

        data_dict = preprocess(text_sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        data_dict["image"] = image
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
        batch["sample_ids"] = torch.tensor(sample_ids, dtype=torch.long)
        return batch


def _first_visual_span(input_ids: torch.Tensor, image_token_id: int) -> Optional[Tuple[int, int]]:
    positions = (input_ids == image_token_id).nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return None
    start = int(positions[0].item())
    non_image_offsets = input_ids[start:].ne(image_token_id).nonzero(as_tuple=False).flatten()
    end = (start + int(non_image_offsets[0].item())) if non_image_offsets.numel() > 0 else int(input_ids.numel())
    return start, end


def _iqr(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return values.new_zeros(())
    q75 = torch.quantile(values.float(), 0.75)
    q25 = torch.quantile(values.float(), 0.25)
    return q75 - q25


def _spatial_entropy(values: torch.Tensor) -> torch.Tensor:
    return _spatial_entropy_batch(values.reshape(1, -1))[0]


def _spatial_entropy_batch(values: torch.Tensor) -> torch.Tensor:
    num_tokens = int(values.shape[-1])
    side = int(num_tokens**0.5)
    if num_tokens <= 1 or side * side != num_tokens:
        return values.new_zeros((values.shape[0],), dtype=torch.float32)

    try:
        import numpy as np
        from scipy.ndimage import label

        scores = values.detach().float().cpu().numpy().reshape(values.shape[0], side, side)
        entropies = []
        for score in scores:
            binary_map = score > score.mean()
            labeled_array, num_features = label(binary_map, structure=np.ones((3, 3), dtype=int))
            if num_features <= 0:
                entropies.append(0.0)
                continue
            sizes = np.array([np.sum(labeled_array == idx) for idx in range(1, num_features + 1)])
            probs = sizes / sizes.sum()
            entropies.append(float(-np.sum(probs * np.log(probs + 1e-10))))
        return values.new_tensor(entropies, dtype=torch.float32)
    except Exception:
        return values.new_zeros((values.shape[0],), dtype=torch.float32)


def _log_minmax_normalize(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.full_like(values.float(), 0.5)

    x = values.float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.log(x.clamp_min(eps))

    lo = x.min()
    hi = x.max()
    if torch.isclose(hi, lo):
        return torch.full_like(x, 0.5)

    return (x - lo) / (hi - lo)


def _cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _math_sdpa_checkpoint_contexts():
    backend = torch.nn.attention.SDPBackend.MATH
    return torch.nn.attention.sdpa_kernel(backend), torch.nn.attention.sdpa_kernel(backend)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, num_key_value_heads * n_rep, seq_len, head_dim)


def _make_4d_causal_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    key_log_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build an additive causal mask, optionally with per-key log-retention bias.

    key_log_bias has shape [B, S].  A negative value at key position j reduces
    every query's attention probability to key j before softmax.  Disallowed
    causal/padding positions are still set to dtype minimum.
    """

    bsz, seq_len = attention_mask.shape
    device = attention_mask.device
    query_pos = torch.arange(seq_len, device=device).view(seq_len, 1)
    key_pos = torch.arange(seq_len, device=device).view(1, seq_len)
    causal = key_pos <= query_pos
    key_valid = attention_mask[:, None, None, :].bool()
    allowed = causal.view(1, 1, seq_len, seq_len) & key_valid
    min_dtype = torch.finfo(dtype).min

    mask = torch.zeros((bsz, 1, seq_len, seq_len), device=device, dtype=dtype)
    if key_log_bias is not None:
        mask = mask + key_log_bias.to(device=device, dtype=dtype)[:, None, None, :]
    return mask.masked_fill(~allowed, min_dtype)


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
    def forward(ctx, logits: torch.Tensor, keep_k: int, temperature: float, max_iter: int):  # type: ignore[override]
        if logits.dim() != 1:
            raise ValueError("_BudgetedSigmoidTopK expects a 1D tensor of valid logits.")

        n = int(logits.numel())
        keep_k = int(keep_k)
        temperature = max(float(temperature), 1e-6)
        max_iter = int(max_iter)

        x = logits.float()
        if n == 0 or keep_k <= 0:
            out = torch.zeros_like(x)
            ctx.save_for_backward(out)
            ctx.temperature = temperature
            ctx.has_free_boundary = False
            return out.to(dtype=logits.dtype)
        if keep_k >= n:
            out = torch.ones_like(x)
            ctx.save_for_backward(out)
            ctx.temperature = temperature
            ctx.has_free_boundary = False
            return out.to(dtype=logits.dtype)

        # Wide but finite brackets.  These make sigmoid mass almost n / 0 at the
        # endpoints, while avoiding +/-inf in lower-precision training.
        lo = x.min() - 50.0 * temperature - 1.0
        hi = x.max() + 50.0 * temperature + 1.0
        target = float(keep_k)
        for _ in range(max(8, max_iter)):
            mid = (lo + hi) * 0.5
            mass = torch.sigmoid((x - mid) / temperature).sum()
            # mass decreases as lambda increases.
            move_lo = mass > target
            lo = torch.where(move_lo, mid, lo)
            hi = torch.where(move_lo, hi, mid)

        lam = (lo + hi) * 0.5
        out = torch.sigmoid((x - lam) / temperature)
        ctx.save_for_backward(out)
        ctx.temperature = temperature
        ctx.has_free_boundary = True
        return out.to(dtype=logits.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (soft,) = ctx.saved_tensors
        if not bool(ctx.has_free_boundary):
            return torch.zeros_like(grad_output), None, None, None

        temperature = max(float(ctx.temperature), 1e-6)
        grad = grad_output.float()
        soft = soft.float()
        slope = soft * (1.0 - soft)
        denom = slope.sum().clamp_min(1e-12)

        # Implicit differentiation of sum_i sigmoid((x_i-lambda)/tau)=k:
        # dL/dx_j = a_j/tau * (g_j - sum_i g_i a_i / sum_i a_i),
        # where a_i = m_i(1-m_i).
        weighted_mean_grad = (grad * slope).sum() / denom
        grad_logits = (slope / temperature) * (grad - weighted_mean_grad)
        return grad_logits.to(dtype=grad_output.dtype), None, None, None


class LearnablePruneWrapper(nn.Module):
    def __init__(
        self,
        llava_model: nn.Module,
        predictor: LearnablePrunePredictor,
        keep_k: int = 64,
        candidate_layers: str = "16-24",
        rank_loss_weight: float = 1.0,
        kl_loss_weight: float = 1.0,
        ce_loss_weight: float = 1.0,
        enable_scale: bool = True,
        enable_rss: bool = False,
        teacher_top_iqr_layers: int = 3,
        topk_hinge_margin: float = 1.0,
        kd_temperature: float = 1.0,
        budgeted_soft_topk_iters: int = 32,
        predictor_layer: int = -1,
        teacher_target_log_dir: Optional[str] = None,
        teacher_target_log_to_console: bool = False,
    ):
        super().__init__()
        self.llava = llava_model
        self.predictor = predictor
        self.keep_k = int(keep_k)
        self.candidate_layers = self._parse_layers(candidate_layers)
        self.rank_loss_weight = float(rank_loss_weight)
        self.kl_loss_weight = float(kl_loss_weight)
        self.ce_loss_weight = float(ce_loss_weight)
        self.enable_scale = bool(enable_scale)
        self.enable_rss = bool(enable_rss)
        self.teacher_top_iqr_layers = int(teacher_top_iqr_layers)
        self.topk_hinge_margin = float(topk_hinge_margin)
        self.kd_temperature = float(kd_temperature)
        self.budgeted_soft_topk_iters = int(budgeted_soft_topk_iters)
        self._training_progress = 0.0
        self._last_soft_budget_error: Optional[torch.Tensor] = None
        self.attention_gate_mode = "hard_st"
        self.attention_min_keep_prob = 0.0
        self.predictor_layer = int(predictor_layer)

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

        for param in self.llava.parameters():
            param.requires_grad = False
        self.llava.eval()

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
                "context_fn": _math_sdpa_checkpoint_contexts,
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

    def _current_kl_loss_weight(self) -> float:
        return self.kl_loss_weight * self._loss_weight_scale(0.2, 2.0)

    def _current_ce_loss_weight(self) -> float:
        return self.ce_loss_weight * self._loss_weight_scale(0.2, 2.0)

    @contextmanager
    def _temporary_attention_backend(self, backend: str):
        configs = []
        for obj in (
            getattr(self.llava, "model", None),
            getattr(self.llava, "config", None),
            getattr(getattr(self.llava, "config", None), "text_config", None),
        ):
            config = getattr(obj, "config", obj)
            if config is not None and hasattr(config, "_attn_implementation"):
                configs.append(config)
        seen = set()
        originals = []
        for config in configs:
            if id(config) in seen:
                continue
            seen.add(id(config))
            originals.append((config, config._attn_implementation))
            config._attn_implementation = backend
        try:
            yield
        finally:
            for config, original in originals:
                config._attn_implementation = original

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

    @staticmethod
    def _parse_layers(spec: str) -> List[int]:
        layers: List[int] = []
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                layers.extend(range(int(start), int(end) + 1))
            else:
                layers.append(int(part))
        return layers

    def _usable_teacher_layers(self, num_layers: int) -> List[int]:
        layers = [idx for idx in self.candidate_layers if 0 <= idx < num_layers]
        return layers or [max(0, num_layers - 1)]

    def _full_forward_with_teacher_states(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[Any, Dict[int, torch.Tensor], List[int]]:
        language_model = self.llava.model
        teacher_layers = self._usable_teacher_layers(len(language_model.layers))
        teacher_states: Dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(layer_idx: int):
            def hook(_module, args, kwargs):
                hidden_states = kwargs.get("hidden_states") if kwargs else None
                if hidden_states is None and args:
                    hidden_states = args[0]
                if hidden_states is not None:
                    teacher_states[layer_idx] = hidden_states.detach()

            return hook

        for layer_idx in teacher_layers:
            hooks.append(language_model.layers[layer_idx].register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True))
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

        missing = [idx for idx in teacher_layers if idx not in teacher_states]
        if missing:
            raise RuntimeError(f"Failed to capture teacher hidden states for layers: {missing}")
        return full_outputs, teacher_states, teacher_layers

    def _build_multimodal_inputs(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        image_features = self.llava.encode_images(images)
        if torch.is_tensor(image_features):
            image_features = [feat for feat in image_features]
        else:
            image_features = list(image_features)

        compact_input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        compact_labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds: List[torch.Tensor] = []
        new_input_ids: List[torch.Tensor] = []
        new_labels: List[torch.Tensor] = []
        cur_image_idx = 0
        embed_tokens = self.llava.get_model().embed_tokens
        image_token_value = int(IMAGE_TOKEN_INDEX)

        for batch_idx, cur_input_ids in enumerate(compact_input_ids):
            num_images = int((cur_input_ids == IMAGE_TOKEN_INDEX).sum().item())
            cur_labels = compact_labels[batch_idx]
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds = embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds, cur_image_features[0:0].to(cur_input_embeds.dtype)], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_input_ids.append(cur_input_ids)
                new_labels.append(cur_labels)
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            text_embeds = embed_tokens(torch.cat(cur_input_ids_noim)) if sum(split_sizes) > 0 else None
            text_embeds_split = torch.split(text_embeds, split_sizes, dim=0) if text_embeds is not None else [
                self.llava.get_input_embeddings().weight.new_zeros((0, self.llava.config.hidden_size))
                for _ in split_sizes
            ]

            cur_new_embeds = []
            cur_new_ids = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_embeds.append(text_embeds_split[i])
                cur_new_ids.append(cur_input_ids_noim[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx].to(device=text_embeds_split[i].device, dtype=text_embeds_split[i].dtype)
                    cur_image_idx += 1
                    visual_len = cur_image_features.shape[0]
                    cur_new_embeds.append(cur_image_features)
                    cur_new_ids.append(torch.full((visual_len,), image_token_value, device=cur_input_ids.device, dtype=cur_input_ids.dtype))
                    cur_new_labels.append(torch.full((visual_len,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            new_input_embeds.append(torch.cat([x.to(self.llava.device) for x in cur_new_embeds], dim=0))
            new_input_ids.append(torch.cat(cur_new_ids, dim=0))
            new_labels.append(torch.cat(cur_new_labels, dim=0))

        tokenizer_model_max_length = getattr(self.llava.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_input_ids = [x[:tokenizer_model_max_length] for x in new_input_ids]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        pad_token_id = int(getattr(self.llava.config, "pad_token_id", 0) or 0)
        padded_embeds = []
        padded_ids = torch.full((batch_size, max_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        padded_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
        padded_attention = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        padded_position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_embeds, cur_ids, cur_labels) in enumerate(zip(new_input_embeds, new_input_ids, new_labels)):
            cur_len = cur_embeds.shape[0]
            if getattr(self.llava.config, "tokenizer_padding_side", "right") == "left":
                padded_embeds.append(
                    torch.cat(
                        [
                            torch.zeros((max_len - cur_len, cur_embeds.shape[1]), dtype=cur_embeds.dtype, device=cur_embeds.device),
                            cur_embeds,
                        ],
                        dim=0,
                    )
                )
                if cur_len > 0:
                    padded_ids[i, -cur_len:] = cur_ids
                    padded_labels[i, -cur_len:] = cur_labels
                    padded_attention[i, -cur_len:] = True
                    padded_position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
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
                    padded_labels[i, :cur_len] = cur_labels
                    padded_attention[i, :cur_len] = True
                    padded_position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        return (
            torch.stack(padded_embeds, dim=0),
            padded_ids,
            padded_attention,
            padded_position_ids,
            padded_labels,
        )

    def _get_visual_token_attention_scores(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        scoring_layer_idx: int,
        image_start_idx: int,
        image_end_idx: int,
        query_indices: torch.Tensor,
    ) -> torch.Tensor:
        language_model = self.llava.model
        scoring_layer = language_model.layers[scoring_layer_idx]
        self_attn = scoring_layer.self_attn
        bsz, seq_len, hidden_size = hidden_states.shape
        if bsz != 1:
            raise ValueError("Importance scoring expects a single sample.")

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
        cos, sin = language_model.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = _repeat_kv(key_states, num_heads // num_kv_heads)
        value_states = _repeat_kv(value_states, num_heads // num_kv_heads)

        q_idx = query_indices.to(dtype=torch.long, device=hidden_states.device)
        key_positions = torch.arange(seq_len, device=hidden_states.device).view(1, 1, 1, seq_len)
        key_valid = attention_mask[:, None, None, :].bool()
        v_visual = value_states[0, :, image_start_idx:image_end_idx, :]
        query_chunk_size = 8
        score_chunks: List[torch.Tensor] = []
        for q_chunk in q_idx.split(query_chunk_size):
            q_states = query_states.index_select(2, q_chunk)
            attn_scores = torch.matmul(q_states, key_states.transpose(2, 3)) * scaling
            causal = key_positions <= q_chunk.view(1, 1, -1, 1)
            attn_scores = attn_scores.float().masked_fill(~(causal & key_valid), torch.finfo(torch.float32).min)
            attn_probs = F.softmax(attn_scores, dim=-1).to(dtype=q_states.dtype)
            z_heads = torch.matmul(attn_probs, value_states)[0].permute(1, 0, 2).contiguous()

            alpha_visual = attn_probs[0, :, :, image_start_idx:image_end_idx].permute(1, 0, 2).contiguous()
            beta = alpha_visual / (1.0 - alpha_visual).clamp(min=1e-6)
            delta_z = beta.unsqueeze(-1) * (z_heads.unsqueeze(2) - v_visual.unsqueeze(0))
            delta_z_cat = delta_z.permute(0, 2, 1, 3).contiguous().view(-1, num_heads * head_dim)
            delta_y = self_attn.o_proj(delta_z_cat).view(q_chunk.numel(), -1, hidden_size)
            score_chunks.append(delta_y.norm(dim=-1))
        if not score_chunks:
            return hidden_states.new_zeros((0, image_end_idx - image_start_idx))
        return torch.cat(score_chunks, dim=0)

    def _teacher_targets(
        self,
        teacher_hidden_states: Dict[int, torch.Tensor],
        teacher_layers: List[int],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[Tuple[int, int]]]]:
        bsz = input_ids.shape[0]
        device = input_ids.device
        image_token_id = int(IMAGE_TOKEN_INDEX)
        spans = [_first_visual_span(input_ids[b], image_token_id) for b in range(bsz)]
        max_visual = max((end - start for span in spans if span is not None for start, end in [span]), default=0)
        scores = torch.zeros((bsz, max_visual), device=device, dtype=torch.float32)
        valid_masks = torch.zeros((bsz, max_visual), device=device, dtype=torch.bool)
        if max_visual == 0:
            return scores, valid_masks, spans

        for b, span in enumerate(spans):
            if span is None:
                continue
            start, end = span
            seq_len = int(input_ids.shape[1])
            valid_queries = torch.arange(end, seq_len, device=device)
            valid_queries = valid_queries[attention_mask[b, end:].bool()]
            if valid_queries.numel() == 0:
                valid_positions = attention_mask[b].bool().nonzero(as_tuple=False).flatten()
                if valid_positions.numel() == 0:
                    continue
                valid_queries = valid_positions[-1:]

            sample_scores: List[torch.Tensor] = []
            sample_iqrs: List[torch.Tensor] = []
            sample_attention_mask = attention_mask[b : b + 1]
            sample_position_ids = position_ids[b : b + 1]
            for layer_idx in teacher_layers:
                layer_hidden = teacher_hidden_states[layer_idx][b : b + 1]
                importance_scores = self._get_visual_token_attention_scores(
                    hidden_states=layer_hidden,
                    attention_mask=sample_attention_mask,
                    position_ids=sample_position_ids,
                    scoring_layer_idx=layer_idx,
                    image_start_idx=start,
                    image_end_idx=end,
                    query_indices=valid_queries,
                ).mean(dim=0)
                importance_scores = importance_scores.float()
                sample_scores.append(importance_scores)
                sample_iqrs.append(_iqr(importance_scores))

            if not sample_scores:
                continue
            top_layers = min(max(1, self.teacher_top_iqr_layers), len(sample_scores))
            score_stack = torch.stack(sample_scores, dim=0)
            spatial_entropy_values = _spatial_entropy_batch(score_stack)
            iqr_values = torch.stack(sample_iqrs)
            spatial_entropy_quality = _log_minmax_normalize(spatial_entropy_values)
            iqr_quality = _log_minmax_normalize(iqr_values)
            layer_quality = 0.6 * spatial_entropy_quality + 0.4 * iqr_quality
            top_layer_indices = torch.topk(layer_quality, k=top_layers).indices
            sample_id = None
            if sample_ids is not None:
                sample_id = int(sample_ids[b].detach().cpu().item())
            self._log_teacher_target_selection(
                sample_id=sample_id,
                local_batch_index=b,
                visual_start=start,
                visual_end=end,
                query_count=int(valid_queries.numel()),
                teacher_layers=teacher_layers,
                top_layer_indices=top_layer_indices,
                spatial_entropy_values=spatial_entropy_values,
                iqr_values=iqr_values,
                layer_quality=layer_quality,
            )
            pooled_scores = score_stack.index_select(0, top_layer_indices.to(device=score_stack.device)).mean(dim=0)
            if self.enable_rss and inputs_embeds is not None and pooled_scores.numel() > 0:
                visual_embeds = inputs_embeds[b : b + 1, start:end, :].detach().float()
                pooled_scores = self._apply_rss_algorithm(visual_embeds, pooled_scores.float())
            scores[b, : pooled_scores.numel()] = pooled_scores
            valid_masks[b, : pooled_scores.numel()] = True
        return scores, valid_masks, spans

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
        teacher_layers: List[int],
        top_layer_indices: torch.Tensor,
        spatial_entropy_values: torch.Tensor,
        iqr_values: torch.Tensor,
        layer_quality: torch.Tensor,
    ) -> None:
        if self._teacher_target_logger is None and not self.teacher_target_log_to_console:
            return

        selected_indices = [int(x) for x in top_layer_indices.detach().cpu().tolist()]
        selected_layers = [int(teacher_layers[i]) for i in selected_indices]
        record = {
            "global_step": int(self._teacher_target_global_step),
            "log_index": int(self._teacher_target_log_count),
            "rank": int(self._teacher_target_rank),
            "sample_id": sample_id,
            "local_batch_index": int(local_batch_index),
            "visual_span": [int(visual_start), int(visual_end)],
            "visual_token_count": int(visual_end - visual_start),
            "query_count": int(query_count),
            "candidate_layers": [int(x) for x in teacher_layers],
            "selected_candidate_indices": selected_indices,
            "selected_layers": selected_layers,
            "pooled_layer_count": int(len(selected_layers)),
            "spatial_entropy": [float(x) for x in spatial_entropy_values.detach().float().cpu().tolist()],
            "iqr": [float(x) for x in iqr_values.detach().float().cpu().tolist()],
            "layer_quality": [float(x) for x in layer_quality.detach().float().cpu().tolist()],
        }
        message = json.dumps(record, ensure_ascii=False)
        if self._teacher_target_logger is not None:
            self._teacher_target_logger.info(message)
        if self.teacher_target_log_to_console and self._teacher_target_rank == 0:
            print(f"[teacher-target] {message}", flush=True)
        self._teacher_target_log_count += 1

    def _budgeted_soft_topk_gate(
        self,
        valid_logits: torch.Tensor,
        keep_k: int,
    ) -> torch.Tensor:
        return _BudgetedSigmoidTopK.apply(
            valid_logits,
            int(keep_k),
            1.0,
            int(self.budgeted_soft_topk_iters),
        )

    def _retention_probabilities(
        self,
        predictor_logits: torch.Tensor,
        teacher_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return hard-ST visual retention gates and log-biases."""
        probs = predictor_logits.new_ones(predictor_logits.shape)
        log_biases = torch.zeros(
            predictor_logits.shape,
            device=predictor_logits.device,
            dtype=torch.float32,
        )
        budget_errors: List[torch.Tensor] = []
        min_log_bias = torch.finfo(torch.float32).min

        for b in range(predictor_logits.shape[0]):
            valid = teacher_valid[b].bool()
            valid_count = int(valid.sum().item())
            if valid_count == 0:
                continue
            keep_k = min(self.keep_k, valid_count)
            valid_logits = predictor_logits[b, valid]
            if keep_k >= valid_count:
                probs[b, valid] = 1.0
                log_biases[b, valid] = 0.0
                budget_errors.append(valid_logits.new_zeros(()))
                continue

            soft = self._budgeted_soft_topk_gate(valid_logits, keep_k=keep_k)
            budget_errors.append((soft.detach().float().sum() - float(keep_k)).abs())

            hard = torch.zeros_like(valid_logits)
            topk = torch.topk(valid_logits, k=keep_k).indices
            hard[topk] = 1.0
            gate = hard + soft - soft.detach()
            probs[b, valid] = gate

            soft_log_bias = torch.log(soft.float().clamp_min(1e-6))
            hard_log_bias = torch.where(
                hard.bool(),
                torch.zeros_like(soft_log_bias),
                torch.full_like(soft_log_bias, min_log_bias),
            )
            log_biases[b, valid] = hard_log_bias + soft_log_bias - soft_log_bias.detach()

        if budget_errors:
            self._last_soft_budget_error = torch.stack([x.float() for x in budget_errors]).mean()
        else:
            self._last_soft_budget_error = predictor_logits.new_zeros(())
        return probs, log_biases

    def _selector_loss(
        self,
        predictor_logits: torch.Tensor,
        teacher_scores: torch.Tensor,
        teacher_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = predictor_logits.sum() * 0.0
        rank_loss = zero
        spearman_total = zero
        topk_overlap_total = zero
        boundary_accuracy_total = zero
        valid_rows = 0
        boundary_rows = 0

        for b in range(predictor_logits.shape[0]):
            row_valid = teacher_valid[b].bool()
            valid_count = int(row_valid.sum().item())
            if valid_count <= 1:
                continue
            valid_rows += 1
            teacher_vals_raw = teacher_scores[b, row_valid].float()
            pred_vals = predictor_logits[b, row_valid].float()

            teacher_ranks = torch.argsort(torch.argsort(teacher_vals_raw.float())).float()
            pred_ranks = torch.argsort(torch.argsort(pred_vals.float())).float()
            teacher_ranks = teacher_ranks - teacher_ranks.mean()
            pred_ranks = pred_ranks - pred_ranks.mean()
            spearman = (teacher_ranks * pred_ranks).sum() / (
                teacher_ranks.square().sum().sqrt() * pred_ranks.square().sum().sqrt()
            ).clamp(min=1e-6)
            spearman_total = spearman_total + spearman

            keep_k = min(self.keep_k, valid_count)
            if keep_k <= 0:
                continue

            teacher_order = torch.argsort(teacher_vals_raw, descending=True)
            pred_order = torch.argsort(pred_vals, descending=True)
            teacher_top = torch.zeros(valid_count, device=pred_vals.device, dtype=torch.bool)
            pred_top = torch.zeros(valid_count, device=pred_vals.device, dtype=torch.bool)
            teacher_top[teacher_order[:keep_k]] = True
            pred_top[pred_order[:keep_k]] = True
            topk_overlap_total = topk_overlap_total + (teacher_top & pred_top).float().sum() / float(keep_k)

            if keep_k >= valid_count:
                continue

            pos_idx = teacher_order[:keep_k]
            neg_mask = torch.ones(valid_count, device=pred_vals.device, dtype=torch.bool)
            neg_mask[pos_idx] = False

            margin_augmented_scores = pred_vals + self.topk_hinge_margin * neg_mask.to(dtype=pred_vals.dtype)
            violating_topk = torch.topk(margin_augmented_scores, k=keep_k, largest=True).values
            row_loss = (violating_topk.sum() - pred_vals[pos_idx].sum()) / float(keep_k)
            rank_loss = rank_loss + row_loss.clamp_min(0.0)

            neg_idx = neg_mask.nonzero(as_tuple=False).flatten()
            if neg_idx.numel() > 0:
                hard_neg_k = min(keep_k, int(neg_idx.numel()))
                hard_neg_logits = torch.topk(pred_vals[neg_idx], k=hard_neg_k, largest=True).values
                pos_logits = pred_vals[pos_idx]
                boundary_accuracy_total = boundary_accuracy_total + (
                    pos_logits[:, None] > hard_neg_logits[None, :]
                ).float().mean()
                boundary_rows += 1

        denom = max(1, valid_rows)
        rank_loss = rank_loss / denom
        selector_spearman = spearman_total / denom
        selector_topk_overlap = topk_overlap_total / denom
        selector_boundary_accuracy = boundary_accuracy_total / max(1, boundary_rows)
        total = self._current_rank_loss_weight() * rank_loss
        return total, rank_loss, selector_spearman, selector_topk_overlap, selector_boundary_accuracy

    def _build_visual_key_log_bias(
        self,
        input_ids: torch.Tensor,
        spans: List[Optional[Tuple[int, int]]],
        retention_log_bias: torch.Tensor,
    ) -> torch.Tensor:
        key_log_bias = retention_log_bias.new_zeros(input_ids.shape, dtype=torch.float32)
        for b, span in enumerate(spans):
            if span is None:
                continue
            start, end = span
            visual_len = end - start
            key_log_bias[b, start:end] = retention_log_bias[b, :visual_len].float()
        return key_log_bias

    def _kl_loss(
        self,
        full_hidden_states: torch.Tensor,
        pruned_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        teacher_hidden = full_hidden_states.detach()
        student_hidden = pruned_hidden_states
        valid = attention_mask.bool()
        if labels is not None and labels.shape == attention_mask.shape:
            teacher_hidden = teacher_hidden[:, :-1, :]
            student_hidden = student_hidden[:, :-1, :]
            valid = valid[:, :-1] & labels[:, 1:].ne(IGNORE_INDEX)
        if not bool(valid.any().item()):
            return student_hidden.sum() * 0.0

        with torch.no_grad():
            teacher_logits = self.llava.lm_head(teacher_hidden[valid]).detach()
        pruned_logits = self.llava.lm_head(student_hidden[valid])
        temperature = max(self.kd_temperature, 1e-6)
        student_logp = F.log_softmax(pruned_logits.float() / temperature, dim=-1)
        teacher_p = F.softmax(teacher_logits.float() / temperature, dim=-1)
        return F.kl_div(student_logp, teacher_p, reduction="batchmean") * (temperature * temperature)

    def _ce_loss(self, pruned_hidden_states: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels is None or labels.shape != attention_mask.shape:
            return pruned_hidden_states.sum() * 0.0
        shifted_hidden = pruned_hidden_states[:, :-1, :]
        shifted_labels = labels[:, 1:]
        valid = attention_mask[:, :-1].bool() & shifted_labels.ne(IGNORE_INDEX)
        if not bool(valid.any().item()):
            return pruned_hidden_states.sum() * 0.0
        logits = self.llava.lm_head(shifted_hidden[valid])
        return F.cross_entropy(logits.float(), shifted_labels[valid].to(dtype=torch.long))

    def _build_predictor_inputs(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_token_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Only prompt tokens are visible to the predictor.  Answer tokens are not used.
        prompt_text_mask = attention_mask.bool() & labels.eq(IGNORE_INDEX) & ~visual_token_mask
        predictor_token_mask = visual_token_mask | prompt_text_mask
        lengths = predictor_token_mask.long().sum(dim=1)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        bsz, _, hidden_size = inputs_embeds.shape
        predictor_inputs = inputs_embeds.new_zeros((bsz, max_len, hidden_size))
        predictor_attention_mask = torch.zeros((bsz, max_len), device=inputs_embeds.device, dtype=torch.bool)
        predictor_visual_mask = torch.zeros((bsz, max_len), device=inputs_embeds.device, dtype=torch.bool)
        predictor_text_mask = torch.zeros((bsz, max_len), device=inputs_embeds.device, dtype=torch.bool)
        if max_len > 0:
            batch_idx, seq_idx = predictor_token_mask.nonzero(as_tuple=True)
            slot_idx = predictor_token_mask.long().cumsum(dim=1)[batch_idx, seq_idx] - 1
            predictor_inputs[batch_idx, slot_idx] = inputs_embeds[batch_idx, seq_idx]
            predictor_attention_mask[batch_idx, slot_idx] = True
            predictor_visual_mask[batch_idx, slot_idx] = visual_token_mask[batch_idx, seq_idx]
            predictor_text_mask[batch_idx, slot_idx] = prompt_text_mask[batch_idx, seq_idx]
        return predictor_inputs, predictor_visual_mask, predictor_text_mask, predictor_attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
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
        ) = self._build_multimodal_inputs(
            input_ids=input_ids,
            images=image_tensor,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )

        with torch.no_grad():
            full_outputs, teacher_hidden_states, teacher_layers = self._full_forward_with_teacher_states(
                inputs_embeds=inputs_embeds,
                attention_mask=expanded_attention_mask,
                position_ids=original_position_ids,
            )
            teacher_scores, teacher_valid, spans = self._teacher_targets(
                teacher_hidden_states,
                teacher_layers,
                expanded_input_ids,
                expanded_attention_mask,
                original_position_ids,
                inputs_embeds=inputs_embeds,
                sample_ids=sample_ids,
            )
            del teacher_hidden_states

        max_visual = teacher_valid.shape[1]
        if max_visual == 0:
            zero = sum(param.sum() for param in self.predictor.parameters()) * 0.0
            return {
                "loss": zero,
                "total_loss": zero.detach(),
                "logits": inputs_embeds.new_zeros((expanded_input_ids.shape[0], 0)),
                "selector_loss": zero.detach(),
                "rank_loss": zero.detach(),
                "selector_spearman": zero.detach(),
                "selector_topk_overlap": zero.detach(),
                "selector_boundary_accuracy": zero.detach(),
                "kl_loss": zero.detach(),
                "ce_loss": zero.detach(),
                "retention_mean": zero.detach(),
                "retention_top_mean": zero.detach(),
                "retention_bottom_mean": zero.detach(),
                "soft_budget_error": zero.detach(),
                "weighted_rank_loss": zero.detach(),
                "weighted_kl_loss": zero.detach(),
                "weighted_ce_loss": zero.detach(),
                "current_rank_loss_weight": zero.detach() + self._current_rank_loss_weight(),
                "current_kl_loss_weight": zero.detach() + self._current_kl_loss_weight(),
                "current_ce_loss_weight": zero.detach() + self._current_ce_loss_weight(),
            }

        visual_token_mask = torch.zeros(expanded_input_ids.shape, device=expanded_input_ids.device, dtype=torch.bool)
        for b, span in enumerate(spans):
            if span is not None:
                start, end = span
                visual_token_mask[b, start:end] = True
        (
            predictor_inputs,
            predictor_visual_mask,
            predictor_text_mask,
            predictor_attention_mask,
        ) = self._build_predictor_inputs(
            inputs_embeds=inputs_embeds.detach(),
            attention_mask=expanded_attention_mask,
            visual_token_mask=visual_token_mask,
            labels=expanded_labels,
        )
        predictor_inputs = predictor_inputs.to(dtype=next(self.predictor.parameters()).dtype)
        predictor_logits = self.predictor(
            predictor_inputs,
            visual_token_mask=predictor_visual_mask,
            text_token_mask=predictor_text_mask,
            attention_mask=predictor_attention_mask,
        )

        retention_probs, retention_log_bias = self._retention_probabilities(predictor_logits, teacher_valid)
        (
            selector_loss,
            rank_loss,
            selector_spearman,
            selector_topk_overlap,
            selector_boundary_accuracy,
        ) = self._selector_loss(
            predictor_logits=predictor_logits,
            teacher_scores=teacher_scores,
            teacher_valid=teacher_valid,
        )

        key_log_bias = self._build_visual_key_log_bias(expanded_input_ids, spans, retention_log_bias)
        soft_prune_attention_mask = _make_4d_causal_mask(
            attention_mask=expanded_attention_mask,
            dtype=inputs_embeds.dtype,
            key_log_bias=key_log_bias,
        )
        # Math SDPA keeps gradients through additive masks and avoids the dense eager attention path.
        with (
            self._temporary_attention_backend("sdpa"),
            self._temporary_decoder_checkpoint_training(),
            torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH),
        ):
            pruned_outputs = self.llava.model(
                inputs_embeds=inputs_embeds,
                attention_mask=soft_prune_attention_mask,
                position_ids=original_position_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        kl_loss = self._kl_loss(
            full_outputs.last_hidden_state,
            pruned_outputs.last_hidden_state,
            expanded_attention_mask,
            labels=expanded_labels,
        )
        ce_loss = self._ce_loss(
            pruned_outputs.last_hidden_state,
            expanded_attention_mask,
            expanded_labels,
        )

        weighted_rank_loss = self._current_rank_loss_weight() * rank_loss
        weighted_kl_loss = self._current_kl_loss_weight() * kl_loss
        weighted_ce_loss = self._current_ce_loss_weight() * ce_loss
        loss = weighted_rank_loss + weighted_kl_loss + weighted_ce_loss

        if not bool(torch.isfinite(loss).item()):
            details = {
                "selector_loss": float(selector_loss.detach().float().item()),
                "rank_loss": float(rank_loss.detach().float().item()),
                "kl_loss": float(kl_loss.detach().float().item()),
                "ce_loss": float(ce_loss.detach().float().item()),
            }
            raise FloatingPointError(f"Non-finite learnable-prune loss: {details}")

        valid_probs = retention_probs[teacher_valid].detach().float()
        retention_mean = valid_probs.mean() if valid_probs.numel() > 0 else loss.detach() * 0.0
        top_means: List[torch.Tensor] = []
        bottom_means: List[torch.Tensor] = []
        for b in range(predictor_logits.shape[0]):
            row_valid = teacher_valid[b].bool()
            valid_count = int(row_valid.sum().item())
            if valid_count <= 1:
                continue
            keep_k = min(self.keep_k, valid_count)
            pred_vals = predictor_logits[b, row_valid].detach().float()
            row_probs = retention_probs[b, row_valid].detach().float()
            order = torch.argsort(pred_vals, descending=True)
            top_means.append(row_probs[order[:keep_k]].mean())
            if keep_k < valid_count:
                bottom_means.append(row_probs[order[keep_k:]].mean())
        retention_top_mean = torch.stack(top_means).mean() if top_means else loss.detach() * 0.0
        retention_bottom_mean = torch.stack(bottom_means).mean() if bottom_means else loss.detach() * 0.0

        return {
            "loss": loss,
            "total_loss": loss.detach(),
            "logits": predictor_logits,
            "selector_loss": selector_loss.detach(),
            "rank_loss": rank_loss.detach(),
            "selector_spearman": selector_spearman.detach(),
            "selector_topk_overlap": selector_topk_overlap.detach(),
            "selector_boundary_accuracy": selector_boundary_accuracy.detach(),
            "kl_loss": kl_loss.detach(),
            "ce_loss": ce_loss.detach(),
            "retention_mean": retention_mean.detach(),
            "retention_top_mean": retention_top_mean.detach(),
            "retention_bottom_mean": retention_bottom_mean.detach(),
            "soft_budget_error": (self._last_soft_budget_error.detach() if self._last_soft_budget_error is not None else loss.detach() * 0.0),
            "weighted_rank_loss": weighted_rank_loss.detach(),
            "weighted_kl_loss": weighted_kl_loss.detach(),
            "weighted_ce_loss": weighted_ce_loss.detach(),
            "current_rank_loss_weight": loss.detach() * 0.0 + self._current_rank_loss_weight(),
            "current_kl_loss_weight": loss.detach() * 0.0 + self._current_kl_loss_weight(),
            "current_ce_loss_weight": loss.detach() * 0.0 + self._current_ce_loss_weight(),
        }


class LearnablePruneTrainer(Trainer):
    loss_metric_keys = (
        "total_loss",
        "selector_loss",
        "rank_loss",
        "selector_spearman",
        "selector_topk_overlap",
        "selector_boundary_accuracy",
        "kl_loss",
        "ce_loss",
        "retention_mean",
        "retention_top_mean",
        "retention_bottom_mean",
        "soft_budget_error",
        "weighted_rank_loss",
        "weighted_kl_loss",
        "weighted_ce_loss",
        "current_rank_loss_weight",
        "current_kl_loss_weight",
        "current_ce_loss_weight",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self._pending_loss_metrics: List[Dict[str, torch.Tensor]] = []

    def _metric_scalar(self, value: torch.Tensor) -> float:
        tensor = value.detach().float().reshape(1)
        if hasattr(self, "accelerator") and self.accelerator is not None:
            tensor = self.accelerator.gather_for_metrics(tensor)
        return float(tensor.mean().item())

    def _current_train_progress(self) -> float:
        total_steps = int(getattr(self.state, "max_steps", 0) or getattr(self.args, "max_steps", 0) or 0)
        if total_steps <= 1:
            return 0.0
        step = int(getattr(self.state, "global_step", 0) or 0)
        return float(min(max(step / float(max(1, total_steps - 1)), 0.0), 1.0))

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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._maybe_set_training_progress(model, self._current_train_progress())
        self._maybe_set_training_log_step(model, int(getattr(self.state, "global_step", 0) or 0))
        outputs = model(**inputs)
        loss = outputs["loss"]
        self._pending_loss_metrics.append(
            {key: outputs[key].detach().float().reshape(1) for key in self.loss_metric_keys if key in outputs}
        )
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if self._pending_loss_metrics:
            keys = sorted({key for metrics in self._pending_loss_metrics for key in metrics})
            for key in keys:
                values = [metrics[key] for metrics in self._pending_loss_metrics if key in metrics]
                if values:
                    logs[key] = self._metric_scalar(torch.stack(values, dim=0).mean())
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
        config = {
            "training_variant": "one_stage_dynamic_kl_budgeted_hard_st_topk_precision_hinge",
            "keep_k": model.keep_k,
            "candidate_layers": model.candidate_layers,
            "rank_loss_weight": model.rank_loss_weight,
            "kl_loss_weight": model.kl_loss_weight,
            "ce_loss_weight": model.ce_loss_weight,
            "enable_scale": model.enable_scale,
            "enable_rss": model.enable_rss,
            "teacher_top_iqr_layers": model.teacher_top_iqr_layers,
            "topk_hinge_margin": model.topk_hinge_margin,
            "teacher_layer_quality": "0.6*log_minmax(spatial_entropy)+0.4*log_minmax(iqr)",
            "teacher_layer_pooling": "single_best_layer",
            "kd_temperature": model.kd_temperature,
            "attention_min_keep_prob": 0.0,
            "attention_gate_mode": model.attention_gate_mode,
            "budgeted_soft_topk_iters": model.budgeted_soft_topk_iters,
            "predictor_layer": model.predictor_layer,
            "predictor_layer_enabled": False,
            "teacher_target_logging": model._teacher_target_logger is not None or model.teacher_target_log_to_console,
            "predictor_input_context": "pre_llm_visual_tokens_plus_unlabeled_prompt_tokens",
            "kl_mask_type": "covipal_style_visual_key_log_retention_bias",
            "predictor_input_size": model.predictor.input_size,
            "predictor_hidden_size": model.predictor.hidden_size,
            "predictor_heads": model.predictor.num_heads,
            "predictor_mlp_ratio": model.predictor.mlp_ratio,
            "predictor_layers": getattr(model.predictor, "num_layers", None),
            "predictor_use_final_full_attention": getattr(model.predictor, "use_final_full_attention", None),
        }
        torch.save(config, os.path.join(output_dir, "learnable_prune_config.pt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one-stage Dynamic-KL learnable visual-token pruner")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--vision_tower", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--modeling_file",
        type=str,
        default=None,
        help="Kept for CLI compatibility with the HF trainer; official LLaVA training ignores this.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="Randomly sample this fraction of the training set after max_samples is applied. Use 0.1 for 10%%.",
    )
    parser.add_argument("--keep_k", type=int, default=64)
    parser.add_argument("--candidate_layers", type=str, default="16-24")
    parser.add_argument(
        "--predictor_layer",
        type=int,
        default=-1,
        help="Reserved for future mid-layer predictor inputs; currently saved to config only.",
    )
    parser.add_argument("--predictor_hidden_size", type=int, default=512)
    parser.add_argument("--predictor_heads", type=int, default=8)
    parser.add_argument("--predictor_layers", type=int, default=4)
    parser.add_argument("--predictor_mlp_ratio", type=int, default=2)
    parser.add_argument("--predictor_use_final_full_attention", type=parse_bool_flag, default=False)

    parser.add_argument("--rank_loss_weight", type=float, default=1.0)
    parser.add_argument("--kl_loss_weight", type=float, default=1.0)
    parser.add_argument("--ce_loss_weight", type=float, default=1.0)
    parser.add_argument("--enable_scale", type=parse_bool_flag, default=True)
    parser.add_argument("--enable_rss", type=parse_bool_flag, default=False)
    parser.add_argument("--teacher_top_iqr_layers", type=int, default=3)
    parser.add_argument("--topk_hinge_margin", type=float, default=1.0)
    parser.add_argument("--kd_temperature", type=float, default=1.0)
    parser.add_argument("--budgeted_soft_topk_iters", type=int, default=32)
    parser.add_argument("--teacher_target_log_dir", type=str, default=None)
    parser.add_argument("--teacher_target_log_to_console", type=parse_bool_flag, default=False)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--loss_type", type=str, default="SFT", choices=["SFT", "NTP", "sft", "ntp"])
    parser.add_argument("--gradient_checkpointing", type=parse_bool_flag, default=False)
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

    if args.modeling_file:
        print("--modeling_file is ignored because this trainer uses official LlavaLlamaForCausalLM.")

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
        torch_dtype=dtype,
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
            "context_fn": _math_sdpa_checkpoint_contexts,
        }
        try:
            llava.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        except TypeError:
            llava.gradient_checkpointing_enable()

    hidden_size = int(llava.config.hidden_size)
    predictor = LearnablePrunePredictor(
        input_size=hidden_size,
        hidden_size=args.predictor_hidden_size,
        num_heads=args.predictor_heads,
        mlp_ratio=args.predictor_mlp_ratio,
        num_layers=args.predictor_layers,
        use_final_full_attention=args.predictor_use_final_full_attention,
    )
    model = LearnablePruneWrapper(
        llava_model=llava,
        predictor=predictor,
        keep_k=args.keep_k,
        candidate_layers=args.candidate_layers,
        rank_loss_weight=args.rank_loss_weight,
        kl_loss_weight=args.kl_loss_weight,
        ce_loss_weight=args.ce_loss_weight,
        enable_scale=args.enable_scale,
        enable_rss=args.enable_rss,
        teacher_top_iqr_layers=args.teacher_top_iqr_layers,
        topk_hinge_margin=args.topk_hinge_margin,
        kd_temperature=args.kd_temperature,
        budgeted_soft_topk_iters=args.budgeted_soft_topk_iters,
        predictor_layer=args.predictor_layer,
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
        report_to=report_to,
        run_name=args.run_name,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
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
