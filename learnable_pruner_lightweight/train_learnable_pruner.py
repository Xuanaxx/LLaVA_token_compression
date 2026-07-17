#!/usr/bin/env python3
# coding: utf-8
"""One-stage fixed-layer JS-divergence training for a lightweight LLaVA visual-token pruner.

This trainer implements a single integrated objective:

    L = w_js * JS(full, soft-pruned)
      + w_rank * TopK precision structured hinge
      + w_ce * CE(labels, soft-pruned)

The selector teacher is computed from one fixed decoder layer, controlled by
``--teacher_layer`` and defaulting to layer 18.  This version intentionally does
not run the dynamic candidate-layer search used by the original trainer.

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
import math
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
from llava.mm_utils import get_anyres_image_grid_shape, process_anyres_image, tokenizer_image_token  # noqa: E402
from llava.model.llava_arch import unpad_image  # noqa: E402
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM  # noqa: E402


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")
_VISUAL_COORDINATE_CACHE: Dict[Tuple[Any, ...], torch.Tensor] = {}


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
    def __init__(
        self,
        data_path: str,
        image_folder: str,
        image_processor,
        image_aspect_ratio: str = "pad",
        image_grid_pinpoints=None,
    ):
        self.data_path = data_path
        self.image_folder = image_folder
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
        self.image_grid_pinpoints = image_grid_pinpoints
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
            image_size = tuple(int(x) for x in image.size)
            if self.data_args.image_aspect_ratio == "pad":
                image = _expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            elif self.data_args.image_aspect_ratio == "anyres":
                if self.data_args.image_grid_pinpoints is None:
                    raise ValueError("image_grid_pinpoints is required when --image_aspect_ratio anyres.")
                image = process_anyres_image(image, processor, self.data_args.image_grid_pinpoints)
            else:
                image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            text_sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            text_sources = copy.deepcopy([e["conversations"] for e in sources])
            crop_size = self.data_args.image_processor.crop_size
            h = crop_size["height"] if isinstance(crop_size, dict) else crop_size
            w = crop_size["width"] if isinstance(crop_size, dict) else crop_size
            image = torch.zeros(3, h, w)
            image_size = (int(w), int(h))

        data_dict = preprocess(text_sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        data_dict["image"] = image
        data_dict["image_size"] = image_size
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
        if all("image_size" in instance for instance in instances):
            batch["image_sizes"] = [tuple(int(x) for x in instance["image_size"]) for instance in instances]
        batch["sample_ids"] = torch.tensor(sample_ids, dtype=torch.long)
        return batch


def _cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _optimized_sdpa_context():
    """Use math SDPA for differentiable dense attention-mask biases.

    PyTorch's memory-efficient CUDA backend can accept the forward inputs used
    here but fail during backward with an incorrectly aligned LSE buffer when a
    4D additive mask requires gradients (especially under checkpoint replay).
    Math SDPA is the reliable backend for this uncommon mask-gradient path.
    """

    return torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)


def _optimized_sdpa_checkpoint_contexts():
    return _optimized_sdpa_context(), _optimized_sdpa_context()


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

    _, seq_len = attention_mask.shape
    device = attention_mask.device
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((seq_len, seq_len), min_dtype, device=device, dtype=dtype).triu_(diagonal=1)
    if key_log_bias is None:
        key_bias = torch.zeros_like(attention_mask, dtype=dtype)
    else:
        key_bias = key_log_bias.to(device=device, dtype=dtype)
    key_bias = key_bias.masked_fill(~attention_mask.bool(), min_dtype)
    # Broadcasting avoids materializing a BxSxS boolean `allowed` tensor and a
    # second dense zero tensor before producing the additive attention mask.
    return causal_mask[None, None, :, :] + key_bias[:, None, None, :]


def _image_size_tuple(image_sizes: Any, image_idx: int) -> Tuple[int, int]:
    if image_sizes is None:
        raise ValueError("image_sizes is required for LLaVA-NeXT anyres spatial merging.")
    image_size = image_sizes[image_idx]
    if torch.is_tensor(image_size):
        image_size = image_size.detach().cpu().tolist()
    elif isinstance(image_size, np.ndarray):
        image_size = image_size.tolist()
    if len(image_size) != 2:
        raise ValueError(f"Expected image_sizes[{image_idx}] to contain (width, height), got: {image_size}")
    return int(image_size[0]), int(image_size[1])


def _normalized_grid_coordinates(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    append_newline: bool = False,
) -> torch.Tensor:
    """Return row-major normalized (x, y) coordinates matching merged image features."""
    cache_key = (
        int(height), int(width), bool(append_newline), device.type, device.index, dtype
    )
    cached = _VISUAL_COORDINATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    y = torch.linspace(0.0, 1.0, max(1, int(height)), device=device, dtype=torch.float32)
    x = torch.linspace(0.0, 1.0, max(1, int(width)), device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack((xx, yy), dim=-1)
    if append_newline:
        # Newline embeddings are not image patches. A small out-of-range x
        # coordinate distinguishes them while preserving their row coordinate.
        newline = torch.stack((torch.full_like(y, 1.05), y), dim=-1)[:, None, :]
        coords = torch.cat((coords, newline), dim=1)
    coords = coords.reshape(-1, 2).to(dtype=dtype)
    _VISUAL_COORDINATE_CACHE[cache_key] = coords
    return coords


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
    def forward(  # type: ignore[override]
        ctx,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
        keep_k: int,
        temperature: float,
        max_iter: int,
    ):
        if logits.dim() != 2 or valid_mask.shape != logits.shape:
            raise ValueError("_BudgetedSigmoidTopK expects matching [B, N] logits and masks.")

        valid = valid_mask.bool()
        counts = valid.sum(dim=-1)
        target = counts.clamp(max=int(keep_k))
        keep_k = int(keep_k)
        temperature = max(float(temperature), 1e-6)
        max_iter = int(max_iter)

        x = logits.float()
        free_rows = (target > 0) & (target < counts)
        out = valid.to(dtype=x.dtype) * (target >= counts).unsqueeze(-1).to(dtype=x.dtype)

        # Wide but finite brackets.  These make sigmoid mass almost n / 0 at the
        # endpoints, while avoiding +/-inf in lower-precision training.
        pos_inf = torch.tensor(torch.inf, device=x.device, dtype=x.dtype)
        neg_inf = torch.tensor(-torch.inf, device=x.device, dtype=x.dtype)
        lo = x.masked_fill(~valid, pos_inf).amin(dim=-1) - 50.0 * temperature - 1.0
        hi = x.masked_fill(~valid, neg_inf).amax(dim=-1) + 50.0 * temperature + 1.0
        lo = torch.where(free_rows, lo, torch.zeros_like(lo))
        hi = torch.where(free_rows, hi, torch.zeros_like(hi))
        target_float = target.to(dtype=x.dtype)
        for _ in range(max(8, max_iter)):
            mid = (lo + hi) * 0.5
            mass = (torch.sigmoid((x - mid[:, None]) / temperature) * valid).sum(dim=-1)
            # mass decreases as lambda increases.
            move_lo = (mass > target_float) & free_rows
            lo = torch.where(move_lo, mid, lo)
            hi = torch.where(free_rows & ~move_lo, mid, hi)

        lam = (lo + hi) * 0.5
        relaxed = torch.sigmoid((x - lam[:, None]) / temperature) * valid
        out = torch.where(free_rows[:, None], relaxed, out)
        ctx.save_for_backward(out, valid, free_rows)
        ctx.temperature = temperature
        return out.to(dtype=logits.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        soft, valid, free_rows = ctx.saved_tensors
        temperature = max(float(ctx.temperature), 1e-6)
        grad = grad_output.float()
        soft = soft.float()
        slope = soft * (1.0 - soft) * valid
        denom = slope.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # Implicit differentiation of sum_i sigmoid((x_i-lambda)/tau)=k:
        # dL/dx_j = a_j/tau * (g_j - sum_i g_i a_i / sum_i a_i),
        # where a_i = m_i(1-m_i).
        weighted_mean_grad = (grad * slope).sum(dim=-1, keepdim=True) / denom
        grad_logits = (slope / temperature) * (grad - weighted_mean_grad)
        grad_logits = grad_logits * free_rows[:, None]
        return grad_logits.to(dtype=grad_output.dtype), None, None, None, None


class LearnablePruneWrapper(nn.Module):
    def __init__(
        self,
        llava_model: nn.Module,
        predictor: LearnablePrunePredictor,
        keep_k: int = 64,
        teacher_layer: int = 18,
        rank_loss_weight: float = 1.0,
        js_loss_weight: float = 1.0,
        ce_loss_weight: float = 1.0,
        enable_scale: bool = True,
        enable_rss: bool = False,
        topk_hinge_margin: float = 1.0,
        kd_temperature: float = 1.0,
        budgeted_soft_topk_iters: int = 32,
        teacher_target_log_dir: Optional[str] = None,
        teacher_target_log_to_console: bool = False,
    ):
        super().__init__()
        self.llava = llava_model
        self.predictor = predictor
        self.keep_k = int(keep_k)
        self.teacher_layer = int(teacher_layer)
        self.rank_loss_weight = float(rank_loss_weight)
        self.js_loss_weight = float(js_loss_weight)
        self.ce_loss_weight = float(ce_loss_weight)
        self.enable_scale = bool(enable_scale)
        self.enable_rss = bool(enable_rss)
        self.topk_hinge_margin = float(topk_hinge_margin)
        self.kd_temperature = float(kd_temperature)
        self.budgeted_soft_topk_iters = int(budgeted_soft_topk_iters)
        self._training_progress = 0.0
        self._last_soft_budget_error: Optional[torch.Tensor] = None
        self.attention_gate_mode = "hard_st"

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
                "context_fn": _optimized_sdpa_checkpoint_contexts,
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

    def _current_js_loss_weight(self) -> float:
        return self.js_loss_weight * self._loss_weight_scale(0.2, 2.0)

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

    def _full_forward_with_teacher_qkv(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[Any, Dict[str, torch.Tensor], int]:
        language_model = self.llava.model
        num_layers = len(language_model.layers)
        if not 0 <= self.teacher_layer < num_layers:
            raise ValueError(
                f"--teacher_layer must be in [0, {num_layers - 1}] for this model; got {self.teacher_layer}."
            )
        teacher_layer = self.teacher_layer
        teacher_qkv: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_projection_hook(name: str):
            def hook(_module, _args, output):
                teacher_qkv[name] = output.detach()
            return hook

        self_attn = language_model.layers[teacher_layer].self_attn
        hooks.extend(
            (
                self_attn.q_proj.register_forward_hook(make_projection_hook("query")),
                self_attn.k_proj.register_forward_hook(make_projection_hook("key")),
                self_attn.v_proj.register_forward_hook(make_projection_hook("value")),
            )
        )
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

        if set(teacher_qkv) != {"query", "key", "value"}:
            raise RuntimeError(f"Failed to capture teacher Q/K/V projections for layer: {teacher_layer}")
        return full_outputs, teacher_qkv, teacher_layer

    def _encode_and_merge_image_features(
        self, images: Any, image_sizes: Optional[Any] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Match LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal image merging."""

        if isinstance(images, (list, tuple)) or (torch.is_tensor(images) and images.ndim == 5):
            if isinstance(images, (list, tuple)):
                image_batches = [image.unsqueeze(0) if image.ndim == 3 else image for image in images]
            else:
                image_batches = [image for image in images]

            concat_images = torch.cat([image for image in image_batches], dim=0)
            image_features = self.llava.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in image_batches]
            image_features = torch.split(image_features, split_sizes, dim=0)

            mm_patch_merge_type = getattr(self.llava.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.llava.config, "image_aspect_ratio", "square")
            if mm_patch_merge_type == "flat":
                merged = [feature.flatten(0, 1) for feature in image_features]
                coordinates = []
                side = int(self.llava.get_vision_tower().num_patches_per_side)
                for feature, flat_feature in zip(image_features, merged):
                    grid = _normalized_grid_coordinates(
                        side, side, device=flat_feature.device, dtype=flat_feature.dtype
                    )
                    coordinates.append(grid.repeat(feature.shape[0], 1))
                return merged, coordinates
            if not mm_patch_merge_type.startswith("spatial"):
                raise ValueError(f"Unexpected mm_patch_merge_type: {mm_patch_merge_type}")

            merged_features: List[torch.Tensor] = []
            merged_coordinates: List[torch.Tensor] = []
            vision_tower = self.llava.get_vision_tower()
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    patch_image_feature = image_feature[1:]
                    height = width = vision_tower.num_patches_per_side
                    if height * width != base_image_feature.shape[0]:
                        raise ValueError(
                            f"Unexpected base image feature length {base_image_feature.shape[0]} for {height}x{width} patches."
                        )
                    if image_aspect_ratio != "anyres":
                        raise NotImplementedError(
                            f"Spatial patch merging currently expects image_aspect_ratio='anyres', got {image_aspect_ratio!r}."
                        )
                    image_size = _image_size_tuple(image_sizes, image_idx)
                    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                        image_size,
                        self.llava.config.image_grid_pinpoints,
                        vision_tower.config.image_size,
                    )
                    patch_image_feature = patch_image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    if "unpad" in mm_patch_merge_type:
                        patch_image_feature = patch_image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        patch_image_feature = patch_image_feature.flatten(1, 2).flatten(2, 3)
                        patch_image_feature = unpad_image(patch_image_feature, image_size)
                        patch_height, patch_width = patch_image_feature.shape[-2:]
                        image_newline = self.llava.model.image_newline.to(
                            device=patch_image_feature.device,
                            dtype=patch_image_feature.dtype,
                        )
                        patch_image_feature = torch.cat(
                            (
                                patch_image_feature,
                                image_newline[:, None, None].expand(*patch_image_feature.shape[:-1], 1),
                            ),
                            dim=-1,
                        )
                        patch_image_feature = patch_image_feature.flatten(1, 2).transpose(0, 1)
                        patch_coordinates = _normalized_grid_coordinates(
                            patch_height,
                            patch_width,
                            device=patch_image_feature.device,
                            dtype=patch_image_feature.dtype,
                            append_newline=True,
                        )
                    else:
                        patch_image_feature = patch_image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        patch_image_feature = patch_image_feature.flatten(0, 3)
                        patch_coordinates = _normalized_grid_coordinates(
                            num_patch_height * height,
                            num_patch_width * width,
                            device=patch_image_feature.device,
                            dtype=patch_image_feature.dtype,
                        )
                    image_feature = torch.cat((base_image_feature, patch_image_feature), dim=0)
                    base_coordinates = _normalized_grid_coordinates(
                        height, width, device=image_feature.device, dtype=image_feature.dtype
                    )
                    image_coordinates = torch.cat((base_coordinates, patch_coordinates), dim=0)
                else:
                    image_feature = image_feature[0]
                    height = width = vision_tower.num_patches_per_side
                    if "unpad" in mm_patch_merge_type:
                        image_newline = self.llava.model.image_newline.to(
                            device=image_feature.device,
                            dtype=image_feature.dtype,
                        )
                        image_feature = torch.cat((image_feature, image_newline[None]), dim=0)
                        image_coordinates = _normalized_grid_coordinates(
                            height,
                            width,
                            device=image_feature.device,
                            dtype=image_feature.dtype,
                        )
                        image_coordinates = torch.cat(
                            (image_coordinates, image_coordinates.new_tensor([[1.05, 1.0]])), dim=0
                        )
                    else:
                        image_coordinates = _normalized_grid_coordinates(
                            height, width, device=image_feature.device, dtype=image_feature.dtype
                        )
                merged_features.append(image_feature)
                merged_coordinates.append(image_coordinates)
            return merged_features, merged_coordinates

        image_features = self.llava.encode_images(images)
        if torch.is_tensor(image_features):
            merged = [feature for feature in image_features]
        else:
            merged = list(image_features)
        coordinates = []
        for feature in merged:
            side = int(round(feature.shape[0] ** 0.5))
            if side * side == feature.shape[0]:
                coordinates.append(
                    _normalized_grid_coordinates(side, side, device=feature.device, dtype=feature.dtype)
                )
            else:
                x = torch.linspace(0.0, 1.0, feature.shape[0], device=feature.device, dtype=feature.dtype)
                coordinates.append(torch.stack((x, torch.zeros_like(x)), dim=-1))
        return merged, coordinates

    def _build_multimodal_inputs(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        image_sizes: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        image_features, image_coordinates = self._encode_and_merge_image_features(images, image_sizes=image_sizes)

        compact_input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        compact_labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        embed_tokens = self.llava.get_model().embed_tokens
        safe_input_ids = input_ids.masked_fill(input_ids.eq(IMAGE_TOKEN_INDEX), 0)
        input_embeds = embed_tokens(safe_input_ids)
        compact_input_embeds = [
            cur_embeds[cur_attention_mask]
            for cur_embeds, cur_attention_mask in zip(input_embeds, attention_mask)
        ]

        new_input_embeds: List[torch.Tensor] = []
        new_input_ids: List[torch.Tensor] = []
        new_labels: List[torch.Tensor] = []
        new_visual_coordinates: List[torch.Tensor] = []
        cur_image_idx = 0
        image_token_value = int(IMAGE_TOKEN_INDEX)

        for batch_idx, cur_input_ids in enumerate(compact_input_ids):
            num_images = int((cur_input_ids == IMAGE_TOKEN_INDEX).sum().item())
            cur_labels = compact_labels[batch_idx]
            cur_input_embeds = compact_input_embeds[batch_idx]
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds = torch.cat([cur_input_embeds, cur_image_features[0:0].to(cur_input_embeds.dtype)], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_input_ids.append(cur_input_ids)
                new_labels.append(cur_labels)
                new_visual_coordinates.append(cur_input_embeds.new_zeros((cur_input_embeds.shape[0], 2)))
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_input_embeds_noim = []
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_input_embeds_noim.append(cur_input_embeds[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])

            cur_new_embeds = []
            cur_new_ids = []
            cur_new_labels = []
            cur_new_coordinates = []
            for i in range(num_images + 1):
                cur_new_embeds.append(cur_input_embeds_noim[i])
                cur_new_ids.append(cur_input_ids_noim[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_coordinates.append(cur_input_embeds_noim[i].new_zeros((cur_input_embeds_noim[i].shape[0], 2)))
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx].to(
                        device=cur_input_embeds.device,
                        dtype=cur_input_embeds.dtype,
                    )
                    cur_image_idx += 1
                    cur_image_coordinates = image_coordinates[cur_image_idx - 1].to(
                        device=cur_input_embeds.device, dtype=cur_input_embeds.dtype
                    )
                    visual_len = cur_image_features.shape[0]
                    cur_new_embeds.append(cur_image_features)
                    cur_new_ids.append(torch.full((visual_len,), image_token_value, device=cur_input_ids.device, dtype=cur_input_ids.dtype))
                    cur_new_labels.append(torch.full((visual_len,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_coordinates.append(cur_image_coordinates)

            new_input_embeds.append(torch.cat(cur_new_embeds, dim=0))
            new_input_ids.append(torch.cat(cur_new_ids, dim=0))
            new_labels.append(torch.cat(cur_new_labels, dim=0))
            new_visual_coordinates.append(torch.cat(cur_new_coordinates, dim=0))

        tokenizer_model_max_length = getattr(self.llava.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_input_ids = [x[:tokenizer_model_max_length] for x in new_input_ids]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_visual_coordinates = [x[:tokenizer_model_max_length] for x in new_visual_coordinates]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        pad_token_id = int(getattr(self.llava.config, "pad_token_id", 0) or 0)
        hidden_size = new_input_embeds[0].shape[-1]
        padded_embeds = new_input_embeds[0].new_zeros((batch_size, max_len, hidden_size))
        padded_ids = torch.full((batch_size, max_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        padded_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
        padded_attention = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        padded_position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        padded_visual_coordinates = padded_embeds.new_zeros((batch_size, max_len, 2))
        position_template = torch.arange(max_len, dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_embeds, cur_ids, cur_labels) in enumerate(zip(new_input_embeds, new_input_ids, new_labels)):
            cur_len = cur_embeds.shape[0]
            if getattr(self.llava.config, "tokenizer_padding_side", "right") == "left":
                if cur_len > 0:
                    padded_embeds[i, -cur_len:] = cur_embeds
                    padded_ids[i, -cur_len:] = cur_ids
                    padded_labels[i, -cur_len:] = cur_labels
                    padded_attention[i, -cur_len:] = True
                    padded_position_ids[i, -cur_len:] = position_template[:cur_len]
                    padded_visual_coordinates[i, -cur_len:] = new_visual_coordinates[i]
            else:
                if cur_len > 0:
                    padded_embeds[i, :cur_len] = cur_embeds
                    padded_ids[i, :cur_len] = cur_ids
                    padded_labels[i, :cur_len] = cur_labels
                    padded_attention[i, :cur_len] = True
                    padded_position_ids[i, :cur_len] = position_template[:cur_len]
                    padded_visual_coordinates[i, :cur_len] = new_visual_coordinates[i]

        return (
            padded_embeds,
            padded_ids,
            padded_attention,
            padded_position_ids,
            padded_labels,
            padded_visual_coordinates,
        )

    @staticmethod
    def _pack_mask_positions(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack true positions of a [B, S] mask into [B, max_count]."""

        mask = mask.bool()
        counts = mask.sum(dim=-1)
        max_count = int(counts.max().item()) if counts.numel() else 0
        positions = torch.zeros((mask.shape[0], max_count), device=mask.device, dtype=torch.long)
        valid = torch.arange(max_count, device=mask.device)[None, :] < counts[:, None]
        if max_count:
            batch_idx, seq_idx = mask.nonzero(as_tuple=True)
            slot_idx = mask.long().cumsum(dim=-1)[batch_idx, seq_idx] - 1
            positions[batch_idx, slot_idx] = seq_idx
        return positions, valid

    def _get_visual_token_attention_scores(
        self,
        projected_qkv: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        scoring_layer_idx: int,
        visual_positions: torch.Tensor,
        visual_valid: torch.Tensor,
        query_positions: torch.Tensor,
        query_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Leave-one-visual-token-out scores without anyres padding GEMMs.

        Q/K/V and RoPE are still computed once for the complete teacher batch.
        The expensive query-key and query-visual interactions are evaluated per
        row at their true lengths: LLaVA-NeXT batches can differ by thousands of
        visual/query tokens, so padding those matrices to both batch maxima does
        substantially more work than a tiny batch-row loop.
        """

        language_model = self.llava.model
        scoring_layer = language_model.layers[scoring_layer_idx]
        self_attn = scoring_layer.self_attn
        query_projected = projected_qkv["query"]
        key_projected = projected_qkv["key"]
        value_projected = projected_qkv["value"]
        bsz, seq_len, hidden_size = query_projected.shape
        if visual_positions.shape[1] == 0 or query_positions.shape[1] == 0:
            return query_projected.new_zeros((bsz, visual_positions.shape[1]), dtype=torch.float32)

        num_heads = getattr(self_attn.config, "num_attention_heads", None)
        if num_heads is None:
            num_heads = getattr(self_attn, "num_heads")
        num_heads = int(num_heads)
        num_kv_heads = int(getattr(self_attn.config, "num_key_value_heads", num_heads))
        head_dim = int(self_attn.head_dim)
        scaling = float(getattr(self_attn, "scaling", head_dim ** -0.5))

        query_states = query_projected.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_projected.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_projected.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        cos, sin = language_model.rotary_emb(query_projected, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = _repeat_kv(key_states, num_heads // num_kv_heads)
        value_states = _repeat_kv(value_states, num_heads // num_kv_heads)

        visual_positions = visual_positions.to(dtype=torch.long, device=query_projected.device)
        visual_valid = visual_valid.to(device=query_projected.device)
        query_positions = query_positions.to(dtype=torch.long, device=query_projected.device)
        query_valid = query_valid.to(device=query_projected.device)
        score_sum = query_projected.new_zeros((bsz, visual_positions.shape[1]), dtype=torch.float32)

        # One device-to-host synchronization supplies all small row sizes;
        # it replaces a much costlier padded B x H x Qmax x Smax computation.
        row_sizes = torch.stack(
            (attention_mask.sum(dim=-1), visual_valid.sum(dim=-1), query_valid.sum(dim=-1)),
            dim=-1,
        ).to(device="cpu", dtype=torch.long).tolist()
        left_padding = getattr(self.llava.config, "tokenizer_padding_side", "right") == "left"
        for batch_idx, (key_count, visual_count, query_count) in enumerate(row_sizes):
            if visual_count == 0 or query_count == 0:
                continue

            # _build_multimodal_inputs always emits one contiguous valid span.
            # Slicing it avoids materializing nonzero indices and copying K/V
            # with index_select while retaining left/right-padding semantics.
            key_start = seq_len - key_count if left_padding else 0
            key_end = key_start + key_count
            row_key = key_states[batch_idx, :, key_start:key_end, :]
            row_value = value_states[batch_idx, :, key_start:key_end, :]
            row_visual_positions = visual_positions[batch_idx, :visual_count]
            row_query_positions = query_positions[batch_idx, :query_count]
            row_key_positions = torch.arange(key_start, key_end, device=query_projected.device)
            visual_key_columns = row_visual_positions - key_start
            row_v_visual = row_value.index_select(1, visual_key_columns).permute(1, 0, 2).contiguous()
            row_score_sum = score_sum[batch_idx, :visual_count]

            # Eight queries keeps the temporary [Q, Nv, hidden] delta tensor
            # bounded while providing enough parallel work for the projection.
            for chunk_start in range(0, query_count, 8):
                q_pos = row_query_positions[chunk_start : chunk_start + 8]
                chunk_len = q_pos.shape[0]
                q_states = query_states[batch_idx].index_select(1, q_pos)
                attn_scores = torch.matmul(q_states, row_key.transpose(1, 2)) * scaling
                causal = row_key_positions[None, None, :] <= q_pos[None, :, None]
                attn_scores = attn_scores.float().masked_fill(~causal, torch.finfo(torch.float32).min)
                attn_probs = F.softmax(attn_scores, dim=-1).to(dtype=q_states.dtype)
                z_heads = torch.matmul(attn_probs, row_value).permute(1, 0, 2).contiguous()

                # Visual tokens are guaranteed to be valid keys. Convert their
                # sequence positions to columns in the compact key matrix.
                alpha_visual = attn_probs.index_select(-1, visual_key_columns).permute(1, 2, 0).contiguous()
                beta = alpha_visual / (1.0 - alpha_visual).clamp(min=1e-6)
                delta_z = beta.unsqueeze(-1) * (z_heads.unsqueeze(1) - row_v_visual.unsqueeze(0))
                delta_y = self_attn.o_proj(delta_z.flatten(0, 1).flatten(1, 2)).view(
                    chunk_len, visual_count, hidden_size
                )
                row_score_sum.add_(delta_y.float().norm(dim=-1).sum(dim=0))

            row_score_sum.div_(float(query_count))
        return score_sum

    def _teacher_targets(
        self,
        teacher_qkv: Dict[str, torch.Tensor],
        teacher_layer: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = input_ids.shape[0]
        device = input_ids.device
        image_token_id = int(IMAGE_TOKEN_INDEX)
        visual_mask = input_ids.eq(image_token_id) & attention_mask.bool()
        visual_positions, valid_masks = self._pack_mask_positions(visual_mask)
        if visual_positions.shape[1] == 0:
            return torch.zeros_like(valid_masks, dtype=torch.float32), valid_masks

        sequence_positions = torch.arange(input_ids.shape[1], device=device)[None, :]
        visual_end = torch.where(
            visual_mask,
            sequence_positions,
            torch.full_like(sequence_positions, -1),
        ).amax(dim=-1) + 1
        # Teacher queries must be available at inference. Exclude supervised
        # assistant/answer tokens and score only prompt text after the image.
        query_mask = (
            attention_mask.bool()
            & labels.eq(IGNORE_INDEX)
            & ~visual_mask
            & (sequence_positions >= visual_end[:, None])
        )
        query_positions, query_valid = self._pack_mask_positions(query_mask)
        # Do not manufacture arbitrary TopK labels from an all-zero teacher in
        # the unusual case where no inference-visible text follows the image.
        valid_masks = valid_masks & query_valid.any(dim=-1, keepdim=True)

        scores = self._get_visual_token_attention_scores(
            projected_qkv=teacher_qkv,
            attention_mask=attention_mask,
            position_ids=position_ids,
            scoring_layer_idx=teacher_layer,
            visual_positions=visual_positions,
            visual_valid=valid_masks,
            query_positions=query_positions,
            query_valid=query_valid,
        ).float()

        visual_counts = valid_masks.sum(dim=-1)
        visual_starts = visual_positions[:, 0]
        if self._teacher_target_logger is not None or self.teacher_target_log_to_console:
            for b in range(bsz):
                start = int(visual_starts[b].item())
                visual_count = int(visual_counts[b].item())
                sample_id = int(sample_ids[b].item()) if sample_ids is not None else None
                self._log_teacher_target_selection(
                    sample_id=sample_id,
                    local_batch_index=b,
                    visual_start=start,
                    visual_end=start + visual_count,
                    query_count=int(query_valid[b].sum().item()),
                    teacher_layer=teacher_layer,
                )
        if self.enable_rss and inputs_embeds is not None:
            # RSS depends on a variable-sized Nv x Nv similarity matrix. It is
            # opt-in and remains row-wise to avoid padding anyres batches to the
            # largest quadratic matrix.
            for b in range(bsz):
                start = int(visual_starts[b].item())
                visual_count = int(valid_masks[b].sum().item())
                visual_embeds = inputs_embeds[b : b + 1, start : start + visual_count, :].detach().float()
                scores[b, :visual_count] = self._apply_rss_algorithm(
                    visual_embeds,
                    scores[b, :visual_count],
                )
        return scores, valid_masks

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
        teacher_layer: int,
    ) -> None:
        if self._teacher_target_logger is None and not self.teacher_target_log_to_console:
            return

        record = {
            "global_step": int(self._teacher_target_global_step),
            "log_index": int(self._teacher_target_log_count),
            "rank": int(self._teacher_target_rank),
            "sample_id": sample_id,
            "local_batch_index": int(local_batch_index),
            "visual_span": [int(visual_start), int(visual_end)],
            "visual_token_count": int(visual_end - visual_start),
            "query_count": int(query_count),
            "teacher_layer": int(teacher_layer),
            "selected_layers": [int(teacher_layer)],
            "pooled_layer_count": 1,
        }
        message = json.dumps(record, ensure_ascii=False)
        if self._teacher_target_logger is not None:
            self._teacher_target_logger.info(message)
        if self.teacher_target_log_to_console and self._teacher_target_rank == 0:
            print(f"[teacher-target] {message}", flush=True)
        self._teacher_target_log_count += 1

    def _budgeted_soft_topk_gate(
        self,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return _BudgetedSigmoidTopK.apply(
            logits,
            valid_mask,
            int(self.keep_k),
            1.0,
            int(self.budgeted_soft_topk_iters),
        )

    def _retention_probabilities(
        self,
        predictor_logits: torch.Tensor,
        teacher_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return hard-ST visual retention gates and log-biases."""
        valid = teacher_valid.bool()
        soft = self._budgeted_soft_topk_gate(predictor_logits, valid)
        valid_counts = valid.sum(dim=-1)
        keep_counts = valid_counts.clamp(max=self.keep_k)
        self._last_soft_budget_error = (
            soft.detach().float().sum(dim=-1) - keep_counts.float()
        ).abs().mean()

        topk_count = min(self.keep_k, predictor_logits.shape[1])
        hard = torch.zeros_like(predictor_logits)
        if topk_count > 0:
            masked_logits = predictor_logits.masked_fill(~valid, torch.finfo(predictor_logits.dtype).min)
            topk_indices = torch.topk(masked_logits, k=topk_count, dim=-1).indices
            hard.scatter_(1, topk_indices, 1.0)
        hard = hard * valid.to(dtype=hard.dtype)
        gate = hard + soft - soft.detach()
        probs = torch.where(valid, gate, torch.ones_like(gate))

        min_log_bias = torch.finfo(torch.float32).min
        soft_log_bias = torch.log(soft.float().clamp_min(1e-6))
        hard_log_bias = torch.where(
            hard.bool(),
            torch.zeros_like(soft_log_bias),
            torch.full_like(soft_log_bias, min_log_bias),
        )
        log_biases = hard_log_bias + soft_log_bias - soft_log_bias.detach()
        log_biases = torch.where(valid, log_biases, torch.zeros_like(log_biases))
        return probs, log_biases

    def _selector_loss(
        self,
        predictor_logits: torch.Tensor,
        teacher_scores: torch.Tensor,
        teacher_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = teacher_valid.bool()
        counts = valid.sum(dim=-1)
        metric_rows = counts > 1
        metric_denom = metric_rows.sum().clamp_min(1).float()
        pred = predictor_logits.float()
        teacher = teacher_scores.float()
        pos_inf = torch.tensor(torch.inf, device=pred.device, dtype=pred.dtype)
        neg_inf = torch.tensor(-torch.inf, device=pred.device, dtype=pred.dtype)

        teacher_ranks = torch.argsort(
            torch.argsort(teacher.masked_fill(~valid, pos_inf), dim=-1), dim=-1
        ).float()
        pred_ranks = torch.argsort(
            torch.argsort(pred.masked_fill(~valid, pos_inf), dim=-1), dim=-1
        ).float()
        rank_mean = (counts.float() - 1.0).clamp_min(0.0)[:, None] * 0.5
        teacher_ranks = (teacher_ranks - rank_mean) * valid
        pred_ranks = (pred_ranks - rank_mean) * valid
        spearman_per_row = (teacher_ranks * pred_ranks).sum(dim=-1) / (
            teacher_ranks.square().sum(dim=-1).sqrt()
            * pred_ranks.square().sum(dim=-1).sqrt()
        ).clamp_min(1e-6)
        selector_spearman = (spearman_per_row * metric_rows).sum() / metric_denom

        topk_count = min(self.keep_k, pred.shape[1])
        if topk_count == 0:
            zero = pred.sum() * 0.0
            return zero, selector_spearman, zero, zero
        teacher_top_idx = torch.topk(teacher.masked_fill(~valid, neg_inf), k=topk_count, dim=-1).indices
        pred_top_idx = torch.topk(pred.masked_fill(~valid, neg_inf), k=topk_count, dim=-1).indices
        teacher_top = torch.zeros_like(valid).scatter(1, teacher_top_idx, True) & valid
        pred_top = torch.zeros_like(valid).scatter(1, pred_top_idx, True) & valid
        keep_counts = counts.clamp(max=self.keep_k).clamp_min(1)
        overlap_per_row = (teacher_top & pred_top).sum(dim=-1).float() / keep_counts.float()
        selector_topk_overlap = (overlap_per_row * metric_rows).sum() / metric_denom

        boundary_rows = counts > self.keep_k
        boundary_denom = boundary_rows.sum().clamp_min(1).float()
        negatives = valid & ~teacher_top
        margin_scores = (pred + self.topk_hinge_margin * negatives).masked_fill(~valid, neg_inf)
        violating = torch.topk(margin_scores, k=topk_count, dim=-1).values.sum(dim=-1)
        positive = pred.gather(1, teacher_top_idx).sum(dim=-1)
        row_rank_loss = ((violating - positive) / float(topk_count)).clamp_min(0.0)
        rank_loss = (row_rank_loss * boundary_rows).sum() / metric_denom

        hard_neg_idx = torch.topk(pred.masked_fill(~negatives, neg_inf), k=topk_count, dim=-1).indices
        pos_logits = pred.gather(1, teacher_top_idx)
        neg_logits = pred.gather(1, hard_neg_idx)
        neg_counts = (counts - self.keep_k).clamp(min=0, max=topk_count)
        neg_slots = torch.arange(topk_count, device=pred.device)[None, :] < neg_counts[:, None]
        comparisons = (pos_logits[:, :, None] > neg_logits[:, None, :]) & neg_slots[:, None, :]
        comparison_count = (topk_count * neg_counts).clamp_min(1).float()
        boundary_per_row = comparisons.sum(dim=(1, 2)).float() / comparison_count
        selector_boundary_accuracy = (boundary_per_row * boundary_rows).sum() / boundary_denom
        return rank_loss, selector_spearman, selector_topk_overlap, selector_boundary_accuracy

    def _build_visual_key_log_bias(
        self,
        visual_token_mask: torch.Tensor,
        teacher_valid: torch.Tensor,
        retention_log_bias: torch.Tensor,
    ) -> torch.Tensor:
        key_log_bias = retention_log_bias.new_zeros(visual_token_mask.shape, dtype=torch.float32)
        key_log_bias[visual_token_mask] = retention_log_bias[teacher_valid].float()
        return key_log_bias

    def _answer_token_mask(
        self,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        valid = attention_mask.bool()
        shifted_labels: Optional[torch.Tensor] = None
        if labels is not None and labels.shape == attention_mask.shape:
            shifted_labels = labels[:, 1:]
            valid = valid[:, :-1] & shifted_labels.ne(IGNORE_INDEX)
        return valid, shifted_labels

    def _teacher_answer_logits(
        self,
        full_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        valid, shifted_labels = self._answer_token_mask(attention_mask, labels)
        teacher_hidden = full_hidden_states[:, :-1, :] if shifted_labels is not None else full_hidden_states
        return self.llava.lm_head(teacher_hidden[valid]).detach()

    def _js_ce_losses(
        self,
        teacher_logits: torch.Tensor,
        pruned_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute JS divergence and CE from one shared student LM-head GEMM."""

        student_hidden = pruned_hidden_states
        valid, shifted_labels = self._answer_token_mask(attention_mask, labels)
        if shifted_labels is not None:
            student_hidden = student_hidden[:, :-1, :]
        if teacher_logits.shape[0] == 0:
            zero = student_hidden.sum() * 0.0
            return zero, zero

        student_logits = self.llava.lm_head(student_hidden[valid])
        temperature = max(self.kd_temperature, 1e-6)
        teacher_logp = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
        student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
        mixture_logp = torch.logaddexp(teacher_logp, student_logp) - math.log(2.0)
        js_per_token = 0.5 * (
            F.kl_div(mixture_logp, teacher_logp, log_target=True, reduction="none").sum(dim=-1)
            + F.kl_div(mixture_logp, student_logp, log_target=True, reduction="none").sum(dim=-1)
        )
        js_loss = js_per_token.mean() * (temperature * temperature)

        if shifted_labels is None:
            ce_loss = student_logits.sum() * 0.0
        elif temperature == 1.0:
            # At T=1 student_logp is exactly the log-softmax required by CE.
            # Reusing it avoids a second full-vocabulary fp32 log-softmax pass.
            ce_loss = F.nll_loss(
                student_logp,
                shifted_labels[valid].to(dtype=torch.long),
            )
        else:
            ce_loss = F.cross_entropy(
                student_logits.float(),
                shifted_labels[valid].to(dtype=torch.long),
            )
        return js_loss, ce_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_sizes: Optional[Any] = None,
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
            expanded_visual_coordinates,
        ) = self._build_multimodal_inputs(
            input_ids=input_ids,
            images=image_tensor,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            image_sizes=image_sizes,
        )

        with torch.no_grad():
            full_outputs, teacher_qkv, teacher_layer = self._full_forward_with_teacher_qkv(
                inputs_embeds=inputs_embeds,
                attention_mask=expanded_attention_mask,
                position_ids=original_position_ids,
            )
            teacher_scores, teacher_valid = self._teacher_targets(
                teacher_qkv,
                teacher_layer,
                expanded_input_ids,
                expanded_attention_mask,
                expanded_labels,
                original_position_ids,
                inputs_embeds=inputs_embeds,
                sample_ids=sample_ids,
            )
            teacher_logits = self._teacher_answer_logits(
                full_outputs.last_hidden_state,
                expanded_attention_mask,
                expanded_labels,
            )
            del teacher_qkv
            del full_outputs

        max_visual = teacher_valid.shape[1]
        if max_visual == 0:
            zero = sum(param.sum() for param in self.predictor.parameters()) * 0.0
            return {
                "loss": zero,
                "total_loss": zero.detach(),
                "logits": inputs_embeds.new_zeros((expanded_input_ids.shape[0], 0)),
                "rank_loss": zero.detach(),
                "selector_spearman": zero.detach(),
                "selector_topk_overlap": zero.detach(),
                "selector_boundary_accuracy": zero.detach(),
                "js_loss": zero.detach(),
                "ce_loss": zero.detach(),
                "retention_mean": zero.detach(),
                "retention_top_mean": zero.detach(),
                "retention_bottom_mean": zero.detach(),
                "soft_budget_error": zero.detach(),
                "weighted_rank_loss": zero.detach(),
                "weighted_js_loss": zero.detach(),
                "weighted_ce_loss": zero.detach(),
                "loss_weight_progress": zero.detach() + self._training_progress,
                "current_rank_loss_weight": zero.detach() + self._current_rank_loss_weight(),
                "current_js_loss_weight": zero.detach() + self._current_js_loss_weight(),
                "current_ce_loss_weight": zero.detach() + self._current_ce_loss_weight(),
            }

        visual_token_mask = expanded_input_ids.eq(int(IMAGE_TOKEN_INDEX)) & expanded_attention_mask.bool()
        prompt_text_mask = (
            expanded_attention_mask.bool()
            & expanded_labels.eq(IGNORE_INDEX)
            & ~visual_token_mask
        )
        predictor_logits = self.predictor(
            inputs_embeds.detach(),
            visual_token_mask=visual_token_mask,
            text_token_mask=prompt_text_mask,
            attention_mask=expanded_attention_mask,
            visual_coordinates=expanded_visual_coordinates,
        )

        retention_probs, retention_log_bias = self._retention_probabilities(predictor_logits, teacher_valid)
        (
            rank_loss,
            selector_spearman,
            selector_topk_overlap,
            selector_boundary_accuracy,
        ) = self._selector_loss(
            predictor_logits=predictor_logits,
            teacher_scores=teacher_scores,
            teacher_valid=teacher_valid,
        )

        key_log_bias = self._build_visual_key_log_bias(
            visual_token_mask,
            teacher_valid,
            retention_log_bias,
        )
        soft_prune_attention_mask = _make_4d_causal_mask(
            attention_mask=expanded_attention_mask,
            dtype=inputs_embeds.dtype,
            key_log_bias=key_log_bias,
        )
        # A differentiable dense additive mask requires math SDPA; the efficient
        # CUDA backend currently fails its LSE alignment check during backward.
        with (
            self._temporary_attention_backend("sdpa"),
            self._temporary_decoder_checkpoint_training(),
            _optimized_sdpa_context(),
        ):
            pruned_outputs = self.llava.model(
                inputs_embeds=inputs_embeds,
                attention_mask=soft_prune_attention_mask,
                position_ids=original_position_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        js_loss, ce_loss = self._js_ce_losses(
            teacher_logits,
            pruned_outputs.last_hidden_state,
            expanded_attention_mask,
            labels=expanded_labels,
        )

        weighted_rank_loss = self._current_rank_loss_weight() * rank_loss
        weighted_js_loss = self._current_js_loss_weight() * js_loss
        weighted_ce_loss = self._current_ce_loss_weight() * ce_loss
        loss = weighted_rank_loss + weighted_js_loss + weighted_ce_loss

        if not bool(torch.isfinite(loss).item()):
            details = {
                "rank_loss": float(rank_loss.detach().float().item()),
                "js_loss": float(js_loss.detach().float().item()),
                "ce_loss": float(ce_loss.detach().float().item()),
            }
            raise FloatingPointError(f"Non-finite learnable-prune loss: {details}")

        detached_probs = retention_probs.detach().float()
        valid_probs = detached_probs[teacher_valid]
        retention_mean = valid_probs.mean() if valid_probs.numel() > 0 else loss.detach() * 0.0
        valid_counts = teacher_valid.sum(dim=-1)
        topk_count = min(self.keep_k, predictor_logits.shape[1])
        masked_pred = predictor_logits.detach().float().masked_fill(~teacher_valid, -torch.inf)
        top_idx = torch.topk(masked_pred, k=topk_count, dim=-1).indices
        top_mask = torch.zeros_like(teacher_valid).scatter(1, top_idx, True) & teacher_valid
        metric_rows = valid_counts > 1
        metric_denom = metric_rows.sum().clamp_min(1).float()
        keep_counts = valid_counts.clamp(max=self.keep_k).clamp_min(1)
        top_per_row = (detached_probs * top_mask).sum(dim=-1) / keep_counts.float()
        retention_top_mean = (top_per_row * metric_rows).sum() / metric_denom
        bottom_mask = teacher_valid & ~top_mask
        bottom_rows = valid_counts > self.keep_k
        bottom_counts = bottom_mask.sum(dim=-1).clamp_min(1)
        bottom_per_row = (detached_probs * bottom_mask).sum(dim=-1) / bottom_counts.float()
        retention_bottom_mean = (
            (bottom_per_row * bottom_rows).sum() / bottom_rows.sum().clamp_min(1).float()
        )

        return {
            "loss": loss,
            "total_loss": loss.detach(),
            "logits": predictor_logits,
            "rank_loss": rank_loss.detach(),
            "selector_spearman": selector_spearman.detach(),
            "selector_topk_overlap": selector_topk_overlap.detach(),
            "selector_boundary_accuracy": selector_boundary_accuracy.detach(),
            "js_loss": js_loss.detach(),
            "ce_loss": ce_loss.detach(),
            "retention_mean": retention_mean.detach(),
            "retention_top_mean": retention_top_mean.detach(),
            "retention_bottom_mean": retention_bottom_mean.detach(),
            "soft_budget_error": (self._last_soft_budget_error.detach() if self._last_soft_budget_error is not None else loss.detach() * 0.0),
            "weighted_rank_loss": weighted_rank_loss.detach(),
            "weighted_js_loss": weighted_js_loss.detach(),
            "weighted_ce_loss": weighted_ce_loss.detach(),
            "loss_weight_progress": loss.detach() * 0.0 + self._training_progress,
            "current_rank_loss_weight": loss.detach() * 0.0 + self._current_rank_loss_weight(),
            "current_js_loss_weight": loss.detach() * 0.0 + self._current_js_loss_weight(),
            "current_ce_loss_weight": loss.detach() * 0.0 + self._current_ce_loss_weight(),
        }


class LearnablePruneTrainer(Trainer):
    loss_metric_keys = (
        "total_loss",
        "rank_loss",
        "selector_spearman",
        "selector_topk_overlap",
        "selector_boundary_accuracy",
        "js_loss",
        "ce_loss",
        "retention_mean",
        "retention_top_mean",
        "retention_bottom_mean",
        "soft_budget_error",
        "weighted_rank_loss",
        "weighted_js_loss",
        "weighted_ce_loss",
        "loss_weight_progress",
        "current_rank_loss_weight",
        "current_js_loss_weight",
        "current_ce_loss_weight",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self._pending_loss_metrics: List[Dict[str, torch.Tensor]] = []
        self._scale_total_steps_cache: Optional[int] = None

    def _metric_scalar(self, value: torch.Tensor) -> float:
        tensor = value.detach().float().reshape(1)
        if hasattr(self, "accelerator") and self.accelerator is not None:
            tensor = self.accelerator.gather_for_metrics(tensor)
        return float(tensor.mean().item())

    def _current_train_progress(self) -> float:
        total_steps = self._infer_total_train_steps()
        if total_steps <= 1:
            return 0.0
        step = int(getattr(self.state, "global_step", 0) or 0)
        return float(min(max(step / float(max(1, total_steps - 1)), 0.0), 1.0))

    def _infer_total_train_steps(self) -> int:
        for value in (getattr(self.state, "max_steps", 0), getattr(self.args, "max_steps", 0)):
            value = int(value or 0)
            if value > 0:
                self._scale_total_steps_cache = value
                return value

        if self._scale_total_steps_cache is not None:
            return self._scale_total_steps_cache

        try:
            dataloader_len = len(self.get_train_dataloader())
        except Exception:
            dataloader_len = 0
        if dataloader_len <= 0:
            return 0

        grad_accum = max(1, int(getattr(self.args, "gradient_accumulation_steps", 1) or 1))
        updates_per_epoch = max(1, math.ceil(dataloader_len / grad_accum))
        num_epochs = max(float(getattr(self.args, "num_train_epochs", 1.0) or 1.0), 0.0)
        self._scale_total_steps_cache = max(1, math.ceil(updates_per_epoch * num_epochs))
        return self._scale_total_steps_cache

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
            "training_variant": "one_stage_fixed_layer_js_budgeted_hard_st_topk_precision_hinge_tclm_multihead_rank_swiglu",
            "keep_k": model.keep_k,
            "teacher_layer": model.teacher_layer,
            "rank_loss_weight": model.rank_loss_weight,
            "js_loss_weight": model.js_loss_weight,
            "ce_loss_weight": model.ce_loss_weight,
            "enable_scale": model.enable_scale,
            "enable_rss": model.enable_rss,
            "topk_hinge_margin": model.topk_hinge_margin,
            "teacher_layer_quality": "fixed_layer",
            "teacher_layer_pooling": "fixed_single_layer",
            "kd_temperature": model.kd_temperature,
            "attention_gate_mode": model.attention_gate_mode,
            "budgeted_soft_topk_iters": model.budgeted_soft_topk_iters,
            "teacher_target_logging": model._teacher_target_logger is not None or model.teacher_target_log_to_console,
            "predictor_input_context": "pre_llm_visual_tokens_plus_unlabeled_prompt_tokens",
            "distillation_loss": "jensen_shannon_full_vocabulary",
            "distillation_mask_type": "covipal_style_visual_key_log_retention_bias",
            "predictor_input_size": model.predictor.input_size,
            "predictor_hidden_size": model.predictor.hidden_size,
            "predictor_rank": getattr(model.predictor, "rank", None),
            "predictor_num_heads": getattr(model.predictor, "num_heads", None),
            "predictor_rank_mlp_ratio": getattr(model.predictor, "rank_mlp_ratio", None),
            "predictor_use_visual_position": getattr(model.predictor, "use_visual_position", None),
            "predictor_use_true_visual_coordinates": True,
            "predictor_attention_implementation": "fused_sdpa",
            "predictor_use_text_position": getattr(model.predictor, "use_text_position", None),
        }
        torch.save(config, os.path.join(output_dir, "learnable_prune_config.pt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fixed-layer lightweight learnable visual-token pruner")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--vision_tower", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="Randomly sample this fraction of the training set after max_samples is applied. Use 0.1 for 10%%.",
    )
    parser.add_argument("--keep_k", type=int, default=160)
    parser.add_argument("--teacher_layer", type=int, default=18)
    parser.add_argument("--predictor_hidden_size", type=int, default=512)
    parser.add_argument("--predictor_rank", type=int, default=256)
    parser.add_argument("--predictor_num_heads", type=int, default=4)
    parser.add_argument("--predictor_rank_mlp_ratio", type=int, default=2)
    parser.add_argument("--predictor_use_visual_position", type=parse_bool_flag, default=True)
    parser.add_argument("--predictor_use_text_position", type=parse_bool_flag, default=True)

    parser.add_argument("--rank_loss_weight", type=float, default=0.5)
    parser.add_argument(
        "--js_loss_weight",
        "--kl_loss_weight",
        dest="js_loss_weight",
        type=float,
        default=6.0,
        help="JS-divergence loss weight. --kl_loss_weight remains a deprecated compatibility alias.",
    )
    parser.add_argument("--ce_loss_weight", type=float, default=0.5)
    parser.add_argument("--enable_scale", type=parse_bool_flag, default=True)
    parser.add_argument("--enable_rss", type=parse_bool_flag, default=False)
    parser.add_argument("--topk_hinge_margin", type=float, default=1.0)
    parser.add_argument("--kd_temperature", type=float, default=1.0)
    parser.add_argument("--budgeted_soft_topk_iters", type=int, default=16)
    parser.add_argument("--teacher_target_log_dir", type=str, default=None)
    parser.add_argument("--teacher_target_log_to_console", type=parse_bool_flag, default=False)

    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--gradient_checkpointing", type=parse_bool_flag, default=True)
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
            "context_fn": _optimized_sdpa_checkpoint_contexts,
        }
        try:
            llava.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        except TypeError:
            llava.gradient_checkpointing_enable()

    hidden_size = int(llava.config.hidden_size)
    predictor = LearnablePrunePredictor(
        input_size=hidden_size,
        hidden_size=args.predictor_hidden_size,
        rank=args.predictor_rank,
        num_heads=args.predictor_num_heads,
        rank_mlp_ratio=args.predictor_rank_mlp_ratio,
        use_visual_position=args.predictor_use_visual_position,
        use_text_position=args.predictor_use_text_position,
    )
    model = LearnablePruneWrapper(
        llava_model=llava,
        predictor=predictor,
        keep_k=args.keep_k,
        teacher_layer=args.teacher_layer,
        rank_loss_weight=args.rank_loss_weight,
        js_loss_weight=args.js_loss_weight,
        ce_loss_weight=args.ce_loss_weight,
        enable_scale=args.enable_scale,
        enable_rss=args.enable_rss,
        topk_hinge_margin=args.topk_hinge_margin,
        kd_temperature=args.kd_temperature,
        budgeted_soft_topk_iters=args.budgeted_soft_topk_iters,
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
        image_grid_pinpoints=getattr(llava.config, "image_grid_pinpoints", None),
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
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        dataloader_prefetch_factor=(args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None),
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
