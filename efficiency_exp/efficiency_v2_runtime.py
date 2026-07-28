#!/usr/bin/env python3
"""Shared model/runtime utilities for protocol-v2 efficiency measurements.

This module contains only the reusable model adapters, input preparation,
provenance helpers, and analytical token/cache utilities needed by the v2
benchmark and its correctness audit.  It intentionally has no benchmark CLI
or result-writing entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq
import torch
from PIL import Image, ImageOps


HERE = Path(__file__).resolve().parent
OUR_REPO = HERE.parent
LEARNPRUNER_DIR = Path(
    "/data1/chenzixuan/MLLM_Token_Compression_Workdir/src/LearnPruner"
)
OFFICIAL_MODEL = Path("/data1/chenzixuan/model/liuhaotian/llava-v1.5-7b")
HF_MODEL = Path("/data1/chenzixuan/model/llava-hf/llava-1.5-7b-hf")
OURS_CHECKPOINT = Path(
    "/data1/chenzixuan/train_output/"
    "official_llava_v1.5_7b_learnable_prune_lightweight_top80pctscope_"
    "multibudget64_128_192_layer18_sample0.2/checkpoint-500"
)
PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "USER: <image>\nDescribe this image in detail. ASSISTANT:"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(path: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "status": status}
    except Exception as exc:
        return {"error": repr(exc)}


def expand_to_square(
    image: Image.Image,
    image_mean: Sequence[float],
) -> Image.Image:
    """Match official LLaVA's ``image_aspect_ratio=pad`` preprocessing."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width == height:
        return image
    background = tuple(
        int(round(float(channel) * 255.0)) for channel in image_mean
    )
    side = max(width, height)
    result = Image.new("RGB", (side, side), background)
    result.paste(image, ((side - width) // 2, (side - height) // 2))
    return result


@dataclass
class PreparedInput:
    kwargs: dict[str, Any]
    prompt_text_tokens: int
    input_sequence_tokens: int
    original_size: tuple[int, int]


class FirstLayerTokenTracer:
    """Capture the first prefill sequence length seen by every LLM layer."""

    def __init__(self, layers: Iterable[torch.nn.Module]):
        self.layers = list(layers)
        self.tokens: list[int | None] = [None] * len(self.layers)
        self.handles = [
            # The optimized custom decoder intentionally calls a layer's
            # submodules directly, bypassing ``layer.forward``. Every path
            # still enters input_layernorm exactly once per layer invocation.
            getattr(layer, "input_layernorm", layer).register_forward_pre_hook(
                self._make_hook(index)
            )
            for index, layer in enumerate(self.layers)
        ]

    def _make_hook(self, index: int):
        def hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            if self.tokens[index] is not None:
                return
            if not args or not torch.is_tensor(args[0]):
                raise RuntimeError(
                    f"Could not inspect decoder layer {index} hidden states"
                )
            self.tokens[index] = int(args[0].shape[-2])

        return hook

    def reset(self) -> None:
        self.tokens = [None] * len(self.layers)

    def result(self) -> list[int]:
        missing = [
            index
            for index, value in enumerate(self.tokens)
            if value is None
        ]
        if missing:
            raise RuntimeError(
                "Decoder layers were not called during generation: "
                f"{missing}"
            )
        return [int(value) for value in self.tokens if value is not None]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class ModelAdapter:
    """Load and normalize the official/HF benchmark implementations."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        attach_token_tracer: bool = True,
    ):
        self.args = args
        self.device = torch.device(args.device)
        self.dtype = getattr(torch, args.dtype)
        self.model: torch.nn.Module
        self.tokenizer: Any
        self.processor: Any
        self.backend: str
        if args.method in {"vanilla", "ours"}:
            self._load_official()
        else:
            self._load_hf()
        self.model.eval()
        self.layers = self._decoder_layers()
        self.tracer = (
            FirstLayerTokenTracer(self.layers)
            if attach_token_tracer
            else None
        )
        self.text_config = self._text_config()
        self.vision_config = self._vision_config()
        self.projector = self._projector()
        image_processor = getattr(
            self.processor,
            "image_processor",
            self.processor,
        )
        self.image_mean = tuple(
            float(value) for value in image_processor.image_mean
        )
        self.model_parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        self.auxiliary_parameter_count = self._auxiliary_parameter_count()

    def _load_official(self) -> None:
        if str(OUR_REPO) not in sys.path:
            sys.path.insert(0, str(OUR_REPO))
        if self.args.method == "ours":
            checkpoint = self.args.ours_checkpoint.expanduser().resolve()
            if not (checkpoint / "learnable_prune_config.pt").is_file():
                raise FileNotFoundError(
                    checkpoint / "learnable_prune_config.pt"
                )
            os.environ.update(
                {
                    "LEARNABLE_PRUNE_CHECKPOINT": str(checkpoint),
                    "LEARNABLE_PRUNE_AVG_TOKEN_BUDGET": str(
                        self.args.ours_budget
                    ),
                    "ENABLE_PREDICTOR": "1",
                    "ENABLE_FINALWIPE": "1",
                    "LEARNABLE_PRUNE_DIRECT_SDPA": "1",
                    "LEARNABLE_PRUNE_PROFILE_INTERNAL": "0",
                    "LEARNABLE_PRUNE_DEBUG": "0",
                    "LEARNABLE_PRUNE_SCOPE_GRAPH_CACHE_SIZE": "4",
                    "LEARNABLE_PRUNE_IMPLICIT_CAUSAL": (
                        "1" if self.args.ours_implicit_causal else "0"
                    ),
                }
            )
            # Do not set LEARNABLE_TOPK/SCOPE/MID here: the selected checkpoint
            # profile is authoritative (64 -> keep_k=110, scope=137, mid=32).

        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model

        model_path = self.args.official_model.expanduser().resolve()
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(model_path / "config.json")
        load_kwargs: dict[str, Any] = {
            "device_map": str(self.device),
            "dtype": self.args.dtype,
            "attn_implementation": self.args.attn_implementation,
        }
        if self.args.method == "ours":
            load_kwargs[
                "learnable_prune_lightweight_scope_finalwipe_model"
            ] = True
        (
            self.tokenizer,
            self.model,
            self.processor,
            _,
        ) = load_pretrained_model(
            str(model_path),
            None,
            get_model_name_from_path(str(model_path)),
            **load_kwargs,
        )
        if self.args.method == "ours":
            # Load before recording parameter/memory metadata and warmup.
            self.model.load_learnable_prune_checkpoint(str(checkpoint))
        self.backend = "official_llava"

    def _load_hf(self) -> None:
        from transformers import AutoProcessor
        from transformers import (
            LlavaForConditionalGeneration as HFLlava,
        )

        model_path = self.args.hf_model.expanduser().resolve()
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(model_path / "config.json")
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.tokenizer = self.processor.tokenizer
        model_class: Any = HFLlava
        if self.args.method == "learnpruner":
            checkpoint = (
                self.args.learnpruner_checkpoint.expanduser().resolve()
            )
            if not (checkpoint / "learnpruner_config.pt").is_file():
                raise FileNotFoundError(
                    checkpoint / "learnpruner_config.pt"
                )
            if str(LEARNPRUNER_DIR) not in sys.path:
                sys.path.insert(0, str(LEARNPRUNER_DIR))
            from modeling_llava_learnpruner import (
                LlavaForConditionalGeneration as LearnPrunerLlava,
            )

            model_class = LearnPrunerLlava

        self.model = model_class.from_pretrained(
            str(model_path),
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation=self.args.attn_implementation,
        )
        self.model.to(self.device)
        if self.args.method == "learnpruner":
            self.model.load_learnpruner_checkpoint(
                str(
                    self.args.learnpruner_checkpoint
                    .expanduser()
                    .resolve()
                )
            )
            config = self.model.get_learnpruner_config()
            expected = {
                "stage1_tokens": self.args.learnpruner_stage1_tokens,
                "stage2_tokens": self.args.learnpruner_stage2_tokens,
                "prune_layer": self.args.learnpruner_prune_layer,
                "diversity_ratio": self.args.learnpruner_diversity_ratio,
            }
            mismatches = {
                name: (config.get(name), value)
                for name, value in expected.items()
                if config.get(name) != value
            }
            if mismatches:
                raise ValueError(
                    "LearnPruner runtime settings are checkpoint-owned; "
                    f"CLI/checkpoint mismatch: {mismatches}"
                )
        self.backend = "huggingface_llava"

    def _decoder_layers(self) -> Sequence[torch.nn.Module]:
        if self.backend == "official_llava":
            return self.model.get_model().layers
        return self.model.model.language_model.layers

    def _text_config(self) -> Any:
        return getattr(
            self.model.config,
            "text_config",
            self.model.config,
        )

    def _vision_config(self) -> Any:
        if self.backend == "official_llava":
            return self.model.get_vision_tower().config
        return self.model.config.vision_config

    def _projector(self) -> torch.nn.Module:
        if self.backend == "official_llava":
            return self.model.get_model().mm_projector
        return self.model.model.multi_modal_projector

    def _auxiliary_parameter_count(self) -> int:
        candidates = []
        if hasattr(self.model, "learnable_prune_predictor"):
            candidates.append(self.model.learnable_prune_predictor)
        if hasattr(
            getattr(self.model, "model", None),
            "learnpruner_predictor",
        ):
            candidates.append(self.model.model.learnpruner_predictor)
        return sum(
            parameter.numel()
            for module in candidates
            for parameter in module.parameters()
        )

    def reset_pruner_stats(self) -> None:
        if hasattr(self.model, "reset_learnable_prune_stats"):
            self.model.reset_learnable_prune_stats()
        if hasattr(self.model, "reset_learnpruner_stats"):
            self.model.reset_learnpruner_stats()

    def latest_pruner_stats(self) -> dict[str, Any]:
        stats: list[dict[str, Any]] = []
        if hasattr(self.model, "get_learnable_prune_stats"):
            stats = self.model.get_learnable_prune_stats()
        elif hasattr(self.model, "get_learnpruner_stats"):
            stats = self.model.get_learnpruner_stats()
        return dict(stats[-1]) if stats else {}

    def prepare(self, row: dict[str, Any]) -> PreparedInput:
        with Image.open(io.BytesIO(row["binary"])) as raw_image:
            original_size = tuple(int(value) for value in raw_image.size)
            image = expand_to_square(raw_image, self.image_mean)
        if self.backend == "official_llava":
            from llava.constants import IMAGE_TOKEN_INDEX
            from llava.mm_utils import process_images, tokenizer_image_token

            input_ids = tokenizer_image_token(
                PROMPT,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0)
            prompt_text_tokens = int(
                input_ids.shape[1]
                - input_ids.eq(IMAGE_TOKEN_INDEX).sum()
            )
            attention_mask = torch.ones_like(input_ids)
            pixel_values = process_images(
                [image],
                self.processor,
                self.model.config,
            )
            if isinstance(pixel_values, list):
                raise ValueError(
                    "LLaVA-1.5 benchmark expects one dense image tensor"
                )
            kwargs = {
                "inputs": input_ids.to(self.device),
                "attention_mask": attention_mask.to(self.device),
                "images": pixel_values.to(
                    device=self.device,
                    dtype=self.dtype,
                ),
                "image_sizes": [original_size],
            }
            expanded_tokens = prompt_text_tokens + int(
                (
                    self.vision_config.image_size
                    // self.vision_config.patch_size
                )
                ** 2
            )
            return PreparedInput(
                kwargs,
                prompt_text_tokens,
                expanded_tokens,
                original_size,
            )

        batch = self.processor(
            images=image,
            text=PROMPT,
            return_tensors="pt",
        )
        kwargs = {}
        for key, value in batch.items():
            kwargs[key] = value.to(
                device=self.device,
                dtype=(
                    self.dtype
                    if torch.is_floating_point(value)
                    else None
                ),
            )
        image_token_id = int(self.model.config.image_token_id)
        image_tokens = int(kwargs["input_ids"].eq(image_token_id).sum())
        prompt_text_tokens = (
            int(kwargs["attention_mask"].sum()) - image_tokens
        )
        return PreparedInput(
            kwargs,
            prompt_text_tokens,
            int(kwargs["attention_mask"].sum()),
            original_size,
        )

    def generate(
        self,
        prepared: PreparedInput,
        new_tokens: int,
    ) -> torch.Tensor:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        generation_kwargs = {
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "max_new_tokens": int(new_tokens),
            "min_new_tokens": int(new_tokens),
            "pad_token_id": pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        return self.model.generate(
            **prepared.kwargs,
            **generation_kwargs,
        )

    def decode_tail(
        self,
        output: torch.Tensor,
        new_tokens: int,
    ) -> str:
        return self.tokenizer.decode(
            output[0, -new_tokens:],
            skip_special_tokens=True,
        ).strip()

    def close(self) -> None:
        if self.tracer is not None:
            self.tracer.close()


def timed_generate(
    adapter: ModelAdapter,
    prepared: PreparedInput,
    new_tokens: int,
) -> tuple[torch.Tensor, float, float]:
    """Time one complete generation for the v2 correctness audit."""
    torch.cuda.synchronize(adapter.device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    output = adapter.generate(prepared, new_tokens)
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return (
        output,
        float(start_event.elapsed_time(end_event)),
        wall_ms,
    )


def kv_cache_bytes(
    adapter: ModelAdapter,
    layer_tokens: Sequence[int],
) -> int:
    config = adapter.text_config
    heads = int(config.num_attention_heads)
    kv_heads = int(
        getattr(config, "num_key_value_heads", heads)
    )
    head_dim = int(config.hidden_size) // heads
    dtype_bytes = torch.tensor(
        [],
        dtype=adapter.dtype,
    ).element_size()
    return int(
        sum(layer_tokens)
        * 2
        * kv_heads
        * head_dim
        * dtype_bytes
    )


def canonical_layer_tokens(
    args: argparse.Namespace,
    adapter: ModelAdapter,
    traced_tokens: Sequence[int],
    pruner_stats: dict[str, Any],
    prompt_text_tokens: int,
) -> list[int]:
    """Return physical decoder lengths, excluding auxiliary scoring calls."""
    if args.method != "ours" or not pruner_stats:
        return [int(value) for value in traced_tokens]
    num_layers = len(adapter.layers)
    if pruner_stats.get("predictor_only"):
        sequence_tokens = (
            prompt_text_tokens
            + int(pruner_stats["kept_visual_tokens"])
        )
        return [sequence_tokens] * num_layers
    mid_layer = int(pruner_stats["mid_pruning_layer_idx"])
    wipe_layer = int(pruner_stats["final_wipe_layer_idx"])
    if not (0 <= mid_layer <= wipe_layer <= num_layers):
        raise RuntimeError(
            "Invalid pruning boundaries: "
            f"mid={mid_layer}, wipe={wipe_layer}, layers={num_layers}"
        )
    scoped_sequence = (
        prompt_text_tokens
        + int(pruner_stats["kept_visual_tokens"])
    )
    mid_sequence = int(pruner_stats["mid_sequence_tokens"])
    final_sequence = int(pruner_stats["final_sequence_tokens"])
    return (
        [scoped_sequence] * mid_layer
        + [mid_sequence] * (wipe_layer - mid_layer)
        + [final_sequence] * (num_layers - wipe_layer)
    )


def validate_environment(args: argparse.Namespace) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != args.expected_cuda_visible_devices:
        raise RuntimeError(
            "Expected CUDA_VISIBLE_DEVICES="
            f"{args.expected_cuda_visible_devices!r}, got {visible!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Exactly one visible GPU is required for an isolated benchmark, "
            f"got {torch.cuda.device_count()}"
        )
    if not str(args.device).startswith("cuda"):
        raise ValueError("--device must be CUDA")
    torch.cuda.set_device(torch.device(args.device))


def load_rows(
    sample_path: Path,
    num_samples: int,
) -> list[dict[str, Any]]:
    if not sample_path.is_file():
        raise FileNotFoundError(
            f"Missing {sample_path}; "
            "run prepare_detailcaps_sample.py first"
        )
    table = pq.read_table(sample_path)
    required = {"dataset_index", "source", "image", "binary"}
    if not required.issubset(table.column_names):
        missing = sorted(required - set(table.column_names))
        raise ValueError(f"Sample parquet lacks columns {missing}")
    if num_samples <= 0 or num_samples > table.num_rows:
        raise ValueError(
            f"--num-samples must be in [1, {table.num_rows}]"
        )
    return table.slice(0, num_samples).to_pylist()
