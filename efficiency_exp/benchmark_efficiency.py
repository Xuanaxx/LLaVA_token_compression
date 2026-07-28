#!/usr/bin/env python3
"""Benchmark one LLaVA token-compression method on a fixed DetailCaps sample.

Each method runs in a separate process.  CUDA events surround the complete
``generate`` call so that vision encoding is included for both the official
LLaVA and Hugging Face implementations.  A one-token generation measures
prefill/TTFT, and an independent fixed-length generation measures total time.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq
import torch
from PIL import Image, ImageOps
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
OUR_REPO = HERE.parent
LEARNPRUNER_DIR = Path(
    "/data1/chenzixuan/MLLM_Token_Compression_Workdir/src/LearnPruner"
)
OFFICIAL_MODEL = Path("/data1/chenzixuan/model/liuhaotian/llava-v1.5-7b")
HF_MODEL = Path("/data1/chenzixuan/model/llava-hf/llava-1.5-7b-hf")
LEARNPRUNER_CHECKPOINT = Path(
    "/data1/chenzixuan/train_output/"
    "learnpruner_llava15_7b_paper_aligned_smoke_cuda2"
)
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
RESULT_FIELDS = [
    "method",
    "sample_order",
    "dataset_index",
    "source",
    "image",
    "prompt_text_tokens",
    "output_tokens",
    "avg_visual_tokens",
    "estimated_prefill_tflops",
    "prefill_cuda_ms",
    "total_cuda_ms",
    "decode_cuda_ms_per_token",
    "total_wall_ms",
    "throughput_output_tokens_per_s",
    "kv_cache_mb",
    "peak_allocated_gb",
    "peak_reserved_gb",
    "layer_prefill_tokens",
    "pruner_stats",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("vanilla", "learnpruner", "ours", "vanilla_hf"), required=True
    )
    parser.add_argument(
        "--sample-parquet",
        type=Path,
        default=HERE / "samples" / "detailcaps_seed42_n500.parquet",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-cuda-visible-devices", default="7")
    parser.add_argument("--official-model", type=Path, default=OFFICIAL_MODEL)
    parser.add_argument("--hf-model", type=Path, default=HF_MODEL)
    parser.add_argument(
        "--learnpruner-checkpoint", type=Path, default=LEARNPRUNER_CHECKPOINT
    )
    parser.add_argument("--ours-checkpoint", type=Path, default=OURS_CHECKPOINT)
    parser.add_argument("--ours-budget", type=int, choices=(64, 128, 192), default=64)
    parser.add_argument("--learnpruner-stage1-tokens", type=int, default=111)
    parser.add_argument("--learnpruner-stage2-tokens", type=int, default=37)
    parser.add_argument("--learnpruner-prune-layer", type=int, default=12)
    parser.add_argument("--learnpruner-diversity-ratio", type=float, default=0.1)
    parser.add_argument(
        "--ours-implicit-causal",
        action="store_true",
        help="Use the semantics-equivalent implicit SDPA causal mask optimization.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def expand_to_square(image: Image.Image, image_mean: Sequence[float]) -> Image.Image:
    """Match official LLaVA's ``image_aspect_ratio=pad`` preprocessing."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width == height:
        return image
    background = tuple(int(round(float(channel) * 255.0)) for channel in image_mean)
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
    """Capture the sequence length at the first (prefill) call of every LLM layer."""

    def __init__(self, layers: Iterable[torch.nn.Module]):
        self.layers = list(layers)
        self.tokens: list[int | None] = [None] * len(self.layers)
        self.handles = [
            # The optimized custom decoder intentionally calls a layer's
            # submodules directly, bypassing ``layer.forward``.  Every path
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
                raise RuntimeError(f"Could not inspect decoder layer {index} hidden states")
            self.tokens[index] = int(args[0].shape[-2])

        return hook

    def reset(self) -> None:
        self.tokens = [None] * len(self.layers)

    def result(self) -> list[int]:
        missing = [index for index, value in enumerate(self.tokens) if value is None]
        if missing:
            raise RuntimeError(f"Decoder layers were not called during generation: {missing}")
        return [int(value) for value in self.tokens if value is not None]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class ModelAdapter:
    def __init__(self, args: argparse.Namespace, *, attach_token_tracer: bool = True):
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
            FirstLayerTokenTracer(self.layers) if attach_token_tracer else None
        )
        self.text_config = self._text_config()
        self.vision_config = self._vision_config()
        self.projector = self._projector()
        image_processor = getattr(self.processor, "image_processor", self.processor)
        self.image_mean = tuple(float(x) for x in image_processor.image_mean)
        self.model_parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.auxiliary_parameter_count = self._auxiliary_parameter_count()

    def _load_official(self) -> None:
        if str(OUR_REPO) not in sys.path:
            sys.path.insert(0, str(OUR_REPO))
        if self.args.method == "ours":
            checkpoint = self.args.ours_checkpoint.expanduser().resolve()
            if not (checkpoint / "learnable_prune_config.pt").is_file():
                raise FileNotFoundError(checkpoint / "learnable_prune_config.pt")
            os.environ.update(
                {
                    "LEARNABLE_PRUNE_CHECKPOINT": str(checkpoint),
                    "LEARNABLE_PRUNE_AVG_TOKEN_BUDGET": str(self.args.ours_budget),
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
            load_kwargs["learnable_prune_lightweight_scope_finalwipe_model"] = True
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            str(model_path),
            None,
            get_model_name_from_path(str(model_path)),
            **load_kwargs,
        )
        if self.args.method == "ours":
            # Load before recording parameter/memory metadata and before warmup.
            self.model.load_learnable_prune_checkpoint(str(checkpoint))
        self.backend = "official_llava"

    def _load_hf(self) -> None:
        from transformers import AutoProcessor, LlavaForConditionalGeneration as HFLlava

        model_path = self.args.hf_model.expanduser().resolve()
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(model_path / "config.json")
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.tokenizer = self.processor.tokenizer
        model_class: Any = HFLlava
        if self.args.method == "learnpruner":
            checkpoint = self.args.learnpruner_checkpoint.expanduser().resolve()
            if not (checkpoint / "learnpruner_config.pt").is_file():
                raise FileNotFoundError(checkpoint / "learnpruner_config.pt")
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
                str(self.args.learnpruner_checkpoint.expanduser().resolve())
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
        return getattr(self.model.config, "text_config", self.model.config)

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
        if hasattr(getattr(self.model, "model", None), "learnpruner_predictor"):
            candidates.append(self.model.model.learnpruner_predictor)
        return sum(parameter.numel() for module in candidates for parameter in module.parameters())

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
            original_size = tuple(int(x) for x in raw_image.size)
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
            prompt_text_tokens = int(input_ids.shape[1] - input_ids.eq(IMAGE_TOKEN_INDEX).sum())
            attention_mask = torch.ones_like(input_ids)
            pixel_values = process_images(
                [image], self.processor, self.model.config
            )
            if isinstance(pixel_values, list):
                raise ValueError("LLaVA-1.5 benchmark expects one dense image tensor")
            kwargs = {
                "inputs": input_ids.to(self.device),
                "attention_mask": attention_mask.to(self.device),
                "images": pixel_values.to(device=self.device, dtype=self.dtype),
                "image_sizes": [original_size],
            }
            expanded_tokens = prompt_text_tokens + int(
                (self.vision_config.image_size // self.vision_config.patch_size) ** 2
            )
            return PreparedInput(kwargs, prompt_text_tokens, expanded_tokens, original_size)

        batch = self.processor(images=image, text=PROMPT, return_tensors="pt")
        kwargs = {}
        for key, value in batch.items():
            kwargs[key] = value.to(
                device=self.device,
                dtype=self.dtype if torch.is_floating_point(value) else None,
            )
        image_token_id = int(self.model.config.image_token_id)
        image_tokens = int(kwargs["input_ids"].eq(image_token_id).sum())
        prompt_text_tokens = int(kwargs["attention_mask"].sum()) - image_tokens
        return PreparedInput(
            kwargs,
            prompt_text_tokens,
            int(kwargs["attention_mask"].sum()),
            original_size,
        )

    def generate(self, prepared: PreparedInput, new_tokens: int) -> torch.Tensor:
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
        return self.model.generate(**prepared.kwargs, **generation_kwargs)

    def decode_tail(self, output: torch.Tensor, new_tokens: int) -> str:
        return self.tokenizer.decode(output[0, -new_tokens:], skip_special_tokens=True).strip()

    def close(self) -> None:
        if self.tracer is not None:
            self.tracer.close()


def timed_generate(
    adapter: ModelAdapter,
    prepared: PreparedInput,
    new_tokens: int,
) -> tuple[torch.Tensor, float, float]:
    torch.cuda.synchronize(adapter.device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    output = adapter.generate(prepared, new_tokens)
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return output, float(start_event.elapsed_time(end_event)), wall_ms


def estimate_prefill_flops(
    adapter: ModelAdapter,
    layer_tokens: Sequence[int],
) -> float:
    """Analytical FLOPs for vision tower + projector + actual LLM layer lengths.

    A multiply-add counts as two FLOPs.  This mirrors the scale of Table 5 and
    deliberately excludes the tiny, implementation-specific pruning selection
    kernels, which are fully represented in measured latency.
    """
    text = adapter.text_config
    hidden = int(text.hidden_size)
    intermediate = int(text.intermediate_size)
    heads = int(text.num_attention_heads)
    kv_heads = int(getattr(text, "num_key_value_heads", heads))
    head_dim = hidden // heads
    kv_hidden = kv_heads * head_dim
    decoder_flops = 0.0
    for tokens in layer_tokens:
        projection_weights = hidden * (hidden + 2 * kv_hidden + hidden)
        mlp_weights = 3 * hidden * intermediate
        decoder_flops += 2.0 * tokens * (projection_weights + mlp_weights)
        decoder_flops += 4.0 * tokens * tokens * heads * head_dim

    # Generation requests only the final prefill logit in current Transformers.
    decoder_flops += 2.0 * hidden * int(text.vocab_size)

    vision = adapter.vision_config
    vision_hidden = int(vision.hidden_size)
    vision_intermediate = int(vision.intermediate_size)
    vision_layers = int(vision.num_hidden_layers)
    image_size = int(vision.image_size)
    patch_size = int(vision.patch_size)
    patch_tokens = (image_size // patch_size) ** 2
    vision_tokens = patch_tokens + 1  # CLIP class token is computed then dropped.
    vision_linear = 2.0 * vision_tokens * (
        4 * vision_hidden * vision_hidden + 2 * vision_hidden * vision_intermediate
    )
    vision_attention = 4.0 * vision_tokens * vision_tokens * vision_hidden
    patch_embedding = 2.0 * patch_tokens * (3 * patch_size * patch_size) * vision_hidden
    vision_flops = vision_layers * (vision_linear + vision_attention) + patch_embedding

    projector_weights = sum(
        int(module.in_features) * int(module.out_features)
        for module in adapter.projector.modules()
        if isinstance(module, torch.nn.Linear)
    )
    projector_flops = 2.0 * patch_tokens * projector_weights
    return decoder_flops + vision_flops + projector_flops


def kv_cache_bytes(adapter: ModelAdapter, layer_tokens: Sequence[int]) -> int:
    config = adapter.text_config
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(config.hidden_size) // heads
    dtype_bytes = torch.tensor([], dtype=adapter.dtype).element_size()
    return int(sum(layer_tokens) * 2 * kv_heads * head_dim * dtype_bytes)


def canonical_layer_tokens(
    args: argparse.Namespace,
    adapter: ModelAdapter,
    traced_tokens: Sequence[int],
    pruner_stats: dict[str, Any],
    prompt_text_tokens: int,
) -> list[int]:
    """Return actual decoder-layer lengths, excluding auxiliary scoring calls.

    The custom method's middle-layer scorer reuses that layer's input norm
    before the compact sequence enters the real decoder layer.  A submodule
    hook therefore observes the scorer's longer tensor first.  Its inference
    stats describe the physical decoder stages unambiguously, so reconstruct
    that one method's layer lengths from those recorded stage boundaries.
    """
    if args.method != "ours" or not pruner_stats:
        return [int(value) for value in traced_tokens]
    num_layers = len(adapter.layers)
    if pruner_stats.get("predictor_only"):
        sequence_tokens = prompt_text_tokens + int(pruner_stats["kept_visual_tokens"])
        return [sequence_tokens] * num_layers
    mid_layer = int(pruner_stats["mid_pruning_layer_idx"])
    wipe_layer = int(pruner_stats["final_wipe_layer_idx"])
    if not (0 <= mid_layer <= wipe_layer <= num_layers):
        raise RuntimeError(
            f"Invalid pruning boundaries: mid={mid_layer}, wipe={wipe_layer}, layers={num_layers}"
        )
    scoped_sequence = prompt_text_tokens + int(pruner_stats["kept_visual_tokens"])
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
            f"Expected CUDA_VISIBLE_DEVICES={args.expected_cuda_visible_devices!r}, got {visible!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Exactly one visible GPU is required for an isolated benchmark, got {torch.cuda.device_count()}"
        )
    if not str(args.device).startswith("cuda"):
        raise ValueError("--device must be CUDA")
    torch.cuda.set_device(torch.device(args.device))


def load_rows(sample_path: Path, num_samples: int) -> list[dict[str, Any]]:
    if not sample_path.is_file():
        raise FileNotFoundError(
            f"Missing {sample_path}; run prepare_detailcaps_sample.py first"
        )
    table = pq.read_table(sample_path)
    required = {"dataset_index", "source", "image", "binary"}
    if not required.issubset(table.column_names):
        raise ValueError(f"Sample parquet lacks columns {sorted(required - set(table.column_names))}")
    if num_samples <= 0 or num_samples > table.num_rows:
        raise ValueError(f"--num-samples must be in [1, {table.num_rows}]")
    return table.slice(0, num_samples).to_pylist()


def completed_indices(output: Path) -> set[int]:
    if not output.is_file():
        return set()
    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != RESULT_FIELDS:
            raise RuntimeError(f"Existing result has an incompatible header: {reader.fieldnames}")
        return {int(row["dataset_index"]) for row in reader}


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    adapter: ModelAdapter,
    sample_path: Path,
) -> None:
    properties = torch.cuda.get_device_properties(adapter.device)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": args.method,
        "backend": adapter.backend,
        "protocol": {
            "dataset": "CAPTURE / DetailCaps-4870",
            "sample_parquet": str(sample_path),
            "sample_sha256": file_sha256(sample_path),
            "sampling_seed": args.seed,
            "num_samples": args.num_samples,
            "prompt": PROMPT,
            "batch_size": 1,
            "warmup_samples": args.warmup,
            "output_tokens": args.max_new_tokens,
            "force_fixed_output_length": True,
            "dtype": args.dtype,
            "attention_implementation": args.attn_implementation,
            "prefill_definition": "CUDA time for complete generate(min=max_new_tokens=1)",
            "total_definition": (
                "CUDA time for independent complete generate with fixed output length"
            ),
            "preprocessing_in_timing": False,
            "vision_encoder_in_timing": True,
            "tflops_definition": (
                "Analytical prefill FLOPs (2 FLOPs/MAC) for CLIP vision tower, "
                "projector, and actual per-layer LLM sequence lengths; pruning kernels excluded"
            ),
            "kv_cache_definition": "Analytical bytes from actual per-layer prefill lengths",
            "gpu_memory_definition": "torch.cuda.max_memory_allocated including weights",
        },
        "paths": {
            "official_model": str(args.official_model.expanduser().resolve()),
            "hf_model": str(args.hf_model.expanduser().resolve()),
            "learnpruner_checkpoint": str(
                args.learnpruner_checkpoint.expanduser().resolve()
            ),
            "ours_checkpoint": str(args.ours_checkpoint.expanduser().resolve()),
        },
        "pruning": {
            "ours_budget": args.ours_budget,
            "ours_implicit_causal": args.ours_implicit_causal,
            "learnpruner_stage1_tokens": args.learnpruner_stage1_tokens,
            "learnpruner_stage2_tokens": args.learnpruner_stage2_tokens,
            "learnpruner_prune_layer_index": args.learnpruner_prune_layer,
            "learnpruner_diversity_ratio": args.learnpruner_diversity_ratio,
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "gpu": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_index": adapter.device.index,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
        },
        "model": {
            "parameter_count": adapter.model_parameter_count,
            "auxiliary_parameter_count": adapter.auxiliary_parameter_count,
            "decoder_layers": len(adapter.layers),
        },
        "git": {
            "ours": git_info(OUR_REPO),
            "learnpruner_workdir": git_info(LEARNPRUNER_DIR.parents[1]),
        },
        "command": sys.argv,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_environment(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sample_path = args.sample_parquet.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (HERE / "results" / f"{args.method}_seed{args.seed}_n{args.num_samples}.tsv")
    )
    metadata_output = output.with_suffix(".metadata.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(f"{output} exists; pass --resume or --overwrite")
    if args.overwrite:
        output.unlink(missing_ok=True)
        metadata_output.unlink(missing_ok=True)
    done = completed_indices(output) if args.resume else set()
    rows = load_rows(sample_path, args.num_samples)
    requested_indices = {int(row["dataset_index"]) for row in rows}
    if not done.issubset(requested_indices):
        raise RuntimeError("Existing output contains dataset indices outside the requested sample")

    print(
        f"Loading method={args.method} on physical CUDA {os.environ['CUDA_VISIBLE_DEVICES']} "
        f"({len(done)}/{len(rows)} samples already complete)",
        flush=True,
    )
    adapter = ModelAdapter(args)
    write_metadata(metadata_output, args, adapter, sample_path)
    print(
        f"GPU={torch.cuda.get_device_name(adapter.device)}, backend={adapter.backend}, "
        f"parameters={adapter.model_parameter_count / 1e9:.3f}B, "
        f"auxiliary={adapter.auxiliary_parameter_count / 1e6:.3f}M",
        flush=True,
    )

    warmup_rows = rows[: min(args.warmup, len(rows))]
    with torch.inference_mode():
        for row in tqdm(warmup_rows, desc=f"Warmup {args.method}"):
            prepared = adapter.prepare(row)
            adapter.generate(prepared, 1)
            adapter.generate(prepared, args.max_new_tokens)
            del prepared
    torch.cuda.synchronize(adapter.device)
    gc.collect()
    baseline_allocated_gb = torch.cuda.memory_allocated(adapter.device) / 1e9
    print(f"Post-warmup allocated memory: {baseline_allocated_gb:.3f} GB", flush=True)

    file_exists = output.exists() and output.stat().st_size > 0
    handle = output.open("a" if file_exists else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
    if not file_exists:
        writer.writeheader()
        handle.flush()

    measured = 0
    try:
        progress = tqdm(rows, desc=f"Benchmark {args.method}")
        for sample_order, row in enumerate(progress):
            dataset_index = int(row["dataset_index"])
            if dataset_index in done:
                continue
            prepared = adapter.prepare(row)
            adapter.reset_pruner_stats()

            # Complete one-token generation: includes image encoder, projector,
            # pruning, LLM prefill, and the first-token logits/sampling step.
            adapter.tracer.reset()
            with torch.inference_mode():
                prefill_output, prefill_cuda_ms, _ = timed_generate(adapter, prepared, 1)
            del prefill_output

            # Independent fixed-length generation.  Suppressing EOS through
            # min_new_tokens makes total latency comparable across methods.
            adapter.tracer.reset()
            torch.cuda.reset_peak_memory_stats(adapter.device)
            with torch.inference_mode():
                total_output, total_cuda_ms, total_wall_ms = timed_generate(
                    adapter, prepared, args.max_new_tokens
                )
            traced_layer_tokens = adapter.tracer.result()
            peak_allocated_gb = torch.cuda.max_memory_allocated(adapter.device) / 1e9
            peak_reserved_gb = torch.cuda.max_memory_reserved(adapter.device) / 1e9
            pruner_stats = adapter.latest_pruner_stats()
            layer_tokens = canonical_layer_tokens(
                args,
                adapter,
                traced_layer_tokens,
                pruner_stats,
                prepared.prompt_text_tokens,
            )
            if measured == 0:
                print(
                    f"First output preview: {adapter.decode_tail(total_output, args.max_new_tokens)!r}",
                    flush=True,
                )

            if len(layer_tokens) != int(adapter.text_config.num_hidden_layers):
                raise RuntimeError(
                    f"Captured {len(layer_tokens)} layers, expected "
                    f"{adapter.text_config.num_hidden_layers}"
                )
            visual_tokens = [
                tokens - prepared.prompt_text_tokens for tokens in layer_tokens
            ]
            if any(tokens < 0 for tokens in visual_tokens):
                raise RuntimeError(
                    f"Negative visual token count: layers={layer_tokens}, "
                    f"text={prepared.prompt_text_tokens}"
                )
            avg_visual_tokens = sum(visual_tokens) / len(visual_tokens)
            estimated_tflops = estimate_prefill_flops(adapter, layer_tokens) / 1e12
            cache_mb = kv_cache_bytes(adapter, layer_tokens) / 1e6
            decode_steps = args.max_new_tokens - 1
            decode_ms_per_token = (
                max(0.0, total_cuda_ms - prefill_cuda_ms) / decode_steps
                if decode_steps > 0
                else math.nan
            )
            throughput = args.max_new_tokens / (total_wall_ms / 1000.0)
            writer.writerow(
                {
                    "method": args.method,
                    "sample_order": sample_order,
                    "dataset_index": dataset_index,
                    "source": row["source"],
                    "image": row["image"],
                    "prompt_text_tokens": prepared.prompt_text_tokens,
                    "output_tokens": args.max_new_tokens,
                    "avg_visual_tokens": f"{avg_visual_tokens:.9f}",
                    "estimated_prefill_tflops": f"{estimated_tflops:.9f}",
                    "prefill_cuda_ms": f"{prefill_cuda_ms:.9f}",
                    "total_cuda_ms": f"{total_cuda_ms:.9f}",
                    "decode_cuda_ms_per_token": f"{decode_ms_per_token:.9f}",
                    "total_wall_ms": f"{total_wall_ms:.9f}",
                    "throughput_output_tokens_per_s": f"{throughput:.9f}",
                    "kv_cache_mb": f"{cache_mb:.9f}",
                    "peak_allocated_gb": f"{peak_allocated_gb:.9f}",
                    "peak_reserved_gb": f"{peak_reserved_gb:.9f}",
                    "layer_prefill_tokens": ",".join(str(x) for x in layer_tokens),
                    "pruner_stats": json.dumps(
                        pruner_stats, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
            handle.flush()
            measured += 1
            progress.set_postfix(
                prefill=f"{prefill_cuda_ms:.1f}ms",
                total=f"{total_cuda_ms:.1f}ms",
                visual=f"{avg_visual_tokens:.2f}",
            )
            del total_output, prepared
    finally:
        handle.close()
        adapter.close()

    final_done = completed_indices(output)
    if final_done != requested_indices:
        missing = sorted(requested_indices - final_done)
        raise RuntimeError(f"Incomplete output: {len(missing)} rows missing, first={missing[:10]}")
    print(f"Completed {len(final_done)} samples: {output}", flush=True)


if __name__ == "__main__":
    main()
