#!/usr/bin/env python3
"""Profile LLaVA-1.5 vanilla vs. learnable visual-token pruning on GQA.

The benchmark times model.forward calls with CUDA events.  For every generate()
call, the first forward is prefill and all remaining cached forwards are decode.
Image loading/preprocessing, prompt tokenization, and CPU overhead are excluded.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/data1/chenzixuan/model/liuhaotian/llava-v1.5-7b"
DEFAULT_RUN = "/data1/chenzixuan/train_output/official_llava_learnable_prune_lightweight_top64_layer18_sample0.2"


@dataclass
class SampleTiming:
    sample_index: int
    question_id: str
    prefill_ms: float
    decode_ms: float
    decode_steps: int


class ForwardTimer:
    """Record top-level forwards without changing the signature Transformers inspects."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._pending: list[torch.cuda.Event] = []
        self._handles: list[Any] = []

    def __enter__(self) -> "ForwardTimer":
        def before_forward(_model: torch.nn.Module, _args: tuple[Any, ...]) -> None:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._pending.append(start)

        def after_forward(_model: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
            if not self._pending:
                raise RuntimeError("Forward timing hook lost its matching start event")
            start = self._pending.pop()
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.events.append((start, end))
            return output

        self._handles = [
            self.model.register_forward_pre_hook(before_forward),
            self.model.register_forward_hook(after_forward),
        ]
        return self

    def __exit__(self, *_: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def elapsed_ms(self) -> list[float]:
        torch.cuda.synchronize()
        return [float(start.elapsed_time(end)) for start, end in self.events]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LLaVA-1.5 vanilla and learnable-pruned prefill/decode speed on GQA."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--prune-run", default=DEFAULT_RUN,
                        help="Training run directory or a directory containing learnable_prune_config.pt.")
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit predictor checkpoint; overrides --prune-run auto-discovery.")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--attn-anchor", choices=("query", "last"), default="query")
    parser.add_argument("--avg-token-budget", type=int, choices=(64, 128, 192), default=192)
    parser.add_argument("--scope-target-count", type=int, default=None)
    parser.add_argument("--mid-pruning-layer-idx", type=int, default=12)
    parser.add_argument("--mid-target-count", type=int, default=None)
    parser.add_argument("--final-wipe-layer-idx", type=int, default=25)
    parser.add_argument("--output", default=str(REPO_ROOT / "inference" / "profile_speed_up_results.json"))
    parser.add_argument("--hf-token", default=None,
                        help="Hugging Face token. If omitted, use the locally configured token.")
    return parser.parse_args()


def resolve_checkpoint(run: str, explicit: str | None) -> Path:
    if explicit:
        checkpoint = Path(explicit).expanduser()
        if not (checkpoint / "learnable_prune_config.pt").is_file():
            raise FileNotFoundError(f"Invalid predictor checkpoint: {checkpoint}")
        return checkpoint
    root = Path(run).expanduser()
    if root.name.startswith("checkpoint-") and (root / "learnable_prune_config.pt").is_file():
        return root
    candidates = [p for p in root.glob("checkpoint-*") if (p / "learnable_prune_config.pt").is_file()]
    if candidates:
        return max(candidates, key=lambda p: int(p.name.rsplit("-", 1)[-1]))
    if (root / "learnable_prune_config.pt").is_file():
        return root
    raise FileNotFoundError(f"No checkpoint containing learnable_prune_config.pt under {root}")


def configure_pruning(args: argparse.Namespace, checkpoint: Path) -> None:
    scope_defaults = {64: 137, 128: 272, 192: 408}
    mid_defaults = {64: 32, 128: 64, 192: 96}
    values = {
        "LEARNABLE_PRUNE_CHECKPOINT": str(checkpoint),
        "LEARNABLE_TOPK": "64",
        "ENABLE_PREDICTOR": "1",
        "ENABLE_FINALWIPE": "1",
        "SCOPE_TARGET_COUNT": str(args.scope_target_count or scope_defaults[args.avg_token_budget]),
        "MID_PRUNING_LAYER_IDX": str(args.mid_pruning_layer_idx),
        "MID_TARGET_COUNT": str(args.mid_target_count or mid_defaults[args.avg_token_budget]),
        "FINAL_WIPE_LAYER_IDX": str(args.final_wipe_layer_idx),
    }
    os.environ.update(values)


def load_gqa_sample(num_samples: int, seed: int, token: str | None) -> list[dict[str, Any]]:
    token_arg: Any = token if token else True
    instructions = load_dataset(
        "lmms-lab/GQA", "testdev_balanced_instructions", split="testdev", token=token_arg
    )
    if num_samples > len(instructions):
        raise ValueError(f"Requested {num_samples} samples, but GQA testdev has {len(instructions)}")
    indices = random.Random(seed).sample(range(len(instructions)), num_samples)
    selected = [instructions[i] for i in indices]

    images = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", token=token_arg)
    # Reading the id column does not decode every PIL image. Decode only the 200 selected rows.
    image_to_row = {str(image_id): i for i, image_id in enumerate(images["id"])}
    result = []
    for dataset_index, doc in zip(indices, selected):
        image_id = str(doc["imageId"])
        if image_id not in image_to_row:
            raise KeyError(f"GQA image {image_id!r} was not found in the image dataset")
        result.append({
            "dataset_index": dataset_index,
            "question_id": str(doc.get("questionId", doc.get("id", dataset_index))),
            "question": doc["question"],
            "image": images[image_to_row[image_id]]["image"].convert("RGB"),
        })
    return result


def load_model(args: argparse.Namespace, pruned: bool):
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    load_kwargs: dict[str, Any] = {
        "device_map": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
    }
    if pruned:
        load_kwargs["learnable_prune_lightweight_scope_finalwipe_model"] = True
    tokenizer, model, processor, _ = load_pretrained_model(
        args.model_path, None, get_model_name_from_path(args.model_path), **load_kwargs
    )
    model.eval()
    return tokenizer, model, processor


def prepare_sample(sample: dict[str, Any], tokenizer: Any, processor: Any, config: Any, device: torch.device):
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    question = sample["question"] + "\nAnswer the question using a single word or phrase."
    if getattr(config, "mm_use_im_start_end", False):
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        question = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    image_tensor = process_images([sample["image"]], processor, config).to(device=device)
    return input_ids, image_tensor, [sample["image"].size]


def benchmark_model(args: argparse.Namespace, samples: list[dict[str, Any]], pruned: bool) -> list[SampleTiming]:
    tokenizer, model, processor = load_model(args, pruned)
    device = torch.device(args.device)
    image_dtype = next(model.parameters()).dtype

    def prepare(sample: dict[str, Any]):
        input_ids, image_tensor, image_sizes = prepare_sample(sample, tokenizer, processor, model.config, device)
        return input_ids, image_tensor.to(dtype=image_dtype), image_sizes

    generation_kwargs = dict(
        do_sample=False, temperature=0.0, top_p=1.0, num_beams=1,
        max_new_tokens=args.max_new_tokens, use_cache=True,
    )
    if pruned:
        generation_kwargs["attn_anchor"] = args.attn_anchor

    warmup_samples = samples[: min(args.warmup, len(samples))]
    with torch.inference_mode():
        for sample in tqdm(warmup_samples, desc=f"Warmup ({'pruned' if pruned else 'vanilla'})"):
            input_ids, images, image_sizes = prepare(sample)
            model.generate(input_ids, images=images, image_sizes=image_sizes, **generation_kwargs)
    torch.cuda.synchronize()

    timings: list[SampleTiming] = []
    with torch.inference_mode():
        for sample in tqdm(samples, desc=f"Profile ({'pruned' if pruned else 'vanilla'})"):
            input_ids, images, image_sizes = prepare(sample)
            with ForwardTimer(model) as timer:
                model.generate(input_ids, images=images, image_sizes=image_sizes, **generation_kwargs)
            calls = timer.elapsed_ms()
            if not calls:
                raise RuntimeError("generate() made no model.forward calls")
            timings.append(SampleTiming(
                sample_index=int(sample["dataset_index"]),
                question_id=sample["question_id"],
                prefill_ms=calls[0],
                decode_ms=sum(calls[1:]),
                decode_steps=len(calls) - 1,
            ))

    del model, tokenizer, processor
    gc.collect()
    torch.cuda.empty_cache()
    return timings


def aggregate(rows: list[SampleTiming]) -> dict[str, float | int]:
    prefill_ms = sum(x.prefill_ms for x in rows)
    decode_ms = sum(x.decode_ms for x in rows)
    decode_steps = sum(x.decode_steps for x in rows)
    return {
        "samples": len(rows),
        "prefill_total_ms": prefill_ms,
        "prefill_mean_ms": prefill_ms / len(rows),
        "decode_total_ms": decode_ms,
        "decode_steps": decode_steps,
        "decode_mean_ms_per_token": decode_ms / decode_steps if decode_steps else float("nan"),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This profiler requires CUDA because it uses CUDA events")
    if not str(args.device).startswith("cuda"):
        raise ValueError("--device must be a CUDA device")
    sys.path.insert(0, str(REPO_ROOT))
    checkpoint = resolve_checkpoint(args.prune_run, args.checkpoint)
    configure_pruning(args, checkpoint)
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Sampling {args.num_samples} GQA examples (seed={args.seed})", flush=True)
    samples = load_gqa_sample(args.num_samples, args.seed, args.hf_token)
    # Run vanilla first so importing the custom model happens only after all pruning env vars are set.
    vanilla = benchmark_model(args, samples, pruned=False)
    pruned = benchmark_model(args, samples, pruned=True)
    vanilla_summary, pruned_summary = aggregate(vanilla), aggregate(pruned)
    speedup = {
        "prefill": vanilla_summary["prefill_mean_ms"] / pruned_summary["prefill_mean_ms"],
        "decode": vanilla_summary["decode_mean_ms_per_token"] / pruned_summary["decode_mean_ms_per_token"],
    }
    payload = {
        "config": {**vars(args), "checkpoint": str(checkpoint)},
        "vanilla": vanilla_summary,
        "pruned": pruned_summary,
        "speedup": speedup,
        "per_sample": {
            "vanilla": [asdict(x) for x in vanilla],
            "pruned": [asdict(x) for x in pruned],
        },
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print("\nResults (vanilla / pruned)")
    print(f"Prefill: {vanilla_summary['prefill_mean_ms']:.3f} / {pruned_summary['prefill_mean_ms']:.3f} ms, speedup {speedup['prefill']:.3f}x")
    print(f"Decode:  {vanilla_summary['decode_mean_ms_per_token']:.3f} / {pruned_summary['decode_mean_ms_per_token']:.3f} ms/token, speedup {speedup['decode']:.3f}x")
    print(f"Saved detailed results to {output}")


if __name__ == "__main__":
    main()
