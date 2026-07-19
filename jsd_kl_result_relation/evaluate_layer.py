#!/usr/bin/env python3
"""Evaluate scoring layers and record full-vs-top-k decoding divergence."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence

import torch
from datasets import load_dataset
from PIL import ImageDraw
from tqdm import tqdm
from transformers.cache_utils import Cache, StaticCache

from jsd_kl_result_relation.metrics import (
    DIVERGENCE_METRICS,
    distribution_divergence_tensors,
    mean_finite,
)
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model


DATASETS = {
    "refcoco": "lmms-lab/RefCOCO",
    "refcoco_plus": "lmms-lab/RefCOCOplus",
}
PROMPT = "Provide a short description for this region."
DEFAULT_CAPTION_METRICS = (
    "Bleu_4",
    "Bleu_3",
    "Bleu_2",
    "Bleu_1",
    "METEOR",
    "ROUGE_L",
    "CIDEr",
)


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _answer_list(answer: Any) -> list[str]:
    if isinstance(answer, (list, tuple)):
        return [str(item) for item in answer]
    return [str(answer)]


def sampled_indices(dataset_size: int, sample_size: int, seed: int) -> list[int]:
    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size}")
    if dataset_size < sample_size:
        raise ValueError(
            f"Requested exactly {sample_size} samples, but dataset only has {dataset_size}"
        )
    rng = random.Random(seed)
    indices = list(range(dataset_size))
    rng.shuffle(indices)
    return sorted(indices[:sample_size])


def parse_layers(spec: str) -> list[int]:
    layers: list[int] = []
    for item in spec.replace(" ", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            layers.extend(range(start, end + step, step))
        else:
            layers.append(int(item))
    return list(dict.fromkeys(layers))


def _write_or_validate_sample_manifest(
    path: Path, indices: Sequence[int], config: dict[str, Any]
) -> None:
    payload = {"config": config, "indices": [int(index) for index in indices]}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"Existing sample manifest does not match this run: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_records(path: Path, expected: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for key, value in expected.items():
                if record.get(key) != value:
                    raise ValueError(
                        f"Resume record mismatch at {path}:{line_number}: {key}={record.get(key)!r}, expected {value!r}"
                    )
            dataset_index = int(record["dataset_index"])
            if dataset_index in records:
                raise ValueError(f"Duplicate dataset_index={dataset_index} in {path}")
            records[dataset_index] = record
    return records


def _boxed_image(doc: dict[str, Any]):
    image = doc["image"].convert("RGB").copy()
    x, y, width, height = [float(value) for value in doc["bbox"]]
    ImageDraw.Draw(image).rectangle([x, y, x + width, y + height], outline="red")
    return image


def _build_prompt_input(
    tokenizer, conv_template: str, device: torch.device
) -> torch.Tensor:
    conversation = conv_templates[conv_template].copy()
    conversation.append_message(
        conversation.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n{PROMPT}"
    )
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    return (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )


def _vision_dtype(model) -> torch.dtype:
    tower = model.get_vision_tower()
    try:
        return next(tower.parameters()).dtype
    except (StopIteration, AttributeError):
        return next(model.parameters()).dtype


def _prepare_image(model, image_processor, image, device: torch.device):
    image_tensor = process_images([image], image_processor, model.config)
    dtype = _vision_dtype(model)
    if isinstance(image_tensor, list):
        return [item.to(device=device, dtype=dtype) for item in image_tensor]
    return image_tensor.to(device=device, dtype=dtype)


def score_and_prune_many(
    model,
    expanded_input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    scoring_layers: Sequence[int],
    topk: int,
) -> dict[int, dict[str, Any]]:
    """Collect requested layer scores in one prefix pass and apply direct top-k."""

    visual_positions = (
        (expanded_input_ids[0].eq(IMAGE_TOKEN_INDEX) & attention_mask[0].bool())
        .nonzero(as_tuple=False)
        .flatten()
    )
    if visual_positions.numel() == 0:
        raise ValueError("Expanded multimodal sequence contains no visual tokens")
    requested = sorted(set(int(layer) for layer in scoring_layers))
    if not requested:
        return {}
    query_indices = model._build_query_indices(
        expanded_input_ids, attention_mask, visual_positions
    )
    hidden_states = inputs_embeds
    cache_position = torch.arange(
        hidden_states.shape[1], device=hidden_states.device, dtype=torch.long
    )
    causal_mask = model._prepare_mask(
        attention_mask,
        hidden_states,
        position_ids=position_ids,
        cache_position=cache_position,
    )
    position_embeddings = None
    if hasattr(model.model, "rotary_emb"):
        try:
            position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
        except TypeError:
            position_embeddings = None

    requested_set = set(requested)
    last_requested_layer = requested[-1]
    pruning_by_layer: dict[int, dict[str, Any]] = {}
    for layer_idx in range(last_requested_layer + 1):
        if layer_idx in requested_set:
            scores = model._visual_token_scores(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                scoring_layer_idx=layer_idx,
                visual_positions=visual_positions,
                query_indices=query_indices,
            )
            keep_positions = model._topk_keep_positions(
                expanded_input_ids,
                attention_mask,
                scores,
                topk,
            )
            pruning_by_layer[layer_idx] = {
                "pruned_embeds": inputs_embeds.index_select(1, keep_positions),
                "pruned_position_ids": position_ids.index_select(1, keep_positions),
                "original_visual_tokens": int(visual_positions.numel()),
                "kept_visual_tokens": int(min(max(topk, 1), visual_positions.numel())),
                "query_tokens": int(query_indices.numel()),
            }
        if layer_idx < last_requested_layer:
            hidden_states, _, _ = model._run_layer(
                model.model.layers[layer_idx],
                hidden_states,
                causal_mask,
                position_ids,
                use_cache=False,
                output_attentions=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
    return pruning_by_layer


def score_and_prune(
    model,
    expanded_input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    scoring_layer: int,
    topk: int,
) -> dict[str, Any]:
    """Backward-compatible one-layer wrapper around :func:`score_and_prune_many`."""

    return score_and_prune_many(
        model=model,
        expanded_input_ids=expanded_input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        scoring_layers=[scoring_layer],
        topk=topk,
    )[int(scoring_layer)]


def _eos_ids(tokenizer) -> set[int]:
    value = tokenizer.eos_token_id
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    return {int(value)}


def _copy_to_static_cache(
    model, source_cache: Cache, max_cache_len: int
) -> StaticCache:
    """Fork a populated dynamic cache once into preallocated copy-on-write storage."""

    if max_cache_len < int(source_cache.get_seq_length()):
        raise ValueError(
            f"Static cache length {max_cache_len} is smaller than source length {source_cache.get_seq_length()}"
        )
    target = StaticCache(config=model.config, max_cache_len=max_cache_len)
    source_layers = getattr(source_cache, "layers", None)
    target_layers = getattr(target, "layers", None)
    if (
        source_layers is None
        or target_layers is None
        or len(source_layers) != len(target_layers)
    ):
        raise TypeError(f"Unsupported cache structure: {type(source_cache).__name__}")
    for source_layer, target_layer in zip(source_layers, target_layers):
        if not getattr(source_layer, "is_initialized", False):
            continue
        target_layer.lazy_initialization(source_layer.keys)
        sequence_length = int(source_layer.keys.shape[-2])
        target_layer.keys[:, :, :sequence_length].copy_(source_layer.keys)
        target_layer.values[:, :, :sequence_length].copy_(source_layer.values)
    return target


@torch.inference_mode()
def full_reference_prefill(
    model,
    full_embeds: torch.Tensor,
    full_attention_mask: torch.Tensor,
    full_position_ids: torch.Tensor,
) -> tuple[torch.Tensor, Cache]:
    """Compute the layer-independent unpruned logits/cache once per sample."""

    full_length = int(full_embeds.shape[1])
    hidden_states, cache, _, _ = model._manual_decode(
        inputs_embeds=full_embeds,
        attention_mask=full_attention_mask,
        position_ids=full_position_ids,
        cache_position=torch.arange(
            full_length, device=full_embeds.device, dtype=torch.long
        ),
        use_cache=True,
    )
    if not isinstance(cache, Cache):
        raise TypeError(f"Expected a transformers Cache, got {type(cache).__name__}")
    return model.lm_head(hidden_states[:, -1, :]), cache


@torch.inference_mode()
def paired_greedy_generate(
    model,
    tokenizer,
    full_logits: torch.Tensor,
    full_cache: Cache,
    full_length: int,
    full_attention_mask: torch.Tensor,
    next_semantic_position: int,
    pruned_embeds: torch.Tensor,
    pruned_position_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[list[int], dict[str, Any]]:
    """Decode on the pruned trajectory and compare aligned next-token distributions.

    Both branches receive the token selected by the pruned branch at every step.  The
    comparison therefore isolates visual-token pruning under an identical text prefix.
    """

    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    device = pruned_embeds.device
    pruned_length = int(pruned_embeds.shape[1])
    full_static_cache = _copy_to_static_cache(
        model,
        full_cache,
        max_cache_len=full_length + max_new_tokens,
    )
    pruned_static_cache = StaticCache(
        config=model.config,
        max_cache_len=pruned_length + max_new_tokens,
    )
    full_decode_attention = torch.ones(
        (1, full_length + max_new_tokens),
        dtype=full_attention_mask.dtype,
        device=device,
    )
    full_decode_attention[:, :full_length].copy_(full_attention_mask)
    pruned_decode_attention = torch.ones(
        (1, pruned_length + max_new_tokens),
        dtype=full_attention_mask.dtype,
        device=device,
    )
    pruned_hidden, pruned_static_cache, _, _ = model._manual_decode(
        inputs_embeds=pruned_embeds,
        attention_mask=pruned_decode_attention[:, :pruned_length],
        position_ids=pruned_position_ids,
        past_key_values=pruned_static_cache,
        cache_position=torch.arange(pruned_length, device=device, dtype=torch.long),
        use_cache=True,
    )
    pruned_logits = model.lm_head(pruned_hidden[:, -1, :])
    eos_ids = _eos_ids(tokenizer)
    generated: list[int] = []
    step_metrics: list[dict[str, torch.Tensor]] = []
    semantic_positions = torch.arange(
        next_semantic_position,
        next_semantic_position + max_new_tokens,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    full_cache_positions = torch.arange(
        full_length,
        full_length + max_new_tokens,
        dtype=torch.long,
        device=device,
    )
    pruned_cache_positions = torch.arange(
        pruned_length,
        pruned_length + max_new_tokens,
        dtype=torch.long,
        device=device,
    )

    for step in range(max_new_tokens):
        step_metrics.append(distribution_divergence_tensors(full_logits, pruned_logits))
        next_token_tensor = torch.argmax(pruned_logits, dim=-1)
        next_token = int(next_token_tensor.item())
        generated.append(next_token)
        if next_token in eos_ids or step == max_new_tokens - 1:
            break

        token_embed = model.get_model().embed_tokens(next_token_tensor.view(1, 1))
        semantic_position = semantic_positions[:, step : step + 1]
        full_hidden, full_static_cache, _, _ = model._manual_decode(
            inputs_embeds=token_embed,
            attention_mask=full_decode_attention[:, : full_length + step + 1],
            position_ids=semantic_position,
            past_key_values=full_static_cache,
            cache_position=full_cache_positions[step : step + 1],
            use_cache=True,
        )
        pruned_hidden, pruned_static_cache, _, _ = model._manual_decode(
            inputs_embeds=token_embed,
            attention_mask=pruned_decode_attention[:, : pruned_length + step + 1],
            position_ids=semantic_position,
            past_key_values=pruned_static_cache,
            cache_position=pruned_cache_positions[step : step + 1],
            use_cache=True,
        )
        full_logits = model.lm_head(full_hidden[:, -1, :])
        pruned_logits = model.lm_head(pruned_hidden[:, -1, :])

    jsd = torch.stack([row["jsd"] for row in step_metrics])
    kl_full_to_pruned = torch.stack([row["kl_full_to_pruned"] for row in step_metrics])
    kl_pruned_to_full = torch.stack([row["kl_pruned_to_full"] for row in step_metrics])
    top1_match = torch.stack(
        [row["top1_match"].to(dtype=torch.float32) for row in step_metrics]
    )
    packed = (
        torch.stack(
            [
                jsd[0],
                kl_full_to_pruned[0],
                kl_pruned_to_full[0],
                top1_match[0],
                jsd.mean(),
                kl_full_to_pruned.mean(),
                kl_pruned_to_full.mean(),
                top1_match.mean(),
            ]
        )
        .cpu()
        .tolist()
    )
    metrics = {
        "first_token_jsd": float(packed[0]),
        "first_token_kl_full_to_pruned": float(packed[1]),
        "first_token_kl_pruned_to_full": float(packed[2]),
        "first_token_top1_match": bool(packed[3]),
        "generation_mean_jsd": float(packed[4]),
        "generation_mean_kl_full_to_pruned": float(packed[5]),
        "generation_mean_kl_pruned_to_full": float(packed[6]),
        "generation_top1_match_rate": float(packed[7]),
        "num_compared_steps": int(len(step_metrics)),
    }
    return generated, metrics


def _caption_performance_many(
    records_by_layer: dict[int, Sequence[dict[str, Any]]],
    requested_metrics: Sequence[str],
) -> dict[int, dict[str, float]]:
    """Score every layer with one shared PTB-tokenization/scorer setup."""

    from pycocoevalcap.eval import Bleu, Cider, Meteor, Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    unsupported = sorted(set(requested_metrics) - set(DEFAULT_CAPTION_METRICS))
    if unsupported:
        raise ValueError(
            f"Unsupported caption metrics {unsupported}; supported={DEFAULT_CAPTION_METRICS}"
        )
    if not records_by_layer:
        return {}

    layers = list(records_by_layer)
    reference_records = list(records_by_layer[layers[0]])
    num_records = len(reference_records)
    ground_truth_raw = {
        image_id: [{"caption": answer} for answer in record["answer"]]
        for image_id, record in enumerate(reference_records)
    }
    generated_raw: dict[int, list[dict[str, str]]] = {}
    for layer_offset, layer in enumerate(layers):
        records = list(records_by_layer[layer])
        if len(records) != num_records:
            raise ValueError(
                f"Layer {layer} has {len(records)} caption records; expected {num_records}"
            )
        for image_id, (reference, record) in enumerate(zip(reference_records, records)):
            if record["dataset_index"] != reference["dataset_index"]:
                raise ValueError(
                    f"Layer {layer} has a different dataset index at record {image_id}"
                )
            if record["answer"] != reference["answer"]:
                raise ValueError(
                    f"Layer {layer} has different references at record {image_id}"
                )
            generated_raw[layer_offset * num_records + image_id] = [
                {"caption": record["prediction"]}
            ]

    tokenizer = PTBTokenizer()
    ground_truth = tokenizer.tokenize(ground_truth_raw)
    generated_all = tokenizer.tokenize(generated_raw)

    bleu_metrics = tuple(
        metric for metric in requested_metrics if metric.startswith("Bleu_")
    )
    scorers = {
        "METEOR": Meteor() if "METEOR" in requested_metrics else None,
        "ROUGE_L": Rouge() if "ROUGE_L" in requested_metrics else None,
        "CIDEr": Cider() if "CIDEr" in requested_metrics else None,
    }
    bleu_scorer = Bleu(4) if bleu_metrics else None
    output: dict[int, dict[str, float]] = {}
    for layer_offset, layer in enumerate(layers):
        generated = {
            image_id: generated_all[layer_offset * num_records + image_id]
            for image_id in range(num_records)
        }
        scores: dict[str, float] = {}
        if bleu_scorer is not None:
            bleu_score, _ = bleu_scorer.compute_score(ground_truth, generated)
            for metric in bleu_metrics:
                scores[metric] = float(bleu_score[int(metric.split("_")[-1]) - 1])
        for metric, scorer in scorers.items():
            if scorer is not None:
                score, _ = scorer.compute_score(ground_truth, generated)
                scores[metric] = float(score)
        output[layer] = {metric: scores[metric] for metric in requested_metrics}
    return output


def _caption_performance(
    records: Sequence[dict[str, Any]],
    requested_metrics: Sequence[str],
) -> dict[str, float]:
    """Backward-compatible single-layer caption aggregation wrapper."""

    return _caption_performance_many({0: records}, requested_metrics)[0]


def _summary(
    records: Sequence[dict[str, Any]],
    config: dict[str, Any],
    indices: Sequence[int],
    performance: dict[str, float],
) -> dict[str, Any]:
    divergence = {
        metric: mean_finite(record.get(metric) for record in records)
        for metric in DIVERGENCE_METRICS
    }
    divergence.update(
        {
            "first_token_top1_match_rate": mean_finite(
                1.0 if record.get("first_token_top1_match") else 0.0
                for record in records
            ),
            "generation_top1_match_rate": mean_finite(
                record.get("generation_top1_match_rate") for record in records
            ),
            "mean_compared_steps": mean_finite(
                record.get("num_compared_steps") for record in records
            ),
        }
    )
    index_bytes = ",".join(str(index) for index in indices).encode("utf-8")
    return {
        "config": config,
        "num_records": int(len(records)),
        "sample_indices_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "mean_divergence": divergence,
        "performance": performance,
    }


def _print_cuda_memory(device: torch.device) -> None:
    gib = float(1024**3)
    print(
        "CUDA_MEMORY "
        f"device={device} "
        f"allocated_gib={torch.cuda.memory_allocated(device) / gib:.3f} "
        f"peak_allocated_gib={torch.cuda.max_memory_allocated(device) / gib:.3f} "
        f"reserved_gib={torch.cuda.memory_reserved(device) / gib:.3f} "
        f"peak_reserved_gib={torch.cuda.max_memory_reserved(device) / gib:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--task", required=True, choices=sorted(DATASETS))
    parser.add_argument("--dataset-split", default="val")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--sample-shard-index", type=int, default=0)
    parser.add_argument("--sample-shard-count", type=int, default=1)
    layer_group = parser.add_mutually_exclusive_group(required=True)
    layer_group.add_argument("--scoring-layer", type=int)
    layer_group.add_argument(
        "--scoring-layers", help="Comma/space list with optional ranges, e.g. 0-31"
    )
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--caption-metrics", default=",".join(DEFAULT_CAPTION_METRICS))
    parser.add_argument("--conv-template", default="vicuna_v1")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32")
    )
    parser.add_argument("--records-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--sample-indices-json", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    if args.topk <= 0:
        parser.error("--topk must be positive")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.sample_shard_count <= 0:
        parser.error("--sample-shard-count must be positive")
    if not 0 <= args.sample_shard_index < args.sample_shard_count:
        parser.error("--sample-shard-index must be in [0, --sample-shard-count)")
    scoring_layers = (
        [int(args.scoring_layer)]
        if args.scoring_layer is not None
        else parse_layers(str(args.scoring_layers))
    )
    if not scoring_layers or any(layer < 0 for layer in scoring_layers):
        parser.error("scoring layers must contain non-negative integers")
    if args.scoring_layers is not None:
        if args.output_root is None:
            parser.error("--output-root is required with --scoring-layers")
        if any(
            path is not None
            for path in (
                args.records_jsonl,
                args.summary_json,
                args.sample_indices_json,
            )
        ):
            parser.error(
                "per-layer output paths cannot be combined with --scoring-layers"
            )
        shard_suffix = (
            ""
            if args.sample_shard_count == 1
            else f".shard_{args.sample_shard_index + 1}_of_{args.sample_shard_count}"
        )
        layer_paths = {
            layer: {
                "records": args.output_root
                / f"scoring_layer_{layer}"
                / f"records{shard_suffix}.jsonl",
                "summary": args.output_root
                / f"scoring_layer_{layer}"
                / "layer_summary.json",
                "indices": args.output_root
                / f"scoring_layer_{layer}"
                / f"sample_indices{shard_suffix}.json",
            }
            for layer in scoring_layers
        }
    else:
        if args.sample_shard_count != 1:
            parser.error("sample sharding requires --scoring-layers and --output-root")
        if any(
            path is None
            for path in (
                args.records_jsonl,
                args.summary_json,
                args.sample_indices_json,
            )
        ):
            parser.error(
                "--records-jsonl, --summary-json and --sample-indices-json are required with --scoring-layer"
            )
        layer_paths = {
            scoring_layers[0]: {
                "records": args.records_jsonl,
                "summary": args.summary_json,
                "indices": args.sample_indices_json,
            }
        }
    caption_metrics = [
        item.strip() for item in args.caption_metrics.split(",") if item.strip()
    ]
    if not caption_metrics:
        parser.error("--caption-metrics cannot be empty")
    unsupported_caption_metrics = sorted(
        set(caption_metrics) - set(DEFAULT_CAPTION_METRICS)
    )
    if unsupported_caption_metrics:
        parser.error(
            f"unsupported caption metrics {unsupported_caption_metrics}; "
            f"supported={DEFAULT_CAPTION_METRICS}"
        )

    dataset_path = DATASETS[args.task]
    dataset = load_dataset(dataset_path, split=args.dataset_split)
    all_indices = sampled_indices(len(dataset), args.sample_size, args.sample_seed)
    worker_indices = all_indices[args.sample_shard_index :: args.sample_shard_count]
    dataset_fingerprint = str(getattr(dataset, "_fingerprint", ""))
    configs: dict[int, dict[str, Any]] = {}
    expected_record_configs: dict[int, dict[str, Any]] = {}
    processed_by_layer: dict[int, dict[int, dict[str, Any]]] = {}
    for layer in scoring_layers:
        config = {
            "model": args.model_name,
            "model_path": str(Path(args.model_path).resolve()),
            "task": args.task,
            "dataset_path": dataset_path,
            "dataset_split": args.dataset_split,
            "dataset_fingerprint": dataset_fingerprint,
            "sample_size": int(args.sample_size),
            "sample_seed": int(args.sample_seed),
            "scoring_layer": int(layer),
            "topk": int(args.topk),
            "max_new_tokens": int(args.max_new_tokens),
            "caption_metrics": caption_metrics,
            "conv_template": args.conv_template,
            "dtype": args.dtype,
            "generation": "greedy_pruned_trajectory",
            "distribution_reference": "unpruned_with_shared_pruned_text_prefix",
            "kl_direction_primary": "full_to_pruned",
        }
        config_fingerprint = hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        expected_record_config = {
            "model": args.model_name,
            "task": args.task,
            "scoring_layer": int(layer),
            "topk": int(args.topk),
            "config_sha256": config_fingerprint,
        }
        paths = layer_paths[layer]
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        _write_or_validate_sample_manifest(paths["indices"], worker_indices, config)
        processed = _load_records(paths["records"], expected_record_config)
        unexpected = sorted(set(processed) - set(worker_indices))
        if unexpected:
            raise ValueError(
                f"Layer {layer} records contain indices outside the sample manifest: {unexpected[:10]}"
            )
        configs[layer] = config
        expected_record_configs[layer] = expected_record_config
        processed_by_layer[layer] = processed

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        None,
        get_model_name_from_path(args.model_path),
        device_map="cuda",
        best_layer_sweep_model=True,
        dtype=args.dtype,
        attn_implementation="eager",
    )
    model.eval()
    invalid_layers = [
        layer for layer in scoring_layers if not 0 <= layer < len(model.model.layers)
    ]
    if invalid_layers:
        raise ValueError(
            f"Layers {invalid_layers} are outside [0, {len(model.model.layers) - 1}]"
        )
    device = next(model.parameters()).device
    input_ids = _build_prompt_input(tokenizer, args.conv_template, device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with ExitStack() as stack:
        record_handles = {
            layer: stack.enter_context(
                layer_paths[layer]["records"].open("a", encoding="utf-8", buffering=1)
            )
            for layer in scoring_layers
        }
        for dataset_index in tqdm(
            worker_indices,
            desc=(
                f"{args.task} shard={args.sample_shard_index + 1}/{args.sample_shard_count} "
                f"layers={min(scoring_layers)}-{max(scoring_layers)} topk={args.topk}"
            ),
        ):
            pending_layers = [
                layer
                for layer in scoring_layers
                if dataset_index not in processed_by_layer[layer]
            ]
            if not pending_layers:
                continue
            doc = dataset[int(dataset_index)]
            image = _boxed_image(doc)
            image_tensor = _prepare_image(model, image_processor, image, device)

            with torch.inference_mode():
                inputs_embeds, expanded_ids, expanded_mask, position_ids = (
                    model._embed_multimodal_for_generation(
                        input_ids,
                        image_tensor,
                        attention_mask,
                        None,
                        image_sizes=[image.size],
                    )
                )
                pruning_by_layer = score_and_prune_many(
                    model=model,
                    expanded_input_ids=expanded_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=expanded_mask,
                    position_ids=position_ids,
                    scoring_layers=pending_layers,
                    topk=args.topk,
                )
                full_logits, full_cache = full_reference_prefill(
                    model=model,
                    full_embeds=inputs_embeds,
                    full_attention_mask=expanded_mask,
                    full_position_ids=position_ids,
                )
                full_length = int(inputs_embeds.shape[1])
                next_semantic_position = int(position_ids.max().item()) + 1

                for layer in pending_layers:
                    pruning = pruning_by_layer[layer]
                    generated_ids, divergence = paired_greedy_generate(
                        model=model,
                        tokenizer=tokenizer,
                        full_logits=full_logits,
                        full_cache=full_cache,
                        full_length=full_length,
                        full_attention_mask=expanded_mask,
                        next_semantic_position=next_semantic_position,
                        pruned_embeds=pruning["pruned_embeds"],
                        pruned_position_ids=pruning["pruned_position_ids"],
                        max_new_tokens=args.max_new_tokens,
                    )
                    prediction = tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()
                    record: dict[str, Any] = {
                        **expected_record_configs[layer],
                        "dataset_index": int(dataset_index),
                        "question_id": _json_scalar(
                            doc.get("question_id", dataset_index)
                        ),
                        "answer": _answer_list(doc.get("answer", [])),
                        "prediction": prediction,
                        "generated_token_ids": generated_ids,
                        "original_visual_tokens": pruning["original_visual_tokens"],
                        "kept_visual_tokens": pruning["kept_visual_tokens"],
                        "original_sequence_tokens": full_length,
                        "pruned_sequence_tokens": int(
                            pruning["pruned_embeds"].shape[1]
                        ),
                        "query_tokens": pruning["query_tokens"],
                        **divergence,
                    }
                    record_handles[layer].write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                    processed_by_layer[layer][dataset_index] = record

            del inputs_embeds, image_tensor, pruning_by_layer, full_cache, full_logits

    ordered_records_by_layer = {
        layer: [
            processed_by_layer[layer][index]
            for index in worker_indices
            if index in processed_by_layer[layer]
        ]
        for layer in scoring_layers
    }
    for layer, ordered_records in ordered_records_by_layer.items():
        if len(ordered_records) != len(worker_indices):
            raise RuntimeError(
                f"Layer {layer}: expected {len(worker_indices)} shard records, found {len(ordered_records)}"
            )
    if args.sample_shard_count > 1:
        print(
            f"SHARD_COMPLETE model={args.model_name} task={args.task} "
            f"shard={args.sample_shard_index + 1}/{args.sample_shard_count} "
            f"records_per_layer={len(worker_indices)} layers={len(scoring_layers)}"
        )
        if device.type == "cuda":
            _print_cuda_memory(device)
        return

    performance_by_layer = _caption_performance_many(
        ordered_records_by_layer,
        caption_metrics,
    )
    for layer, ordered_records in ordered_records_by_layer.items():
        summary = _summary(
            ordered_records,
            configs[layer],
            all_indices,
            performance_by_layer[layer],
        )
        summary_path = layer_paths[layer]["summary"]
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"COMPLETE model={args.model_name} task={args.task} layer={layer} "
            f"records={len(ordered_records)} summary={summary_path}"
        )
    if device.type == "cuda":
        _print_cuda_memory(device)


if __name__ == "__main__":
    main()
