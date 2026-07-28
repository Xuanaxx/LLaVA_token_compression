#!/usr/bin/env python3
"""Validate and summarize the three-arm formal efficiency-v2 experiment.

The formal table contains exactly Vanilla (official LLaVA), LearnPruner
(Transformers LLaVA), and Ours (official LLaVA), with 500 DetailCaps samples,
four complete repetitions, 20 process warmups, and fixed 32-token generation.

Hard failures are limited to protocol/provenance violations: missing or mixed
rows, changed hashes, invalid numeric values, non-32-token generations,
invalid per-layer traces, and disagreement between each adjacent instrumented
and uninstrumented generation. Clock variation and performance targets are
reported as warnings because the formal run uses an unlocked GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from benchmark_efficiency_v2 import (
    CLOCK_MODES,
    CLOCK_MODE_MONITORED_UNLOCKED,
    FORMAL_HF_MODEL,
    FORMAL_LEARNPRUNER_CHECKPOINT,
    FORMAL_LEARNPRUNER_DIVERSITY_RATIO,
    FORMAL_LEARNPRUNER_PRUNE_LAYER,
    FORMAL_LEARNPRUNER_STAGE1_TOKENS,
    FORMAL_LEARNPRUNER_STAGE2_TOKENS,
    FORMAL_LOCKED_SM_CLOCK_MHZ,
    FORMAL_MAX_NEW_TOKENS,
    FORMAL_OFFICIAL_MODEL,
    FORMAL_OURS_BUDGET,
    FORMAL_OURS_CHECKPOINT,
    FORMAL_PHYSICAL_GPU_ID,
    FORMAL_SEED,
    FORMAL_WARMUP,
    LEARNPRUNER_CHECKPOINT_USAGE,
    PROMPT,
    PROTOCOL_VERSION,
    RESULT_FIELDS,
    checkpoint_hashes,
    selected_source_hashes,
)


ARMS = ("vanilla", "learnpruner", "ours")
EXPECTED_N = 500
EXPECTED_REPETITIONS = 4
EXPECTED_OUTPUT_TOKENS = 32
EXPECTED_DECODER_LAYERS = 32
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

BACKENDS = {
    "vanilla": "official_llava",
    "learnpruner": "huggingface_llava",
    "ours": "official_llava",
}
DISPLAY_NAMES = {
    "vanilla": "Vanilla",
    "learnpruner": "LearnPruner",
    "ours": "Ours",
}

LATENCY_METRICS = (
    "vision_cuda_ms",
    "projector_cuda_ms",
    "llm_prefill_cuda_ms",
    "multimodal_prefill_cuda_ms",
    "ttft_cuda_ms",
    "decode_cuda_ms",
    "tpot_cuda_ms",
    "e2e_cuda_ms",
    "e2e_uninstrumented_cuda_ms",
    "ttft_wall_ms",
    "decode_wall_ms",
    "tpot_wall_ms",
    "e2e_wall_ms",
    "e2e_uninstrumented_wall_ms",
)
THROUGHPUT_METRICS = (
    "throughput_output_tokens_per_s",
    "throughput_uninstrumented_output_tokens_per_s",
    "decode_throughput_tokens_per_s",
)
OBSERVER_METRICS = (
    "observer_e2e_cuda_ratio",
    "observer_e2e_wall_ratio",
)
RESOURCE_METRICS = (
    "avg_visual_tokens",
    "estimated_prefill_tflops",
    "analytical_prefill_kv_cache_mb",
    "baseline_allocated_gb",
    "peak_allocated_gb",
    "incremental_peak_allocated_gb",
    "peak_reserved_gb",
)
TELEMETRY_METRICS = (
    "gpu_sm_clock_mhz_before",
    "gpu_sm_clock_mhz_after",
    "gpu_power_w_before",
    "gpu_power_w_after",
    "gpu_temperature_c_before",
    "gpu_temperature_c_after",
    "gpu_utilization_pct_before",
    "gpu_utilization_pct_after",
)
METRICS = (
    LATENCY_METRICS
    + THROUGHPUT_METRICS
    + OBSERVER_METRICS
    + RESOURCE_METRICS
    + TELEMETRY_METRICS
)
PRIMARY_SPEEDUP_METRICS = (
    "multimodal_prefill_cuda_ms",
    "ttft_cuda_ms",
    "decode_cuda_ms",
    "tpot_cuda_ms",
    "e2e_uninstrumented_cuda_ms",
    "throughput_uninstrumented_output_tokens_per_s",
    "decode_throughput_tokens_per_s",
)
COMPARISONS = (
    ("vanilla", "ours", "matched_official_backend"),
    ("vanilla", "learnpruner", "cross_backend_raw_system"),
    ("learnpruner", "ours", "cross_backend_raw_system"),
)


class SummaryError(RuntimeError):
    """A hard protocol/provenance failure."""


@dataclass(frozen=True)
class MetricStats:
    count: int
    mean: float
    std: float
    median: float
    p10: float
    p90: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class SpeedupStats:
    baseline: str
    current: str
    comparison_type: str
    metric: str
    direction: str
    count: int
    mean_speedup: float
    median_speedup: float
    paired_median: float
    paired_p10: float
    paired_p90: float


@dataclass
class MethodData:
    rows: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    values: dict[str, dict[tuple[int, int], float]] = field(
        default_factory=lambda: {metric: {} for metric in METRICS}
    )
    config_ids: set[str] = field(default_factory=set)


@dataclass
class LoadedResults:
    methods: dict[str, MethodData]
    input_provenance: dict[str, Any]
    sample_sha256: str
    gpu_uuid: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--physical-gpu-id",
        default=FORMAL_PHYSICAL_GPU_ID,
    )
    parser.add_argument("--run-id", required=False, default="formal")
    parser.add_argument(
        "--clock-mode",
        choices=CLOCK_MODES,
        default=CLOCK_MODE_MONITORED_UNLOCKED,
    )
    parser.add_argument("--expected-n", type=int, default=EXPECTED_N)
    parser.add_argument(
        "--expected-repetitions",
        type=int,
        default=EXPECTED_REPETITIONS,
    )
    parser.add_argument(
        "--expected-output-tokens",
        type=int,
        default=EXPECTED_OUTPUT_TOKENS,
    )
    # Retained for CLI compatibility. These bounds only produce warnings.
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument(
        "--max-latency-p90-p10-ratio",
        type=float,
        default=1.15,
    )
    parser.add_argument(
        "--max-clock-p90-p10-ratio",
        type=float,
        default=1.10,
    )
    parser.add_argument(
        "--max-cross-method-clock-median-ratio",
        type=float,
        default=1.05,
    )
    parser.add_argument(
        "--max-observer-e2e-ratio",
        type=float,
        default=1.15,
    )
    parser.add_argument(
        "--max-observer-asymmetry-ratio",
        type=float,
        default=1.05,
    )
    return parser.parse_args(argv)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise SummaryError("Cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def close_enough(
    left: float,
    right: float,
    *,
    absolute: float = 1e-4,
    relative: float = 5e-4,
) -> bool:
    return abs(left - right) <= max(
        absolute,
        relative * max(abs(left), abs(right)),
    )


def finite_float(
    raw: Any,
    *,
    label: str,
    nonnegative: bool = True,
    positive: bool = False,
) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SummaryError(f"Invalid numeric value for {label}: {raw!r}") from exc
    if not math.isfinite(value):
        raise SummaryError(f"Non-finite value for {label}: {value!r}")
    if nonnegative and value < 0.0:
        raise SummaryError(f"Negative value for {label}: {value!r}")
    if positive and value <= 0.0:
        raise SummaryError(f"Nonpositive value for {label}: {value!r}")
    return value


def integer(raw: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(raw, bool):
        raise SummaryError(f"Boolean is not an integer for {label}")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SummaryError(f"Invalid integer for {label}: {raw!r}") from exc
    if str(raw).strip() not in {str(value), f"+{value}"}:
        raise SummaryError(f"Non-canonical integer for {label}: {raw!r}")
    if minimum is not None and value < minimum:
        raise SummaryError(f"{label}={value} is below {minimum}")
    return value


def discover_input_files(args: argparse.Namespace) -> list[Path]:
    root = (
        args.result_root.expanduser().resolve()
        / PROTOCOL_VERSION
        / f"cuda{args.physical_gpu_id}"
    )
    if not root.is_dir():
        raise SummaryError(f"Missing protocol-v2 result directory: {root}")
    candidates = sorted(
        root.glob(f"*/*/{args.run_id}_rep*_seed*_n*.tsv")
    )
    if not candidates:
        raise SummaryError(
            f"No result TSVs for exact run-id {args.run_id!r} under {root}"
        )
    foreign = sorted(
        {
            path.parent.parent.name
            for path in candidates
            if path.parent.parent.name not in ARMS
        }
    )
    if foreign:
        raise SummaryError(
            "The three-method summary found foreign arms for this run-id: "
            f"{foreign}"
        )
    expected_file_count = len(ARMS) * EXPECTED_REPETITIONS
    if len(candidates) != expected_file_count:
        raise SummaryError(
            f"Expected exactly {expected_file_count} TSVs, found "
            f"{len(candidates)}"
        )
    return candidates


def snapshot_input_provenance(paths: Sequence[Path]) -> dict[str, Any]:
    records: list[dict[str, str]] = []
    for result_path in paths:
        metadata_path = result_path.with_suffix(".metadata.json")
        if not metadata_path.is_file():
            raise SummaryError(f"Missing metadata: {metadata_path}")
        records.append(
            {
                "result_tsv": str(result_path.resolve()),
                "result_tsv_sha256": file_sha256(result_path),
                "metadata_json": str(metadata_path.resolve()),
                "metadata_json_sha256": file_sha256(metadata_path),
            }
        )
    return {
        "files": records,
        "aggregate_sha256": sha256_text(canonical_json(records)),
    }


def current_hashes(method: str) -> tuple[dict[str, Any], dict[str, Any]]:
    hash_args = argparse.Namespace(
        method=method,
        official_model=FORMAL_OFFICIAL_MODEL,
        hf_model=FORMAL_HF_MODEL,
        ours_checkpoint=FORMAL_OURS_CHECKPOINT,
        learnpruner_checkpoint=(
            FORMAL_LEARNPRUNER_CHECKPOINT.expanduser().resolve()
        ),
    )
    return selected_source_hashes(hash_args), checkpoint_hashes(hash_args)


def validate_sample_hash(
    protocol_config: dict[str, Any],
    cache: dict[Path, str],
) -> None:
    raw_path = protocol_config.get("sample_path")
    if not isinstance(raw_path, str):
        raise SummaryError("protocol_config.sample_path is missing")
    sample_path = Path(raw_path).expanduser().resolve()
    if not sample_path.is_file():
        raise SummaryError(f"Recorded sample parquet is missing: {sample_path}")
    observed = cache.get(sample_path)
    if observed is None:
        observed = file_sha256(sample_path)
        cache[sample_path] = observed
    if observed != protocol_config.get("sample_sha256"):
        raise SummaryError(
            f"Sample parquet hash changed: {sample_path}, "
            f"observed={observed}, recorded={protocol_config.get('sample_sha256')}"
        )


def expected_protocol_values(
    args: argparse.Namespace,
    arm: str,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": args.run_id,
        "method": arm,
        "arm_id": arm,
        "backend_family": BACKENDS[arm],
        "physical_gpu_id": str(args.physical_gpu_id),
        "clock_mode": args.clock_mode,
        "locked_sm_clock_mhz": (
            None
            if args.clock_mode == CLOCK_MODE_MONITORED_UNLOCKED
            else FORMAL_LOCKED_SM_CLOCK_MHZ
        ),
        "num_samples": EXPECTED_N,
        "sampling_seed": FORMAL_SEED,
        "prompt": PROMPT,
        "warmup": FORMAL_WARMUP,
        "max_new_tokens": EXPECTED_OUTPUT_TOKENS,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "batch_size": 1,
        "greedy": True,
        "use_cache": True,
        "force_fixed_output_length": True,
        "trace_validation_enabled": True,
    }


def validate_method_protocol(
    protocol_config: dict[str, Any],
    arm: str,
) -> None:
    pruning = protocol_config.get("pruning")
    if not isinstance(pruning, dict):
        raise SummaryError(f"Missing pruning configuration for {arm}")
    required = {
        "ours_budget": FORMAL_OURS_BUDGET,
        "learnpruner_stage1_tokens": FORMAL_LEARNPRUNER_STAGE1_TOKENS,
        "learnpruner_stage2_tokens": FORMAL_LEARNPRUNER_STAGE2_TOKENS,
        "learnpruner_prune_layer": FORMAL_LEARNPRUNER_PRUNE_LAYER,
        "learnpruner_diversity_ratio": FORMAL_LEARNPRUNER_DIVERSITY_RATIO,
    }
    mismatches = {
        key: (pruning.get(key), expected)
        for key, expected in required.items()
        if pruning.get(key) != expected
    }
    if mismatches:
        raise SummaryError(
            f"Formal pruning configuration changed for {arm}: "
            f"{canonical_json(mismatches)}"
        )
    if arm == "ours":
        ours_required = {
            "ours_cached_bool_mask": "off",
            "ours_profile_internal": "off",
            "ours_implicit_causal": True,
            "ours_fixed_length_greedy": "on",
            "ours_triton_rms": "on",
            "ours_cuda_graph_prefill": "on",
            "ours_cuda_graph_prefill_cache_size": 2,
            "ours_cuda_graph_decode": "on",
            "ours_static_kv_decode": "off",
        }
        ours_mismatches = {
            key: (pruning.get(key), expected)
            for key, expected in ours_required.items()
            if pruning.get(key) != expected
        }
        if ours_mismatches:
            raise SummaryError(
                "Ours did not use the admitted v2 path: "
                f"{canonical_json(ours_mismatches)}"
            )
        audit = protocol_config.get("ours_combined_audit")
        if (
            not isinstance(audit, dict)
            or audit.get("passed") is not True
            or audit.get("status") != "pass"
        ):
            raise SummaryError("Ours is missing its successful admission audit")
        artifact = Path(str(audit.get("artifact_path", ""))).expanduser().resolve()
        if (
            not artifact.is_file()
            or file_sha256(artifact) != audit.get("artifact_sha256")
        ):
            raise SummaryError("Ours admission-audit artifact hash changed")
    elif protocol_config.get("ours_combined_audit") is not None:
        raise SummaryError(f"Unexpected Ours audit attached to {arm}")


def validate_metadata(
    args: argparse.Namespace,
    path: Path,
    metadata: dict[str, Any],
    arm: str,
    sample_hash_cache: dict[Path, str],
) -> tuple[int, str, int, list[int], dict[str, Any]]:
    expected_backend = BACKENDS[arm]
    try:
        repetition = int(metadata["repetition"])
        config_id = str(metadata["config_id"])
        decoder_layers = int(metadata["model"]["decoder_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SummaryError(f"Incomplete metadata for {path}") from exc
    if repetition not in range(EXPECTED_REPETITIONS):
        raise SummaryError(f"Invalid repetition {repetition} in {path}")
    if decoder_layers != EXPECTED_DECODER_LAYERS:
        raise SummaryError(
            f"Expected {EXPECTED_DECODER_LAYERS} decoder layers in {path}, "
            f"found {decoder_layers}"
        )
    expected_metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": args.run_id,
        "method": arm,
        "arm_id": arm,
        "backend": expected_backend,
        "clock_mode": args.clock_mode,
        "locked_sm_clock_mhz": (
            None
            if args.clock_mode == CLOCK_MODE_MONITORED_UNLOCKED
            else FORMAL_LOCKED_SM_CLOCK_MHZ
        ),
        "result_fields": RESULT_FIELDS,
    }
    metadata_mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if metadata_mismatches:
        raise SummaryError(
            f"Metadata protocol mismatch for {path}: "
            f"{canonical_json(metadata_mismatches)}"
        )
    if path.parent.name != config_id or path.parent.parent.name != arm:
        raise SummaryError(f"Path does not encode arm/config id: {path}")

    protocol_config = metadata.get("protocol_config")
    if not isinstance(protocol_config, dict):
        raise SummaryError(f"Missing protocol_config in {path}")
    recomputed_config_id = sha256_text(canonical_json(protocol_config))[:16]
    if recomputed_config_id != config_id:
        raise SummaryError(
            f"Config hash mismatch for {path}: metadata={config_id}, "
            f"recomputed={recomputed_config_id}"
        )
    expected_protocol = expected_protocol_values(args, arm)
    protocol_mismatches = {
        key: (protocol_config.get(key), expected)
        for key, expected in expected_protocol.items()
        if protocol_config.get(key) != expected
    }
    if protocol_mismatches:
        raise SummaryError(
            f"Protocol config mismatch for {path}: "
            f"{canonical_json(protocol_mismatches)}"
        )
    expected_paths = {
        "official_model": str(FORMAL_OFFICIAL_MODEL),
        "hf_model": str(FORMAL_HF_MODEL),
        "ours_checkpoint": str(FORMAL_OURS_CHECKPOINT),
        "learnpruner_checkpoint": str(
            FORMAL_LEARNPRUNER_CHECKPOINT.expanduser().resolve()
        ),
    }
    if protocol_config.get("paths") != expected_paths:
        raise SummaryError(f"Formal model/checkpoint paths changed in {path}")
    if (
        protocol_config.get("learnpruner_checkpoint_usage")
        != LEARNPRUNER_CHECKPOINT_USAGE
        or metadata.get("learnpruner_checkpoint_usage")
        != LEARNPRUNER_CHECKPOINT_USAGE
    ):
        raise SummaryError(
            f"Missing latency-only LearnPruner checkpoint disclaimer in {path}"
        )
    validate_method_protocol(protocol_config, arm)
    validate_sample_hash(protocol_config, sample_hash_cache)

    expected_sources, expected_checkpoints = current_hashes(arm)
    if protocol_config.get("source_hashes") != expected_sources:
        raise SummaryError(
            f"Current source hashes differ from the timed run for {path}"
        )
    if protocol_config.get("checkpoint_hashes") != expected_checkpoints:
        raise SummaryError(
            f"Current checkpoint hashes differ from the timed run for {path}"
        )
    if not all(
        isinstance(value, str) and value
        for value in expected_sources.values()
        if value is not None
    ):
        raise SummaryError(f"Invalid current source hash set for {arm}")
    if not all(
        isinstance(value, str) and value
        for value in expected_checkpoints.values()
        if value is not None
    ):
        raise SummaryError(f"Invalid current checkpoint hash set for {arm}")

    trace = metadata.get("trace_validation")
    if not isinstance(trace, dict):
        raise SummaryError(f"Missing untimed layer trace in {path}")
    try:
        canonical_trace = [
            int(value) for value in trace["canonical_layer_tokens"]
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise SummaryError(f"Invalid untimed layer trace in {path}") from exc
    if len(canonical_trace) != decoder_layers:
        raise SummaryError(f"Untimed layer trace length mismatch in {path}")

    if arm == "learnpruner":
        loaded_config = metadata["model"].get("learnpruner_loaded_config")
        required_loaded = {
            "implementation_mode": "paper_aligned",
            "stage1_tokens": FORMAL_LEARNPRUNER_STAGE1_TOKENS,
            "stage2_tokens": FORMAL_LEARNPRUNER_STAGE2_TOKENS,
            "prune_layer": FORMAL_LEARNPRUNER_PRUNE_LAYER,
            "prune_layer_numbering": "one_based",
        }
        if not isinstance(loaded_config, dict) or any(
            loaded_config.get(key) != expected
            for key, expected in required_loaded.items()
        ):
            raise SummaryError(
                f"Loaded LearnPruner configuration is not paper-aligned in {path}"
            )
    return repetition, config_id, decoder_layers, canonical_trace, protocol_config


def parse_json_list(
    raw: str,
    *,
    label: str,
    expected_length: int | None = None,
) -> list[Any]:
    try:
        value = json.loads(raw)
    except Exception as exc:
        raise SummaryError(f"Invalid JSON for {label}") from exc
    if not isinstance(value, list):
        raise SummaryError(f"{label} is not a list")
    if expected_length is not None and len(value) != expected_length:
        raise SummaryError(
            f"{label} has length {len(value)}, expected {expected_length}"
        )
    return value


def expected_layer_trace(
    arm: str,
    prompt_tokens: int,
    input_tokens: int,
    pruner_stats: dict[str, Any],
) -> list[int]:
    if arm == "vanilla":
        expected_input = prompt_tokens + 576
        if input_tokens != expected_input or pruner_stats:
            raise SummaryError("Vanilla input/pruner trace is inconsistent")
        return [expected_input] * EXPECTED_DECODER_LAYERS

    if arm == "learnpruner":
        required_stats = {
            "implementation_mode": "paper_aligned",
            "stage1_visual_tokens": FORMAL_LEARNPRUNER_STAGE1_TOKENS,
            "stage2_visual_tokens": FORMAL_LEARNPRUNER_STAGE2_TOKENS,
            "stage2_paper_layer": FORMAL_LEARNPRUNER_PRUNE_LAYER,
            "stage2_layer_idx": FORMAL_LEARNPRUNER_PRUNE_LAYER - 1,
        }
        if any(
            pruner_stats.get(key) != expected
            for key, expected in required_stats.items()
        ):
            raise SummaryError("LearnPruner row is not paper-aligned 111/37")
        final_sequence = prompt_tokens + FORMAL_LEARNPRUNER_STAGE2_TOKENS
        if pruner_stats.get("final_sequence_tokens") != final_sequence:
            raise SummaryError("LearnPruner final sequence length is inconsistent")
        return (
            [prompt_tokens + FORMAL_LEARNPRUNER_STAGE1_TOKENS]
            * FORMAL_LEARNPRUNER_PRUNE_LAYER
            + [final_sequence]
            * (EXPECTED_DECODER_LAYERS - FORMAL_LEARNPRUNER_PRUNE_LAYER)
        )

    required_stats = {
        "predictor_only": False,
        "avg_token_budget_profile": FORMAL_OURS_BUDGET,
        "kept_visual_tokens": 137,
        "mid_pruning_layer_idx": 12,
        "mid_target_visual_tokens": 32,
        "mid_sequence_tokens": prompt_tokens + 32,
        "final_wipe_layer_idx": 25,
        "final_sequence_tokens": prompt_tokens,
    }
    if any(
        pruner_stats.get(key) != expected
        for key, expected in required_stats.items()
    ):
        raise SummaryError("Ours row does not match the formal budget-64 trace")
    if input_tokens != prompt_tokens + 576:
        raise SummaryError("Ours expanded input length is inconsistent")
    return (
        [prompt_tokens + 137] * 12
        + [prompt_tokens + 32] * 13
        + [prompt_tokens] * 7
    )


def validate_graph_fields(
    arm: str,
    row: dict[str, str],
    generated_hash: str,
    label: str,
) -> None:
    callbacks = parse_json_list(
        row["cuda_graph_decode_callback_steps"],
        label=f"{label}.cuda_graph_decode_callback_steps",
    )
    if arm == "ours":
        if row["cuda_graph_residency_warmup_tail_sha256"] != generated_hash:
            raise SummaryError(f"Ours graph-residency hash mismatch in {label}")
        runner_id = integer(
            row["cuda_graph_prefill_callback_runner_id"],
            label=f"{label}.prefill_runner_id",
            minimum=1,
        )
        if (
            runner_id < 1
            or row["cuda_graph_prefill_pair_status"]
            != "stable_runner_two_cache_hit_replays_no_fallback"
            or callbacks != list(range(1, EXPECTED_OUTPUT_TOKENS))
            or row["cuda_graph_runner_cache_status"] != "single_runner_reused"
            or row["triton_rms_runtime_status"]
            != "prefill_only_enabled_no_runtime_fallback"
        ):
            raise SummaryError(f"Ours graph/runtime protocol mismatch in {label}")
        return
    if (
        row["cuda_graph_residency_warmup_tail_sha256"] != ""
        or row["cuda_graph_prefill_callback_runner_id"] != ""
        or row["cuda_graph_prefill_pair_status"] != "disabled"
        or callbacks
        or row["cuda_graph_runner_cache_status"] != "disabled"
        or row["triton_rms_runtime_status"] != "disabled"
    ):
        raise SummaryError(f"Unexpected Ours-only runtime fields in {label}")


def validate_timing_arithmetic(
    row: dict[str, str],
    parsed_metrics: dict[str, float],
    label: str,
) -> None:
    expected_decode_steps = EXPECTED_OUTPUT_TOKENS - 1
    decode_cuda = [
        finite_float(value, label=f"{label}.decode_step_cuda[{index}]")
        for index, value in enumerate(
            parse_json_list(
                row["decode_step_cuda_ms"],
                label=f"{label}.decode_step_cuda_ms",
                expected_length=expected_decode_steps,
            )
        )
    ]
    decode_wall = [
        finite_float(value, label=f"{label}.decode_step_wall[{index}]")
        for index, value in enumerate(
            parse_json_list(
                row["decode_step_wall_ms"],
                label=f"{label}.decode_step_wall_ms",
                expected_length=expected_decode_steps,
            )
        )
    ]
    decode_forward = [
        finite_float(value, label=f"{label}.decode_forward[{index}]")
        for index, value in enumerate(
            parse_json_list(
                row["decode_forward_cuda_ms"],
                label=f"{label}.decode_forward_cuda_ms",
                expected_length=expected_decode_steps,
            )
        )
    ]
    if any(
        forward > interval + 1e-3
        for forward, interval in zip(decode_forward, decode_cuda)
    ):
        raise SummaryError(f"Decode forward exceeds its interval in {label}")
    consistency = (
        (
            sum(decode_cuda),
            parsed_metrics["decode_cuda_ms"],
            "decode CUDA sum",
        ),
        (
            statistics.fmean(decode_cuda),
            parsed_metrics["tpot_cuda_ms"],
            "TPOT CUDA",
        ),
        (
            sum(decode_wall),
            parsed_metrics["decode_wall_ms"],
            "decode wall sum",
        ),
        (
            statistics.fmean(decode_wall),
            parsed_metrics["tpot_wall_ms"],
            "TPOT wall",
        ),
        (
            parsed_metrics["ttft_cuda_ms"]
            + parsed_metrics["decode_cuda_ms"],
            parsed_metrics["e2e_cuda_ms"],
            "CUDA TTFT+decode",
        ),
        (
            parsed_metrics["ttft_wall_ms"]
            + parsed_metrics["decode_wall_ms"],
            parsed_metrics["e2e_wall_ms"],
            "wall TTFT+decode",
        ),
    )
    for observed, expected, name in consistency:
        if not close_enough(observed, expected):
            raise SummaryError(
                f"{name} mismatch in {label}: {observed} vs {expected}"
            )


def validate_row(
    args: argparse.Namespace,
    row: dict[str, str],
    *,
    arm: str,
    repetition: int,
    decoder_layers: int,
    label: str,
) -> tuple[
    tuple[int, int],
    dict[str, Any],
    dict[str, float],
    tuple[Any, ...],
]:
    expected_static = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": args.run_id,
        "method": arm,
        "arm_id": arm,
        "backend": BACKENDS[arm],
        "locked_sm_clock_mhz": (
            ""
            if args.clock_mode == CLOCK_MODE_MONITORED_UNLOCKED
            else str(FORMAL_LOCKED_SM_CLOCK_MHZ)
        ),
        "repetition": str(repetition),
        "generation_variant": (
            "fast_triton_prefill_rms_cuda_graph_prefill_cuda_graph_decode"
            if arm == "ours"
            else "generation_mixin"
        ),
    }
    static_mismatches = {
        key: (row.get(key), expected)
        for key, expected in expected_static.items()
        if row.get(key) != expected
    }
    if static_mismatches:
        raise SummaryError(
            f"Row protocol mismatch in {label}: "
            f"{canonical_json(static_mismatches)}"
        )

    sample_order = integer(
        row["sample_order"],
        label=f"{label}.sample_order",
        minimum=0,
    )
    dataset_index = integer(
        row["dataset_index"],
        label=f"{label}.dataset_index",
        minimum=0,
    )
    prompt_tokens = integer(
        row["prompt_text_tokens"],
        label=f"{label}.prompt_text_tokens",
        minimum=1,
    )
    input_tokens = integer(
        row["input_sequence_tokens"],
        label=f"{label}.input_sequence_tokens",
        minimum=1,
    )
    projector_input = integer(
        row["projector_input_tokens"],
        label=f"{label}.projector_input_tokens",
        minimum=1,
    )
    projector_output = integer(
        row["projector_output_tokens"],
        label=f"{label}.projector_output_tokens",
        minimum=1,
    )
    expected_projector = (
        FORMAL_LEARNPRUNER_STAGE1_TOKENS
        if arm == "learnpruner"
        else 576
    )
    if projector_input != expected_projector or projector_output != expected_projector:
        raise SummaryError(
            f"Projector trace mismatch in {label}: "
            f"{projector_input}/{projector_output}, expected {expected_projector}"
        )

    requested = integer(
        row["requested_output_tokens"],
        label=f"{label}.requested_output_tokens",
    )
    actual = integer(
        row["actual_output_tokens"],
        label=f"{label}.actual_output_tokens",
    )
    forwards = integer(
        row["forward_steps"],
        label=f"{label}.forward_steps",
    )
    output_sequence_length = integer(
        row["output_sequence_length"],
        label=f"{label}.output_sequence_length",
        minimum=EXPECTED_OUTPUT_TOKENS,
    )
    if (
        requested != EXPECTED_OUTPUT_TOKENS
        or actual != EXPECTED_OUTPUT_TOKENS
        or forwards != EXPECTED_OUTPUT_TOKENS
        or output_sequence_length < EXPECTED_OUTPUT_TOKENS
    ):
        raise SummaryError(f"Fixed 32-token generation violated in {label}")

    token_ids = parse_json_list(
        row["generated_token_ids"],
        label=f"{label}.generated_token_ids",
        expected_length=EXPECTED_OUTPUT_TOKENS,
    )
    if any(isinstance(token, bool) or not isinstance(token, int) for token in token_ids):
        raise SummaryError(f"Generated token IDs are not integers in {label}")
    generated_hash = sha256_text(compact_json(token_ids))
    if (
        generated_hash != row["generated_tail_sha256"]
        or generated_hash != row["uninstrumented_generated_tail_sha256"]
    ):
        raise SummaryError(
            "Instrumented/uninstrumented generated-tail hash mismatch in "
            f"{label}"
        )
    expected_order = (
        "instrumented_first"
        if (repetition + sample_order) % 2 == 0
        else "uninstrumented_first"
    )
    if row["measurement_order"] != expected_order:
        raise SummaryError(f"Adjacent measurement order mismatch in {label}")
    validate_graph_fields(arm, row, generated_hash, label)

    try:
        pruner_stats = json.loads(row["pruner_stats"])
    except Exception as exc:
        raise SummaryError(f"Invalid pruner_stats JSON in {label}") from exc
    if not isinstance(pruner_stats, dict):
        raise SummaryError(f"pruner_stats is not an object in {label}")
    layer_tokens = [
        integer(value, label=f"{label}.layer_token")
        for value in row["layer_prefill_tokens"].split(",")
    ]
    if len(layer_tokens) != decoder_layers:
        raise SummaryError(f"Per-layer trace length mismatch in {label}")
    expected_layers = expected_layer_trace(
        arm,
        prompt_tokens,
        input_tokens,
        pruner_stats,
    )
    if layer_tokens != expected_layers:
        raise SummaryError(f"Per-layer token trace mismatch in {label}")

    parsed_metrics: dict[str, float] = {}
    for metric in METRICS:
        parsed_metrics[metric] = finite_float(
            row[metric],
            label=f"{label}.{metric}",
            positive=(
                metric
                in {
                    "e2e_cuda_ms",
                    "e2e_uninstrumented_cuda_ms",
                    "e2e_wall_ms",
                    "e2e_uninstrumented_wall_ms",
                    "throughput_output_tokens_per_s",
                    "throughput_uninstrumented_output_tokens_per_s",
                    "decode_throughput_tokens_per_s",
                    "gpu_sm_clock_mhz_before",
                    "gpu_sm_clock_mhz_after",
                    "gpu_power_w_before",
                    "gpu_power_w_after",
                    "gpu_temperature_c_before",
                    "gpu_temperature_c_after",
                }
            ),
        )
    observer_cuda_delta = finite_float(
        row["observer_e2e_cuda_delta_ms"],
        label=f"{label}.observer_e2e_cuda_delta_ms",
        nonnegative=False,
    )
    observer_wall_delta = finite_float(
        row["observer_e2e_wall_delta_ms"],
        label=f"{label}.observer_e2e_wall_delta_ms",
        nonnegative=False,
    )
    expected_avg_visual = statistics.fmean(
        token_count - prompt_tokens for token_count in layer_tokens
    )
    if not close_enough(
        parsed_metrics["avg_visual_tokens"],
        expected_avg_visual,
        absolute=1e-7,
        relative=1e-8,
    ):
        raise SummaryError(f"Average visual-token count mismatch in {label}")
    if not close_enough(
        observer_cuda_delta,
        parsed_metrics["e2e_cuda_ms"]
        - parsed_metrics["e2e_uninstrumented_cuda_ms"],
    ) or not close_enough(
        observer_wall_delta,
        parsed_metrics["e2e_wall_ms"]
        - parsed_metrics["e2e_uninstrumented_wall_ms"],
    ):
        raise SummaryError(f"Observer delta mismatch in {label}")
    expected_cuda_ratio = (
        parsed_metrics["e2e_cuda_ms"]
        / parsed_metrics["e2e_uninstrumented_cuda_ms"]
    )
    expected_wall_ratio = (
        parsed_metrics["e2e_wall_ms"]
        / parsed_metrics["e2e_uninstrumented_wall_ms"]
    )
    if not close_enough(
        parsed_metrics["observer_e2e_cuda_ratio"],
        expected_cuda_ratio,
    ) or not close_enough(
        parsed_metrics["observer_e2e_wall_ratio"],
        expected_wall_ratio,
    ):
        raise SummaryError(f"Observer ratio mismatch in {label}")
    expected_throughput = EXPECTED_OUTPUT_TOKENS / (
        parsed_metrics["e2e_uninstrumented_wall_ms"] / 1000.0
    )
    expected_decode_throughput = (EXPECTED_OUTPUT_TOKENS - 1) / (
        parsed_metrics["decode_wall_ms"] / 1000.0
    )
    if not close_enough(
        parsed_metrics["throughput_uninstrumented_output_tokens_per_s"],
        expected_throughput,
    ) or not close_enough(
        parsed_metrics["decode_throughput_tokens_per_s"],
        expected_decode_throughput,
    ):
        raise SummaryError(f"Throughput arithmetic mismatch in {label}")
    validate_timing_arithmetic(row, parsed_metrics, label)

    key = (repetition, dataset_index)
    identity = {
        "sample_order": sample_order,
        "dataset_index": dataset_index,
        "source": row["source"],
        "image": row["image"],
        "layer_tokens": layer_tokens,
        "generated_hash": generated_hash,
    }
    aggregate_record = (
        sample_order,
        dataset_index,
        row["generated_tail_sha256"],
        row["uninstrumented_generated_tail_sha256"],
        row["measurement_order"],
        row["cuda_graph_residency_warmup_tail_sha256"],
        row["cuda_graph_prefill_callback_runner_id"],
        row["cuda_graph_prefill_pair_status"],
        row["cuda_graph_decode_callback_steps"],
        row["cuda_graph_runner_cache_status"],
        row["triton_rms_runtime_status"],
        projector_input,
        projector_output,
    )
    return key, identity, parsed_metrics, aggregate_record


def load_results(args: argparse.Namespace) -> LoadedResults:
    files = discover_input_files(args)
    provenance_before = snapshot_input_provenance(files)
    methods = {arm: MethodData() for arm in ARMS}
    file_slots: set[tuple[str, int]] = set()
    sample_hash_cache: dict[Path, str] = {}
    common_protocol_values: dict[str, set[str]] = {
        key: set()
        for key in (
            "physical_gpu_id",
            "gpu_uuid",
            "gpu_name",
            "gpu_driver_version",
            "clock_mode",
            "locked_sm_clock_mhz",
            "sample_path",
            "sample_sha256",
            "num_samples",
            "sampling_seed",
            "prompt",
            "warmup",
            "max_new_tokens",
            "dtype",
            "attention_implementation",
            "batch_size",
            "greedy",
            "use_cache",
            "force_fixed_output_length",
            "paths",
            "runtime",
        )
    }

    for path in files:
        arm = path.parent.parent.name
        metadata_path = path.with_suffix(".metadata.json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SummaryError(f"Invalid metadata JSON: {metadata_path}") from exc
        (
            repetition,
            config_id,
            decoder_layers,
            canonical_trace,
            protocol_config,
        ) = validate_metadata(
            args,
            path,
            metadata,
            arm,
            sample_hash_cache,
        )
        slot = (arm, repetition)
        if slot in file_slots:
            raise SummaryError(f"Duplicate result file for {slot}")
        file_slots.add(slot)
        filename = (
            f"{args.run_id}_rep{repetition:02d}_seed{FORMAL_SEED}_"
            f"n{EXPECTED_N}.tsv"
        )
        if path.name != filename:
            raise SummaryError(
                f"Noncanonical result filename {path.name!r}; expected {filename!r}"
            )

        aggregate_records: list[tuple[Any, ...]] = []
        file_rows = 0
        sample_orders: set[int] = set()
        first_trace: list[int] | None = None
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != RESULT_FIELDS:
                raise SummaryError(f"Unexpected TSV schema in {path}")
            for line_number, row in enumerate(reader, start=2):
                label = f"{path}:{line_number}"
                key, identity, parsed_metrics, aggregate_record = validate_row(
                    args,
                    row,
                    arm=arm,
                    repetition=repetition,
                    decoder_layers=decoder_layers,
                    label=label,
                )
                if key in methods[arm].rows:
                    raise SummaryError(f"Duplicate key {key} for {arm}")
                methods[arm].rows[key] = identity
                for metric, value in parsed_metrics.items():
                    methods[arm].values[metric][key] = value
                sample_order = int(identity["sample_order"])
                if sample_order in sample_orders:
                    raise SummaryError(
                        f"Duplicate sample_order={sample_order} in {path}"
                    )
                sample_orders.add(sample_order)
                if sample_order == 0:
                    first_trace = list(identity["layer_tokens"])
                aggregate_records.append(aggregate_record)
                file_rows += 1

        if file_rows != EXPECTED_N or sample_orders != set(range(EXPECTED_N)):
            raise SummaryError(
                f"Incomplete sample order in {path}: rows={file_rows}, "
                f"unique_orders={len(sample_orders)}"
            )
        if first_trace != canonical_trace:
            raise SummaryError(
                f"Measured first-row trace differs from untimed trace in {path}"
            )
        expected_tsv_hash = file_sha256(path)
        expected_aggregate = sha256_text(
            canonical_json(sorted(aggregate_records))
        )
        finalized_metadata = {
            "completed_rows": EXPECTED_N,
            "result_tsv_sha256": expected_tsv_hash,
            "generated_tail_hash_aggregate": expected_aggregate,
        }
        finalized_mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in finalized_metadata.items()
            if metadata.get(key) != expected
        }
        if finalized_mismatches:
            raise SummaryError(
                f"Finalized TSV/metadata hash mismatch for {path}: "
                f"{canonical_json(finalized_mismatches)}"
            )
        if not isinstance(metadata.get("completed_at_utc"), str):
            raise SummaryError(f"Result metadata is not finalized: {path}")
        methods[arm].config_ids.add(config_id)
        for key in common_protocol_values:
            if key not in protocol_config:
                raise SummaryError(
                    f"Missing shared protocol field {key!r} in {path}"
                )
            common_protocol_values[key].add(
                canonical_json(protocol_config[key])
            )

    expected_slots = {
        (arm, repetition)
        for arm in ARMS
        for repetition in range(EXPECTED_REPETITIONS)
    }
    if file_slots != expected_slots:
        raise SummaryError(
            f"Method/repetition file slots differ: "
            f"missing={sorted(expected_slots - file_slots)}, "
            f"extra={sorted(file_slots - expected_slots)}"
        )
    expected_rows = EXPECTED_N * EXPECTED_REPETITIONS
    for arm, method_data in methods.items():
        if len(method_data.config_ids) != 1:
            raise SummaryError(
                f"{arm} has multiple/missing config ids: "
                f"{sorted(method_data.config_ids)}"
            )
        if len(method_data.rows) != expected_rows:
            raise SummaryError(
                f"{arm} has {len(method_data.rows)} rows, expected {expected_rows}"
            )
    mixed_protocol = {
        key: sorted(values)
        for key, values in common_protocol_values.items()
        if len(values) != 1
    }
    if mixed_protocol:
        raise SummaryError(
            "Methods/repetitions do not share one exact protocol: "
            f"{canonical_json(mixed_protocol)}"
        )

    reference = methods["vanilla"].rows
    reference_keys = set(reference)
    for arm in ARMS[1:]:
        if set(methods[arm].rows) != reference_keys:
            raise SummaryError(f"Paired repetition/dataset keys differ for {arm}")
        for key in reference_keys:
            left = reference[key]
            right = methods[arm].rows[key]
            if (
                left["sample_order"],
                left["source"],
                left["image"],
            ) != (
                right["sample_order"],
                right["source"],
                right["image"],
            ):
                raise SummaryError(
                    f"Ordered sample identity differs for {arm}, key={key}"
                )

    provenance_after = snapshot_input_provenance(files)
    if provenance_after != provenance_before:
        raise SummaryError(
            "Input TSV/metadata bytes changed during validation"
        )
    sample_sha256 = next(iter(common_protocol_values["sample_sha256"]))
    gpu_uuid = next(iter(common_protocol_values["gpu_uuid"]))
    return LoadedResults(
        methods=methods,
        input_provenance=provenance_after,
        sample_sha256=json.loads(sample_sha256),
        gpu_uuid=json.loads(gpu_uuid),
    )


def metric_stats(values: Sequence[float]) -> MetricStats:
    if not values:
        raise SummaryError("Cannot summarize an empty metric")
    return MetricStats(
        count=len(values),
        mean=statistics.fmean(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
        median=statistics.median(values),
        p10=percentile(values, 0.10),
        p90=percentile(values, 0.90),
        minimum=min(values),
        maximum=max(values),
    )


def build_metric_stats(
    loaded: LoadedResults,
) -> dict[str, dict[str, MetricStats]]:
    return {
        arm: {
            metric: metric_stats(list(method.values[metric].values()))
            for metric in METRICS
        }
        for arm, method in loaded.methods.items()
    }


def build_speedups(
    loaded: LoadedResults,
    stats: dict[str, dict[str, MetricStats]],
) -> list[SpeedupStats]:
    results: list[SpeedupStats] = []
    for baseline, current, comparison_type in COMPARISONS:
        keys = sorted(loaded.methods[baseline].rows)
        for metric in PRIMARY_SPEEDUP_METRICS:
            higher_is_better = metric in THROUGHPUT_METRICS
            baseline_values = loaded.methods[baseline].values[metric]
            current_values = loaded.methods[current].values[metric]
            if higher_is_better:
                paired = [
                    current_values[key] / baseline_values[key]
                    for key in keys
                ]
                mean_speedup = (
                    stats[current][metric].mean
                    / stats[baseline][metric].mean
                )
                median_speedup = (
                    stats[current][metric].median
                    / stats[baseline][metric].median
                )
                direction = "current_over_baseline"
            else:
                paired = [
                    baseline_values[key] / current_values[key]
                    for key in keys
                ]
                mean_speedup = (
                    stats[baseline][metric].mean
                    / stats[current][metric].mean
                )
                median_speedup = (
                    stats[baseline][metric].median
                    / stats[current][metric].median
                )
                direction = "baseline_over_current"
            if any(not math.isfinite(value) or value <= 0 for value in paired):
                raise SummaryError(
                    f"Invalid paired speedup for {baseline}/{current}/{metric}"
                )
            results.append(
                SpeedupStats(
                    baseline=baseline,
                    current=current,
                    comparison_type=comparison_type,
                    metric=metric,
                    direction=direction,
                    count=len(paired),
                    mean_speedup=mean_speedup,
                    median_speedup=median_speedup,
                    paired_median=statistics.median(paired),
                    paired_p10=percentile(paired, 0.10),
                    paired_p90=percentile(paired, 0.90),
                )
            )
    return results


def warning_messages(
    args: argparse.Namespace,
    stats: dict[str, dict[str, MetricStats]],
    speedups: Sequence[SpeedupStats],
) -> list[str]:
    warnings: list[str] = []
    clock_medians: list[float] = []
    for arm in ARMS:
        clocks = (
            stats[arm]["gpu_sm_clock_mhz_before"],
            stats[arm]["gpu_sm_clock_mhz_after"],
        )
        combined_clock_values = [
            value
            for metric in (
                "gpu_sm_clock_mhz_before",
                "gpu_sm_clock_mhz_after",
            )
            for value in (
                stats[arm][metric].minimum,
                stats[arm][metric].p10,
                stats[arm][metric].median,
                stats[arm][metric].p90,
                stats[arm][metric].maximum,
            )
        ]
        clock_ratio = percentile(combined_clock_values, 0.90) / max(
            percentile(combined_clock_values, 0.10),
            1e-12,
        )
        clock_median = statistics.fmean(
            metric.median for metric in clocks
        )
        clock_medians.append(clock_median)
        if clock_ratio > args.max_clock_p90_p10_ratio:
            warnings.append(
                f"{arm} unlocked SM-clock p90/p10={clock_ratio:.3f} exceeds "
                f"{args.max_clock_p90_p10_ratio:.3f}"
            )
        e2e = stats[arm]["e2e_uninstrumented_cuda_ms"]
        latency_ratio = e2e.p90 / max(e2e.p10, 1e-12)
        if latency_ratio > args.max_latency_p90_p10_ratio:
            warnings.append(
                f"{arm} E2E p90/p10={latency_ratio:.3f} exceeds "
                f"{args.max_latency_p90_p10_ratio:.3f}"
            )
        observer_lower = 1.0 / args.max_observer_e2e_ratio
        for metric in OBSERVER_METRICS:
            median = stats[arm][metric].median
            if not observer_lower <= median <= args.max_observer_e2e_ratio:
                warnings.append(
                    f"{arm} {metric} median={median:.3f} is outside "
                    f"[{observer_lower:.3f}, {args.max_observer_e2e_ratio:.3f}]"
                )
    cross_clock_ratio = max(clock_medians) / max(min(clock_medians), 1e-12)
    if cross_clock_ratio > args.max_cross_method_clock_median_ratio:
        warnings.append(
            f"Cross-method unlocked SM-clock median ratio="
            f"{cross_clock_ratio:.3f} exceeds "
            f"{args.max_cross_method_clock_median_ratio:.3f}"
        )

    speedup_index = {
        (item.baseline, item.current, item.metric): item
        for item in speedups
    }
    old_targets = (
        ("vanilla", "ours", "multimodal_prefill_cuda_ms", 1.50),
        ("vanilla", "ours", "decode_cuda_ms", 1.20),
        ("vanilla", "ours", "tpot_cuda_ms", 1.20),
        ("vanilla", "ours", "e2e_uninstrumented_cuda_ms", 1.20),
        ("vanilla", "learnpruner", "multimodal_prefill_cuda_ms", 1.00),
        ("vanilla", "learnpruner", "e2e_uninstrumented_cuda_ms", 1.00),
    )
    for baseline, current, metric, target in old_targets:
        result = speedup_index[(baseline, current, metric)]
        if result.mean_speedup < target:
            warnings.append(
                f"Informational performance target: {baseline}/{current} "
                f"{metric} mean speedup={result.mean_speedup:.3f}x < "
                f"{target:.3f}x"
            )
    return warnings


def write_metrics(
    path: Path,
    stats: dict[str, dict[str, MetricStats]],
) -> None:
    fields = ("method", "metric", *MetricStats.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for arm in ARMS:
            for metric in METRICS:
                writer.writerow(
                    {
                        "method": arm,
                        "metric": metric,
                        **asdict(stats[arm][metric]),
                    }
                )


def write_speedups(path: Path, speedups: Sequence[SpeedupStats]) -> None:
    fields = tuple(SpeedupStats.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for result in speedups:
            writer.writerow(asdict(result))


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    stats: dict[str, dict[str, MetricStats]],
    speedups: Sequence[SpeedupStats],
    warnings: Sequence[str],
) -> None:
    displayed_metrics = (
        "multimodal_prefill_cuda_ms",
        "ttft_cuda_ms",
        "tpot_cuda_ms",
        "decode_cuda_ms",
        "e2e_uninstrumented_cuda_ms",
        "throughput_uninstrumented_output_tokens_per_s",
    )
    labels = {
        "multimodal_prefill_cuda_ms": "Prefill CUDA (ms)",
        "ttft_cuda_ms": "TTFT CUDA (ms)",
        "tpot_cuda_ms": "TPOT CUDA (ms)",
        "decode_cuda_ms": "Decode CUDA (ms)",
        "e2e_uninstrumented_cuda_ms": "E2E CUDA (ms)",
        "throughput_uninstrumented_output_tokens_per_s": "Throughput (tok/s)",
    }
    lines = [
        "# Efficiency v2 — three-method formal result",
        "",
        f"Run `{args.run_id}`; DetailCaps n={EXPECTED_N}; "
        f"{EXPECTED_REPETITIONS} complete repetitions; warmup={FORMAL_WARMUP}; "
        f"fixed generation={EXPECTED_OUTPUT_TOKENS} tokens; "
        f"clock mode=`{args.clock_mode}`.",
        "",
        "Hard gate: **PASS**. Protocol, completeness, hashes, finite values, "
        "fixed output length, per-layer traces, and adjacent "
        "instrumented/uninstrumented token hashes were validated.",
        "",
        "Clock-distribution and performance thresholds are informational "
        "warnings only because the GPU clock was not locked.",
        "",
        "## Aggregate metrics",
        "",
        "| Method | "
        + " | ".join(labels[metric] for metric in displayed_metrics)
        + " |",
        "|---|" + "|".join("---:" for _ in displayed_metrics) + "|",
    ]
    for arm in ARMS:
        cells = [
            f"{stats[arm][metric].median:.3f}"
            for metric in displayed_metrics
        ]
        lines.append(
            f"| {DISPLAY_NAMES[arm]} | " + " | ".join(cells) + " |"
        )
    lines.extend(
        [
            "",
            "Values are medians over 2,000 paired rows per method "
            "(500 images × 4 repetitions).",
            "",
            "## Speedups",
            "",
            "| Comparison | Type | Metric | Mean speedup | Median speedup |",
            "|---|---|---|---:|---:|",
        ]
    )
    for result in speedups:
        lines.append(
            f"| {DISPLAY_NAMES[result.baseline]} / "
            f"{DISPLAY_NAMES[result.current]} | "
            f"{result.comparison_type} | {result.metric} | "
            f"{result.mean_speedup:.3f}× | "
            f"{result.median_speedup:.3f}× |"
        )
    lines.extend(
        [
            "",
            "Vanilla-vs-LearnPruner is a **cross-backend raw-system "
            "comparison**: Vanilla uses official LLaVA while LearnPruner uses "
            "Transformers LLaVA. It must not be interpreted as a "
            "matched-backend algorithm-only speedup.",
            "",
            "Vanilla-vs-Ours is the matched official-LLaVA comparison. "
            "LearnPruner-vs-Ours is also a cross-backend raw-system ranking.",
            "",
            "## Warnings",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_cli(args: argparse.Namespace) -> None:
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise SummaryError("Invalid --run-id")
    required = {
        "--expected-n": (args.expected_n, EXPECTED_N),
        "--expected-repetitions": (
            args.expected_repetitions,
            EXPECTED_REPETITIONS,
        ),
        "--expected-output-tokens": (
            args.expected_output_tokens,
            EXPECTED_OUTPUT_TOKENS,
        ),
        "--physical-gpu-id": (
            str(args.physical_gpu_id),
            FORMAL_PHYSICAL_GPU_ID,
        ),
    }
    mismatches = {
        key: (observed, expected)
        for key, (observed, expected) in required.items()
        if observed != expected
    }
    if mismatches:
        raise SummaryError(
            "This summarizer accepts only the formal 500x4x32 protocol on "
            f"physical GPU {FORMAL_PHYSICAL_GPU_ID}: "
            f"{canonical_json(mismatches)}"
        )
    for name in (
        "max_latency_p90_p10_ratio",
        "max_clock_p90_p10_ratio",
        "max_cross_method_clock_median_ratio",
        "max_observer_e2e_ratio",
        "max_observer_asymmetry_ratio",
    ):
        if float(getattr(args, name)) <= 1.0:
            raise SummaryError(f"--{name.replace('_', '-')} must exceed one")


def run(args: argparse.Namespace) -> int:
    validate_cli(args)
    loaded = load_results(args)
    stats = build_metric_stats(loaded)
    speedups = build_speedups(loaded, stats)
    warnings = warning_messages(args, stats, speedups)

    output_dir = (
        args.result_root.expanduser().resolve()
        / PROTOCOL_VERSION
        / f"cuda{args.physical_gpu_id}"
        / "summary"
        / args.run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.tsv"
    speedups_path = output_dir / "speedups.tsv"
    summary_path = output_dir / "summary.md"
    gate_path = output_dir / "gate.json"
    write_metrics(metrics_path, stats)
    write_speedups(speedups_path, speedups)
    write_markdown(summary_path, args, stats, speedups, warnings)

    output_hashes = {
        path.name: file_sha256(path)
        for path in (metrics_path, speedups_path, summary_path)
    }
    gate = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": args.run_id,
        "passed": True,
        "status": "pass",
        "hard_failures": [],
        "warnings": warnings,
        "hard_gate_scope": [
            "protocol",
            "completeness",
            "input_and_provenance_hashes",
            "finite_numeric_values",
            "fixed_32_token_generation",
            "per_layer_token_trace",
            "instrumented_uninstrumented_generated_tail_hash_equality",
        ],
        "nonblocking_diagnostics": [
            "unlocked_clock_distribution",
            "latency_stability",
            "observer_overhead",
            "performance_thresholds",
        ],
        "validated_protocol": {
            "methods": list(ARMS),
            "physical_gpu_id": FORMAL_PHYSICAL_GPU_ID,
            "gpu_uuid": loaded.gpu_uuid,
            "clock_mode": args.clock_mode,
            "locked_sm_clock_mhz": (
                None
                if args.clock_mode == CLOCK_MODE_MONITORED_UNLOCKED
                else FORMAL_LOCKED_SM_CLOCK_MHZ
            ),
            "sample_sha256": loaded.sample_sha256,
            "samples_per_repetition": EXPECTED_N,
            "repetitions": EXPECTED_REPETITIONS,
            "rows_per_method": EXPECTED_N * EXPECTED_REPETITIONS,
            "sampling_seed": FORMAL_SEED,
            "warmup_generations_per_process": FORMAL_WARMUP,
            "output_tokens": EXPECTED_OUTPUT_TOKENS,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "batch_size": 1,
        },
        "comparison_semantics": {
            "vanilla_vs_ours": "matched_official_backend",
            "vanilla_vs_learnpruner": (
                "cross_backend_raw_system_comparison_not_algorithm_only"
            ),
            "learnpruner_vs_ours": "cross_backend_raw_system_comparison",
        },
        "input_provenance": loaded.input_provenance,
        "output_sha256": output_hashes,
        "learnpruner_checkpoint_usage": LEARNPRUNER_CHECKPOINT_USAGE,
    }
    gate_path.write_text(
        json.dumps(gate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote three-method v2 summary: {output_dir}")
    print("Hard gate: PASS")
    if warnings:
        print(f"Nonblocking warnings: {len(warnings)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (SummaryError, OSError, ValueError, ZeroDivisionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
