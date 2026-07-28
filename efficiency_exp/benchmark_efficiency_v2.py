#!/usr/bin/env python3
"""Strict, resumable inference-efficiency benchmark for one LLaVA method.

Protocol v2 measures an adjacent pair of token-identical fixed-length
generations. One has only outer events and is the authoritative E2E result;
the other records the first multimodal forward, time-to-first-token, and every
subsequent decode interval as instrumented diagnostics. Their order alternates.
No decode metric is obtained by subtracting an independent prefill run.

The formal run contains exactly three methods:

* ``vanilla`` (official LLaVA) versus ``ours`` (official LLaVA);
* ``learnpruner`` through its required Transformers LLaVA backend.

The Vanilla/LearnPruner row is reported only as a raw cross-backend system
comparison; it is not an implementation-matched algorithmic speedup.

Token-length tracing is performed once as an untimed validation probe.  No
token tracing hook is installed during a measured generation.
"""

from __future__ import annotations

import argparse
import csv
import functools
import gc
import hashlib
import inspect
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm import tqdm

try:
    from .efficiency_v2_runtime import (
        HERE,
        HF_MODEL,
        LEARNPRUNER_DIR,
        OFFICIAL_MODEL,
        OUR_REPO,
        OURS_CHECKPOINT,
        PROMPT,
        FirstLayerTokenTracer,
        ModelAdapter,
        PreparedInput,
        canonical_layer_tokens,
        file_sha256,
        git_info,
        kv_cache_bytes,
        load_rows,
        validate_environment,
    )
except ImportError:
    from efficiency_v2_runtime import (
        HERE,
        HF_MODEL,
        LEARNPRUNER_DIR,
        OFFICIAL_MODEL,
        OUR_REPO,
        OURS_CHECKPOINT,
        PROMPT,
        FirstLayerTokenTracer,
        ModelAdapter,
        PreparedInput,
        canonical_layer_tokens,
        file_sha256,
        git_info,
        kv_cache_bytes,
        load_rows,
        validate_environment,
    )


PROTOCOL_VERSION = "efficiency-v2"
DEFAULT_RESULT_ROOT = HERE / "results"
FORMAL_LEARNPRUNER_CHECKPOINT = Path(
    "/data1/chenzixuan/train_output/"
    "learnpruner_llava15_7b_paper_aligned_smoke_cuda2"
)
FORMAL_OFFICIAL_MODEL = OFFICIAL_MODEL.expanduser().resolve()
FORMAL_HF_MODEL = HF_MODEL.expanduser().resolve()
FORMAL_OURS_CHECKPOINT = OURS_CHECKPOINT.expanduser().resolve()
FORMAL_PHYSICAL_GPU_ID = "2"
FORMAL_LOCKED_SM_CLOCK_MHZ = 1980
CLOCK_MODE_MONITORED_UNLOCKED = "monitored_unlocked"
CLOCK_MODE_MANAGED_LOCKED = "managed_locked"
CLOCK_MODE_EXTERNAL_LOCKED = "external_locked"
CLOCK_MODES = (
    CLOCK_MODE_MONITORED_UNLOCKED,
    CLOCK_MODE_MANAGED_LOCKED,
    CLOCK_MODE_EXTERNAL_LOCKED,
)
FORMAL_DEFAULT_CLOCK_MODE = CLOCK_MODE_MONITORED_UNLOCKED
FORMAL_MAX_NEW_TOKENS = 32
FORMAL_SEED = 42
FORMAL_WARMUP = 20
FORMAL_OURS_BUDGET = 64
FORMAL_LEARNPRUNER_STAGE1_TOKENS = 111
FORMAL_LEARNPRUNER_STAGE2_TOKENS = 37
FORMAL_LEARNPRUNER_PRUNE_LAYER = 12
FORMAL_LEARNPRUNER_DIVERSITY_RATIO = 0.1
LEARNPRUNER_CHECKPOINT_USAGE = {
    "role": "latency-only",
    "paper_aligned": True,
    "accuracy_checkpoint": False,
    "note": (
        "not an accuracy checkpoint; fixed 111/37 token shapes are used only "
        "for paper-aligned architecture inference latency"
    ),
}
OURS_COMBINED_AUDIT_PATH = (
    HERE / "results/audits/ours_combined_gate_prefill_seed42_n32.json"
)
OURS_COMBINED_AUDIT_SCRIPT = HERE / "audit_ours_combined_gate.py"
OURS_COMBINED_AUDIT_SHA256 = (
    "2a19b236e122c5fce6c37197d4a9d0b7ced7752459ab467c7f7c750f50a16ea2"
)
OURS_COMBINED_AUDIT_SCHEMA = "ours-combined-gate-v2"
OURS_COMBINED_AUDIT_RUN_ID = "20260728T150611.940305Z-pid1369780"
OURS_COMBINED_FULL_OUTPUT_SHA256 = (
    "5de840343e7df8c772160542b440b0973682cf33caa5fd6112d253cbaeef00de"
)
OURS_COMBINED_PREFILL_OUTPUT_SHA256 = (
    "92e2b8b73c50060437115a111c981c70fa6136dded62dcd6c510d38de1e4d018"
)
OURS_COMBINED_AUDIT_CHECKS = {
    "capture_excluded": True,
    "combined_callback_trace_exact_cases": 32,
    "decode_callback_pairs_per_full_candidate": 31,
    "distinct_candidate_full_token_hashes": 32,
    "full_token_mismatch_count": 0,
    "graph_fallback_or_cache_miss_cases": 0,
    "implicit_mask_materialization_calls": 0,
    "prefill_callback_pairs_per_candidate": 1,
    "prefill_callback_trace_exact_cases": 32,
    "prefill_capture_count": 1,
    "prefill_replay_count": 66,
    "prefill_signature_mismatch_count": 0,
    "prefill_token_mismatch_count": 0,
    "same_decode_runner_identity_cases": 32,
    "same_prefill_runner_identity_cases": 32,
    "sample_count": 32,
    "triton_multirow_launch_observed": True,
    "triton_runtime_disabled": False,
    "triton_singleton_launch_count": 0,
}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
METHODS = ("vanilla", "ours", "learnpruner")
RESULT_FIELDS = [
    "protocol_version",
    "config_id",
    "run_id",
    "method",
    "arm_id",
    "generation_variant",
    "backend",
    "locked_sm_clock_mhz",
    "repetition",
    "sample_order",
    "dataset_index",
    "source",
    "image",
    "prompt_text_tokens",
    "input_sequence_tokens",
    "projector_input_tokens",
    "projector_output_tokens",
    "requested_output_tokens",
    "actual_output_tokens",
    "output_sequence_length",
    "generated_token_ids",
    "generated_tail_sha256",
    "uninstrumented_generated_tail_sha256",
    "forward_steps",
    "measurement_order",
    "cuda_graph_residency_warmup_tail_sha256",
    "cuda_graph_prefill_callback_runner_id",
    "cuda_graph_prefill_pair_status",
    "cuda_graph_decode_callback_steps",
    "cuda_graph_runner_cache_status",
    "triton_rms_runtime_status",
    "avg_visual_tokens",
    "estimated_prefill_tflops",
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
    "observer_e2e_cuda_delta_ms",
    "observer_e2e_cuda_ratio",
    "observer_e2e_wall_delta_ms",
    "observer_e2e_wall_ratio",
    "throughput_output_tokens_per_s",
    "throughput_uninstrumented_output_tokens_per_s",
    "decode_throughput_tokens_per_s",
    "prefill_forward_cuda_ms",
    "decode_forward_cuda_ms",
    "decode_step_cuda_ms",
    "decode_step_wall_ms",
    "analytical_prefill_kv_cache_mb",
    "baseline_allocated_gb",
    "peak_allocated_gb",
    "incremental_peak_allocated_gb",
    "peak_reserved_gb",
    "gpu_sm_clock_mhz_before",
    "gpu_sm_clock_mhz_after",
    "gpu_power_w_before",
    "gpu_power_w_after",
    "gpu_temperature_c_before",
    "gpu_temperature_c_after",
    "gpu_utilization_pct_before",
    "gpu_utilization_pct_after",
    "layer_prefill_tokens",
    "pruner_stats",
]


class ProtocolError(RuntimeError):
    """A protocol violation that invalidates a benchmark run."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--sample-parquet",
        type=Path,
        default=HERE / "samples" / "detailcaps_seed42_n500.parquet",
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional explicit TSV path. It must contain protocol version, physical "
            "GPU, method, and computed config id. Prefer --result-root."
        ),
    )
    parser.add_argument("--run-id", default="formal")
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--warmup", type=int, default=FORMAL_WARMUP)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=FORMAL_MAX_NEW_TOKENS,
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--physical-gpu-id",
        default=FORMAL_PHYSICAL_GPU_ID,
        help="Physical NVML/nvidia-smi GPU index before CUDA_VISIBLE_DEVICES remapping.",
    )
    parser.add_argument(
        "--expected-cuda-visible-devices",
        default=FORMAL_PHYSICAL_GPU_ID,
        help="Exact CUDA_VISIBLE_DEVICES value required by the process.",
    )
    parser.add_argument(
        "--clock-mode",
        choices=CLOCK_MODES,
        default=FORMAL_DEFAULT_CLOCK_MODE,
        help=(
            "GPU clock policy recorded in protocol metadata. The default "
            "monitors an unlocked GPU without requiring administrator access; "
            "managed_locked/external_locked retain fixed-clock compatibility."
        ),
    )
    parser.add_argument(
        "--locked-sm-clock-mhz",
        type=int,
        default=None,
        help=(
            "Fixed SM clock for managed_locked/external_locked modes. It must "
            f"be exactly {FORMAL_LOCKED_SM_CLOCK_MHZ} MHz. Leave unset for "
            "monitored_unlocked."
        ),
    )
    parser.add_argument("--official-model", type=Path, default=OFFICIAL_MODEL)
    parser.add_argument("--hf-model", type=Path, default=HF_MODEL)
    parser.add_argument(
        "--learnpruner-checkpoint",
        type=Path,
        default=FORMAL_LEARNPRUNER_CHECKPOINT,
    )
    parser.add_argument("--ours-checkpoint", type=Path, default=OURS_CHECKPOINT)
    parser.add_argument(
        "--ours-budget",
        type=int,
        choices=(64, 128, 192),
        default=FORMAL_OURS_BUDGET,
    )
    parser.add_argument(
        "--learnpruner-stage1-tokens",
        type=int,
        default=FORMAL_LEARNPRUNER_STAGE1_TOKENS,
    )
    parser.add_argument(
        "--learnpruner-stage2-tokens",
        type=int,
        default=FORMAL_LEARNPRUNER_STAGE2_TOKENS,
    )
    parser.add_argument(
        "--learnpruner-prune-layer",
        type=int,
        default=FORMAL_LEARNPRUNER_PRUNE_LAYER,
    )
    parser.add_argument(
        "--learnpruner-diversity-ratio",
        type=float,
        default=FORMAL_LEARNPRUNER_DIVERSITY_RATIO,
    )
    parser.add_argument(
        "--ours-implicit-causal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Explicitly control LEARNABLE_PRUNE_IMPLICIT_CAUSAL. Formal Ours "
            "uses --ours-implicit-causal; the slow semantic reference uses "
            "--no-ours-implicit-causal."
        ),
    )
    parser.add_argument(
        "--ours-fixed-length-greedy",
        choices=("on", "off"),
        default="on",
        help=(
            "Explicitly control LEARNABLE_PRUNE_FIXED_LENGTH_GREEDY. For Ours, "
            "'off' is recorded as the slow semantic-reference arm."
        ),
    )
    parser.add_argument(
        "--ours-triton-rms",
        choices=("on", "off"),
        default="off",
        help=(
            "Explicitly control the exact prefill-only "
            "LEARNABLE_PRUNE_TRITON_RMS specialization. Formal Ours uses 'on'; "
            "decode sequence length 1 must retain authoritative torch RMS."
        ),
    )
    parser.add_argument(
        "--ours-cuda-graph-prefill",
        choices=("on", "off"),
        default="off",
        help=(
            "Explicitly control LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL. Formal Ours "
            "uses 'on' as a steady-state optimization and requires one warmed "
            "prefill replay callback pair in every instrumented generation; "
            "capture/initialization are warmup-only and excluded from latency."
        ),
    )
    parser.add_argument(
        "--ours-cuda-graph-decode",
        choices=("on", "off"),
        default="off",
        help=(
            "Explicitly control LEARNABLE_PRUNE_CUDA_GRAPH_DECODE. Formal Ours "
            "uses 'on' as a steady-state optimization and requires exact replay "
            "callbacks for every instrumented decode step; graph capture is "
            "warmup-only and excluded from latency."
        ),
    )
    parser.add_argument(
        "--ours-static-kv-decode",
        choices=("on", "off"),
        default="off",
        help=(
            "Explicitly control LEARNABLE_PRUNE_STATIC_KV_DECODE. Protocol v2 "
            "rejects 'on' until full-sample parity validation passes."
        ),
    )
    parser.add_argument(
        "--skip-trace-validation",
        action="store_true",
        help="Skip the untimed one-sample token-length validation probe.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def optional_sha256(path: Path) -> str | None:
    return file_sha256(path) if path.is_file() else None


def combined_audit_provenance(
    args: argparse.Namespace,
    sample_path: Path,
    sample_sha256: str,
) -> dict[str, Any] | None:
    """Load and strictly bind the persisted combined Ours admission artifact."""
    if args.method != "ours":
        return None
    artifact_path = OURS_COMBINED_AUDIT_PATH.expanduser().resolve()
    if not artifact_path.is_file():
        raise ProtocolError(
            f"Missing required combined Ours audit artifact: {artifact_path}"
        )
    observed_artifact_hash = file_sha256(artifact_path)
    if observed_artifact_hash != OURS_COMBINED_AUDIT_SHA256:
        raise ProtocolError(
            "Combined Ours audit artifact hash mismatch: "
            f"observed={observed_artifact_hash}, "
            f"expected={OURS_COMBINED_AUDIT_SHA256}"
        )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    checks = artifact.get("checks", {})
    protocol = artifact.get("protocol", {})
    paths = artifact.get("paths", {})
    digests = artifact.get("digests", {})
    path_validation = artifact.get("path_validation", {})
    capture = artifact.get("capture", {})
    final_prefill_diagnostics = artifact.get(
        "prefill_graph_diagnostics_final",
        {},
    )
    output_hashes = artifact.get("output_hashes", {})
    expected_core = {
        "schema_version": OURS_COMBINED_AUDIT_SCHEMA,
        "run_id": OURS_COMBINED_AUDIT_RUN_ID,
        "status": "pass",
        "passed": True,
    }
    core_mismatches = {
        key: (artifact.get(key), value)
        for key, value in expected_core.items()
        if artifact.get(key) != value
    }
    expected_checks = dict(OURS_COMBINED_AUDIT_CHECKS)
    check_mismatches = (
        {}
        if checks == expected_checks
        else {"observed": checks, "expected": expected_checks}
    )
    expected_protocol = {
        "dataset": "CAPTURE / DetailCaps-4870",
        "selection": "first 32 parquet rows in stored order",
        "num_samples": 32,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "batch_size": 1,
        "min_new_tokens": 32,
        "max_new_tokens": 32,
        "capture_excluded": True,
        "prefill_definition": (
            "outer CUDA event around independent generate(min=max=1)"
        ),
        "e2e_definition": (
            "outer CUDA event around independent "
            "generate(min=max=max_new_tokens)"
        ),
        "decode_definition": (
            "paired E2E minus matching-arm independent prefill"
        ),
        "preprocessing_in_timing": False,
        "vision_encoder_in_timing": True,
        "prefill_graph_scope": (
            "three decoder segments only; predictor/SCOPE, middle "
            "scoring/TopK/sort/packing, and final wipe remain eager"
        ),
    }
    protocol_mismatches = {
        key: (protocol.get(key), value)
        for key, value in expected_protocol.items()
        if protocol.get(key) != value
    }
    common_arm_switches = {
        "LEARNABLE_PRUNE_CACHED_BOOL_MASK": "0",
        "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL_CACHE_SIZE": "2",
        "LEARNABLE_PRUNE_DIRECT_SDPA": "1",
        "LEARNABLE_PRUNE_PROFILE_INTERNAL": "0",
        "LEARNABLE_PRUNE_STATIC_KV_DECODE": "0",
    }
    expected_candidate_switches = {
        **common_arm_switches,
        "LEARNABLE_PRUNE_CUDA_GRAPH_DECODE": "1",
        "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL": "1",
        "LEARNABLE_PRUNE_FIXED_LENGTH_GREEDY": "1",
        "LEARNABLE_PRUNE_IMPLICIT_CAUSAL": "1",
        "LEARNABLE_PRUNE_TRITON_RMS": "1",
    }
    expected_eager_switches = {
        **common_arm_switches,
        "LEARNABLE_PRUNE_CUDA_GRAPH_DECODE": "0",
        "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL": "0",
        "LEARNABLE_PRUNE_FIXED_LENGTH_GREEDY": "1",
        "LEARNABLE_PRUNE_IMPLICIT_CAUSAL": "1",
        "LEARNABLE_PRUNE_TRITON_RMS": "1",
    }
    expected_slow_switches = {
        **common_arm_switches,
        "LEARNABLE_PRUNE_CUDA_GRAPH_DECODE": "0",
        "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL": "0",
        "LEARNABLE_PRUNE_FIXED_LENGTH_GREEDY": "0",
        "LEARNABLE_PRUNE_IMPLICIT_CAUSAL": "0",
        "LEARNABLE_PRUNE_TRITON_RMS": "0",
    }
    arms = protocol.get("arms", {})
    expected_arms = {
        "candidate": expected_candidate_switches,
        "eager": expected_eager_switches,
        "slow": expected_slow_switches,
    }
    arm_mismatches = {
        arm: (arms.get(arm), expected)
        for arm, expected in expected_arms.items()
        if arms.get(arm) != expected
    }
    expected_output_hashes = {
        **{
            arm: OURS_COMBINED_FULL_OUTPUT_SHA256
            for arm in ("slow", "eager", "candidate")
        },
        **{
            f"{arm}_prefill": OURS_COMBINED_PREFILL_OUTPUT_SHA256
            for arm in ("slow", "eager", "candidate")
        },
    }
    output_hash_mismatches = {
        arm: (
            output_hashes.get(arm, {}).get(
                "aggregate_canonical_json_sha256"
            ),
            expected_hash,
        )
        for arm, expected_hash in expected_output_hashes.items()
        if output_hashes.get(arm, {}).get(
            "aggregate_canonical_json_sha256"
        )
        != expected_hash
    }
    expected_digests = {
        "sample_parquet_sha256": sample_sha256,
        "official_modeling_source_sha256": optional_sha256(
            OUR_REPO
            / "llava/model/learnable_prune_lightweight_scope_finalwipe/"
            "official_modeling.py"
        ),
        "llava_llama_source_sha256": optional_sha256(
            OUR_REPO / "llava/model/language_model/llava_llama.py"
        ),
        "official_model_config_sha256": optional_sha256(
            args.official_model.expanduser().resolve() / "config.json"
        ),
        "checkpoint_config_sha256": optional_sha256(
            args.ours_checkpoint.expanduser().resolve()
            / "learnable_prune_config.pt"
        ),
        "checkpoint_predictor_sha256": optional_sha256(
            args.ours_checkpoint.expanduser().resolve() / "predictor.pt"
        ),
        "script_sha256": optional_sha256(
            OURS_COMBINED_AUDIT_SCRIPT.expanduser().resolve()
        ),
    }
    digest_mismatches = {
        key: (digests.get(key), value)
        for key, value in expected_digests.items()
        if value is None or digests.get(key) != value
    }
    expected_paths = {
        "artifact": str(artifact_path),
        "script": str(OURS_COMBINED_AUDIT_SCRIPT.expanduser().resolve()),
        "sample_parquet": str(sample_path),
        "official_model": str(args.official_model.expanduser().resolve()),
        "ours_checkpoint": str(args.ours_checkpoint.expanduser().resolve()),
    }
    path_mismatches = {
        key: (paths.get(key), value)
        for key, value in expected_paths.items()
        if paths.get(key) != value
    }
    path_validation_expected = {
        "outside_measured_samples": True,
        "multirow_eligible_count": 66,
        "singleton_call_count": 130,
        "singleton_eligible_count": 0,
        "singleton_launch_count": 0,
        "token_equal": True,
        "triton_runtime_disabled_before": False,
        "triton_runtime_disabled_after": False,
        "implicit_prepare_mask_calls": 0,
        "slow_prepare_mask_calls": 3,
    }
    path_validation_mismatches = {
        key: (path_validation.get(key), value)
        for key, value in path_validation_expected.items()
        if path_validation.get(key) != value
    }

    def positive_finite(value: Any) -> bool:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(number) and number > 0.0

    def prefill_diagnostics_mismatches(
        value: Any,
        *,
        cache_hits: int,
        replay_successes: int,
        last_status: str,
        label: str,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {label: (value, "diagnostics object")}
        expected_scalars = {
            "enabled": False,
            "cache_capacity": 2,
            "cache_entries": 1,
            "cache_hits": cache_hits,
            "cache_misses": 1,
            "cached_failures": 0,
            "capture_failures": 0,
            "captures": 1,
            "eager_fallbacks": 0,
            "evictions": 0,
            "hook_fallbacks": 0,
            "last_error": None,
            "last_runner_id": 1,
            "last_status": last_status,
            "replay_failures": 0,
            "replay_successes": replay_successes,
            "signature_mismatches": 0,
        }
        mismatch: dict[str, Any] = {
            key: (value.get(key), expected)
            for key, expected in expected_scalars.items()
            if value.get(key) != expected
        }
        runners = value.get("runners")
        if not isinstance(runners, list) or len(runners) != 1:
            mismatch["runners"] = (runners, "one healthy runner")
            return {label: mismatch}
        runner = runners[0]
        expected_runner = {
            "runner_id": 1,
            "segment_shapes": [186, 81, 49],
            "replay_count": replay_successes,
        }
        if not isinstance(runner, dict):
            mismatch["runner"] = (runner, expected_runner)
            return {label: mismatch}
        for key, expected in expected_runner.items():
            if runner.get(key) != expected:
                mismatch[f"runner.{key}"] = (runner.get(key), expected)
        capture_wall_ms = runner.get("capture_wall_ms")
        initialization_wall_ms = runner.get("initialization_wall_ms")
        if not positive_finite(capture_wall_ms):
            mismatch["runner.capture_wall_ms"] = (
                capture_wall_ms,
                "positive finite",
            )
        if not positive_finite(initialization_wall_ms):
            mismatch["runner.initialization_wall_ms"] = (
                initialization_wall_ms,
                "positive finite",
            )
        if value.get("total_capture_wall_ms") != capture_wall_ms:
            mismatch["total_capture_wall_ms"] = (
                value.get("total_capture_wall_ms"),
                capture_wall_ms,
            )
        if (
            value.get("total_initialization_wall_ms")
            != initialization_wall_ms
        ):
            mismatch["total_initialization_wall_ms"] = (
                value.get("total_initialization_wall_ms"),
                initialization_wall_ms,
            )
        return {label: mismatch} if mismatch else {}

    capture_mismatches: dict[str, Any] = {}
    expected_capture_scalars = {
        "excluded_from_paired_timings": True,
        "decode_runner_cache_entries": 1,
        "prefill_runner_cache_entries": 1,
        "prefill_runner_id": 1,
        "callback_trace": [],
    }
    for key, expected in expected_capture_scalars.items():
        if capture.get(key) != expected:
            capture_mismatches[key] = (capture.get(key), expected)
    for key in (
        "cuda_ms",
        "wall_ms",
        "decode_runner_identity",
        "prefill_runner_identity",
        "estimated_incremental_cuda_ms_vs_candidate_p50",
        "estimated_incremental_wall_ms_vs_candidate_p50",
    ):
        if not positive_finite(capture.get(key)):
            capture_mismatches[key] = (
                capture.get(key),
                "positive finite",
            )
    expected_post_capture_trace = [
        ["prefill_start", 1],
        ["prefill_end", 1],
        *[
            [phase, step]
            for step in range(1, 32)
            for phase in ("decode_start", "decode_end")
        ],
    ]
    if (
        capture.get("post_capture_validation_trace")
        != expected_post_capture_trace
    ):
        capture_mismatches["post_capture_validation_trace"] = (
            capture.get("post_capture_validation_trace"),
            expected_post_capture_trace,
        )
    capture_mismatches.update(
        prefill_diagnostics_mismatches(
            capture.get("prefill_diagnostics"),
            cache_hits=0,
            replay_successes=1,
            last_status="captured_then_replayed",
            label="prefill_diagnostics",
        )
    )
    capture_mismatches.update(
        prefill_diagnostics_mismatches(
            capture.get("post_capture_prefill_diagnostics"),
            cache_hits=1,
            replay_successes=2,
            last_status="cache_hit_replayed",
            label="post_capture_prefill_diagnostics",
        )
    )
    capture_mismatches.update(
        prefill_diagnostics_mismatches(
            final_prefill_diagnostics,
            cache_hits=65,
            replay_successes=66,
            last_status="cache_hit_replayed",
            label="prefill_graph_diagnostics_final",
        )
    )
    mismatches = {
        name: value
        for name, value in {
            "core": core_mismatches,
            "checks": check_mismatches,
            "protocol": protocol_mismatches,
            "arms": arm_mismatches,
            "output_hashes": output_hash_mismatches,
            "digests": digest_mismatches,
            "paths": path_mismatches,
            "path_validation": path_validation_mismatches,
            "capture": capture_mismatches,
        }.items()
        if value
    }
    if mismatches:
        raise ProtocolError(
            "Persisted combined Ours audit failed strict revalidation: "
            f"{canonical_json(mismatches)}"
        )
    summary = artifact.get("summary", {})
    prefill_runner = capture["prefill_diagnostics"]["runners"][0]
    return {
        "artifact_path": str(artifact_path),
        "artifact_sha256": observed_artifact_hash,
        **expected_core,
        "protocol": expected_protocol,
        "checks": expected_checks,
        "output_hashes": {
            "full_aggregate_canonical_json_sha256": (
                OURS_COMBINED_FULL_OUTPUT_SHA256
            ),
            "prefill_aggregate_canonical_json_sha256": (
                OURS_COMBINED_PREFILL_OUTPUT_SHA256
            ),
        },
        "capture": {
            "cuda_ms": float(capture["cuda_ms"]),
            "wall_ms": float(capture["wall_ms"]),
            "excluded_from_paired_timings": True,
            "decode_runner_cache_entries": 1,
            "prefill_runner_cache_entries": 1,
            "prefill_runner_id": 1,
            "prefill_segment_capture_wall_ms": float(
                prefill_runner["capture_wall_ms"]
            ),
            "prefill_initialization_wall_ms": float(
                prefill_runner["initialization_wall_ms"]
            ),
            "prefill_replay_counts": {
                "capture": 1,
                "post_capture": 2,
                "final": 66,
            },
        },
        "runtime_probe": path_validation_expected,
        "p50": {
            metric: float(summary[metric]["p50"])
            for metric in (
                "slow_prefill_cuda_ms",
                "eager_prefill_cuda_ms",
                "candidate_prefill_cuda_ms",
                "slow_e2e_cuda_ms",
                "eager_e2e_cuda_ms",
                "candidate_e2e_cuda_ms",
                "slow_decode_cuda_ms",
                "eager_decode_cuda_ms",
                "candidate_decode_cuda_ms",
                "candidate_vs_slow_e2e_speedup",
                "candidate_vs_eager_e2e_speedup",
                "candidate_vs_slow_decode_speedup",
                "candidate_vs_eager_decode_speedup",
            )
        },
        "digests": expected_digests,
        "timing_role": (
            "admission evidence only; its independent-prefill subtraction "
            "decode is not used as formal v2 TPOT/decode"
        ),
    }


def selected_source_hashes(args: argparse.Namespace) -> dict[str, str | None]:
    from transformers.models.llava import modeling_llava as hf_llava_modeling

    learnpruner_modeling = LEARNPRUNER_DIR / "modeling_llava_learnpruner.py"
    ours_modeling = (
        OUR_REPO
        / "llava/model/learnable_prune_lightweight_scope_finalwipe/official_modeling.py"
    )
    vanilla_modeling = OUR_REPO / "llava/model/language_model/llava_llama.py"
    return {
        "benchmark_v2": optional_sha256(Path(__file__).resolve()),
        "benchmark_adapter": optional_sha256(
            HERE / "efficiency_v2_runtime.py"
        ),
        "runner_v2": optional_sha256(HERE / "run_benchmark_v2.sh"),
        "summarizer_v2": optional_sha256(HERE / "summarize_results_v2.py"),
        "combined_audit_script": (
            optional_sha256(OURS_COMBINED_AUDIT_SCRIPT)
            if args.method == "ours"
            else None
        ),
        "combined_audit_artifact": (
            optional_sha256(OURS_COMBINED_AUDIT_PATH)
            if args.method == "ours"
            else None
        ),
        "vanilla_modeling": optional_sha256(vanilla_modeling),
        "hf_llava_modeling": (
            optional_sha256(Path(inspect.getfile(hf_llava_modeling)))
            if args.method in {"vanilla_hf", "learnpruner"}
            else None
        ),
        "ours_modeling": (
            optional_sha256(ours_modeling) if args.method == "ours" else None
        ),
        "learnpruner_modeling": (
            optional_sha256(learnpruner_modeling)
            if args.method == "learnpruner"
            else None
        ),
    }


def checkpoint_hashes(args: argparse.Namespace) -> dict[str, str | None]:
    ours = args.ours_checkpoint.expanduser().resolve()
    learnpruner = args.learnpruner_checkpoint.expanduser().resolve()
    return {
        "official_config": optional_sha256(
            args.official_model.expanduser().resolve() / "config.json"
        ),
        "hf_config": optional_sha256(args.hf_model.expanduser().resolve() / "config.json"),
        "ours_config": (
            optional_sha256(ours / "learnable_prune_config.pt")
            if args.method == "ours"
            else None
        ),
        "ours_predictor": (
            optional_sha256(ours / "predictor.pt") if args.method == "ours" else None
        ),
        "learnpruner_config": (
            optional_sha256(learnpruner / "learnpruner_config.pt")
            if args.method == "learnpruner"
            else None
        ),
        "learnpruner_predictor": (
            optional_sha256(learnpruner / "predictor.pt")
            if args.method == "learnpruner"
            else None
        ),
    }


def runtime_identity() -> dict[str, str | None]:
    """Return resume-critical software identity, not just descriptive metadata."""
    transformers = __import__("transformers")
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": __import__("numpy").__version__,
        "cuda_runtime": torch.version.cuda,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "cudnn_version": (
            str(torch.backends.cudnn.version())
            if torch.backends.cudnn.version() is not None
            else None
        ),
        "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
        "cuda_matmul_allow_tf32": str(
            bool(torch.backends.cuda.matmul.allow_tf32)
        ).lower(),
        "cudnn_allow_tf32": str(
            bool(torch.backends.cudnn.allow_tf32)
        ).lower(),
    }


def learnpruner_checkpoint_usage(args: argparse.Namespace) -> dict[str, Any]:
    selected = args.learnpruner_checkpoint.expanduser().resolve()
    if selected == FORMAL_LEARNPRUNER_CHECKPOINT.expanduser().resolve():
        return dict(LEARNPRUNER_CHECKPOINT_USAGE)
    return {
        "role": "compatibility-sanity-only",
        "paper_aligned": False,
        "accuracy_checkpoint": False,
        "note": (
            "non-formal checkpoint override; excluded from the protocol-v2 "
            "primary table and hard gate"
        ),
    }


def arm_id(args: argparse.Namespace) -> str:
    if args.method == "ours" and args.ours_fixed_length_greedy == "off":
        return "ours_slow"
    return args.method


def generation_variant(args: argparse.Namespace) -> str:
    if args.method != "ours":
        return "generation_mixin"
    fast = "fast" if args.ours_fixed_length_greedy == "on" else "slow"
    triton = (
        "triton_prefill_rms"
        if args.ours_triton_rms == "on"
        else "torch_rms"
    )
    prefill = (
        "cuda_graph_prefill"
        if args.ours_cuda_graph_prefill == "on"
        else "eager_prefill"
    )
    decode = (
        "cuda_graph_decode"
        if args.ours_cuda_graph_decode == "on"
        else "eager_decode"
    )
    return f"{fast}_{triton}_{prefill}_{decode}"


@dataclass(frozen=True)
class GpuSnapshot:
    uuid: str | None = None
    name: str | None = None
    driver_version: str | None = None
    sm_clock_mhz: float | None = None
    power_w: float | None = None
    temperature_c: float | None = None
    utilization_pct: float | None = None


class NvmlMonitor:
    """Low-overhead, out-of-timed-region clock/power/temperature sampling."""

    def __init__(self, physical_gpu_id: str):
        self.physical_gpu_id = int(physical_gpu_id)
        self._pynvml: Any = None
        self._handle: Any = None
        self.error: str | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.physical_gpu_id)
        except Exception as exc:  # pragma: no cover - depends on the GPU host
            self.error = repr(exc)

    def snapshot(self) -> GpuSnapshot:
        if self._pynvml is None or self._handle is None:
            return GpuSnapshot()
        nvml = self._pynvml
        try:
            utilization = nvml.nvmlDeviceGetUtilizationRates(self._handle)
            uuid = nvml.nvmlDeviceGetUUID(self._handle)
            name = nvml.nvmlDeviceGetName(self._handle)
            driver_version = nvml.nvmlSystemGetDriverVersion()
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8")
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode("utf-8")
            return GpuSnapshot(
                uuid=str(uuid),
                name=str(name),
                driver_version=str(driver_version),
                sm_clock_mhz=float(
                    nvml.nvmlDeviceGetClockInfo(self._handle, nvml.NVML_CLOCK_SM)
                ),
                power_w=float(nvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0,
                temperature_c=float(
                    nvml.nvmlDeviceGetTemperature(
                        self._handle, nvml.NVML_TEMPERATURE_GPU
                    )
                ),
                utilization_pct=float(utilization.gpu),
            )
        except Exception as exc:  # pragma: no cover - depends on the GPU host
            self.error = repr(exc)
            return GpuSnapshot()

    def close(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


def validate_nvml_snapshot(snapshot: GpuSnapshot, monitor: NvmlMonitor) -> None:
    required = (
        "uuid",
        "name",
        "driver_version",
        "sm_clock_mhz",
        "power_w",
        "temperature_c",
        "utilization_pct",
    )
    missing = [name for name in required if getattr(snapshot, name) is None]
    if missing:
        raise ProtocolError(
            "Strict protocol v2 requires NVML provenance/telemetry before model "
            f"load; missing={missing}, error={monitor.error}"
        )


def validate_formal_sm_clock(
    snapshot: GpuSnapshot,
    *,
    clock_mode: str,
    label: str,
) -> None:
    """Validate one NVML clock observation under the selected formal policy."""
    observed = snapshot.sm_clock_mhz
    if observed is None or not math.isfinite(float(observed)) or float(observed) <= 0.0:
        raise ProtocolError(
            "Formal protocol v2 requires a positive finite monitored SM clock; "
            f"observed={observed!r} at {label}"
        )
    if (
        clock_mode != CLOCK_MODE_MONITORED_UNLOCKED
        and abs(float(observed) - FORMAL_LOCKED_SM_CLOCK_MHZ) > 0.5
    ):
        raise ProtocolError(
            f"Clock mode {clock_mode!r} requires a locked "
            f"{FORMAL_LOCKED_SM_CLOCK_MHZ} MHz SM clock; "
            f"observed={observed!r} at {label}"
        )


def protocol_payload(
    args: argparse.Namespace,
    sample_path: Path,
    sample_sha256: str,
    gpu: GpuSnapshot,
) -> dict[str, Any]:
    combined_audit = combined_audit_provenance(
        args,
        sample_path,
        sample_sha256,
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": args.run_id,
        "method": args.method,
        "arm_id": arm_id(args),
        "generation_variant": generation_variant(args),
        "backend_family": (
            "official_llava" if args.method in {"vanilla", "ours"} else "huggingface_llava"
        ),
        "physical_gpu_id": str(args.physical_gpu_id),
        "gpu_uuid": gpu.uuid,
        "gpu_name": gpu.name,
        "gpu_driver_version": gpu.driver_version,
        "clock_mode": args.clock_mode,
        "locked_sm_clock_mhz": args.locked_sm_clock_mhz,
        "sample_path": str(sample_path),
        "sample_sha256": sample_sha256,
        "num_samples": int(args.num_samples),
        "sampling_seed": int(args.seed),
        "prompt": PROMPT,
        "warmup": int(args.warmup),
        "max_new_tokens": int(args.max_new_tokens),
        "dtype": args.dtype,
        "attention_implementation": args.attn_implementation,
        "batch_size": 1,
        "greedy": True,
        "use_cache": True,
        "force_fixed_output_length": True,
        "trace_validation_enabled": not args.skip_trace_validation,
        "measurement_pair": {
            "generations_per_sample": 2,
            "authoritative": "uninstrumented_outer_events",
            "diagnostic": "instrumented_direct_stages",
            "order": "(repetition + sample_order) parity",
            "require_generated_tail_equality": True,
            "graph_replay_evidence": (
                "one exact prefill callback pair plus exact ordered decode "
                "callbacks in every instrumented generation; both adjacent "
                "calls must add warmed cache-hit replays, and capture, fallback, "
                "cache replacement, or replay failure is fatal when enabled"
            ),
        },
        "paths": {
            "official_model": str(args.official_model.expanduser().resolve()),
            "hf_model": str(args.hf_model.expanduser().resolve()),
            "ours_checkpoint": str(args.ours_checkpoint.expanduser().resolve()),
            "learnpruner_checkpoint": str(
                args.learnpruner_checkpoint.expanduser().resolve()
            ),
        },
        "learnpruner_checkpoint_usage": learnpruner_checkpoint_usage(args),
        "ours_combined_audit": combined_audit,
        "pruning": {
            "ours_budget": int(args.ours_budget),
            "ours_cached_bool_mask": "off",
            "ours_profile_internal": "off",
            "ours_implicit_causal": bool(args.ours_implicit_causal),
            "ours_fixed_length_greedy": args.ours_fixed_length_greedy,
            "ours_triton_rms": args.ours_triton_rms,
            "ours_cuda_graph_prefill": args.ours_cuda_graph_prefill,
            "ours_cuda_graph_prefill_cache_size": 2,
            "ours_cuda_graph_decode": args.ours_cuda_graph_decode,
            "ours_static_kv_decode": args.ours_static_kv_decode,
            "learnpruner_stage1_tokens": int(args.learnpruner_stage1_tokens),
            "learnpruner_stage2_tokens": int(args.learnpruner_stage2_tokens),
            "learnpruner_prune_layer": int(args.learnpruner_prune_layer),
            "learnpruner_diversity_ratio": float(args.learnpruner_diversity_ratio),
        },
        "optimization_validation": {
            "fixed_length_greedy": {
                "formal_enabled": (
                    args.method == "ours"
                    and args.ours_fixed_length_greedy == "on"
                ),
                "challenging_cases_checked": 6,
                "generated_sequence_mismatches": 0,
            },
            "triton_rms": {
                "formal_enabled": (
                    args.method == "ours"
                    and args.ours_fixed_length_greedy == "on"
                    and args.ours_triton_rms == "on"
                ),
                "decision": (
                    "enabled_exact_prefill_only_with_native_runtime_probe"
                    if (
                        args.method == "ours"
                        and args.ours_fixed_length_greedy == "on"
                        and args.ours_triton_rms == "on"
                    )
                    else "disabled_for_non_rms_or_semantic_reference_arm"
                ),
                "domain": {
                    "dtype": "bfloat16",
                    "hidden_size": 4096,
                    "minimum_sequence_length": 16,
                    "decode_sequence_length_one": "authoritative_torch_fallback",
                    "pinned_torch_git_version": (
                        "5811a8d7da873dd699ff6687092c225caffcf1bb"
                    ),
                },
                "superseded_unrestricted_kernel_audit": {
                    "audit_cases": 32,
                    "generated_sequence_mismatches": 6,
                    "decision": "replaced_by_exact_prefill_only_kernel",
                },
                "combined_gate_artifact_sha256": (
                    combined_audit["artifact_sha256"]
                    if combined_audit is not None
                    else None
                ),
                "formal_run_requirements": {
                    "untimed_optimized_reference_hash_equality": True,
                    "multirow_launch_count_gt_zero": True,
                    "singleton_eligibility_false": True,
                    "singleton_launch_count": 0,
                    "runtime_fallback_is_fatal": True,
                    "adjacent_instrumented_uninstrumented_hash_equality": True,
                    "full_fast_slow_hash_equality": True,
                },
            },
            "cuda_graph_decode": {
                "formal_enabled": (
                    args.method == "ours"
                    and args.ours_fixed_length_greedy == "on"
                    and args.ours_cuda_graph_decode == "on"
                ),
                "decision": (
                    "enabled_for_formal_steady_state_with_native_per_row_gate"
                    if (
                        args.method == "ours"
                        and args.ours_fixed_length_greedy == "on"
                        and args.ours_cuda_graph_decode == "on"
                    )
                    else "disabled_for_non_graph_or_semantic_reference_arm"
                ),
                "combined_gate_artifact_sha256": (
                    combined_audit["artifact_sha256"]
                    if combined_audit is not None
                    else None
                ),
                "formal_run_requirements": {
                    "callback_steps_per_instrumented_generation": (
                        int(args.max_new_tokens) - 1
                        if (
                            args.method == "ours"
                            and args.ours_fixed_length_greedy == "on"
                            and args.ours_cuda_graph_decode == "on"
                        )
                        else 0
                    ),
                    "callback_step_sequence": (
                        "1..max_new_tokens-1_interleaved_start_end"
                    ),
                    "fallback_is_fatal": True,
                    "single_runner_identity_across_adjacent_pair": True,
                    "adjacent_instrumented_uninstrumented_hash_equality": True,
                    "full_fast_slow_hash_equality": True,
                    "capture_excluded_from_steady_state": True,
                },
            },
            "cuda_graph_prefill": {
                "formal_enabled": (
                    args.method == "ours"
                    and args.ours_fixed_length_greedy == "on"
                    and args.ours_cuda_graph_prefill == "on"
                ),
                "decision": (
                    "enabled_for_formal_steady_state_with_native_per_pair_gate"
                    if (
                        args.method == "ours"
                        and args.ours_fixed_length_greedy == "on"
                        and args.ours_cuda_graph_prefill == "on"
                    )
                    else "disabled_for_non_graph_or_semantic_reference_arm"
                ),
                "combined_gate_artifact_sha256": (
                    combined_audit["artifact_sha256"]
                    if combined_audit is not None
                    else None
                ),
                "formal_run_requirements": {
                    "callback_pairs_per_instrumented_generation": (
                        1
                        if (
                            args.method == "ours"
                            and args.ours_fixed_length_greedy == "on"
                            and args.ours_cuda_graph_prefill == "on"
                        )
                        else 0
                    ),
                    "callback_sequence": (
                        "prefill_start_then_prefill_end_with_same_runner_id"
                    ),
                    "adjacent_pair_cache_hit_replays": (
                        2
                        if (
                            args.method == "ours"
                            and args.ours_fixed_length_greedy == "on"
                            and args.ours_cuda_graph_prefill == "on"
                        )
                        else 0
                    ),
                    "fallback_is_fatal": True,
                    "single_runner_identity_across_adjacent_pair": True,
                    "capture_and_initialization_excluded_from_steady_state": True,
                    "adjacent_instrumented_uninstrumented_hash_equality": True,
                    "full_fast_slow_hash_equality": True,
                },
            },
            "static_kv_decode": {
                "formal_enabled": False,
                "decision": "disabled_pending_full_sample_parity_audit",
            },
        },
        "runtime": runtime_identity(),
        "source_hashes": selected_source_hashes(args),
        "checkpoint_hashes": checkpoint_hashes(args),
    }
    return payload


def config_id(payload: dict[str, Any]) -> str:
    return sha256_text(canonical_json(payload))[:16]


def output_path_for(
    args: argparse.Namespace,
    computed_config_id: str,
) -> Path:
    gpu_component = f"cuda{args.physical_gpu_id}"
    required_components = {
        PROTOCOL_VERSION,
        gpu_component,
        arm_id(args),
        computed_config_id,
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        components = set(output.parts)
        missing = sorted(required_components - components)
        if missing:
            raise ProtocolError(
                "Explicit --output must encode protocol/GPU/method/config id as path "
                f"components; missing={missing}, required={sorted(required_components)}"
            )
        return output
    stem = (
        f"{args.run_id}_rep{args.repetition:02d}_seed{args.seed}_"
        f"n{args.num_samples}.tsv"
    )
    return (
        args.result_root.expanduser().resolve()
        / PROTOCOL_VERSION
        / gpu_component
        / arm_id(args)
        / computed_config_id
        / stem
    )


def read_completed_rows(path: Path) -> tuple[set[int], int]:
    if not path.is_file():
        return set(), 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != RESULT_FIELDS:
            raise ProtocolError(
                f"Existing result header is not {PROTOCOL_VERSION}: {reader.fieldnames}"
            )
        indices: list[int] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                indices.append(int(row["dataset_index"]))
            except Exception as exc:
                raise ProtocolError(
                    f"Invalid dataset index in {path} line {line_number}"
                ) from exc
    if len(set(indices)) != len(indices):
        raise ProtocolError(f"Duplicate dataset indices in resumable result {path}")
    return set(indices), len(indices)


def resume_guard(
    metadata_path: Path,
    *,
    expected_config_id: str,
    expected_payload: dict[str, Any],
    expected_repetition: int,
    result_path: Path | None = None,
) -> dict[str, Any]:
    if not metadata_path.is_file():
        raise ProtocolError(
            f"Refusing resume without matching metadata: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "config_id": expected_config_id,
        "protocol_config": expected_payload,
        "repetition": int(expected_repetition),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ProtocolError(
            "Refusing resume because metadata does not exactly match the current "
            f"protocol: {canonical_json(mismatches)}"
        )
    recorded_result_sha256 = metadata.get("result_tsv_sha256")
    if recorded_result_sha256 is not None:
        if result_path is None or not result_path.is_file():
            raise ProtocolError(
                "Refusing finalized resume without its bound result TSV"
            )
        observed_result_sha256 = file_sha256(result_path)
        if (
            not isinstance(recorded_result_sha256, str)
            or observed_result_sha256 != recorded_result_sha256
        ):
            raise ProtocolError(
                "Refusing resume because the finalized result TSV hash changed: "
                f"observed={observed_result_sha256}, "
                f"expected={recorded_result_sha256}"
            )
    return metadata


def resume_rows_guard(
    path: Path,
    *,
    args: argparse.Namespace,
    expected_config_id: str,
    expected_rows: Sequence[dict[str, Any]],
) -> None:
    """Reject partially resumed TSVs whose rows are not from this exact run."""
    if not path.is_file():
        return
    expected_by_index = {
        int(row["dataset_index"]): (sample_order, row)
        for sample_order, row in enumerate(expected_rows)
    }
    expected_static = {
        "protocol_version": PROTOCOL_VERSION,
        "config_id": expected_config_id,
        "run_id": args.run_id,
        "method": args.method,
        "arm_id": arm_id(args),
        "generation_variant": generation_variant(args),
        "backend": (
            "official_llava"
            if args.method in {"vanilla", "ours"}
            else "huggingface_llava"
        ),
        "locked_sm_clock_mhz": (
            ""
            if args.locked_sm_clock_mhz is None
            else str(args.locked_sm_clock_mhz)
        ),
        "repetition": str(args.repetition),
        "projector_input_tokens": (
            "111" if args.method == "learnpruner" else "576"
        ),
        "projector_output_tokens": (
            "111" if args.method == "learnpruner" else "576"
        ),
        "requested_output_tokens": str(args.max_new_tokens),
        "actual_output_tokens": str(args.max_new_tokens),
        "forward_steps": str(args.max_new_tokens),
        "cuda_graph_prefill_pair_status": (
            "stable_runner_two_cache_hit_replays_no_fallback"
            if (
                args.method == "ours"
                and args.ours_fixed_length_greedy == "on"
                and args.ours_cuda_graph_prefill == "on"
            )
            else "disabled"
        ),
        "cuda_graph_decode_callback_steps": json_compact(
            list(range(1, args.max_new_tokens))
            if (
                args.method == "ours"
                and args.ours_fixed_length_greedy == "on"
                and args.ours_cuda_graph_decode == "on"
            )
            else []
        ),
        "cuda_graph_runner_cache_status": (
            "single_runner_reused"
            if (
                args.method == "ours"
                and args.ours_fixed_length_greedy == "on"
                and args.ours_cuda_graph_decode == "on"
            )
            else "disabled"
        ),
        "triton_rms_runtime_status": (
            "prefill_only_enabled_no_runtime_fallback"
            if args.method == "ours" and args.ours_triton_rms == "on"
            else "disabled"
        ),
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != RESULT_FIELDS:
            raise ProtocolError(f"Existing result header is not {PROTOCOL_VERSION}")
        for line_number, row in enumerate(reader, start=2):
            try:
                dataset_index = int(row["dataset_index"])
            except Exception as exc:
                raise ProtocolError(
                    f"Invalid dataset index in {path} line {line_number}"
                ) from exc
            if dataset_index not in expected_by_index:
                raise ProtocolError(
                    f"Foreign dataset index {dataset_index} in {path}:{line_number}"
                )
            sample_order, expected_sample = expected_by_index[dataset_index]
            expected = {
                **expected_static,
                "sample_order": str(sample_order),
                "dataset_index": str(dataset_index),
                "source": str(expected_sample["source"]),
                "image": str(expected_sample["image"]),
                "measurement_order": (
                    "instrumented_first"
                    if (args.repetition + sample_order) % 2 == 0
                    else "uninstrumented_first"
                ),
            }
            mismatches = {
                key: (row.get(key), value)
                for key, value in expected.items()
                if row.get(key) != value
            }
            if mismatches:
                raise ProtocolError(
                    f"Refusing mixed/stale resume row {path}:{line_number}: "
                    f"{canonical_json(mismatches)}"
                )
            prefill_graph_enabled = (
                args.method == "ours"
                and args.ours_fixed_length_greedy == "on"
                and args.ours_cuda_graph_prefill == "on"
            )
            graph_residency_enabled = (
                args.method == "ours"
                and args.ours_fixed_length_greedy == "on"
                and (
                    args.ours_cuda_graph_prefill == "on"
                    or args.ours_cuda_graph_decode == "on"
                )
            )
            callback_runner_text = row.get(
                "cuda_graph_prefill_callback_runner_id",
                "",
            )
            try:
                callback_runner_id = (
                    int(callback_runner_text)
                    if callback_runner_text
                    else None
                )
            except Exception as exc:
                raise ProtocolError(
                    "Invalid CUDA-graph prefill callback runner in "
                    f"{path}:{line_number}"
                ) from exc
            if (
                prefill_graph_enabled
                and (
                    callback_runner_id is None
                    or callback_runner_id < 1
                )
            ) or (
                not prefill_graph_enabled
                and callback_runner_id is not None
            ) or (
                graph_residency_enabled
                and row.get(
                    "cuda_graph_residency_warmup_tail_sha256"
                )
                != row.get("generated_tail_sha256")
            ) or (
                not graph_residency_enabled
                and row.get(
                    "cuda_graph_residency_warmup_tail_sha256"
                )
            ):
                raise ProtocolError(
                    "Invalid CUDA-graph prefill/residency provenance in "
                    f"{path}:{line_number}"
                )
            if not row.get("generated_token_ids") or not row.get(
                "generated_tail_sha256"
            ) or not row.get("uninstrumented_generated_tail_sha256"):
                raise ProtocolError(
                    f"Incomplete generated-token provenance in {path}:{line_number}"
                )
            try:
                generated_ids = [
                    int(value)
                    for value in json.loads(row["generated_token_ids"])
                ]
            except Exception as exc:
                raise ProtocolError(
                    f"Invalid generated-token provenance in {path}:{line_number}"
                ) from exc
            recomputed_hash = sha256_text(json_compact(generated_ids))
            if (
                len(generated_ids) != args.max_new_tokens
                or row["generated_tail_sha256"] != recomputed_hash
                or row["uninstrumented_generated_tail_sha256"]
                != recomputed_hash
            ):
                raise ProtocolError(
                    f"Generated-token hash mismatch in resumable row "
                    f"{path}:{line_number}"
                )


def _cuda_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def _elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(start.elapsed_time(end))


@dataclass(frozen=True)
class StageMeasurements:
    vision_cuda_ms: float
    projector_cuda_ms: float
    llm_prefill_cuda_ms: float
    multimodal_prefill_cuda_ms: float
    ttft_cuda_ms: float
    decode_cuda_ms: float
    tpot_cuda_ms: float
    e2e_cuda_ms: float
    ttft_wall_ms: float
    decode_wall_ms: float
    tpot_wall_ms: float
    e2e_wall_ms: float
    prefill_forward_cuda_ms: float
    decode_forward_cuda_ms: list[float]
    decode_step_cuda_ms: list[float]
    decode_step_wall_ms: list[float]
    forward_steps: int
    cuda_graph_prefill_callback_runner_id: int | None
    cuda_graph_decode_callback_steps: list[int]
    projector_input_tokens: int
    projector_output_tokens: int


class GenerationStageTimer:
    """Record direct generation-stage boundaries with lightweight CUDA events."""

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        require_cuda_graph_prefill: bool = False,
        require_cuda_graph_decode: bool = False,
    ):
        self.adapter = adapter
        self.model = adapter.model
        self.require_cuda_graph_prefill = bool(require_cuda_graph_prefill)
        self.require_cuda_graph_decode = bool(require_cuda_graph_decode)
        self.vision_module = self._vision_module()
        self.projector_module = adapter.projector
        self.step_start_module = getattr(
            adapter.layers[0], "input_layernorm", adapter.layers[0]
        )
        self.step_end_module = self._lm_head_module()
        self.handles: list[Any] = []
        self.step_starts: list[torch.cuda.Event] = []
        self.step_ends: list[torch.cuda.Event] = []
        self.step_wall_starts: list[float] = []
        self.graph_decode_starts: set[int] = set()
        self.graph_decode_ends: set[int] = set()
        self.active_graph_decode_step: int | None = None
        self.graph_prefill_started = False
        self.graph_prefill_ended = False
        self.active_graph_prefill_runner_id: int | None = None
        self.graph_prefill_runner_id: int | None = None
        self.vision_start: torch.cuda.Event | None = None
        self.vision_end: torch.cuda.Event | None = None
        self.projector_start: torch.cuda.Event | None = None
        self.projector_end: torch.cuda.Event | None = None
        self.projector_input_tokens: int | None = None
        self.projector_output_tokens: int | None = None

    def _vision_module(self) -> torch.nn.Module:
        if self.adapter.backend == "official_llava":
            module = self.adapter.model.get_vision_tower()
        else:
            module = self.adapter.model.model.vision_tower
        if isinstance(module, (list, tuple)):
            module = module[0]
        if not isinstance(module, torch.nn.Module):
            nested = getattr(module, "vision_tower", None)
            if isinstance(nested, torch.nn.Module):
                module = nested
        if not isinstance(module, torch.nn.Module):
            raise ProtocolError(f"Could not resolve vision module: {type(module)!r}")
        return module

    def _lm_head_module(self) -> torch.nn.Module:
        candidates = [
            getattr(self.adapter.model, "lm_head", None),
            getattr(
                getattr(
                    getattr(self.adapter.model, "model", None),
                    "language_model",
                    None,
                ),
                "lm_head",
                None,
            ),
            getattr(
                getattr(self.adapter.model, "language_model", None),
                "lm_head",
                None,
            ),
        ]
        for candidate in candidates:
            if isinstance(candidate, torch.nn.Module):
                return candidate
        raise ProtocolError("Could not resolve the language-model output head")

    def _record_once_pre(self, attribute: str):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...]) -> None:
            if getattr(self, attribute) is None:
                event = _cuda_event()
                event.record()
                setattr(self, attribute, event)

        return hook

    def _record_once_post(self, attribute: str):
        def hook(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            _output: Any,
        ) -> None:
            if getattr(self, attribute) is None:
                event = _cuda_event()
                event.record()
                setattr(self, attribute, event)

        return hook

    def _record_step_start(self) -> None:
        event = _cuda_event()
        event.record()
        self.step_starts.append(event)
        self.step_wall_starts.append(time.perf_counter())

    def _record_step_end(self) -> None:
        event = _cuda_event()
        event.record()
        self.step_ends.append(event)

    @staticmethod
    def _sequence_tokens(value: Any, boundary: str) -> int:
        if not torch.is_tensor(value) or value.ndim < 2:
            raise ProtocolError(
                f"Projector {boundary} is not a token tensor: "
                f"{type(value)!r}"
            )
        tokens = int(value.shape[-2])
        if tokens <= 0:
            raise ProtocolError(
                f"Projector {boundary} has invalid token count {tokens}"
            )
        return tokens

    def _projector_pre(
        self,
        _module: torch.nn.Module,
        args: tuple[Any, ...],
    ) -> None:
        if self.projector_start is None:
            if not args:
                raise ProtocolError("Projector pre-hook received no input")
            self.projector_input_tokens = self._sequence_tokens(
                args[0],
                "input",
            )
            self.projector_start = _cuda_event()
            self.projector_start.record()

    def _projector_post(
        self,
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        if self.projector_end is None:
            self.projector_output_tokens = self._sequence_tokens(
                output,
                "output",
            )
            self.projector_end = _cuda_event()
            self.projector_end.record()

    def _step_pre(self, _module: torch.nn.Module, _args: tuple[Any, ...]) -> None:
        self._record_step_start()

    def _step_post(
        self,
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        _output: Any,
    ) -> None:
        self._record_step_end()

    def _cuda_graph_decode_boundary(self, phase: str, step: int) -> None:
        """Merge graph-replay markers into the ordinary module-hook timeline."""
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ProtocolError(f"Invalid CUDA-graph decode step: {step!r}")
        if phase == "decode_start":
            if (
                self.active_graph_decode_step is not None
                or step in self.graph_decode_starts
                or step != len(self.step_starts)
            ):
                raise ProtocolError(
                    "Out-of-order/duplicate CUDA-graph decode start: "
                    f"step={step}, recorded_starts={len(self.step_starts)}, "
                    f"active={self.active_graph_decode_step}"
                )
            self.graph_decode_starts.add(step)
            self.active_graph_decode_step = step
            self._record_step_start()
            return
        if phase == "decode_end":
            if (
                step in self.graph_decode_ends
                or step not in self.graph_decode_starts
                or step != len(self.step_ends)
                or self.active_graph_decode_step != step
            ):
                raise ProtocolError(
                    "Out-of-order/duplicate CUDA-graph decode end: "
                    f"step={step}, recorded_ends={len(self.step_ends)}, "
                    f"active={self.active_graph_decode_step}"
                )
            self.graph_decode_ends.add(step)
            self.active_graph_decode_step = None
            self._record_step_end()
            return
        raise ProtocolError(f"Unknown CUDA-graph boundary phase: {phase!r}")

    def _cuda_graph_prefill_boundary(
        self,
        phase: str,
        runner_id: int,
    ) -> None:
        """Use replay markers for prefill because graph replay bypasses hooks."""
        if (
            isinstance(runner_id, bool)
            or not isinstance(runner_id, int)
            or runner_id < 1
        ):
            raise ProtocolError(
                f"Invalid CUDA-graph prefill runner id: {runner_id!r}"
            )
        if phase == "prefill_start":
            if (
                self.graph_prefill_started
                or self.graph_prefill_ended
                or self.active_graph_prefill_runner_id is not None
                or self.step_starts
                or self.step_ends
            ):
                raise ProtocolError(
                    "Out-of-order/duplicate CUDA-graph prefill start: "
                    f"runner_id={runner_id}, starts={len(self.step_starts)}, "
                    f"ends={len(self.step_ends)}, "
                    f"active={self.active_graph_prefill_runner_id}"
                )
            self.graph_prefill_started = True
            self.active_graph_prefill_runner_id = runner_id
            self.graph_prefill_runner_id = runner_id
            self._record_step_start()
            return
        if phase == "prefill_end":
            if (
                not self.graph_prefill_started
                or self.graph_prefill_ended
                or self.active_graph_prefill_runner_id != runner_id
                or self.graph_prefill_runner_id != runner_id
                or len(self.step_starts) != 1
                or self.step_ends
            ):
                raise ProtocolError(
                    "Out-of-order/mismatched CUDA-graph prefill end: "
                    f"runner_id={runner_id}, "
                    f"recorded_runner={self.graph_prefill_runner_id}, "
                    f"starts={len(self.step_starts)}, "
                    f"ends={len(self.step_ends)}, "
                    f"active={self.active_graph_prefill_runner_id}"
                )
            self.graph_prefill_ended = True
            self.active_graph_prefill_runner_id = None
            self._record_step_end()
            return
        raise ProtocolError(
            f"Unknown CUDA-graph prefill boundary phase: {phase!r}"
        )

    def _register(self) -> None:
        self.handles = [
            # Layer-0 input norm and lm_head are shared boundaries for upstream
            # GenerationMixin and Ours' direct fixed-length loop. The latter
            # bypasses repeated top-level model.forward calls but still invokes
            # layer 0 through _manual_decode and the same lm_head each step.
            self.step_start_module.register_forward_pre_hook(self._step_pre),
            self.step_end_module.register_forward_hook(self._step_post),
            self.vision_module.register_forward_pre_hook(
                self._record_once_pre("vision_start")
            ),
            self.vision_module.register_forward_hook(
                self._record_once_post("vision_end")
            ),
            self.projector_module.register_forward_pre_hook(
                self._projector_pre
            ),
            self.projector_module.register_forward_hook(
                self._projector_post
            ),
        ]
        graph_boundary_register = getattr(
            self.model,
            "register_cuda_graph_decode_boundary_callback",
            None,
        )
        if callable(graph_boundary_register):
            self.handles.append(
                graph_boundary_register(self._cuda_graph_decode_boundary)
            )
        elif self.require_cuda_graph_decode:
            self._remove()
            raise ProtocolError(
                "Formal CUDA-graph decode requires the replay-boundary callback API"
            )
        prefill_boundary_register = getattr(
            self.model,
            "register_cuda_graph_prefill_boundary_callback",
            None,
        )
        if self.require_cuda_graph_prefill:
            if not callable(prefill_boundary_register):
                self._remove()
                raise ProtocolError(
                    "Formal CUDA-graph prefill requires the replay-boundary "
                    "callback API"
                )
            self.handles.append(
                prefill_boundary_register(
                    self._cuda_graph_prefill_boundary
                )
            )

    def _remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def run(self, generate_call: Any, expected_steps: int) -> tuple[Any, StageMeasurements]:
        self._register()
        torch.cuda.synchronize(self.adapter.device)
        overall_start = _cuda_event()
        overall_end = _cuda_event()
        wall_start = time.perf_counter()
        overall_start.record()
        try:
            output = generate_call()
            overall_end.record()
            overall_end.synchronize()
            wall_end = time.perf_counter()
        finally:
            self._remove()

        if len(self.step_starts) != len(self.step_ends):
            raise ProtocolError(
                f"Unbalanced forward events: starts={len(self.step_starts)}, "
                f"ends={len(self.step_ends)}"
            )
        if self.graph_decode_starts != self.graph_decode_ends:
            raise ProtocolError(
                "Unbalanced CUDA-graph decode callbacks: "
                f"starts={sorted(self.graph_decode_starts)}, "
                f"ends={sorted(self.graph_decode_ends)}"
            )
        if self.active_graph_decode_step is not None:
            raise ProtocolError(
                "CUDA-graph decode callback remained active at generation end: "
                f"step={self.active_graph_decode_step}"
            )
        if self.active_graph_prefill_runner_id is not None:
            raise ProtocolError(
                "CUDA-graph prefill callback remained active at generation end: "
                f"runner_id={self.active_graph_prefill_runner_id}"
            )
        if self.require_cuda_graph_prefill:
            if (
                not self.graph_prefill_started
                or not self.graph_prefill_ended
                or self.graph_prefill_runner_id is None
            ):
                raise ProtocolError(
                    "CUDA-graph prefill replay coverage mismatch/fallback: "
                    f"started={self.graph_prefill_started}, "
                    f"ended={self.graph_prefill_ended}, "
                    f"runner_id={self.graph_prefill_runner_id}"
                )
        elif (
            self.graph_prefill_started
            or self.graph_prefill_ended
            or self.graph_prefill_runner_id is not None
        ):
            raise ProtocolError(
                "Unexpected CUDA-graph prefill callbacks while the path is disabled"
            )
        expected_graph_steps = (
            set(range(1, expected_steps))
            if self.require_cuda_graph_decode
            else set()
        )
        if (
            self.graph_decode_starts != expected_graph_steps
            or self.graph_decode_ends != expected_graph_steps
        ):
            raise ProtocolError(
                "CUDA-graph decode replay coverage mismatch/fallback: "
                f"required={self.require_cuda_graph_decode}, "
                f"expected={sorted(expected_graph_steps)}, "
                f"starts={sorted(self.graph_decode_starts)}, "
                f"ends={sorted(self.graph_decode_ends)}"
            )
        if len(self.step_starts) != expected_steps:
            raise ProtocolError(
                f"Expected {expected_steps} generation forwards/output tokens, "
                f"observed {len(self.step_starts)}"
            )
        required_events = {
            "vision_start": self.vision_start,
            "vision_end": self.vision_end,
            "projector_start": self.projector_start,
            "projector_end": self.projector_end,
        }
        missing = [name for name, event in required_events.items() if event is None]
        if missing:
            raise ProtocolError(f"Missing timed stage boundary event(s): {missing}")
        if (
            self.projector_input_tokens is None
            or self.projector_output_tokens is None
            or self.projector_input_tokens != self.projector_output_tokens
        ):
            raise ProtocolError(
                "Projector token tracing failed or changed sequence length: "
                f"input={self.projector_input_tokens}, "
                f"output={self.projector_output_tokens}"
            )
        if expected_steps < 2:
            raise ProtocolError("Protocol v2 requires at least two output tokens")

        vision_ms = _elapsed_ms(self.vision_start, self.vision_end)  # type: ignore[arg-type]
        projector_ms = _elapsed_ms(  # type: ignore[arg-type]
            self.projector_start, self.projector_end
        )
        llm_prefill_ms = _elapsed_ms(self.step_starts[0], self.step_ends[0])
        multimodal_prefill_ms = _elapsed_ms(overall_start, self.step_ends[0])
        ttft_ms = _elapsed_ms(overall_start, self.step_starts[1])
        decode_ms = _elapsed_ms(self.step_starts[1], overall_end)
        e2e_ms = _elapsed_ms(overall_start, overall_end)

        decode_step_cuda: list[float] = []
        decode_step_wall: list[float] = []
        for index in range(1, expected_steps):
            cuda_end = (
                self.step_starts[index + 1]
                if index + 1 < expected_steps
                else overall_end
            )
            wall_interval_end = (
                self.step_wall_starts[index + 1]
                if index + 1 < expected_steps
                else wall_end
            )
            decode_step_cuda.append(_elapsed_ms(self.step_starts[index], cuda_end))
            decode_step_wall.append(
                (wall_interval_end - self.step_wall_starts[index]) * 1000.0
            )
        decode_forward_cuda = [
            _elapsed_ms(self.step_starts[index], self.step_ends[index])
            for index in range(1, expected_steps)
        ]
        ttft_wall_ms = (self.step_wall_starts[1] - wall_start) * 1000.0
        e2e_wall_ms = (wall_end - wall_start) * 1000.0
        decode_wall_ms = (wall_end - self.step_wall_starts[1]) * 1000.0
        measurements = StageMeasurements(
            vision_cuda_ms=vision_ms,
            projector_cuda_ms=projector_ms,
            llm_prefill_cuda_ms=llm_prefill_ms,
            multimodal_prefill_cuda_ms=multimodal_prefill_ms,
            ttft_cuda_ms=ttft_ms,
            decode_cuda_ms=decode_ms,
            tpot_cuda_ms=sum(decode_step_cuda) / len(decode_step_cuda),
            e2e_cuda_ms=e2e_ms,
            ttft_wall_ms=ttft_wall_ms,
            decode_wall_ms=decode_wall_ms,
            tpot_wall_ms=sum(decode_step_wall) / len(decode_step_wall),
            e2e_wall_ms=e2e_wall_ms,
            prefill_forward_cuda_ms=_elapsed_ms(
                self.step_starts[0], self.step_ends[0]
            ),
            decode_forward_cuda_ms=decode_forward_cuda,
            decode_step_cuda_ms=decode_step_cuda,
            decode_step_wall_ms=decode_step_wall,
            forward_steps=len(self.step_starts),
            cuda_graph_prefill_callback_runner_id=(
                self.graph_prefill_runner_id
            ),
            cuda_graph_decode_callback_steps=sorted(
                self.graph_decode_starts
            ),
            projector_input_tokens=self.projector_input_tokens,
            projector_output_tokens=self.projector_output_tokens,
        )
        return output, measurements


def generation_kwargs(adapter: ModelAdapter, new_tokens: int) -> dict[str, Any]:
    pad_token_id = adapter.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = adapter.tokenizer.eos_token_id
    return {
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "max_new_tokens": int(new_tokens),
        "min_new_tokens": int(new_tokens),
        "pad_token_id": pad_token_id,
        "eos_token_id": adapter.tokenizer.eos_token_id,
    }


def generate_v2(
    adapter: ModelAdapter,
    prepared: PreparedInput,
    new_tokens: int,
) -> Any:
    return adapter.model.generate(
        **prepared.kwargs,
        **generation_kwargs(adapter, new_tokens),
    )


def timed_generate_uninstrumented(
    adapter: ModelAdapter,
    prepared: PreparedInput,
    new_tokens: int,
) -> tuple[Any, float, float]:
    """Authoritative outer timing with no module hook or boundary callback."""
    torch.cuda.synchronize(adapter.device)
    start = _cuda_event()
    end = _cuda_event()
    wall_start = time.perf_counter()
    start.record()
    output = generate_v2(adapter, prepared, new_tokens)
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return output, _elapsed_ms(start, end), wall_ms


def output_sequences(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    sequences = getattr(output, "sequences", None)
    if not torch.is_tensor(sequences):
        raise ProtocolError(f"Generation output has no tensor sequences: {type(output)!r}")
    return sequences


def actual_output_tokens(
    output: Any,
    prepared: PreparedInput,
    forward_steps: int,
) -> tuple[int, int, list[int], str]:
    sequences = output_sequences(output)
    sequence_length = int(sequences.shape[-1])
    input_tensor = prepared.kwargs.get("input_ids")
    if input_tensor is None:
        input_tensor = prepared.kwargs.get("inputs")
    if not torch.is_tensor(input_tensor):
        raise ProtocolError("Prepared generation input has no input-id tensor")
    plausible_lengths = {
        forward_steps,
        int(input_tensor.shape[-1]) + forward_steps,
    }
    if sequence_length not in plausible_lengths:
        raise ProtocolError(
            f"Generated sequence length {sequence_length} is incompatible with "
            f"{forward_steps} forward steps and input lengths {sorted(plausible_lengths)}"
        )
    tail = [
        int(token_id)
        for token_id in sequences[0, -forward_steps:].detach().cpu().tolist()
    ]
    tail_json = json_compact(tail)
    return forward_steps, sequence_length, tail, sha256_text(tail_json)


def untimed_graph_residency_warmup(
    args: argparse.Namespace,
    adapter: ModelAdapter,
    prepared: PreparedInput,
) -> str:
    """Make the current sample's graph signatures resident outside timing."""
    enabled = (
        args.method == "ours"
        and args.ours_fixed_length_greedy == "on"
        and (
            args.ours_cuda_graph_prefill == "on"
            or args.ours_cuda_graph_decode == "on"
        )
    )
    if not enabled:
        return ""
    adapter.reset_pruner_stats()
    with torch.inference_mode():
        output = generate_v2(
            adapter,
            prepared,
            args.max_new_tokens,
        )
    torch.cuda.synchronize(adapter.device)
    actual, _sequence_length, _token_ids, tail_hash = (
        actual_output_tokens(
            output,
            prepared,
            args.max_new_tokens,
        )
    )
    if actual != args.max_new_tokens:
        raise ProtocolError(
            "Untimed graph-residency warmup did not produce the fixed "
            f"output length: actual={actual}, expected={args.max_new_tokens}"
        )
    del output
    return tail_hash


def estimate_prefill_flops_v2(
    adapter: ModelAdapter,
    layer_tokens: Sequence[int],
    projector_tokens: int,
) -> float:
    """Analytical prefill FLOPs using the projector's observed token count."""
    if isinstance(projector_tokens, bool) or int(projector_tokens) <= 0:
        raise ProtocolError(
            f"Invalid projector token count for FLOP estimate: {projector_tokens}"
        )
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
        decoder_flops += 2.0 * tokens * (
            projection_weights + mlp_weights
        )
        decoder_flops += 4.0 * tokens * tokens * heads * head_dim

    # Generation requests only the final prefill logit.
    decoder_flops += 2.0 * hidden * int(text.vocab_size)

    vision = adapter.vision_config
    vision_hidden = int(vision.hidden_size)
    vision_intermediate = int(vision.intermediate_size)
    vision_layers = int(vision.num_hidden_layers)
    image_size = int(vision.image_size)
    patch_size = int(vision.patch_size)
    patch_tokens = (image_size // patch_size) ** 2
    vision_tokens = patch_tokens + 1
    vision_linear = 2.0 * vision_tokens * (
        4 * vision_hidden * vision_hidden
        + 2 * vision_hidden * vision_intermediate
    )
    vision_attention = (
        4.0 * vision_tokens * vision_tokens * vision_hidden
    )
    patch_embedding = (
        2.0
        * patch_tokens
        * (3 * patch_size * patch_size)
        * vision_hidden
    )
    vision_flops = (
        vision_layers * (vision_linear + vision_attention)
        + patch_embedding
    )

    projector_weights = sum(
        int(module.in_features) * int(module.out_features)
        for module in adapter.projector.modules()
        if isinstance(module, torch.nn.Linear)
    )
    projector_flops = (
        2.0 * int(projector_tokens) * projector_weights
    )
    return decoder_flops + vision_flops + projector_flops


def infer_layer_tokens(
    args: argparse.Namespace,
    adapter: ModelAdapter,
    prepared: PreparedInput,
    pruner_stats: dict[str, Any],
) -> list[int]:
    num_layers = len(adapter.layers)
    if args.method in {"vanilla", "vanilla_hf"}:
        return [int(prepared.input_sequence_tokens)] * num_layers
    if args.method == "ours":
        if not pruner_stats:
            raise ProtocolError("Ours did not report pruning statistics")
        placeholder_trace = [int(prepared.input_sequence_tokens)] * num_layers
        return canonical_layer_tokens(
            args,
            adapter,
            placeholder_trace,
            pruner_stats,
            prepared.prompt_text_tokens,
        )
    if not pruner_stats:
        raise ProtocolError("LearnPruner did not report pruning statistics")
    stage1_sequence = (
        prepared.prompt_text_tokens + int(pruner_stats["stage1_visual_tokens"])
    )
    final_sequence = int(pruner_stats["final_sequence_tokens"])
    stage2_layer_idx = int(pruner_stats["stage2_layer_idx"])
    full_layers = stage2_layer_idx + 1
    if not (1 <= full_layers <= num_layers):
        raise ProtocolError(
            f"Invalid LearnPruner layer boundary {stage2_layer_idx} for {num_layers} layers"
        )
    return [stage1_sequence] * full_layers + [final_sequence] * (
        num_layers - full_layers
    )


def untimed_trace_validation(
    args: argparse.Namespace,
    adapter: ModelAdapter,
    row: dict[str, Any],
) -> dict[str, Any]:
    prepared = adapter.prepare(row)
    tracer = FirstLayerTokenTracer(adapter.layers)
    adapter.reset_pruner_stats()
    old_prefill_graph = os.environ.get(
        "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL"
    )
    force_eager_trace = (
        args.method == "ours"
        and args.ours_cuda_graph_prefill == "on"
    )
    if force_eager_trace:
        # Layer hooks cannot observe kernels replayed from a captured graph.
        # This validation is explicitly untimed and restores the formal switch.
        os.environ["LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL"] = "0"
    try:
        with torch.inference_mode():
            output = adapter.generate(prepared, 1)
        torch.cuda.synchronize(adapter.device)
        traced = tracer.result()
        stats = adapter.latest_pruner_stats()
        canonical = canonical_layer_tokens(
            args,
            adapter,
            traced,
            stats,
            prepared.prompt_text_tokens,
        )
        inferred = infer_layer_tokens(args, adapter, prepared, stats)
        if canonical != inferred:
            raise ProtocolError(
                "Untimed token trace disagrees with v2 layer-token inference: "
                f"traced={canonical}, inferred={inferred}"
            )
        if args.method == "ours":
            # The middle scorer intentionally reuses one decoder layer's input
            # norm before that layer's real compact-sequence invocation. The
            # first-call tracer is ambiguous only at that one documented index.
            scoring_layer = (
                int(stats["mid_scoring_layer_idx"])
                if "mid_scoring_layer_idx" in stats
                else None
            )
            mismatched_layers = [
                index
                for index, (observed, expected) in enumerate(zip(traced, inferred))
                if index != scoring_layer and int(observed) != int(expected)
            ]
            if mismatched_layers:
                raise ProtocolError(
                    "Ours untimed trace disagrees outside its scorer layer: "
                    f"layers={mismatched_layers}, traced={traced}, inferred={inferred}"
                )
        return {
            "dataset_index": int(row["dataset_index"]),
            "traced_layer_tokens": traced,
            "canonical_layer_tokens": canonical,
            "pruner_stats": stats,
            "cuda_graph_prefill_forced_off_for_trace": force_eager_trace,
        }
    finally:
        if old_prefill_graph is None:
            os.environ.pop(
                "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL",
                None,
            )
        else:
            os.environ[
                "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL"
            ] = old_prefill_graph
        tracer.close()
        del prepared
        if "output" in locals():
            del output


def untimed_generation_path_validation(
    args: argparse.Namespace,
    adapter: ModelAdapter,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Prove that Ours selected the requested fast/slow control flow."""
    if args.method != "ours":
        raise ProtocolError("Generation-path validation is specific to Ours")
    prepared = adapter.prepare(row)
    top_level_forwards = 0
    graph_callback_events: list[tuple[str, int]] = []
    prefill_graph_callback_events: list[tuple[str, int]] = []

    original_forward = adapter.model.forward
    missing_instance_forward = object()
    previous_instance_forward = adapter.model.__dict__.get(
        "forward",
        missing_instance_forward,
    )

    @functools.wraps(original_forward)
    def count_forward(*forward_args: Any, **forward_kwargs: Any) -> Any:
        nonlocal top_level_forwards
        top_level_forwards += 1
        return original_forward(*forward_args, **forward_kwargs)

    # The admitted fixed loop deliberately calls self.forward directly so
    # graph-backed tensors never pass through arbitrary Module post-hooks.
    # Count that exact method without changing Module hook semantics, and
    # restore the instance attribute losslessly in the finally block.
    setattr(adapter.model, "forward", count_forward)
    graph_handle: Any | None = None
    prefill_graph_handle: Any | None = None
    try:
        graph_boundary_register = getattr(
            adapter.model,
            "register_cuda_graph_decode_boundary_callback",
            None,
        )
        if callable(graph_boundary_register):
            graph_handle = graph_boundary_register(
                lambda phase, step: graph_callback_events.append(
                    (str(phase), int(step))
                )
            )
        elif args.ours_cuda_graph_decode == "on":
            raise ProtocolError(
                "Formal CUDA-graph decode requires the replay-boundary callback API"
            )
        prefill_graph_boundary_register = getattr(
            adapter.model,
            "register_cuda_graph_prefill_boundary_callback",
            None,
        )
        if args.ours_cuda_graph_prefill == "on":
            if not callable(prefill_graph_boundary_register):
                raise ProtocolError(
                    "Formal CUDA-graph prefill requires the replay-boundary "
                    "callback API"
                )
            prefill_graph_handle = prefill_graph_boundary_register(
                lambda phase, runner_id: (
                    prefill_graph_callback_events.append(
                        (str(phase), int(runner_id))
                    )
                )
            )
        adapter.reset_pruner_stats()
        with torch.inference_mode():
            output = generate_v2(adapter, prepared, args.max_new_tokens)
        torch.cuda.synchronize(adapter.device)
    finally:
        if graph_handle is not None:
            graph_handle.remove()
        if prefill_graph_handle is not None:
            prefill_graph_handle.remove()
        if previous_instance_forward is missing_instance_forward:
            delattr(adapter.model, "forward")
        else:
            setattr(
                adapter.model,
                "forward",
                previous_instance_forward,
            )
    expected_forwards = (
        1
        if args.ours_fixed_length_greedy == "on"
        else int(args.max_new_tokens)
    )
    if top_level_forwards != expected_forwards:
        raise ProtocolError(
            "Ours did not execute the requested generation path: "
            f"fixed_length={args.ours_fixed_length_greedy}, "
            f"top_level_forwards={top_level_forwards}, "
            f"expected={expected_forwards}"
        )
    expected_graph_events = (
        [
            event
            for step in range(1, int(args.max_new_tokens))
            for event in (("decode_start", step), ("decode_end", step))
        ]
        if args.ours_cuda_graph_decode == "on"
        else []
    )
    if graph_callback_events != expected_graph_events:
        raise ProtocolError(
            "Ours CUDA-graph generation path fell back or emitted an invalid "
            "replay timeline: "
            f"switch={args.ours_cuda_graph_decode}, "
            f"observed={graph_callback_events}, "
            f"expected={expected_graph_events}"
        )
    expected_prefill_graph_phases = (
        ["prefill_start", "prefill_end"]
        if args.ours_cuda_graph_prefill == "on"
        else []
    )
    observed_prefill_graph_phases = [
        phase for phase, _runner_id in prefill_graph_callback_events
    ]
    prefill_graph_runner_ids = {
        runner_id
        for _phase, runner_id in prefill_graph_callback_events
    }
    if (
        observed_prefill_graph_phases != expected_prefill_graph_phases
        or (
            args.ours_cuda_graph_prefill == "on"
            and (
                len(prefill_graph_runner_ids) != 1
                or next(iter(prefill_graph_runner_ids)) < 1
            )
        )
        or (
            args.ours_cuda_graph_prefill == "off"
            and prefill_graph_runner_ids
        )
    ):
        raise ProtocolError(
            "Ours CUDA-graph prefill path fell back or emitted an invalid "
            "replay timeline: "
            f"switch={args.ours_cuda_graph_prefill}, "
            f"observed={prefill_graph_callback_events}, "
            f"expected_phases={expected_prefill_graph_phases}"
        )
    prefill_snapshot = validated_cuda_graph_prefill_snapshot(
        args,
        adapter,
    )
    if prefill_snapshot is not None and (
        prefill_snapshot["last_runner_id"]
        != next(iter(prefill_graph_runner_ids))
    ):
        raise ProtocolError(
            "Ours CUDA-graph prefill callback runner disagrees with diagnostics"
        )
    actual_tokens, sequence_length, _tokens, tail_hash = actual_output_tokens(
        output,
        prepared,
        args.max_new_tokens,
    )
    del output, prepared
    return {
        "dataset_index": int(row["dataset_index"]),
        "fixed_length_greedy": args.ours_fixed_length_greedy,
        "observed_top_level_forwards": top_level_forwards,
        "expected_top_level_forwards": expected_forwards,
        "forward_count_instrumentation": (
            "temporary_model_forward_method_wrapper_restored"
        ),
        "cuda_graph_decode": args.ours_cuda_graph_decode,
        "observed_cuda_graph_callback_events": [
            [phase, step] for phase, step in graph_callback_events
        ],
        "expected_cuda_graph_callback_events": [
            [phase, step] for phase, step in expected_graph_events
        ],
        "cuda_graph_fallback_or_cache_miss": False,
        "cuda_graph_prefill": args.ours_cuda_graph_prefill,
        "observed_cuda_graph_prefill_callback_events": [
            [phase, runner_id]
            for phase, runner_id in prefill_graph_callback_events
        ],
        "expected_cuda_graph_prefill_callback_phases": (
            expected_prefill_graph_phases
        ),
        "cuda_graph_prefill_callback_runner_id": (
            next(iter(prefill_graph_runner_ids))
            if prefill_graph_runner_ids
            else None
        ),
        "cuda_graph_prefill_fallback_or_cache_miss": False,
        "actual_output_tokens": actual_tokens,
        "output_sequence_length": sequence_length,
        "generated_tail_sha256": tail_hash,
    }


def validated_cuda_graph_runner_identity(
    args: argparse.Namespace,
    adapter: ModelAdapter,
) -> int | None:
    """Return the sole healthy graph runner's identity for pair-level reuse checks."""
    enabled = (
        args.method == "ours"
        and args.ours_fixed_length_greedy == "on"
        and args.ours_cuda_graph_decode == "on"
    )
    if not enabled:
        return None
    cache = getattr(
        adapter.model,
        "_fixed_greedy_cuda_graph_runners",
        None,
    )
    if cache is None or not hasattr(cache, "values"):
        raise ProtocolError(
            "Formal CUDA-graph decode did not expose its runner cache"
        )
    runners = list(cache.values())
    if len(runners) != 1:
        raise ProtocolError(
            "Formal CUDA-graph decode requires one warmed runner: "
            f"observed_cache_entries={len(runners)}"
        )
    runner = runners[0]
    if isinstance(runner, BaseException) or not callable(
        getattr(runner, "run", None)
    ):
        raise ProtocolError(
            "CUDA-graph runner cache contains a failed/non-runnable entry: "
            f"{type(runner)!r}"
        )
    return id(runner)


def validated_cuda_graph_prefill_snapshot(
    args: argparse.Namespace,
    adapter: ModelAdapter,
) -> dict[str, Any] | None:
    """Return strict JSON-friendly evidence for the one warmed prefill runner."""
    enabled = (
        args.method == "ours"
        and args.ours_fixed_length_greedy == "on"
        and args.ours_cuda_graph_prefill == "on"
    )
    if not enabled:
        return None
    getter = getattr(
        adapter.model,
        "get_cuda_graph_prefill_diagnostics",
        None,
    )
    if not callable(getter):
        raise ProtocolError(
            "Formal CUDA-graph prefill did not expose diagnostics"
        )
    diagnostics = getter()
    if not isinstance(diagnostics, dict):
        raise ProtocolError(
            "CUDA-graph prefill diagnostics are not a mapping"
        )
    integer_fields = (
        "cache_capacity",
        "cache_entries",
        "cached_failures",
        "cache_hits",
        "cache_misses",
        "captures",
        "capture_failures",
        "replay_successes",
        "replay_failures",
        "eager_fallbacks",
        "hook_fallbacks",
        "signature_mismatches",
        "evictions",
    )
    normalized: dict[str, Any] = {"enabled": diagnostics.get("enabled")}
    for field in integer_fields:
        value = diagnostics.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProtocolError(
                "Invalid CUDA-graph prefill diagnostic counter: "
                f"{field}={value!r}"
            )
        normalized[field] = int(value)
    for field in (
        "total_capture_wall_ms",
        "total_initialization_wall_ms",
    ):
        value = diagnostics.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProtocolError(
                "Invalid CUDA-graph prefill timing diagnostic: "
                f"{field}={value!r}"
            )
        normalized[field] = _finite_float(float(value), field)
    normalized["last_status"] = diagnostics.get("last_status")
    normalized["last_runner_id"] = diagnostics.get("last_runner_id")
    normalized["last_error"] = diagnostics.get("last_error")

    runners = diagnostics.get("runners")
    normalized_runners: list[dict[str, Any]] = []
    if not isinstance(runners, list):
        raise ProtocolError(
            "CUDA-graph prefill diagnostics runners are not a list"
        )
    for runner in runners:
        if not isinstance(runner, dict):
            raise ProtocolError(
                "CUDA-graph prefill runner diagnostic is not a mapping"
            )
        runner_id = runner.get("runner_id")
        segment_shapes = runner.get("segment_shapes")
        replay_count = runner.get("replay_count")
        if (
            isinstance(runner_id, bool)
            or not isinstance(runner_id, int)
            or runner_id < 1
            or not isinstance(segment_shapes, list)
            or len(segment_shapes) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in segment_shapes
            )
            or isinstance(replay_count, bool)
            or not isinstance(replay_count, int)
            or replay_count < 1
        ):
            raise ProtocolError(
                "Invalid CUDA-graph prefill runner diagnostic: "
                f"{canonical_json(runner)}"
            )
        normalized_runners.append(
            {
                "runner_id": int(runner_id),
                "segment_shapes": [int(value) for value in segment_shapes],
                "capture_wall_ms": _finite_float(
                    float(runner.get("capture_wall_ms", -1.0)),
                    "prefill_runner_capture_wall_ms",
                ),
                "initialization_wall_ms": _finite_float(
                    float(runner.get("initialization_wall_ms", -1.0)),
                    "prefill_runner_initialization_wall_ms",
                ),
                "replay_count": int(replay_count),
            }
        )
    normalized["runners"] = normalized_runners

    fatal_counters = {
        field: normalized[field]
        for field in (
            "cached_failures",
            "capture_failures",
            "replay_failures",
            "eager_fallbacks",
            "hook_fallbacks",
            "signature_mismatches",
        )
        if normalized[field] != 0
    }
    if (
        normalized["enabled"] is not True
        or normalized["cache_capacity"] < 1
        or not (
            1
            <= normalized["cache_entries"]
            <= normalized["cache_capacity"]
        )
        or normalized["cache_misses"] != normalized["captures"]
        or normalized["captures"] < 1
        or len(normalized_runners) != normalized["cache_entries"]
        or (
            normalized["evictions"] == 0
            and normalized["replay_successes"]
            != sum(
                runner["replay_count"] for runner in normalized_runners
            )
        )
        or normalized["replay_successes"]
        < sum(runner["replay_count"] for runner in normalized_runners)
        or normalized["last_status"]
        not in {"captured_then_replayed", "cache_hit_replayed"}
        or normalized["last_runner_id"]
        not in {runner["runner_id"] for runner in normalized_runners}
        or normalized["last_error"] is not None
        or normalized["total_capture_wall_ms"] <= 0.0
        or normalized["total_initialization_wall_ms"] <= 0.0
        or normalized_runners[0]["capture_wall_ms"] <= 0.0
        or normalized_runners[0]["initialization_wall_ms"] <= 0.0
        or fatal_counters
    ):
        raise ProtocolError(
            "CUDA-graph prefill has no healthy warmed steady-state runner: "
            f"{canonical_json(normalized)}"
        )
    return normalized


def validate_cuda_graph_prefill_pair(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    callback_runner_id: int | None,
) -> str:
    """Require exactly two stable cache-hit replays in an adjacent pair."""
    if before is None and after is None:
        if callback_runner_id is not None:
            raise ProtocolError(
                "CUDA-graph prefill callback appeared while the path is disabled"
            )
        return "disabled"
    if before is None or after is None:
        raise ProtocolError(
            "CUDA-graph prefill diagnostics changed enabled state inside a pair"
        )
    before_runners = {
        int(runner["runner_id"]): runner for runner in before["runners"]
    }
    after_runners = {
        int(runner["runner_id"]): runner for runner in after["runners"]
    }
    stable_fields = (
        "enabled",
        "cache_capacity",
        "cache_entries",
        "cached_failures",
        "cache_misses",
        "captures",
        "capture_failures",
        "replay_failures",
        "eager_fallbacks",
        "hook_fallbacks",
        "signature_mismatches",
        "total_capture_wall_ms",
        "total_initialization_wall_ms",
    )
    changed = {
        field: (before.get(field), after.get(field))
        for field in stable_fields
        if before.get(field) != after.get(field)
    }
    if (
        isinstance(callback_runner_id, bool)
        or not isinstance(callback_runner_id, int)
        or callback_runner_id not in before_runners
        or callback_runner_id not in after_runners
    ):
        raise ProtocolError(
            "CUDA-graph prefill callback did not identify a resident runner: "
            f"callback_runner_id={callback_runner_id}, "
            f"before_ids={sorted(before_runners)}, "
            f"after_ids={sorted(after_runners)}"
        )
    expected_runner_id = callback_runner_id
    before_runner = before_runners[expected_runner_id]
    after_runner = after_runners[expected_runner_id]
    changed_runner_cache = (
        {
            runner_id: runner
            for runner_id, runner in before_runners.items()
            if runner_id != expected_runner_id
        }
        != {
            runner_id: runner
            for runner_id, runner in after_runners.items()
            if runner_id != expected_runner_id
        }
    )
    stable_selected_runner = {
        field: before_runner.get(field)
        for field in (
            "runner_id",
            "segment_shapes",
            "capture_wall_ms",
            "initialization_wall_ms",
        )
    } == {
        field: after_runner.get(field)
        for field in (
            "runner_id",
            "segment_shapes",
            "capture_wall_ms",
            "initialization_wall_ms",
        )
    }
    if (
        changed
        or changed_runner_cache
        or not stable_selected_runner
        or before["evictions"] != after["evictions"]
        or int(after["cache_hits"]) != int(before["cache_hits"]) + 2
        or int(after["replay_successes"])
        != int(before["replay_successes"]) + 2
        or int(after_runner["replay_count"])
        != int(before_runner["replay_count"]) + 2
        or after["last_status"] != "cache_hit_replayed"
        or after["last_runner_id"] != expected_runner_id
        or after["last_error"] is not None
    ):
        raise ProtocolError(
            "CUDA-graph prefill did not perform exactly two stable warmed "
            "cache-hit replays in the adjacent pair: "
            f"callback_runner_id={callback_runner_id}, "
            f"changed={canonical_json(changed)}, "
            f"changed_runner_cache={changed_runner_cache}, "
            f"stable_selected_runner={stable_selected_runner}, "
            f"before={canonical_json(before)}, "
            f"after={canonical_json(after)}"
        )
    return "stable_runner_two_cache_hit_replays_no_fallback"


def _ours_modeling_module(adapter: ModelAdapter) -> Any:
    module_name = type(adapter.model).__module__
    module = sys.modules.get(module_name)
    if module is None:
        raise ProtocolError(
            f"Could not resolve loaded Ours modeling module {module_name!r}"
        )
    return module


def validated_triton_rms_runtime_status(
    args: argparse.Namespace,
    adapter: ModelAdapter,
) -> str:
    enabled = args.method == "ours" and args.ours_triton_rms == "on"
    if not enabled:
        return "disabled"
    module = _ours_modeling_module(adapter)
    expected_git_version = getattr(
        module,
        "_TRITON_RMS_EXACT_TORCH_GIT_VERSION",
        None,
    )
    checks = {
        "runtime_disabled": getattr(
            module, "_TRITON_RMS_RUNTIME_DISABLED", None
        ),
        "kernel_available": getattr(module, "_triton_rms_kernel", None)
        is not None,
        "expected_torch_git_version": expected_git_version,
        "actual_torch_git_version": getattr(
            torch.version, "git_version", None
        ),
    }
    if (
        checks["runtime_disabled"] is not False
        or checks["kernel_available"] is not True
        or not isinstance(expected_git_version, str)
        or checks["actual_torch_git_version"] != expected_git_version
    ):
        raise ProtocolError(
            "Prefill-only Triton RMS is unavailable or entered runtime "
            f"fallback: {canonical_json(checks)}"
        )
    return "prefill_only_enabled_no_runtime_fallback"


def untimed_triton_rms_path_validation(
    args: argparse.Namespace,
    adapter: ModelAdapter,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Exercise and prove the narrow prefill-only RMS domain outside timing."""
    if args.method != "ours" or args.ours_triton_rms != "on":
        raise ProtocolError(
            "Triton-RMS path validation requires the enabled Ours arm"
        )
    module = _ours_modeling_module(adapter)
    original_apply = getattr(module, "_apply_rms_norm", None)
    original_launch = getattr(module, "_launch_triton_rms_norm", None)
    eligibility = getattr(module, "_triton_rms_norm_is_eligible", None)
    if not all(callable(value) for value in (original_apply, original_launch, eligibility)):
        raise ProtocolError(
            "Loaded Ours module lacks the audited Triton-RMS validation API"
        )

    observations: list[tuple[int, bool]] = []
    launch_sequence_lengths: list[int] = []

    def observed_apply(
        norm: torch.nn.Module,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = (
            int(hidden_states.shape[1])
            if torch.is_tensor(hidden_states) and hidden_states.ndim == 3
            else -1
        )
        observations.append(
            (sequence_length, bool(eligibility(norm, hidden_states)))
        )
        return original_apply(norm, hidden_states)

    def observed_launch(
        norm: torch.nn.Module,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        launch_sequence_lengths.append(int(hidden_states.shape[1]))
        original_launch(norm, hidden_states, output)

    old_graph = os.environ.get("LEARNABLE_PRUNE_CUDA_GRAPH_DECODE")
    old_prefill_graph = os.environ.get(
        "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL"
    )
    old_rms = os.environ.get("LEARNABLE_PRUNE_TRITON_RMS")
    prepared = adapter.prepare(row)
    optimized_output: Any | None = None
    reference_output: Any | None = None
    try:
        # Force eager decode so the validation sees both multi-row prefill and
        # singleton decode eligibility. The warmed formal graph runner remains
        # cached and is restored immediately after this untimed probe.
        os.environ["LEARNABLE_PRUNE_CUDA_GRAPH_DECODE"] = "0"
        os.environ["LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL"] = "0"
        os.environ["LEARNABLE_PRUNE_TRITON_RMS"] = "1"
        try:
            setattr(module, "_apply_rms_norm", observed_apply)
            setattr(module, "_launch_triton_rms_norm", observed_launch)
            adapter.reset_pruner_stats()
            with torch.inference_mode():
                optimized_output = generate_v2(adapter, prepared, 2)
            torch.cuda.synchronize(adapter.device)
        finally:
            setattr(module, "_apply_rms_norm", original_apply)
            setattr(module, "_launch_triton_rms_norm", original_launch)

        os.environ["LEARNABLE_PRUNE_TRITON_RMS"] = "0"
        adapter.reset_pruner_stats()
        with torch.inference_mode():
            reference_output = generate_v2(adapter, prepared, 2)
        torch.cuda.synchronize(adapter.device)
    finally:
        if old_graph is None:
            os.environ.pop("LEARNABLE_PRUNE_CUDA_GRAPH_DECODE", None)
        else:
            os.environ["LEARNABLE_PRUNE_CUDA_GRAPH_DECODE"] = old_graph
        if old_prefill_graph is None:
            os.environ.pop(
                "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL",
                None,
            )
        else:
            os.environ[
                "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL"
            ] = old_prefill_graph
        if old_rms is None:
            os.environ.pop("LEARNABLE_PRUNE_TRITON_RMS", None)
        else:
            os.environ["LEARNABLE_PRUNE_TRITON_RMS"] = old_rms

    optimized_info = actual_output_tokens(
        optimized_output,
        prepared,
        2,
    )
    reference_info = actual_output_tokens(
        reference_output,
        prepared,
        2,
    )
    if optimized_info != reference_info:
        raise ProtocolError(
            "Untimed prefill-only Triton-RMS probe changed generated tokens: "
            f"optimized_hash={optimized_info[-1]}, "
            f"reference_hash={reference_info[-1]}"
        )
    multirow_eligible = [
        sequence_length
        for sequence_length, eligible_value in observations
        if sequence_length > 1 and eligible_value
    ]
    singleton_observations = [
        eligible_value
        for sequence_length, eligible_value in observations
        if sequence_length == 1
    ]
    if (
        not multirow_eligible
        or not launch_sequence_lengths
        or not singleton_observations
        or any(singleton_observations)
        or any(sequence_length <= 1 for sequence_length in launch_sequence_lengths)
    ):
        raise ProtocolError(
            "Triton-RMS eligibility/launch domain was not strictly prefill-only: "
            f"multirow_eligible={len(multirow_eligible)}, "
            f"singleton_observations={singleton_observations}, "
            f"launch_sequence_lengths={launch_sequence_lengths}"
        )
    runtime_status = validated_triton_rms_runtime_status(args, adapter)
    optimized_hash = optimized_info[-1]
    del prepared, optimized_output, reference_output
    return {
        "dataset_index": int(row["dataset_index"]),
        "probe_output_tokens": 2,
        "graph_forced_off_during_probe": True,
        "decode_graph_forced_off_during_probe": True,
        "prefill_graph_forced_off_during_probe": True,
        "optimized_reference_generated_tail_equal": True,
        "generated_tail_sha256": optimized_hash,
        "eligibility_observation_count": len(observations),
        "multirow_eligible_count": len(multirow_eligible),
        "singleton_observation_count": len(singleton_observations),
        "singleton_eligible_count": sum(singleton_observations),
        "launch_count": len(launch_sequence_lengths),
        "launch_sequence_lengths": launch_sequence_lengths,
        "singleton_launch_count": sum(
            sequence_length == 1
            for sequence_length in launch_sequence_lengths
        ),
        "runtime_status": runtime_status,
    }


def _finite_float(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ProtocolError(f"Invalid nonnegative {name}: {value}")
    return value


def snapshot_value(value: float | None) -> str:
    return "" if value is None else f"{value:.9f}"


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def loaded_learnpruner_config(
    args: argparse.Namespace,
    adapter: ModelAdapter,
) -> dict[str, Any] | None:
    if args.method != "learnpruner":
        return None
    inner_model = getattr(adapter.model, "model", None)
    config = getattr(inner_model, "learnpruner_config", None)
    if not isinstance(config, dict):
        raise ProtocolError("LearnPruner checkpoint did not expose its loaded config")
    if (
        args.learnpruner_checkpoint.expanduser().resolve()
        == FORMAL_LEARNPRUNER_CHECKPOINT.expanduser().resolve()
    ):
        expected = {
            "implementation_mode": "paper_aligned",
            "stage1_tokens": 111,
            "stage2_tokens": 37,
            "prune_layer": 12,
            "prune_layer_numbering": "one_based",
        }
        mismatches = {
            key: (config.get(key), value)
            for key, value in expected.items()
            if config.get(key) != value
        }
        if mismatches:
            raise ProtocolError(
                "Formal LearnPruner checkpoint is not paper-aligned 111/37 "
                f"latency configuration: {canonical_json(mismatches)}"
            )
    return dict(config)


def write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    adapter: ModelAdapter,
    protocol_config: dict[str, Any],
    computed_config_id: str,
    gpu_before_load: GpuSnapshot,
    trace_validation: dict[str, Any] | None,
    generation_path_validation: dict[str, Any] | None,
    triton_rms_path_validation: dict[str, Any] | None,
    prefill_graph_warmup_diagnostics: dict[str, Any] | None,
    output: Path,
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(adapter.device)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "config_id": computed_config_id,
        "run_id": args.run_id,
        "method": args.method,
        "arm_id": arm_id(args),
        "generation_variant": generation_variant(args),
        "backend": adapter.backend,
        "clock_mode": args.clock_mode,
        "locked_sm_clock_mhz": args.locked_sm_clock_mhz,
        "repetition": int(args.repetition),
        "output": str(output),
        "result_fields": RESULT_FIELDS,
        "protocol_config": protocol_config,
        "timing_boundaries": {
            "preprocessing_in_timing": False,
            "vision_encoder_in_timing": True,
            "authoritative_e2e": (
                "a complete adjacent fixed-length generate with only outer CUDA "
                "events; no module hooks or graph-boundary callback is registered"
            ),
            "cuda_graph_capture": (
                "graph construction/capture and initialization occur only during "
                "untimed warmup/residency calls and are excluded from steady-state "
                "latency and transient peak-memory timing; persistent graph "
                "allocations remain in baseline allocated memory"
            ),
            "per_sample_graph_residency_warmup": (
                "immediately before each adjacent measured pair, one untimed "
                "no-hook fixed-length generation makes that sample's prefill and "
                "decode graph signatures resident; it is synchronized and its "
                "generated-tail hash must equal both measured calls"
            ),
            "measurement_order": (
                "instrumented-first and uninstrumented-first alternate by "
                "(repetition + sample_order) parity"
            ),
            "observer_overhead": (
                "instrumented E2E divided by adjacent uninstrumented E2E; "
                "generated-tail hashes must match"
            ),
            "multimodal_prefill": (
                "overall CUDA start through end of first model forward; includes "
                "vision, projector, pruning, LLM prefill, and LM head"
            ),
            "llm_prefill": (
                "for ordinary eager paths: first decoder-layer input "
                "normalization through first lm_head completion; for formal "
                "prefill graph replay: prefill_start immediately after "
                "vision/projector and predictor/SCOPE through prefill_end after "
                "late graph, final RMS/lm_head, and heterogeneous cache assembly"
            ),
            "ttft": (
                "overall CUDA start through the second invocation of decoder "
                "layer 0 in the same generate call; includes first-token selection"
            ),
            "decode": (
                "second invocation of decoder layer 0 through generation end in "
                "the same generate call"
            ),
            "tpot": (
                "direct intervals from each decode layer-0 start to the next "
                "layer-0 start, with the final interval ending at generation end; "
                "a CUDA-graph replay uses equivalent same-stream callbacks "
                "immediately outside each replay; this is an instrumented "
                "diagnostic rather than the authoritative outer E2E"
            ),
        },
        "analytical_metric_definitions": {
            "estimated_prefill_flops": (
                "vision FLOPs use the full CLIP patch grid; projector FLOPs "
                "use the projector input/output token count traced in the same "
                "instrumented generation (576 for Vanilla/Ours, 111 for formal "
                "LearnPruner); decoder FLOPs use per-layer traced/inferred "
                "post-pruning sequence lengths"
            ),
            "projector_token_trace_required": True,
            "projector_must_preserve_sequence_length": True,
        },
        "ours_runtime_switches": {
            "LEARNABLE_PRUNE_CACHED_BOOL_MASK": (
                "0" if args.method == "ours" else None
            ),
            "LEARNABLE_PRUNE_DIRECT_SDPA": (
                "1" if args.method == "ours" else None
            ),
            "LEARNABLE_PRUNE_IMPLICIT_CAUSAL": (
                ("1" if args.ours_implicit_causal else "0")
                if args.method == "ours"
                else None
            ),
            "LEARNABLE_PRUNE_FIXED_LENGTH_GREEDY": (
                "1"
                if args.method == "ours"
                and args.ours_fixed_length_greedy == "on"
                else "0"
                if args.method == "ours"
                else None
            ),
            "LEARNABLE_PRUNE_TRITON_RMS": (
                (
                    "1" if args.ours_triton_rms == "on" else "0"
                )
                if args.method == "ours"
                else None
            ),
            "LEARNABLE_PRUNE_CUDA_GRAPH_DECODE": (
                (
                    "1"
                    if args.ours_cuda_graph_decode == "on"
                    else "0"
                )
                if args.method == "ours"
                else None
            ),
            "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL": (
                (
                    "1"
                    if args.ours_cuda_graph_prefill == "on"
                    else "0"
                )
                if args.method == "ours"
                else None
            ),
            "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL_CACHE_SIZE": (
                "2" if args.method == "ours" else None
            ),
            "LEARNABLE_PRUNE_STATIC_KV_DECODE": (
                "0" if args.method == "ours" else None
            ),
            "LEARNABLE_PRUNE_PROFILE_INTERNAL": (
                "0" if args.method == "ours" else None
            ),
        },
        "optimization_validation": protocol_config["optimization_validation"],
        "gpu": {
            **asdict(gpu_before_load),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_index": adapter.device.index,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": int(properties.total_memory),
            "nvml_physical_index": str(args.physical_gpu_id),
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": __import__("numpy").__version__,
            "cuda_runtime": torch.version.cuda,
            "vanilla_forward_supports_logits_to_keep": (
                "logits_to_keep"
                in inspect.signature(adapter.model.forward).parameters
                if args.method == "vanilla"
                else None
            ),
        },
        "model": {
            "parameter_count": adapter.model_parameter_count,
            "auxiliary_parameter_count": adapter.auxiliary_parameter_count,
            "decoder_layers": len(adapter.layers),
            "learnpruner_loaded_config": loaded_learnpruner_config(
                args,
                adapter,
            ),
        },
        "learnpruner_checkpoint_usage": learnpruner_checkpoint_usage(args),
        "trace_validation": trace_validation,
        "generation_path_validation": generation_path_validation,
        "triton_rms_path_validation": triton_rms_path_validation,
        "cuda_graph_prefill_warmup_diagnostics": (
            prefill_graph_warmup_diagnostics
        ),
        "git": {
            "ours": git_info(OUR_REPO),
            "learnpruner_workdir": git_info(LEARNPRUNER_DIR.parents[1]),
        },
        "command": sys.argv,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def validate_args(args: argparse.Namespace) -> None:
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise ValueError(
            "--run-id may contain only letters, digits, dot, underscore, and hyphen"
        )
    if args.repetition < 0:
        raise ValueError("--repetition must be nonnegative")
    if args.max_new_tokens != FORMAL_MAX_NEW_TOKENS:
        raise ValueError(
            "Formal protocol v2 requires --max-new-tokens "
            f"{FORMAL_MAX_NEW_TOKENS}; its persisted admission artifact is "
            "not valid for another generation length"
        )
    if args.seed != FORMAL_SEED:
        raise ValueError(
            f"Formal protocol v2 requires --seed {FORMAL_SEED}"
        )
    if args.warmup != FORMAL_WARMUP:
        raise ValueError(
            f"Formal protocol v2 requires --warmup {FORMAL_WARMUP}"
        )
    if args.clock_mode == CLOCK_MODE_MONITORED_UNLOCKED:
        if args.locked_sm_clock_mhz is not None:
            raise ValueError(
                "monitored_unlocked requires --locked-sm-clock-mhz to be unset"
            )
    elif args.locked_sm_clock_mhz != FORMAL_LOCKED_SM_CLOCK_MHZ:
        raise ValueError(
            f"{args.clock_mode} requires --locked-sm-clock-mhz "
            f"{FORMAL_LOCKED_SM_CLOCK_MHZ}"
        )
    expected_paths = {
        "official_model": FORMAL_OFFICIAL_MODEL,
        "hf_model": FORMAL_HF_MODEL,
        "ours_checkpoint": FORMAL_OURS_CHECKPOINT,
        "learnpruner_checkpoint": (
            FORMAL_LEARNPRUNER_CHECKPOINT.expanduser().resolve()
        ),
    }
    observed_paths = {
        "official_model": args.official_model.expanduser().resolve(),
        "hf_model": args.hf_model.expanduser().resolve(),
        "ours_checkpoint": args.ours_checkpoint.expanduser().resolve(),
        "learnpruner_checkpoint": (
            args.learnpruner_checkpoint.expanduser().resolve()
        ),
    }
    path_mismatches = {
        key: (str(observed_paths[key]), str(expected))
        for key, expected in expected_paths.items()
        if observed_paths[key] != expected
    }
    if path_mismatches:
        raise ValueError(
            "Formal protocol v2 model/checkpoint paths are immutable: "
            f"{canonical_json(path_mismatches)}"
        )
    if (
        str(args.physical_gpu_id) != FORMAL_PHYSICAL_GPU_ID
        or str(args.expected_cuda_visible_devices)
        != FORMAL_PHYSICAL_GPU_ID
    ):
        raise ValueError(
            "Formal protocol v2 requires physical GPU and "
            f"CUDA_VISIBLE_DEVICES expectation {FORMAL_PHYSICAL_GPU_ID}"
        )
    formal_pruning_values = {
        "ours_budget": (args.ours_budget, FORMAL_OURS_BUDGET),
        "learnpruner_stage1_tokens": (
            args.learnpruner_stage1_tokens,
            FORMAL_LEARNPRUNER_STAGE1_TOKENS,
        ),
        "learnpruner_stage2_tokens": (
            args.learnpruner_stage2_tokens,
            FORMAL_LEARNPRUNER_STAGE2_TOKENS,
        ),
        "learnpruner_prune_layer": (
            args.learnpruner_prune_layer,
            FORMAL_LEARNPRUNER_PRUNE_LAYER,
        ),
        "learnpruner_diversity_ratio": (
            args.learnpruner_diversity_ratio,
            FORMAL_LEARNPRUNER_DIVERSITY_RATIO,
        ),
    }
    pruning_mismatches = {
        key: (observed, expected)
        for key, (observed, expected) in formal_pruning_values.items()
        if observed != expected
    }
    if pruning_mismatches:
        raise ValueError(
            "Formal protocol v2 pruning configuration is immutable: "
            f"{canonical_json(pruning_mismatches)}"
        )
    if args.dtype != "bfloat16":
        raise ValueError(
            "Formal protocol v2 requires the audited bfloat16 dtype"
        )
    if args.attn_implementation != "sdpa":
        raise ValueError(
            "Formal protocol v2 requires the audited SDPA implementation"
        )
    if args.ours_triton_rms == "on" and args.method != "ours":
        raise ValueError(
            "Prefill-only Triton RMS is implemented only for the Ours arm"
        )
    if (
        args.ours_triton_rms == "on"
        and args.ours_fixed_length_greedy != "on"
    ):
        raise ValueError(
            "Prefill-only Triton RMS is admitted only for Ours fast generation"
        )
    if args.ours_triton_rms == "on" and args.dtype != "bfloat16":
        raise ValueError(
            "Prefill-only Triton RMS requires the audited bfloat16 dtype"
        )
    if args.ours_triton_rms == "on" and args.skip_trace_validation:
        raise ValueError(
            "Prefill-only Triton RMS requires its untimed runtime-path probe"
        )
    if args.ours_cuda_graph_decode == "on" and args.method != "ours":
        raise ValueError(
            "CUDA-graph decode is implemented only for the Ours arm"
        )
    if (
        args.ours_cuda_graph_decode == "on"
        and args.ours_fixed_length_greedy != "on"
    ):
        raise ValueError(
            "CUDA-graph decode requires Ours fixed-length greedy generation"
        )
    if args.ours_cuda_graph_decode == "on" and args.warmup < 1:
        raise ValueError(
            "CUDA-graph decode requires at least one untimed warmup so graph "
            "construction/capture cannot enter a measured pair"
        )
    if args.ours_cuda_graph_decode == "on" and args.max_new_tokens > 64:
        raise ValueError(
            "CUDA-graph decode supports at most 64 fixed output tokens"
        )
    if (
        args.ours_cuda_graph_decode == "on"
        and args.attn_implementation != "sdpa"
    ):
        raise ValueError(
            "CUDA-graph decode requires the audited SDPA implementation"
        )
    if args.ours_cuda_graph_decode == "on" and args.skip_trace_validation:
        raise ValueError(
            "CUDA-graph decode requires untimed generation-path validation"
        )
    if args.ours_cuda_graph_prefill == "on" and args.method != "ours":
        raise ValueError(
            "CUDA-graph prefill is implemented only for the Ours arm"
        )
    if (
        args.ours_cuda_graph_prefill == "on"
        and args.ours_fixed_length_greedy != "on"
    ):
        raise ValueError(
            "CUDA-graph prefill requires Ours fixed-length greedy generation"
        )
    if (
        args.ours_cuda_graph_prefill == "on"
        and args.attn_implementation != "sdpa"
    ):
        raise ValueError(
            "CUDA-graph prefill requires the audited SDPA implementation"
        )
    if args.ours_cuda_graph_prefill == "on" and not args.ours_implicit_causal:
        raise ValueError(
            "CUDA-graph prefill requires implicit causal SDPA"
        )
    if args.ours_cuda_graph_prefill == "on" and args.warmup < 1:
        raise ValueError(
            "CUDA-graph prefill requires at least one untimed warmup so graph "
            "construction/initialization cannot enter a measured pair"
        )
    if args.ours_cuda_graph_prefill == "on" and args.skip_trace_validation:
        raise ValueError(
            "CUDA-graph prefill requires untimed generation-path validation"
        )
    if (
        args.ours_cuda_graph_prefill == "on"
        and args.ours_triton_rms != "on"
    ):
        raise ValueError(
            "Formal CUDA-graph prefill is admitted only with the exact "
            "prefill-only Triton RMS path"
        )
    if (
        args.ours_cuda_graph_prefill == "on"
        and args.ours_cuda_graph_decode != "on"
    ):
        raise ValueError(
            "Formal CUDA-graph prefill is admitted only with CUDA-graph decode"
        )
    if args.ours_static_kv_decode != "off":
        raise ValueError(
            "Protocol v2 forbids static-KV decode until full-sample token parity "
            "is validated"
        )


def finalize_metadata_output_hash(
    metadata_path: Path,
    output: Path,
    *,
    prefill_graph_final_diagnostics: dict[str, Any] | None,
) -> str:
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    ordered = sorted(
        (
            int(row["sample_order"]),
            int(row["dataset_index"]),
            row["generated_tail_sha256"],
            row["uninstrumented_generated_tail_sha256"],
            row["measurement_order"],
            row["cuda_graph_residency_warmup_tail_sha256"],
            row["cuda_graph_prefill_callback_runner_id"],
            row["cuda_graph_prefill_pair_status"],
            row["cuda_graph_decode_callback_steps"],
            row["cuda_graph_runner_cache_status"],
            row["triton_rms_runtime_status"],
            int(row["projector_input_tokens"]),
            int(row["projector_output_tokens"]),
        )
        for row in rows
    )
    aggregate = sha256_text(canonical_json(ordered))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["completed_rows"] = len(rows)
    metadata["generated_tail_hash_aggregate"] = aggregate
    metadata["result_tsv_sha256"] = file_sha256(output)
    metadata["cuda_graph_prefill_final_diagnostics"] = (
        prefill_graph_final_diagnostics
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    validate_environment(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sample_path = args.sample_parquet.expanduser().resolve()
    rows = load_rows(sample_path, args.num_samples)
    sample_digest = file_sha256(sample_path)
    monitor = NvmlMonitor(args.physical_gpu_id)
    gpu_before_load = monitor.snapshot()
    validate_nvml_snapshot(gpu_before_load, monitor)
    validate_formal_sm_clock(
        gpu_before_load,
        clock_mode=args.clock_mode,
        label="before model load",
    )
    protocol_config = protocol_payload(
        args, sample_path, sample_digest, gpu_before_load
    )
    computed_config_id = config_id(protocol_config)
    output = output_path_for(args, computed_config_id)
    metadata_output = output.with_suffix(".metadata.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        output.unlink(missing_ok=True)
        metadata_output.unlink(missing_ok=True)
    elif output.exists() and not args.resume:
        raise FileExistsError(f"{output} exists; pass --resume or --overwrite")

    if output.exists():
        resume_guard(
            metadata_output,
            expected_config_id=computed_config_id,
            expected_payload=protocol_config,
            expected_repetition=args.repetition,
            result_path=output,
        )
    elif metadata_output.exists():
        if not args.resume:
            raise ProtocolError(
                f"Refusing a new run with orphan metadata: {metadata_output}"
            )
        resume_guard(
            metadata_output,
            expected_config_id=computed_config_id,
            expected_payload=protocol_config,
            expected_repetition=args.repetition,
        )
    completed, existing_rows = read_completed_rows(output)
    requested_indices = {int(row["dataset_index"]) for row in rows}
    if not completed.issubset(requested_indices):
        raise ProtocolError(
            "Existing v2 result contains dataset indices outside this exact sample"
        )
    resume_rows_guard(
        output,
        args=args,
        expected_config_id=computed_config_id,
        expected_rows=rows,
    )

    print(
        f"{PROTOCOL_VERSION} method={args.method} physical_cuda={args.physical_gpu_id} "
        f"config={computed_config_id} rep={args.repetition} "
        f"resume={existing_rows}/{len(rows)} output={output}",
        flush=True,
    )
    if args.method == "ours":
        os.environ["LEARNABLE_PRUNE_CACHED_BOOL_MASK"] = "0"
        os.environ["LEARNABLE_PRUNE_FIXED_LENGTH_GREEDY"] = (
            "1" if args.ours_fixed_length_greedy == "on" else "0"
        )
        # The admitted kernel is restricted to audited multi-row prefill
        # shapes. Singleton decode stays on authoritative torch RMS, and a
        # runtime kernel failure is a hard protocol error.
        os.environ["LEARNABLE_PRUNE_TRITON_RMS"] = (
            "1" if args.ours_triton_rms == "on" else "0"
        )
        # Graph construction/capture is warmed before measurement. Every
        # instrumented sample must subsequently emit all replay boundaries;
        # eager fallback or a cache miss is a protocol error.
        os.environ["LEARNABLE_PRUNE_CUDA_GRAPH_DECODE"] = (
            "1" if args.ours_cuda_graph_decode == "on" else "0"
        )
        os.environ["LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL"] = (
            "1" if args.ours_cuda_graph_prefill == "on" else "0"
        )
        os.environ["LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL_CACHE_SIZE"] = "2"
        # Static KV remains outside formal v2 pending its full-sample audit.
        os.environ["LEARNABLE_PRUNE_STATIC_KV_DECODE"] = "0"
    adapter = ModelAdapter(args, attach_token_tracer=False)
    loaded_learnpruner_config(args, adapter)
    trace_validation: dict[str, Any] | None = None
    generation_path_validation: dict[str, Any] | None = None
    triton_rms_path_validation: dict[str, Any] | None = None
    prefill_graph_warmup_diagnostics: dict[str, Any] | None = None
    try:
        warmup_rows = rows[: min(args.warmup, len(rows))]
        with torch.inference_mode():
            for row in tqdm(warmup_rows, desc=f"Warmup v2 {args.method}"):
                prepared = adapter.prepare(row)
                adapter.generate(prepared, args.max_new_tokens)
                del prepared
        torch.cuda.synchronize(adapter.device)

        if not args.skip_trace_validation and not output.exists():
            trace_validation = untimed_trace_validation(args, adapter, rows[0])
            if args.method == "ours":
                if args.ours_triton_rms == "on":
                    triton_rms_path_validation = (
                        untimed_triton_rms_path_validation(
                            args,
                            adapter,
                            rows[0],
                        )
                    )
                generation_path_validation = (
                    untimed_generation_path_validation(
                        args,
                        adapter,
                        rows[0],
                    )
                )
            # Restore a no-hook steady-state call after the trace probe.
            prepared = adapter.prepare(rows[0])
            with torch.inference_mode():
                adapter.generate(prepared, args.max_new_tokens)
            torch.cuda.synchronize(adapter.device)
            del prepared

        prefill_graph_warmup_diagnostics = (
            validated_cuda_graph_prefill_snapshot(args, adapter)
        )
        if output.exists():
            metadata = json.loads(metadata_output.read_text(encoding="utf-8"))
            trace_validation = metadata.get("trace_validation")
            generation_path_validation = metadata.get(
                "generation_path_validation"
            )
            triton_rms_path_validation = metadata.get(
                "triton_rms_path_validation"
            )
            persisted_warmup_diagnostics = metadata.get(
                "cuda_graph_prefill_warmup_diagnostics"
            )
            if persisted_warmup_diagnostics is None:
                persisted_warmup_diagnostics = (
                    prefill_graph_warmup_diagnostics
                )
            prefill_graph_warmup_diagnostics = (
                persisted_warmup_diagnostics
            )
        else:
            write_metadata(
                metadata_output,
                args=args,
                adapter=adapter,
                protocol_config=protocol_config,
                computed_config_id=computed_config_id,
                gpu_before_load=gpu_before_load,
                trace_validation=trace_validation,
                generation_path_validation=generation_path_validation,
                triton_rms_path_validation=triton_rms_path_validation,
                prefill_graph_warmup_diagnostics=(
                    prefill_graph_warmup_diagnostics
                ),
                output=output,
            )

        file_exists = output.exists() and output.stat().st_size > 0
        handle = output.open(
            "a" if file_exists else "w", encoding="utf-8", newline=""
        )
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        if not file_exists:
            writer.writeheader()
            handle.flush()

        try:
            progress = tqdm(rows, desc=f"Benchmark v2 {args.method}")
            for sample_order, row in enumerate(progress):
                dataset_index = int(row["dataset_index"])
                if dataset_index in completed:
                    continue
                prepared = adapter.prepare(row)
                graph_residency_warmup_tail_sha256 = (
                    untimed_graph_residency_warmup(
                        args,
                        adapter,
                        prepared,
                    )
                )

                def run_instrumented() -> tuple[
                    StageMeasurements,
                    int,
                    int,
                    list[int],
                    str,
                    dict[str, Any],
                ]:
                    adapter.reset_pruner_stats()
                    timer = GenerationStageTimer(
                        adapter,
                        require_cuda_graph_prefill=(
                            args.method == "ours"
                            and args.ours_fixed_length_greedy == "on"
                            and args.ours_cuda_graph_prefill == "on"
                        ),
                        require_cuda_graph_decode=(
                            args.method == "ours"
                            and args.ours_fixed_length_greedy == "on"
                            and args.ours_cuda_graph_decode == "on"
                        ),
                    )
                    with torch.inference_mode():
                        generated, measurements = timer.run(
                            lambda: generate_v2(
                                adapter,
                                prepared,
                                args.max_new_tokens,
                            ),
                            expected_steps=args.max_new_tokens,
                        )
                    output_info = actual_output_tokens(
                        generated,
                        prepared,
                        measurements.forward_steps,
                    )
                    stats = adapter.latest_pruner_stats()
                    del generated
                    return measurements, *output_info, stats

                def run_uninstrumented() -> tuple[
                    float,
                    float,
                    int,
                    int,
                    list[int],
                    str,
                    dict[str, Any],
                    int,
                    int,
                    int,
                    GpuSnapshot,
                    GpuSnapshot,
                ]:
                    adapter.reset_pruner_stats()
                    baseline = torch.cuda.memory_allocated(adapter.device)
                    torch.cuda.reset_peak_memory_stats(adapter.device)
                    before = monitor.snapshot()
                    validate_nvml_snapshot(before, monitor)
                    validate_formal_sm_clock(
                        before,
                        clock_mode=args.clock_mode,
                        label=(
                            f"repetition {args.repetition} sample "
                            f"{dataset_index} before authoritative generation"
                        ),
                    )
                    with torch.inference_mode():
                        generated, cuda_ms, wall_ms = (
                            timed_generate_uninstrumented(
                                adapter,
                                prepared,
                                args.max_new_tokens,
                            )
                        )
                    after = monitor.snapshot()
                    validate_nvml_snapshot(after, monitor)
                    validate_formal_sm_clock(
                        after,
                        clock_mode=args.clock_mode,
                        label=(
                            f"repetition {args.repetition} sample "
                            f"{dataset_index} after authoritative generation"
                        ),
                    )
                    peak = torch.cuda.max_memory_allocated(adapter.device)
                    reserved = torch.cuda.max_memory_reserved(adapter.device)
                    output_info = actual_output_tokens(
                        generated,
                        prepared,
                        args.max_new_tokens,
                    )
                    stats = adapter.latest_pruner_stats()
                    del generated
                    return (
                        cuda_ms,
                        wall_ms,
                        *output_info,
                        stats,
                        baseline,
                        peak,
                        reserved,
                        before,
                        after,
                    )

                graph_runner_identity_before = (
                    validated_cuda_graph_runner_identity(args, adapter)
                )
                prefill_graph_snapshot_before = (
                    validated_cuda_graph_prefill_snapshot(args, adapter)
                )
                triton_rms_status_before = (
                    validated_triton_rms_runtime_status(args, adapter)
                )
                instrumented_first = (
                    args.repetition + sample_order
                ) % 2 == 0
                measurement_order = (
                    "instrumented_first"
                    if instrumented_first
                    else "uninstrumented_first"
                )
                if instrumented_first:
                    (
                        measured,
                        actual_tokens,
                        output_sequence_length,
                        generated_token_ids,
                        generated_tail_sha256,
                        instrumented_pruner_stats,
                    ) = run_instrumented()
                    (
                        e2e_uninstrumented_cuda_ms,
                        e2e_uninstrumented_wall_ms,
                        uninstrumented_actual_tokens,
                        uninstrumented_sequence_length,
                        uninstrumented_token_ids,
                        uninstrumented_tail_sha256,
                        pruner_stats,
                        baseline_allocated,
                        peak_allocated,
                        peak_reserved,
                        gpu_before,
                        gpu_after,
                    ) = run_uninstrumented()
                else:
                    (
                        e2e_uninstrumented_cuda_ms,
                        e2e_uninstrumented_wall_ms,
                        uninstrumented_actual_tokens,
                        uninstrumented_sequence_length,
                        uninstrumented_token_ids,
                        uninstrumented_tail_sha256,
                        pruner_stats,
                        baseline_allocated,
                        peak_allocated,
                        peak_reserved,
                        gpu_before,
                        gpu_after,
                    ) = run_uninstrumented()
                    (
                        measured,
                        actual_tokens,
                        output_sequence_length,
                        generated_token_ids,
                        generated_tail_sha256,
                        instrumented_pruner_stats,
                    ) = run_instrumented()
                graph_runner_identity_after = (
                    validated_cuda_graph_runner_identity(args, adapter)
                )
                prefill_graph_snapshot_after = (
                    validated_cuda_graph_prefill_snapshot(args, adapter)
                )
                triton_rms_status_after = (
                    validated_triton_rms_runtime_status(args, adapter)
                )
                if graph_runner_identity_after != graph_runner_identity_before:
                    raise ProtocolError(
                        "CUDA-graph runner cache miss/replacement occurred inside "
                        "the adjacent measurement pair: "
                        f"before={graph_runner_identity_before}, "
                        f"after={graph_runner_identity_after}"
                    )
                graph_runner_cache_status = (
                    "single_runner_reused"
                    if graph_runner_identity_after is not None
                    else "disabled"
                )
                prefill_graph_pair_status = (
                    validate_cuda_graph_prefill_pair(
                        prefill_graph_snapshot_before,
                        prefill_graph_snapshot_after,
                        measured.cuda_graph_prefill_callback_runner_id,
                    )
                )
                if triton_rms_status_after != triton_rms_status_before:
                    raise ProtocolError(
                        "Triton-RMS runtime state changed inside the adjacent "
                        "measurement pair: "
                        f"before={triton_rms_status_before}, "
                        f"after={triton_rms_status_after}"
                    )
                if actual_tokens != args.max_new_tokens:
                    raise ProtocolError(
                        f"Actual output length {actual_tokens} != requested "
                        f"{args.max_new_tokens}"
                    )
                if (
                    uninstrumented_actual_tokens != actual_tokens
                    or uninstrumented_sequence_length != output_sequence_length
                    or uninstrumented_token_ids != generated_token_ids
                    or uninstrumented_tail_sha256 != generated_tail_sha256
                ):
                    raise ProtocolError(
                        "Adjacent instrumented/uninstrumented generations differ: "
                        f"instrumented_hash={generated_tail_sha256}, "
                        f"uninstrumented_hash={uninstrumented_tail_sha256}"
                    )
                if (
                    graph_residency_warmup_tail_sha256
                    and graph_residency_warmup_tail_sha256
                    != generated_tail_sha256
                ):
                    raise ProtocolError(
                        "Untimed graph-residency warmup changed the generated "
                        "tail relative to the adjacent measured calls: "
                        f"warmup_hash={graph_residency_warmup_tail_sha256}, "
                        f"measured_hash={generated_tail_sha256}"
                    )
                if instrumented_pruner_stats != pruner_stats:
                    raise ProtocolError(
                        "Adjacent instrumented/uninstrumented pruner stats differ"
                    )
                layer_tokens = infer_layer_tokens(
                    args, adapter, prepared, pruner_stats
                )
                if len(layer_tokens) != len(adapter.layers):
                    raise ProtocolError(
                        f"Layer-token length {len(layer_tokens)} != "
                        f"decoder layers {len(adapter.layers)}"
                    )
                if trace_validation is not None and sample_order == 0:
                    expected_trace = trace_validation["canonical_layer_tokens"]
                    if layer_tokens != expected_trace:
                        raise ProtocolError(
                            "First measured layer lengths disagree with untimed "
                            f"trace: measured={layer_tokens}, trace={expected_trace}"
                        )
                visual_tokens = [
                    token_count - prepared.prompt_text_tokens
                    for token_count in layer_tokens
                ]
                if any(value < 0 for value in visual_tokens):
                    raise ProtocolError(f"Negative visual-token counts: {visual_tokens}")
                avg_visual_tokens = sum(visual_tokens) / len(visual_tokens)
                estimated_tflops = (
                    estimate_prefill_flops_v2(
                        adapter,
                        layer_tokens,
                        measured.projector_input_tokens,
                    )
                    / 1e12
                )
                analytical_cache_mb = (
                    kv_cache_bytes(adapter, layer_tokens) / 1e6
                )
                throughput = actual_tokens / (measured.e2e_wall_ms / 1000.0)
                uninstrumented_throughput = actual_tokens / (
                    e2e_uninstrumented_wall_ms / 1000.0
                )
                decode_tokens = actual_tokens - 1
                decode_throughput = decode_tokens / (
                    measured.decode_wall_ms / 1000.0
                )
                numeric_metrics = {
                    name: _finite_float(getattr(measured, name), name)
                    for name in (
                        "vision_cuda_ms",
                        "projector_cuda_ms",
                        "llm_prefill_cuda_ms",
                        "multimodal_prefill_cuda_ms",
                        "ttft_cuda_ms",
                        "decode_cuda_ms",
                        "tpot_cuda_ms",
                        "e2e_cuda_ms",
                        "ttft_wall_ms",
                        "decode_wall_ms",
                        "tpot_wall_ms",
                        "e2e_wall_ms",
                        "prefill_forward_cuda_ms",
                    )
                }
                e2e_uninstrumented_cuda_ms = _finite_float(
                    e2e_uninstrumented_cuda_ms,
                    "e2e_uninstrumented_cuda_ms",
                )
                e2e_uninstrumented_wall_ms = _finite_float(
                    e2e_uninstrumented_wall_ms,
                    "e2e_uninstrumented_wall_ms",
                )
                if (
                    e2e_uninstrumented_cuda_ms <= 0.0
                    or e2e_uninstrumented_wall_ms <= 0.0
                ):
                    raise ProtocolError(
                        "Uninstrumented E2E timing must be strictly positive"
                    )
                observer_cuda_delta = (
                    measured.e2e_cuda_ms - e2e_uninstrumented_cuda_ms
                )
                observer_wall_delta = (
                    measured.e2e_wall_ms - e2e_uninstrumented_wall_ms
                )
                observer_cuda_ratio = (
                    measured.e2e_cuda_ms / e2e_uninstrumented_cuda_ms
                )
                observer_wall_ratio = (
                    measured.e2e_wall_ms / e2e_uninstrumented_wall_ms
                )
                result_row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "config_id": computed_config_id,
                        "run_id": args.run_id,
                        "method": args.method,
                        "arm_id": arm_id(args),
                        "generation_variant": generation_variant(args),
                        "backend": adapter.backend,
                        "locked_sm_clock_mhz": (
                            ""
                            if args.locked_sm_clock_mhz is None
                            else args.locked_sm_clock_mhz
                        ),
                        "repetition": args.repetition,
                        "sample_order": sample_order,
                        "dataset_index": dataset_index,
                        "source": row["source"],
                        "image": row["image"],
                        "prompt_text_tokens": prepared.prompt_text_tokens,
                        "input_sequence_tokens": prepared.input_sequence_tokens,
                        "projector_input_tokens": (
                            measured.projector_input_tokens
                        ),
                        "projector_output_tokens": (
                            measured.projector_output_tokens
                        ),
                        "requested_output_tokens": args.max_new_tokens,
                        "actual_output_tokens": actual_tokens,
                        "output_sequence_length": output_sequence_length,
                        "generated_token_ids": json_compact(generated_token_ids),
                        "generated_tail_sha256": generated_tail_sha256,
                        "uninstrumented_generated_tail_sha256": (
                            uninstrumented_tail_sha256
                        ),
                        "forward_steps": measured.forward_steps,
                        "measurement_order": measurement_order,
                        "cuda_graph_residency_warmup_tail_sha256": (
                            graph_residency_warmup_tail_sha256
                        ),
                        "cuda_graph_prefill_callback_runner_id": (
                            ""
                            if measured.cuda_graph_prefill_callback_runner_id
                            is None
                            else measured.cuda_graph_prefill_callback_runner_id
                        ),
                        "cuda_graph_prefill_pair_status": (
                            prefill_graph_pair_status
                        ),
                        "cuda_graph_decode_callback_steps": json_compact(
                            measured.cuda_graph_decode_callback_steps
                        ),
                        "cuda_graph_runner_cache_status": (
                            graph_runner_cache_status
                        ),
                        "triton_rms_runtime_status": triton_rms_status_after,
                        "avg_visual_tokens": f"{avg_visual_tokens:.9f}",
                        "estimated_prefill_tflops": f"{estimated_tflops:.9f}",
                        **{
                            name: f"{value:.9f}"
                            for name, value in numeric_metrics.items()
                        },
                        "e2e_uninstrumented_cuda_ms": (
                            f"{e2e_uninstrumented_cuda_ms:.9f}"
                        ),
                        "e2e_uninstrumented_wall_ms": (
                            f"{e2e_uninstrumented_wall_ms:.9f}"
                        ),
                        "observer_e2e_cuda_delta_ms": (
                            f"{observer_cuda_delta:.9f}"
                        ),
                        "observer_e2e_cuda_ratio": (
                            f"{observer_cuda_ratio:.9f}"
                        ),
                        "observer_e2e_wall_delta_ms": (
                            f"{observer_wall_delta:.9f}"
                        ),
                        "observer_e2e_wall_ratio": (
                            f"{observer_wall_ratio:.9f}"
                        ),
                        "throughput_output_tokens_per_s": f"{throughput:.9f}",
                        "throughput_uninstrumented_output_tokens_per_s": (
                            f"{uninstrumented_throughput:.9f}"
                        ),
                        "decode_throughput_tokens_per_s": (
                            f"{decode_throughput:.9f}"
                        ),
                        "decode_forward_cuda_ms": json_compact(
                            measured.decode_forward_cuda_ms
                        ),
                        "decode_step_cuda_ms": json_compact(
                            measured.decode_step_cuda_ms
                        ),
                        "decode_step_wall_ms": json_compact(
                            measured.decode_step_wall_ms
                        ),
                        "analytical_prefill_kv_cache_mb": (
                            f"{analytical_cache_mb:.9f}"
                        ),
                        "baseline_allocated_gb": f"{baseline_allocated / 1e9:.9f}",
                        "peak_allocated_gb": f"{peak_allocated / 1e9:.9f}",
                        "incremental_peak_allocated_gb": (
                            f"{max(0, peak_allocated - baseline_allocated) / 1e9:.9f}"
                        ),
                        "peak_reserved_gb": f"{peak_reserved / 1e9:.9f}",
                        "gpu_sm_clock_mhz_before": snapshot_value(
                            gpu_before.sm_clock_mhz
                        ),
                        "gpu_sm_clock_mhz_after": snapshot_value(
                            gpu_after.sm_clock_mhz
                        ),
                        "gpu_power_w_before": snapshot_value(gpu_before.power_w),
                        "gpu_power_w_after": snapshot_value(gpu_after.power_w),
                        "gpu_temperature_c_before": snapshot_value(
                            gpu_before.temperature_c
                        ),
                        "gpu_temperature_c_after": snapshot_value(
                            gpu_after.temperature_c
                        ),
                        "gpu_utilization_pct_before": snapshot_value(
                            gpu_before.utilization_pct
                        ),
                        "gpu_utilization_pct_after": snapshot_value(
                            gpu_after.utilization_pct
                        ),
                        "layer_prefill_tokens": ",".join(
                            str(value) for value in layer_tokens
                        ),
                        "pruner_stats": json_compact(pruner_stats),
                    }
                missing_fields = set(RESULT_FIELDS) - set(result_row)
                extra_fields = set(result_row) - set(RESULT_FIELDS)
                if missing_fields or extra_fields:
                    raise ProtocolError(
                        "Internal result schema mismatch: "
                        f"missing={sorted(missing_fields)}, "
                        f"extra={sorted(extra_fields)}"
                    )
                writer.writerow(result_row)
                handle.flush()
                progress.set_postfix(
                    prefill=f"{measured.multimodal_prefill_cuda_ms:.1f}ms",
                    tpot=f"{measured.tpot_cuda_ms:.1f}ms",
                    e2e=f"{e2e_uninstrumented_cuda_ms:.1f}ms",
                    observer=f"{observer_cuda_ratio:.3f}x",
                    clock=(
                        "NA"
                        if gpu_after.sm_clock_mhz is None
                        else f"{gpu_after.sm_clock_mhz:.0f}"
                    ),
                )
                del prepared
        finally:
            handle.close()

        final_indices, final_rows = read_completed_rows(output)
        if final_indices != requested_indices or final_rows != len(rows):
            missing = sorted(requested_indices - final_indices)
            raise ProtocolError(
                f"Incomplete v2 output: rows={final_rows}/{len(rows)}, "
                f"missing={missing[:10]}"
            )
        aggregate_hash = finalize_metadata_output_hash(
            metadata_output,
            output,
            prefill_graph_final_diagnostics=(
                validated_cuda_graph_prefill_snapshot(args, adapter)
            ),
        )
        print(
            f"Completed {PROTOCOL_VERSION}: method={args.method}, "
            f"config={computed_config_id}, rows={final_rows}, "
            f"output_hash={aggregate_hash}, output={output}",
            flush=True,
        )
        return 0
    finally:
        adapter.close()
        monitor.close()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
