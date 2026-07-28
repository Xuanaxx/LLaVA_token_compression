#!/usr/bin/env python3
"""Audit the exact Ours inference optimizations on paired DetailCaps inputs.

This is a correctness/admission gate, not the formal multi-method benchmark.
It compares:

* ``slow``: upstream fixed-length generation with explicit causal masks;
* ``eager``: custom fixed greedy + implicit causal + exact prefill RMSNorm;
* ``candidate``: the eager arm plus exact three-segment prefill and decode
  CUDA Graph replay.

The output is an atomic, machine-readable JSON artifact.  A failed run also
writes its partial evidence and traceback before returning a non-zero status.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEFAULT_SAMPLE = HERE / "samples" / "detailcaps_seed42_n500.parquet"
DEFAULT_CHECKPOINT = Path(
    "/data1/chenzixuan/train_output/"
    "official_llava_v1.5_7b_learnable_prune_lightweight_top80pctscope_"
    "multibudget64_128_192_layer18_sample0.2/checkpoint-500"
)
DEFAULT_MODEL = Path("/data1/chenzixuan/model/liuhaotian/llava-v1.5-7b")
RUNTIME_SWITCH_NAMES = (
    "LEARNABLE_PRUNE_FIXED_LENGTH_GREEDY",
    "LEARNABLE_PRUNE_IMPLICIT_CAUSAL",
    "LEARNABLE_PRUNE_TRITON_RMS",
    "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL",
    "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL_CACHE_SIZE",
    "LEARNABLE_PRUNE_CUDA_GRAPH_DECODE",
    "LEARNABLE_PRUNE_STATIC_KV_DECODE",
    "LEARNABLE_PRUNE_DIRECT_SDPA",
    "LEARNABLE_PRUNE_CACHED_BOOL_MASK",
    "LEARNABLE_PRUNE_PROFILE_INTERNAL",
)
ARMS = {
    "slow": {
        "LEARNABLE_PRUNE_FIXED_LENGTH_GREEDY": "0",
        "LEARNABLE_PRUNE_IMPLICIT_CAUSAL": "0",
        "LEARNABLE_PRUNE_TRITON_RMS": "0",
        "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL": "0",
        "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL_CACHE_SIZE": "2",
        "LEARNABLE_PRUNE_CUDA_GRAPH_DECODE": "0",
        "LEARNABLE_PRUNE_STATIC_KV_DECODE": "0",
        "LEARNABLE_PRUNE_DIRECT_SDPA": "1",
        "LEARNABLE_PRUNE_CACHED_BOOL_MASK": "0",
        "LEARNABLE_PRUNE_PROFILE_INTERNAL": "0",
    },
    "eager": {
        "LEARNABLE_PRUNE_FIXED_LENGTH_GREEDY": "1",
        "LEARNABLE_PRUNE_IMPLICIT_CAUSAL": "1",
        "LEARNABLE_PRUNE_TRITON_RMS": "1",
        "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL": "0",
        "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL_CACHE_SIZE": "2",
        "LEARNABLE_PRUNE_CUDA_GRAPH_DECODE": "0",
        "LEARNABLE_PRUNE_STATIC_KV_DECODE": "0",
        "LEARNABLE_PRUNE_DIRECT_SDPA": "1",
        "LEARNABLE_PRUNE_CACHED_BOOL_MASK": "0",
        "LEARNABLE_PRUNE_PROFILE_INTERNAL": "0",
    },
    "candidate": {
        "LEARNABLE_PRUNE_FIXED_LENGTH_GREEDY": "1",
        "LEARNABLE_PRUNE_IMPLICIT_CAUSAL": "1",
        "LEARNABLE_PRUNE_TRITON_RMS": "1",
        "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL": "1",
        "LEARNABLE_PRUNE_CUDA_GRAPH_PREFILL_CACHE_SIZE": "2",
        "LEARNABLE_PRUNE_CUDA_GRAPH_DECODE": "1",
        "LEARNABLE_PRUNE_STATIC_KV_DECODE": "0",
        "LEARNABLE_PRUNE_DIRECT_SDPA": "1",
        "LEARNABLE_PRUNE_CACHED_BOOL_MASK": "0",
        "LEARNABLE_PRUNE_PROFILE_INTERNAL": "0",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-parquet", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--official-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ours-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ours-budget", type=int, choices=(64, 128, 192), default=64)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-cuda-visible-devices", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            HERE
            / "results"
            / "audits"
            / "ours_combined_gate_prefill_seed42_n32.json"
        ),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def command_output(command: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception as exc:
        return f"<error: {exc!r}>"


def repo_provenance() -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "commit": command_output(["git", "rev-parse", "HEAD"], REPO),
        "status": command_output(["git", "status", "--short"], REPO).splitlines(),
        "tracked_diff_sha256": bytes_sha256(diff),
    }


def json_tokens(tensor: torch.Tensor, count: int) -> list[list[int]]:
    tail = tensor.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if tail.ndim != 2 or tail.shape[0] != 1 or tail.shape[1] < count:
        raise AssertionError(
            f"Expected [1, >= {count}] output tokens, got {tuple(tail.shape)}"
        )
    return [[int(value) for value in tail[0, -count:].tolist()]]


def token_bytes(tokens: list[list[int]]) -> bytes:
    return np.asarray(tokens, dtype="<i8").tobytes(order="C")


def token_hash(tokens: list[list[int]]) -> str:
    return bytes_sha256(token_bytes(tokens))


def canonical_json_hash(tokens: list[list[list[int]]]) -> str:
    payload = json.dumps(tokens, separators=(",", ":"), ensure_ascii=True)
    return bytes_sha256(payload.encode("utf-8"))


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "stdev": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


@contextlib.contextmanager
def runtime_arm(name: str) -> Iterator[None]:
    flags = ARMS[name]
    previous = {key: os.environ.get(key) for key in RUNTIME_SWITCH_NAMES}
    os.environ.update(flags)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def expected_graph_trace(steps: int) -> list[list[Any]]:
    return [
        [phase, step]
        for step in range(1, steps + 1)
        for phase in ("decode_start", "decode_end")
    ]


def expected_prefill_trace(runner_id: int) -> list[list[Any]]:
    return [
        ["prefill_start", int(runner_id)],
        ["prefill_end", int(runner_id)],
    ]


def expected_candidate_trace(
    runner_id: int,
    decode_steps: int,
) -> list[list[Any]]:
    return expected_prefill_trace(runner_id) + expected_graph_trace(
        decode_steps
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit artifact: {output}")

    sample_path = args.sample_parquet.expanduser().resolve()
    checkpoint = args.ours_checkpoint.expanduser().resolve()
    official_model = args.official_model.expanduser().resolve()
    script_path = Path(__file__).resolve()
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + f"-pid{os.getpid()}"
    )
    audit: dict[str, Any] = {
        "schema_version": "ours-combined-gate-v2",
        "run_id": run_id,
        "status": "running",
        "created_at_utc": created_at,
        "command": sys.argv,
        "paths": {
            "artifact": str(output),
            "script": str(script_path),
            "sample_parquet": str(sample_path),
            "official_model": str(official_model),
            "ours_checkpoint": str(checkpoint),
        },
        "digests": {
            "script_sha256": file_sha256(script_path),
            "sample_parquet_sha256": file_sha256(sample_path),
            "checkpoint_config_sha256": file_sha256(
                checkpoint / "learnable_prune_config.pt"
            ),
            "checkpoint_predictor_sha256": file_sha256(
                checkpoint / "predictor.pt"
            ),
            "official_model_config_sha256": file_sha256(
                official_model / "config.json"
            ),
            "official_modeling_source_sha256": file_sha256(
                REPO
                / "llava/model/learnable_prune_lightweight_scope_finalwipe/"
                "official_modeling.py"
            ),
            "llava_llama_source_sha256": file_sha256(
                REPO / "llava/model/language_model/llava_llama.py"
            ),
        },
        "repo": repo_provenance(),
        "protocol": {
            "dataset": "CAPTURE / DetailCaps-4870",
            "selection": f"first {args.num_samples} parquet rows in stored order",
            "num_samples": args.num_samples,
            "batch_size": 1,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.max_new_tokens,
            "preprocessing_in_timing": False,
            "vision_encoder_in_timing": True,
            "prefill_definition": "outer CUDA event around independent generate(min=max=1)",
            "e2e_definition": (
                "outer CUDA event around independent generate(min=max=max_new_tokens)"
            ),
            "decode_definition": "paired E2E minus matching-arm independent prefill",
            "pair_order": {
                "even_rows": "slow then eager then candidate",
                "odd_rows": "candidate then eager then slow",
                "eager_position": "between slow and candidate",
            },
            "capture_excluded": True,
            "prefill_graph_scope": (
                "three decoder segments only; predictor/SCOPE, middle "
                "scoring/TopK/sort/packing, and final wipe remain eager"
            ),
            "arms": ARMS,
        },
        "rows": [],
    }
    adapter = None
    callback_handles: list[Any] = []
    boundary_trace: list[list[Any]] = []
    try:
        from efficiency_exp import efficiency_v2_runtime as benchmark

        benchmark_args = argparse.Namespace(
            method="ours",
            sample_parquet=sample_path,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            dtype="bfloat16",
            attn_implementation="sdpa",
            device=args.device,
            expected_cuda_visible_devices=(
                args.expected_cuda_visible_devices
            ),
            official_model=official_model,
            ours_checkpoint=checkpoint,
            ours_budget=args.ours_budget,
            ours_implicit_causal=True,
        )
        benchmark.validate_environment(benchmark_args)

        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        rows = benchmark.load_rows(sample_path, args.num_samples)
        audit["protocol"]["dataset_indices"] = [
            int(row["dataset_index"]) for row in rows
        ]

        os.environ.update(ARMS["eager"])
        adapter = benchmark.ModelAdapter(
            benchmark_args,
            attach_token_tracer=False,
        )
        model = adapter.model
        device = adapter.device
        properties = torch.cuda.get_device_properties(device)
        audit["software"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_git_version": getattr(torch.version, "git_version", None),
            "transformers": __import__("transformers").__version__,
            "triton": getattr(__import__("triton"), "__version__", None),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
        audit["gpu"] = {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_device": str(device),
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": int(properties.total_memory),
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version",
                    "--format=csv,noheader",
                ]
            ).splitlines(),
        }
        audit["model"] = {
            "parameter_count": int(adapter.model_parameter_count),
            "auxiliary_parameter_count": int(adapter.auxiliary_parameter_count),
            "decoder_layers": len(adapter.layers),
            "ours_budget": args.ours_budget,
        }

        from llava.model.learnable_prune_lightweight_scope_finalwipe import (
            official_modeling as official,
        )

        boundary_trace = []

        def plain_generate(
            arm: str,
            prepared: Any,
            new_tokens: int,
        ) -> tuple[torch.Tensor, list[list[Any]]]:
            boundary_trace.clear()
            with torch.inference_mode(), runtime_arm(arm):
                result = adapter.generate(prepared, new_tokens)
                torch.cuda.synchronize(device)
            return result, [list(event) for event in boundary_trace]

        def timed_generate(
            arm: str,
            prepared: Any,
            new_tokens: int,
        ) -> dict[str, Any]:
            boundary_trace.clear()
            with torch.inference_mode(), runtime_arm(arm):
                result, cuda_ms, wall_ms = benchmark.timed_generate(
                    adapter,
                    prepared,
                    new_tokens,
                )
            tokens = json_tokens(result, new_tokens)
            return {
                "cuda_ms": float(cuda_ms),
                "wall_ms": float(wall_ms),
                "tokens": tokens,
                "token_sha256_int64_le": token_hash(tokens),
                "callback_trace": [
                    list(event) for event in boundary_trace
                ],
            }

        first_prepared = adapter.prepare(rows[0])

        # Instrument a separate validation call, never a measured sample.
        eligibility_events: list[dict[str, Any]] = []
        launch_events: list[dict[str, Any]] = []
        original_eligibility = official._triton_rms_norm_is_eligible
        original_launch = official._launch_triton_rms_norm
        original_prepare_mask = model._prepare_mask
        implicit_prepare_mask_calls = 0
        slow_prepare_mask_calls = 0
        prepare_counter_target = "implicit"

        def traced_eligibility(norm: Any, hidden_states: torch.Tensor) -> bool:
            eligible = bool(original_eligibility(norm, hidden_states))
            eligibility_events.append(
                {
                    "shape": [int(value) for value in hidden_states.shape],
                    "eligible": eligible,
                    "norm_type": type(norm).__name__,
                }
            )
            return eligible

        def traced_launch(
            norm: Any,
            hidden_states: torch.Tensor,
            result: torch.Tensor,
        ) -> None:
            launch_events.append(
                {
                    "shape": [int(value) for value in hidden_states.shape],
                    "norm_type": type(norm).__name__,
                }
            )
            return original_launch(norm, hidden_states, result)

        def traced_prepare_mask(*mask_args: Any, **mask_kwargs: Any) -> Any:
            nonlocal implicit_prepare_mask_calls, slow_prepare_mask_calls
            if prepare_counter_target == "implicit":
                implicit_prepare_mask_calls += 1
            else:
                slow_prepare_mask_calls += 1
            return original_prepare_mask(*mask_args, **mask_kwargs)

        official._triton_rms_norm_is_eligible = traced_eligibility
        official._launch_triton_rms_norm = traced_launch
        model._prepare_mask = traced_prepare_mask
        try:
            runtime_disabled_before = bool(
                official._TRITON_RMS_RUNTIME_DISABLED
            )
            probe_eager, probe_eager_trace = plain_generate(
                "eager", first_prepared, 2
            )
            prepare_counter_target = "slow"
            probe_slow, probe_slow_trace = plain_generate(
                "slow", first_prepared, 2
            )
            runtime_disabled_after = bool(
                official._TRITON_RMS_RUNTIME_DISABLED
            )
        finally:
            official._triton_rms_norm_is_eligible = original_eligibility
            official._launch_triton_rms_norm = original_launch
            del model._prepare_mask

        probe_eager_tokens = json_tokens(probe_eager, 2)
        probe_slow_tokens = json_tokens(probe_slow, 2)
        singleton_eligibility = [
            event
            for event in eligibility_events
            if event["shape"][1] == 1
        ]
        multirow_eligibility = [
            event
            for event in eligibility_events
            if event["shape"][1] > 1
        ]
        audit["path_validation"] = {
            "outside_measured_samples": True,
            "token_equal": probe_eager_tokens == probe_slow_tokens,
            "eager_tokens": probe_eager_tokens,
            "slow_tokens": probe_slow_tokens,
            "eager_callback_trace": probe_eager_trace,
            "slow_callback_trace": probe_slow_trace,
            "triton_runtime_disabled_before": runtime_disabled_before,
            "triton_runtime_disabled_after": runtime_disabled_after,
            "rms_eligibility_events": eligibility_events,
            "rms_launch_events": launch_events,
            "multirow_eligible_count": sum(
                bool(event["eligible"]) for event in multirow_eligibility
            ),
            "singleton_call_count": len(singleton_eligibility),
            "singleton_eligible_count": sum(
                bool(event["eligible"]) for event in singleton_eligibility
            ),
            "singleton_launch_count": sum(
                event["shape"][1] == 1 for event in launch_events
            ),
            "implicit_prepare_mask_calls": implicit_prepare_mask_calls,
            "slow_prepare_mask_calls": slow_prepare_mask_calls,
            "measured_sample_note": (
                "All function wrappers were restored before warmup/capture/"
                "paired timing. RMS eligibility is deterministic from the "
                "recorded flags, shapes, dtype, device, torch git version, "
                "module state, and runtime-disabled state."
            ),
        }
        if probe_eager_tokens != probe_slow_tokens:
            raise AssertionError("Two-token optimized path changed token IDs")
        if probe_eager_trace or probe_slow_trace:
            raise AssertionError("Graph callbacks fired with graph disabled")
        if runtime_disabled_before or runtime_disabled_after:
            raise AssertionError("Triton RMS runtime was disabled")
        if not launch_events or not multirow_eligibility:
            raise AssertionError("Validation probe did not launch prefill RMS")
        if any(event["shape"][1] == 1 for event in launch_events):
            raise AssertionError("Singleton decode incorrectly launched Triton RMS")
        if any(event["eligible"] for event in singleton_eligibility):
            raise AssertionError("Singleton decode was RMS-eligible")
        if implicit_prepare_mask_calls != 0 or slow_prepare_mask_calls <= 0:
            raise AssertionError(
                "Implicit/explicit causal-mask validation did not take expected paths"
            )

        # Warm every non-graph path, then measure graph construction separately.
        warm_slow, warm_slow_trace = plain_generate(
            "slow", first_prepared, args.max_new_tokens
        )
        warm_eager, warm_eager_trace = plain_generate(
            "eager", first_prepared, args.max_new_tokens
        )
        if warm_slow_trace or warm_eager_trace:
            raise AssertionError("Graph callbacks fired during eager warmup")
        warm_slow_tokens = json_tokens(warm_slow, args.max_new_tokens)
        warm_eager_tokens = json_tokens(warm_eager, args.max_new_tokens)
        if warm_slow_tokens != warm_eager_tokens:
            raise AssertionError("Warm eager output changed token IDs")
        if model._fixed_greedy_cuda_graph_runners:
            raise AssertionError("Decode CUDA Graph runner existed before capture")
        if model._prefill_cuda_graph_runners:
            raise AssertionError("Prefill CUDA Graph runner existed before capture")

        # Both prefill and decode construction happen without callbacks and
        # outside paired timings. Callback-aware calls below are cache-hit-only.
        capture = timed_generate(
            "candidate", first_prepared, args.max_new_tokens
        )
        if capture["callback_trace"]:
            raise AssertionError("Capture/init unexpectedly emitted callbacks")
        if capture["tokens"] != warm_slow_tokens:
            raise AssertionError("Capture call changed token IDs")
        decode_graph_cache = model._fixed_greedy_cuda_graph_runners
        decode_runners = [
            value
            for value in decode_graph_cache.values()
            if isinstance(value, official._FixedGreedyCudaGraphRunner)
        ]
        if len(decode_graph_cache) != 1 or len(decode_runners) != 1:
            raise AssertionError("Expected exactly one live decode graph runner")
        decode_runner = decode_runners[0]
        decode_runner_identity = id(decode_runner)

        prefill_graph_cache = model._prefill_cuda_graph_runners
        prefill_runners = [
            value
            for value in prefill_graph_cache.values()
            if isinstance(
                value,
                official._ThreeSegmentPrefillCudaGraphRunner,
            )
        ]
        if len(prefill_graph_cache) != 1 or len(prefill_runners) != 1:
            raise AssertionError("Expected exactly one live prefill graph runner")
        prefill_runner = prefill_runners[0]
        prefill_runner_identity = id(prefill_runner)
        prefill_runner_id = int(prefill_runner.runner_id)
        prefill_diagnostics_after_capture = (
            model.get_cuda_graph_prefill_diagnostics()
        )
        if (
            prefill_diagnostics_after_capture["captures"] != 1
            or prefill_diagnostics_after_capture["replay_successes"] != 1
            or prefill_diagnostics_after_capture["signature_mismatches"] != 0
            or prefill_diagnostics_after_capture["last_runner_id"]
            != prefill_runner_id
        ):
            raise AssertionError(
                "Prefill capture diagnostics did not prove one capture/replay"
            )

        audit["capture"] = {
            "cuda_ms": capture["cuda_ms"],
            "wall_ms": capture["wall_ms"],
            "tokens": capture["tokens"],
            "token_sha256_int64_le": capture["token_sha256_int64_le"],
            "callback_trace": capture["callback_trace"],
            "decode_runner_identity": decode_runner_identity,
            "decode_runner_cache_entries": len(decode_graph_cache),
            "prefill_runner_identity": prefill_runner_identity,
            "prefill_runner_id": prefill_runner_id,
            "prefill_runner_cache_entries": len(prefill_graph_cache),
            "prefill_diagnostics": prefill_diagnostics_after_capture,
            "excluded_from_paired_timings": True,
        }

        callback_handles.extend(
            [
                model.register_cuda_graph_prefill_boundary_callback(
                    lambda phase, runner_id: boundary_trace.append(
                        [str(phase), int(runner_id)]
                    )
                ),
                model.register_cuda_graph_decode_boundary_callback(
                    lambda phase, step: boundary_trace.append(
                        [str(phase), int(step)]
                    )
                ),
            ]
        )
        expected_prefill = expected_prefill_trace(prefill_runner_id)
        expected_full_trace = expected_candidate_trace(
            prefill_runner_id,
            args.max_new_tokens - 1,
        )
        validation_replay, validation_trace = plain_generate(
            "candidate",
            first_prepared,
            args.max_new_tokens,
        )
        if validation_trace != expected_full_trace:
            raise AssertionError(
                "Warmed candidate validation did not emit exact combined trace"
            )
        if json_tokens(validation_replay, args.max_new_tokens) != warm_slow_tokens:
            raise AssertionError("Warmed candidate validation changed token IDs")
        audit["capture"]["post_capture_validation_trace"] = validation_trace
        audit["capture"]["post_capture_prefill_diagnostics"] = (
            model.get_cuda_graph_prefill_diagnostics()
        )

        torch.cuda.reset_peak_memory_stats(device)
        aggregate_tokens: dict[str, list[list[list[int]]]] = {
            "slow_prefill": [],
            "eager_prefill": [],
            "candidate_prefill": [],
            "slow": [],
            "eager": [],
            "candidate": [],
        }
        raw: dict[str, list[float]] = {
            "slow_prefill_cuda_ms": [],
            "eager_prefill_cuda_ms": [],
            "candidate_prefill_cuda_ms": [],
            "slow_e2e_cuda_ms": [],
            "eager_e2e_cuda_ms": [],
            "candidate_e2e_cuda_ms": [],
            "slow_decode_cuda_ms": [],
            "eager_decode_cuda_ms": [],
            "candidate_decode_cuda_ms": [],
            "candidate_vs_slow_e2e_speedup": [],
            "candidate_vs_slow_decode_speedup": [],
            "candidate_vs_eager_e2e_speedup": [],
            "candidate_vs_eager_decode_speedup": [],
        }

        for row_index, row in enumerate(rows):
            prepared = first_prepared if row_index == 0 else adapter.prepare(row)
            prefill_order = (
                ("slow", "eager", "candidate")
                if row_index % 2 == 0
                else ("candidate", "eager", "slow")
            )
            prefill_results = {
                arm: timed_generate(arm, prepared, 1)
                for arm in prefill_order
            }
            if prefill_results["slow"]["callback_trace"]:
                raise AssertionError(
                    f"Graph callback fired in row {row_index} slow prefill"
                )
            if prefill_results["eager"]["callback_trace"]:
                raise AssertionError(
                    f"Graph callback fired in row {row_index} eager prefill"
                )
            if (
                prefill_results["candidate"]["callback_trace"]
                != expected_prefill
            ):
                raise AssertionError(
                    f"Prefill graph replay/fallback trace mismatch at row "
                    f"{row_index}"
                )
            if not (
                prefill_results["slow"]["tokens"]
                == prefill_results["eager"]["tokens"]
                == prefill_results["candidate"]["tokens"]
            ):
                raise AssertionError(
                    f"Prefill first-token mismatch at row {row_index}"
                )

            full_order = (
                ("slow", "eager", "candidate")
                if row_index % 2 == 0
                else ("candidate", "eager", "slow")
            )
            full_results = {
                arm: timed_generate(
                    arm,
                    prepared,
                    args.max_new_tokens,
                )
                for arm in full_order
            }
            if full_results["slow"]["callback_trace"]:
                raise AssertionError(
                    f"Graph callback fired in row {row_index} slow arm"
                )
            if full_results["eager"]["callback_trace"]:
                raise AssertionError(
                    f"Graph callback fired in row {row_index} eager arm"
                )
            if (
                full_results["candidate"]["callback_trace"]
                != expected_full_trace
            ):
                raise AssertionError(
                    f"Combined graph replay/fallback trace mismatch at row "
                    f"{row_index}"
                )
            if not (
                full_results["slow"]["tokens"]
                == full_results["eager"]["tokens"]
                == full_results["candidate"]["tokens"]
            ):
                raise AssertionError(
                    f"Generated-token mismatch at DetailCaps row {row_index}"
                )
            current_decode_runners = [
                value
                for value in model._fixed_greedy_cuda_graph_runners.values()
                if isinstance(value, official._FixedGreedyCudaGraphRunner)
            ]
            if (
                len(model._fixed_greedy_cuda_graph_runners) != 1
                or len(current_decode_runners) != 1
                or id(current_decode_runners[0])
                != decode_runner_identity
            ):
                raise AssertionError(
                    f"Decode CUDA Graph runner miss/fallback at row {row_index}"
                )
            current_prefill_runners = [
                value
                for value in model._prefill_cuda_graph_runners.values()
                if isinstance(
                    value,
                    official._ThreeSegmentPrefillCudaGraphRunner,
                )
            ]
            if (
                len(model._prefill_cuda_graph_runners) != 1
                or len(current_prefill_runners) != 1
                or id(current_prefill_runners[0])
                != prefill_runner_identity
                or int(current_prefill_runners[0].runner_id)
                != prefill_runner_id
            ):
                raise AssertionError(
                    f"Prefill CUDA Graph runner miss/fallback at row "
                    f"{row_index}"
                )
            if official._TRITON_RMS_RUNTIME_DISABLED:
                raise AssertionError(
                    f"Triton RMS runtime disabled at row {row_index}"
                )

            slow_prefill = prefill_results["slow"]["cuda_ms"]
            eager_prefill = prefill_results["eager"]["cuda_ms"]
            candidate_prefill = prefill_results["candidate"]["cuda_ms"]
            slow_e2e = full_results["slow"]["cuda_ms"]
            eager_e2e = full_results["eager"]["cuda_ms"]
            candidate_e2e = full_results["candidate"]["cuda_ms"]
            slow_decode = slow_e2e - slow_prefill
            eager_decode = eager_e2e - eager_prefill
            candidate_decode = candidate_e2e - candidate_prefill
            metrics = {
                "slow_prefill_cuda_ms": slow_prefill,
                "eager_prefill_cuda_ms": eager_prefill,
                "candidate_prefill_cuda_ms": candidate_prefill,
                "slow_e2e_cuda_ms": slow_e2e,
                "eager_e2e_cuda_ms": eager_e2e,
                "candidate_e2e_cuda_ms": candidate_e2e,
                "slow_decode_cuda_ms": slow_decode,
                "eager_decode_cuda_ms": eager_decode,
                "candidate_decode_cuda_ms": candidate_decode,
                "candidate_vs_slow_e2e_speedup": slow_e2e / candidate_e2e,
                "candidate_vs_slow_decode_speedup": slow_decode / candidate_decode,
                "candidate_vs_eager_e2e_speedup": eager_e2e / candidate_e2e,
                "candidate_vs_eager_decode_speedup": eager_decode / candidate_decode,
            }
            for name, value in metrics.items():
                raw[name].append(float(value))
            for arm in ("slow", "eager", "candidate"):
                aggregate_tokens[f"{arm}_prefill"].append(
                    prefill_results[arm]["tokens"]
                )
            for arm in ("slow", "eager", "candidate"):
                aggregate_tokens[arm].append(full_results[arm]["tokens"])

            audit["rows"].append(
                {
                    "row_index": row_index,
                    "dataset_index": int(row["dataset_index"]),
                    "source": str(row["source"]),
                    "image": str(row["image"]),
                    "prefill_order": list(prefill_order),
                    "full_order": list(full_order),
                    "prefill": prefill_results,
                    "full": full_results,
                    "metrics": metrics,
                    "same_decode_runner_identity": True,
                    "same_prefill_runner_identity": True,
                    "prefill_runner_id": prefill_runner_id,
                    "triton_runtime_disabled": False,
                }
            )
            print(
                f"row={row_index:02d} "
                f"prefill={slow_prefill:.3f}/{eager_prefill:.3f}"
                f"->{candidate_prefill:.3f} "
                f"e2e={slow_e2e:.3f}/{eager_e2e:.3f}->{candidate_e2e:.3f} "
                f"speedup={metrics['candidate_vs_slow_e2e_speedup']:.3f}x",
                flush=True,
            )

        output_hashes: dict[str, dict[str, str]] = {}
        for arm, token_rows in aggregate_tokens.items():
            raw_bytes = b"".join(
                token_bytes(tokens) for tokens in token_rows
            )
            output_hashes[arm] = {
                "aggregate_int64_le_bytes_sha256": bytes_sha256(raw_bytes),
                "aggregate_canonical_json_sha256": canonical_json_hash(
                    token_rows
                ),
            }
        if not (
            output_hashes["slow"]
            == output_hashes["eager"]
            == output_hashes["candidate"]
        ):
            raise AssertionError("Aggregate full-output hashes differ")
        if not (
            output_hashes["slow_prefill"]
            == output_hashes["eager_prefill"]
            == output_hashes["candidate_prefill"]
        ):
            raise AssertionError("Aggregate prefill-output hashes differ")
        distinct_candidate_hashes = {
            row["full"]["candidate"]["token_sha256_int64_le"]
            for row in audit["rows"]
        }
        if len(audit["rows"]) >= 2 and len(distinct_candidate_hashes) < 2:
            raise AssertionError(
                "Different DetailCaps images produced only one candidate "
                "token hash; static-input dependence was not demonstrated"
            )

        audit["raw_paired_arrays"] = raw
        audit["summary"] = {
            name: distribution(values) for name, values in raw.items()
        }
        audit["output_hashes"] = output_hashes
        audit["memory"] = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes_after_capture": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_reserved_bytes_after_capture": int(
                torch.cuda.max_memory_reserved(device)
            ),
        }
        audit["capture"]["estimated_incremental_cuda_ms_vs_candidate_p50"] = (
            audit["capture"]["cuda_ms"]
            - audit["summary"]["candidate_e2e_cuda_ms"]["p50"]
        )
        audit["capture"]["estimated_incremental_wall_ms_vs_candidate_p50"] = (
            audit["capture"]["wall_ms"]
            - statistics.median(
                [
                    float(row["full"]["candidate"]["wall_ms"])
                    for row in audit["rows"]
                ]
            )
        )
        final_prefill_diagnostics = (
            model.get_cuda_graph_prefill_diagnostics()
        )
        expected_prefill_replays = 2 + 2 * len(audit["rows"])
        if (
            final_prefill_diagnostics["captures"] != 1
            or final_prefill_diagnostics["capture_failures"] != 0
            or final_prefill_diagnostics["replay_failures"] != 0
            or final_prefill_diagnostics["eager_fallbacks"] != 0
            or final_prefill_diagnostics["hook_fallbacks"] != 0
            or final_prefill_diagnostics["signature_mismatches"] != 0
            or final_prefill_diagnostics["cache_entries"] != 1
            or final_prefill_diagnostics["replay_successes"]
            != expected_prefill_replays
            or final_prefill_diagnostics["runners"][0]["replay_count"]
            != expected_prefill_replays
            or final_prefill_diagnostics["last_runner_id"]
            != prefill_runner_id
        ):
            raise AssertionError(
                "Final prefill graph diagnostics show capture/replay "
                "fallback, failure, or runner drift"
            )
        audit["prefill_graph_diagnostics_final"] = (
            final_prefill_diagnostics
        )
        audit["checks"] = {
            "sample_count": len(audit["rows"]),
            "full_token_mismatch_count": 0,
            "prefill_token_mismatch_count": 0,
            "prefill_callback_pairs_per_candidate": 1,
            "decode_callback_pairs_per_full_candidate": (
                args.max_new_tokens - 1
            ),
            "prefill_callback_trace_exact_cases": len(audit["rows"]),
            "combined_callback_trace_exact_cases": len(audit["rows"]),
            "graph_fallback_or_cache_miss_cases": 0,
            "same_prefill_runner_identity_cases": len(audit["rows"]),
            "same_decode_runner_identity_cases": len(audit["rows"]),
            "prefill_capture_count": 1,
            "prefill_replay_count": expected_prefill_replays,
            "prefill_signature_mismatch_count": 0,
            "distinct_candidate_full_token_hashes": len(
                distinct_candidate_hashes
            ),
            "capture_excluded": True,
            "triton_multirow_launch_observed": bool(launch_events),
            "triton_singleton_launch_count": 0,
            "triton_runtime_disabled": False,
            "implicit_mask_materialization_calls": 0,
        }
        audit["status"] = "pass"
        audit["passed"] = True
        audit["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:
        audit["status"] = "fail"
        audit["passed"] = False
        audit["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        audit["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "repr": repr(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        for callback_handle in callback_handles:
            callback_handle.remove()
        if adapter is not None:
            adapter.close()
        atomic_write_json(output, audit)
        print(f"AUDIT_ARTIFACT={output}", flush=True)
        print(f"AUDIT_STATUS={audit['status']}", flush=True)


if __name__ == "__main__":
    main()
