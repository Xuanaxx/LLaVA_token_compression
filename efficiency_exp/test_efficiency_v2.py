#!/usr/bin/env python3
"""CPU contract tests for the formal three-method efficiency-v2 pipeline."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import benchmark_efficiency_v2 as benchmark
import efficiency_v2_runtime as runtime
import summarize_results_v2 as summary


FORMAL_OURS_FLAGS = (
    "--ours-implicit-causal",
    "--ours-fixed-length-greedy",
    "on",
    "--ours-triton-rms",
    "on",
    "--ours-cuda-graph-prefill",
    "on",
    "--ours-cuda-graph-decode",
    "on",
    "--ours-static-kv-decode",
    "off",
)


def formal_args(method: str):
    arguments = ["--method", method]
    if method == "ours":
        arguments.extend(FORMAL_OURS_FLAGS)
    return benchmark.parse_args(arguments)


class FormalProtocolContractTests(unittest.TestCase):
    def test_result_schema_and_methods_are_exactly_three_arm_v2(self) -> None:
        self.assertEqual(
            benchmark.METHODS,
            ("vanilla", "ours", "learnpruner"),
        )
        self.assertEqual(
            summary.ARMS,
            ("vanilla", "learnpruner", "ours"),
        )
        self.assertEqual(
            len(benchmark.RESULT_FIELDS),
            len(set(benchmark.RESULT_FIELDS)),
        )
        self.assertEqual(benchmark.RESULT_FIELDS, summary.RESULT_FIELDS)
        self.assertTrue(
            {
                "cuda_graph_residency_warmup_tail_sha256",
                "cuda_graph_prefill_callback_runner_id",
                "cuda_graph_prefill_pair_status",
                "cuda_graph_decode_callback_steps",
                "cuda_graph_runner_cache_status",
                "triton_rms_runtime_status",
                "layer_prefill_tokens",
            }.issubset(benchmark.RESULT_FIELDS)
        )
    def _run_fake_unlocked_runner(
        self,
    ) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            python_log = root / "python.log"
            nvidia_log = root / "nvidia.log"

            fake_python = fake_bin / "fake-python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"${PYTHON_LOG}\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            fake_nvidia_smi = fake_bin / "nvidia-smi"
            fake_nvidia_smi.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"${NVIDIA_LOG}\"\n",
                encoding="utf-8",
            )
            fake_nvidia_smi.chmod(0o755)

            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment.get('PATH', '')}",
                    "PYTHON_BIN": str(fake_python),
                    "PYTHON_LOG": str(python_log),
                    "NVIDIA_LOG": str(nvidia_log),
                    "RESULT_ROOT": str(root / "results"),
                    "SAMPLE_DIR": str(root / "samples"),
                    "NUM_SAMPLES": "500",
                    "REPETITIONS": "4",
                    "WARMUP": "20",
                    "MAX_NEW_TOKENS": "32",
                    "PHYSICAL_GPU": "2",
                    "RUN_ID": "three_arm_cpu_contract",
                    "LOCKED_SM_CLOCK_MHZ": "",
                    "GPU_LOCK_PATH": str(root / "runner.lock"),
                    "DEFER_SUMMARY": "true",
                }
            )
            environment.pop("CLOCK_LOCK_MODE", None)
            completed = subprocess.run(
                ["/bin/bash", str(HERE / "run_benchmark_v2.sh")],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            python_lines = (
                python_log.read_text(encoding="utf-8").splitlines()
                if python_log.exists()
                else []
            )
            nvidia_lines = (
                nvidia_log.read_text(encoding="utf-8").splitlines()
                if nvidia_log.exists()
                else []
            )
            return completed, python_lines, nvidia_lines

    def test_runner_is_unlocked_and_four_by_three_near_balanced(self) -> None:
        completed, python_lines, nvidia_lines = (
            self._run_fake_unlocked_runner()
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(any("-lgc" in line for line in nvidia_lines))
        self.assertFalse(any("-rgc" in line for line in nvidia_lines))
        self.assertIn(
            "Clock mode unlocked: runner will not lock or reset the GPU clock.",
            completed.stdout,
        )

        benchmark_commands = [
            shlex.split(line)
            for line in python_lines
            if any(
                token.endswith("benchmark_efficiency_v2.py")
                for token in shlex.split(line)
            )
        ]
        self.assertEqual(len(benchmark_commands), 12)
        observed_orders: dict[int, list[str]] = {}
        positions = {
            method: Counter() for method in benchmark.METHODS
        }
        for command in benchmark_commands:
            method = command[command.index("--method") + 1]
            repetition = int(
                command[command.index("--repetition") + 1]
            )
            observed_orders.setdefault(repetition, []).append(method)
            position = len(observed_orders[repetition]) - 1
            positions[method][position] += 1
            self.assertEqual(
                command[command.index("--num-samples") + 1],
                "500",
            )
            self.assertEqual(
                command[command.index("--warmup") + 1],
                "20",
            )
            self.assertEqual(
                command[command.index("--max-new-tokens") + 1],
                "32",
            )
            self.assertEqual(
                command[command.index("--clock-mode") + 1],
                benchmark.CLOCK_MODE_MONITORED_UNLOCKED,
            )
            self.assertNotIn("--locked-sm-clock-mhz", command)
            if method == "ours":
                for flag in (
                    "--ours-implicit-causal",
                    "--ours-triton-rms",
                    "--ours-cuda-graph-prefill",
                    "--ours-cuda-graph-decode",
                    "--ours-static-kv-decode",
                ):
                    self.assertIn(flag, command)

        expected_orders = {
            0: ["vanilla", "learnpruner", "ours"],
            1: ["vanilla", "ours", "learnpruner"],
            2: ["learnpruner", "ours", "vanilla"],
            3: ["ours", "vanilla", "learnpruner"],
        }
        self.assertEqual(observed_orders, expected_orders)
        for method, counts in positions.items():
            self.assertEqual(sum(counts.values()), 4, method)
            exposures = [counts[position] for position in range(3)]
            self.assertLessEqual(max(exposures) - min(exposures), 1)
        for other in ("vanilla", "learnpruner"):
            ours_before = sum(
                order.index("ours") < order.index(other)
                for order in observed_orders.values()
            )
            self.assertEqual(ours_before, 2)

    def test_formal_benchmark_cli_accepts_only_admitted_settings(self) -> None:
        for method in ("vanilla", "learnpruner", "ours"):
            with self.subTest(method=method):
                args = formal_args(method)
                benchmark.validate_args(args)
                self.assertEqual(args.max_new_tokens, 32)
                self.assertEqual(args.warmup, 20)
                self.assertEqual(args.seed, 42)
                self.assertEqual(
                    args.clock_mode,
                    benchmark.CLOCK_MODE_MONITORED_UNLOCKED,
                )

        invalid_cases = (
            (["--max-new-tokens", "31"], "max-new-tokens 32"),
            (["--warmup", "19"], "warmup 20"),
            (["--seed", "7"], "seed 42"),
            (["--dtype", "float16"], "bfloat16"),
            (["--attn-implementation", "eager"], "SDPA"),
            (["--physical-gpu-id", "1"], "physical GPU"),
            (
                ["--locked-sm-clock-mhz", "1980"],
                "monitored_unlocked",
            ),
            (["--ours-budget", "128"], "pruning configuration"),
        )
        for extra, message in invalid_cases:
            with self.subTest(extra=extra):
                args = benchmark.parse_args(
                    ["--method", "vanilla", *extra]
                )
                with self.assertRaisesRegex(ValueError, message):
                    benchmark.validate_args(args)

        non_ours_graph = benchmark.parse_args(
            [
                "--method",
                "learnpruner",
                "--ours-cuda-graph-decode",
                "on",
            ]
        )
        with self.assertRaisesRegex(ValueError, "only for the Ours arm"):
            benchmark.validate_args(non_ours_graph)

    def test_formal_summarizer_cli_is_fixed_500_by_4_by_32(self) -> None:
        valid = summary.parse_args(["--run-id", "formal_cpu_contract"])
        summary.validate_cli(valid)
        invalid_cases = (
            (["--expected-n", "499"], "500x4x32"),
            (["--expected-repetitions", "3"], "500x4x32"),
            (["--expected-output-tokens", "31"], "500x4x32"),
            (["--physical-gpu-id", "1"], "500x4x32"),
        )
        for extra, message in invalid_cases:
            with self.subTest(extra=extra):
                args = summary.parse_args(
                    ["--run-id", "formal_cpu_contract", *extra]
                )
                with self.assertRaisesRegex(summary.SummaryError, message):
                    summary.validate_cli(args)

    def test_fixed_generation_kwargs_force_exactly_32_tokens(self) -> None:
        class Tokenizer:
            pad_token_id = None
            eos_token_id = 2

        class Adapter:
            tokenizer = Tokenizer()

        kwargs = benchmark.generation_kwargs(Adapter(), 32)
        self.assertEqual(kwargs["min_new_tokens"], 32)
        self.assertEqual(kwargs["max_new_tokens"], 32)
        self.assertFalse(kwargs["do_sample"])
        self.assertEqual(kwargs["num_beams"], 1)
        self.assertTrue(kwargs["use_cache"])

    def test_pinned_audit_and_current_runtime_are_strictly_bound(self) -> None:
        args = formal_args("ours")
        artifact_path = benchmark.OURS_COMBINED_AUDIT_PATH.resolve()
        self.assertTrue(artifact_path.is_file())
        self.assertEqual(
            benchmark.file_sha256(artifact_path),
            benchmark.OURS_COMBINED_AUDIT_SHA256,
        )
        artifact = json.loads(
            artifact_path.read_text(encoding="utf-8")
        )
        self.assertEqual(
            artifact["schema_version"],
            benchmark.OURS_COMBINED_AUDIT_SCHEMA,
        )
        self.assertEqual(
            artifact["run_id"],
            benchmark.OURS_COMBINED_AUDIT_RUN_ID,
        )
        self.assertEqual(
            artifact["checks"],
            benchmark.OURS_COMBINED_AUDIT_CHECKS,
        )
        self.assertEqual(
            artifact["digests"]["script_sha256"],
            benchmark.file_sha256(
                benchmark.OURS_COMBINED_AUDIT_SCRIPT
            ),
        )

        sample_path = args.sample_parquet.resolve()
        provenance = benchmark.combined_audit_provenance(
            args,
            sample_path,
            benchmark.file_sha256(sample_path),
        )
        self.assertIsNotNone(provenance)
        self.assertTrue(provenance["passed"])
        self.assertEqual(
            provenance["artifact_sha256"],
            benchmark.OURS_COMBINED_AUDIT_SHA256,
        )

        source_hashes = benchmark.selected_source_hashes(args)
        runtime_path = HERE / "efficiency_v2_runtime.py"
        self.assertEqual(
            source_hashes["benchmark_adapter"],
            runtime.file_sha256(runtime_path),
        )
        self.assertEqual(
            benchmark.ModelAdapter,
            runtime.ModelAdapter,
        )
        self.assertEqual(
            benchmark.PreparedInput,
            runtime.PreparedInput,
        )

    def test_formal_arm_and_generation_labels(self) -> None:
        for method in ("vanilla", "learnpruner"):
            args = formal_args(method)
            self.assertEqual(benchmark.arm_id(args), method)
            self.assertEqual(
                benchmark.generation_variant(args),
                "generation_mixin",
            )
        ours = formal_args("ours")
        self.assertEqual(benchmark.arm_id(ours), "ours")
        self.assertEqual(
            benchmark.generation_variant(ours),
            "fast_triton_prefill_rms_cuda_graph_prefill_cuda_graph_decode",
        )


class SummaryValidationTests(unittest.TestCase):
    def test_three_method_layer_traces(self) -> None:
        prompt_tokens = 49
        vanilla = summary.expected_layer_trace(
            "vanilla",
            prompt_tokens,
            prompt_tokens + 576,
            {},
        )
        self.assertEqual(vanilla, [625] * 32)

        learnpruner_stats = {
            "implementation_mode": "paper_aligned",
            "stage1_visual_tokens": 111,
            "stage2_visual_tokens": 37,
            "stage2_paper_layer": 12,
            "stage2_layer_idx": 11,
            "final_sequence_tokens": 86,
        }
        learnpruner = summary.expected_layer_trace(
            "learnpruner",
            prompt_tokens,
            prompt_tokens + 111,
            learnpruner_stats,
        )
        self.assertEqual(
            learnpruner,
            [160] * 12 + [86] * 20,
        )

        ours_stats = {
            "predictor_only": False,
            "avg_token_budget_profile": 64,
            "kept_visual_tokens": 137,
            "mid_pruning_layer_idx": 12,
            "mid_target_visual_tokens": 32,
            "mid_sequence_tokens": 81,
            "final_wipe_layer_idx": 25,
            "final_sequence_tokens": 49,
        }
        ours = summary.expected_layer_trace(
            "ours",
            prompt_tokens,
            prompt_tokens + 576,
            ours_stats,
        )
        self.assertEqual(
            ours,
            [186] * 12 + [81] * 13 + [49] * 7,
        )
        with self.assertRaises(summary.SummaryError):
            summary.expected_layer_trace(
                "learnpruner",
                prompt_tokens,
                prompt_tokens + 111,
                {**learnpruner_stats, "stage2_layer_idx": 12},
            )
        with self.assertRaises(summary.SummaryError):
            summary.expected_layer_trace(
                "ours",
                prompt_tokens,
                prompt_tokens + 576,
                {**ours_stats, "kept_visual_tokens": 136},
            )

    def test_ours_graph_fields_and_non_ours_empty_fields(self) -> None:
        generated_hash = "a" * 64
        ours_row = {
            "cuda_graph_residency_warmup_tail_sha256": generated_hash,
            "cuda_graph_prefill_callback_runner_id": "7",
            "cuda_graph_prefill_pair_status": (
                "stable_runner_two_cache_hit_replays_no_fallback"
            ),
            "cuda_graph_decode_callback_steps": json.dumps(
                list(range(1, 32))
            ),
            "cuda_graph_runner_cache_status": "single_runner_reused",
            "triton_rms_runtime_status": (
                "prefill_only_enabled_no_runtime_fallback"
            ),
        }
        summary.validate_graph_fields(
            "ours",
            ours_row,
            generated_hash,
            "ours-row",
        )
        broken = {
            **ours_row,
            "cuda_graph_decode_callback_steps": json.dumps(
                list(range(1, 31))
            ),
        }
        with self.assertRaisesRegex(
            summary.SummaryError,
            "graph/runtime protocol",
        ):
            summary.validate_graph_fields(
                "ours",
                broken,
                generated_hash,
                "ours-row",
            )

        disabled = {
            "cuda_graph_residency_warmup_tail_sha256": "",
            "cuda_graph_prefill_callback_runner_id": "",
            "cuda_graph_prefill_pair_status": "disabled",
            "cuda_graph_decode_callback_steps": "[]",
            "cuda_graph_runner_cache_status": "disabled",
            "triton_rms_runtime_status": "disabled",
        }
        for arm in ("vanilla", "learnpruner"):
            summary.validate_graph_fields(
                arm,
                disabled,
                generated_hash,
                f"{arm}-row",
            )

    def test_numeric_parsing_statistics_and_timing_arithmetic(self) -> None:
        for raw in ("nan", "inf", "-inf"):
            with self.subTest(raw=raw):
                with self.assertRaises(summary.SummaryError):
                    summary.finite_float(raw, label="metric")
        with self.assertRaises(summary.SummaryError):
            summary.finite_float("-1", label="metric")

        stats = summary.metric_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(stats.count, 5)
        self.assertEqual(stats.mean, 3.0)
        self.assertEqual(stats.median, 3.0)
        self.assertAlmostEqual(stats.p10, 1.4)
        self.assertAlmostEqual(stats.p90, 4.6)
        self.assertEqual(stats.minimum, 1.0)
        self.assertEqual(stats.maximum, 5.0)

        row = {
            "decode_step_cuda_ms": json.dumps([2.0] * 31),
            "decode_step_wall_ms": json.dumps([3.0] * 31),
            "decode_forward_cuda_ms": json.dumps([1.5] * 31),
        }
        parsed = {
            "decode_cuda_ms": 62.0,
            "tpot_cuda_ms": 2.0,
            "decode_wall_ms": 93.0,
            "tpot_wall_ms": 3.0,
            "ttft_cuda_ms": 10.0,
            "e2e_cuda_ms": 72.0,
            "ttft_wall_ms": 20.0,
            "e2e_wall_ms": 113.0,
        }
        summary.validate_timing_arithmetic(row, parsed, "row")
        with self.assertRaisesRegex(
            summary.SummaryError,
            "CUDA TTFT\\+decode",
        ):
            summary.validate_timing_arithmetic(
                row,
                {**parsed, "e2e_cuda_ms": 73.0},
                "row",
            )

    def test_speedups_preserve_cross_backend_labels_and_direction(self) -> None:
        keys = ((0, 10), (0, 20))
        latency = {
            "vanilla": (100.0, 120.0),
            "learnpruner": (80.0, 96.0),
            "ours": (50.0, 60.0),
        }
        throughput = {
            "vanilla": (10.0, 12.0),
            "learnpruner": (12.5, 15.0),
            "ours": (20.0, 24.0),
        }
        methods = {
            arm: summary.MethodData() for arm in summary.ARMS
        }
        stats: dict[str, dict[str, summary.MetricStats]] = {
            arm: {} for arm in summary.ARMS
        }
        for arm, method in methods.items():
            method.rows = {key: {} for key in keys}
            for metric in summary.PRIMARY_SPEEDUP_METRICS:
                values = (
                    throughput[arm]
                    if metric in summary.THROUGHPUT_METRICS
                    else latency[arm]
                )
                method.values[metric] = dict(zip(keys, values))
                stats[arm][metric] = summary.metric_stats(values)
        loaded = summary.LoadedResults(
            methods=methods,
            input_provenance={},
            sample_sha256="sample",
            gpu_uuid="gpu",
        )
        speedups = summary.build_speedups(loaded, stats)
        self.assertEqual(
            len(speedups),
            len(summary.COMPARISONS)
            * len(summary.PRIMARY_SPEEDUP_METRICS),
        )
        labels = {
            (item.baseline, item.current): item.comparison_type
            for item in speedups
        }
        self.assertEqual(
            labels[("vanilla", "ours")],
            "matched_official_backend",
        )
        self.assertEqual(
            labels[("vanilla", "learnpruner")],
            "cross_backend_raw_system",
        )
        self.assertEqual(
            labels[("learnpruner", "ours")],
            "cross_backend_raw_system",
        )
        self.assertEqual(
            summary.BACKENDS,
            {
                "vanilla": "official_llava",
                "learnpruner": "huggingface_llava",
                "ours": "official_llava",
            },
        )
        for item in speedups:
            expected_direction = (
                "current_over_baseline"
                if item.metric in summary.THROUGHPUT_METRICS
                else "baseline_over_current"
            )
            self.assertEqual(item.direction, expected_direction)
            self.assertTrue(math.isfinite(item.mean_speedup))
            self.assertGreater(item.mean_speedup, 0.0)


if __name__ == "__main__":
    unittest.main()
