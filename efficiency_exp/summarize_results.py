#!/usr/bin/env python3
"""Validate and summarize the three DetailCaps efficiency benchmark runs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


METHODS = ("vanilla", "learnpruner", "ours")
METRICS = (
    "avg_visual_tokens",
    "estimated_prefill_tflops",
    "prefill_cuda_ms",
    "total_cuda_ms",
    "decode_cuda_ms_per_token",
    "kv_cache_mb",
    "peak_allocated_gb",
    "total_wall_ms",
)
LATENCY_METRICS = (
    "prefill_cuda_ms",
    "total_cuda_ms",
    "decode_cuda_ms_per_token",
    "total_wall_ms",
)
DISPLAY_NAMES = {
    "vanilla": "Vanilla",
    "learnpruner": "LearnPruner",
    "ours": "Ours",
}
METRIC_LABELS = {
    "avg_visual_tokens": "Visual tokens",
    "estimated_prefill_tflops": "Est. prefill TFLOPs",
    "prefill_cuda_ms": "Prefill CUDA (ms)",
    "total_cuda_ms": "Total CUDA (ms)",
    "decode_cuda_ms_per_token": "Decode CUDA (ms/token)",
    "kv_cache_mb": "KV cache (MB)",
    "peak_allocated_gb": "Peak allocated (GB)",
    "total_wall_ms": "Wall time (ms)",
}


class SummaryError(RuntimeError):
    """An input or output error that should be shown without a traceback."""


@dataclass(frozen=True)
class MetricStats:
    mean: float
    std: float
    p50: float
    p95: float


@dataclass(frozen=True)
class MethodResults:
    indices: frozenset[str]
    values: dict[str, list[float]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory containing the per-method TSV files (default: %(default)s).",
    )
    parser.add_argument("--n", type=int, default=500, help="Expected rows per method.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed embedded in input filenames."
    )
    return parser.parse_args(argv)


def input_path(input_dir: Path, method: str, seed: int, n: int) -> Path:
    return input_dir / f"{method}_seed{seed}_n{n}.tsv"


def read_method(path: Path, expected_n: int) -> MethodResults:
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise SummaryError(f"cannot open input file {path}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise SummaryError(f"input file has no TSV header: {path}")
        required = {"dataset_index", *METRICS}
        missing_columns = sorted(required.difference(reader.fieldnames))
        if missing_columns:
            raise SummaryError(
                f"input file {path} is missing required column(s): "
                + ", ".join(missing_columns)
            )

        indices: list[str] = []
        values = {metric: [] for metric in METRICS}
        for line_number, row in enumerate(reader, start=2):
            dataset_index = (row.get("dataset_index") or "").strip()
            if not dataset_index:
                raise SummaryError(f"empty dataset_index in {path} at line {line_number}")
            indices.append(dataset_index)

            for metric in METRICS:
                raw_value = (row.get(metric) or "").strip()
                try:
                    value = float(raw_value)
                except ValueError as exc:
                    raise SummaryError(
                        f"invalid {metric}={raw_value!r} in {path} at line {line_number}"
                    ) from exc
                if not math.isfinite(value):
                    raise SummaryError(
                        f"non-finite {metric}={raw_value!r} in {path} at line {line_number}"
                    )
                values[metric].append(value)

    if len(indices) != expected_n:
        raise SummaryError(
            f"row-count mismatch for {path}: found {len(indices)}, expected {expected_n}"
        )
    unique_indices = frozenset(indices)
    if len(unique_indices) != len(indices):
        seen: set[str] = set()
        duplicates: list[str] = []
        for index in indices:
            if index in seen and index not in duplicates:
                duplicates.append(index)
            seen.add(index)
        preview = ", ".join(repr(index) for index in duplicates[:5])
        suffix = " ..." if len(duplicates) > 5 else ""
        raise SummaryError(f"duplicate dataset_index in {path}: {preview}{suffix}")
    return MethodResults(indices=unique_indices, values=values)


def validate_index_sets(results: dict[str, MethodResults]) -> None:
    reference_method = METHODS[0]
    reference = results[reference_method].indices
    for method in METHODS[1:]:
        current = results[method].indices
        if current == reference:
            continue
        missing = sorted(reference - current)[:5]
        extra = sorted(current - reference)[:5]
        raise SummaryError(
            "dataset_index set mismatch: "
            f"{method} differs from {reference_method}; "
            f"missing={missing!r}, extra={extra!r}"
        )


def percentile(values: Sequence[float], probability: float) -> float:
    """Return the linearly interpolated percentile used by NumPy/Pandas defaults."""

    ordered = sorted(values)
    if not ordered:
        raise SummaryError("cannot summarize an empty metric")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def summarize(values: Sequence[float]) -> MetricStats:
    if not values:
        raise SummaryError("cannot summarize an empty metric")
    # Sample standard deviation (ddof=1), matching pandas.DataFrame.std().
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return MetricStats(
        mean=statistics.fmean(values),
        std=std,
        p50=percentile(values, 0.50),
        p95=percentile(values, 0.95),
    )


def make_summary(
    results: dict[str, MethodResults],
) -> dict[str, dict[str, MetricStats]]:
    return {
        method: {
            metric: summarize(results[method].values[metric]) for metric in METRICS
        }
        for method in METHODS
    }


def ratio(numerator: float, denominator: float, description: str) -> float:
    if denominator <= 0.0:
        raise SummaryError(
            f"cannot compute {description}: denominator must be positive, got {denominator}"
        )
    return numerator / denominator


def speedups(
    summary: dict[str, dict[str, MetricStats]],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    versus_vanilla: dict[str, dict[str, float]] = {}
    for method in METHODS:
        versus_vanilla[method] = {
            metric: ratio(
                summary["vanilla"][metric].mean,
                summary[method][metric].mean,
                f"Vanilla/{DISPLAY_NAMES[method]} speedup for {metric}",
            )
            for metric in LATENCY_METRICS
        }
    learnpruner_over_ours = {
        metric: ratio(
            summary["learnpruner"][metric].mean,
            summary["ours"][metric].mean,
            f"LearnPruner/Ours speedup for {metric}",
        )
        for metric in LATENCY_METRICS
    }
    return versus_vanilla, learnpruner_over_ours


def format_number(value: float) -> str:
    return f"{value:.10g}"


def write_summary_tsv(
    path: Path,
    n: int,
    seed: int,
    summary: dict[str, dict[str, MetricStats]],
    versus_vanilla: dict[str, dict[str, float]],
    learnpruner_over_ours: dict[str, float],
) -> None:
    fieldnames = (
        "method",
        "metric",
        "n",
        "seed",
        "mean",
        "std",
        "p50",
        "p95",
        "vanilla_over_current",
        "learnpruner_over_ours",
    )
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for method in METHODS:
                for metric in METRICS:
                    stats = summary[method][metric]
                    is_latency = metric in LATENCY_METRICS
                    writer.writerow(
                        {
                            "method": method,
                            "metric": metric,
                            "n": n,
                            "seed": seed,
                            "mean": format_number(stats.mean),
                            "std": format_number(stats.std),
                            "p50": format_number(stats.p50),
                            "p95": format_number(stats.p95),
                            "vanilla_over_current": (
                                format_number(versus_vanilla[method][metric])
                                if is_latency
                                else ""
                            ),
                            "learnpruner_over_ours": (
                                format_number(learnpruner_over_ours[metric])
                                if is_latency and method == "ours"
                                else ""
                            ),
                        }
                    )
    except OSError as exc:
        raise SummaryError(f"cannot write summary TSV {path}: {exc}") from exc


def markdown_stat(stats: MetricStats) -> str:
    return (
        f"{stats.mean:.3f} ± {stats.std:.3f} "
        f"[{stats.p50:.3f}, {stats.p95:.3f}]"
    )


def write_summary_markdown(
    path: Path,
    n: int,
    seed: int,
    summary: dict[str, dict[str, MetricStats]],
    versus_vanilla: dict[str, dict[str, float]],
    learnpruner_over_ours: dict[str, float],
    gate_passed: bool,
) -> None:
    lines = [
        "# Table 5 — Efficiency",
        "",
        f"DetailCaps sample: n={n}, seed={seed}. Values are mean ± sample std "
        "[p50, p95].",
        "",
        "| Method | " + " | ".join(METRIC_LABELS[metric] for metric in METRICS) + " |",
        "|:--|" + "--:|" * len(METRICS),
    ]
    for method in METHODS:
        cells = [markdown_stat(summary[method][metric]) for metric in METRICS]
        lines.append(f"| {DISPLAY_NAMES[method]} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Latency speedups",
            "",
            "Each value is baseline mean / current mean; values above 1× are faster.",
            "",
            "| Comparison | "
            + " | ".join(METRIC_LABELS[metric] for metric in LATENCY_METRICS)
            + " |",
            "|:--|" + "--:|" * len(LATENCY_METRICS),
        ]
    )
    for method in METHODS:
        cells = [
            f"{versus_vanilla[method][metric]:.3f}×"
            for metric in LATENCY_METRICS
        ]
        lines.append(
            f"| Vanilla / {DISPLAY_NAMES[method]} | " + " | ".join(cells) + " |"
        )
    cells = [f"{learnpruner_over_ours[metric]:.3f}×" for metric in LATENCY_METRICS]
    lines.append("| LearnPruner / Ours | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Strict latency gate",
            "",
            (
                "**PASS** — Ours has lower mean prefill and total CUDA latency than "
                "LearnPruner."
                if gate_passed
                else "**FAIL** — Ours must have strictly lower mean prefill and total "
                "CUDA latency than LearnPruner."
            ),
            "",
        ]
    )
    try:
        path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        raise SummaryError(f"cannot write Markdown summary {path}: {exc}") from exc


def gate_passes(summary: dict[str, dict[str, MetricStats]]) -> bool:
    return all(
        summary["ours"][metric].mean < summary["learnpruner"][metric].mean
        for metric in ("prefill_cuda_ms", "total_cuda_ms")
    )


def gate_failure_message(summary: dict[str, dict[str, MetricStats]]) -> str:
    comparisons = []
    for metric in ("prefill_cuda_ms", "total_cuda_ms"):
        comparisons.append(
            f"{metric}: ours={summary['ours'][metric].mean:.6g} ms, "
            f"learnpruner={summary['learnpruner'][metric].mean:.6g} ms"
        )
    return "strict latency gate failed (requires ours < learnpruner): " + "; ".join(
        comparisons
    )


def run(args: argparse.Namespace) -> int:
    if args.n <= 0:
        raise SummaryError(f"--n must be positive, got {args.n}")
    input_dir = args.input_dir.expanduser().resolve()
    paths = {
        method: input_path(input_dir, method, args.seed, args.n) for method in METHODS
    }
    missing = [str(paths[method]) for method in METHODS if not paths[method].is_file()]
    if missing:
        raise SummaryError("missing required input file(s): " + ", ".join(missing))

    results = {method: read_method(paths[method], args.n) for method in METHODS}
    validate_index_sets(results)
    summary = make_summary(results)
    versus_vanilla, learnpruner_over_ours = speedups(summary)
    passed = gate_passes(summary)

    summary_tsv = input_dir / "summary.tsv"
    summary_md = input_dir / "summary.md"
    write_summary_tsv(
        summary_tsv,
        args.n,
        args.seed,
        summary,
        versus_vanilla,
        learnpruner_over_ours,
    )
    write_summary_markdown(
        summary_md,
        args.n,
        args.seed,
        summary,
        versus_vanilla,
        learnpruner_over_ours,
        passed,
    )
    print(f"Wrote {summary_tsv}")
    print(f"Wrote {summary_md}")

    if not passed:
        print(f"error: {gate_failure_message(summary)}", file=sys.stderr)
        return 2
    print("Strict latency gate: PASS (ours < learnpruner for prefill and total CUDA means)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except SummaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
