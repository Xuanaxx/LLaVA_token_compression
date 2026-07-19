#!/usr/bin/env python3
"""Correlate layer-level pruning divergence with pruned RefCOCO performance."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from jsd_kl_result_relation.metrics import (
    DIVERGENCE_METRICS,
    correlations,
    standardize_within_groups,
)


LAYER_RE = re.compile(r"scoring_layer_(?P<layer>\d+)$")


def load_layer_rows(
    base_root: Path, expected_samples: int | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for path in sorted(base_root.rglob("layer_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        config = data.get("config", {})
        layer_dir = path.parent
        match = LAYER_RE.fullmatch(layer_dir.name)
        if match is None:
            continue
        model = str(config.get("model") or layer_dir.parent.parent.name)
        task = str(config.get("task") or layer_dir.parent.name)
        layer = int(config.get("scoring_layer", match.group("layer")))
        key = (model, task, layer)
        if key in seen:
            raise ValueError(f"Duplicate summary for {key}: {path}")
        seen.add(key)
        n_records = data.get("num_records")
        if expected_samples is not None and n_records != expected_samples:
            raise ValueError(
                f"Expected {expected_samples} records for {key}, found {n_records!r} in {path}"
            )
        divergence = data.get("mean_divergence", {})
        performance = data.get("performance", {})
        missing = [
            metric
            for metric in DIVERGENCE_METRICS
            if not isinstance(divergence.get(metric), (int, float))
        ]
        if missing:
            raise ValueError(f"Missing divergence metrics {missing} in {path}")
        if not performance or not all(
            isinstance(value, (int, float)) for value in performance.values()
        ):
            raise ValueError(f"Missing numeric performance metrics in {path}")
        rows.append(
            {
                "model": model,
                "task": task,
                "layer": layer,
                "num_records": n_records,
                "sample_indices_sha256": data.get("sample_indices_sha256"),
                "divergence": {
                    metric: float(divergence[metric]) for metric in DIVERGENCE_METRICS
                },
                "performance": {
                    metric: float(value) for metric, value in performance.items()
                },
                "summary_path": str(path),
            }
        )
    if not rows:
        raise ValueError(f"No layer_summary.json files found under {base_root}")
    return rows


def _validate_sample_alignment(rows: Sequence[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[(row["model"], row["task"])].add(str(row.get("sample_indices_sha256")))
    mismatched = {key: values for key, values in grouped.items() if len(values) != 1}
    if mismatched:
        raise ValueError(
            f"Scoring layers used different sampled examples: {mismatched}"
        )


def _available_performance_metrics(rows: Sequence[dict[str, Any]]) -> list[str]:
    common: set[str] | None = None
    for row in rows:
        metrics = set(row["performance"])
        common = metrics if common is None else common & metrics
    if not common:
        raise ValueError("Layer summaries have no common performance metric")
    return sorted(common)


def _best_row(rows: Sequence[dict[str, Any]], key, maximize: bool) -> dict[str, Any]:
    ordered = sorted(
        rows, key=lambda row: ((-1 if maximize else 1) * key(row), row["layer"])
    )
    return ordered[0]


def analyze_correlations(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    _validate_sample_alignment(rows)
    by_model_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model_task[(row["model"], row["task"])].append(row)

    models: dict[str, Any] = defaultdict(lambda: {"tasks": {}})
    for (model, task), task_rows in sorted(by_model_task.items()):
        task_rows = sorted(task_rows, key=lambda row: row["layer"])
        performance_metrics = _available_performance_metrics(task_rows)
        metric_results: dict[str, Any] = {}
        for performance_metric in performance_metrics:
            y = [row["performance"][performance_metric] for row in task_rows]
            metric_results[performance_metric] = {
                divergence_metric: correlations(
                    [row["divergence"][divergence_metric] for row in task_rows], y
                )
                for divergence_metric in DIVERGENCE_METRICS
            }
        primary_metric = (
            "CIDEr" if "CIDEr" in performance_metrics else performance_metrics[0]
        )
        best_performance = _best_row(
            task_rows, lambda row: row["performance"][primary_metric], maximize=True
        )
        minimum_divergence = {
            metric: {
                "layer": _best_row(
                    task_rows, lambda row, name=metric: row["divergence"][name], False
                )["layer"],
                "value": _best_row(
                    task_rows, lambda row, name=metric: row["divergence"][name], False
                )["divergence"][metric],
            }
            for metric in DIVERGENCE_METRICS
        }
        models[model]["tasks"][task] = {
            "num_layers": len(task_rows),
            "layers": [row["layer"] for row in task_rows],
            "primary_performance_metric": primary_metric,
            "best_pruned_performance": {
                "layer": best_performance["layer"],
                "value": best_performance["performance"][primary_metric],
            },
            "minimum_divergence": minimum_divergence,
            "correlations_divergence_vs_pruned_performance": metric_results,
        }

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    for model, model_rows in sorted(by_model.items()):
        tasks = sorted({row["task"] for row in model_rows})
        common_metrics = _available_performance_metrics(model_rows)
        pooled: dict[str, Any] = {}
        groups = [row["task"] for row in model_rows]
        for performance_metric in common_metrics:
            performance_z = standardize_within_groups(
                [row["performance"][performance_metric] for row in model_rows], groups
            )
            pooled[performance_metric] = {}
            for divergence_metric in DIVERGENCE_METRICS:
                divergence_z = standardize_within_groups(
                    [row["divergence"][divergence_metric] for row in model_rows], groups
                )
                pooled[performance_metric][divergence_metric] = correlations(
                    divergence_z, performance_z
                )
        models[model]["pooled_task_standardized_correlations"] = pooled
        models[model]["pooled_tasks"] = tasks

    return {
        "method": {
            "unit": "scoring_layer",
            "x": "mean full-vs-pruned vocabulary divergence over the shared pruned decoding trajectory",
            "y": "caption metric of the top-k-pruned model",
            "correlations": ["Pearson", "Spearman"],
            "pooled_analysis": "both x and y are z-scored within each task before concatenation",
            "interpretation": "a negative coefficient means lower divergence is associated with better pruned performance",
        },
        "num_layer_rows": len(rows),
        "models": dict(models),
    }


def write_rows_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    performance_metrics = sorted(
        {metric for row in rows for metric in row["performance"]}
    )
    fieldnames = [
        "model",
        "task",
        "layer",
        "num_records",
        *DIVERGENCE_METRICS,
        *(f"performance_{metric}" for metric in performance_metrics),
        "summary_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in sorted(
            rows, key=lambda item: (item["model"], item["task"], item["layer"])
        ):
            flat = {
                "model": row["model"],
                "task": row["task"],
                "layer": row["layer"],
                "num_records": row["num_records"],
                "summary_path": row["summary_path"],
                **row["divergence"],
                **{
                    f"performance_{metric}": value
                    for metric, value in row["performance"].items()
                },
            }
            writer.writerow(flat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int)
    args = parser.parse_args()

    rows = load_layer_rows(args.base_root, expected_samples=args.expected_samples)
    output = analyze_correlations(rows)
    output["base_root"] = str(args.base_root)
    output["rows"] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_rows_tsv(args.tsv, rows)
    print(f"Saved correlations to {args.output}")
    print(f"Saved layer table to {args.tsv}")


if __name__ == "__main__":
    main()
