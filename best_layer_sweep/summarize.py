#!/usr/bin/env python3
"""Aggregate lmms-eval layer-sweep results and report the best layer per model."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


LAYER_RE = re.compile(r"scoring_layer_(?P<layer>\d+)$")
RESULT_RE = re.compile(r"(?P<timestamp>\d{8}_\d{6})_results\.json$")
PREFERRED_METRICS = (
    "exact_match,none",
    "pope_accuracy,none",
    "accuracy,none",
    "acc,none",
    "relaxed_overall,none",
    "mme_perception_score,none",
    "mme_cognition_score,none",
    "exact_match",
    "pope_accuracy",
    "accuracy",
    "acc",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _latest_result(layer_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[str, float, Path, dict[str, Any]]] = []
    for path in layer_dir.rglob("*_results.json"):
        try:
            data = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        match = RESULT_RE.match(path.name)
        timestamp = str(data.get("date") or (match.group("timestamp") if match else ""))
        candidates.append((timestamp, path.stat().st_mtime, path, data))
    if not candidates:
        return None
    _, _, path, data = max(candidates, key=lambda item: (item[0], item[1], str(item[2])))
    return path, data


def _select_metric(task_result: dict[str, Any], requested: str | None) -> tuple[str, float]:
    if requested:
        value = task_result.get(requested)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Requested metric {requested!r} is absent or non-numeric")
        return requested, float(value)
    for metric in PREFERRED_METRICS:
        value = task_result.get(metric)
        if isinstance(value, (int, float)):
            return metric, float(value)
    for metric, value in sorted(task_result.items()):
        if metric != "alias" and "stderr" not in metric and isinstance(value, (int, float)):
            return metric, float(value)
    raise ValueError(f"No numeric performance metric in {sorted(task_result)}")


def _parse_metric_overrides(values: Iterable[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Metric override must be TASK=METRIC, got {value!r}")
        task, metric = value.split("=", 1)
        overrides[task.strip()] = metric.strip()
    return overrides


def summarize(
    base_root: Path,
    metric_overrides: dict[str, str] | None = None,
    expected_samples: int | None = None,
) -> dict[str, Any]:
    overrides = metric_overrides or {}
    rows: list[dict[str, Any]] = []
    for layer_dir in sorted(base_root.rglob("scoring_layer_*")):
        match = LAYER_RE.fullmatch(layer_dir.name)
        if not layer_dir.is_dir() or match is None:
            continue
        try:
            relative = layer_dir.relative_to(base_root)
            model_name, task = relative.parts[-3], relative.parts[-2]
        except (ValueError, IndexError):
            continue
        latest = _latest_result(layer_dir)
        if latest is None:
            continue
        result_path, data = latest
        task_result = data.get("results", {}).get(task)
        if not isinstance(task_result, dict):
            continue
        metric, score = _select_metric(task_result, overrides.get(task))
        sample_info = data.get("n-samples", {}).get(task, {})
        n_samples = sample_info.get("effective", sample_info.get("original")) if isinstance(sample_info, dict) else None
        if expected_samples is not None and n_samples != expected_samples:
            raise ValueError(
                f"Expected {expected_samples} evaluated samples for {model_name}/{task}/layer {match.group('layer')}, "
                f"found {n_samples!r} in {result_path}"
            )
        rows.append(
            {
                "model": model_name,
                "task": task,
                "layer": int(match.group("layer")),
                "metric": metric,
                "score": score,
                "n_samples": n_samples,
                "result_path": str(result_path),
            }
        )
    if not rows:
        raise ValueError(f"No usable layer results found under {base_root}")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["task"])].append(row)

    task_summaries: dict[str, dict[str, Any]] = defaultdict(dict)
    normalized: dict[tuple[str, int], list[float]] = defaultdict(list)
    model_tasks: dict[str, set[str]] = defaultdict(set)
    for (model, task), task_rows in sorted(grouped.items()):
        ranked = sorted(task_rows, key=lambda row: (-row["score"], row["layer"]))
        best_score = ranked[0]["score"]
        denominator = abs(best_score) if abs(best_score) > 1e-12 else 1.0
        for row in task_rows:
            row["relative_to_task_best"] = float(row["score"] / denominator)
            normalized[(model, row["layer"])].append(row["relative_to_task_best"])
        model_tasks[model].add(task)
        task_summaries[model][task] = {
            "best_layer": ranked[0]["layer"],
            "best_score": best_score,
            "metric": ranked[0]["metric"],
            "num_layers": len(ranked),
        }

    models: dict[str, Any] = {}
    for model, tasks in sorted(model_tasks.items()):
        candidates = []
        for (candidate_model, layer), values in normalized.items():
            if candidate_model == model and len(values) == len(tasks):
                candidates.append({"layer": layer, "mean_relative_score": sum(values) / len(values)})
        if not candidates:
            raise ValueError(f"No layer has complete task coverage for model {model}")
        ranking = sorted(candidates, key=lambda row: (-row["mean_relative_score"], row["layer"]))
        models[model] = {
            "best_layer": ranking[0]["layer"],
            "best_mean_relative_score": ranking[0]["mean_relative_score"],
            "tasks": task_summaries[model],
            "ranking": ranking,
        }

    output: dict[str, Any] = {"base_root": str(base_root), "models": models, "rows": rows}
    if len(models) == 1:
        only_model, only_summary = next(iter(models.items()))
        output.update({"model": only_model, "best_layer": only_summary["best_layer"]})
    return output


def _write_tsv(path: Path, output: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(output["rows"], key=lambda row: (row["model"], row["task"], row["layer"]))
    with path.open("w", encoding="utf-8") as handle:
        handle.write("model\ttask\tlayer\tmetric\tscore\trelative_to_task_best\tn_samples\tresult_path\n")
        for row in rows:
            handle.write(
                f"{row['model']}\t{row['task']}\t{row['layer']}\t{row['metric']}\t{row['score']}\t"
                f"{row['relative_to_task_best']}\t{row.get('n_samples', '')}\t{row['result_path']}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tsv", type=Path)
    parser.add_argument("--metric", action="append", default=[], metavar="TASK=METRIC")
    parser.add_argument("--expected-samples", type=int)
    args = parser.parse_args()
    output = summarize(args.base_root, _parse_metric_overrides(args.metric), args.expected_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.tsv:
        _write_tsv(args.tsv, output)
    for model, model_summary in output["models"].items():
        print(
            f"BEST_LAYER model={model} layer={model_summary['best_layer']} "
            f"mean_relative_score={model_summary['best_mean_relative_score']:.8f}"
        )


if __name__ == "__main__":
    main()
