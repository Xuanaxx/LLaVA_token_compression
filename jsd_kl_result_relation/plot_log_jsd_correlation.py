#!/usr/bin/env python3
"""Plot layer-level RefCOCO performance against log-transformed JSD."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, t

from jsd_kl_result_relation.evaluate_layer import (
    DEFAULT_CAPTION_METRICS,
    _caption_performance_many,
)
from jsd_kl_result_relation.metrics import correlations, mean_finite


LAYER_RE = re.compile(r"scoring_layer_(?P<layer>\d+)$")
SHARD_RE = re.compile(r"records\.shard_(?P<shard>\d+)_of_(?P<count>\d+)\.jsonl$")
JSD_METRICS = ("first_token_jsd", "generation_mean_jsd")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    return records


def _ordered_shards(layer_dir: Path) -> list[tuple[int, int, Path]]:
    shards: list[tuple[int, int, Path]] = []
    for path in layer_dir.glob("records.shard_*_of_*.jsonl"):
        match = SHARD_RE.fullmatch(path.name)
        if match is not None:
            shards.append((int(match.group("shard")), int(match.group("count")), path))
    shards.sort()
    if not shards:
        raise FileNotFoundError(f"No sample-sharded records found in {layer_dir}")
    counts = {count for _, count, _ in shards}
    if len(counts) != 1:
        raise ValueError(f"Inconsistent shard counts in {layer_dir}: {sorted(counts)}")
    shard_count = counts.pop()
    if [shard for shard, _, _ in shards] != list(range(1, shard_count + 1)):
        raise ValueError(f"Incomplete shard set in {layer_dir}")
    return shards


def load_sharded_experiment(
    input_root: Path, expected_samples: int
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    """Load complete layer shards without creating canonical output files."""

    layer_dirs: list[tuple[int, Path]] = []
    for path in input_root.iterdir():
        match = LAYER_RE.fullmatch(path.name) if path.is_dir() else None
        if match is not None:
            layer_dirs.append((int(match.group("layer")), path))
    layer_dirs.sort()
    if not layer_dirs:
        raise ValueError(f"No scoring_layer_* directories found under {input_root}")

    records_by_layer: dict[int, list[dict[str, Any]]] = {}
    shared_indices: list[int] | None = None
    shared_config: dict[str, Any] | None = None
    for layer, layer_dir in layer_dirs:
        records_by_index: dict[int, dict[str, Any]] = {}
        layer_config: dict[str, Any] | None = None
        for shard, count, records_path in _ordered_shards(layer_dir):
            manifest_path = layer_dir / f"sample_indices.shard_{shard}_of_{count}.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing shard manifest: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = manifest.get("config")
            if not isinstance(config, dict):
                raise ValueError(f"Missing configuration in {manifest_path}")
            if layer_config is None:
                layer_config = config
            elif config != layer_config:
                raise ValueError(f"Shard configuration mismatch in {layer_dir}")

            manifest_indices = [int(index) for index in manifest.get("indices", [])]
            shard_records = _read_jsonl(records_path)
            shard_by_index = {
                int(record["dataset_index"]): record for record in shard_records
            }
            if len(shard_by_index) != len(shard_records):
                raise ValueError(f"Duplicate dataset indices in {records_path}")
            if set(shard_by_index) != set(manifest_indices):
                raise ValueError(f"Records do not match manifest in {records_path}")
            overlap = set(records_by_index) & set(shard_by_index)
            if overlap:
                raise ValueError(
                    f"Layer {layer} has samples repeated across shards: {sorted(overlap)[:5]}"
                )
            records_by_index.update(shard_by_index)

        if layer_config is None:
            raise RuntimeError(f"No configuration found for layer {layer}")
        if int(layer_config.get("scoring_layer", -1)) != layer:
            raise ValueError(f"Configuration layer mismatch in {layer_dir}")
        indices = sorted(records_by_index)
        if len(indices) != expected_samples:
            raise ValueError(
                f"Layer {layer} has {len(indices)} records; expected {expected_samples}"
            )
        if shared_indices is None:
            shared_indices = indices
        elif indices != shared_indices:
            raise ValueError(f"Layer {layer} used a different sampled example set")

        comparable_config = {
            key: value for key, value in layer_config.items() if key != "scoring_layer"
        }
        if shared_config is None:
            shared_config = comparable_config
        elif comparable_config != shared_config:
            raise ValueError(f"Layer {layer} has a different non-layer configuration")
        records_by_layer[layer] = [records_by_index[index] for index in indices]

    if shared_config is None:
        raise RuntimeError("No experiment configuration was loaded")
    return records_by_layer, shared_config


def _scipy_correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "n": int(len(x)),
        "pearson": float(pearson.statistic),
        "pearson_pvalue": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
    }


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = intercept + slope * x
    residual = y - fitted
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(1.0 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)),
    }


def build_layer_rows(
    records_by_layer: dict[int, Sequence[dict[str, Any]]],
    requested_metrics: Sequence[str],
) -> list[dict[str, Any]]:
    performance_by_layer = _caption_performance_many(
        records_by_layer, requested_metrics
    )
    rows: list[dict[str, Any]] = []
    for layer, records in records_by_layer.items():
        divergence = {
            metric: mean_finite(record.get(metric) for record in records)
            for metric in JSD_METRICS
        }
        if any(value is None or value <= 0.0 for value in divergence.values()):
            raise ValueError(f"Layer {layer} contains a non-positive mean JSD")
        rows.append(
            {
                "layer": layer,
                **{metric: float(divergence[metric]) for metric in JSD_METRICS},
                **{
                    f"log_{metric}": float(np.log(divergence[metric]))
                    for metric in JSD_METRICS
                },
                **{
                    f"performance_{metric}": float(value)
                    for metric, value in performance_by_layer[layer].items()
                },
            }
        )
    return sorted(rows, key=lambda row: row["layer"])


def analyze_log_jsd(
    rows: Sequence[dict[str, Any]], performance_metrics: Sequence[str]
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for performance_metric in performance_metrics:
        y = np.asarray(
            [row[f"performance_{performance_metric}"] for row in rows],
            dtype=np.float64,
        )
        results[performance_metric] = {}
        for jsd_metric in JSD_METRICS:
            raw = np.asarray([row[jsd_metric] for row in rows], dtype=np.float64)
            log = np.log(raw)
            results[performance_metric][jsd_metric] = {
                "raw_jsd": correlations(raw, y),
                "log_jsd": _scipy_correlations(log, y),
                "linear_fit_on_log_jsd": _linear_fit(log, y),
            }
    return results


def _confidence_band(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit = _linear_fit(x, y)
    fitted = fit["intercept"] + fit["slope"] * grid
    residual = y - (fit["intercept"] + fit["slope"] * x)
    dof = len(x) - 2
    residual_scale = np.sqrt(np.sum(residual**2) / dof)
    centered_sum = np.sum((x - x.mean()) ** 2)
    standard_error = residual_scale * np.sqrt(
        1.0 / len(x) + (grid - x.mean()) ** 2 / centered_sum
    )
    radius = t.ppf(0.975, dof) * standard_error
    return fitted, fitted - radius, fitted + radius


def plot_cider_log_jsd(
    rows: Sequence[dict[str, Any]], output_base: Path
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    layers = np.asarray([row["layer"] for row in rows], dtype=np.int64)
    y = np.asarray([row["performance_CIDEr"] for row in rows], dtype=np.float64)
    panel_specs = (
        ("first_token_jsd", "(a) First-token divergence", r"$\ln(\mathrm{JSD}_{\mathrm{first}})$"),
        (
            "generation_mean_jsd",
            "(b) Generation-average divergence",
            r"$\ln(\overline{\mathrm{JSD}}_{\mathrm{generation}})$",
        ),
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(7.05, 3.0), sharey=True, constrained_layout=True
    )
    scatter = None
    for axis, (metric, title, xlabel) in zip(axes, panel_specs):
        x = np.asarray([row[f"log_{metric}"] for row in rows], dtype=np.float64)
        grid = np.linspace(x.min(), x.max(), 300)
        fitted, lower, upper = _confidence_band(x, y, grid)
        scatter = axis.scatter(
            x,
            y,
            c=layers,
            cmap="viridis",
            vmin=layers.min(),
            vmax=layers.max(),
            s=39,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            zorder=3,
        )
        axis.fill_between(
            grid, lower, upper, color="#D55E00", alpha=0.13, linewidth=0, zorder=1
        )
        axis.plot(grid, fitted, color="#D55E00", linewidth=1.6, zorder=2)
        stats = _scipy_correlations(x, y)
        axis.text(
            0.04,
            0.06,
            rf"Pearson $r={stats['pearson']:.3f}$" + "\n" + rf"Spearman $\rho={stats['spearman']:.3f}$",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.88,
                "linewidth": 0.6,
            },
        )
        for layer in (0, 19):
            index = int(np.flatnonzero(layers == layer)[0])
            offset = (-4, 6) if layer == 0 else (4, 5)
            axis.annotate(
                f"L{layer}",
                (x[index], y[index]),
                xytext=offset,
                textcoords="offset points",
                ha="right" if layer == 0 else "left",
                va="bottom",
                fontsize=7.5,
                color="0.15",
            )
        axis.set_title(title, pad=5)
        axis.set_xlabel(xlabel)
        axis.grid(True, color="0.88", linewidth=0.55, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("CIDEr after Top-64 pruning")
    if scatter is None:
        raise RuntimeError("No scatter plot was created")
    colorbar = fig.colorbar(scatter, ax=axes, pad=0.015, fraction=0.035)
    colorbar.set_label("Scoring layer", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=500)
    parser.add_argument(
        "--output-stem", default="refcoco_top64_log_jsd_vs_cider"
    )
    args = parser.parse_args()

    records_by_layer, config = load_sharded_experiment(
        args.input_root, args.expected_samples
    )
    configured_metrics = config.get("caption_metrics", DEFAULT_CAPTION_METRICS)
    if not isinstance(configured_metrics, list):
        configured_metrics = list(DEFAULT_CAPTION_METRICS)
    performance_metrics = list(configured_metrics)
    if "CIDEr" not in performance_metrics:
        performance_metrics.append("CIDEr")
    rows = build_layer_rows(records_by_layer, performance_metrics)
    analysis = {
        "method": {
            "unit": "scoring_layer",
            "num_layers": len(rows),
            "samples_per_layer": args.expected_samples,
            "log_transform": "natural logarithm of the layer-mean JSD",
            "fit": "ordinary least squares with performance as the response",
            "confidence_band": "pointwise 95% confidence interval for the fitted mean",
        },
        "input_root": str(args.input_root),
        "config": config,
        "correlations": analyze_log_jsd(rows, performance_metrics),
        "rows": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output_dir / args.output_stem
    plot_cider_log_jsd(rows, output_base)
    json_path = output_base.with_name(f"{output_base.name}_correlations.json")
    json_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tsv_path = output_base.with_name(f"{output_base.name}_layer_metrics.tsv")
    write_rows(tsv_path, rows)

    print(f"Saved figure: {output_base.with_suffix('.pdf')}")
    print(f"Saved figure: {output_base.with_suffix('.png')}")
    print(f"Saved correlations: {json_path}")
    print(f"Saved layer table: {tsv_path}")
    for metric in JSD_METRICS:
        stats = analysis["correlations"]["CIDEr"][metric]["log_jsd"]
        print(
            f"CIDEr vs log({metric}): Pearson={stats['pearson']:.6f} "
            f"(p={stats['pearson_pvalue']:.3g}), "
            f"Spearman={stats['spearman']:.6f} "
            f"(p={stats['spearman_pvalue']:.3g})"
        )


if __name__ == "__main__":
    main()
