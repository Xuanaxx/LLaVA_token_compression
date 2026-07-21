#!/usr/bin/env python3
"""Plot RefCOCO and RefCOCO+ layer-level CIDEr against log-transformed JSD."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import pearsonr, spearmanr, t


TASK_SPECS = (
    ("refcoco", "RefCOCO", "o", "#D55E00", "-"),
    ("refcoco_plus", "RefCOCO+", "^", "#0072B2", "--"),
)
JSD_SPECS = (
    ("first_token_jsd", "(a) First-token divergence", r"$\ln(\mathrm{JSD}_{\mathrm{first}})$"),
    (
        "generation_mean_jsd",
        "(b) Generation-average divergence",
        r"$\ln(\overline{\mathrm{JSD}}_{\mathrm{generation}})$",
    ),
)


def load_rows(path: Path, expected_samples: int) -> dict[str, list[dict[str, Any]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle, delimiter="\t"))

    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for task, _, _, _, _ in TASK_SPECS:
        rows = sorted(
            (row for row in raw_rows if row["task"] == task),
            key=lambda row: int(row["layer"]),
        )
        layers = [int(row["layer"]) for row in rows]
        if layers != list(range(32)):
            raise ValueError(f"Expected layers 0--31 for {task}, found {layers}")
        counts = {int(row["num_records"]) for row in rows}
        if counts != {expected_samples}:
            raise ValueError(
                f"Expected {expected_samples} records per layer for {task}, found {counts}"
            )
        rows_by_task[task] = rows
    return rows_by_task


def correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "n": int(len(x)),
        "pearson": float(pearson.statistic),
        "pearson_pvalue": float(pearson.pvalue),
        "spearman": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
    }


def confidence_band(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = intercept + slope * grid
    residual = y - (intercept + slope * x)
    dof = len(x) - 2
    residual_scale = np.sqrt(np.sum(residual**2) / dof)
    centered_sum = np.sum((x - x.mean()) ** 2)
    standard_error = residual_scale * np.sqrt(
        1.0 / len(x) + (grid - x.mean()) ** 2 / centered_sum
    )
    radius = t.ppf(0.975, dof) * standard_error
    return fitted, fitted - radius, fitted + radius


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / values.std(ddof=0)


def analyze(rows_by_task: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    task_results: dict[str, Any] = {}
    pooled_results: dict[str, Any] = {}
    for metric, _, _ in JSD_SPECS:
        pooled_x: list[float] = []
        pooled_y: list[float] = []
        for task, _, _, _, _ in TASK_SPECS:
            rows = rows_by_task[task]
            x = np.log(np.asarray([float(row[metric]) for row in rows]))
            y = np.asarray([float(row["performance_CIDEr"]) for row in rows])
            task_results.setdefault(task, {})[metric] = correlations(x, y)
            pooled_x.extend(zscore(x))
            pooled_y.extend(zscore(y))
        pooled_results[metric] = correlations(
            np.asarray(pooled_x), np.asarray(pooled_y)
        )
    return {
        "method": {
            "unit": "scoring_layer",
            "samples_per_layer": 500,
            "x": "natural logarithm of layer-mean JSD",
            "y": "CIDEr of the top-64-pruned model",
            "pooled_analysis": "both axes are z-scored within task before concatenation",
        },
        "tasks": task_results,
        "pooled_task_standardized": pooled_results,
    }


def configure_style() -> None:
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


def plot(
    rows_by_task: dict[str, list[dict[str, Any]]],
    analysis: dict[str, Any],
    output_base: Path,
) -> None:
    configure_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(7.05, 3.05), sharey=True, constrained_layout=True
    )
    scatter = None

    for axis, (metric, title, xlabel) in zip(axes, JSD_SPECS):
        stats_lines: list[str] = []
        layer19_points: list[tuple[float, float]] = []
        for task, display, marker, line_color, line_style in TASK_SPECS:
            rows = rows_by_task[task]
            layers = np.asarray([int(row["layer"]) for row in rows])
            x = np.log(np.asarray([float(row[metric]) for row in rows]))
            y = np.asarray([float(row["performance_CIDEr"]) for row in rows])
            grid = np.linspace(x.min(), x.max(), 300)
            fitted, lower, upper = confidence_band(x, y, grid)

            scatter = axis.scatter(
                x,
                y,
                c=layers,
                cmap="viridis",
                vmin=0,
                vmax=31,
                marker=marker,
                s=36 if marker == "o" else 43,
                edgecolor="white",
                linewidth=0.45,
                alpha=0.88,
                zorder=3,
            )
            axis.fill_between(
                grid,
                lower,
                upper,
                color=line_color,
                alpha=0.075,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                grid,
                fitted,
                color=line_color,
                linestyle=line_style,
                linewidth=1.55,
                zorder=2,
            )
            stats = analysis["tasks"][task][metric]
            stats_lines.append(
                rf"{display}: $r={stats['pearson']:.3f}$, $\rho={stats['spearman']:.3f}$"
            )
            index19 = int(np.flatnonzero(layers == 19)[0])
            layer19_points.append((float(x[index19]), float(y[index19])))

        mean_l19_x = float(np.mean([point[0] for point in layer19_points]))
        mean_l19_y = float(np.mean([point[1] for point in layer19_points]))
        axis.annotate(
            "L19 (both)",
            (mean_l19_x, mean_l19_y),
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=7.3,
            color="0.15",
        )
        axis.text(
            0.04,
            0.06,
            "\n".join(stats_lines),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.35,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.9,
                "linewidth": 0.6,
            },
        )
        axis.set_title(title, pad=5)
        axis.set_xlabel(xlabel)
        axis.grid(True, color="0.88", linewidth=0.55, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel("CIDEr after Top-64 pruning")
    if scatter is None:
        raise RuntimeError("No scatter plot was created")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle=line_style,
            marker=marker,
            markerfacecolor="0.45",
            markeredgecolor="white",
            markersize=5,
            linewidth=1.55,
            label=display,
        )
        for _, display, marker, color, line_style in TASK_SPECS
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.7,
    )
    colorbar = fig.colorbar(scatter, ax=axes, pad=0.015, fraction=0.035)
    colorbar.set_label("Scoring layer", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=500)
    parser.add_argument(
        "--output-stem",
        default="refcoco_refcocoplus_top64_log_jsd_vs_cider",
    )
    args = parser.parse_args()

    rows_by_task = load_rows(args.input_tsv, args.expected_samples)
    analysis = analyze(rows_by_task)
    output_base = args.output_dir / args.output_stem
    plot(rows_by_task, analysis, output_base)
    analysis_path = output_base.with_name(f"{output_base.name}_correlations.json")
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved figure: {output_base.with_suffix('.pdf')}")
    print(f"Saved figure: {output_base.with_suffix('.png')}")
    print(f"Saved correlations: {analysis_path}")
    for metric, _, _ in JSD_SPECS:
        stats = analysis["pooled_task_standardized"][metric]
        print(
            f"Pooled CIDEr vs log({metric}): Pearson={stats['pearson']:.6f} "
            f"(p={stats['pearson_pvalue']:.3g}), "
            f"Spearman={stats['spearman']:.6f} "
            f"(p={stats['spearman_pvalue']:.3g})"
        )


if __name__ == "__main__":
    main()
