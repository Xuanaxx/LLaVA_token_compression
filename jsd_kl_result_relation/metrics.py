#!/usr/bin/env python3
"""Numerically stable distribution and correlation helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


DIVERGENCE_METRICS = (
    "first_token_jsd",
    "first_token_kl_full_to_pruned",
    "first_token_kl_pruned_to_full",
    "generation_mean_jsd",
    "generation_mean_kl_full_to_pruned",
    "generation_mean_kl_pruned_to_full",
)


def distribution_divergence_tensors(
    full_logits: torch.Tensor,
    pruned_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return exact divergence metrics without synchronizing the accelerator."""

    full = full_logits.detach().float().reshape(-1)
    pruned = pruned_logits.detach().float().reshape(-1)
    if full.shape != pruned.shape:
        raise ValueError(
            f"Logit shapes differ: {tuple(full.shape)} != {tuple(pruned.shape)}"
        )

    full_logp = F.log_softmax(full, dim=-1)
    pruned_logp = F.log_softmax(pruned, dim=-1)
    full_p = full_logp.exp()
    pruned_p = pruned_logp.exp()
    mixture_logp = (0.5 * (full_p + pruned_p)).clamp_min(1e-30).log()
    kl_full_to_pruned = (full_p * (full_logp - pruned_logp)).sum()
    kl_pruned_to_full = (pruned_p * (pruned_logp - full_logp)).sum()
    jsd = 0.5 * (
        (full_p * (full_logp - mixture_logp)).sum()
        + (pruned_p * (pruned_logp - mixture_logp)).sum()
    )
    return {
        "jsd": jsd,
        "kl_full_to_pruned": kl_full_to_pruned,
        "kl_pruned_to_full": kl_pruned_to_full,
        "top1_match": torch.argmax(full).eq(torch.argmax(pruned)),
    }


def distribution_divergence(
    full_logits: torch.Tensor, pruned_logits: torch.Tensor
) -> dict[str, float | bool]:
    """Compare two categorical distributions represented by unnormalized logits."""

    tensors = distribution_divergence_tensors(full_logits, pruned_logits)
    values = (
        torch.stack(
            [
                tensors["jsd"],
                tensors["kl_full_to_pruned"],
                tensors["kl_pruned_to_full"],
                tensors["top1_match"].to(dtype=torch.float32),
            ]
        )
        .cpu()
        .tolist()
    )
    return {
        "jsd": float(values[0]),
        "kl_full_to_pruned": float(values[1]),
        "kl_pruned_to_full": float(values[2]),
        "top1_match": bool(values[3]),
    }


def mean_finite(values: Iterable[float | int | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return float(np.mean(finite)) if finite else None


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, matching scipy.stats.rankdata(method='average')."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError(
            f"Correlation inputs differ in shape: {x_arr.shape} != {y_arr.shape}"
        )
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[finite], y_arr[finite]
    if len(x_arr) < 2 or np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def correlations(
    x: Sequence[float], y: Sequence[float]
) -> dict[str, float | int | None]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError(
            f"Correlation inputs differ in shape: {x_arr.shape} != {y_arr.shape}"
        )
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[finite], y_arr[finite]
    return {
        "n": int(len(x_arr)),
        "pearson": pearson_correlation(x_arr, y_arr),
        "spearman": (
            pearson_correlation(rankdata(x_arr), rankdata(y_arr))
            if len(x_arr)
            else None
        ),
    }


def standardize_within_groups(
    values: Sequence[float], groups: Sequence[str]
) -> np.ndarray:
    """Z-score values independently in every task before pooled correlation."""

    arr = np.asarray(values, dtype=np.float64)
    group_arr = np.asarray(groups)
    if arr.shape != group_arr.shape:
        raise ValueError(
            f"Values/groups differ in shape: {arr.shape} != {group_arr.shape}"
        )
    result = np.full(arr.shape, np.nan, dtype=np.float64)
    for group in np.unique(group_arr):
        mask = (group_arr == group) & np.isfinite(arr)
        if mask.sum() < 2:
            continue
        std = float(np.std(arr[mask]))
        if std < 1e-12:
            continue
        result[mask] = (arr[mask] - float(np.mean(arr[mask]))) / std
    return result
