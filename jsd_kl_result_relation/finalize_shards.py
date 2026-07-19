#!/usr/bin/env python3
"""Deterministically merge sample-sharded layer records and write summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from jsd_kl_result_relation.evaluate_layer import (
    _caption_performance_many,
    _load_records,
    _summary,
    _write_or_validate_sample_manifest,
    parse_layers,
)


def _config_fingerprint(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _base_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "scoring_layer"}


def finalize_shards(
    output_root: Path,
    scoring_layers: Sequence[int],
    shard_count: int,
    expected_samples: int,
) -> None:
    if shard_count <= 1:
        raise ValueError(f"shard_count must be greater than one, got {shard_count}")
    layers = list(dict.fromkeys(int(layer) for layer in scoring_layers))
    if not layers:
        raise ValueError("scoring_layers cannot be empty")

    configs: dict[int, dict[str, Any]] = {}
    records_by_layer: dict[int, list[dict[str, Any]]] = {}
    global_indices: list[int] | None = None
    common_config: dict[str, Any] | None = None

    for layer in layers:
        layer_dir = output_root / f"scoring_layer_{layer}"
        layer_config: dict[str, Any] | None = None
        indices_by_shard: list[list[int]] = []
        records_by_index: dict[int, dict[str, Any]] = {}

        for shard_index in range(shard_count):
            shard_number = shard_index + 1
            suffix = f".shard_{shard_number}_of_{shard_count}"
            manifest_path = layer_dir / f"sample_indices{suffix}.json"
            records_path = layer_dir / f"records{suffix}.jsonl"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing shard manifest: {manifest_path}")
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            config = payload.get("config")
            if not isinstance(config, dict):
                raise ValueError(f"Invalid config in {manifest_path}")
            if layer_config is None:
                layer_config = config
            elif config != layer_config:
                raise ValueError(f"Shard config mismatch in {manifest_path}")
            if int(config.get("scoring_layer", -1)) != layer:
                raise ValueError(
                    f"Manifest {manifest_path} describes scoring layer "
                    f"{config.get('scoring_layer')}, expected {layer}"
                )

            shard_indices = [int(index) for index in payload.get("indices", [])]
            if len(shard_indices) != len(set(shard_indices)):
                raise ValueError(f"Duplicate sample indices in {manifest_path}")
            indices_by_shard.append(shard_indices)

            expected_record_config = {
                "model": config["model"],
                "task": config["task"],
                "scoring_layer": int(layer),
                "topk": int(config["topk"]),
                "config_sha256": _config_fingerprint(config),
            }
            shard_records = _load_records(records_path, expected_record_config)
            if set(shard_records) != set(shard_indices):
                missing = sorted(set(shard_indices) - set(shard_records))
                extra = sorted(set(shard_records) - set(shard_indices))
                raise ValueError(
                    f"Incomplete records in {records_path}: missing={missing[:10]} extra={extra[:10]}"
                )
            overlap = set(records_by_index) & set(shard_records)
            if overlap:
                raise ValueError(
                    f"Samples occur in multiple shards for layer {layer}: {sorted(overlap)[:10]}"
                )
            records_by_index.update(shard_records)

        if layer_config is None:
            raise RuntimeError(f"No shard configuration found for layer {layer}")
        merged_indices = sorted(index for part in indices_by_shard for index in part)
        if len(merged_indices) != expected_samples:
            raise ValueError(
                f"Layer {layer} has {len(merged_indices)} merged samples; expected {expected_samples}"
            )
        for shard_index, shard_indices in enumerate(indices_by_shard):
            expected_shard = merged_indices[shard_index::shard_count]
            if shard_indices != expected_shard:
                raise ValueError(
                    f"Layer {layer} shard {shard_index + 1} does not match deterministic strided partition"
                )

        if global_indices is None:
            global_indices = merged_indices
        elif merged_indices != global_indices:
            raise ValueError(f"Layer {layer} used different global sample indices")
        normalized = _base_config(layer_config)
        if common_config is None:
            common_config = normalized
        elif normalized != common_config:
            raise ValueError(f"Layer {layer} has a different non-layer configuration")

        ordered_records = [records_by_index[index] for index in merged_indices]
        configs[layer] = layer_config
        records_by_layer[layer] = ordered_records
        _write_or_validate_sample_manifest(
            layer_dir / "sample_indices.json",
            merged_indices,
            layer_config,
        )
        _atomic_write_text(
            layer_dir / "records.jsonl",
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in ordered_records
            ),
        )

    if global_indices is None or common_config is None:
        raise RuntimeError("No shard outputs were finalized")
    caption_metrics = common_config.get("caption_metrics")
    if not isinstance(caption_metrics, list) or not caption_metrics:
        raise ValueError("Shard config has no caption_metrics list")
    performance_by_layer = _caption_performance_many(
        records_by_layer,
        caption_metrics,
    )
    for layer in layers:
        summary = _summary(
            records_by_layer[layer],
            configs[layer],
            global_indices,
            performance_by_layer[layer],
        )
        summary_path = output_root / f"scoring_layer_{layer}" / "layer_summary.json"
        _atomic_write_text(
            summary_path,
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )
        print(
            f"FINALIZED layer={layer} records={len(records_by_layer[layer])} "
            f"summary={summary_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scoring-layers", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    args = parser.parse_args()
    finalize_shards(
        output_root=args.output_root,
        scoring_layers=parse_layers(args.scoring_layers),
        shard_count=args.shard_count,
        expected_samples=args.expected_samples,
    )


if __name__ == "__main__":
    main()
