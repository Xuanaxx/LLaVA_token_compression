#!/usr/bin/env python3
"""Create one deterministic DetailCaps sample shared by every benchmark method."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


DEFAULT_DATASET = Path("/data3/chenzixuan/DetailCaps-4870/DetailCaps-4870.parquet")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing cached sample after validating the requested configuration.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    parquet_file = pq.ParquetFile(dataset)
    total_rows = parquet_file.metadata.num_rows
    if args.num_samples > total_rows:
        raise ValueError(f"Requested {args.num_samples} rows from a {total_rows}-row dataset")

    stem = f"detailcaps_seed{args.seed}_n{args.num_samples}"
    sample_path = output_dir / f"{stem}.parquet"
    manifest_path = output_dir / f"{stem}_manifest.tsv"
    metadata_path = output_dir / f"{stem}_metadata.json"
    outputs = (sample_path, manifest_path, metadata_path)
    if any(path.exists() for path in outputs) and not args.force:
        if all(path.is_file() for path in outputs):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected = {
                "dataset": str(dataset),
                "dataset_size_bytes": dataset.stat().st_size,
                "dataset_rows": total_rows,
                "seed": args.seed,
                "num_samples": args.num_samples,
            }
            mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if mismatches:
                raise RuntimeError(f"Existing cached sample has incompatible metadata: {mismatches}")
            cached_rows = pq.ParquetFile(sample_path).metadata.num_rows
            if cached_rows != args.num_samples:
                raise RuntimeError(
                    f"Existing sample has {cached_rows} rows, expected {args.num_samples}; use --force"
                )
            print(f"Reusing validated sample: {sample_path}")
            return
        raise RuntimeError(f"Partial cached sample exists under {output_dir}; use --force")

    output_dir.mkdir(parents=True, exist_ok=True)
    indices = random.Random(args.seed).sample(range(total_rows), args.num_samples)
    # DetailCaps has one row group. Read only the three columns needed by the
    # efficiency benchmark, then persist the selected rows so three GPU runs do
    # not each scan the original 926 MiB parquet file.
    table = pq.read_table(dataset, columns=["source", "image", "binary"])
    selected = table.take(pa.array(indices, type=pa.int64()))
    selected = selected.append_column(
        "dataset_index", pa.array(indices, type=pa.int64())
    ).select(["dataset_index", "source", "image", "binary"])

    rows = selected.to_pylist()
    for row_number, row in enumerate(rows):
        try:
            with Image.open(io.BytesIO(row["binary"])) as image:
                image.verify()
        except Exception as exc:
            raise RuntimeError(
                f"Invalid image at sampled row {row_number} (dataset index {row['dataset_index']})"
            ) from exc

    pq.write_table(selected, sample_path, compression="zstd")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_order", "dataset_index", "source", "image"],
            delimiter="\t",
        )
        writer.writeheader()
        for sample_order, row in enumerate(rows):
            writer.writerow(
                {
                    "sample_order": sample_order,
                    "dataset_index": row["dataset_index"],
                    "source": row["source"],
                    "image": row["image"],
                }
            )

    source_counts: dict[str, int] = {}
    for row in rows:
        source = str(row["source"])
        source_counts[source] = source_counts.get(source, 0) + 1
    metadata = {
        "dataset": str(dataset),
        "dataset_size_bytes": dataset.stat().st_size,
        "dataset_rows": total_rows,
        "dataset_sha256": sha256(dataset),
        "seed": args.seed,
        "num_samples": args.num_samples,
        "sampling": "random.Random(seed).sample(range(dataset_rows), num_samples)",
        "sample_parquet": str(sample_path),
        "sample_parquet_sha256": sha256(sample_path),
        "source_counts": dict(sorted(source_counts.items())),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Created {sample_path} ({len(rows)} rows)")
    print(f"Source counts: {metadata['source_counts']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
