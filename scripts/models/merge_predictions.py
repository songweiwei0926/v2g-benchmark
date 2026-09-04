#!/usr/bin/env python3
"""Merge all model predictions into a single parquet file.

Concatenates baselines, published, ENCODE-supplementary, Borzoi, Enformer and
AlphaGenome prediction frames into one long-format table conforming to the
canonical prediction schema. Missing inputs are skipped gracefully.

CLI (Snakemake interface)
-------------------------
    merge_predictions.py
        --baselines <parquet> --published <parquet> --encode-supplementary <parquet>
        --borzoi <parquet> --enformer <parquet> --alphagenome <parquet>
        --output <parquet>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import polars as pl

from v2gbench.io.parquet import read_parquet, write_parquet

# Canonical prediction columns (order preserved on output).
PREDICTION_COLUMNS = [
    "model_id", "model_family", "benchmark_id",
    "variant_id", "element_id", "gene_id", "context_id",
    "raw_score", "ranking_score", "signed_score",
    "coverage", "applicability", "source_mode",
]


def _safe_read(path: Optional[str], label: str) -> Optional[pl.DataFrame]:
    """Read a parquet file if it exists and is non-empty; return None otherwise."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[merge_predictions] SKIP {label}: {p} does not exist")
        return None
    df = read_parquet(p)
    if df.height == 0:
        print(f"[merge_predictions] SKIP {label}: {p} is empty")
        return None
    print(f"[merge_predictions] {label}: {df.height} rows from {p}")
    return df


def _align_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure the frame has all canonical prediction columns (fill missing with null/0)."""
    for col in PREDICTION_COLUMNS:
        if col not in df.columns:
            if col in ("coverage",):
                df = df.with_columns(pl.lit(0).cast(pl.Int64).alias(col))
            elif col in ("ranking_score", "raw_score"):
                df = df.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
            elif col == "applicability":
                df = df.with_columns(pl.lit("APPLICABLE").alias(col))
            elif col == "source_mode":
                df = df.with_columns(pl.lit("published_prediction").alias(col))
            else:
                df = df.with_columns(pl.lit(None).alias(col))
    # Keep extra columns (e.g. borzoi_sd_across_replicates) but order canonical first.
    extra = [c for c in df.columns if c not in PREDICTION_COLUMNS]
    return df.select(PREDICTION_COLUMNS + extra)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge all model predictions into a single parquet file."
    )
    parser.add_argument("--baselines", required=False, default=None, help="Baselines predictions parquet")
    parser.add_argument("--published", required=False, default=None, help="Published predictions parquet")
    parser.add_argument("--encode-supplementary", required=False, default=None, help="ENCODE supplementary predictions parquet")
    parser.add_argument("--borzoi", required=False, default=None, help="Borzoi predictions parquet")
    parser.add_argument("--enformer", required=False, default=None, help="Enformer predictions parquet")
    parser.add_argument("--alphagenome", required=False, default=None, help="AlphaGenome predictions parquet")
    parser.add_argument("--output", required=True, help="Output merged predictions parquet path")
    args = parser.parse_args()

    frames: List[pl.DataFrame] = []
    for label, path in [
        ("baselines", args.baselines),
        ("published", args.published),
        ("encode_supplementary", args.encode_supplementary),
        ("borzoi", args.borzoi),
        ("enformer", args.enformer),
        ("alphagenome", args.alphagenome),
    ]:
        df = _safe_read(path, label)
        if df is not None:
            frames.append(_align_columns(df))

    if not frames:
        print("[merge_predictions] WARNING: no input frames found; writing empty schema.")
        merged = pl.DataFrame(schema={c: pl.Utf8 for c in PREDICTION_COLUMNS})
    else:
        merged = pl.concat(frames, how="vertical_relaxed")

    write_parquet(merged, args.output)
    print(f"[merge_predictions] Wrote {merged.height} merged rows -> {args.output}")

    if "model_id" in merged.columns and merged.height > 0:
        print("[merge_predictions] Models:")
        print(merged.group_by("model_id").len().sort("model_id"))


if __name__ == "__main__":
    main()
