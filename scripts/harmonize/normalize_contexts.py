#!/usr/bin/env python3
"""Normalize contexts from variant data and build a context mapping table.

Extracts unique context labels (cell type / tissue / cell line) from the
normalized variants parquet, normalizes them using
``v2gbench.harmonize.contexts``, and builds a full context mapping table
via ``v2gbench.harmonize.map_contexts``.

Outputs:
  - ``contexts_normalized.parquet`` — unique source contexts with normalized
    names and canonical context IDs.
  - ``context_mapping.parquet`` — full mapping table with matched canonical
    context, ontology ID, mapping method, level, and confidence.

CLI (Snakemake):
    python scripts/harmonize/normalize_contexts.py \
        --config config/context_mapping.yaml \
        --variants data/processed/variants_normalized.parquet \
        --output data/processed/contexts_normalized.parquet \
        --mapping-output data/processed/context_mapping.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from v2gbench.harmonize.contexts import normalize_context, load_context_aliases
from v2gbench.harmonize.map_contexts import (
    build_context_mapping_table,
    filter_primary_mapping,
    write_context_mapping,
)
from v2gbench.io.parquet import read_parquet, write_parquet
from v2gbench.schemas.context import make_context_id

# Column names that may hold context labels in the variants parquet.
_CONTEXT_COL_CANDIDATES = [
    "context",
    "tissue",
    "cell_type",
    "cell_line",
    "tissue_name",
    "celltype",
    "source_context",
    "context_name",
]


def _find_context_column(df: pl.DataFrame) -> str | None:
    """Find the column holding context labels in *df*."""
    for col in _CONTEXT_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize contexts and build context mapping table"
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to context_mapping.yaml")
    parser.add_argument("--variants", required=True, type=Path,
                        help="Path to normalized variants parquet")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output contexts_normalized.parquet path")
    parser.add_argument("--mapping-output", required=True, type=Path,
                        help="Output context_mapping.parquet path")
    args = parser.parse_args()

    if not args.variants.exists():
        print(f"ERROR: Variants parquet not found: {args.variants}", file=sys.stderr)
        return 1
    if not args.config.exists():
        print(f"ERROR: Context mapping config not found: {args.config}", file=sys.stderr)
        return 1

    print(f"Loading variants: {args.variants}")
    variants_df = read_parquet(args.variants)
    print(f"  Loaded {variants_df.height} variants, columns: {variants_df.columns}")

    # Find context column.
    ctx_col = _find_context_column(variants_df)
    if ctx_col is None:
        print(f"  WARNING: No context column found in variants parquet.", file=sys.stderr)
        print(f"  Expected one of: {_CONTEXT_COL_CANDIDATES}", file=sys.stderr)
        print(f"  Writing empty context outputs.", file=sys.stderr)
        empty_ctx = pl.DataFrame(schema={
            "source_context": pl.Utf8,
            "context_id": pl.Utf8,
            "normalized_name": pl.Utf8,
        })
        write_parquet(empty_ctx, args.output)
        empty_map = pl.DataFrame(schema={
            "source_context": pl.Utf8,
            "matched_context": pl.Utf8,
            "matched_ontology_id": pl.Utf8,
            "mapping_method": pl.Utf8,
            "mapping_level": pl.Int64,
            "ontology_distance": pl.Int64,
            "mapping_confidence": pl.Float64,
        })
        write_context_mapping(empty_map, args.mapping_output)
        return 0

    print(f"  Using context column: '{ctx_col}'")

    # Extract unique contexts.
    unique_contexts = (
        variants_df.select(pl.col(ctx_col).cast(pl.Utf8))
        .filter(pl.col(ctx_col).is_not_null() & (pl.col(ctx_col).str.strip() != ""))
        .unique()
        .to_series()
        .to_list()
    )
    print(f"  Unique contexts: {len(unique_contexts)}")

    # Normalize each context.
    ctx_rows = []
    for ctx in unique_contexts:
        norm = normalize_context(ctx)
        ctx_rows.append({
            "source_context": ctx,
            "context_id": make_context_id(ctx),
            "normalized_name": norm,
        })
    contexts_df = pl.DataFrame(ctx_rows)
    write_parquet(contexts_df, args.output)
    print(f"\nWrote normalized contexts: {args.output} ({contexts_df.height} rows)")

    # Build full mapping table.
    print(f"\nBuilding context mapping table ...")
    print(f"  Config: {args.config}")
    mapping_df = build_context_mapping_table(
        source_contexts=unique_contexts,
        config_path=args.config,
    )
    write_context_mapping(mapping_df, args.mapping_output)
    print(f"  Wrote full mapping: {args.mapping_output} ({mapping_df.height} rows)")

    # Report mapping summary.
    if not mapping_df.is_empty():
        method_counts = mapping_df.group_by("mapping_method").len().sort("len", descending=True)
        print(f"\n  Mapping method breakdown:")
        for row in method_counts.iter_rows(named=True):
            print(f"    {row['mapping_method']}: {row['len']}")

        # Primary (high-confidence) subset.
        primary = filter_primary_mapping(mapping_df, min_confidence=0.8)
        print(f"  Primary mapping (confidence >= 0.8): {primary.height}/{mapping_df.height}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
