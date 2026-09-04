#!/usr/bin/env python3
"""Apply the 6-level context mapping to a set of source contexts.

Utility script (not directly called by Snakemake) that applies the context
mapping cascade to a list of source context labels and prints / saves the
results.

Usage examples:
    # Map contexts from a file (one per line):
    python scripts/harmonize/map_contexts.py \
        --config config/context_mapping.yaml \
        --input contexts.txt \
        --output context_mapping.parquet

    # Map contexts passed on the command line:
    python scripts/harmonize/map_contexts.py \
        --config config/context_mapping.yaml \
        --contexts K562 "Whole Blood" Liver "T cell"

    # Map contexts extracted from a variants parquet:
    python scripts/harmonize/map_contexts.py \
        --config config/context_mapping.yaml \
        --variants data/processed/variants_normalized.parquet \
        --output context_mapping.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import polars as pl

from v2gbench.harmonize.contexts import normalize_context, load_context_aliases, map_context
from v2gbench.harmonize.map_contexts import (
    build_context_mapping_table,
    filter_primary_mapping,
    write_context_mapping,
)
from v2gbench.io.parquet import read_parquet, write_parquet

_CONTEXT_COL_CANDIDATES = [
    "context", "tissue", "cell_type", "cell_line",
    "tissue_name", "celltype", "source_context", "context_name",
]


def _read_contexts_from_file(path: Path) -> list[str]:
    """Read contexts from a text file (one per line)."""
    contexts = []
    with open(path) as fh:
        for line in fh:
            ctx = line.strip()
            if ctx:
                contexts.append(ctx)
    return contexts


def _read_contexts_from_parquet(path: Path) -> list[str]:
    """Extract unique contexts from a variants parquet."""
    df = read_parquet(path)
    ctx_col = None
    for col in _CONTEXT_COL_CANDIDATES:
        if col in df.columns:
            ctx_col = col
            break
    if ctx_col is None:
        print(f"ERROR: No context column found in {path}", file=sys.stderr)
        print(f"  Expected one of: {_CONTEXT_COL_CANDIDATES}", file=sys.stderr)
        return []
    return (
        df.select(pl.col(ctx_col).cast(pl.Utf8))
        .filter(pl.col(ctx_col).is_not_null() & (pl.col(ctx_col).str.strip() != ""))
        .unique()
        .to_series()
        .to_list()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply 6-level context mapping to source contexts"
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to context_mapping.yaml")
    parser.add_argument("--contexts", nargs="*", default=None,
                        help="Source context labels to map (space-separated)")
    parser.add_argument("--input", type=Path, default=None,
                        help="Text file with one context per line")
    parser.add_argument("--variants", type=Path, default=None,
                        help="Variants parquet to extract contexts from")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output parquet path (optional)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Filter to mappings with confidence >= this (default: 0.0 = all)")
    parser.add_argument("--primary-only", action="store_true",
                        help="Output only primary mappings (confidence >= 0.8)")
    args = parser.parse_args()

    # Gather source contexts.
    source_contexts: list[str] = []
    if args.contexts:
        source_contexts.extend(args.contexts)
    if args.input:
        source_contexts.extend(_read_contexts_from_file(args.input))
    if args.variants:
        source_contexts.extend(_read_contexts_from_parquet(args.variants))

    if not source_contexts:
        print("ERROR: No source contexts provided. Use --contexts, --input, or --variants.",
              file=sys.stderr)
        return 1

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_contexts: list[str] = []
    for ctx in source_contexts:
        if ctx not in seen:
            seen.add(ctx)
            unique_contexts.append(ctx)

    print(f"Mapping {len(unique_contexts)} source contexts ...")
    print(f"  Config: {args.config}")

    # Build the full mapping table.
    mapping_df = build_context_mapping_table(
        source_contexts=unique_contexts,
        config_path=args.config,
    )

    # Apply filtering.
    if args.primary_only:
        mapping_df = filter_primary_mapping(mapping_df, min_confidence=0.8)
        print(f"  Primary only: {mapping_df.height} rows")
    elif args.min_confidence > 0.0:
        mapping_df = mapping_df.filter(pl.col("mapping_confidence") >= args.min_confidence)
        print(f"  Confidence >= {args.min_confidence}: {mapping_df.height} rows")

    # Print results.
    if not mapping_df.is_empty():
        print(f"\n{'Source':<30} {'Matched':<25} {'Method':<25} {'Level':>5} {'Conf':>6}")
        print("-" * 95)
        for row in mapping_df.iter_rows(named=True):
            src = row.get("source_context", "") or ""
            matched = row.get("matched_context", "") or "(unmapped)"
            method = row.get("mapping_method", "") or ""
            level = row.get("mapping_level")
            conf = row.get("mapping_confidence", 0.0)
            level_str = str(level) if level is not None else "-"
            print(f"{src:<30} {matched:<25} {method:<25} {level_str:>5} {conf:>6.2f}")
    else:
        print("  No mappings produced.")

    # Write output if requested.
    if args.output:
        write_context_mapping(mapping_df, args.output)
        print(f"\nWrote mapping table: {args.output} ({mapping_df.height} rows)")

    # Summary.
    if not mapping_df.is_empty():
        mapped = mapping_df.filter(pl.col("mapping_method") != "unmapped").height
        unmapped = mapping_df.filter(pl.col("mapping_method") == "unmapped").height
        print(f"\nSummary: {mapped} mapped, {unmapped} unmapped out of {mapping_df.height}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
