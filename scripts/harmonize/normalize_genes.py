#!/usr/bin/env python3
"""Parse a GENCODE GTF and build the canonical gene master table.

Uses ``v2gbench.harmonize.genes.build_gene_master_table`` to produce a
parquet with columns: ``gene_id``, ``gene_symbol``, ``chrom``, ``start``,
``end``, ``strand``, ``tss``, ``gene_type``, ``canonical_transcript``,
``exon_intervals`` (JSON-encoded).

CLI (Snakemake):
    python scripts/harmonize/normalize_genes.py \
        --gtf data/reference/gencode.v47.genes.gtf \
        --output data/reference/gene_master.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from v2gbench.harmonize.genes import build_gene_master_table
from v2gbench.io.parquet import read_parquet


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse GENCODE GTF and build gene master table"
    )
    parser.add_argument("--gtf", required=True, type=Path, help="Path to GENCODE GTF (may be .gz)")
    parser.add_argument("--output", required=True, type=Path, help="Output parquet path")
    args = parser.parse_args()

    if not args.gtf.exists():
        print(f"ERROR: GTF not found: {args.gtf}", file=sys.stderr)
        return 1

    print(f"Building gene master table from: {args.gtf}")
    print(f"  Output: {args.output}")

    try:
        result_path = build_gene_master_table(args.gtf, args.output)
    except Exception as exc:
        print(f"ERROR: Failed to build gene master table: {exc}", file=sys.stderr)
        return 1

    # Report summary.
    try:
        df = read_parquet(result_path)
        print(f"\nGene master table complete:")
        print(f"  Genes: {df.height}")
        print(f"  Columns: {df.columns}")
        if "gene_type" in df.columns:
            type_counts = df.group_by("gene_type").len().sort("len", descending=True).head(10)
            print(f"  Top gene types:")
            for row in type_counts.iter_rows(named=True):
                print(f"    {row['gene_type']}: {row['len']}")
    except Exception as exc:
        print(f"  (Could not read back summary: {exc})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
