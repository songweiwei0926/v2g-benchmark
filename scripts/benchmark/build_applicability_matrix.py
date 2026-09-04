#!/usr/bin/env python3
"""Build the model x context applicability matrix.

For every (model, context) pair, determine whether the model can produce a
prediction for that context, returning one of the APPLICABILITY_STATUS values.
The resulting long-format matrix is used to compute prediction coverage and
to define fair-comparison subsets.

CLI (Snakemake interface)
-------------------------
    build_applicability_matrix.py
        --models <models.yaml> --contexts <parquet> --output <tsv>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from v2gbench.benchmark.applicability import build_applicability_matrix
from v2gbench.io.parquet import read_parquet
from v2gbench.io.parquet import write_tsv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build model x context applicability matrix."
    )
    parser.add_argument("--models", required=True, help="Path to models.yaml")
    parser.add_argument("--contexts", required=True, help="Contexts parquet path")
    parser.add_argument("--output", required=True, help="Output applicability matrix TSV path")
    args = parser.parse_args()

    contexts_df = read_parquet(args.contexts)
    print(f"[build_applicability_matrix] contexts: {contexts_df.height}")

    matrix = build_applicability_matrix(args.models, contexts_df, output_path=None)
    print(f"[build_applicability_matrix] matrix rows: {matrix.height}")

    write_tsv(matrix, args.output)
    print(f"[build_applicability_matrix] Wrote -> {args.output}")

    # Summary by applicability status.
    if "applicability" in matrix.columns and matrix.height > 0:
        print("[build_applicability_matrix] Status summary:")
        print(matrix.group_by("applicability").len().sort("applicability"))


if __name__ == "__main__":
    main()
