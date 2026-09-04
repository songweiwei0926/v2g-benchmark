#!/usr/bin/env python3
"""Build the training-overlap (leakage) registry from models.yaml.

For each model declared in the model registry, record its training provenance
(dataset, cell type, assay, label type, variants/genes/pairs) and a coarse
benchmark-overlap flag. The registry is used downstream to assign per-pair
leakage categories and to filter strict no-leakage subsets.

CLI (Snakemake interface)
-------------------------
    build_leakage_registry.py
        --models <models.yaml> --output <parquet> --registry-output <tsv>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from v2gbench.benchmark.leakage import build_leakage_registry
from v2gbench.io.parquet import write_parquet, write_tsv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training-overlap (leakage) registry from models.yaml."
    )
    parser.add_argument("--models", required=True, help="Path to models.yaml")
    parser.add_argument("--output", required=True, help="Output registry parquet path")
    parser.add_argument("--registry-output", required=True, help="Output registry TSV path")
    args = parser.parse_args()

    print(f"[build_leakage_registry] Loading model config: {args.models}")
    registry = build_leakage_registry(args.models, output_path=args.output)
    print(f"[build_leakage_registry] registry rows: {registry.height}")

    write_parquet(registry, args.output)
    write_tsv(registry, args.registry_output)
    print(f"[build_leakage_registry] Wrote parquet -> {args.output}")
    print(f"[build_leakage_registry] Wrote TSV -> {args.registry_output}")

    # Summary by benchmark_overlap flag.
    if "benchmark_overlap" in registry.columns and registry.height > 0:
        print("[build_leakage_registry] Overlap summary:")
        print(registry.group_by("benchmark_overlap").len().sort("benchmark_overlap"))


if __name__ == "__main__":
    main()
