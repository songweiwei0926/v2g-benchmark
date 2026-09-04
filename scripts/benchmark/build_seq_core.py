#!/usr/bin/env python3
"""Build the SEQ_CORE deterministic stratified sample (<=5000 pairs).

Sequence models are expensive, so the benchmark defines a fixed, reproducible
SEQ_CORE subset of at most ``max_variant_contexts`` (default 5000)
variant-context pairs on which all sequence models are compared fairly.
Selection is deterministic (SHA256-ordered) and stratified across benchmark
source, distance bin, PIP bin, nearest status, chromosome and context
supergroup.

CLI (Snakemake interface)
-------------------------
    build_seq_core.py
        --evidence <parquet> --candidates <parquet> --output <parquet>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from v2gbench.benchmark.seq_core import build_seq_core
from v2gbench.io.parquet import read_parquet, write_parquet
from v2gbench.utils.config import load_config


def _find_root() -> Path:
    cur = Path.cwd()
    while cur != cur.parent:
        if (cur / "config" / "project.yaml").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SEQ_CORE deterministic stratified sample (<=5000 pairs)."
    )
    parser.add_argument("--evidence", required=True, help="Evidence-long parquet path")
    parser.add_argument("--candidates", required=True, help="Candidate pairs parquet path")
    parser.add_argument("--output", required=True, help="Output SEQ_CORE parquet path")
    args = parser.parse_args()

    root = _find_root()
    project_cfg = load_config(root / "config" / "project.yaml") if (root / "config" / "project.yaml").exists() else {}
    seq_cfg = project_cfg.get("sequence_core", {})
    seed = int(project_cfg.get("project", {}).get("seed", 20260904))
    max_vc = int(seq_cfg.get("max_variant_contexts", 5000))

    evidence_df = read_parquet(args.evidence)
    candidates_df = read_parquet(args.candidates)
    print(f"[build_seq_core] evidence rows={evidence_df.height} candidates rows={candidates_df.height}")
    print(f"[build_seq_core] max_variant_contexts={max_vc} seed={seed}")

    # The evidence frame is the primary input for stratified sampling. If it
    # lacks distance_to_tss / is_nearest, join candidate columns to enrich
    # the strata.
    enrich_cols = [c for c in ("distance_to_tss", "is_nearest", "chrom") if c not in evidence_df.columns and c in candidates_df.columns]
    if enrich_cols and {"variant_id", "gene_id", "context_id"}.issubset(evidence_df.columns) and {"variant_id", "gene_id", "context_id"}.issubset(candidates_df.columns):
        evidence_df = evidence_df.join(
            candidates_df.select(["variant_id", "gene_id", "context_id"] + enrich_cols).unique(),
            on=["variant_id", "gene_id", "context_id"],
            how="left",
        )

    seq_core = build_seq_core(
        evidence_df,
        max_variant_contexts=max_vc,
        seed=seed,
        output_path=args.output,
    )
    print(f"[build_seq_core] SEQ_CORE rows: {seq_core.height}")

    n_vc = seq_core.select(["variant_id", "context_id"]).n_unique() if {"variant_id", "context_id"}.issubset(seq_core.columns) else 0
    print(f"[build_seq_core] unique variant-context pairs: {n_vc}")
    print(f"[build_seq_core] Wrote -> {args.output}")


if __name__ == "__main__":
    main()
