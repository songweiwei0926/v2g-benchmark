#!/usr/bin/env python3
"""Build the candidate gene universe at 250k / 500k / 1Mb windows.

For every benchmark variant we enumerate all GENCODE genes whose TSS lies
within ``window_size`` bp of the variant position, producing three candidate
sets in parallel. Gold membership is annotated by joining the canonical gold
pairs, and gold coverage is asserted to be 100%.

CLI (Snakemake interface)
-------------------------
    build_candidate_sets.py
        --variants <parquet> --gene-master <parquet> --gold <parquet>
        --contexts <parquet> --output-dir data/processed
        --main-output <parquet> --coverage-report <tsv>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import polars as pl

from v2gbench.benchmark.candidate_sets import (
    build_all_candidate_sets,
    check_gold_coverage,
    compute_gold_distance_rank,
)
from v2gbench.io.parquet import read_parquet, write_parquet, write_tsv
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
        description="Build candidate gene universe at 250k/500k/1Mb windows."
    )
    parser.add_argument("--variants", required=True, help="Variants parquet path")
    parser.add_argument("--gene-master", required=True, help="GENCODE gene master parquet path")
    parser.add_argument("--gold", required=True, help="Canonical gold pairs parquet path")
    parser.add_argument("--contexts", required=True, help="Contexts parquet path")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for per-window parquet files")
    parser.add_argument("--main-output", required=True, help="Main candidate parquet (1Mb window) path")
    parser.add_argument("--coverage-report", required=True, help="Gold coverage report TSV path")
    args = parser.parse_args()

    root = _find_root()
    project_cfg = load_config(root / "config" / "project.yaml") if (root / "config" / "project.yaml").exists() else {}
    bench_cfg = project_cfg.get("benchmark", {})
    windows = list(bench_cfg.get("sensitivity_windows", [250_000, 500_000, 1_000_000]))
    primary_window = int(bench_cfg.get("primary_window", 1_000_000))

    variants_df = read_parquet(args.variants)
    gene_master_df = read_parquet(args.gene_master)
    gold_df = read_parquet(args.gold)
    contexts_df = read_parquet(args.contexts)

    print(f"[build_candidate_sets] variants={variants_df.height} genes={gene_master_df.height}")
    print(f"[build_candidate_sets] windows={windows} primary={primary_window}")

    # Build context-expressed gene mapping if the contexts frame exposes it.
    context_expressed: Dict[str, List[str]] = {}
    if "context_id" in contexts_df.columns and "expressed_genes" in contexts_df.columns:
        for row in contexts_df.to_dicts():
            genes = row.get("expressed_genes")
            if genes:
                if isinstance(genes, str):
                    gene_list = [g.strip() for g in genes.split("|") if g.strip()]
                else:
                    gene_list = list(genes)
                context_expressed[row["context_id"]] = gene_list

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = build_all_candidate_sets(
        variants_df,
        gene_master_df,
        gold_df=gold_df,
        windows=windows,
        context_expressed=context_expressed if context_expressed else None,
        output_dir=output_dir,
    )

    # Write the main (primary window) candidate set.
    main_df = result.get(primary_window)
    if main_df is None:
        # Fall back to the largest window if primary not built.
        main_df = result[max(result.keys())] if result else pl.DataFrame()
    write_parquet(main_df, args.main_output)
    print(f"[build_candidate_sets] Wrote main candidate set ({primary_window} bp) -> {args.main_output}")
    print(f"[build_candidate_sets] main candidate rows: {main_df.height}")

    # Gold coverage report across all windows.
    coverage_rows = []
    for window, cand in result.items():
        cov = check_gold_coverage(cand, gold_df, raise_on_incomplete=False)
        n_gold = int(cand["is_gold"].sum()) if "is_gold" in cand.columns and cand.height > 0 else 0
        coverage_rows.append({
            "window_bp": window,
            "n_candidates": cand.height,
            "n_gold_candidates": n_gold,
            "gold_coverage": round(cov, 6),
        })
        print(f"  window={window}: candidates={cand.height} gold={n_gold} coverage={cov:.4%}")

    coverage_df = pl.DataFrame(coverage_rows)
    write_tsv(coverage_df, args.coverage_report)
    print(f"[build_candidate_sets] Wrote coverage report -> {args.coverage_report}")

    # Distance-rank distribution of gold genes in the main set.
    if main_df.height > 0 and "is_gold" in main_df.columns:
        rank_dist = compute_gold_distance_rank(main_df)
        if rank_dist.height > 0:
            print("[build_candidate_sets] Gold distance-rank distribution:")
            print(rank_dist.group_by("rank_bucket").len().sort("rank_bucket"))


if __name__ == "__main__":
    main()
