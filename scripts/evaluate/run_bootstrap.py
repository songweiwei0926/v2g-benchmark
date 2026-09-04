#!/usr/bin/env python3
"""Run 2000-replicate bootstrap + paired model comparisons.

Bootstrap resamples sampling units (variants/loci), NOT individual pairs,
preserving the within-variant ranking structure. For each model we compute
bootstrap CIs for the ranking metrics; for each model pair we compute the
paired bootstrap delta distribution, CI, and p-value with BH FDR correction.

Seed = 20260904 for all deterministic operations.

CLI (Snakemake interface)
-------------------------
    run_bootstrap.py
        --predictions <parquet> --candidates <parquet>
        --output <parquet> --paired-output <tsv>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl

from v2gbench.io.parquet import read_parquet, write_parquet, write_tsv
from v2gbench.metrics.ranking import compute_mrr, compute_top1, compute_recall_at_k, compute_ndcg
from v2gbench.statistics.bootstrap import bootstrap_metrics, compute_ci
from v2gbench.statistics.paired import pairwise_comparison, format_comparison_results
from v2gbench.utils.config import load_config


def _find_root() -> Path:
    cur = Path.cwd()
    while cur != cur.parent:
        if (cur / "config" / "project.yaml").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def _prepare_eval_frame(predictions_df: pl.DataFrame, candidate_df: pl.DataFrame) -> pl.DataFrame:
    key_cols = [c for c in ("variant_id", "gene_id", "context_id")
                if c in predictions_df.columns and c in candidate_df.columns]
    if not key_cols:
        raise ValueError("predictions and candidates share no join keys")
    cand_cols = key_cols + [c for c in ("is_gold", "distance_to_tss") if c in candidate_df.columns]
    cand_small = candidate_df.select(cand_cols).unique(subset=key_cols)
    joined = predictions_df.join(cand_small, on=key_cols, how="left")
    if "is_gold" not in joined.columns:
        joined = joined.with_columns(pl.lit(0).cast(pl.Int64).alias("is_gold"))
    else:
        joined = joined.with_columns(pl.col("is_gold").fill_null(0).cast(pl.Int64))
    joined = joined.with_columns(pl.col("ranking_score").fill_null(0.0).cast(pl.Float64))
    return joined


# Metric functions for bootstrap (each maps a per-model sub-frame -> float).
METRIC_FNS = {
    "MRR": compute_mrr,
    "Top1": compute_top1,
    "Recall@3": lambda df: compute_recall_at_k(df, 3),
    "Recall@5": lambda df: compute_recall_at_k(df, 5),
    "NDCG": compute_ndcg,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 2000-replicate bootstrap + paired model comparisons."
    )
    parser.add_argument("--predictions", required=True, help="Predictions parquet path")
    parser.add_argument("--candidates", required=True, help="Candidate pairs parquet path")
    parser.add_argument("--output", required=True, help="Output bootstrap results parquet path")
    parser.add_argument("--paired-output", required=True, help="Output paired comparison TSV path")
    args = parser.parse_args()

    root = _find_root()
    project_cfg = load_config(root / "config" / "project.yaml") if (root / "config" / "project.yaml").exists() else {}
    seed = int(project_cfg.get("project", {}).get("seed", 20260904))
    n_reps = int(project_cfg.get("bootstrap", {}).get("replicates", 2000))

    predictions_df = read_parquet(args.predictions)
    candidate_df = read_parquet(args.candidates)
    print(f"[run_bootstrap] predictions={predictions_df.height} candidates={candidate_df.height}")
    print(f"[run_bootstrap] replicates={n_reps} seed={seed}")

    eval_df = _prepare_eval_frame(predictions_df, candidate_df)

    if "model_id" not in eval_df.columns:
        raise ValueError("predictions must contain a 'model_id' column")

    # --- Per-model bootstrap CIs ---
    bootstrap_rows: List[Dict] = []
    model_ids = [str(m[0] if isinstance(m, tuple) else m) for m, _ in eval_df.group_by("model_id", maintain_order=True)]
    print(f"[run_bootstrap] Models: {model_ids}")

    for model_id in model_ids:
        sub = eval_df.filter(pl.col("model_id") == model_id)
        print(f"[run_bootstrap] Bootstrapping {model_id} ({sub.height} pairs) ...")
        for metric_name, fn in METRIC_FNS.items():
            # Point estimate.
            try:
                point = float(fn(sub))
            except Exception:
                point = float("nan")
            # Bootstrap distribution.
            reps = bootstrap_metrics(sub, fn, n_replicates=n_reps, unit="variant_id", seed=seed)
            ci_low, ci_high = compute_ci(reps)
            valid = reps[~np.isnan(reps)]
            se = float(np.std(valid)) if valid.size > 0 else float("nan")
            bootstrap_rows.append({
                "model_id": model_id,
                "metric": metric_name,
                "point_estimate": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "std_error": se,
                "n_replicates": int(valid.size),
            })

    bootstrap_df = pl.DataFrame(bootstrap_rows) if bootstrap_rows else pl.DataFrame(schema={
        "model_id": pl.Utf8, "metric": pl.Utf8, "point_estimate": pl.Float64,
        "ci_low": pl.Float64, "ci_high": pl.Float64, "std_error": pl.Float64, "n_replicates": pl.Int64,
    })
    write_parquet(bootstrap_df, args.output)
    print(f"[run_bootstrap] Wrote bootstrap CIs -> {args.output}")

    # --- Paired comparisons (all-pairs, on MRR as the primary metric) ---
    print("[run_bootstrap] Running paired comparisons (MRR) ...")
    comparisons = pairwise_comparison(
        eval_df,
        metric_fn=compute_mrr,
        n_replicates=n_reps,
        seed=seed,
        unit="variant_id",
    )
    paired_df = format_comparison_results(comparisons)
    write_tsv(paired_df, args.paired_output)
    print(f"[run_bootstrap] Wrote paired comparisons -> {args.paired_output}")
    if paired_df.height > 0:
        print(paired_df)


if __name__ == "__main__":
    main()
