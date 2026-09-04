#!/usr/bin/env python3
"""Compute ranking metrics (MRR, Top-1, Recall@3/5, NDCG).

The evaluation code is model-agnostic: it reads only model_id,
ranking_score, and the gold label (is_gold). Tie-breaking follows the
frozen project rules: score descending, distance ascending, gene_id
alphabetical. Missing predictions are kept with ranking_score=0.

CLI (Snakemake interface)
-------------------------
    evaluate_ranking.py
        --predictions <parquet> --candidates <parquet> --output <tsv>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import polars as pl

from v2gbench.io.parquet import read_parquet, write_tsv
from v2gbench.metrics.ranking import compute_all_ranking_metrics


def _prepare_eval_frame(predictions_df: pl.DataFrame, candidate_df: pl.DataFrame) -> pl.DataFrame:
    """Join predictions to candidates to get is_gold + distance_to_tss for tie-breaking."""
    key_cols = [c for c in ("variant_id", "gene_id", "context_id")
                if c in predictions_df.columns and c in candidate_df.columns]
    if not key_cols:
        raise ValueError("predictions and candidates share no join keys")

    # Candidate columns needed: is_gold, distance_to_tss.
    cand_cols = key_cols + [c for c in ("is_gold", "distance_to_tss") if c in candidate_df.columns]
    cand_small = candidate_df.select(cand_cols).unique(subset=key_cols)

    joined = predictions_df.join(cand_small, on=key_cols, how="left")

    # Ensure is_gold exists (default 0).
    if "is_gold" not in joined.columns:
        joined = joined.with_columns(pl.lit(0).cast(pl.Int64).alias("is_gold"))
    else:
        joined = joined.with_columns(pl.col("is_gold").fill_null(0).cast(pl.Int64))

    # Ensure ranking_score is non-null (missing -> 0).
    joined = joined.with_columns(pl.col("ranking_score").fill_null(0.0).cast(pl.Float64))

    return joined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ranking metrics (MRR, Top-1, Recall@3/5, NDCG)."
    )
    parser.add_argument("--predictions", required=True, help="Predictions parquet path")
    parser.add_argument("--candidates", required=True, help="Candidate pairs parquet path")
    parser.add_argument("--output", required=True, help="Output metrics TSV path")
    args = parser.parse_args()

    predictions_df = read_parquet(args.predictions)
    candidate_df = read_parquet(args.candidates)
    print(f"[evaluate_ranking] predictions={predictions_df.height} candidates={candidate_df.height}")

    eval_df = _prepare_eval_frame(predictions_df, candidate_df)
    print(f"[evaluate_ranking] joined eval frame: {eval_df.height} rows")

    if "model_id" not in eval_df.columns:
        raise ValueError("predictions must contain a 'model_id' column")

    # Compute metrics per model (model-agnostic: only ranking_score + is_gold used).
    results: List[Dict] = []
    for model_id, sub in eval_df.group_by("model_id", maintain_order=True):
        model_id_val = model_id[0] if isinstance(model_id, tuple) else model_id
        metrics = compute_all_ranking_metrics(sub)
        metrics["model_id"] = str(model_id_val)
        metrics["n_variants"] = int(sub["variant_id"].n_unique()) if "variant_id" in sub.columns else 0
        metrics["n_pairs"] = int(sub.height)
        results.append(metrics)

    if not results:
        print("[evaluate_ranking] WARNING: no models found in predictions.")
        out_df = pl.DataFrame(schema={
            "model_id": pl.Utf8, "MRR": pl.Float64, "Top1": pl.Float64,
            "Recall@3": pl.Float64, "Recall@5": pl.Float64, "NDCG": pl.Float64,
            "n_variants": pl.Int64, "n_pairs": pl.Int64,
        })
    else:
        out_df = pl.DataFrame(results).select([
            "model_id", "MRR", "Top1", "Recall@3", "Recall@5", "NDCG", "n_variants", "n_pairs",
        ]).sort("model_id")

    write_tsv(out_df, args.output)
    print(f"[evaluate_ranking] Wrote metrics -> {args.output}")
    print(out_df)


if __name__ == "__main__":
    main()
