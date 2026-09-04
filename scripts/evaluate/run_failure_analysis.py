#!/usr/bin/env python3
"""Identify error cases and interesting loci.

For each model, finds cases where the gold gene was ranked poorly (error
cases) and loci that are consistently hard or easy across all models
(interesting loci). Error cases are defined as gold genes ranked below the
median or outside the top-k. Interesting loci are variants where all models
fail (hard) or all models succeed (easy).

The evaluation code is model-agnostic.

CLI (Snakemake interface)
-------------------------
    run_failure_analysis.py
        --predictions <parquet> --candidates <parquet> --evidence <parquet>
        --output <parquet> --interesting-output <parquet>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl

from v2gbench.io.parquet import read_parquet, write_parquet
from v2gbench.metrics.ranking import rank_candidates


def _prepare_eval_frame(predictions_df: pl.DataFrame, candidate_df: pl.DataFrame, evidence_df: pl.DataFrame) -> pl.DataFrame:
    key_cols = [c for c in ("variant_id", "gene_id", "context_id")
                if c in predictions_df.columns and c in candidate_df.columns]
    if not key_cols:
        raise ValueError("predictions and candidates share no join keys")
    cand_cols = key_cols + [c for c in ("is_gold", "distance_to_tss", "distance_rank") if c in candidate_df.columns]
    cand_small = candidate_df.select(cand_cols).unique(subset=key_cols)
    joined = predictions_df.join(cand_small, on=key_cols, how="left")
    if "is_gold" not in joined.columns:
        joined = joined.with_columns(pl.lit(0).cast(pl.Int64).alias("is_gold"))
    else:
        joined = joined.with_columns(pl.col("is_gold").fill_null(0).cast(pl.Int64))
    joined = joined.with_columns(pl.col("ranking_score").fill_null(0.0).cast(pl.Float64))

    # Join evidence for context.
    ev_key = [c for c in key_cols if c in evidence_df.columns]
    if ev_key:
        ev_cols = ev_key + [c for c in ("evidence_type", "source_dataset", "pip") if c in evidence_df.columns]
        ev_small = evidence_df.select(ev_cols).unique(subset=ev_key)
        joined = joined.join(ev_small, on=ev_key, how="left")
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identify error cases and interesting loci."
    )
    parser.add_argument("--predictions", required=True, help="Predictions parquet path")
    parser.add_argument("--candidates", required=True, help="Candidate pairs parquet path")
    parser.add_argument("--evidence", required=True, help="Evidence-long parquet path")
    parser.add_argument("--output", required=True, help="Output error cases parquet path")
    parser.add_argument("--interesting-output", required=True, help="Output interesting loci parquet path")
    args = parser.parse_args()

    predictions_df = read_parquet(args.predictions)
    candidate_df = read_parquet(args.candidates)
    evidence_df = read_parquet(args.evidence)
    print(f"[run_failure_analysis] predictions={predictions_df.height} candidates={candidate_df.height}")

    eval_df = _prepare_eval_frame(predictions_df, candidate_df, evidence_df)

    if "model_id" not in eval_df.columns:
        raise ValueError("predictions must contain a 'model_id' column")

    # Rank candidates per (model, variant) with deterministic tie-breaking.
    error_rows: List[Dict] = []
    model_ids = [str(m[0] if isinstance(m, tuple) else m) for m, _ in eval_df.group_by("model_id", maintain_order=True)]
    print(f"[run_failure_analysis] Models: {model_ids}")

    # Per-model per-variant gold rank.
    per_model_gold_ranks: Dict[str, pl.DataFrame] = {}
    for model_id in model_ids:
        sub = eval_df.filter(pl.col("model_id") == model_id)
        ranked = rank_candidates(sub)
        gold = ranked.filter(pl.col("is_gold") == 1)
        if gold.height == 0:
            continue
        # Best gold rank per variant.
        best_gold = gold.sort("rank").group_by("variant_id", maintain_order=True).first()
        best_gold = best_gold.with_columns(pl.lit(model_id).alias("model_id"))
        per_model_gold_ranks[model_id] = best_gold

        # Error cases: gold gene ranked > 3 (outside top-3) or below median.
        errors = best_gold.filter(pl.col("rank") > 3)
        for row in errors.to_dicts():
            error_rows.append({
                "model_id": model_id,
                "variant_id": row["variant_id"],
                "gene_id": row["gene_id"],
                "gold_rank": int(row["rank"]),
                "ranking_score": float(row.get("ranking_score", 0.0)),
                "distance_to_tss": row.get("distance_to_tss"),
                "evidence_type": row.get("evidence_type"),
                "source_dataset": row.get("source_dataset"),
            })

    error_df = pl.DataFrame(error_rows) if error_rows else pl.DataFrame(schema={
        "model_id": pl.Utf8, "variant_id": pl.Utf8, "gene_id": pl.Utf8,
        "gold_rank": pl.Int64, "ranking_score": pl.Float64,
        "distance_to_tss": pl.Int64, "evidence_type": pl.Utf8, "source_dataset": pl.Utf8,
    })
    write_parquet(error_df, args.output)
    print(f"[run_failure_analysis] Wrote {error_df.height} error cases -> {args.output}")

    # --- Interesting loci: variants where all models fail (hard) or all succeed (easy) ---
    if per_model_gold_ranks:
        # Combine all per-model gold ranks.
        all_gold = pl.concat(list(per_model_gold_ranks.values()), how="vertical_relaxed")

        # Per variant: fraction of models with gold in top-3 (success rate).
        variant_success = (
            all_gold.with_columns((pl.col("rank") <= 3).alias("success"))
            .group_by("variant_id")
            .agg(
                pl.col("success").mean().alias("success_rate"),
                pl.col("model_id").n_unique().alias("n_models"),
                pl.col("rank").mean().alias("mean_gold_rank"),
                pl.col("rank").max().alias("max_gold_rank"),
                pl.col("rank").min().alias("min_gold_rank"),
            )
        )

        # Hard loci: success_rate == 0 (all models fail).
        hard = variant_success.filter(pl.col("success_rate") == 0.0).with_columns(pl.lit("hard").alias("locus_type"))
        # Easy loci: success_rate == 1.0 (all models succeed).
        easy = variant_success.filter(pl.col("success_rate") == 1.0).with_columns(pl.lit("easy").alias("locus_type"))
        # Ambiguous: in between.
        amb = variant_success.filter((pl.col("success_rate") > 0.0) & (pl.col("success_rate") < 1.0)).with_columns(pl.lit("ambiguous").alias("locus_type"))

        interesting = pl.concat([hard, easy, amb], how="vertical_relaxed").sort(["locus_type", "variant_id"])
    else:
        interesting = pl.DataFrame(schema={
            "variant_id": pl.Utf8, "success_rate": pl.Float64, "n_models": pl.Int64,
            "mean_gold_rank": pl.Float64, "max_gold_rank": pl.Int64, "min_gold_rank": pl.Int64,
            "locus_type": pl.Utf8,
        })

    write_parquet(interesting, args.interesting_output)
    print(f"[run_failure_analysis] Wrote {interesting.height} interesting loci -> {args.interesting_output}")
    if interesting.height > 0:
        print(interesting.group_by("locus_type").len().sort("locus_type"))


if __name__ == "__main__":
    main()
