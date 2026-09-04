#!/usr/bin/env python3
"""Compute direction-of-effect metrics.

Evaluates whether a model correctly predicts the sign of a variant's effect
on a gene (up-regulating vs down-regulating). Metrics: direction accuracy,
balanced accuracy, direction MCC, and Spearman correlation with effect size.

The evaluation code is model-agnostic: it reads only model_id, signed_score,
and the ground-truth effect_direction / effect_size.

CLI (Snakemake interface)
-------------------------
    evaluate_direction.py
        --predictions <parquet> --evidence <parquet> --output <tsv>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import polars as pl

from v2gbench.io.parquet import read_parquet, write_tsv
from v2gbench.metrics.direction import (
    compute_direction_accuracy,
    compute_balanced_accuracy,
    compute_direction_mcc,
)
from v2gbench.metrics.effect_size import compute_spearman as compute_effect_spearman


def _prepare_eval_frame(predictions_df: pl.DataFrame, evidence_df: pl.DataFrame) -> pl.DataFrame:
    """Join predictions to evidence for effect_direction + effect_size."""
    key_cols = [c for c in ("variant_id", "gene_id", "context_id")
                if c in predictions_df.columns and c in evidence_df.columns]
    if not key_cols:
        raise ValueError("predictions and evidence share no join keys")

    ev_cols = key_cols + [c for c in ("effect_direction", "effect_size") if c in evidence_df.columns]
    ev_small = evidence_df.select(ev_cols).unique(subset=key_cols)

    joined = predictions_df.join(ev_small, on=key_cols, how="left")

    # Convert effect_direction string ("up"/"down"/"none"/"unknown") to int {+1,-1}.
    if "effect_direction" in joined.columns:
        joined = joined.with_columns(
            pl.when(pl.col("effect_direction") == "up").then(1)
            .when(pl.col("effect_direction") == "down").then(-1)
            .otherwise(None)
            .cast(pl.Float64)
            .alias("effect_direction")
        )
    else:
        joined = joined.with_columns(pl.lit(None).cast(pl.Float64).alias("effect_direction"))

    # Ensure signed_score exists.
    if "signed_score" not in joined.columns:
        joined = joined.with_columns(pl.lit(None).cast(pl.Float64).alias("signed_score"))

    return joined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute direction-of-effect metrics."
    )
    parser.add_argument("--predictions", required=True, help="Predictions parquet path")
    parser.add_argument("--evidence", required=True, help="Evidence-long parquet path")
    parser.add_argument("--output", required=True, help="Output metrics TSV path")
    args = parser.parse_args()

    predictions_df = read_parquet(args.predictions)
    evidence_df = read_parquet(args.evidence)
    print(f"[evaluate_direction] predictions={predictions_df.height} evidence={evidence_df.height}")

    eval_df = _prepare_eval_frame(predictions_df, evidence_df)
    print(f"[evaluate_direction] joined eval frame: {eval_df.height} rows")

    if "model_id" not in eval_df.columns:
        raise ValueError("predictions must contain a 'model_id' column")

    results: List[Dict] = []
    for model_id, sub in eval_df.group_by("model_id", maintain_order=True):
        model_id_val = model_id[0] if isinstance(model_id, tuple) else model_id
        metrics = {
            "model_id": str(model_id_val),
            "DirectionAccuracy": compute_direction_accuracy(sub),
            "BalancedAccuracy": compute_balanced_accuracy(sub),
            "DirectionMCC": compute_direction_mcc(sub),
            "Spearman": compute_effect_spearman(sub),
            "n_pairs": int(sub.height),
        }
        results.append(metrics)

    if not results:
        out_df = pl.DataFrame(schema={
            "model_id": pl.Utf8, "DirectionAccuracy": pl.Float64,
            "BalancedAccuracy": pl.Float64, "DirectionMCC": pl.Float64,
            "Spearman": pl.Float64, "n_pairs": pl.Int64,
        })
    else:
        out_df = pl.DataFrame(results).select([
            "model_id", "DirectionAccuracy", "BalancedAccuracy",
            "DirectionMCC", "Spearman", "n_pairs",
        ]).sort("model_id")

    write_tsv(out_df, args.output)
    print(f"[evaluate_direction] Wrote metrics -> {args.output}")
    print(out_df)


if __name__ == "__main__":
    main()
