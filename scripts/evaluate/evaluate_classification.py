#!/usr/bin/env python3
"""Compute classification metrics (AUPRC, AUROC, MCC).

The evaluation code is model-agnostic: it reads only model_id,
ranking_score, and the gold label. Only confident labels (0/1) are used;
unknown labels (-1) are skipped. MCC uses a 0.5 threshold on ranking_score.

CLI (Snakemake interface)
-------------------------
    evaluate_classification.py
        --predictions <parquet> --evidence <parquet> --output <tsv>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import polars as pl

from v2gbench.io.parquet import read_parquet, write_tsv
from v2gbench.metrics.classification import compute_all_classification_metrics


def _prepare_eval_frame(predictions_df: pl.DataFrame, evidence_df: pl.DataFrame) -> pl.DataFrame:
    """Join predictions to evidence to get the label column."""
    key_cols = [c for c in ("variant_id", "gene_id", "context_id")
                if c in predictions_df.columns and c in evidence_df.columns]
    if not key_cols:
        raise ValueError("predictions and evidence share no join keys")

    # Evidence provides the label; prefer 'label', fall back to a derived is_gold.
    ev_cols = key_cols[:]
    if "label" in evidence_df.columns:
        ev_cols.append("label")
    elif "is_gold" in evidence_df.columns:
        ev_cols.append("is_gold")
    ev_small = evidence_df.select(ev_cols).unique(subset=key_cols)

    joined = predictions_df.join(ev_small, on=key_cols, how="left")

    # Normalize to a 'label' column in {0, 1, -1}.
    if "label" not in joined.columns and "is_gold" in joined.columns:
        joined = joined.with_columns(pl.col("is_gold").cast(pl.Int64).alias("label"))
    if "label" in joined.columns:
        joined = joined.with_columns(pl.col("label").fill_null(-1).cast(pl.Int64))
    else:
        joined = joined.with_columns(pl.lit(-1).cast(pl.Int64).alias("label"))

    joined = joined.with_columns(pl.col("ranking_score").fill_null(0.0).cast(pl.Float64))
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute classification metrics (AUPRC, AUROC, MCC)."
    )
    parser.add_argument("--predictions", required=True, help="Predictions parquet path")
    parser.add_argument("--evidence", required=True, help="Evidence-long parquet path")
    parser.add_argument("--output", required=True, help="Output metrics TSV path")
    args = parser.parse_args()

    predictions_df = read_parquet(args.predictions)
    evidence_df = read_parquet(args.evidence)
    print(f"[evaluate_classification] predictions={predictions_df.height} evidence={evidence_df.height}")

    eval_df = _prepare_eval_frame(predictions_df, evidence_df)
    print(f"[evaluate_classification] joined eval frame: {eval_df.height} rows")

    if "model_id" not in eval_df.columns:
        raise ValueError("predictions must contain a 'model_id' column")

    results: List[Dict] = []
    for model_id, sub in eval_df.group_by("model_id", maintain_order=True):
        model_id_val = model_id[0] if isinstance(model_id, tuple) else model_id
        metrics = compute_all_classification_metrics(sub)
        metrics["model_id"] = str(model_id_val)
        n_pos = int((sub["label"] == 1).sum()) if "label" in sub.columns else 0
        n_neg = int((sub["label"] == 0).sum()) if "label" in sub.columns else 0
        metrics["n_positive"] = n_pos
        metrics["n_negative"] = n_neg
        results.append(metrics)

    if not results:
        out_df = pl.DataFrame(schema={
            "model_id": pl.Utf8, "AUPRC": pl.Float64, "AUROC": pl.Float64,
            "MCC": pl.Float64, "n_positive": pl.Int64, "n_negative": pl.Int64,
        })
    else:
        out_df = pl.DataFrame(results).select([
            "model_id", "AUPRC", "AUROC", "MCC", "n_positive", "n_negative",
        ]).sort("model_id")

    write_tsv(out_df, args.output)
    print(f"[evaluate_classification] Wrote metrics -> {args.output}")
    print(out_df)


if __name__ == "__main__":
    main()
