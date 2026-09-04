#!/usr/bin/env python3
"""Train integrated ensemble models (rank-mean, logistic, XGBoost).

Three ensembles combine per-model predictions into a single ranking:

    integrated_rank      — mean percentile rank across models (no training).
    integrated_logistic  — L2 logistic regression, nested chromosome-fold CV.
    integrated_xgboost   — XGBoost, chromosome-separated outer CV.

The feature matrix is built from a long predictions frame pivoted wide,
with explicit missing-indicator columns per feature. Chromosome-level folds
prevent leakage.

CLI (Snakemake interface)
-------------------------
    train_integrated.py
        --predictions <parquet> --candidates <parquet> --evidence <parquet>
        --output-dir predictions/integrated --feature-importance-output <tsv>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import polars as pl

from v2gbench.io.parquet import read_parquet, write_parquet, write_tsv
from v2gbench.models.integrated import (
    IntegratedRankAdapter,
    IntegratedLogisticAdapter,
    IntegratedXGBoostAdapter,
    prepare_features,
    chromosome_folds,
)
from v2gbench.utils.config import load_config


def _find_root() -> Path:
    cur = Path.cwd()
    while cur != cur.parent:
        if (cur / "config" / "project.yaml").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def _collect_feature_importance(logistic_result, xgboost_result, feature_cols) -> pl.DataFrame:
    """Extract feature importance from logistic and XGBoost models."""
    rows: List[Dict[str, Any]] = []

    # Logistic: average coefficient magnitude across folds.
    if logistic_result and "folds" in logistic_result:
        coefs = []
        for fold_info in logistic_result["folds"].values():
            model = fold_info.get("model")
            if model is not None and hasattr(model, "coef_"):
                coefs.append(np.abs(model.coef_[0]))
        if coefs:
            mean_coef = np.mean(coefs, axis=0)
            for feat, val in zip(feature_cols, mean_coef):
                rows.append({"model": "integrated_logistic", "feature": feat, "importance": float(val)})

    # XGBoost: average feature importance across folds.
    if xgboost_result and "folds" in xgboost_result:
        importances = []
        for fold_info in xgboost_result["folds"].values():
            model = fold_info.get("model")
            if model is not None and hasattr(model, "feature_importances_"):
                importances.append(model.feature_importances_)
        if importances:
            mean_imp = np.mean(importances, axis=0)
            all_cols = feature_cols + [f"{m}_missing" for m in feature_cols]
            for feat, val in zip(all_cols, mean_imp):
                rows.append({"model": "integrated_xgboost", "feature": feat, "importance": float(val)})

    if not rows:
        return pl.DataFrame(schema={"model": pl.Utf8, "feature": pl.Utf8, "importance": pl.Float64})
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train integrated models (rank-mean, logistic, XGBoost)."
    )
    parser.add_argument("--predictions", required=True, help="Merged per-model predictions parquet path")
    parser.add_argument("--candidates", required=True, help="Candidate pairs parquet path")
    parser.add_argument("--evidence", required=True, help="Evidence-long parquet path")
    parser.add_argument("--output-dir", default="predictions/integrated", help="Output directory for integrated predictions")
    parser.add_argument("--feature-importance-output", required=True, help="Feature importance TSV path")
    args = parser.parse_args()

    root = _find_root()
    project_cfg = load_config(root / "config" / "project.yaml") if (root / "config" / "project.yaml").exists() else {}
    feature_list = list(project_cfg.get("integrated_features", [
        "log_distance", "abc", "encode_re2g", "sce2g", "pgboost",
        "borzoi_abs", "alphagenome_abs", "enformer_abs",
    ]))
    seed = int(project_cfg.get("project", {}).get("seed", 20260904))

    predictions_df = read_parquet(args.predictions)
    candidate_df = read_parquet(args.candidates)
    evidence_df = read_parquet(args.evidence)
    print(f"[train_integrated] predictions={predictions_df.height} candidates={candidate_df.height}")
    print(f"[train_integrated] features={feature_list}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure candidate_df has a chrom column for fold assignment.
    if "chrom" not in candidate_df.columns and "variant_id" in candidate_df.columns:
        candidate_df = candidate_df.with_columns(
            pl.col("variant_id").map_elements(
                lambda v: v.split(":")[1] if v and ":" in v else "unknown",
                return_dtype=pl.Utf8,
            ).alias("chrom")
        )

    # Augment predictions with log_distance feature if not already present.
    # log_distance is derived from the candidate table's distance_to_tss.
    if "log_distance" not in predictions_df.columns and "distance_to_tss" in candidate_df.columns:
        dist = candidate_df.select(["variant_id", "gene_id", "context_id", "distance_to_tss"]).with_columns(
            (pl.col("distance_to_tss").cast(pl.Float64).log1p()).alias("log_distance")
        )
        predictions_df = predictions_df.join(dist, on=["variant_id", "gene_id", "context_id"], how="left")
        # If log_distance ended up as a per-pair column, emit it as a pseudo-model row.
        if "log_distance" in predictions_df.columns and "log_distance" not in [m for m in predictions_df["model_id"].unique().to_list()]:
            ld_rows = predictions_df.select(
                ["variant_id", "gene_id", "context_id", "log_distance"]
            ).unique().filter(pl.col("log_distance").is_not_null()).with_columns(
                pl.lit("log_distance").alias("model_id"),
                pl.col("log_distance").alias("ranking_score"),
            )
            predictions_df = pl.concat([
                predictions_df.select([c for c in predictions_df.columns if c != "log_distance"]),
                ld_rows.select(["model_id", "variant_id", "gene_id", "context_id", "ranking_score"]),
            ], how="diagonal_relaxed")

    inputs: Dict[str, Any] = {
        "predictions_df": predictions_df,
        "candidate_df": candidate_df,
    }

    all_frames: List[pl.DataFrame] = []
    logistic_result = None
    xgboost_result = None
    feat_cols: List[str] = []

    # --- integrated_rank (no training) ---
    print("[train_integrated] Training integrated_rank (mean percentile rank) ...")
    rank_adapter = IntegratedRankAdapter(config={"feature_list": feature_list})
    rank_preds = rank_adapter.score(inputs)
    write_parquet(rank_preds, output_dir / "integrated_rank.parquet")
    print(f"  integrated_rank: rows={rank_preds.height}")
    all_frames.append(rank_preds)

    # --- integrated_logistic ---
    print("[train_integrated] Training integrated_logistic (L2 logistic, chromosome CV) ...")
    logistic_adapter = IntegratedLogisticAdapter(config={"feature_list": feature_list})
    try:
        logistic_preds = logistic_adapter.score(inputs)
        write_parquet(logistic_preds, output_dir / "integrated_logistic.parquet")
        print(f"  integrated_logistic: rows={logistic_preds.height}")
        all_frames.append(logistic_preds)
        # Re-run training to capture results for feature importance.
        feat_df, feat_cols = prepare_features(predictions_df, candidate_df, feature_list)
        X = feat_df.select(feat_cols).to_numpy()
        y = feat_df["is_gold"].to_numpy()
        chroms = feat_df["chrom"].to_list()
        folds = chromosome_folds(chroms)
        logistic_result = __import__("v2gbench.models.integrated", fromlist=["train_logistic"]).train_logistic(X, y, chroms, folds)
    except Exception as exc:
        print(f"  integrated_logistic: ERROR {exc}")

    # --- integrated_xgboost ---
    print("[train_integrated] Training integrated_xgboost (XGBoost, chromosome CV) ...")
    xgb_adapter = IntegratedXGBoostAdapter(config={"feature_list": feature_list})
    try:
        xgb_preds = xgb_adapter.score(inputs)
        write_parquet(xgb_preds, output_dir / "integrated_xgboost.parquet")
        print(f"  integrated_xgboost: rows={xgb_preds.height}")
        all_frames.append(xgb_preds)
        if not feat_cols:
            feat_df, feat_cols = prepare_features(predictions_df, candidate_df, feature_list)
        all_cols = feat_cols + [f"{m}_missing" for m in feat_cols]
        X = feat_df.select(all_cols).to_numpy()
        y = feat_df["is_gold"].to_numpy()
        chroms = feat_df["chrom"].to_list()
        folds = chromosome_folds(chroms)
        xgboost_result = __import__("v2gbench.models.integrated", fromlist=["train_xgboost"]).train_xgboost(X, y, chroms, folds)
    except Exception as exc:
        print(f"  integrated_xgboost: ERROR {exc}")

    # --- Feature importance ---
    fi_df = _collect_feature_importance(logistic_result, xgboost_result, feat_cols or feature_list)
    write_tsv(fi_df, args.feature_importance_output)
    print(f"[train_integrated] Wrote feature importance -> {args.feature_importance_output}")

    # --- Combined output ---
    if all_frames:
        combined = pl.concat(all_frames, how="vertical_relaxed")
        write_parquet(combined, output_dir / "integrated_all.parquet")
        print(f"[train_integrated] Wrote combined -> {output_dir / 'integrated_all.parquet'} ({combined.height} rows)")


if __name__ == "__main__":
    main()
