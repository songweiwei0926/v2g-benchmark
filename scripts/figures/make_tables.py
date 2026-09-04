#!/usr/bin/env python3
"""Generate result tables and SupplementaryTables.xlsx for the V2G benchmark.

Usage:
    python scripts/figures/make_tables.py \
        --ranking results/metrics/ranking_metrics.tsv \
        --classification results/metrics/classification_metrics.tsv \
        --bootstrap results/bootstrap/paired_comparisons.tsv \
        --stratified results/stratified/stratified_metrics.parquet \
        --failures results/failures/error_cases.parquet \
        --candidates data/processed/candidate_1m.parquet \
        --evidence data/processed/evidence_long.parquet \
        --predictions data/processed/all_model_predictions.parquet \
        --integrated results/metrics/integrated_metrics.tsv \
        --feature-importance results/tables/integrated_feature_importance.tsv \
        --applicability results/tables/model_context_matrix.tsv \
        --leakage data/processed/leakage_registry.parquet \
        --encode-configs results/tables/all_encode_configurations.tsv \
        --output-dir results/tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from v2gbench.plotting import (
    make_main_results_table,
    make_supplementary_tables,
    make_model_exclusions_table,
    make_mandatory_completion_matrix,
)
from v2gbench.utils.config import load_config


def _read_tsv(path: str | None) -> pl.DataFrame | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return pl.read_csv(p, separator="\t")


def _read_parquet(path: str | None) -> pl.DataFrame | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return pl.read_parquet(p)


def _build_excluded_models(models_yaml_path: str) -> pl.DataFrame:
    """Load excluded models from models.yaml."""
    cfg = load_config(models_yaml_path)
    excluded = cfg.get("excluded_models", [])
    if isinstance(excluded, str):
        excluded = [excluded]
    if not excluded:
        excluded = []
    return pl.DataFrame({
        "model_id": excluded,
        "exclusion_reason": ["listed in excluded_models"] * len(excluded),
    })


def _build_completion_data(
    metrics_df: pl.DataFrame,
    predictions_df: pl.DataFrame | None,
    models_yaml_path: str,
) -> pl.DataFrame:
    """Build mandatory model completion matrix.

    For each mandatory model, check if it appears in metrics and predictions.
    """
    cfg = load_config(models_yaml_path)
    mandatory = cfg.get("main_figure_models", [])
    if isinstance(mandatory, str):
        mandatory = [mandatory]

    all_models = set(mandatory)
    if predictions_df is not None and "model_id" in predictions_df.columns:
        pred_models = set(predictions_df["model_id"].unique().to_list())
    else:
        pred_models = set()

    if metrics_df is not None and "model_id" in metrics_df.columns:
        metric_models = set(metrics_df["model_id"].unique().to_list())
    else:
        metric_models = set()

    rows = []
    for m in mandatory:
        has_metrics = m in metric_models
        has_predictions = m in pred_models
        status = "PASS" if (has_metrics and has_predictions) else "FAIL"
        rows.append({
            "model_id": m,
            "has_metrics": has_metrics,
            "has_predictions": has_predictions,
            "status": status,
        })
    return pl.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate V2G benchmark result tables."
    )
    parser.add_argument("--ranking", type=str, required=True)
    parser.add_argument("--classification", type=str, default=None)
    parser.add_argument("--bootstrap", type=str, default=None)
    parser.add_argument("--stratified", type=str, default=None)
    parser.add_argument("--failures", type=str, default=None)
    parser.add_argument("--candidates", type=str, default=None)
    parser.add_argument("--evidence", type=str, default=None)
    parser.add_argument("--predictions", type=str, default=None)
    parser.add_argument("--integrated", type=str, default=None)
    parser.add_argument("--feature-importance", type=str, default=None)
    parser.add_argument("--applicability", type=str, default=None)
    parser.add_argument("--leakage", type=str, default=None)
    parser.add_argument("--encode-configs", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--models-yaml", type=str, default="config/models.yaml")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ranking_df = _read_tsv(args.ranking)
    classification_df = _read_tsv(args.classification)
    stratified_df = _read_parquet(args.stratified)
    failures_df = _read_parquet(args.failures)
    candidates_df = _read_parquet(args.candidates)
    evidence_df = _read_parquet(args.evidence)
    predictions_df = _read_parquet(args.predictions)
    integrated_df = _read_tsv(args.integrated)
    feature_imp_df = _read_tsv(args.feature_importance)
    applicability_df = _read_tsv(args.applicability)
    leakage_df = _read_parquet(args.leakage)
    encode_configs_df = _read_tsv(args.encode_configs)
    bootstrap_df = _read_tsv(args.bootstrap)

    if ranking_df is None:
        print("ERROR: --ranking is required")
        return 1

    # Merge ranking + classification into unified metrics
    metrics_df = ranking_df
    if classification_df is not None:
        merge_cols = [c for c in ["model_id", "track", "subset"]
                      if c in metrics_df.columns and c in classification_df.columns]
        if merge_cols:
            metrics_df = metrics_df.join(
                classification_df, on=merge_cols, how="full"
            )
        else:
            metrics_df = pl.concat(
                [metrics_df, classification_df], how="diagonal"
            )

    if integrated_df is not None:
        merge_cols = [c for c in ["model_id", "track", "subset"]
                      if c in metrics_df.columns and c in integrated_df.columns]
        if merge_cols:
            metrics_df = metrics_df.join(
                integrated_df, on=merge_cols, how="full"
            )

    # 1. Main results table
    main_path = str(output_dir / "main_results.tsv")
    make_main_results_table(metrics_df, main_path)
    print(f"  main_results.tsv: {main_path}")

    # 2. Model exclusions table
    excluded_df = _build_excluded_models(args.models_yaml)
    excl_path = str(output_dir / "model_exclusions.tsv")
    make_model_exclusions_table(excluded_df, excl_path)
    print(f"  model_exclusions.tsv: {excl_path}")

    # 3. Mandatory completion matrix
    completion_df = _build_completion_data(
        metrics_df, predictions_df, args.models_yaml
    )
    comp_path = str(output_dir / "mandatory_completion_matrix.tsv")
    make_mandatory_completion_matrix(completion_df, comp_path)
    print(f"  mandatory_completion_matrix.tsv: {comp_path}")

    # 4. Model score QC table
    qc_path = str(output_dir / "model_score_qc.tsv")
    if predictions_df is not None and "model_id" in predictions_df.columns:
        qc_df = (
            predictions_df
            .group_by("model_id")
            .agg(
                pl.col("ranking_score").mean().alias("mean_score"),
                pl.col("ranking_score").std().alias("std_score"),
                pl.col("ranking_score").min().alias("min_score"),
                pl.col("ranking_score").max().alias("max_score"),
                pl.col("variant_id").n_unique().alias("n_variants"),
                pl.col("gene_id").n_unique().alias("n_genes"),
                pl.len().alias("n_predictions"),
            )
            .sort("model_id")
        )
        qc_df.write_csv(qc_path, separator="\t")
    else:
        pl.DataFrame({
            "model_id": [], "mean_score": [], "std_score": [],
            "min_score": [], "max_score": [], "n_variants": [],
            "n_genes": [], "n_predictions": [],
        }).write_csv(qc_path, separator="\t")
    print(f"  model_score_qc.tsv: {qc_path}")

    # 5. SupplementaryTables.xlsx
    all_data: Mapping[str, Any] = {
        "ranking": ranking_df,
        "classification": classification_df,
        "bootstrap": bootstrap_df,
        "stratified": stratified_df,
        "failures": failures_df,
        "candidates": candidates_df,
        "evidence": evidence_df,
        "predictions": predictions_df,
        "integrated": integrated_df,
        "feature_importance": feature_imp_df,
        "applicability": applicability_df,
        "leakage": leakage_df,
        "encode_configs": encode_configs_df,
        "excluded_models": excluded_df,
        "completion_matrix": completion_df,
    }
    supp_path = str(output_dir / "SupplementaryTables.xlsx")
    make_supplementary_tables(all_data, supp_path)
    print(f"  SupplementaryTables.xlsx: {supp_path}")

    print(f"\nAll tables generated in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
