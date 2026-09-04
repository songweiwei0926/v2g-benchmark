#!/usr/bin/env python3
"""Generate main and supplementary figures for the V2G benchmark.

Usage (main figures):
    python scripts/figures/make_figures.py \
        --ranking results/metrics/ranking_metrics.tsv \
        --classification results/metrics/classification_metrics.tsv \
        --stratified results/stratified/stratified_metrics.parquet \
        --bootstrap results/bootstrap/bootstrap_results.parquet \
        --predictions data/processed/all_model_predictions.parquet \
        --evidence data/processed/evidence_long.parquet \
        --candidates data/processed/candidate_1m.parquet \
        --integrated results/metrics/integrated_metrics.tsv \
        --failures results/failures/interesting_loci.parquet \
        --output-dir results/figures

Usage (supplementary only):
    python scripts/figures/make_figures.py \
        --supplementary-only \
        --stratified results/stratified/stratified_metrics.parquet \
        --predictions data/processed/all_model_predictions.parquet \
        --evidence data/processed/evidence_long.parquet \
        --failures results/failures/error_cases.parquet \
        --output-dir results/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from v2gbench.plotting import (
    make_fig1,
    make_fig2,
    make_fig3,
    make_fig4,
    make_fig5,
    make_fig6,
    make_supplementary_figures,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate V2G benchmark figures (SVG)."
    )
    parser.add_argument("--ranking", type=str, default=None)
    parser.add_argument("--classification", type=str, default=None)
    parser.add_argument("--stratified", type=str, default=None)
    parser.add_argument("--bootstrap", type=str, default=None)
    parser.add_argument("--predictions", type=str, default=None)
    parser.add_argument("--evidence", type=str, default=None)
    parser.add_argument("--candidates", type=str, default=None)
    parser.add_argument("--integrated", type=str, default=None)
    parser.add_argument("--failures", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--supplementary-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.supplementary_only:
        stratified_df = _read_parquet(args.stratified)
        predictions_df = _read_parquet(args.predictions)
        evidence_df = _read_parquet(args.evidence)
        failures_df = _read_parquet(args.failures)

        if stratified_df is None:
            print("ERROR: --stratified is required for supplementary figures")
            return 1

        # Use stratified as metrics proxy for supplementary
        metrics_df = stratified_df
        paths = make_supplementary_figures(
            metrics_df, stratified_df, str(output_dir)
        )
        print(f"Generated {len(paths)} supplementary figures in {output_dir}")
        for p in paths:
            print(f"  {p}")
        return 0

    # Main figures
    ranking_df = _read_tsv(args.ranking)
    classification_df = _read_tsv(args.classification)
    stratified_df = _read_parquet(args.stratified)
    predictions_df = _read_parquet(args.predictions)
    evidence_df = _read_parquet(args.evidence)
    candidates_df = _read_parquet(args.candidates)
    integrated_df = _read_tsv(args.integrated)
    failures_df = _read_parquet(args.failures)

    if ranking_df is None:
        print("ERROR: --ranking is required for main figures")
        return 1

    # Merge ranking + classification into a unified metrics_df
    metrics_df = ranking_df
    if classification_df is not None:
        merge_cols = [c for c in ["model_id", "track", "subset"]
                      if c in metrics_df.columns and c in classification_df.columns]
        if merge_cols:
            metrics_df = metrics_df.join(classification_df, on=merge_cols, how="full")
        else:
            metrics_df = pl.concat([metrics_df, classification_df], how="diagonal")

    # Add integrated metrics if available
    if integrated_df is not None:
        merge_cols = [c for c in ["model_id", "track", "subset"]
                      if c in metrics_df.columns and c in integrated_df.columns]
        if merge_cols:
            metrics_df = metrics_df.join(integrated_df, on=merge_cols, how="full")

    # Fig 1: Overview — needs metrics, evidence, candidates
    if evidence_df is not None and candidates_df is not None:
        p1 = make_fig1(metrics_df, evidence_df, candidates_df,
                       str(output_dir / "fig1_overview.svg"))
        print(f"  fig1: {p1}")
    else:
        print("  WARNING: Skipping fig1 (missing evidence or candidates)")

    # Fig 2: Heatmap — needs metrics
    p2 = make_fig2(metrics_df, str(output_dir / "fig2_heatmap.svg"))
    print(f"  fig2: {p2}")

    # Fig 3: Distance — needs stratified
    if stratified_df is not None:
        p3 = make_fig3(stratified_df, str(output_dir / "fig3_distance.svg"))
        print(f"  fig3: {p3}")
    else:
        print("  WARNING: Skipping fig3 (missing stratified)")

    # Fig 4: Context — needs stratified
    if stratified_df is not None:
        p4 = make_fig4(stratified_df, str(output_dir / "fig4_context.svg"))
        print(f"  fig4: {p4}")
    else:
        print("  WARNING: Skipping fig4 (missing stratified)")

    # Fig 5: Complementarity — needs predictions
    if predictions_df is not None:
        p5 = make_fig5(predictions_df, str(output_dir / "fig5_complementarity.svg"))
        print(f"  fig5: {p5}")
    else:
        print("  WARNING: Skipping fig5 (missing predictions)")

    # Fig 6: Integrated — needs metrics (with integrated model rows)
    p6 = make_fig6(metrics_df, str(output_dir / "fig6_integrated.svg"))
    print(f"  fig6: {p6}")

    print(f"\nAll main figures generated in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
