#!/usr/bin/env python3
"""Run stratified analyses (distance, nearest, PIP, context, evidence).

Computes ranking and classification metrics within each stratum defined by
the frozen project parameters: distance bins, nearest-gene rank strata, PIP
bins, context match strata, and evidence type strata. The evaluation code is
model-agnostic.

CLI (Snakemake interface)
-------------------------
    run_stratified.py
        --predictions <parquet> --candidates <parquet> --evidence <parquet>
        --output <parquet>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl

from v2gbench.io.parquet import read_parquet, write_parquet
from v2gbench.metrics.ranking import compute_all_ranking_metrics
from v2gbench.metrics.classification import compute_all_classification_metrics
from v2gbench.statistics.sampling import assign_distance_bin, assign_pip_bin, assign_nearest_rank
from v2gbench.utils.config import load_config


def _find_root() -> Path:
    cur = Path.cwd()
    while cur != cur.parent:
        if (cur / "config" / "project.yaml").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def _prepare_eval_frame(predictions_df: pl.DataFrame, candidate_df: pl.DataFrame, evidence_df: pl.DataFrame) -> pl.DataFrame:
    key_cols = [c for c in ("variant_id", "gene_id", "context_id")
                if c in predictions_df.columns and c in candidate_df.columns]
    if not key_cols:
        raise ValueError("predictions and candidates share no join keys")

    cand_cols = key_cols + [c for c in ("is_gold", "distance_to_tss", "distance_rank", "is_nearest") if c in candidate_df.columns]
    cand_small = candidate_df.select(cand_cols).unique(subset=key_cols)
    joined = predictions_df.join(cand_small, on=key_cols, how="left")

    if "is_gold" not in joined.columns:
        joined = joined.with_columns(pl.lit(0).cast(pl.Int64).alias("is_gold"))
    else:
        joined = joined.with_columns(pl.col("is_gold").fill_null(0).cast(pl.Int64))
    joined = joined.with_columns(pl.col("ranking_score").fill_null(0.0).cast(pl.Float64))

    # Join evidence for label + pip + evidence_type.
    ev_key = [c for c in key_cols if c in evidence_df.columns]
    if ev_key:
        ev_cols = ev_key + [c for c in ("label", "pip", "evidence_type", "effect_direction") if c in evidence_df.columns]
        ev_small = evidence_df.select(ev_cols).unique(subset=ev_key)
        joined = joined.join(ev_small, on=ev_key, how="left")
        if "label" in joined.columns:
            joined = joined.with_columns(pl.col("label").fill_null(-1).cast(pl.Int64))

    return joined


def _distance_bin_label(d, bins):
    b = assign_distance_bin(d, [e[1] for e in bins])
    if b is None:
        return "unknown"
    lo = bins[b - 1][0] if b - 1 < len(bins) else 0
    hi = bins[b - 1][1] if b - 1 < len(bins) else bins[-1][1]
    return f"{lo}-{hi}"


def _stratify(df: pl.DataFrame, stratum_col: str, stratum_label: str) -> List[Tuple[str, pl.DataFrame]]:
    """Split df by a stratum column, returning (label, sub) pairs."""
    if stratum_col not in df.columns:
        return []
    out = []
    for val, sub in df.group_by(stratum_col, maintain_order=True):
        v = val[0] if isinstance(val, tuple) else val
        out.append((f"{stratum_label}:{v}", sub))
    return out


def _compute_metrics(sub: pl.DataFrame) -> Dict:
    metrics = compute_all_ranking_metrics(sub)
    metrics.update(compute_all_classification_metrics(sub))
    metrics["n_pairs"] = int(sub.height)
    metrics["n_variants"] = int(sub["variant_id"].n_unique()) if "variant_id" in sub.columns else 0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stratified analyses (distance, nearest, PIP, context, evidence)."
    )
    parser.add_argument("--predictions", required=True, help="Predictions parquet path")
    parser.add_argument("--candidates", required=True, help="Candidate pairs parquet path")
    parser.add_argument("--evidence", required=True, help="Evidence-long parquet path")
    parser.add_argument("--output", required=True, help="Output stratified results parquet path")
    args = parser.parse_args()

    root = _find_root()
    project_cfg = load_config(root / "config" / "project.yaml") if (root / "config" / "project.yaml").exists() else {}
    distance_bins = project_cfg.get("distance_bins", [[0, 10000], [10000, 50000], [50000, 100000], [100000, 250000], [250000, 500000], [500000, 1000000]])
    pip_bins = project_cfg.get("pip_bins", [[0.50, 0.70], [0.70, 0.90], [0.90, 0.95], [0.95, 1.01]])
    nearest_strata = project_cfg.get("nearest_strata", [1, 2, 3, "4+"])
    evidence_strata = project_cfg.get("evidence_strata", ["CRISPR", "eQTL", "GWAS", "curated_L2G"])

    predictions_df = read_parquet(args.predictions)
    candidate_df = read_parquet(args.candidates)
    evidence_df = read_parquet(args.evidence)
    print(f"[run_stratified] predictions={predictions_df.height} candidates={candidate_df.height} evidence={evidence_df.height}")

    eval_df = _prepare_eval_frame(predictions_df, candidate_df, evidence_df)

    # Assign strata columns.
    # Distance bin.
    if "distance_to_tss" in eval_df.columns:
        edges = [b[1] for b in distance_bins]
        eval_df = eval_df.with_columns(
            pl.col("distance_to_tss").map_elements(
                lambda d: _distance_bin_label(d, distance_bins), return_dtype=pl.Utf8
            ).alias("distance_stratum")
        )

    # Nearest rank stratum.
    if "distance_rank" in eval_df.columns:
        eval_df = eval_df.with_columns(
            pl.col("distance_rank").map_elements(
                lambda r: assign_nearest_rank(r), return_dtype=pl.Utf8
            ).alias("nearest_stratum")
        )

    # PIP bin stratum.
    if "pip" in eval_df.columns:
        pip_edges = [b[1] for b in pip_bins]
        eval_df = eval_df.with_columns(
            pl.col("pip").map_elements(
                lambda p: f"pip_{assign_pip_bin(p, pip_edges)}", return_dtype=pl.Utf8
            ).alias("pip_stratum")
        )

    # Context stratum (context_id).
    if "context_id" in eval_df.columns:
        eval_df = eval_df.with_columns(pl.col("context_id").alias("context_stratum"))

    # Evidence type stratum.
    if "evidence_type" in eval_df.columns:
        eval_df = eval_df.with_columns(pl.col("evidence_type").alias("evidence_stratum"))

    # Compute per-stratum per-model metrics.
    stratum_cols = {
        "distance_stratum": "distance",
        "nearest_stratum": "nearest",
        "pip_stratum": "pip",
        "context_stratum": "context",
        "evidence_stratum": "evidence",
    }

    results: List[Dict] = []
    model_ids = [str(m[0] if isinstance(m, tuple) else m) for m, _ in eval_df.group_by("model_id", maintain_order=True)]

    for stratum_col, stratum_name in stratum_cols.items():
        if stratum_col not in eval_df.columns:
            continue
        print(f"[run_stratified] Stratifying by {stratum_name} ...")
        for stratum_val, sub in eval_df.group_by(stratum_col, maintain_order=True):
            sv = stratum_val[0] if isinstance(stratum_val, tuple) else stratum_val
            for model_id in model_ids:
                model_sub = sub.filter(pl.col("model_id") == model_id)
                if model_sub.height == 0:
                    continue
                metrics = _compute_metrics(model_sub)
                metrics["model_id"] = model_id
                metrics["stratum_type"] = stratum_name
                metrics["stratum_value"] = str(sv)
                results.append(metrics)

    out_df = pl.DataFrame(results) if results else pl.DataFrame()
    write_parquet(out_df, args.output)
    print(f"[run_stratified] Wrote {out_df.height} stratified metric rows -> {args.output}")


if __name__ == "__main__":
    main()
