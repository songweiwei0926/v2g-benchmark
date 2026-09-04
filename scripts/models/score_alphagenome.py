#!/usr/bin/env python3
"""Run AlphaGenome API inference with retry/backoff.

AlphaGenome is accessed through DeepMind's hosted API. For each
(variant, context) pair we issue one request returning gene-centric RNA
scores, convert them into a signed log-fold-change and an absolute value,
plus a per-variant quantile score. The inference loop implements exponential
backoff with jitter, a persistent on-disk cache for resumability, and
periodic checkpointing.

Output columns: alphagenome_signed, alphagenome_abs, alphagenome_quantile.

CLI (Snakemake interface)
-------------------------
    score_alphagenome.py
        --seq-core <parquet> --gene-master <parquet> --output <parquet>
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import polars as pl

from v2gbench.io.parquet import read_parquet, write_parquet
from v2gbench.models.alphagenome import AlphaGenomeAdapter, run_alphagenome_inference
from v2gbench.schemas.variant import parse_variant_id
from v2gbench.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("score_alphagenome")


def _find_root() -> Path:
    cur = Path.cwd()
    while cur != cur.parent:
        if (cur / "config" / "project.yaml").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def _prepare_variants(seq_core_df: pl.DataFrame) -> pl.DataFrame:
    if all(c in seq_core_df.columns for c in ("chrom", "pos", "ref", "alt")):
        return seq_core_df
    if "variant_id" not in seq_core_df.columns:
        raise ValueError("seq_core must contain variant_id or chrom/pos/ref/alt columns")
    parsed = seq_core_df.with_columns(
        pl.col("variant_id").map_elements(lambda v: parse_variant_id(v)["chrom"], return_dtype=pl.Utf8).alias("chrom"),
        pl.col("variant_id").map_elements(lambda v: parse_variant_id(v)["pos"], return_dtype=pl.Int64).alias("pos"),
        pl.col("variant_id").map_elements(lambda v: parse_variant_id(v)["ref"], return_dtype=pl.Utf8).alias("ref"),
        pl.col("variant_id").map_elements(lambda v: parse_variant_id(v)["alt"], return_dtype=pl.Utf8).alias("alt"),
    )
    return parsed


def _build_contexts(seq_core_df: pl.DataFrame) -> List[Dict[str, Any]]:
    """Build the list of context dicts for AlphaGenome from the SEQ_CORE frame."""
    if "context_id" not in seq_core_df.columns:
        return [{"context_id": "default", "ontology_id": "", "scorer": "RNA_SEQ"}]
    # Unique contexts; carry ontology_id if present.
    ctx_cols = ["context_id"]
    if "ontology_id" in seq_core_df.columns:
        ctx_cols.append("ontology_id")
    unique = seq_core_df.select(ctx_cols).unique()
    contexts: List[Dict[str, Any]] = []
    for row in unique.to_dicts():
        contexts.append({
            "context_id": row["context_id"],
            "ontology_id": row.get("ontology_id", ""),
            "scorer": "RNA_SEQ",
        })
    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AlphaGenome API inference with retry/backoff on the SEQ_CORE subset."
    )
    parser.add_argument("--seq-core", required=True, help="SEQ_CORE parquet path")
    parser.add_argument("--gene-master", required=True, help="GENCODE gene master parquet path")
    parser.add_argument("--output", required=True, help="Output predictions parquet path")
    args = parser.parse_args()

    root = _find_root()
    project_cfg = load_config(root / "config" / "project.yaml") if (root / "config" / "project.yaml").exists() else {}
    models_cfg = load_config(root / "config" / "models.yaml") if (root / "config" / "models.yaml").exists() else {}

    seq_core_df = read_parquet(args.seq_core)
    gene_master_df = read_parquet(args.gene_master)
    print(f"[score_alphagenome] seq_core={seq_core_df.height} genes={gene_master_df.height}")

    variants_df = _prepare_variants(seq_core_df).select(
        ["variant_id", "chrom", "pos", "ref", "alt"] + (["context_id"] if "context_id" in seq_core_df.columns else [])
    ).unique()

    ag_cfg = dict(models_cfg.get("models", {}).get("alphagenome", {}))
    adapter = AlphaGenomeAdapter(config=ag_cfg)

    # Resolve API key from env or config.
    api_key = os.environ.get("ALPHAGENOME_API_KEY") or ag_cfg.get("api_key")
    if not api_key:
        print("[score_alphagenome] WARNING: no API key found (set ALPHAGENOME_API_KEY env var).")
        print("[score_alphagenome] Proceeding with inference attempt (will fail without a key).")

    if not adapter.validate_resources():
        print("[score_alphagenome] WARNING: alphagenome package not available or API key missing.")

    output_dir = Path(args.output).parent / "alphagenome_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"

    contexts = _build_contexts(seq_core_df)
    print(f"[score_alphagenome] {len(contexts)} contexts, {variants_df.height} variants")

    print(f"[score_alphagenome] Running AlphaGenome inference ...")
    df = run_alphagenome_inference(
        variants_df,
        gene_master_df,
        output_dir,
        api_key=api_key or "",
        batch_size=int(ag_cfg.get("batch_size", 50)),
        model_version=adapter.model_version,
        cache_dir=cache_dir,
        contexts=contexts,
        scorer=adapter.scorer,
    )

    # Join back to candidate pairs.
    key_cols = [c for c in ("variant_id", "gene_id", "context_id") if c in df.columns and c in seq_core_df.columns]
    if key_cols:
        joined = seq_core_df.join(df, on=key_cols, how="left")
    else:
        joined = seq_core_df

    missing = pl.col("alphagenome_abs").is_null()
    out = joined.select(
        pl.lit("alphagenome").alias("model_id"),
        pl.lit("sequence").alias("model_family"),
        pl.lit(None, dtype=pl.Utf8).alias("benchmark_id"),
        pl.col("variant_id"),
        pl.lit(None, dtype=pl.Utf8).alias("element_id"),
        pl.col("gene_id"),
        pl.col("context_id") if "context_id" in joined.columns else pl.lit("default").alias("context_id"),
        pl.when(missing).then(0.0).otherwise(pl.col("alphagenome_abs")).alias("raw_score"),
        pl.when(missing).then(0.0).otherwise(pl.col("alphagenome_quantile") if "alphagenome_quantile" in joined.columns else pl.col("alphagenome_abs")).alias("ranking_score"),
        pl.when(missing).then(None).otherwise(pl.col("alphagenome_signed")).alias("signed_score"),
        pl.when(missing).then(0).otherwise(1).cast(pl.Int64).alias("coverage"),
        pl.when(missing).then(pl.lit("NOT_APPLICABLE_MISSING_DATA")).otherwise(pl.lit("APPLICABLE")).alias("applicability"),
        pl.lit("remote_inference").alias("source_mode"),
    )

    write_parquet(out, args.output)
    print(f"[score_alphagenome] Wrote {out.height} prediction rows -> {args.output}")


if __name__ == "__main__":
    main()
