#!/usr/bin/env python3
"""Run Borzoi 4-replicate inference.

Borzoi predicts RNA-seq coverage tracks from a 262 kb DNA sequence window.
For each variant we run REF and ALT inference across four independently
trained replicates, aggregate per-exon RNA signal onto candidate genes, and
compute a signed effect (ALT - REF) and its absolute value. Replicates are
ensembled by averaging; cross-replicate SD is reported as uncertainty.

Output columns: borzoi_signed, borzoi_abs, borzoi_sd_across_replicates.

CLI (Snakemake interface)
-------------------------
    score_borzoi.py
        --seq-core <parquet> --gene-master <parquet> --fasta <path> --output <parquet>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import polars as pl

from v2gbench.io.parquet import read_parquet, write_parquet
from v2gbench.models.borzoi import BorzoiAdapter, run_borzoi_inference
from v2gbench.schemas.variant import parse_variant_id
from v2gbench.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("score_borzoi")


def _find_root() -> Path:
    cur = Path.cwd()
    while cur != cur.parent:
        if (cur / "config" / "project.yaml").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def _load_fasta(fasta_path: str):
    """Load a genome FASTA via pyfaidx (lazy import)."""
    try:
        import pyfaidx
    except ImportError as exc:
        raise ImportError(
            "pyfaidx is required for sequence extraction. Install with `pip install pyfaidx`."
        ) from exc
    return pyfaidx.Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)


def _prepare_variants(seq_core_df: pl.DataFrame) -> pl.DataFrame:
    """Ensure the SEQ_CORE frame has chrom/pos/ref/alt columns for inference."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Borzoi 4-replicate inference on the SEQ_CORE subset."
    )
    parser.add_argument("--seq-core", required=True, help="SEQ_CORE parquet path")
    parser.add_argument("--gene-master", required=True, help="GENCODE gene master parquet path")
    parser.add_argument("--fasta", required=True, help="GRCh38 FASTA path")
    parser.add_argument("--output", required=True, help="Output predictions parquet path")
    args = parser.parse_args()

    root = _find_root()
    project_cfg = load_config(root / "config" / "project.yaml") if (root / "config" / "project.yaml").exists() else {}
    models_cfg = load_config(root / "config" / "models.yaml") if (root / "config" / "models.yaml").exists() else {}
    seed = int(project_cfg.get("project", {}).get("seed", 20260904))

    seq_core_df = read_parquet(args.seq_core)
    gene_master_df = read_parquet(args.gene_master)
    print(f"[score_borzoi] seq_core={seq_core_df.height} genes={gene_master_df.height}")

    # Prepare variants with chrom/pos/ref/alt.
    variants_df = _prepare_variants(seq_core_df).select(
        ["variant_id", "chrom", "pos", "ref", "alt"] + (["context_id"] if "context_id" in seq_core_df.columns else [])
    ).unique()

    # Load genome FASTA.
    print(f"[score_borzoi] Loading FASTA: {args.fasta}")
    genome_fasta = _load_fasta(args.fasta)

    # Configure adapter from models.yaml.
    borzoi_cfg = dict(models_cfg.get("models", {}).get("borzoi", {}))
    adapter = BorzoiAdapter(config=borzoi_cfg)

    if not adapter.validate_resources():
        print("[score_borzoi] WARNING: borzoi-pytorch/torch not available or weights missing.")
        print("[score_borzoi] Proceeding with inference attempt (will fail if deps absent).")

    output_dir = Path(args.output).parent / "borzoi_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[score_borzoi] Running {len(adapter.replicates)} replicates over {variants_df.height} variants ...")
    ensemble = run_borzoi_inference(
        variants_df,
        gene_master_df,
        output_dir,
        batch_size=int(borzoi_cfg.get("batch_size", 8)),
        replicates=adapter.replicates,
        weights_dir=borzoi_cfg.get("weights_dir"),
        genome_fasta=genome_fasta,
    )

    # Attach candidate-pair context and assemble the prediction-schema frame.
    # Join ensemble scores back to the SEQ_CORE candidate pairs.
    key_cols = [c for c in ("variant_id", "gene_id") if c in ensemble.columns and c in seq_core_df.columns]
    if key_cols:
        joined = seq_core_df.join(ensemble, on=key_cols, how="left")
    else:
        joined = seq_core_df

    missing = pl.col("borzoi_abs").is_null()
    out = joined.select(
        pl.lit("borzoi").alias("model_id"),
        pl.lit("sequence").alias("model_family"),
        pl.lit(None, dtype=pl.Utf8).alias("benchmark_id"),
        pl.col("variant_id"),
        pl.lit(None, dtype=pl.Utf8).alias("element_id"),
        pl.col("gene_id"),
        pl.col("context_id") if "context_id" in joined.columns else pl.lit("default").alias("context_id"),
        pl.when(missing).then(0.0).otherwise(pl.col("borzoi_abs")).alias("raw_score"),
        pl.when(missing).then(0.0).otherwise(pl.col("borzoi_abs")).alias("ranking_score"),
        pl.when(missing).then(None).otherwise(pl.col("borzoi_signed")).alias("signed_score"),
        pl.when(missing).then(0).otherwise(1).cast(pl.Int64).alias("coverage"),
        pl.when(missing).then(pl.lit("NOT_APPLICABLE_MISSING_DATA")).otherwise(pl.lit("APPLICABLE")).alias("applicability"),
        pl.lit("local_inference").alias("source_mode"),
    )

    # Add SD column if present.
    if "borzoi_sd_across_replicates" in ensemble.columns:
        out = out.with_columns(
            pl.when(missing).then(None).otherwise(pl.col("borzoi_sd_across_replicates")).alias("borzoi_sd_across_replicates")
        )

    write_parquet(out, args.output)
    print(f"[score_borzoi] Wrote {out.height} prediction rows -> {args.output}")


if __name__ == "__main__":
    main()
