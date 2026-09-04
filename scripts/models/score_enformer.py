#!/usr/bin/env python3
"""Run Enformer CAGE-track inference.

Enformer predicts 5,313 human & mouse tracks from a ~115 kb input window at
128 bp resolution. For variant scoring we use the human CAGE tracks, which
directly report TSS activity. For each candidate gene we extract the CAGE
signal in a +/-1 kb window around the gene's TSS for both REF and ALT
sequences and compute a signed effect (ALT - REF).

Output columns: enformer_signed, enformer_abs.

CLI (Snakemake interface)
-------------------------
    score_enformer.py
        --seq-core <parquet> --gene-master <parquet> --fasta <path> --output <parquet>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import polars as pl

from v2gbench.io.parquet import read_parquet, write_parquet
from v2gbench.models.enformer import EnformerAdapter, run_enformer_inference
from v2gbench.schemas.variant import parse_variant_id
from v2gbench.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("score_enformer")


def _find_root() -> Path:
    cur = Path.cwd()
    while cur != cur.parent:
        if (cur / "config" / "project.yaml").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def _load_fasta(fasta_path: str):
    try:
        import pyfaidx
    except ImportError as exc:
        raise ImportError(
            "pyfaidx is required for sequence extraction. Install with `pip install pyfaidx`."
        ) from exc
    return pyfaidx.Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Enformer CAGE-track inference on the SEQ_CORE subset."
    )
    parser.add_argument("--seq-core", required=True, help="SEQ_CORE parquet path")
    parser.add_argument("--gene-master", required=True, help="GENCODE gene master parquet path")
    parser.add_argument("--fasta", required=True, help="GRCh38 FASTA path")
    parser.add_argument("--output", required=True, help="Output predictions parquet path")
    args = parser.parse_args()

    root = _find_root()
    project_cfg = load_config(root / "config" / "project.yaml") if (root / "config" / "project.yaml").exists() else {}
    models_cfg = load_config(root / "config" / "models.yaml") if (root / "config" / "models.yaml").exists() else {}

    seq_core_df = read_parquet(args.seq_core)
    gene_master_df = read_parquet(args.gene_master)
    print(f"[score_enformer] seq_core={seq_core_df.height} genes={gene_master_df.height}")

    variants_df = _prepare_variants(seq_core_df).select(
        ["variant_id", "chrom", "pos", "ref", "alt"] + (["context_id"] if "context_id" in seq_core_df.columns else [])
    ).unique()

    print(f"[score_enformer] Loading FASTA: {args.fasta}")
    genome_fasta = _load_fasta(args.fasta)

    enformer_cfg = dict(models_cfg.get("models", {}).get("enformer", {}))
    adapter = EnformerAdapter(config=enformer_cfg)

    if not adapter.validate_resources():
        print("[score_enformer] WARNING: enformer-pytorch/torch not available.")
        print("[score_enformer] Proceeding with inference attempt (will fail if deps absent).")

    output_dir = Path(args.output).parent / "enformer_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load track metadata if configured.
    track_metadata = None
    tm_path = enformer_cfg.get("track_metadata_path")
    if tm_path and Path(tm_path).exists():
        track_metadata = pl.read_parquet(tm_path)
        print(f"[score_enformer] Loaded track metadata: {track_metadata.height} tracks")

    print(f"[score_enformer] Running Enformer inference over {variants_df.height} variants ...")
    df = run_enformer_inference(
        variants_df,
        gene_master_df,
        output_dir,
        batch_size=int(enformer_cfg.get("batch_size", 8)),
        track_metadata=track_metadata,
        genome_fasta=genome_fasta,
    )

    # Join back to candidate pairs.
    key_cols = [c for c in ("variant_id", "gene_id") if c in df.columns and c in seq_core_df.columns]
    if key_cols:
        joined = seq_core_df.join(df, on=key_cols, how="left")
    else:
        joined = seq_core_df

    missing = pl.col("enformer_abs").is_null()
    out = joined.select(
        pl.lit("enformer").alias("model_id"),
        pl.lit("sequence").alias("model_family"),
        pl.lit(None, dtype=pl.Utf8).alias("benchmark_id"),
        pl.col("variant_id"),
        pl.lit(None, dtype=pl.Utf8).alias("element_id"),
        pl.col("gene_id"),
        pl.col("context_id") if "context_id" in joined.columns else pl.lit("default").alias("context_id"),
        pl.when(missing).then(0.0).otherwise(pl.col("enformer_abs")).alias("raw_score"),
        pl.when(missing).then(0.0).otherwise(pl.col("enformer_abs")).alias("ranking_score"),
        pl.when(missing).then(None).otherwise(pl.col("enformer_signed")).alias("signed_score"),
        pl.when(missing).then(0).otherwise(1).cast(pl.Int64).alias("coverage"),
        pl.when(missing).then(pl.lit("NOT_APPLICABLE_MISSING_DATA")).otherwise(pl.lit("APPLICABLE")).alias("applicability"),
        pl.lit("local_inference").alias("source_mode"),
    )

    write_parquet(out, args.output)
    print(f"[score_enformer] Wrote {out.height} prediction rows -> {args.output}")


if __name__ == "__main__":
    main()
