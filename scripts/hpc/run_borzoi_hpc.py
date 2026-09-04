#!/usr/bin/env python3
"""HPC GPU script for Borzoi 4-replicate inference.

This script runs inside the regvelo HPC container (PyTorch 2.5.1 + CUDA).
It reads seq_core, gene_master, and GRCh38 FASTA from /input,
runs Borzoi inference on GPU, and writes predictions to /output.

Usage (via hpc_run_tool):
    hpc_run_tool('regvelo', 'python /input/run_borzoi_hpc.py', input_files)
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List

# Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "borzoi-pytorch", "pyfaidx", "polars", "pyarrow"])

import numpy as np
import polars as pl
import torch

INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Borzoi parameters
SEQUENCE_LENGTH = 262_144
REPLICATES = ["replicate_0", "replicate_1", "replicate_2", "replicate_3"]
BATCH_SIZE_LADDER = [8, 4, 2, 1]


def parse_variant_id(vid: str) -> Dict[str, Any]:
    """Parse GRCh38:chr1:123456:A:G format."""
    parts = vid.split(":")
    return {
        "genome_build": parts[0],
        "chrom": parts[1],
        "pos": int(parts[2]),
        "ref": parts[3],
        "alt": parts[4],
    }


def one_hot(sequence: str) -> torch.Tensor:
    """Encode ACGT string as one-hot tensor [4, L]."""
    lookup = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, len(sequence)), dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        if base in lookup:
            arr[lookup[base], i] = 1.0
    return torch.from_numpy(arr)


def fetch_sequence(chrom: str, pos: int, ref: str, alt: str,
                   sequence_length: int, genome_fasta) -> tuple[str, str]:
    """Return (ref_seq, alt_seq) centered on the variant."""
    half = sequence_length // 2
    start = pos - 1 - half
    end = start + sequence_length
    if start < 0:
        raise ValueError(f"Variant {chrom}:{pos} too close to chromosome start")
    ref_window = genome_fasta.fetch(chrom, start, end).upper()
    center = half
    if ref_window[center:center + len(ref)] != ref.upper():
        print(f"WARNING: REF mismatch at {chrom}:{pos}", file=sys.stderr)
    alt_window = ref_window[:center] + alt.upper() + ref_window[center + len(ref):]
    if len(alt_window) > sequence_length:
        alt_window = alt_window[:sequence_length]
    elif len(alt_window) < sequence_length:
        alt_window = alt_window + "N" * (sequence_length - len(alt_window))
    return ref_window, alt_window


def aggregate_gene_rna(tracks: np.ndarray, gene_row: Dict, window_start: int) -> float:
    """Sum RNA-seq prediction across gene exons."""
    exon_json = gene_row.get("exon_intervals")
    if not exon_json:
        intervals = [[gene_row["start"], gene_row["end"]]]
    else:
        intervals = json.loads(exon_json) if isinstance(exon_json, str) else exon_json

    total = 0.0
    for s, e in intervals:
        rel_s = (s - window_start) // 32
        rel_e = (e - window_start) // 32
        rel_s = max(0, rel_s)
        rel_e = min(len(tracks), rel_e)
        if rel_e > rel_s:
            total += float(np.nansum(tracks[rel_s:rel_e]))
    return total


def run_replicate(replicate_id: str, variants_df: pl.DataFrame,
                  gene_master_df: pl.DataFrame, genome_fasta,
                  batch_size: int = 8) -> pl.DataFrame:
    """Run one Borzoi replicate on all variants."""
    from borzoi_pytorch import Borzoi

    print(f"Loading Borzoi {replicate_id}...", file=sys.stderr)
    model = Borzoi.from_replicate(replicate_id)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}", file=sys.stderr)

    all_rows: List[Dict[str, Any]] = []
    variants = variants_df.to_dicts()
    n = len(variants)

    # Try batch sizes from the ladder
    current_batch = batch_size
    for bs in BATCH_SIZE_LADDER:
        if bs <= batch_size:
            current_batch = bs
            break

    print(f"Starting inference with batch_size={current_batch}", file=sys.stderr)

    for i in range(0, n, current_batch):
        batch = variants[i:i + current_batch]
        ref_seqs = []
        alt_seqs = []
        valid = []

        for v in batch:
            try:
                ref_seq, alt_seq = fetch_sequence(
                    v["chrom"], v["pos"], v["ref"], v["alt"],
                    SEQUENCE_LENGTH, genome_fasta
                )
                ref_seqs.append(ref_seq)
                alt_seqs.append(alt_seq)
                valid.append(v)
            except Exception as e:
                print(f"Skipping variant {v['variant_id']}: {e}", file=sys.stderr)

        if not valid:
            continue

        try:
            ref_oh = torch.stack([one_hot(s) for s in ref_seqs]).to(device)
            alt_oh = torch.stack([one_hot(s) for s in alt_seqs]).to(device)

            with torch.no_grad():
                ref_pred = model.predict(ref_oh).cpu().numpy()
                alt_pred = model.predict(alt_oh).cpu().numpy()

            for j, v in enumerate(valid):
                ref_rna = ref_pred[j].sum(axis=0)
                alt_rna = alt_pred[j].sum(axis=0)
                window_start = v["pos"] - 1 - SEQUENCE_LENGTH // 2

                genes = gene_master_df.filter(pl.col("chrom") == v["chrom"])
                for gene in genes.to_dicts():
                    ref_val = aggregate_gene_rna(ref_rna, gene, window_start)
                    alt_val = aggregate_gene_rna(alt_rna, gene, window_start)
                    signed = alt_val - ref_val
                    all_rows.append({
                        "variant_id": v["variant_id"],
                        "gene_id": gene["gene_id"],
                        "ref_rna": ref_val,
                        "alt_rna": alt_val,
                        "signed": float(signed),
                        "abs": float(abs(signed)),
                        "replicate": replicate_id,
                    })

            print(f"  Processed {min(i + current_batch, n)}/{n} variants", file=sys.stderr)

        except torch.cuda.OutOfMemoryError:
            print(f"OOM at batch_size={current_batch}, reducing...", file=sys.stderr)
            torch.cuda.empty_cache()
            # Retry with smaller batch
            smaller = max(1, current_batch // 2)
            if smaller < current_batch:
                current_batch = smaller
                # Re-process this batch with smaller size
                for v in valid:
                    try:
                        ref_seq, alt_seq = fetch_sequence(
                            v["chrom"], v["pos"], v["ref"], v["alt"],
                            SEQUENCE_LENGTH, genome_fasta
                        )
                        ref_oh = one_hot(ref_seq).unsqueeze(0).to(device)
                        alt_oh = one_hot(alt_seq).unsqueeze(0).to(device)
                        with torch.no_grad():
                            ref_pred = model.predict(ref_oh).cpu().numpy()
                            alt_pred = model.predict(alt_oh).cpu().numpy()

                        ref_rna = ref_pred[0].sum(axis=0)
                        alt_rna = alt_pred[0].sum(axis=0)
                        window_start = v["pos"] - 1 - SEQUENCE_LENGTH // 2

                        genes = gene_master_df.filter(pl.col("chrom") == v["chrom"])
                        for gene in genes.to_dicts():
                            ref_val = aggregate_gene_rna(ref_rna, gene, window_start)
                            alt_val = aggregate_gene_rna(alt_rna, gene, window_start)
                            signed = alt_val - ref_val
                            all_rows.append({
                                "variant_id": v["variant_id"],
                                "gene_id": gene["gene_id"],
                                "ref_rna": ref_val,
                                "alt_rna": alt_val,
                                "signed": float(signed),
                                "abs": float(abs(signed)),
                                "replicate": replicate_id,
                            })
                    except Exception as e:
                        print(f"  Failed variant {v['variant_id']}: {e}", file=sys.stderr)
                current_batch = smaller
            continue

    schema = {
        "variant_id": pl.Utf8, "gene_id": pl.Utf8,
        "ref_rna": pl.Float64, "alt_rna": pl.Float64,
        "signed": pl.Float64, "abs": pl.Float64,
        "replicate": pl.Utf8,
    }
    return pl.DataFrame(all_rows, schema=schema) if all_rows else pl.DataFrame(schema=schema)


def main():
    # Load inputs
    seq_core_path = INPUT_DIR / "seq_core.parquet"
    gene_master_path = INPUT_DIR / "gene_master.parquet"
    fasta_path = INPUT_DIR / "GRCh38.fa"

    if not seq_core_path.exists():
        print(f"ERROR: {seq_core_path} not found", file=sys.stderr)
        sys.exit(1)

    seq_core_df = pl.read_parquet(seq_core_path)
    gene_master_df = pl.read_parquet(gene_master_path)

    print(f"seq_core: {seq_core_df.height} rows", file=sys.stderr)
    print(f"gene_master: {gene_master_df.height} rows", file=sys.stderr)

    # Parse variant IDs
    variants_df = seq_core_df.with_columns(
        pl.col("variant_id").map_elements(
            lambda v: parse_variant_id(v)["chrom"], return_dtype=pl.Utf8
        ).alias("chrom"),
        pl.col("variant_id").map_elements(
            lambda v: parse_variant_id(v)["pos"], return_dtype=pl.Int64
        ).alias("pos"),
        pl.col("variant_id").map_elements(
            lambda v: parse_variant_id(v)["ref"], return_dtype=pl.Utf8
        ).alias("ref"),
        pl.col("variant_id").map_elements(
            lambda v: parse_variant_id(v)["alt"], return_dtype=pl.Utf8
        ).alias("alt"),
    ).select(["variant_id", "chrom", "pos", "ref", "alt"] +
             (["context_id"] if "context_id" in seq_core_df.columns else [])).unique()

    print(f"unique variants: {variants_df.height}", file=sys.stderr)

    # Load genome FASTA
    import pyfaidx
    genome_fasta = pyfaidx.Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)

    # Run all replicates
    all_replicate_dfs = []
    for rep in REPLICATES:
        rep_df = run_replicate(rep, variants_df, gene_master_df, genome_fasta)
        all_replicate_dfs.append(rep_df)
        print(f"  {rep}: {rep_df.height} rows", file=sys.stderr)

    # Ensemble: average across replicates
    combined = pl.concat(all_replicate_dfs)
    ensemble = (
        combined
        .group_by(["variant_id", "gene_id"])
        .agg(
            pl.col("signed").mean().alias("borzoi_signed"),
            pl.col("abs").mean().alias("borzoi_abs"),
            pl.col("signed").std().alias("borzoi_sd_across_replicates"),
        )
    )

    # Join back to seq_core
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
        pl.lit("hpc_inference").alias("source_mode"),
    )

    if "borzoi_sd_across_replicates" in joined.columns:
        out = out.with_columns(
            pl.when(missing).then(None).otherwise(pl.col("borzoi_sd_across_replicates")).alias("borzoi_sd_across_replicates")
        )

    output_path = OUTPUT_DIR / "borzoi_predictions.parquet"
    out.write_parquet(output_path)
    print(f"Wrote {out.height} predictions -> {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
