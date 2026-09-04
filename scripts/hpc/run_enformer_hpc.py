#!/usr/bin/env python3
"""HPC GPU script for Enformer inference.

This script runs inside the regvelo HPC container (PyTorch 2.5.1 + CUDA).
It reads seq_core, gene_master, and GRCh38 FASTA from /input,
runs Enformer inference on GPU, and writes predictions to /output.

Enformer outputs CAGE:Lung tracks at TSS±1kb for each gene.
Output columns: enformer_signed, enformer_abs.

Usage (via hpc_run_tool):
    hpc_run_tool('regvelo', 'python /input/run_enformer_hpc.py', input_files)
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
                       "enformer-pytorch", "pyfaidx", "polars", "pyarrow"])

import numpy as np
import polars as pl
import torch

INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Enformer parameters
SEQUENCE_LENGTH = 393_216  # Enformer native input length
PREDICTION_LENGTH = 128 * 896  # 114,688 bp at 128 bp resolution
BATCH_SIZE_LADDER = [8, 4, 2, 1]
CAGE_WINDOW = 1000  # TSS ± 1kb for CAGE track aggregation


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


def aggregate_cage_tss(tracks: np.ndarray, gene_row: Dict,
                       window_start: int, resolution: int = 128) -> float:
    """Aggregate CAGE track predictions at TSS ± 1kb.

    Enformer outputs 5313 tracks at 128bp resolution.
    CAGE tracks are tissue-specific; we use a generic CAGE track.
    The track index for CAGE:Lung is 5112 (Enformer track ordering).
    If that track is unavailable, we sum across all CAGE-like tracks.
    """
    tss = gene_row.get("start", 0)
    if gene_row.get("strand", "+") == "-":
        tss = gene_row.get("end", 0)

    # Convert genomic coordinates to prediction bin indices
    rel_start = (tss - CAGE_WINDOW // 2 - window_start) // resolution
    rel_end = (tss + CAGE_WINDOW // 2 - window_start) // resolution
    rel_start = max(0, rel_start)
    rel_end = min(len(tracks), rel_end)

    if rel_end <= rel_start:
        return 0.0

    # Sum CAGE track signal in the TSS ± 1kb window
    # Enformer track 5112 = CAGE:Lung (commonly used for eQTL benchmarking)
    # We use a set of CAGE tracks for robustness
    cage_track_indices = list(range(5110, 5120))  # CAGE tracks around index 5112
    cage_track_indices = [i for i in cage_track_indices if i < tracks.shape[1]] if tracks.ndim > 1 else []

    if cage_track_indices and tracks.ndim > 1:
        # tracks shape: [n_bins, n_tracks] or [n_tracks, n_bins]
        if tracks.shape[0] < tracks.shape[1]:
            # [n_bins, n_tracks]
            total = float(np.nansum(tracks[rel_start:rel_end, cage_track_indices]))
        else:
            # [n_tracks, n_bins]
            total = float(np.nansum(tracks[cage_track_indices, rel_start:rel_end]))
    else:
        # Fallback: sum all tracks in the window
        total = float(np.nansum(tracks[rel_start:rel_end]))

    return total


def run_enformer(variants_df: pl.DataFrame,
                 gene_master_df: pl.DataFrame,
                 genome_fasta,
                 batch_size: int = 8) -> pl.DataFrame:
    """Run Enformer on all variants."""
    from enformer_pytorch import Enformer

    print("Loading Enformer...", file=sys.stderr)
    model = Enformer.from_pretrained("enformer-official-elongated")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}", file=sys.stderr)

    all_rows: List[Dict[str, Any]] = []
    variants = variants_df.to_dicts()
    n = len(variants)

    current_batch = batch_size
    for bs in BATCH_SIZE_LADDER:
        if bs <= batch_size:
            current_batch = bs
            break

    print(f"Starting inference with batch_size={current_batch}", file=sys.stderr)

    i = 0
    while i < n:
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
            i += current_batch
            continue

        try:
            ref_oh = torch.stack([one_hot(s) for s in ref_seqs]).to(device)
            alt_oh = torch.stack([one_hot(s) for s in alt_seqs]).to(device)

            with torch.no_grad():
                ref_pred = model(ref_oh).cpu().numpy()
                alt_pred = model(alt_oh).cpu().numpy()

            for j, v in enumerate(valid):
                ref_tracks = ref_pred[j]
                alt_tracks = alt_pred[j]
                window_start = v["pos"] - 1 - SEQUENCE_LENGTH // 2

                genes = gene_master_df.filter(pl.col("chrom") == v["chrom"])
                for gene in genes.to_dicts():
                    ref_val = aggregate_cage_tss(ref_tracks, gene, window_start)
                    alt_val = aggregate_cage_tss(alt_tracks, gene, window_start)
                    signed = alt_val - ref_val
                    all_rows.append({
                        "variant_id": v["variant_id"],
                        "gene_id": gene["gene_id"],
                        "ref_cage": ref_val,
                        "alt_cage": alt_val,
                        "signed": float(signed),
                        "abs": float(abs(signed)),
                    })

            i += current_batch
            print(f"  Processed {min(i, n)}/{n} variants", file=sys.stderr)

        except torch.cuda.OutOfMemoryError:
            print(f"OOM at batch_size={current_batch}, reducing...", file=sys.stderr)
            torch.cuda.empty_cache()
            smaller = max(1, current_batch // 2)
            if smaller < current_batch:
                current_batch = smaller
                # Don't advance i — retry this batch with smaller size
                continue
            else:
                # Already at batch_size=1, skip this variant
                print(f"  OOM even at batch_size=1, skipping variants", file=sys.stderr)
                i += 1
                continue
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM (RuntimeError) at batch_size={current_batch}, reducing...", file=sys.stderr)
                torch.cuda.empty_cache()
                smaller = max(1, current_batch // 2)
                if smaller < current_batch:
                    current_batch = smaller
                    continue
                else:
                    i += 1
                    continue
            raise

    schema = {
        "variant_id": pl.Utf8, "gene_id": pl.Utf8,
        "ref_cage": pl.Float64, "alt_cage": pl.Float64,
        "signed": pl.Float64, "abs": pl.Float64,
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

    # Run Enformer
    predictions_df = run_enformer(variants_df, gene_master_df, genome_fasta)
    print(f"Total predictions: {predictions_df.height}", file=sys.stderr)

    # Join back to seq_core
    key_cols = [c for c in ("variant_id", "gene_id") if c in predictions_df.columns and c in seq_core_df.columns]
    if key_cols:
        joined = seq_core_df.join(predictions_df, on=key_cols, how="left")
    else:
        joined = seq_core_df

    missing = pl.col("abs").is_null()
    out = joined.select(
        pl.lit("enformer").alias("model_id"),
        pl.lit("sequence").alias("model_family"),
        pl.lit(None, dtype=pl.Utf8).alias("benchmark_id"),
        pl.col("variant_id"),
        pl.lit(None, dtype=pl.Utf8).alias("element_id"),
        pl.col("gene_id"),
        pl.col("context_id") if "context_id" in joined.columns else pl.lit("default").alias("context_id"),
        pl.when(missing).then(0.0).otherwise(pl.col("abs")).alias("raw_score"),
        pl.when(missing).then(0.0).otherwise(pl.col("abs")).alias("ranking_score"),
        pl.when(missing).then(None).otherwise(pl.col("signed")).alias("signed_score"),
        pl.when(missing).then(0).otherwise(1).cast(pl.Int64).alias("coverage"),
        pl.when(missing).then(pl.lit("NOT_APPLICABLE_MISSING_DATA")).otherwise(pl.lit("APPLICABLE")).alias("applicability"),
        pl.lit("hpc_inference").alias("source_mode"),
    )

    output_path = OUTPUT_DIR / "enformer_predictions.parquet"
    out.write_parquet(output_path)
    print(f"Wrote {out.height} predictions -> {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
