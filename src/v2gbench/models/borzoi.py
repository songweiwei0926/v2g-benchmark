"""Borzoi 4-replicate inference adapter (Family 3 — sequence models).

Borzoi predicts RNA-seq coverage tracks from a 262 kb DNA sequence window.
For each variant we run REF and ALT inference, aggregate the per-exon RNA
signal onto each candidate gene, and compute a **signed** effect score
``ALT - REF`` (and its absolute value).  Borzoi ships four independently
trained replicates; we ensemble them by averaging and report the
cross-replicate standard deviation as an uncertainty estimate.

The heavy GPU inference is expected to run on an HPC node (see the project's
SLURM workflow).  This module provides the Python orchestration: model
loading, per-variant scoring, ensembling, an OOM-aware batch loop and the
:class:`ModelAdapter` glue.

Output columns: ``borzoi_signed``, ``borzoi_abs``, ``borzoi_sd_across_replicates``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

from .base import ModelAdapter

logger = logging.getLogger(__name__)

# Default sequence window Borzoi was trained on.
DEFAULT_SEQUENCE_LENGTH = 262_144
# Replicate identifiers (order matters for SD computation).
DEFAULT_REPLICATES = ["replicate_0", "replicate_1", "replicate_2", "replicate_3"]
# Batch sizes tried in order when an OOM is encountered.
BATCH_SIZE_LADDER = (8, 4, 2, 1)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_borzoi_model(replicate_id: str, weights_dir: Optional[str] = None):
    """Load a borzoi-pytorch model for one replicate.

    Parameters
    ----------
    replicate_id:
        One of ``replicate_0`` … ``replicate_3``.
    weights_dir:
        Directory containing the replicate weights.  If ``None`` the
        borzoi-pytorch default download location is used.

    Returns
    -------
    tuple
        ``(model, predict_fn)`` where ``predict_fn`` maps a one-hot tensor
        to RNA-seq track predictions.
    """
    try:
        import torch  # noqa: F401
        from borzoi_pytorch import Borzoi  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "borzoi-pytorch and torch are required for Borzoi inference. "
            "Install with `pip install borzoi-pytorch torch`."
        ) from exc

    model = Borzoi.from_replicate(replicate_id, weights_dir=weights_dir)
    model.eval()
    return model, model.predict


# --------------------------------------------------------------------------- #
# Per-variant scoring
# --------------------------------------------------------------------------- #
def _one_hot(sequence: str):
    """Encode an ACGT (upper-case) string as a one-hot float tensor [4, L]."""
    import torch
    import numpy as np

    lookup = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, len(sequence)), dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        if base in lookup:
            arr[lookup[base], i] = 1.0
        # N or ambiguous → all-zero column
    return torch.from_numpy(arr)


def _fetch_sequence(chrom: str, pos: int, ref: str, alt: str,
                    sequence_length: int, genome_fasta):
    """Return (ref_seq, alt_seq) centred on the variant.

    ``genome_fasta`` is a pyfaidx / pysam-style object with ``fetch(chrom, start, end)``.
    """
    half = sequence_length // 2
    start = pos - 1 - half  # 0-based
    end = start + sequence_length
    if start < 0:
        raise ValueError(f"Variant {chrom}:{pos} too close to chromosome start for window {sequence_length}")
    ref_window = genome_fasta.fetch(chrom, start, end).upper()
    # Sanity: the REF allele should sit at the centre.
    center = half
    if ref_window[center:center + len(ref)] != ref.upper():
        logger.warning("REF mismatch at %s:%s — proceeding anyway", chrom, pos)
    alt_window = ref_window[:center] + alt.upper() + ref_window[center + len(ref):]
    # Pad/truncate to exact length (indels change length).
    if len(alt_window) > sequence_length:
        alt_window = alt_window[:sequence_length]
    elif len(alt_window) < sequence_length:
        alt_window = alt_window + "N" * (sequence_length - len(alt_window))
    return ref_window, alt_window


def _aggregate_gene_rna(tracks, gene_master, gene_id: str, center_offset: int) -> float:
    """Sum RNA-seq prediction across the gene's exons.

    ``tracks`` is a numpy array of shape ``(L,)`` (single track, single sample)
    at Borzoi's native 32 bp resolution.  ``gene_master`` is a row-dict with
    ``exon_intervals`` (JSON list of [start, end] in genomic coords) and
    ``chrom``/``start``/``end``.
    """
    import numpy as np

    exon_json = gene_master.get("exon_intervals")
    if not exon_json:
        # Fall back to the whole gene body.
        intervals = [[gene_master["start"], gene_master["end"]]]
    else:
        intervals = json.loads(exon_json) if isinstance(exon_json, str) else exon_json

    window_start = gene_master["_window_start"]  # set by caller
    total = 0.0
    for s, e in intervals:
        rel_s = (s - window_start) // 32
        rel_e = (e - window_start) // 32
        rel_s = max(0, rel_s)
        rel_e = min(len(tracks), rel_e)
        if rel_e > rel_s:
            total += float(np.nansum(tracks[rel_s:rel_e]))
    return total


def score_variant_borzoi(model, variant: Dict[str, Any], gene_master_df: pl.DataFrame,
                         sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                         genome_fasta=None) -> pl.DataFrame:
    """Score one variant against all its candidate genes with one replicate.

    Returns a frame with columns ``variant_id, gene_id, ref_rna, alt_rna,
    signed, abs``.

    ``variant`` must contain ``variant_id, chrom, pos, ref, alt``.
    ``genome_fasta`` must be provided (pyfaidx/pysam object).
    """
    import torch
    import numpy as np

    chrom, pos = variant["chrom"], int(variant["pos"])
    ref, alt = variant["ref"], variant["alt"]
    half = sequence_length // 2
    window_start = pos - 1 - half

    ref_seq, alt_seq = _fetch_sequence(chrom, pos, ref, alt, sequence_length, genome_fasta)
    ref_oh = _one_hot(ref_seq).unsqueeze(0)  # [1, 4, L]
    alt_oh = _one_hot(alt_seq).unsqueeze(0)

    predict_fn = getattr(model, "predict", model)
    with torch.no_grad():
        ref_pred = predict_fn(ref_oh).cpu().numpy()  # [1, tracks, L//32]
        alt_pred = predict_fn(alt_oh).cpu().numpy()

    # Borzoi RNA tracks are the human RNA-seq output channels.  We sum across
    # all RNA tracks to get a single per-position signal.
    ref_rna = ref_pred[0].sum(axis=0)  # [L//32]
    alt_rna = alt_pred[0].sum(axis=0)

    genes = gene_master_df.filter(pl.col("chrom") == chrom)
    rows: List[Dict[str, Any]] = []
    for gene in genes.to_dicts():
        gm = {**gene, "_window_start": window_start}
        ref_val = _aggregate_gene_rna(ref_rna, gm, gene["gene_id"], half)
        alt_val = _aggregate_gene_rna(alt_rna, gm, gene["gene_id"], half)
        signed = alt_val - ref_val
        rows.append({
            "variant_id": variant["variant_id"],
            "gene_id": gene["gene_id"],
            "ref_rna": ref_val,
            "alt_rna": alt_val,
            "signed": float(signed),
            "abs": float(abs(signed)),
        })
    return pl.DataFrame(rows, schema={
        "variant_id": pl.Utf8, "gene_id": pl.Utf8,
        "ref_rna": pl.Float64, "alt_rna": pl.Float64,
        "signed": pl.Float64, "abs": pl.Float64,
    }) if rows else pl.DataFrame(schema={
        "variant_id": pl.Utf8, "gene_id": pl.Utf8,
        "ref_rna": pl.Float64, "alt_rna": pl.Float64,
        "signed": pl.Float64, "abs": pl.Float64,
    })


# --------------------------------------------------------------------------- #
# Ensembling
# --------------------------------------------------------------------------- #
def ensemble_borzoi_scores(replicate_scores: List[pl.DataFrame]) -> pl.DataFrame:
    """Average signed/abs scores across replicates and compute SD.

    Each entry of ``replicate_scores`` is the frame returned by
    :func:`score_variant_borzoi` for one replicate.  They must share the
    ``(variant_id, gene_id)`` key.
    """
    if not replicate_scores:
        return pl.DataFrame(schema={
            "variant_id": pl.Utf8, "gene_id": pl.Utf8,
            "borzoi_signed": pl.Float64, "borzoi_abs": pl.Float64,
            "borzoi_sd_across_replicates": pl.Float64,
        })

    stacked = pl.concat(
        [df.select("variant_id", "gene_id", "signed", "abs") for df in replicate_scores],
        how="vertical_relaxed",
    )
    ens = (
        stacked.group_by(["variant_id", "gene_id"])
        .agg(
            pl.col("signed").mean().alias("borzoi_signed"),
            pl.col("abs").mean().alias("borzoi_abs"),
            pl.col("signed").std().alias("borzoi_sd_across_replicates"),
        )
    )
    # std over a single replicate is null → fill 0.
    return ens.with_columns(
        pl.col("borzoi_sd_across_replicates").fill_null(0.0)
    )


# --------------------------------------------------------------------------- #
# Main inference loop
# --------------------------------------------------------------------------- #
def run_borzoi_inference(variants_df: pl.DataFrame, gene_master_df: pl.DataFrame,
                         output_dir: str | Path, batch_size: int = 8,
                         replicates: Optional[List[str]] = None,
                         weights_dir: Optional[str] = None,
                         genome_fasta=None) -> pl.DataFrame:
    """Run Borzoi inference for all variants across all replicates.

    Implements an OOM auto-reduce ladder: if a forward pass raises a CUDA OOM,
    the batch size is halved (8 → 4 → 2 → 1) and the failed batch retried.

    Returns the ensembled frame and writes per-replicate + ensemble parquet
    files under ``output_dir``.
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replicates = replicates or DEFAULT_REPLICATES
    variants = variants_df.to_dicts()

    per_replicate: List[pl.DataFrame] = []
    for rep in replicates:
        logger.info("Borzoi: loading replicate %s", rep)
        model, _ = load_borzoi_model(rep, weights_dir=weights_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        rep_rows: List[pl.DataFrame] = []
        bs = batch_size
        i = 0
        while i < len(variants):
            batch = variants[i:i + bs]
            try:
                for v in batch:
                    df = score_variant_borzoi(
                        model, v, gene_master_df,
                        sequence_length=DEFAULT_SEQUENCE_LENGTH,
                        genome_fasta=genome_fasta,
                    )
                    rep_rows.append(df)
                i += bs
            except torch.cuda.OutOfMemoryError:  # type: ignore[attr-defined]
                torch.cuda.empty_cache()
                next_bs = next((b for b in BATCH_SIZE_LADDER if b < bs), None)
                if next_bs is None:
                    raise
                logger.warning("OOM at batch_size=%d → reducing to %d", bs, next_bs)
                bs = next_bs
                continue

        rep_df = pl.concat(rep_rows, how="vertical_relaxed") if rep_rows else pl.DataFrame()
        rep_df.write_parquet(output_dir / f"borzoi_{rep}.parquet")
        per_replicate.append(rep_df)
        del model
        torch.cuda.empty_cache()

    ensemble = ensemble_borzoi_scores(per_replicate)
    ensemble.write_parquet(output_dir / "borzoi_ensemble.parquet")
    return ensemble


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class BorzoiAdapter(ModelAdapter):
    """ModelAdapter wrapping the Borzoi 4-replicate inference pipeline."""

    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        super().__init__(
            model_id="borzoi",
            model_family="sequence",
            config=config,
        )
        self.replicates: List[str] = list(config.get("replicates", DEFAULT_REPLICATES))
        self.sequence_length = int(config.get("sequence_length", DEFAULT_SEQUENCE_LENGTH))
        self.weights_dir = config.get("weights_dir")

    def validate_resources(self) -> bool:
        try:
            import torch  # noqa: F401
            import borzoi_pytorch  # noqa: F401
        except ImportError:
            return False
        if self.weights_dir and not Path(self.weights_dir).exists():
            return False
        return True

    def applicability(self, context_id: str) -> str:
        # Borzoi RNA tracks are bulk-tissue; applicable to all contexts.
        return "APPLICABLE"

    def score(self, inputs: Dict[str, Any]) -> pl.DataFrame:
        variants_df: pl.DataFrame = inputs["variants_df"]
        gene_master_df: pl.DataFrame = inputs["gene_master_df"]
        output_dir = inputs.get("output_dir", "./predictions/borzoi")
        genome_fasta = inputs.get("genome_fasta")
        batch_size = int(self.config.get("batch_size", 8))

        ensemble = run_borzoi_inference(
            variants_df, gene_master_df, output_dir,
            batch_size=batch_size, replicates=self.replicates,
            weights_dir=self.weights_dir, genome_fasta=genome_fasta,
        )
        return self.normalize_score(ensemble)

    def normalize_score(self, df: pl.DataFrame) -> pl.DataFrame:
        # ranking_score = |signed| (magnitude of predicted effect).
        if "borzoi_abs" in df.columns:
            df = df.with_columns(pl.col("borzoi_abs").alias("ranking_score"))
        elif "abs" in df.columns:
            df = df.with_columns(pl.col("abs").alias("ranking_score"))
        return df
