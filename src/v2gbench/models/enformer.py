"""Enformer CAGE-track inference adapter (Family 3 — sequence models).

Enformer predicts 5,313 human & mouse tracks from a ~115 kb input window at
128 bp resolution.  For variant scoring we use the **human CAGE** tracks,
which directly report TSS activity.  For each candidate gene we extract the
CAGE signal in a ±1 kb window around the gene's TSS for both REF and ALT
sequences and compute a signed effect ``ALT - REF``.

Because CAGE tracks are context-specific (cell line / tissue), we match each
benchmark context to the most appropriate Enformer CAGE track(s) via
:func:`match_cage_tracks`.

Output columns: ``enformer_signed``, ``enformer_abs``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import polars as pl

from .base import ModelAdapter

logger = logging.getLogger(__name__)

# Enformer input length (114,688 bp) and output bin resolution (128 bp).
ENFORMER_INPUT_LENGTH = 114_688
ENFORMER_BIN_SIZE = 128
# Half-window around TSS for CAGE extraction (±1 kb → ±8 bins).
TSS_HALF_WINDOW = 1_000
# Number of human tracks Enformer predicts (5,313 total, ~1,962 human CAGE-ish).
N_HUMAN_TRACKS = 5_313


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_enformer_model():
    """Load the enformer-pytorch model (pretrained weights).

    Returns a model callable that maps a one-hot tensor ``[B, 4, L]`` to a
    track tensor ``[B, tracks, L//128]``.
    """
    try:
        import torch  # noqa: F401
        from enformer_pytorch import Enformer  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "enformer-pytorch and torch are required for Enformer inference. "
            "Install with `pip install enformer-pytorch torch`."
        ) from exc

    model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# CAGE track matching
# --------------------------------------------------------------------------- #
def match_cage_tracks(context_id: str, track_metadata: pl.DataFrame) -> List[int]:
    """Find the Enformer human CAGE track indices matching a benchmark context.

    ``track_metadata`` must contain at least ``track_index``, ``species``
    (``"human"``), ``description`` and ideally ``ontology_id``.  Matching is
    case-insensitive on the context name; if no exact match is found, all
    human CAGE tracks are returned (the caller may then average over them).
    """
    if track_metadata is None or len(track_metadata) == 0:
        return []

    human = track_metadata.filter(pl.col("species") == "human") if "species" in track_metadata.columns else track_metadata

    # Try ontology id match first.
    if "ontology_id" in human.columns:
        ont = human.filter(pl.col("ontology_id") == context_id)
        if len(ont) > 0:
            return ont["track_index"].to_list()

    # Fall back to description substring match.
    needle = context_id.lower().replace("_", " ").replace("-", " ")
    if "description" in human.columns:
        desc_match = human.filter(
            pl.col("description").str.to_lowercase().str.contains(needle)
        )
        if len(desc_match) > 0:
            return desc_match["track_index"].to_list()

    # Final fallback: all human CAGE tracks.
    if "track_type" in human.columns:
        cage = human.filter(pl.col("track_type").str.to_lowercase().str.contains("cage"))
        if len(cage) > 0:
            return cage["track_index"].to_list()

    return human["track_index"].to_list() if "track_index" in human.columns else []


# --------------------------------------------------------------------------- #
# Per-variant scoring
# --------------------------------------------------------------------------- #
def _one_hot(sequence: str):
    import torch
    import numpy as np
    lookup = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, len(sequence)), dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        if base in lookup:
            arr[lookup[base], i] = 1.0
    return torch.from_numpy(arr)


def _fetch_sequence(chrom: str, pos: int, ref: str, alt: str,
                    sequence_length: int, genome_fasta):
    half = sequence_length // 2
    start = pos - 1 - half
    end = start + sequence_length
    if start < 0:
        raise ValueError(f"Variant {chrom}:{pos} too close to start for window {sequence_length}")
    ref_window = genome_fasta.fetch(chrom, start, end).upper()
    center = half
    if ref_window[center:center + len(ref)] != ref.upper():
        logger.warning("REF mismatch at %s:%s", chrom, pos)
    alt_window = ref_window[:center] + alt.upper() + ref_window[center + len(ref):]
    if len(alt_window) > sequence_length:
        alt_window = alt_window[:sequence_length]
    elif len(alt_window) < sequence_length:
        alt_window = alt_window + "N" * (sequence_length - len(alt_window))
    return ref_window, alt_window, start


def _extract_tss_cage(tracks, tss_genomic: int, window_start: int,
                      bin_size: int = ENFORMER_BIN_SIZE,
                      half_window: int = TSS_HALF_WINDOW) -> float:
    """Sum CAGE signal in ±half_window around the TSS."""
    import numpy as np
    rel_start = (tss_genomic - window_start - half_window) // bin_size
    rel_end = (tss_genomic - window_start + half_window) // bin_size
    rel_start = max(0, rel_start)
    rel_end = min(tracks.shape[-1], rel_end)
    if rel_end <= rel_start:
        return 0.0
    return float(np.nansum(tracks[..., rel_start:rel_end]))


def score_variant_enformer(model, variant: Dict[str, Any], gene_master_df: pl.DataFrame,
                           cage_track_indices: Sequence[int],
                           sequence_length: int = ENFORMER_INPUT_LENGTH,
                           genome_fasta=None) -> pl.DataFrame:
    """Score one variant against all candidate genes on the same chromosome.

    Returns a frame with ``variant_id, gene_id, enformer_signed, enformer_abs``.
    """
    import torch

    chrom, pos = variant["chrom"], int(variant["pos"])
    ref, alt = variant["ref"], variant["alt"]
    ref_seq, alt_seq, window_start = _fetch_sequence(
        chrom, pos, ref, alt, sequence_length, genome_fasta
    )
    ref_oh = _one_hot(ref_seq).unsqueeze(0)
    alt_oh = _one_hot(alt_seq).unsqueeze(0)

    with torch.no_grad():
        ref_pred = model(ref_oh).cpu().numpy()  # [1, tracks, bins]
        alt_pred = model(alt_oh).cpu().numpy()

    tracks = list(cage_track_indices) or list(range(ref_pred.shape[1]))
    ref_cage = ref_pred[0, tracks, :].sum(axis=0)  # [bins]
    alt_cage = alt_pred[0, tracks, :].sum(axis=0)

    genes = gene_master_df.filter(pl.col("chrom") == chrom)
    rows: List[Dict[str, Any]] = []
    for gene in genes.to_dicts():
        tss = int(gene["tss"])
        ref_val = _extract_tss_cage(ref_cage, tss, window_start)
        alt_val = _extract_tss_cage(alt_cage, tss, window_start)
        signed = alt_val - ref_val
        rows.append({
            "variant_id": variant["variant_id"],
            "gene_id": gene["gene_id"],
            "enformer_signed": float(signed),
            "enformer_abs": float(abs(signed)),
        })
    schema = {"variant_id": pl.Utf8, "gene_id": pl.Utf8,
              "enformer_signed": pl.Float64, "enformer_abs": pl.Float64}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


# --------------------------------------------------------------------------- #
# Main inference loop
# --------------------------------------------------------------------------- #
def run_enformer_inference(variants_df: pl.DataFrame, gene_master_df: pl.DataFrame,
                           output_dir: str | Path, batch_size: int = 8,
                           track_metadata: Optional[pl.DataFrame] = None,
                           genome_fasta=None) -> pl.DataFrame:
    """Run Enformer inference for all variants.

    Variants are scored one at a time (Enformer has no native batching benefit
    for variant scoring beyond memory); ``batch_size`` controls how many
    variants are held in memory before flushing to parquet.
    """
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_enformer_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    variants = variants_df.to_dicts()
    chunks: List[pl.DataFrame] = []
    buffer: List[pl.DataFrame] = []

    for v in variants:
        ctx = v.get("context_id", "")
        cage_idx = match_cage_tracks(ctx, track_metadata) if track_metadata is not None else []
        df = score_variant_enformer(
            model, v, gene_master_df, cage_idx,
            sequence_length=ENFORMER_INPUT_LENGTH, genome_fasta=genome_fasta,
        )
        buffer.append(df)
        if len(buffer) >= batch_size:
            chunks.append(pl.concat(buffer, how="vertical_relaxed"))
            buffer = []
    if buffer:
        chunks.append(pl.concat(buffer, how="vertical_relaxed"))

    out = pl.concat(chunks, how="vertical_relaxed") if chunks else pl.DataFrame(schema={
        "variant_id": pl.Utf8, "gene_id": pl.Utf8,
        "enformer_signed": pl.Float64, "enformer_abs": pl.Float64,
    })
    out.write_parquet(output_dir / "enformer_predictions.parquet")
    return out


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class EnformerAdapter(ModelAdapter):
    """ModelAdapter wrapping Enformer CAGE-track inference."""

    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        super().__init__(model_id="enformer", model_family="sequence", config=config)
        self.track_metadata_path = config.get("track_metadata_path")

    def validate_resources(self) -> bool:
        try:
            import torch  # noqa: F401
            import enformer_pytorch  # noqa: F401
        except ImportError:
            return False
        return True

    def applicability(self, context_id: str) -> str:
        return "APPLICABLE"

    def score(self, inputs: Dict[str, Any]) -> pl.DataFrame:
        variants_df: pl.DataFrame = inputs["variants_df"]
        gene_master_df: pl.DataFrame = inputs["gene_master_df"]
        output_dir = inputs.get("output_dir", "./predictions/enformer")
        genome_fasta = inputs.get("genome_fasta")
        track_metadata = inputs.get("track_metadata")
        if track_metadata is None and self.track_metadata_path:
            track_metadata = pl.read_parquet(self.track_metadata_path)

        df = run_enformer_inference(
            variants_df, gene_master_df, output_dir,
            batch_size=int(self.config.get("batch_size", 8)),
            track_metadata=track_metadata, genome_fasta=genome_fasta,
        )
        return self.normalize_score(df)

    def normalize_score(self, df: pl.DataFrame) -> pl.DataFrame:
        if "enformer_abs" in df.columns:
            df = df.with_columns(pl.col("enformer_abs").alias("ranking_score"))
        return df
