"""SEQ_CORE deterministic stratified sampling.

Sequence models (Enformer / Borzoi / AlphaGenome) are expensive to run, so
the benchmark defines a fixed, reproducible *SEQ_CORE* subset of at most
``max_variant_contexts`` (default 5000) variant-context pairs on which all
sequence models are compared fairly. Selection must be:

* **deterministic** -- the same input always yields the same subset, with no
  dependence on row order or RNG state. We achieve this by ordering
  candidate (variant_id, context_id) pairs by a SHA256 hash.
* **stratified** -- the subset preserves the joint distribution of several
  strata (benchmark source, distance bin, PIP bin, nearest/non-nearest,
  chromosome, context supergroup) so it is representative of the full gold
  registry rather than a random slice.

All functions use :mod:`polars` (never pandas) and SHA256 hashing via
:mod:`v2gbench.utils.hashing`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Union

import polars as pl

from ..io.parquet import read_parquet, write_parquet
from ..utils.hashing import stable_hash

PathLike = Union[str, Path]

# Default strata columns produced by :func:`assign_strata`.
STRATA_COLUMNS = (
    "benchmark_source",
    "distance_bin",
    "pip_bin",
    "nearest_status",
    "chrom",
    "context_supergroup",
)


def hash_variant_context(variant_id: str, context_id: str) -> str:
    """SHA256 hash of ``variant_id|context_id`` for deterministic ordering.

    The hash is hex-encoded (64 chars) and stable across runs, machines, and
    row orders, so sorting on it yields a reproducible global ordering of
    variant-context pairs.
    """
    return stable_hash(variant_id, context_id)


def _distance_bin(distance: int) -> str:
    """Bucket a TSS distance into coarse bins used for stratification."""
    if distance is None:
        return "unknown"
    if distance <= 10_000:
        return "0-10kb"
    if distance <= 50_000:
        return "10-50kb"
    if distance <= 100_000:
        return "50-100kb"
    if distance <= 250_000:
        return "100-250kb"
    if distance <= 500_000:
        return "250-500kb"
    return "500kb+"


def _pip_bin(pip) -> str:
    """Bucket a PIP into coarse bins used for stratification."""
    if pip is None:
        return "unknown"
    if pip < 0.01:
        return "pip<0.01"
    if pip < 0.10:
        return "pip0.01-0.10"
    if pip < 0.50:
        return "pip0.10-0.50"
    if pip < 0.90:
        return "pip0.50-0.90"
    return "pip>=0.90"


def assign_strata(evidence_df: pl.DataFrame) -> pl.DataFrame:
    """Assign strata labels to each evidence row.

    Adds the columns in :data:`STRATA_COLUMNS`:

    * ``benchmark_source``     -- from ``benchmark_id`` / ``source_dataset``.
    * ``distance_bin``         -- coarse TSS-distance bucket (needs
      ``distance_to_tss``; ``"unknown"`` when absent).
    * ``pip_bin``              -- coarse PIP bucket (``"unknown"`` when absent).
    * ``nearest_status``       -- ``"nearest"`` / ``"non-nearest"`` / ``"unknown"``
      (needs ``is_nearest``).
    * ``chrom``                -- chromosome (parsed from ``variant_id`` when
      ``chrom`` is absent).
    * ``context_supergroup``   -- from ``supergroup`` / ``context_supergroup``
      (``"unknown"`` when absent).

    Parameters
    ----------
    evidence_df
        Evidence-long (or candidate-joined) frame.

    Returns
    -------
    pl.DataFrame
        Input frame plus the strata columns.
    """
    if evidence_df.height == 0:
        return evidence_df.with_columns(
            [pl.lit(None).alias(c) for c in STRATA_COLUMNS]
        )

    df = evidence_df

    # benchmark_source
    src_col = "benchmark_id" if "benchmark_id" in df.columns else "source_dataset"
    if src_col in df.columns:
        df = df.with_columns(pl.col(src_col).alias("benchmark_source"))
    else:
        df = df.with_columns(pl.lit("unknown").alias("benchmark_source"))

    # distance_bin
    if "distance_to_tss" in df.columns:
        df = df.with_columns(
            pl.col("distance_to_tss").map_elements(
                _distance_bin, return_dtype=pl.Utf8
            ).alias("distance_bin")
        )
    else:
        df = df.with_columns(pl.lit("unknown").alias("distance_bin"))

    # pip_bin
    if "pip" in df.columns:
        df = df.with_columns(
            pl.col("pip").map_elements(_pip_bin, return_dtype=pl.Utf8).alias("pip_bin")
        )
    else:
        df = df.with_columns(pl.lit("unknown").alias("pip_bin"))

    # nearest_status
    if "is_nearest" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("is_nearest"))
            .then(pl.lit("nearest"))
            .when(pl.col("is_nearest").is_not_null())
            .then(pl.lit("non-nearest"))
            .otherwise(pl.lit("unknown"))
            .alias("nearest_status")
        )
    else:
        df = df.with_columns(pl.lit("unknown").alias("nearest_status"))

    # chrom
    if "chrom" in df.columns:
        df = df.with_columns(pl.col("chrom").alias("chrom"))
    elif "variant_id" in df.columns:
        df = df.with_columns(
            pl.col("variant_id").map_elements(
                lambda v: v.split(":")[1] if v and ":" in v else "unknown",
                return_dtype=pl.Utf8,
            ).alias("chrom")
        )
    else:
        df = df.with_columns(pl.lit("unknown").alias("chrom"))

    # context_supergroup
    sg_col = None
    for cand in ("context_supergroup", "supergroup"):
        if cand in df.columns:
            sg_col = cand
            break
    if sg_col is not None:
        df = df.with_columns(
            pl.col(sg_col).fill_null("unknown").alias("context_supergroup")
        )
    else:
        df = df.with_columns(pl.lit("unknown").alias("context_supergroup"))

    return df


def build_seq_core(
    evidence_df: Union[pl.DataFrame, PathLike],
    max_variant_contexts: int = 5000,
    seed: int = 20260904,
    output_path: Optional[PathLike] = None,
) -> pl.DataFrame:
    """Deterministically sample the SEQ_CORE subset.

    Procedure:

    1. Assign strata via :func:`assign_strata`.
    2. Reduce to unique (variant_id, context_id) pairs, carrying the strata
       and a deterministic ``hash`` = ``hash_variant_context(variant_id,
       context_id)``.
    3. Stratify jointly by :data:`STRATA_COLUMNS`. Within each stratum, order
       pairs by their hash (deterministic) and allocate slots *proportional*
       to the stratum's size, rounding up so small strata are guaranteed
       representation. The global cap ``max_variant_contexts`` is enforced
       after allocation by trimming the highest-hash pairs.
    4. Return the subset of ``evidence_df`` whose (variant_id, context_id)
       pairs were selected.

    Parameters
    ----------
    evidence_df
        Evidence-long frame (in-memory or Parquet path). Must contain
        ``variant_id`` and ``context_id``.
    max_variant_contexts
        Maximum number of (variant_id, context_id) pairs to select.
    seed
        Seed folded into the hash for extra determinism (default project seed).
    output_path
        Optional destination Parquet path for the selected subset.

    Returns
    -------
    pl.DataFrame
        The SEQ_CORE subset of ``evidence_df`` (all evidence rows for the
        selected variant-context pairs).
    """
    if isinstance(evidence_df, (str, Path)):
        evidence_df = read_parquet(evidence_df)

    if evidence_df.height == 0:
        return evidence_df

    if "variant_id" not in evidence_df.columns or "context_id" not in evidence_df.columns:
        raise ValueError(
            "evidence_df must contain variant_id and context_id columns."
        )

    df = assign_strata(evidence_df)

    # Unique variant-context pairs with their strata + deterministic hash.
    pairs = (
        df.group_by(["variant_id", "context_id"], maintain_order=False)
        .agg([pl.col(c).first() for c in STRATA_COLUMNS])
        .with_columns(
            pl.struct(["variant_id", "context_id"])
            .map_elements(
                lambda s: hash_variant_context(s["variant_id"], s["context_id"]),
                return_dtype=pl.Utf8,
            )
            .alias("hash")
        )
    )

    # Fold the seed into the ordering hash so the global sort is seeded.
    pairs = pairs.with_columns(
        pl.struct(["hash", "variant_id", "context_id"])
        .map_elements(
            lambda s: stable_hash(seed, s["variant_id"], s["context_id"]),
            return_dtype=pl.Utf8,
        )
        .alias("order_hash")
    )

    total_pairs = pairs.height
    if total_pairs <= max_variant_contexts:
        # Everything fits -- no sampling needed, but keep deterministic order.
        selected = pairs.select(["variant_id", "context_id"])
    else:
        # Proportional allocation across joint strata.
        strata_counts = (
            pairs.group_by(list(STRATA_COLUMNS))
            .agg(pl.len().alias("stratum_n"))
            .with_columns(
                (pl.col("stratum_n") / total_pairs * max_variant_contexts)
                .ceil()
                .cast(pl.Int64)
                .alias("quota")
            )
        )

        # Join quota back, rank within stratum by order_hash, keep top `quota`.
        pairs = pairs.join(strata_counts, on=list(STRATA_COLUMNS), how="left")
        pairs = pairs.sort(list(STRATA_COLUMNS) + ["order_hash"]).with_columns(
            pl.col("order_hash")
            .rank("ordinal")
            .over(list(STRATA_COLUMNS))
            .alias("within_stratum_rank")
        )
        selected = pairs.filter(pl.col("within_stratum_rank") <= pl.col("quota"))

        # Enforce the global cap: trim the highest order_hash pairs.
        if selected.height > max_variant_contexts:
            selected = selected.sort("order_hash").head(max_variant_contexts)

        selected = selected.select(["variant_id", "context_id"])

    # Pull all evidence rows for the selected variant-context pairs.
    subset = evidence_df.join(
        selected, on=["variant_id", "context_id"], how="inner"
    )

    if output_path is not None:
        write_parquet(subset, output_path)
    return subset


__all__ = [
    "STRATA_COLUMNS",
    "build_seq_core",
    "hash_variant_context",
    "assign_strata",
]
