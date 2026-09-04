"""Candidate gene universe construction.

For every benchmark variant (or enhancer element) we enumerate the set of
GENCODE genes that *could* be the causal target -- all genes whose TSS lies
within ``window_size`` bp of the variant position. This candidate universe
is what models are asked to rank, and it is also what gold coverage is
checked against: **every gold gene must appear in the candidate set**, or the
benchmark is silently unfair to distance-based methods.

Three window sizes are materialised in parallel (250 kb / 500 kb / 1 Mb) so
the same gold registry can be scored under increasingly permissive candidate
universes without re-running the join.

All functions use :mod:`polars` (never pandas) and write Parquet via
:mod:`v2gbench.io.parquet`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import polars as pl

from ..io.parquet import read_parquet, write_parquet
from ..schemas.candidate import CANDIDATE_BASIS, candidate_schema
from ..schemas.variant import parse_variant_id

PathLike = Union[str, Path]


def _variant_position(variants_df: pl.DataFrame) -> pl.DataFrame:
    """Ensure ``chrom`` and ``pos`` columns exist on the variants frame.

    Accepts either explicit ``chrom``/``pos`` columns or a canonical
    ``variant_id`` (``GRCh38:chr1:123456:A:G``) that is parsed on the fly.
    """
    if "chrom" in variants_df.columns and "pos" in variants_df.columns:
        return variants_df
    if "variant_id" not in variants_df.columns:
        raise ValueError(
            "variants_df must have either (chrom, pos) or variant_id columns."
        )
    parsed = variants_df.with_columns(
        pl.col("variant_id").map_elements(
            lambda v: parse_variant_id(v)["chrom"], return_dtype=pl.Utf8
        ).alias("chrom"),
        pl.col("variant_id").map_elements(
            lambda v: parse_variant_id(v)["pos"], return_dtype=pl.Int64
        ).alias("pos"),
    )
    return parsed


def build_candidate_set(
    variants_df: pl.DataFrame,
    gene_master_df: pl.DataFrame,
    window_size: int,
    context_expressed_genes: Optional[Sequence[str]] = None,
    context_id: Optional[str] = None,
) -> pl.DataFrame:
    """Build the candidate gene set for a window size.

    For each variant, every GENCODE gene whose TSS lies within ``window_size``
    bp of the variant position (same chromosome) is emitted as a candidate
    pair. Per candidate we record:

    * ``distance_to_tss`` -- absolute bp distance from variant to gene TSS.
    * ``distance_rank``   -- 1-based rank of the gene by distance within the
      variant's candidate set (1 = nearest).
    * ``is_nearest``      -- True for the single nearest gene per variant.
    * ``candidate_basis`` -- one of ``CONTEXT_TESTED``, ``CONTEXT_EXPRESSED``,
      ``GENCODE_FALLBACK`` (see :data:`CANDIDATE_BASIS`).

    ``is_gold`` / ``gold_confidence`` are *not* set here (no gold frame is
    passed); use :func:`build_all_candidate_sets` to annotate gold membership,
    or join a gold frame afterwards.

    Parameters
    ----------
    variants_df
        Variants frame with ``variant_id`` and/or ``chrom``/``pos``.
    gene_master_df
        GENCODE gene master table with ``gene_id``, ``chrom``, ``tss``.
    window_size
        Maximum absolute distance (bp) between variant and gene TSS.
    context_expressed_genes
        Optional set of gene IDs expressed in the target context. When
        provided, ``candidate_basis`` is ``CONTEXT_EXPRESSED`` for genes in
        the set and ``GENCODE_FALLBACK`` otherwise.
    context_id
        Optional context id stamped onto every candidate row.

    Returns
    -------
    pl.DataFrame
        Candidate frame conforming to ``candidate_schema`` (with
        ``is_gold``/``gold_confidence`` defaulted to 0 / null).
    """
    if variants_df.height == 0 or gene_master_df.height == 0:
        return pl.DataFrame(
            schema={
                "candidate_set_id": pl.Utf8,
                "variant_id": pl.Utf8,
                "gene_id": pl.Utf8,
                "context_id": pl.Utf8,
                "distance_to_tss": pl.Int64,
                "distance_rank": pl.Int64,
                "is_nearest": pl.Boolean,
                "is_gold": pl.Int64,
                "gold_confidence": pl.Float64,
                "candidate_basis": pl.Utf8,
            }
        )

    variants = _variant_position(variants_df)
    genes = gene_master_df.select(["gene_id", "chrom", "tss"])

    # Same-chromosome cross join, then distance filter.
    joined = (
        variants.select(["variant_id", "chrom", "pos"])
        .join(genes, on="chrom", how="inner")
        .with_columns(
            (pl.col("pos") - pl.col("tss")).abs().alias("distance_to_tss"),
        )
        .filter(pl.col("distance_to_tss") <= window_size)
    )

    if joined.height == 0:
        return pl.DataFrame(
            schema={
                "candidate_set_id": pl.Utf8,
                "variant_id": pl.Utf8,
                "gene_id": pl.Utf8,
                "context_id": pl.Utf8,
                "distance_to_tss": pl.Int64,
                "distance_rank": pl.Int64,
                "is_nearest": pl.Boolean,
                "is_gold": pl.Int64,
                "gold_confidence": pl.Float64,
                "candidate_basis": pl.Utf8,
            }
        )

    # Rank genes by distance within each variant (1 = nearest).
    joined = joined.sort(["variant_id", "distance_to_tss", "gene_id"]).with_columns(
        pl.col("distance_to_tss")
        .rank("ordinal")
        .over("variant_id")
        .cast(pl.Int64)
        .alias("distance_rank"),
    )
    joined = joined.with_columns(
        (pl.col("distance_rank") == 1).alias("is_nearest"),
    )

    # Candidate basis.
    if context_expressed_genes is not None:
        expressed_set = set(context_expressed_genes)
        joined = joined.with_columns(
            pl.when(pl.col("gene_id").is_in(list(expressed_set)))
            .then(pl.lit("CONTEXT_EXPRESSED"))
            .otherwise(pl.lit("GENCODE_FALLBACK"))
            .alias("candidate_basis")
        )
    else:
        joined = joined.with_columns(pl.lit("CONTEXT_TESTED").alias("candidate_basis"))

    # Context id + candidate_set_id + gold defaults.
    ctx = context_id if context_id is not None else "default"
    joined = joined.with_columns(
        pl.lit(ctx).alias("context_id"),
        pl.lit(0).cast(pl.Int64).alias("is_gold"),
        pl.lit(None).cast(pl.Float64).alias("gold_confidence"),
    )
    joined = joined.with_columns(
        (pl.lit(f"candidate_{window_size}_") + pl.col("variant_id"))
        .alias("candidate_set_id")
    )

    out = joined.select(
        [
            "candidate_set_id",
            "variant_id",
            "gene_id",
            "context_id",
            "distance_to_tss",
            "distance_rank",
            "is_nearest",
            "is_gold",
            "gold_confidence",
            "candidate_basis",
        ]
    )
    return candidate_schema.validate(out)


def _annotate_gold(
    candidate_df: pl.DataFrame,
    gold_df: pl.DataFrame,
) -> pl.DataFrame:
    """Mark ``is_gold`` / ``gold_confidence`` from a gold pairs frame."""
    if candidate_df.height == 0:
        return candidate_df

    gold_keys = ["variant_id", "gene_id", "context_id"]
    gold_avail = [c for c in gold_keys if c in gold_df.columns]
    if len(gold_avail) < 2:
        # Not enough keys to join -> leave is_gold as default 0.
        return candidate_df

    gold_small = gold_df.select(gold_avail)
    if "confidence" in gold_df.columns:
        gold_small = gold_small.with_columns(
            pl.col("confidence").alias("gold_confidence")
        ).rename({"confidence": "_conf"})
        gold_small = gold_df.select(gold_avail + ["confidence"]).with_columns(
            pl.col("confidence").alias("gold_confidence")
        ).drop("confidence")
    else:
        gold_small = gold_small.with_columns(
            pl.lit(1.0).cast(pl.Float64).alias("gold_confidence")
        )

    annotated = candidate_df.join(
        gold_small, on=gold_avail, how="left"
    )
    # Coalesce: gold_confidence may already exist as null from candidate build.
    if "gold_confidence_right" in annotated.columns:
        annotated = annotated.with_columns(
            pl.coalesce(["gold_confidence_right", "gold_confidence"]).alias(
                "gold_confidence"
            )
        ).drop("gold_confidence_right")

    annotated = annotated.with_columns(
        pl.when(pl.col("gold_confidence").is_not_null())
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int64)
        .alias("is_gold"),
    )
    return annotated.select(candidate_df.columns)


def build_all_candidate_sets(
    variants_df: pl.DataFrame,
    gene_master_df: pl.DataFrame,
    gold_df: Optional[pl.DataFrame] = None,
    windows: Sequence[int] = (250_000, 500_000, 1_000_000),
    context_expressed: Optional[Dict[str, Sequence[str]]] = None,
    output_dir: Optional[PathLike] = None,
) -> Dict[int, pl.DataFrame]:
    """Build candidate sets for multiple window sizes and annotate gold.

    Produces ``candidate_250k.parquet``, ``candidate_500k.parquet``,
    ``candidate_1m.parquet`` (when ``output_dir`` is given) and returns a
    mapping ``{window_size: candidate_df}``.

    Parameters
    ----------
    variants_df
        Variants frame.
    gene_master_df
        GENCODE gene master table.
    gold_df
        Optional gold pairs frame used to set ``is_gold`` / ``gold_confidence``.
    windows
        Window sizes in bp. Defaults to 250 kb / 500 kb / 1 Mb.
    context_expressed
        Optional mapping ``{context_id: [gene_id, ...]}``. When provided,
        candidate sets are built per context using the expressed-gene list
        and concatenated; otherwise a single default context is used.
    output_dir
        Optional directory to write one Parquet file per window.

    Returns
    -------
    dict[int, pl.DataFrame]
        Mapping from window size to its (gold-annotated) candidate frame.
    """
    result: Dict[int, pl.DataFrame] = {}

    for window in windows:
        frames: List[pl.DataFrame] = []
        if context_expressed:
            for ctx_id, expressed in context_expressed.items():
                frames.append(
                    build_candidate_set(
                        variants_df,
                        gene_master_df,
                        window_size=window,
                        context_expressed_genes=expressed,
                        context_id=ctx_id,
                    )
                )
        else:
            frames.append(
                build_candidate_set(
                    variants_df, gene_master_df, window_size=window
                )
            )

        if frames:
            cand = pl.concat(frames, how="vertical_relaxed")
        else:
            cand = pl.DataFrame()

        if gold_df is not None and cand.height > 0:
            cand = _annotate_gold(cand, gold_df)
            cand = candidate_schema.validate(cand)

        result[window] = cand

        if output_dir is not None:
            suffix = (
                f"{window // 1000}k" if window % 1_000_000 else f"{window // 1_000_000}m"
            )
            # 250000 -> 250k, 500000 -> 500k, 1000000 -> 1m
            if window >= 1_000_000 and window % 1_000_000 == 0:
                suffix = f"{window // 1_000_000}m"
            else:
                suffix = f"{window // 1000}k"
            write_parquet(cand, Path(output_dir) / f"candidate_{suffix}.parquet")

    return result


def check_gold_coverage(
    candidate_df: pl.DataFrame,
    gold_df: pl.DataFrame,
    raise_on_incomplete: bool = True,
) -> float:
    """CRITICAL: assert all gold genes appear in the candidate set.

    A gold gene missing from the candidate universe means no model -- not
    even an oracle -- could ever rank it, making the benchmark silently
    biased toward distance-based methods. This function computes the fraction
    of gold (variant/element, gene, context) pairs that are present in the
    candidate frame and, by default, raises if coverage is below 100 %.

    Parameters
    ----------
    candidate_df
        Candidate frame (must contain ``is_gold`` or be joinable on keys).
    gold_df
        Gold pairs frame.
    raise_on_incomplete
        When True (default), raise ``AssertionError`` if coverage < 1.0.

    Returns
    -------
    float
        Coverage fraction in ``[0, 1]``.
    """
    keys = ["variant_id", "gene_id", "context_id"]
    # Use whichever keys are shared; element_id replaces variant_id for Track B.
    if "variant_id" not in gold_df.columns and "element_id" in gold_df.columns:
        keys = ["element_id", "gene_id", "context_id"]
    keys = [k for k in keys if k in gold_df.columns and k in candidate_df.columns]
    if len(keys) < 2:
        raise ValueError(
            "Cannot compute gold coverage: insufficient shared key columns."
        )

    gold_keys = gold_df.select(keys).unique()
    cand_keys = candidate_df.select(keys).unique()

    total = gold_keys.height
    if total == 0:
        return 1.0

    covered = gold_keys.join(cand_keys, on=keys, how="inner").height
    coverage = covered / total

    if raise_on_incomplete and coverage < 1.0:
        missing = gold_keys.join(
            cand_keys, on=keys, how="anti"
        )
        sample = missing.head(5).to_dicts()
        raise AssertionError(
            f"Gold coverage incomplete: {covered}/{total} pairs "
            f"({coverage:.4%}). Missing examples: {sample}"
        )
    return coverage


def compute_gold_distance_rank(candidate_df: pl.DataFrame) -> pl.DataFrame:
    """For each gold gene, report its ``distance_rank`` bucket.

    Buckets the per-gene distance rank into ``1``, ``2``, ``3``, ``4+`` so
    downstream reporting can show how often the gold gene is the nearest,
    second-nearest, etc. candidate.

    Parameters
    ----------
    candidate_df
        Candidate frame with ``is_gold`` and ``distance_rank`` columns.

    Returns
    -------
    pl.DataFrame
        One row per gold candidate with columns
        ``variant_id, gene_id, context_id, distance_rank, rank_bucket``.
    """
    if candidate_df.height == 0:
        return pl.DataFrame(
            schema={
                "variant_id": pl.Utf8,
                "gene_id": pl.Utf8,
                "context_id": pl.Utf8,
                "distance_rank": pl.Int64,
                "rank_bucket": pl.Utf8,
            }
        )

    gold = candidate_df.filter(pl.col("is_gold") == 1)
    if gold.height == 0:
        return pl.DataFrame(
            schema={
                "variant_id": pl.Utf8,
                "gene_id": pl.Utf8,
                "context_id": pl.Utf8,
                "distance_rank": pl.Int64,
                "rank_bucket": pl.Utf8,
            }
        )

    gold = gold.with_columns(
        pl.when(pl.col("distance_rank") <= 3)
        .then(pl.col("distance_rank").cast(pl.Utf8))
        .otherwise(pl.lit("4+"))
        .alias("rank_bucket")
    )
    return gold.select(
        ["variant_id", "gene_id", "context_id", "distance_rank", "rank_bucket"]
    )


__all__ = [
    "build_candidate_set",
    "build_all_candidate_sets",
    "check_gold_coverage",
    "compute_gold_distance_rank",
]
