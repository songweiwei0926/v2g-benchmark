"""Gold evidence registry construction.

This module builds the canonical *gold* evidence registry that downstream
benchmark metrics score against. It combines the per-dataset adapter outputs
into a single long-format evidence table, aggregates them into canonical
(variant/element, gene, context) pairs, and assigns positive / negative /
unknown labels following dataset-appropriate rules:

* eQTL evidence  -> PIP-thresholded labels (primary + sensitivity sweeps).
* CRISPR evidence -> authors' own negative / powered-negative calls
  (never a bare ``P > 0.05``).

GTEx eQTL records that also appear in the eQTL Catalogue are de-duplicated
out of the eQTL Catalogue *main* benchmark so each pair is counted once.

All functions use :mod:`polars` (never pandas) and write Parquet via
:mod:`v2gbench.io.parquet`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import polars as pl

from ..io.parquet import read_parquet, write_parquet
from ..schemas.evidence import (
    EVIDENCE_TYPES,
    LEAKAGE_TYPES,
    canonical_pairs_schema,
    evidence_schema,
)

PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# Column constants -- keep the canonical schema in sync with schemas/evidence.py
# ---------------------------------------------------------------------------
EVIDENCE_COLUMNS: tuple[str, ...] = (
    "benchmark_id",
    "evidence_id",
    "variant_id",
    "element_id",
    "gene_id",
    "context_id",
    "trait_id",
    "evidence_type",
    "label",
    "effect_size",
    "effect_direction",
    "pip",
    "pvalue",
    "source_dataset",
    "source_publication",
    "confidence",
    "training_overlap",
)


def _ensure_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Return a frame guaranteed to contain every canonical evidence column.

    Missing columns are added as ``null`` (or ``0`` for the integer ``label``
    which is non-nullable in the schema). ``evidence_type`` and
    ``training_overlap`` fall back to schema-valid sentinels when absent.
    """
    missing = [c for c in EVIDENCE_COLUMNS if c not in df.columns]
    for col in missing:
        if col == "label":
            df = df.with_columns(pl.lit(0).cast(pl.Int64).alias(col))
        elif col == "evidence_type":
            df = df.with_columns(pl.lit("curated_L2G").alias(col))
        elif col == "training_overlap":
            df = df.with_columns(pl.lit("UNKNOWN").alias(col))
        else:
            df = df.with_columns(pl.lit(None).alias(col))
    # Preserve canonical column order.
    return df.select(list(EVIDENCE_COLUMNS))


def build_evidence_long(
    adapters_output: Union[PathLike, Dict[str, PathLike], Iterable[PathLike]],
    output_path: PathLike,
) -> pl.DataFrame:
    """Combine all dataset adapter outputs into ``evidence_long.parquet``.

    Parameters
    ----------
    adapters_output
        Either a directory containing one Parquet file per adapter, an
        explicit mapping ``{dataset_name: parquet_path}``, or an iterable of
        Parquet paths. Each input is expected to already conform (or be
        close to) the canonical evidence schema; missing columns are filled
        with nulls / safe defaults.
    output_path
        Destination Parquet path for the concatenated long table.

    Returns
    -------
    pl.DataFrame
        The concatenated, schema-validated evidence-long frame.

    Notes
    -----
    A stable ``benchmark_id`` is assigned per source dataset when the input is
    a mapping (``benchmark_id == source_dataset``). When a directory/iterable
    is given, ``benchmark_id`` is taken from the file if present, otherwise
    derived from the file stem.
    """
    frames: List[pl.DataFrame] = []

    if isinstance(adapters_output, dict):
        items: List[tuple] = [(name, path) for name, path in adapters_output.items()]
    elif isinstance(adapters_output, (str, Path)):
        p = Path(adapters_output)
        if not p.exists():
            raise FileNotFoundError(f"adapters_output path does not exist: {p}")
        if p.is_dir():
            items = [(f.stem, f) for f in sorted(p.glob("*.parquet"))]
        else:
            items = [(p.stem, p)]
    else:  # iterable of paths
        items = [(Path(path).stem, Path(path)) for path in adapters_output]

    if not items:
        raise ValueError("No adapter outputs found to build evidence_long.")

    for name, path in items:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Adapter output not found for '{name}': {path}")
        df = read_parquet(path)
        if "benchmark_id" not in df.columns:
            df = df.with_columns(pl.lit(name).alias("benchmark_id"))
        if "source_dataset" not in df.columns:
            df = df.with_columns(pl.lit(name).alias("source_dataset"))
        df = _ensure_columns(df)
        frames.append(df)

    combined = pl.concat(frames, how="vertical_relaxed")

    # Validate against the canonical schema (coerces dtypes, raises on errors).
    combined = evidence_schema.validate(combined)

    write_parquet(combined, output_path)
    return combined


def build_canonical_pairs(
    evidence_long: Union[pl.DataFrame, PathLike],
    output_path: Optional[PathLike] = None,
) -> pl.DataFrame:
    """Aggregate evidence-long into canonical (locus, gene, context) pairs.

    The locus key is the combination of ``variant_id`` / ``element_id``
    (variant-centric Track A/C vs. enhancer-centric Track B; one is typically
    null). Aggregation produces, per pair:

    * ``n_evidence_sources`` -- number of distinct ``source_dataset`` values.
    * ``evidence_sources``   -- pipe-delimited sorted list of those datasets.
    * ``max_confidence``     -- maximum ``confidence`` across supporting rows.

    Parameters
    ----------
    evidence_long
        Either the in-memory evidence frame or a path to its Parquet file.
    output_path
        Optional destination Parquet path.

    Returns
    -------
    pl.DataFrame
        Canonical pairs frame conforming to ``canonical_pairs_schema``.
    """
    if isinstance(evidence_long, (str, Path)):
        evidence_long = read_parquet(evidence_long)

    if evidence_long.height == 0:
        pairs = pl.DataFrame(
            schema={
                "variant_id": pl.Utf8,
                "element_id": pl.Utf8,
                "gene_id": pl.Utf8,
                "context_id": pl.Utf8,
                "n_evidence_sources": pl.Int64,
                "evidence_sources": pl.Utf8,
                "max_confidence": pl.Float64,
            }
        )
    else:
        pairs = (
            evidence_long
            .group_by(["variant_id", "element_id", "gene_id", "context_id"])
            .agg(
                pl.col("source_dataset").n_unique().alias("n_evidence_sources"),
                pl.col("source_dataset")
                .unique()
                .sort()
                .str.join("|")
                .alias("evidence_sources"),
                pl.col("confidence").max().alias("max_confidence"),
            )
            .sort(["variant_id", "element_id", "gene_id", "context_id"])
        )

    pairs = canonical_pairs_schema.validate(pairs)

    if output_path is not None:
        write_parquet(pairs, output_path)
    return pairs


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------
def assign_eqtl_labels(
    evidence_df: pl.DataFrame,
    primary_pip: float = 0.90,
    sensitivity_pips: Sequence[float] = (0.50, 0.70, 0.90, 0.95),
    negative_pip: float = 0.01,
) -> pl.DataFrame:
    """Assign eQTL labels from PIP thresholds.

    A row is labelled:

    * **positive** (``label == 1``) when ``pip >= primary_pip``.
    * **negative** (``label == 0``) when the variant-gene pair was *tested*
      (``pip`` is not null) **and** ``pip <= negative_pip``.
    * **unknown** (``label == -1``) otherwise -- kept for transparency but
      excluded from scoring.

    Sensitivity-threshold columns (``label_pip_0.50`` etc.) are added so a
    single pass yields the primary label plus the full sensitivity sweep.

    Parameters
    ----------
    evidence_df
        Evidence-long frame. Non-eQTL rows are returned unchanged.
    primary_pip
        PIP threshold for the primary positive call.
    sensitivity_pips
        PIP thresholds for the sensitivity sweep. ``primary_pip`` is added
        automatically if missing.
    negative_pip
        PIP threshold at or below which a *tested* pair is called negative.

    Returns
    -------
    pl.DataFrame
        Frame with ``label`` plus one ``label_pip_{threshold}`` column per
        sensitivity threshold (eQTL rows only).
    """
    if evidence_df.height == 0:
        return evidence_df

    is_eqtl = pl.col("evidence_type") == "eQTL"
    eqtl = evidence_df.filter(is_eqtl)
    other = evidence_df.filter(~is_eqtl)

    if eqtl.height == 0:
        return evidence_df

    # Primary label: 1 = positive, 0 = negative (tested & low PIP), -1 = unknown.
    eqtl = eqtl.with_columns(
        pl.when(pl.col("pip") >= primary_pip)
        .then(pl.lit(1))
        .when(pl.col("pip").is_not_null() & (pl.col("pip") <= negative_pip))
        .then(pl.lit(0))
        .otherwise(pl.lit(-1))
        .cast(pl.Int64)
        .alias("label")
    )

    # Sensitivity sweep -- ensure primary threshold is included.
    sens = sorted(set(sensitivity_pips) | {primary_pip})
    for thr in sens:
        col = f"label_pip_{thr:.2f}"
        eqtl = eqtl.with_columns(
            pl.when(pl.col("pip") >= thr)
            .then(pl.lit(1))
            .when(pl.col("pip").is_not_null() & (pl.col("pip") <= negative_pip))
            .then(pl.lit(0))
            .otherwise(pl.lit(-1))
            .cast(pl.Int64)
            .alias(col)
        )

    return pl.concat([other, eqtl], how="vertical_relaxed")


def assign_crispr_labels(evidence_df: pl.DataFrame) -> pl.DataFrame:
    """Assign CRISPR labels using the authors' own negative calls.

    CRISPR benchmark datasets (e.g. the Engreitz lab EPCrisprBenchmark)
    ship author-curated columns that distinguish a *true negative* (the
    perturbation was performed and no effect was observed, with adequate
    power) from a mere ``P > 0.05`` non-significance. We trust those calls
    rather than re-deriving negatives from a p-value threshold, which would
    conflate "underpowered" with "no effect".

    Recognised author-call columns (checked in priority order):

    * ``Regulated`` (EPCrisprBenchmark) -- ``"No"`` -> negative, ``"Yes"`` -> positive.
    * ``authors_negative`` / ``authors_positive`` -- explicit boolean flags.
    * ``powered_negative`` -- boolean; a tested, adequately-powered null.

    Rows lacking any author call are labelled ``-1`` (unknown) and excluded
    from scoring rather than guessed.

    Parameters
    ----------
    evidence_df
        Evidence-long frame; non-CRISPR rows are returned unchanged.

    Returns
    -------
    pl.DataFrame
        Frame with an updated integer ``label`` column.
    """
    if evidence_df.height == 0:
        return evidence_df

    is_crispr = pl.col("evidence_type").is_in(["CRISPRi", "CRISPRa", "PerturbSeq"])
    crispr = evidence_df.filter(is_crispr)
    other = evidence_df.filter(~is_crispr)

    if crispr.height == 0:
        return evidence_df

    cols = set(crispr.columns)
    pos_exprs: List[pl.Expr] = []
    neg_exprs: List[pl.Expr] = []

    if "Regulated" in cols:
        pos_exprs.append(pl.col("Regulated").str.to_lowercase() == "yes")
        neg_exprs.append(pl.col("Regulated").str.to_lowercase() == "no")
    if "authors_positive" in cols:
        pos_exprs.append(pl.col("authors_positive"))
    if "authors_negative" in cols:
        neg_exprs.append(pl.col("authors_negative"))
    if "powered_negative" in cols:
        neg_exprs.append(pl.col("powered_negative"))

    if pos_exprs or neg_exprs:
        positive = pl.lit(False)
        for e in pos_exprs:
            positive = positive | e.fill_null(False)
        negative = pl.lit(False)
        for e in neg_exprs:
            negative = negative | e.fill_null(False)

        label_expr = (
            pl.when(positive).then(pl.lit(1))
            .when(negative).then(pl.lit(0))
            .otherwise(pl.lit(-1))
            .cast(pl.Int64)
        )
        crispr = crispr.with_columns(label_expr.alias("label"))
    else:
        # No author-call columns -> mark unknown rather than guess.
        crispr = crispr.with_columns(pl.lit(-1).cast(pl.Int64).alias("label"))

    return pl.concat([other, crispr], how="vertical_relaxed")


def deduplicate_gtex_eqtl(evidence_df: pl.DataFrame) -> pl.DataFrame:
    """Exclude ``study == GTEx`` rows from the eQTL Catalogue main benchmark.

    GTEx fine-mapped eQTLs are ingested both directly (GTEx V11 adapter) and
    embedded inside the eQTL Catalogue. To avoid double-counting a pair in
    the eQTL Catalogue *main* benchmark, we drop Catalogue rows whose
    ``study`` (or ``source_dataset``) identifies them as GTEx, *only* when a
    native GTEx record already covers the same (variant, gene, context) pair.

    Rows whose ``source_dataset`` is the native GTEx adapter are always kept.

    Parameters
    ----------
    evidence_df
        Evidence-long frame.

    Returns
    -------
    pl.DataFrame
        Filtered frame with GTEx-from-Catalogue duplicates removed.
    """
    if evidence_df.height == 0:
        return evidence_df

    # Identify the GTEx study column robustly.
    study_col = None
    for cand in ("study", "qtl_study", "study_id"):
        if cand in evidence_df.columns:
            study_col = cand
            break

    is_eqtl = pl.col("evidence_type") == "eQTL"
    is_native_gtex = pl.col("source_dataset").str.to_lowercase().str.contains("gtex")
    if study_col is not None:
        is_catalogue_gtex = (
            (~is_native_gtex)
            & is_eqtl
            & pl.col(study_col).str.to_lowercase().str.contains("gtex")
        )
    else:
        # No study column -> fall back to source_dataset only (nothing to drop).
        is_catalogue_gtex = pl.lit(False)

    catalogue_gtex = evidence_df.filter(is_catalogue_gtex)
    if catalogue_gtex.height == 0:
        return evidence_df

    # Pairs covered by the native GTEx adapter.
    native_pairs = (
        evidence_df.filter(is_eqtl & is_native_gtex)
        .select(["variant_id", "gene_id", "context_id"])
        .unique()
    )

    if native_pairs.height == 0:
        # No native GTEx coverage -> keep catalogue GTEx rows (no duplicate).
        return evidence_df

    # Mark catalogue-GTEx rows whose pair is already covered natively.
    to_drop = (
        catalogue_gtex.join(
            native_pairs,
            on=["variant_id", "gene_id", "context_id"],
            how="inner",
        ).select("evidence_id")
    )
    drop_ids = to_drop["evidence_id"].to_list()
    if not drop_ids:
        return evidence_df

    return evidence_df.filter(~pl.col("evidence_id").is_in(drop_ids))


__all__ = [
    "EVIDENCE_COLUMNS",
    "build_evidence_long",
    "build_canonical_pairs",
    "assign_eqtl_labels",
    "assign_crispr_labels",
    "deduplicate_gtex_eqtl",
]
