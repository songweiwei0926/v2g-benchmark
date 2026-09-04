"""Published prediction importers (Family 1 & 2).

These adapters read pre-computed element-to-gene (E2G) or variant-to-gene
predictions produced by external tools and map them onto the canonical
prediction schema.  They do **not** run any model themselves — they only
harmonize column names, score semantics and context identifiers.

Two families are covered:

* **Family 1 — classical E2G**: ABC, ENCODE-rE2G, scE2G, EpiMap, GraphReg.
* **Family 2 — single-cell E2G**: pgBoost, SCENT, Signac, ArchR, Cicero.

Because every published method emits a slightly different file layout, each
importer is a thin function that:

1. Reads the file (parquet / tsv / csv — auto-detected by extension).
2. Renames method-specific columns to the canonical set
   ``{element_id, gene_id, context_id, score}``.
3. Returns a tidy :class:`polars.DataFrame`.

The generic :class:`PublishedPredictionAdapter` wraps an importer and exposes
the :class:`~v2gbench.models.base.ModelAdapter` interface, including the
E2G→V2G conversion (:func:`e2g_to_v2g`) that lifts element scores onto
overlapping variants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import polars as pl

from .base import ModelAdapter

# Aggregation modes for E2G → V2G conversion.
Aggregation = Literal["max", "sum", "mean"]

# Canonical columns every importer must produce.
CANONICAL_COLUMNS = ("element_id", "gene_id", "context_id", "score")


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def _read_table(path: str | Path) -> pl.DataFrame:
    """Read a tabular file, inferring format from the extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pl.read_parquet(path)
    if suffix in (".tsv", ".txt"):
        return pl.read_csv(path, separator="\t")
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix in (".gz",):
        # .tsv.gz / .csv.gz
        inner = path.with_suffix("").suffix.lower()
        if inner == ".tsv":
            return pl.read_csv(path, separator="\t")
        return pl.read_csv(path)
    # Default: try TSV.
    return pl.read_csv(path, separator="\t")


def _coerce_score(df: pl.DataFrame, score_col: str) -> pl.DataFrame:
    """Ensure the score column is Float64 and finite-or-null."""
    if score_col not in df.columns:
        raise ValueError(f"Expected score column '{score_col}' not found in {df.columns}")
    return df.with_columns(pl.col(score_col).cast(pl.Float64).alias("score"))


def _rename(df: pl.DataFrame, mapping: Dict[str, str]) -> pl.DataFrame:
    """Rename columns that exist; leave the rest untouched."""
    present = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(present)


def _ensure_canonical(df: pl.DataFrame, *, source: str) -> pl.DataFrame:
    """Select/derive the canonical columns and drop the rest."""
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[{source}] missing canonical columns: {missing}; have {df.columns}")
    return df.select(list(CANONICAL_COLUMNS))


# --------------------------------------------------------------------------- #
# Family 1 — classical E2G importers
# --------------------------------------------------------------------------- #
def import_abc_predictions(input_path: str | Path, candidate_df: pl.DataFrame) -> pl.DataFrame:
    """Import ABC (Activity-by-Contact) E2G predictions.

    Expected columns (any case-tolerant alias accepted):
        ``chr``/``chrom``, ``start``, ``end``, ``TargetGene``/``gene_id``,
        ``CellType``/``context_id``, ``ABC.Score``/``score``.
    """
    df = _read_table(input_path)
    df = _rename(df, {
        "chr": "chrom", "chrom": "chrom",
        "TargetGene": "gene_id", "gene_id": "gene_id",
        "CellType": "context_id", "context_id": "context_id",
        "ABC.Score": "score", "ABC_Score": "score", "score": "score",
    })
    if "element_id" not in df.columns and {"chrom", "start", "end"}.issubset(df.columns):
        df = df.with_columns(
            (pl.col("chrom") + ":" + pl.col("start").cast(pl.Utf8)
             + "-" + pl.col("end").cast(pl.Utf8)).alias("element_id")
        )
    df = _coerce_score(df, "score")
    return _ensure_canonical(df, source="abc")


def import_re2g_predictions(input_path: str | Path, candidate_df: pl.DataFrame) -> pl.DataFrame:
    """Import ENCODE-rE2G predictions.

    Expected columns: ``chr``, ``start``, ``end``, ``GeneSymbol``/``gene_id``,
    ``CellType``/``context_id``, ``EnhancerToGene``/``score`` (or ``p``).
    """
    df = _read_table(input_path)
    df = _rename(df, {
        "chr": "chrom", "chrom": "chrom",
        "GeneSymbol": "gene_id", "gene_id": "gene_id", "TargetGene": "gene_id",
        "CellType": "context_id", "context_id": "context_id",
        "EnhancerToGene": "score", "score": "score", "p": "score",
        "Prob": "score", "probability": "score",
    })
    if "element_id" not in df.columns and {"chrom", "start", "end"}.issubset(df.columns):
        df = df.with_columns(
            (pl.col("chrom") + ":" + pl.col("start").cast(pl.Utf8)
             + "-" + pl.col("end").cast(pl.Utf8)).alias("element_id")
        )
    df = _coerce_score(df, "score")
    return _ensure_canonical(df, source="re2g")


def import_sce2g_predictions(input_path: str | Path, candidate_df: pl.DataFrame) -> pl.DataFrame:
    """Import scE2G predictions (ATAC and multiome variants share a layout)."""
    df = _read_table(input_path)
    df = _rename(df, {
        "chr": "chrom", "chrom": "chrom",
        "Gene": "gene_id", "gene_id": "gene_id", "TargetGene": "gene_id",
        "CellType": "context_id", "context_id": "context_id",
        "K27ac": "score", "score": "score", "Prob": "score",
    })
    if "element_id" not in df.columns and {"chrom", "start", "end"}.issubset(df.columns):
        df = df.with_columns(
            (pl.col("chrom") + ":" + pl.col("start").cast(pl.Utf8)
             + "-" + pl.col("end").cast(pl.Utf8)).alias("element_id")
        )
    df = _coerce_score(df, "score")
    return _ensure_canonical(df, source="sce2g")


def import_epimap_predictions(input_path: str | Path, candidate_df: pl.DataFrame) -> pl.DataFrame:
    """Import EpiMap enhancer–gene predictions."""
    df = _read_table(input_path)
    df = _rename(df, {
        "chr": "chrom", "chrom": "chrom",
        "gene": "gene_id", "gene_id": "gene_id", "TargetGene": "gene_id",
        "tissue": "context_id", "context_id": "context_id", "CellType": "context_id",
        "score": "score", "ABC": "score", "p": "score",
    })
    if "element_id" not in df.columns and {"chrom", "start", "end"}.issubset(df.columns):
        df = df.with_columns(
            (pl.col("chrom") + ":" + pl.col("start").cast(pl.Utf8)
             + "-" + pl.col("end").cast(pl.Utf8)).alias("element_id")
        )
    df = _coerce_score(df, "score")
    return _ensure_canonical(df, source="epimap")


def import_graphreg_predictions(input_path: str | Path, candidate_df: pl.DataFrame) -> pl.DataFrame:
    """Import GraphReg enhancer–gene predictions."""
    df = _read_table(input_path)
    df = _rename(df, {
        "chr": "chrom", "chrom": "chrom",
        "gene": "gene_id", "gene_id": "gene_id", "TargetGene": "gene_id",
        "cell_type": "context_id", "context_id": "context_id", "CellType": "context_id",
        "score": "score", "importance": "score", "p": "score",
    })
    if "element_id" not in df.columns and {"chrom", "start", "end"}.issubset(df.columns):
        df = df.with_columns(
            (pl.col("chrom") + ":" + pl.col("start").cast(pl.Utf8)
             + "-" + pl.col("end").cast(pl.Utf8)).alias("element_id")
        )
    df = _coerce_score(df, "score")
    return _ensure_canonical(df, source="graphreg")


# --------------------------------------------------------------------------- #
# Family 2 — single-cell E2G importers
# --------------------------------------------------------------------------- #
def import_pgboost_predictions(input_path: str | Path, candidate_df: pl.DataFrame) -> pl.DataFrame:
    """Import pgBoost predictions (Zenodo bundle)."""
    df = _read_table(input_path)
    df = _rename(df, {
        "chr": "chrom", "chrom": "chrom",
        "gene": "gene_id", "gene_id": "gene_id", "TargetGene": "gene_id",
        "cell_type": "context_id", "context_id": "context_id", "CellType": "context_id",
        "score": "score", "p": "score", "prob": "score",
    })
    if "element_id" not in df.columns and {"chrom", "start", "end"}.issubset(df.columns):
        df = df.with_columns(
            (pl.col("chrom") + ":" + pl.col("start").cast(pl.Utf8)
             + "-" + pl.col("end").cast(pl.Utf8)).alias("element_id")
        )
    df = _coerce_score(df, "score")
    return _ensure_canonical(df, source="pgboost")


def import_scent_predictions(input_path: str | Path, candidate_df: pl.DataFrame) -> pl.DataFrame:
    """Import SCENT enhancer–gene predictions."""
    df = _read_table(input_path)
    df = _rename(df, {
        "chr": "chrom", "chrom": "chrom",
        "gene": "gene_id", "gene_id": "gene_id", "TargetGene": "gene_id",
        "cell_type": "context_id", "context_id": "context_id", "CellType": "context_id",
        "score": "score", "p": "score", "rho": "score",
    })
    if "element_id" not in df.columns and {"chrom", "start", "end"}.issubset(df.columns):
        df = df.with_columns(
            (pl.col("chrom") + ":" + pl.col("start").cast(pl.Utf8)
             + "-" + pl.col("end").cast(pl.Utf8)).alias("element_id")
        )
    df = _coerce_score(df, "score")
    return _ensure_canonical(df, source="scent")


def import_signac_predictions(input_path: str | Path, candidate_df: pl.DataFrame) -> pl.DataFrame:
    """Import Signac co-accessibility predictions."""
    df = _read_table(input_path)
    df = _rename(df, {
        "chr": "chrom", "chrom": "chrom",
        "gene": "gene_id", "gene_id": "gene_id", "TargetGene": "gene_id",
        "cell_type": "context_id", "context_id": "context_id", "CellType": "context_id",
        "score": "score", "cor": "score", "p": "score",
    })
    if "element_id" not in df.columns and {"chrom", "start", "end"}.issubset(df.columns):
        df = df.with_columns(
            (pl.col("chrom") + ":" + pl.col("start").cast(pl.Utf8)
             + "-" + pl.col("end").cast(pl.Utf8)).alias("element_id")
        )
    df = _coerce_score(df, "score")
    return _ensure_canonical(df, source="signac")


def import_archr_predictions(input_path: str | Path, candidate_df: pl.DataFrame) -> pl.DataFrame:
    """Import ArchR peak-to-gene linkage predictions."""
    df = _read_table(input_path)
    df = _rename(df, {
        "chr": "chrom", "chrom": "chrom",
        "gene": "gene_id", "gene_id": "gene_id", "TargetGene": "gene_id",
        "cell_type": "context_id", "context_id": "context_id", "CellType": "context_id",
        "score": "score", "pval": "score", "p": "score", "cor": "score",
    })
    if "element_id" not in df.columns and {"chrom", "start", "end"}.issubset(df.columns):
        df = df.with_columns(
            (pl.col("chrom") + ":" + pl.col("start").cast(pl.Utf8)
             + "-" + pl.col("end").cast(pl.Utf8)).alias("element_id")
        )
    df = _coerce_score(df, "score")
    return _ensure_canonical(df, source="archr")


def import_cicero_predictions(input_path: str | Path, candidate_df: pl.DataFrame) -> pl.DataFrame:
    """Import Cicero co-accessibility predictions."""
    df = _read_table(input_path)
    df = _rename(df, {
        "chr": "chrom", "chrom": "chrom",
        "gene": "gene_id", "gene_id": "gene_id", "TargetGene": "gene_id",
        "cell_type": "context_id", "context_id": "context_id", "CellType": "context_id",
        "score": "score", "coaccess": "score", "p": "score",
    })
    if "element_id" not in df.columns and {"chrom", "start", "end"}.issubset(df.columns):
        df = df.with_columns(
            (pl.col("chrom") + ":" + pl.col("start").cast(pl.Utf8)
             + "-" + pl.col("end").cast(pl.Utf8)).alias("element_id")
        )
    df = _coerce_score(df, "score")
    return _ensure_canonical(df, source="cicero")


# Registry mapping model_id → importer function.
IMPORTER_REGISTRY: Dict[str, Any] = {
    "abc": import_abc_predictions,
    "encode_re2g": import_re2g_predictions,
    "sce2g_atac": import_sce2g_predictions,
    "sce2g_multiome": import_sce2g_predictions,
    "sce2g": import_sce2g_predictions,
    "epimap": import_epimap_predictions,
    "graphreg": import_graphreg_predictions,
    "pgboost": import_pgboost_predictions,
    "scent": import_scent_predictions,
    "signac": import_signac_predictions,
    "archr": import_archr_predictions,
    "cicero": import_cicero_predictions,
}


# --------------------------------------------------------------------------- #
# E2G → V2G conversion
# --------------------------------------------------------------------------- #
def e2g_to_v2g(
    e2g_df: pl.DataFrame,
    variants_df: pl.DataFrame,
    aggregation: Aggregation = "max",
) -> pl.DataFrame:
    """Convert element-to-gene scores to variant-to-gene scores.

    For each (variant, gene) pair, the V2G score is the aggregation of the E2G
    scores of all elements that **overlap** the variant and link to that gene:

        score(v, g) = agg_{E : v overlaps E, E→g} score(E, g)

    ``aggregation`` is ``"max"`` (main figure), ``"sum"`` or ``"mean"``
    (supplementary).  Variants with no overlapping element for a gene receive
    no row here — use :func:`handle_missing_prediction` to fill them.

    Parameters
    ----------
    e2g_df:
        Canonical E2G frame with ``element_id`` (``chrom:start-end``),
        ``gene_id``, ``context_id``, ``score``.
    variants_df:
        Variant frame with ``variant_id``, ``chrom``, ``pos``.
    aggregation:
        How to combine multiple element scores for one (variant, gene).
    """
    required_e2g = {"element_id", "gene_id", "context_id", "score"}
    if not required_e2g.issubset(e2g_df.columns):
        raise ValueError(f"e2g_df missing columns: {required_e2g - set(e2g_df.columns)}")
    if not {"variant_id", "chrom", "pos"}.issubset(variants_df.columns):
        raise ValueError("variants_df must have variant_id, chrom, pos")

    # Parse element_id "chrom:start-end" into coordinates.
    parts = e2g_df.select(
        pl.col("element_id").str.split_exact(":", 1).struct.field("field_0").alias("e_chrom"),
        pl.col("element_id").str.split_exact(":", 1).struct.field("field_1").alias("rest"),
    ).with_columns(
        pl.col("rest").str.split_exact("-", 1).struct.field("field_0").cast(pl.Int64).alias("e_start"),
        pl.col("rest").str.split_exact("-", 1).struct.field("field_1").cast(pl.Int64).alias("e_end"),
    ).drop("rest")

    e2g = pl.concat([e2g_df, parts], how="horizontal")

    # Join variants to elements on chromosome, then filter by overlap.
    joined = variants_df.select("variant_id", "chrom", "pos").join(
        e2g.select("e_chrom", "e_start", "e_end", "gene_id", "context_id", "score"),
        left_on="chrom", right_on="e_chrom", how="inner",
    ).filter(
        (pl.col("pos") >= pl.col("e_start")) & (pl.col("pos") <= pl.col("e_end"))
    )

    if len(joined) == 0:
        # No overlaps at all — return an empty canonical frame.
        return pl.DataFrame(schema={
            "variant_id": pl.Utf8, "gene_id": pl.Utf8, "context_id": pl.Utf8,
            "score": pl.Float64,
        })

    agg_expr = {
        "max": pl.col("score").max(),
        "sum": pl.col("score").sum(),
        "mean": pl.col("score").mean(),
    }[aggregation]

    return (
        joined.group_by(["variant_id", "gene_id", "context_id"])
        .agg(agg_expr.alias("score"))
        .select("variant_id", "gene_id", "context_id", "score")
    )


# --------------------------------------------------------------------------- #
# Missing-prediction handling
# --------------------------------------------------------------------------- #
def handle_missing_prediction(candidate_df: pl.DataFrame, model_id: str) -> pl.DataFrame:
    """Build prediction rows for candidate pairs a model did not cover.

    Sets ``coverage = 0``, ``ranking_score = 0.0`` and
    ``applicability = "NOT_APPLICABLE_MISSING_DATA"`` so that downstream
    ranking treats uncovered pairs as the weakest possible links while still
    keeping them in the evaluation table.
    """
    return candidate_df.select(
        pl.lit(model_id).alias("model_id"),
        pl.lit("e2g").alias("model_family"),
        pl.lit(None, dtype=pl.Utf8).alias("benchmark_id"),
        pl.col("variant_id"),
        pl.lit(None, dtype=pl.Utf8).alias("element_id"),
        pl.col("gene_id"),
        pl.col("context_id"),
        pl.lit(0.0).cast(pl.Float64).alias("raw_score"),
        pl.lit(0.0).cast(pl.Float64).alias("ranking_score"),
        pl.lit(None, dtype=pl.Float64).alias("signed_score"),
        pl.lit(0).cast(pl.Int64).alias("coverage"),
        pl.lit("NOT_APPLICABLE_MISSING_DATA").alias("applicability"),
        pl.lit("published_prediction").alias("source_mode"),
    )


# --------------------------------------------------------------------------- #
# Generic adapter
# --------------------------------------------------------------------------- #
class PublishedPredictionAdapter(ModelAdapter):
    """Generic adapter wrapping a published-prediction importer.

    Parameters
    ----------
    model_id:
        Key into :data:`IMPORTER_REGISTRY`.
    config:
        Must contain ``input_path`` (the prediction file) and may contain
        ``aggregation`` (default ``"max"``) for the E2G→V2G step.
    """

    def __init__(self, model_id: str, config: Dict[str, Any]):
        family = config.get("model_family", "e2g")
        super().__init__(model_id=model_id, model_family=family, config=config)
        if model_id not in IMPORTER_REGISTRY:
            raise ValueError(
                f"No importer registered for model_id='{model_id}'. "
                f"Known: {sorted(IMPORTER_REGISTRY)}"
            )
        self._importer = IMPORTER_REGISTRY[model_id]
        self._e2g_cache: Optional[pl.DataFrame] = None

    # -- ModelAdapter interface ------------------------------------------- #
    def validate_resources(self) -> bool:
        path = self.config.get("input_path")
        if path is None:
            return False
        return Path(path).exists()

    def applicability(self, context_id: str) -> str:
        contexts = self.config.get("applicable_contexts")
        if contexts is None:
            return "APPLICABLE"
        return "APPLICABLE" if context_id in set(contexts) else "NOT_APPLICABLE_CONTEXT"

    def _load_e2g(self, candidate_df: pl.DataFrame) -> pl.DataFrame:
        if self._e2g_cache is None:
            self._e2g_cache = self._importer(self.config["input_path"], candidate_df)
        return self._e2g_cache

    def score(self, inputs: Dict[str, Any]) -> pl.DataFrame:
        candidate_df: pl.DataFrame = inputs["candidate_df"]
        variants_df: pl.DataFrame = inputs["variants_df"]
        e2g = self._load_e2g(candidate_df)
        agg = self.config.get("aggregation", "max")
        v2g = e2g_to_v2g(e2g, variants_df, aggregation=agg)

        # Attach to candidate pairs; fill missing with coverage=0.
        covered = candidate_df.join(
            v2g, on=["variant_id", "gene_id", "context_id"], how="left",
        )
        missing_mask = pl.col("score").is_null()
        covered = covered.with_columns(
            pl.when(missing_mask).then(0).otherwise(1).alias("coverage"),
            pl.when(missing_mask).then(0.0).otherwise(pl.col("score")).alias("ranking_score"),
            pl.when(missing_mask).then(pl.lit("NOT_APPLICABLE_MISSING_DATA"))
            .otherwise(pl.lit("APPLICABLE")).alias("applicability"),
        )

        out = covered.select(
            pl.lit(self.model_id).alias("model_id"),
            pl.lit(self.model_family).alias("model_family"),
            pl.lit(None, dtype=pl.Utf8).alias("benchmark_id"),
            pl.col("variant_id"),
            pl.lit(None, dtype=pl.Utf8).alias("element_id"),
            pl.col("gene_id"),
            pl.col("context_id"),
            pl.col("ranking_score").alias("raw_score"),
            pl.col("ranking_score"),
            pl.lit(None, dtype=pl.Float64).alias("signed_score"),
            pl.col("coverage").cast(pl.Int64),
            pl.col("applicability"),
            pl.lit("published_prediction").alias("source_mode"),
        )
        return self.normalize_score(out)

    def normalize_score(self, df: pl.DataFrame) -> pl.DataFrame:
        # Published scores are used as-is; higher = stronger link by convention.
        return df
