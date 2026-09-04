"""Open Targets L2G importer (Family 4 — disease prioritization, Track C only).

Open Targets' "Locus-to-Gene" (L2G) model prioritizes genes at GWAS loci
using a gradient-boosted classifier over feature sets (distance, eQTL,
pQTL, chromatin interaction, etc.).  Because L2G is trained on GWAS–gene
evidence it is only eligible for **Track C** (disease/trait loci) of the
benchmark, not for the CRISPR/PerturbSeq gold-standard tracks.

Two versions are supported:

* ``2021`` — the original published L2G (fixed feature set, parquet dump).
* ``current`` — the latest Open Targets Platform export.

Both share a common schema (``study_id, variant_id, gene_id, l2g_score``)
which we map onto the canonical prediction columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import polars as pl

from .base import ModelAdapter


def _read_table(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pl.read_parquet(path)
    if suffix in (".tsv", ".txt"):
        return pl.read_csv(path, separator="\t")
    if suffix == ".csv":
        return pl.read_csv(path)
    return pl.read_csv(path, separator="\t")


def _normalize_l2g(df: pl.DataFrame, version: str) -> pl.DataFrame:
    """Map an L2G export onto canonical columns."""
    rename = {
        "studyId": "study_id", "study_id": "study_id",
        "variantId": "variant_id", "variant_id": "variant_id",
        "geneId": "gene_id", "gene_id": "gene_id",
        "Y": "l2g_score", "l2g_score": "l2g_score", "score": "l2g_score",
        "l2g": "l2g_score",
    }
    df = df.rename({k: v for k, v in rename.items() if k in df.columns})
    if "l2g_score" not in df.columns:
        raise ValueError(f"L2G export missing a score column; have {df.columns}")
    df = df.with_columns(pl.col("l2g_score").cast(pl.Float64))
    # Context for L2G is the study/trait.
    if "context_id" not in df.columns:
        ctx = df["study_id"] if "study_id" in df.columns else pl.lit("disease")
        df = df.with_columns(ctx.alias("context_id"))
    return df.select(
        pl.col("variant_id"),
        pl.col("gene_id"),
        pl.col("context_id"),
        pl.col("l2g_score").alias("score"),
    ).with_columns(pl.lit(version).alias("version"))


def import_l2g_2021(input_path: str | Path, candidate_df: pl.DataFrame) -> pl.DataFrame:
    """Import the 2021 L2G export and restrict to candidate pairs.

    The 2021 export is a single parquet/tsv with one row per
    (study, variant, gene).  We join to ``candidate_df`` on
    ``(variant_id, gene_id)`` so only benchmarked pairs are retained.
    """
    df = _read_table(input_path)
    norm = _normalize_l2g(df, version="2021")
    if candidate_df is not None and len(candidate_df) > 0:
        norm = norm.join(
            candidate_df.select("variant_id", "gene_id", "context_id"),
            on=["variant_id", "gene_id"], how="inner",
        ).drop("context_id_right") if "context_id_right" in norm.columns else norm
    return norm


def import_l2g_current(input_path: str | Path, candidate_df: pl.DataFrame) -> pl.DataFrame:
    """Import the current Open Targets L2G export."""
    df = _read_table(input_path)
    norm = _normalize_l2g(df, version="current")
    if candidate_df is not None and len(candidate_df) > 0:
        norm = norm.join(
            candidate_df.select("variant_id", "gene_id"),
            on=["variant_id", "gene_id"], how="inner",
        )
    return norm


class OpenTargetsL2GAdapter(ModelAdapter):
    """ModelAdapter for Open Targets L2G (Track C only).

    Parameters
    ----------
    config:
        Must contain ``version`` (``"2021"`` or ``"current"``) and
        ``input_path``.
    """

    def __init__(self, config: Dict[str, Any]):
        version = config.get("version", "current")
        model_id = config.get("model_id", f"opentargets_l2g_{version}")
        super().__init__(model_id=model_id, model_family="disease", config=config)
        self.version = version
        self._importer = import_l2g_2021 if version == "2021" else import_l2g_current
        self._cache: pl.DataFrame | None = None

    def validate_resources(self) -> bool:
        path = self.config.get("input_path")
        return bool(path) and Path(path).exists()

    def applicability(self, context_id: str) -> str:
        # L2G is only applicable to disease/trait (Track C) contexts.
        track = self.config.get("track", "C")
        if str(track).upper() != "C":
            return "NOT_APPLICABLE_CONTEXT"
        return "APPLICABLE"

    def _load(self, candidate_df: pl.DataFrame) -> pl.DataFrame:
        if self._cache is None:
            self._cache = self._importer(self.config["input_path"], candidate_df)
        return self._cache

    def score(self, inputs: Dict[str, Any]) -> pl.DataFrame:
        candidate_df: pl.DataFrame = inputs["candidate_df"]
        l2g = self._load(candidate_df)

        joined = candidate_df.join(
            l2g.select("variant_id", "gene_id", "score"),
            on=["variant_id", "gene_id"], how="left",
        )
        missing = pl.col("score").is_null()
        out = joined.select(
            pl.lit(self.model_id).alias("model_id"),
            pl.lit(self.model_family).alias("model_family"),
            pl.lit(None, dtype=pl.Utf8).alias("benchmark_id"),
            pl.col("variant_id"),
            pl.lit(None, dtype=pl.Utf8).alias("element_id"),
            pl.col("gene_id"),
            pl.col("context_id"),
            pl.when(missing).then(0.0).otherwise(pl.col("score")).alias("raw_score"),
            pl.when(missing).then(0.0).otherwise(pl.col("score")).alias("ranking_score"),
            pl.lit(None, dtype=pl.Float64).alias("signed_score"),
            pl.when(missing).then(0).otherwise(1).cast(pl.Int64).alias("coverage"),
            pl.when(missing).then(pl.lit("NOT_APPLICABLE_MISSING_DATA"))
            .otherwise(pl.lit("APPLICABLE")).alias("applicability"),
            pl.lit("published_prediction").alias("source_mode"),
        )
        return self.normalize_score(out)

    def normalize_score(self, df: pl.DataFrame) -> pl.DataFrame:
        # L2G score is already a probability in [0, 1]; higher = stronger.
        return df
