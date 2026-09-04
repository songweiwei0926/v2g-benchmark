"""Baseline (Family 0) model adapters.

These are simple, training-free negative controls and distance-based
baselines.  They all derive a ``ranking_score`` directly from the candidate
table (which already carries ``distance_to_tss`` and, for the expressed
baseline, an expression flag) and therefore require no external resources.

Implemented baselines
---------------------
* :class:`RandomAdapter`            — SHA256(variant_id, gene_id, seed) → [0,1).
* :class:`NearestTSSAdapter`        — score = -distance_to_tss.
* :class:`InverseDistanceAdapter`   — score = 1 / (distance + 1).
* :class:`ExponentialDistanceAdapter` — score = exp(-distance / lambda).
* :class:`NearestExpressedAdapter`  — -distance_to_tss restricted to expressed genes.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import polars as pl

from ..schemas.prediction import APPLICABILITY_STATUS  # noqa: F401  (re-export convenience)
from ..utils.hashing import hash_to_float
from .base import ModelAdapter


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _prediction_frame(
    candidate_df: pl.DataFrame,
    ranking_score: pl.Expr,
    *,
    model_id: str,
    model_family: str,
    source_mode: str = "derived_baseline",
    raw_score: pl.Expr | None = None,
    signed_score: pl.Expr | None = None,
    coverage_expr: pl.Expr | None = None,
) -> pl.DataFrame:
    """Assemble a prediction-schema frame from a candidate table + score expr.

    Keeps the canonical columns and fills the ones a baseline does not produce
    (``benchmark_id``, ``element_id``, ``signed_score``) with sensible nulls.
    """
    if "variant_id" not in candidate_df.columns:
        raise ValueError("candidate_df must contain a 'variant_id' column")

    out = candidate_df.select(
        pl.lit(model_id).alias("model_id"),
        pl.lit(model_family).alias("model_family"),
        pl.lit(None, dtype=pl.Utf8).alias("benchmark_id"),
        pl.col("variant_id"),
        pl.lit(None, dtype=pl.Utf8).alias("element_id"),
        pl.col("gene_id"),
        pl.col("context_id"),
        (raw_score if raw_score is not None else ranking_score).cast(pl.Float64).alias("raw_score"),
        ranking_score.cast(pl.Float64).alias("ranking_score"),
        (signed_score if signed_score is not None else pl.lit(None, dtype=pl.Float64)).alias("signed_score"),
        (coverage_expr if coverage_expr is not None else pl.lit(1)).cast(pl.Int64).alias("coverage"),
        pl.lit("APPLICABLE").alias("applicability"),
        pl.lit(source_mode).alias("source_mode"),
    )
    return out


# --------------------------------------------------------------------------- #
# Random
# --------------------------------------------------------------------------- #
class RandomAdapter(ModelAdapter):
    """Stable random baseline.

    ``ranking_score = hash_to_float(variant_id, gene_id, seed)`` so the same
    candidate pair always receives the same score for a fixed seed, making the
    baseline fully reproducible.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(
            model_id="random",
            model_family="baseline",
            config=config or {},
        )
        self.seed = int(self.config.get("seed", 20260904))

    def validate_resources(self) -> bool:
        return True  # pure computation

    def applicability(self, context_id: str) -> str:
        return "APPLICABLE"

    def score(self, inputs: Dict[str, Any]) -> pl.DataFrame:
        candidate_df: pl.DataFrame = inputs["candidate_df"]
        seed = self.seed
        # polars map_elements keeps row order; hash_to_float is deterministic.
        score_expr = pl.struct(["variant_id", "gene_id"]).map_elements(
            lambda row: hash_to_float(row["variant_id"], row["gene_id"], seed),
            return_dtype=pl.Float64,
        )
        df = _prediction_frame(
            candidate_df, score_expr, model_id=self.model_id, model_family=self.model_family
        )
        return self.normalize_score(df)

    def normalize_score(self, df: pl.DataFrame) -> pl.DataFrame:
        # hash_to_float is already in [0, 1) and monotonic in "randomness".
        return df


# --------------------------------------------------------------------------- #
# Nearest TSS
# --------------------------------------------------------------------------- #
class NearestTSSAdapter(ModelAdapter):
    """Score = -distance_to_tss (closer gene → higher score)."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(model_id="nearest_tss", model_family="baseline", config=config or {})

    def validate_resources(self) -> bool:
        return True

    def applicability(self, context_id: str) -> str:
        return "APPLICABLE"

    def score(self, inputs: Dict[str, Any]) -> pl.DataFrame:
        candidate_df: pl.DataFrame = inputs["candidate_df"]
        if "distance_to_tss" not in candidate_df.columns:
            raise ValueError("candidate_df must contain 'distance_to_tss'")
        score_expr = -pl.col("distance_to_tss").cast(pl.Float64)
        df = _prediction_frame(
            candidate_df, score_expr, model_id=self.model_id, model_family=self.model_family
        )
        return self.normalize_score(df)

    def normalize_score(self, df: pl.DataFrame) -> pl.DataFrame:
        return df


# --------------------------------------------------------------------------- #
# Inverse distance
# --------------------------------------------------------------------------- #
class InverseDistanceAdapter(ModelAdapter):
    """Score = 1 / (distance_to_tss + 1)."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(model_id="inverse_distance", model_family="baseline", config=config or {})

    def validate_resources(self) -> bool:
        return True

    def applicability(self, context_id: str) -> str:
        return "APPLICABLE"

    def score(self, inputs: Dict[str, Any]) -> pl.DataFrame:
        candidate_df: pl.DataFrame = inputs["candidate_df"]
        if "distance_to_tss" not in candidate_df.columns:
            raise ValueError("candidate_df must contain 'distance_to_tss'")
        score_expr = 1.0 / (pl.col("distance_to_tss").cast(pl.Float64) + 1.0)
        df = _prediction_frame(
            candidate_df, score_expr, model_id=self.model_id, model_family=self.model_family
        )
        return self.normalize_score(df)

    def normalize_score(self, df: pl.DataFrame) -> pl.DataFrame:
        return df


# --------------------------------------------------------------------------- #
# Exponential distance
# --------------------------------------------------------------------------- #
class ExponentialDistanceAdapter(ModelAdapter):
    """Score = exp(-distance / lambda).

    ``lambda`` (the decay length in bp) is read from ``config["lambda"]``.
    Common values are 50_000, 100_000 and 250_000 bp; the model_id is derived
    from the configured lambda (e.g. ``exp_distance_100kb``).
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        lam = int(config.get("lambda", 100000))
        super().__init__(
            model_id=config.get("model_id", _exp_model_id(lam)),
            model_family="baseline",
            config={**config, "lambda": lam},
        )
        self.lambda_bp = lam

    def validate_resources(self) -> bool:
        return True

    def applicability(self, context_id: str) -> str:
        return "APPLICABLE"

    def score(self, inputs: Dict[str, Any]) -> pl.DataFrame:
        candidate_df: pl.DataFrame = inputs["candidate_df"]
        if "distance_to_tss" not in candidate_df.columns:
            raise ValueError("candidate_df must contain 'distance_to_tss'")
        lam = float(self.lambda_bp)
        score_expr = (-pl.col("distance_to_tss").cast(pl.Float64) / lam).exp()
        df = _prediction_frame(
            candidate_df, score_expr, model_id=self.model_id, model_family=self.model_family
        )
        return self.normalize_score(df)

    def normalize_score(self, df: pl.DataFrame) -> pl.DataFrame:
        return df


def _exp_model_id(lam: int) -> str:
    """Human-readable model id from a lambda in bp (50000 → exp_distance_50kb)."""
    kb = lam // 1000
    return f"exp_distance_{kb}kb"


# --------------------------------------------------------------------------- #
# Nearest expressed
# --------------------------------------------------------------------------- #
class NearestExpressedAdapter(ModelAdapter):
    """Like :class:`NearestTSSAdapter` but restricted to expressed genes.

    Candidate pairs whose gene is *not* expressed in the context receive
    ``ranking_score = -inf`` (i.e. ranked last) and ``coverage = 0``.  The
    expression flag is read from the ``is_expressed`` column of the candidate
    table; if that column is absent the adapter falls back to plain
    nearest-TSS behaviour and emits a warning via ``applicability``.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(model_id="nearest_expressed", model_family="baseline", config=config or {})

    def validate_resources(self) -> bool:
        return True

    def applicability(self, context_id: str) -> str:
        return "APPLICABLE"

    def score(self, inputs: Dict[str, Any]) -> pl.DataFrame:
        candidate_df: pl.DataFrame = inputs["candidate_df"]
        if "distance_to_tss" not in candidate_df.columns:
            raise ValueError("candidate_df must contain 'distance_to_tss'")

        if "is_expressed" in candidate_df.columns:
            expressed = pl.col("is_expressed").cast(pl.Boolean)
            score_expr = pl.when(expressed).then(
                -pl.col("distance_to_tss").cast(pl.Float64)
            ).otherwise(float("-inf"))
            coverage_expr = pl.when(expressed).then(1).otherwise(0)
            applic_expr = pl.when(expressed).then(
                pl.lit("APPLICABLE")
            ).otherwise(pl.lit("NOT_APPLICABLE_MISSING_DATA"))
        else:
            # Fallback: no expression information available.
            score_expr = -pl.col("distance_to_tss").cast(pl.Float64)
            coverage_expr = pl.lit(1)
            applic_expr = pl.lit("APPLICABLE")

        df = candidate_df.select(
            pl.lit(self.model_id).alias("model_id"),
            pl.lit(self.model_family).alias("model_family"),
            pl.lit(None, dtype=pl.Utf8).alias("benchmark_id"),
            pl.col("variant_id"),
            pl.lit(None, dtype=pl.Utf8).alias("element_id"),
            pl.col("gene_id"),
            pl.col("context_id"),
            score_expr.alias("raw_score"),
            score_expr.alias("ranking_score"),
            pl.lit(None, dtype=pl.Float64).alias("signed_score"),
            coverage_expr.cast(pl.Int64).alias("coverage"),
            applic_expr.alias("applicability"),
            pl.lit("derived_baseline").alias("source_mode"),
        )
        return self.normalize_score(df)

    def normalize_score(self, df: pl.DataFrame) -> pl.DataFrame:
        return df
