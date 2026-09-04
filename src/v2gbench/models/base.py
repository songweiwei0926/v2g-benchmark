"""Abstract base class for all model adapters.

Every model in the V2G-Benchmark-OneShot benchmark — baselines, published
prediction importers, sequence models, disease prioritizers and integrated
ensembles — implements the :class:`ModelAdapter` interface defined here.

The interface is intentionally minimal so that the benchmark runner can treat
all models uniformly:

* ``validate_resources``  — preflight check (weights / data / API key present).
* ``applicability``       — decide whether a model is applicable for a context.
* ``score``               — produce raw predictions for candidate pairs.
* ``normalize_score``     — map raw scores to a single ``ranking_score``.
* ``qc``                  — lightweight quality-control on the predictions.

The canonical prediction schema (see :mod:`v2gbench.schemas.prediction`)
requires the columns ``model_id``, ``model_family``, ``benchmark_id``,
``variant_id``/``element_id``, ``gene_id``, ``context_id``, ``raw_score``,
``ranking_score``, ``signed_score``, ``coverage``, ``applicability`` and
``source_mode``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import polars as pl

# Re-export the applicability vocabulary so adapters can reference it without
# importing the schema package directly (avoids a circular import in some
# downstream test setups).
APPLICABILITY_STATUS = (
    "APPLICABLE",
    "NOT_APPLICABLE_CONTEXT",
    "NOT_APPLICABLE_VARIANT",
    "NOT_APPLICABLE_ELEMENT",
    "NOT_APPLICABLE_MISSING_DATA",
)

SOURCE_MODES = (
    "published_prediction",
    "local_inference",
    "remote_inference",
    "derived_baseline",
    "derived_ensemble",
)


class ModelAdapter(ABC):
    """Base class for all model adapters.

    Parameters
    ----------
    model_id:
        Stable identifier of the model (matches the key in ``models.yaml``).
    model_family:
        Family label, e.g. ``baseline``, ``e2g``, ``sequence``, ``ensemble``.
    config:
        Model-specific configuration dictionary (loaded from ``models.yaml``).
    """

    def __init__(self, model_id: str, model_family: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.model_family = model_family
        self.config = config

    # ------------------------------------------------------------------ #
    # Abstract interface
    # ------------------------------------------------------------------ #
    @abstractmethod
    def validate_resources(self) -> bool:
        """Check that all required resources (weights, data, API) are available.

        Returns ``True`` when the model is ready to run, ``False`` otherwise.
        Implementations should be cheap (no network calls beyond a HEAD
        request) and side-effect free.
        """
        ...

    @abstractmethod
    def applicability(self, context_id: str) -> str:
        """Determine if the model is applicable for a given context.

        Returns one of the values in :data:`APPLICABILITY_STATUS`.
        """
        ...

    @abstractmethod
    def score(self, inputs: Dict[str, Any]) -> pl.DataFrame:
        """Score variant-gene pairs.

        ``inputs`` is a dictionary that minimally contains ``candidate_df``
        (a :class:`polars.DataFrame` conforming to the candidate schema) and
        may contain additional model-specific keys (e.g. ``variants_df``,
        ``gene_master_df``).  Returns a DataFrame with the prediction-schema
        columns.
        """
        ...

    @abstractmethod
    def normalize_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize raw scores to ``ranking_score`` (higher = stronger link)."""
        ...

    # ------------------------------------------------------------------ #
    # Shared QC
    # ------------------------------------------------------------------ #
    def qc(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Run QC on predictions.

        Returns a dict with NaN fraction, Inf count, score range and the number
        of unique scores.  An empty frame yields ``status="EMPTY"``.
        """
        n = len(df)
        if n == 0:
            return {"status": "EMPTY", "n_rows": 0}
        if "ranking_score" not in df.columns:
            return {"status": "ERROR", "n_rows": n, "missing_column": "ranking_score"}

        col = df["ranking_score"]
        nan_frac = col.null_count() / n
        # ``is_infinite`` only exists for float dtypes; guard for safety.
        try:
            inf_count = int(col.is_infinite().sum())
        except Exception:
            inf_count = 0

        status = "PASS" if (nan_frac < 0.5 and inf_count == 0) else "WARN"
        return {
            "status": status,
            "n_rows": n,
            "nan_fraction": float(nan_frac),
            "inf_count": inf_count,
            "score_min": float(col.min()),
            "score_max": float(col.max()),
            "n_unique_scores": int(col.n_unique()),
        }
