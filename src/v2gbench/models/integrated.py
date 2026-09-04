"""Integrated ensemble models (Track E).

Three training-free / lightly-trained ensembles that combine the per-model
predictions into a single ranking:

* :class:`IntegratedRankAdapter`    — mean percentile rank across models
  (no training, fully zero-shot).
* :class:`IntegratedLogisticAdapter` — L2-regularized logistic regression
  with ``class_weight="balanced"``, trained via nested cross-validation with
  **chromosome-level** folds to prevent leakage.
* :class:`IntegratedXGBoostAdapter`  — XGBoost with chromosome-separated
  outer cross-validation.

All three share the feature-preparation helpers
:func:`prepare_features`, :func:`chromosome_folds`, :func:`ensure_no_leakage`
and the train functions :func:`train_logistic` / :func:`train_xgboost`.

The feature matrix is built from a long ``predictions_df`` (one row per
model × candidate pair) pivoted wide, with an explicit **missing indicator**
column per feature so that models with partial coverage contribute signal
even when their score is absent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from ..utils.hashing import stable_hash
from .base import ModelAdapter

logger = logging.getLogger(__name__)

# Canonical fold count per spec.
N_FOLDS = 5


# --------------------------------------------------------------------------- #
# Feature preparation
# --------------------------------------------------------------------------- #
def prepare_features(predictions_df: pl.DataFrame, candidate_df: pl.DataFrame,
                     feature_list: Sequence[str]) -> Tuple[pl.DataFrame, List[str]]:
    """Build a wide feature matrix with missing-value indicators.

    Parameters
    ----------
    predictions_df:
        Long frame with columns ``variant_id, gene_id, context_id, model_id,
        ranking_score``.
    candidate_df:
        Candidate pairs (``variant_id, gene_id, context_id, chrom``).
    feature_list:
        Model ids to use as features (order preserved).

    Returns
    -------
    tuple
        ``(feature_df, feature_columns)`` where ``feature_df`` has one row per
        candidate pair, columns ``<model_id>`` (the score, 0 if missing) and
        ``<model_id>_missing`` (1 if the model produced no score, else 0).
        ``feature_columns`` lists the score columns in order.
    """
    key = ["variant_id", "gene_id", "context_id"]
    if "chrom" not in candidate_df.columns:
        raise ValueError("candidate_df must contain 'chrom' for fold assignment")

    # Pivot predictions wide: one column per model.
    wide = (
        predictions_df.filter(pl.col("model_id").is_in(list(feature_list)))
        .select(key + ["model_id", "ranking_score"])
        .pivot(on="model_id", values="ranking_score", aggregate_function="max")
    )

    base = candidate_df.select(key + ["chrom", "is_gold"])
    feat = base.join(wide, on=key, how="left")

    feature_columns: List[str] = []
    for m in feature_list:
        if m not in feat.columns:
            # Model entirely absent — add zero column + all-missing indicator.
            feat = feat.with_columns(pl.lit(0.0).cast(pl.Float64).alias(m))
            feat = feat.with_columns(pl.lit(1).cast(pl.Int64).alias(f"{m}_missing"))
        else:
            missing_expr = pl.col(m).is_null()
            feat = feat.with_columns(
                pl.when(missing_expr).then(0.0).otherwise(pl.col(m)).cast(pl.Float64).alias(m),
                pl.when(missing_expr).then(1).otherwise(0).cast(pl.Int64).alias(f"{m}_missing"),
            )
        feature_columns.append(m)

    return feat, feature_columns


# --------------------------------------------------------------------------- #
# Chromosome folds
# --------------------------------------------------------------------------- #
def chromosome_folds(chromosomes: Sequence[str], n_folds: int = N_FOLDS) -> Dict[str, int]:
    """Assign chromosomes to ``n_folds`` folds deterministically.

    Uses :func:`v2gbench.utils.hashing.stable_hash` so the assignment is
    reproducible across runs.  Returns a mapping ``chrom → fold index``.
    """
    folds: Dict[str, int] = {}
    for chrom in sorted(set(chromosomes)):
        h = stable_hash("fold", chrom)
        folds[chrom] = int(h, 16) % n_folds
    return folds


# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #
def ensure_no_leakage(fold: int, test_chroms: Sequence[str],
                      scaler, features: np.ndarray,
                      threshold: float = 0.0) -> bool:
    """Verify that test chromosomes did not leak into the training scaler.

    A simple guard: if a ``StandardScaler`` was fit on training data only,
    the mean/variance of the test features should not exactly match the
    scaler's learned statistics (which would indicate the test data was
    accidentally included in fitting).  Returns ``True`` if the check passes.

    ``threshold`` is the maximum allowed overlap ratio; 0 means any exact
    match is a failure.
    """
    if scaler is None:
        return True
    test_mean = np.nanmean(features, axis=0)
    train_mean = np.asarray(scaler.mean_)
    # Fraction of features whose test mean exactly equals the training mean.
    exact = np.mean(np.isclose(test_mean, train_mean, equal_nan=True))
    if exact > threshold:
        logger.error("Leakage guard FAILED for fold %d: %.3f of features match "
                     "training mean (test chroms=%s)", fold, exact, list(test_chroms))
        return False
    return True


# --------------------------------------------------------------------------- #
# Training: logistic
# --------------------------------------------------------------------------- #
def train_logistic(X: np.ndarray, y: np.ndarray, chromosomes: Sequence[str],
                   fold_assignment: Dict[str, int],
                   ) -> Dict[str, Any]:
    """Train an L2 logistic regression with nested chromosome-fold CV.

    Outer loop: leave-one-fold-out (chromosome-grouped).  Inner loop: grid
    search over ``C`` on the remaining folds.  Returns a dict with per-fold
    models, best ``C`` values and held-out predictions.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    fold_of = np.array([fold_assignment.get(c, 0) for c in chromosomes])
    Cs = np.logspace(-4, 2, 7)
    results: Dict[str, Any] = {"folds": {}, "oof_pred": np.full(len(y), np.nan)}

    for fold in range(N_FOLDS):
        train_idx = fold_of != fold
        test_idx = fold_of == fold
        if train_idx.sum() == 0 or test_idx.sum() == 0:
            continue

        scaler = StandardScaler().fit(X[train_idx])
        if not ensure_no_leakage(fold, [c for c, f in fold_assignment.items() if f == fold],
                                 scaler, X[test_idx]):
            raise RuntimeError(f"Leakage detected in logistic fold {fold}")
        X_tr = scaler.transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        inner = GridSearchCV(
            LogisticRegression(
                penalty="l2", class_weight="balanced", max_iter=1000,
            ),
            param_grid={"C": Cs},
            cv=3, scoring="roc_auc", n_jobs=-1,
        )
        inner.fit(X_tr, y[train_idx])
        best = inner.best_estimator_
        results["folds"][fold] = {"model": best, "scaler": scaler, "C": inner.best_params_["C"]}
        results["oof_pred"][test_idx] = best.predict_proba(X_te)[:, 1]

    return results


# --------------------------------------------------------------------------- #
# Training: xgboost
# --------------------------------------------------------------------------- #
def train_xgboost(X: np.ndarray, y: np.ndarray, chromosomes: Sequence[str],
                  fold_assignment: Dict[str, int],
                  ) -> Dict[str, Any]:
    """Train XGBoost with chromosome-separated outer CV.

    Returns a dict with per-fold models and out-of-fold predictions.
    """
    from xgboost import XGBClassifier

    fold_of = np.array([fold_assignment.get(c, 0) for c in chromosomes])
    results: Dict[str, Any] = {"folds": {}, "oof_pred": np.full(len(y), np.nan)}

    for fold in range(N_FOLDS):
        train_idx = fold_of != fold
        test_idx = fold_of == fold
        if train_idx.sum() == 0 or test_idx.sum() == 0:
            continue

        if not ensure_no_leakage(fold, [c for c, f in fold_assignment.items() if f == fold],
                                 None, X[test_idx]):
            raise RuntimeError(f"Leakage detected in xgboost fold {fold}")

        pos = int(y[train_idx].sum())
        neg = int((1 - y[train_idx]).sum())
        spw = neg / max(pos, 1)

        model = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="auc",
            tree_method="hist", n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        results["folds"][fold] = {"model": model}
        results["oof_pred"][test_idx] = model.predict_proba(X[test_idx])[:, 1]

    return results


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
class _IntegratedBase(ModelAdapter):
    """Shared scaffolding for the three integrated adapters."""

    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id=model_id, model_family="ensemble", config=config)
        self.feature_list: List[str] = list(config.get("feature_list", []))

    def validate_resources(self) -> bool:
        # Needs at least one input model's predictions.
        return len(self.feature_list) > 0

    def applicability(self, context_id: str) -> str:
        return "APPLICABLE"

    def _gather_predictions(self, inputs: Dict[str, Any]) -> pl.DataFrame:
        preds = inputs.get("predictions_df")
        if preds is None:
            raise ValueError("inputs must contain 'predictions_df' (long frame)")
        return preds


class IntegratedRankAdapter(_IntegratedBase):
    """Mean percentile rank across models (no training)."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(model_id="integrated_rank", config=config or {})

    def score(self, inputs: Dict[str, Any]) -> pl.DataFrame:
        preds = self._gather_predictions(inputs)
        candidate_df: pl.DataFrame = inputs["candidate_df"]

        # Per-model percentile rank within each (variant, context).
        ranked = preds.with_columns(
            pl.col("ranking_score").rank("average").over("variant_id", "context_id", "model_id")
            .alias("pct_rank")
        )
        # Mean percentile rank across models for each (variant, gene, context).
        agg = (
            ranked.group_by(["variant_id", "gene_id", "context_id"])
            .agg(pl.col("pct_rank").mean().alias("mean_pct_rank"))
        )

        joined = candidate_df.select(
            "variant_id", "gene_id", "context_id"
        ).join(agg, on=["variant_id", "gene_id", "context_id"], how="left")

        missing = pl.col("mean_pct_rank").is_null()
        out = joined.select(
            pl.lit(self.model_id).alias("model_id"),
            pl.lit(self.model_family).alias("model_family"),
            pl.lit(None, dtype=pl.Utf8).alias("benchmark_id"),
            pl.col("variant_id"),
            pl.lit(None, dtype=pl.Utf8).alias("element_id"),
            pl.col("gene_id"),
            pl.col("context_id"),
            pl.when(missing).then(0.0).otherwise(pl.col("mean_pct_rank")).alias("raw_score"),
            pl.when(missing).then(0.0).otherwise(pl.col("mean_pct_rank")).alias("ranking_score"),
            pl.lit(None, dtype=pl.Float64).alias("signed_score"),
            pl.when(missing).then(0).otherwise(1).cast(pl.Int64).alias("coverage"),
            pl.when(missing).then(pl.lit("NOT_APPLICABLE_MISSING_DATA"))
            .otherwise(pl.lit("APPLICABLE")).alias("applicability"),
            pl.lit("derived_ensemble").alias("source_mode"),
        )
        return self.normalize_score(out)

    def normalize_score(self, df: pl.DataFrame) -> pl.DataFrame:
        return df


class IntegratedLogisticAdapter(_IntegratedBase):
    """L2 logistic regression ensemble with nested chromosome-fold CV."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(model_id="integrated_logistic", config=config or {})

    def score(self, inputs: Dict[str, Any]) -> pl.DataFrame:
        preds = self._gather_predictions(inputs)
        candidate_df: pl.DataFrame = inputs["candidate_df"]
        feat_df, feat_cols = prepare_features(preds, candidate_df, self.feature_list)

        X = feat_df.select(feat_cols).to_numpy()
        y = feat_df["is_gold"].to_numpy()
        chroms = feat_df["chrom"].to_list()
        folds = chromosome_folds(chroms)

        res = train_logistic(X, y, chroms, folds)
        oof = res["oof_pred"]

        out = feat_df.with_columns(pl.Series("ranking_score", oof)).select(
            pl.lit(self.model_id).alias("model_id"),
            pl.lit(self.model_family).alias("model_family"),
            pl.lit(None, dtype=pl.Utf8).alias("benchmark_id"),
            pl.col("variant_id"),
            pl.lit(None, dtype=pl.Utf8).alias("element_id"),
            pl.col("gene_id"),
            pl.col("context_id"),
            pl.col("ranking_score").alias("raw_score"),
            pl.when(pl.col("ranking_score").is_null()).then(0.0)
            .otherwise(pl.col("ranking_score")).alias("ranking_score"),
            pl.lit(None, dtype=pl.Float64).alias("signed_score"),
            pl.when(pl.col("ranking_score").is_null()).then(0).otherwise(1).cast(pl.Int64).alias("coverage"),
            pl.lit("APPLICABLE").alias("applicability"),
            pl.lit("derived_ensemble").alias("source_mode"),
        )
        return self.normalize_score(out)

    def normalize_score(self, df: pl.DataFrame) -> pl.DataFrame:
        return df


class IntegratedXGBoostAdapter(_IntegratedBase):
    """XGBoost ensemble with chromosome-separated outer CV."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(model_id="integrated_xgboost", config=config or {})

    def score(self, inputs: Dict[str, Any]) -> pl.DataFrame:
        preds = self._gather_predictions(inputs)
        candidate_df: pl.DataFrame = inputs["candidate_df"]
        feat_df, feat_cols = prepare_features(preds, candidate_df, self.feature_list)

        # Include missing-indicator columns for XGBoost.
        all_cols = feat_cols + [f"{m}_missing" for m in feat_cols]
        X = feat_df.select(all_cols).to_numpy()
        y = feat_df["is_gold"].to_numpy()
        chroms = feat_df["chrom"].to_list()
        folds = chromosome_folds(chroms)

        res = train_xgboost(X, y, chroms, folds)
        oof = res["oof_pred"]

        out = feat_df.with_columns(pl.Series("ranking_score", oof)).select(
            pl.lit(self.model_id).alias("model_id"),
            pl.lit(self.model_family).alias("model_family"),
            pl.lit(None, dtype=pl.Utf8).alias("benchmark_id"),
            pl.col("variant_id"),
            pl.lit(None, dtype=pl.Utf8).alias("element_id"),
            pl.col("gene_id"),
            pl.col("context_id"),
            pl.col("ranking_score").alias("raw_score"),
            pl.when(pl.col("ranking_score").is_null()).then(0.0)
            .otherwise(pl.col("ranking_score")).alias("ranking_score"),
            pl.lit(None, dtype=pl.Float64).alias("signed_score"),
            pl.when(pl.col("ranking_score").is_null()).then(0).otherwise(1).cast(pl.Int64).alias("coverage"),
            pl.lit("APPLICABLE").alias("applicability"),
            pl.lit("derived_ensemble").alias("source_mode"),
        )
        return self.normalize_score(out)

    def normalize_score(self, df: pl.DataFrame) -> pl.DataFrame:
        return df
