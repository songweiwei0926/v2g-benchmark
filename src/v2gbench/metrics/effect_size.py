"""Effect-size regression metrics for variant-to-gene prediction.

These metrics evaluate how well a model's predicted effect *magnitude* tracks
the ground-truth effect size.

Expected input columns:

    predicted_effect_size : float – model-predicted effect magnitude
    effect_size           : float – ground-truth effect magnitude

(If only ``signed_score`` is available it is used as the predicted effect size.)

The evaluation code is model-agnostic: it never reads ``model_id``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

__all__ = [
    "compute_pearson",
    "compute_spearman",
    "compute_r2",
]


def _effect_pairs(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted, actual) effect-size arrays for valid rows."""
    if "effect_size" not in df.columns:
        raise ValueError("effect-size metrics require an 'effect_size' column")

    if "predicted_effect_size" in df.columns:
        pred_col = "predicted_effect_size"
    elif "signed_score" in df.columns:
        pred_col = "signed_score"
    else:
        raise ValueError(
            "effect-size metrics require 'predicted_effect_size' "
            "or 'signed_score' column"
        )

    sub = df.filter(
        pl.col(pred_col).is_not_null() & pl.col("effect_size").is_not_null()
    )
    if sub.is_empty():
        return np.array([]), np.array([])
    pred = sub[pred_col].cast(pl.Float64).to_numpy()
    actual = sub["effect_size"].cast(pl.Float64).to_numpy()
    return pred, actual


def compute_pearson(df: pl.DataFrame) -> float:
    """Pearson correlation between predicted and actual effect size.

    Returns NaN if fewer than 3 valid pairs or if either vector has zero
    variance.
    """
    pred, actual = _effect_pairs(df)
    if pred.size < 3:
        return float("nan")
    if np.std(pred) == 0 or np.std(actual) == 0:
        return float("nan")
    r, _ = pearsonr(pred, actual)
    return float(r)


def compute_spearman(df: pl.DataFrame) -> float:
    """Spearman rank correlation between predicted and actual effect size.

    Returns NaN if fewer than 3 valid pairs.
    """
    pred, actual = _effect_pairs(df)
    if pred.size < 3:
        return float("nan")
    rho, _ = spearmanr(pred, actual)
    return float(rho)


def compute_r2(df: pl.DataFrame) -> float:
    """Coefficient of determination (R²) between predicted and actual effect size.

    R² = 1 - SS_res / SS_tot.  Returns NaN if there are fewer than 2 valid
    pairs or if the actual values have zero variance.
    """
    pred, actual = _effect_pairs(df)
    if pred.size < 2:
        return float("nan")
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)
