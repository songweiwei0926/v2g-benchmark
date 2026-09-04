"""Direction-of-effect metrics for variant-to-gene prediction.

These metrics evaluate whether a model correctly predicts the *sign* of a
variant's effect on a gene (up-regulating vs down-regulating).

Expected input columns:

    signed_score      : float – model-predicted signed effect
                                (positive = up, negative = down)
    effect_direction  : int   – ground-truth direction in {+1, -1}
                                (or {1, 0} which is mapped to {+1, -1})
    effect_size       : float – ground-truth magnitude of effect (optional,
                                used by ``compute_spearman``)

The evaluation code is model-agnostic: it never reads ``model_id``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

__all__ = [
    "compute_direction_accuracy",
    "compute_balanced_accuracy",
    "compute_direction_mcc",
    "compute_spearman",
]


def _sign(x: np.ndarray) -> np.ndarray:
    """Sign array returning +1 / -1 (zeros mapped to +1 by convention)."""
    s = np.sign(x)
    s[s == 0] = 1
    return s


def _direction_pairs(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_sign, true_sign) for rows with both columns present."""
    if "signed_score" not in df.columns:
        raise ValueError("direction metrics require a 'signed_score' column")
    if "effect_direction" not in df.columns:
        raise ValueError("direction metrics require an 'effect_direction' column")

    sub = df.filter(
        pl.col("signed_score").is_not_null()
        & pl.col("effect_direction").is_not_null()
    )
    if sub.is_empty():
        return np.array([]), np.array([])

    pred = sub["signed_score"].cast(pl.Float64).to_numpy()
    true = sub["effect_direction"].cast(pl.Float64).to_numpy()

    # Normalise {0, 1} encodings to {-1, +1}.
    true_sign = _sign(np.where(true == 0, -1.0, true))
    pred_sign = _sign(pred)
    return pred_sign, true_sign


def compute_direction_accuracy(df: pl.DataFrame) -> float:
    """Fraction of pairs where ``sign(signed_score) == sign(effect_direction)``."""
    pred, true = _direction_pairs(df)
    if pred.size == 0:
        return float("nan")
    return float(np.mean(pred == true))


def compute_balanced_accuracy(df: pl.DataFrame) -> float:
    """Balanced accuracy for direction prediction (up vs down).

    Balanced accuracy = (sensitivity + specificity) / 2, robust to class
    imbalance.  Returns NaN if fewer than two classes are present.
    """
    pred, true = _direction_pairs(df)
    if pred.size == 0 or len(np.unique(true)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(true, pred))


def compute_direction_mcc(df: pl.DataFrame) -> float:
    """Matthews Correlation Coefficient for direction (up/down) prediction."""
    pred, true = _direction_pairs(df)
    if pred.size == 0 or len(np.unique(true)) < 2:
        return float("nan")
    return float(matthews_corrcoef(true, pred))


def compute_spearman(df: pl.DataFrame) -> float:
    """Spearman correlation between ``signed_score`` and ``effect_size``.

    Requires an ``effect_size`` column.  Rows missing either value are dropped.
    Returns NaN if fewer than 3 valid pairs (Spearman is undefined otherwise).
    """
    if "effect_size" not in df.columns:
        raise ValueError("compute_spearman requires an 'effect_size' column")
    sub = df.filter(
        pl.col("signed_score").is_not_null()
        & pl.col("effect_size").is_not_null()
    )
    if sub.height < 3:
        return float("nan")
    rho, _ = spearmanr(
        sub["signed_score"].cast(pl.Float64).to_numpy(),
        sub["effect_size"].cast(pl.Float64).to_numpy(),
    )
    return float(rho)
