"""Classification metrics for variant-to-gene prediction.

The input DataFrame is expected to contain, at minimum:

    label            : int   – ground-truth label in {0, 1, -1}
                               (1 = positive / gold, 0 = confident negative,
                                -1 / other = unknown / unlabeled)
    score            : float – model confidence / probability in [0, 1]
                               (used for threshold-based and curve metrics)

For curve metrics (AUPRC, AUROC) only rows with ``label`` in {0, 1} are used –
unknown labels are skipped.  For MCC a hard prediction is derived by thresholding
``score`` at ``threshold`` (default 0.5) and compared against the {0, 1} labels.

The evaluation code is model-agnostic: it never reads ``model_id``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)

__all__ = [
    "compute_auprc",
    "compute_auroc",
    "compute_mcc",
    "compute_all_classification_metrics",
]


def _labeled_pairs(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_score) restricted to confident labels in {0, 1}.

    Accepts either an ``is_gold`` column (int in {0,1}) or a ``label`` column
    (int in {-1, 0, 1}); unknowns (-1) are dropped.  The score column may be
    named ``score`` or ``ranking_score``.
    """
    if "label" in df.columns:
        label_col = "label"
    elif "is_gold" in df.columns:
        label_col = "is_gold"
    else:
        raise ValueError(
            "classification metrics require a 'label' or 'is_gold' column"
        )

    if "score" in df.columns:
        score_col = "score"
    elif "ranking_score" in df.columns:
        score_col = "ranking_score"
    else:
        raise ValueError(
            "classification metrics require a 'score' or 'ranking_score' column"
        )

    sub = df.filter(pl.col(label_col).is_in([0, 1]))
    if sub.is_empty():
        return np.array([]), np.array([])
    y_true = sub[label_col].cast(pl.Float64).to_numpy()
    y_score = sub[score_col].cast(pl.Float64).to_numpy()
    return y_true, y_score


def compute_auprc(df: pl.DataFrame) -> float:
    """Area Under the Precision-Recall Curve (average precision).

    Only positive (label 1) and confident negative (label 0) rows are used;
    unknown labels are skipped.  This is the primary classification metric.
    """
    y_true, y_score = _labeled_pairs(df)
    if y_true.size < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def compute_auroc(df: pl.DataFrame) -> float:
    """Area Under the ROC Curve.

    Only positive (label 1) and confident negative (label 0) rows are used.
    Returns NaN if fewer than two classes are present.
    """
    y_true, y_score = _labeled_pairs(df)
    if y_true.size < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_mcc(df: pl.DataFrame, threshold: float = 0.5) -> float:
    """Matthews Correlation Coefficient at a fixed score threshold.

    A hard prediction is derived as ``score >= threshold`` and compared against
    the {0, 1} labels.  Returns NaN if fewer than two classes are present in
    either the labels or the predictions.
    """
    y_true, y_score = _labeled_pairs(df)
    if y_true.size < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    y_pred = (y_score >= threshold).astype(int)
    if len(np.unique(y_pred)) < 2 and len(np.unique(y_true)) < 2:
        return float("nan")
    return float(matthews_corrcoef(y_true, y_pred))


def compute_all_classification_metrics(df: pl.DataFrame) -> Dict[str, float]:
    """Compute the full classification-metric suite.

    Returns a dict with keys: ``AUPRC``, ``AUROC``, ``MCC``.
    """
    return {
        "AUPRC": compute_auprc(df),
        "AUROC": compute_auroc(df),
        "MCC": compute_mcc(df),
    }
