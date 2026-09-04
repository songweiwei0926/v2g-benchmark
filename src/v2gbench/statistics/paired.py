"""Paired model comparison with bootstrap and FDR correction.

This module provides all-pairs bootstrap comparison of models on a shared
evaluation set, plus Benjamini–Hochberg FDR correction for the resulting
family of p-values.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import polars as pl

from .bootstrap import (
    bootstrap_paired,
    compute_bootstrap_p_value,
    compute_ci,
)

__all__ = [
    "pairwise_comparison",
    "benjamini_hochberg",
    "format_comparison_results",
]

MetricFn = Callable[[pl.DataFrame], float]


def _split_by_model(df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
    """Split a combined DataFrame into per-model sub-frames."""
    if "model_id" not in df.columns:
        raise ValueError("pairwise_comparison requires a 'model_id' column")
    out: Dict[str, pl.DataFrame] = {}
    for key, sub in df.group_by("model_id", maintain_order=True):
        # polars returns the group key as a tuple (one element per grouping
        # column); unwrap the single-column case to a plain string.
        model_id = key[0] if isinstance(key, tuple) else key
        out[str(model_id)] = sub
    return out


def pairwise_comparison(
    df: pl.DataFrame,
    metric_fn: MetricFn,
    n_replicates: int = 2000,
    seed: int = 20260904,
    unit: str = "variant_id",
) -> List[Dict]:
    """All-pairs bootstrap comparison of models.

    For every unordered pair of models ``(A, B)`` present in ``df`` (column
    ``model_id``), compute the paired bootstrap distribution of
    ``delta = metric(A) - metric(B)`` and derive:

    * ``delta_mean`` – mean of the bootstrap deltas (point estimate)
    * ``ci_low``, ``ci_high`` – 95% percentile CI of the delta
    * ``p_value`` – two-sided bootstrap p-value against delta == 0

    Parameters
    ----------
    df:
        Combined evaluation DataFrame with a ``model_id`` column.
    metric_fn:
        Callable mapping a per-model DataFrame to a scalar metric.
    n_replicates:
        Number of bootstrap replicates per pair.
    seed:
        Base RNG seed (per-replicate seeds derived deterministically).
    unit:
        Sampling-unit column (default ``"variant_id"``).

    Returns
    -------
    list[dict]
        One dict per model pair with keys ``model_a``, ``model_b``,
        ``delta_mean``, ``ci_low``, ``ci_high``, ``p_value``.
    """
    per_model = _split_by_model(df)
    model_ids = list(per_model.keys())
    results: List[Dict] = []
    for a, b in combinations(model_ids, 2):
        deltas = bootstrap_paired(
            per_model[a],
            per_model[b],
            metric_fn,
            n_replicates=n_replicates,
            unit=unit,
            seed=seed,
        )
        valid = deltas[~np.isnan(deltas)]
        if valid.size == 0:
            delta_mean = float("nan")
            ci_low, ci_high = float("nan"), float("nan")
            p_value = float("nan")
        else:
            delta_mean = float(np.mean(valid))
            ci_low, ci_high = compute_ci(valid)
            p_value = compute_bootstrap_p_value(valid, null=0.0)
        results.append(
            {
                "model_a": a,
                "model_b": b,
                "delta_mean": delta_mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
            }
        )
    return results


def benjamini_hochberg(
    p_values: Sequence[float], q: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg FDR correction.

    Parameters
    ----------
    p_values:
        Sequence of raw p-values.
    q:
        Target false-discovery rate.

    Returns
    -------
    (q_values, significant) : tuple of np.ndarray
        ``q_values`` are the BH-adjusted p-values (monotonicised), and
        ``significant`` is a boolean array flagging rejections at level ``q``.
        NaN p-values are treated as non-significant with q-value NaN.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    q_values = np.full(n, np.nan, dtype=float)
    significant = np.zeros(n, dtype=bool)
    if n == 0:
        return q_values, significant

    valid_mask = ~np.isnan(p)
    valid_idx = np.where(valid_mask)[0]
    valid_p = p[valid_mask]
    m = valid_p.size
    if m == 0:
        return q_values, significant

    order = np.argsort(valid_p)
    ranked = valid_p[order]
    # Raw BH q-values: p_i * m / i  (i is 1-indexed rank).
    raw_q = ranked * m / np.arange(1, m + 1)
    # Monotonise from the largest rank downwards (cumulative min reversed).
    raw_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    raw_q = np.clip(raw_q, 0.0, 1.0)

    # Map back to original positions.
    q_valid = np.empty(m, dtype=float)
    q_valid[order] = raw_q
    q_values[valid_idx] = q_valid
    significant[valid_idx] = q_valid <= q
    return q_values, significant


def format_comparison_results(comparisons: List[Dict]) -> pl.DataFrame:
    """Format pairwise comparison results as a tidy DataFrame.

    Adds Benjamini–Hochberg q-values and a ``significant`` flag (at q = 0.05)
    to the list of comparison dicts produced by :func:`pairwise_comparison`.

    Returns
    -------
    pl.DataFrame
        Columns: ``model_a``, ``model_b``, ``delta_mean``, ``ci_low``,
        ``ci_high``, ``p_value``, ``q_value``, ``significant``.
    """
    if not comparisons:
        return pl.DataFrame(
            schema={
                "model_a": pl.Utf8,
                "model_b": pl.Utf8,
                "delta_mean": pl.Float64,
                "ci_low": pl.Float64,
                "ci_high": pl.Float64,
                "p_value": pl.Float64,
                "q_value": pl.Float64,
                "significant": pl.Boolean,
            }
        )

    p_values = [c["p_value"] for c in comparisons]
    q_values, significant = benjamini_hochberg(p_values, q=0.05)

    rows = []
    for c, qv, sig in zip(comparisons, q_values, significant):
        rows.append(
            {
                "model_a": c["model_a"],
                "model_b": c["model_b"],
                "delta_mean": c["delta_mean"],
                "ci_low": c["ci_low"],
                "ci_high": c["ci_high"],
                "p_value": c["p_value"],
                "q_value": float(qv) if not np.isnan(qv) else None,
                "significant": bool(sig),
            }
        )
    return pl.DataFrame(rows)
