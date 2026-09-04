"""Bootstrap confidence intervals for V2G-Benchmark metrics.

The bootstrap resamples **sampling units** (variants / loci), not individual
(variant, gene) pairs.  This respects the paired structure of the data: all
candidate-gene rows for a resampled variant are kept together, preserving the
within-variant ranking that the metrics depend on.

Seeding is deterministic via :mod:`v2gbench.utils.hashing` so that bootstrap
replicates are reproducible across runs and machines.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np
import polars as pl

from ..utils.hashing import stable_hash

__all__ = [
    "bootstrap_metrics",
    "bootstrap_paired",
    "compute_ci",
    "compute_bootstrap_p_value",
]

MetricFn = Callable[[pl.DataFrame], float]
UnitCol = Union[str, Sequence[str]]


def _unit_cols(unit: UnitCol) -> list[str]:
    """Normalise the sampling-unit specification to a list of column names."""
    if isinstance(unit, str):
        return [unit]
    return list(unit)


def _unit_key(row: dict, cols: list[str]) -> str:
    """Build a single string key for a sampling unit from its column values."""
    return "|".join(str(row[c]) for c in cols)


def _seed_for(replicate: int, seed: int) -> int:
    """Derive a deterministic per-replicate RNG seed from the base seed."""
    return int(stable_hash(str(seed), str(replicate))[:16], 16) % (2**32)


def _resample_variants(
    df: pl.DataFrame, unit_cols: list[str], rng: np.random.Generator
) -> pl.DataFrame:
    """Return a bootstrap resample of ``df`` keeping sampling units intact."""
    # Unique sampling units.
    units = df.select(unit_cols).unique(maintain_order=True)
    unit_keys = np.array(
        [_unit_key(r, unit_cols) for r in units.to_dicts()]
    )
    n = unit_keys.size
    if n == 0:
        return df.clear()
    sampled_idx = rng.integers(0, n, size=n)
    sampled_keys = unit_keys[sampled_idx]

    # Build a lookup from unit key -> original rows, then concatenate.
    key_col = pl.Series("_unit_key", unit_keys)
    units = units.with_columns(key_col)
    df_keyed = df.with_columns(
        pl.Series(
            "_unit_key",
            [_unit_key(r, unit_cols) for r in df.to_dicts()],
        )
    )

    parts = []
    for key in sampled_keys:
        parts.append(df_keyed.filter(pl.col("_unit_key") == key))
    if not parts:
        return df.clear()
    out = pl.concat(parts, how="vertical_relaxed").drop("_unit_key")
    return out


def bootstrap_metrics(
    df: pl.DataFrame,
    metric_fn: MetricFn,
    n_replicates: int = 2000,
    unit: UnitCol = "variant_id",
    seed: int = 20260904,
) -> np.ndarray:
    """Bootstrap a metric by resampling sampling units with replacement.

    Parameters
    ----------
    df:
        Long-format evaluation DataFrame.
    metric_fn:
        Callable mapping a DataFrame to a scalar metric value.
    n_replicates:
        Number of bootstrap replicates.
    unit:
        Sampling-unit column name(s).  All rows sharing the same unit value are
        resampled together.  Default ``"variant_id"``.
    seed:
        Base RNG seed; per-replicate seeds are derived deterministically.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_replicates,)`` with the metric value for each
        replicate.  NaN replicates (e.g. metric undefined on a degenerate
        resample) are preserved.
    """
    cols = _unit_cols(unit)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"bootstrap_metrics: missing unit columns: {missing}")

    values = np.empty(n_replicates, dtype=float)
    values.fill(np.nan)
    for r in range(n_replicates):
        rng = np.random.default_rng(_seed_for(r, seed))
        resampled = _resample_variants(df, cols, rng)
        if resampled.is_empty():
            continue
        try:
            val = metric_fn(resampled)
        except Exception:
            val = float("nan")
        values[r] = val
    return values


def bootstrap_paired(
    df_a: pl.DataFrame,
    df_b: pl.DataFrame,
    metric_fn: MetricFn,
    n_replicates: int = 2000,
    unit: UnitCol = "variant_id",
    seed: int = 20260904,
) -> np.ndarray:
    """Paired bootstrap comparison of two models.

    The **same** resampled set of sampling units is used for both models on
    every replicate, so the per-replicate metric difference
    ``delta = metric(df_a) - metric(df_b)`` is a paired observation.

    Parameters
    ----------
    df_a, df_b:
        Evaluation DataFrames for model A and model B.  They must share the
        same sampling-unit column(s).
    metric_fn:
        Callable mapping a DataFrame to a scalar metric value.
    n_replicates, unit, seed:
        See :func:`bootstrap_metrics`.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_replicates,)`` with ``metric_a - metric_b`` per
        replicate.
    """
    cols = _unit_cols(unit)
    for name, frame in (("df_a", df_a), ("df_b", df_b)):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"bootstrap_paired: {name} missing unit columns: {missing}")

    # Use the union of sampling units present in either frame as the
    # resampling universe, so both models are evaluated on the same units.
    units_a = df_a.select(cols).unique()
    units_b = df_b.select(cols).unique()
    universe = pl.concat([units_a, units_b], how="vertical_relaxed").unique(
        maintain_order=True
    )
    unit_keys = np.array([_unit_key(r, cols) for r in universe.to_dicts()])
    n = unit_keys.size
    if n == 0:
        return np.full(n_replicates, np.nan)

    def _keyed(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            pl.Series("_unit_key", [_unit_key(r, cols) for r in frame.to_dicts()])
        )

    a_keyed = _keyed(df_a)
    b_keyed = _keyed(df_b)

    deltas = np.empty(n_replicates, dtype=float)
    deltas.fill(np.nan)
    for r in range(n_replicates):
        rng = np.random.default_rng(_seed_for(r, seed))
        idx = rng.integers(0, n, size=n)
        sampled_keys = unit_keys[idx]
        a_parts = [a_keyed.filter(pl.col("_unit_key") == k) for k in sampled_keys]
        b_parts = [b_keyed.filter(pl.col("_unit_key") == k) for k in sampled_keys]
        a_sub = (
            pl.concat(a_parts, how="vertical_relaxed").drop("_unit_key")
            if a_parts
            else df_a.clear()
        )
        b_sub = (
            pl.concat(b_parts, how="vertical_relaxed").drop("_unit_key")
            if b_parts
            else df_b.clear()
        )
        try:
            va = metric_fn(a_sub)
            vb = metric_fn(b_sub)
            deltas[r] = float(va) - float(vb)
        except Exception:
            deltas[r] = float("nan")
    return deltas


def compute_ci(
    bootstrap_values: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute a percentile confidence interval from bootstrap replicates.

    Parameters
    ----------
    bootstrap_values:
        1-D array of bootstrap replicate values (NaNs are ignored).
    confidence:
        Coverage probability in (0, 1).  Default 0.95 → 2.5th/97.5th percentiles.

    Returns
    -------
    (low, high) : tuple of floats
    """
    if confidence <= 0 or confidence >= 1:
        raise ValueError("confidence must be in (0, 1)")
    arr = np.asarray(bootstrap_values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    low = float(np.percentile(arr, 100 * alpha))
    high = float(np.percentile(arr, 100 * (1 - alpha)))
    return low, high


def compute_bootstrap_p_value(
    bootstrap_deltas: np.ndarray, null: float = 0.0
) -> float:
    """Two-sided bootstrap p-value for a paired difference.

    Computes the fraction of bootstrap deltas whose absolute value is at least
    as extreme as ``|observed|`` relative to ``null``, using the bootstrap
    distribution itself as the null.  A small continuity correction is applied
    to avoid p == 0.

    Parameters
    ----------
    bootstrap_deltas:
        1-D array of bootstrap replicate deltas (metric_a - metric_b).
    null:
        Null-hypothesis value of the difference (default 0).

    Returns
    -------
    float
        Two-sided p-value in [1/(2n), 1].
    """
    arr = np.asarray(bootstrap_deltas, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n == 0:
        return float("nan")
    # Centre the bootstrap distribution on the null and count extremes.
    centred = arr - np.mean(arr) + null
    extreme = np.mean(np.abs(centred) >= np.abs(null))
    # Continuity correction: never return exactly 0.
    p = float((extreme * n + 1) / (n + 1))
    return min(max(p, 1.0 / (2 * n)), 1.0)
