"""Deterministic sampling and binning utilities for V2G-Benchmark.

All sampling is **SHA256-based and deterministic**: given the same input
DataFrame and seed, the exact same rows are selected on every run and every
machine.  This is critical for reproducible benchmark sub-setting (e.g.
stratified evaluation by distance or PIP bin).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import polars as pl

from ..utils.hashing import hash_to_float

__all__ = [
    "deterministic_stratified_sample",
    "assign_distance_bin",
    "assign_pip_bin",
    "assign_nearest_rank",
]


# ---------------------------------------------------------------------------
# Deterministic stratified sampling
# ---------------------------------------------------------------------------
def deterministic_stratified_sample(
    df: pl.DataFrame,
    strata_cols: Sequence[str],
    n_samples: int,
    seed: int = 20260904,
) -> pl.DataFrame:
    """Deterministic stratified sampling using SHA256 hashing.

    Within each stratum (unique combination of ``strata_cols``), rows are
    assigned a deterministic pseudo-random uniform key in [0, 1) derived from
    ``SHA256(seed | row_key)``.  The rows with the smallest keys are selected,
    so the sample is reproducible and independent of row order.

    Allocation is **proportional**: each stratum receives
    ``round(n_samples * stratum_weight)`` rows (at least 1 if the stratum is
    non-empty and the budget allows), with any remainder distributed to the
    largest strata.

    Parameters
    ----------
    df:
        Input DataFrame.
    strata_cols:
        Column(s) defining the strata.
    n_samples:
        Total number of rows to sample.
    seed:
        Base seed for the deterministic hash.

    Returns
    -------
    pl.DataFrame
        A reproducible stratified sample of ``n_samples`` rows.
    """
    if n_samples <= 0:
        return df.clear()
    cols = list(strata_cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"deterministic_stratified_sample: missing columns: {missing}")
    if df.is_empty():
        return df.clear()

    # Build a stable per-row key from the strata columns (plus row index for
    # uniqueness within a stratum).
    key_cols = [pl.col(c).cast(pl.Utf8) for c in cols]
    # Use the physical row index to disambiguate identical stratum members.
    df = df.with_row_index("_row_idx")
    df = df.with_columns(
        pl.concat_str(
            [pl.lit(str(seed)), *key_cols, pl.col("_row_idx").cast(pl.Utf8)],
            separator="|",
        ).alias("_row_key")
    )
    # Deterministic uniform key in [0, 1).
    df = df.with_columns(
        pl.col("_row_key").map_elements(
            lambda s: hash_to_float(s), return_dtype=pl.Float64
        ).alias("_u")
    )

    # Stratum sizes.
    strata = df.group_by(cols, maintain_order=True).len(name="_stratum_n")
    total = df.height

    # Proportional allocation with a minimum of 1 per non-empty stratum.
    weights = strata["_stratum_n"].to_numpy() / total
    alloc = np.maximum(1, np.round(weights * n_samples).astype(int))
    # Adjust to hit n_samples exactly.
    while alloc.sum() > n_samples:
        # Remove from the stratum with the largest allocation (and >1).
        order = np.argsort(-alloc)
        for i in order:
            if alloc[i] > 1:
                alloc[i] -= 1
                break
        else:
            break
    while alloc.sum() < n_samples:
        # Add to the largest stratum.
        i = int(np.argmax(alloc))
        alloc[i] += 1

    alloc_map = {
        _stratum_key(row, cols): int(a)
        for row, a in zip(strata.to_dicts(), alloc)
    }

    parts: List[pl.DataFrame] = []
    for stratum_key, sub in df.group_by(cols, maintain_order=True):
        k = alloc_map.get(_stratum_key_from_group(stratum_key, cols), 0)
        if k <= 0:
            continue
        picked = sub.sort("_u").head(k)
        parts.append(picked)

    if not parts:
        out = df.clear()
    else:
        out = pl.concat(parts, how="vertical_relaxed")
    return out.drop(["_row_idx", "_row_key", "_u"])


def _stratum_key(row: dict, cols: list[str]) -> str:
    return "|".join(str(row[c]) for c in cols)


def _stratum_key_from_group(group_key, cols: list[str]) -> str:
    """Normalise a polars group_by key (tuple or scalar) into a string key."""
    if isinstance(group_key, tuple):
        return "|".join(str(x) for x in group_key)
    return str(group_key)


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------
def assign_distance_bin(
    distance: Union[float, int, None], bins: Sequence[float]
) -> Optional[int]:
    """Assign a genomic distance to a 1-indexed bin.

    ``bins`` is a monotonically increasing sequence of upper bin edges
    (exclusive).  A value falls in bin ``i`` if ``bins[i-1] <= distance < bins[i]``
    (with bin 0 covering ``distance < bins[0]``).  Values at or above the last
    edge are placed in the final bin.  ``None`` / NaN distances return ``None``.

    Examples
    --------
    >>> bins = [1e4, 1e5, 1e6]
    >>> assign_distance_bin(500, bins)   # bin 1
    1
    >>> assign_distance_bin(5e5, bins)   # bin 3
    3
    """
    if distance is None:
        return None
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return None
    if np.isnan(d):
        return None
    edges = list(bins)
    if not edges:
        return None
    for i, edge in enumerate(edges):
        if d < edge:
            return i + 1
    return len(edges)


def assign_pip_bin(
    pip: Union[float, None], bins: Sequence[float]
) -> Optional[int]:
    """Assign a PIP (posterior inclusion probability) value to a 1-indexed bin.

    ``bins`` is a monotonically increasing sequence of upper bin edges in [0, 1].
    Semantics mirror :func:`assign_distance_bin`.  ``None`` / NaN returns ``None``.
    """
    if pip is None:
        return None
    try:
        p = float(pip)
    except (TypeError, ValueError):
        return None
    if np.isnan(p):
        return None
    edges = list(bins)
    if not edges:
        return None
    for i, edge in enumerate(edges):
        if p <= edge:
            return i + 1
    return len(edges)


def assign_nearest_rank(rank: Union[int, None]) -> Optional[str]:
    """Map a nearest-gene distance rank to a coarse category.

    Returns one of ``"1"``, ``"2"``, ``"3"``, ``"4+"`` for ranks 1, 2, 3 and
    4-or-greater respectively.  ``None`` / NaN / non-positive ranks return
    ``None``.
    """
    if rank is None:
        return None
    try:
        r = int(rank)
    except (TypeError, ValueError):
        return None
    if r <= 0:
        return None
    if r == 1:
        return "1"
    if r == 2:
        return "2"
    if r == 3:
        return "3"
    return "4+"
