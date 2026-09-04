"""Shared matplotlib/seaborn style settings for V2G-Benchmark figures.

This module centralises the rcParams and colorblind-friendly palettes used by
every figure module so that the styling is consistent and DRY.  Importing
``apply_style()`` at the top of each figure function guarantees the SVG output
uses ``svg.fonttype='none'`` (editable text) and the Liberation Sans font
family (with Arimo and DejaVu Sans as fallbacks).
"""

from __future__ import annotations

from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

__all__ = [
    "apply_style",
    "COLORBLIND_PALETTE",
    "CONTEXT_COLORS",
    "OUTCOME_COLORS",
    "METRIC_CMAPS",
    "save_svg",
]

# Colorblind-friendly Okabe-Ito palette (8 colours, perceptually distinct).
COLORBLIND_PALETTE: list[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]

# Context-match level colours (ordered worst -> best).
CONTEXT_COLORS: dict[str, str] = {
    "unmatched": "#CC79A7",
    "coarse_matched": "#E69F00",
    "closely_matched": "#56B4E9",
    "exact": "#009E73",
}

# Quadrant outcome colours for the complementarity scatter.
OUTCOME_COLORS: dict[str, str] = {
    "both_correct": "#009E73",
    "sequence_only": "#0072B2",
    "e2g_only": "#E69F00",
    "both_fail": "#999999",
}

# Diverging / sequential cmaps per metric for heatmaps.
METRIC_CMAPS: dict[str, str] = {
    "MRR": "viridis",
    "Top1": "magma",
    "AUPRC": "cividis",
    "coverage": "rocket_r",
}


def apply_style() -> None:
    """Apply the canonical V2G-Benchmark matplotlib/seaborn style.

    Sets editable SVG fonts (``svg.fonttype='none'``), the Liberation Sans
    font stack, a clean seaborn context, and sensible figure defaults.
    """
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
    mpl.rcParams["pdf.fonttype"] = 42  # editable text in PDFs too
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.dpi"] = 100
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    sns.set_theme(
        context="paper",
        style="ticks",
        palette=COLORBLIND_PALETTE,
        rc={
            "svg.fonttype": "none",
            "font.family": ["Liberation Sans", "Arimo", "DejaVu Sans"],
        },
    )


def save_svg(fig: plt.Figure, output_path: str) -> None:
    """Save *fig* to *output_path* as SVG with editable fonts."""
    apply_style()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _to_pandas(df) -> "object":  # noqa: D401
    """Best-effort conversion of polars/duckdb/pandas frames to pandas."""
    try:
        import polars as pl  # noqa: F401

        if isinstance(df, pl.DataFrame):
            return df.to_pandas()
    except Exception:
        pass
    return df


def safe_get(df, col: str, default=None):
    """Return *col* from a pandas/polars frame or *default* if absent."""
    pdf = _to_pandas(df)
    if col in getattr(pdf, "columns", []):
        return pdf[col]
    return default


def bootstrap_ci(
    values: Iterable[float], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) bootstrap CI for *values*."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(arr.mean()), float(lo), float(hi)
