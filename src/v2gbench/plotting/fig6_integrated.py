"""Figure 6 — Integrated model comparison.

Compares four integration strategies on chromosome-held-out results:

* best_standalone — the best single model per benchmark
* rank_average    — average rank across component models
* logistic        — logistic-regression meta-model
* xgboost         — gradient-boosted meta-model

Bar plot of MRR with bootstrap 95% confidence intervals.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ._style import COLORBLIND_PALETTE, apply_style, bootstrap_ci, save_svg, _to_pandas

__all__ = ["make_fig6"]

# Integration strategies in display order.
_STRATEGIES = ["best_standalone", "rank_average", "logistic", "xgboost"]


def _col(df, *candidates):
    pdf = _to_pandas(df)
    for c in candidates:
        if c in pdf.columns:
            return c
    return None


def make_fig6(metrics_df, output_path: str) -> str:
    """Generate Figure 6 (integrated model comparison) and save as SVG.

    Parameters
    ----------
    metrics_df : polars.DataFrame | pandas.DataFrame
        Long metrics frame for the chromosome-held-out evaluation. Expected
        columns: a strategy/model identifier (``strategy`` / ``model`` /
        ``model_id``), an ``MRR`` column, and optionally a ``benchmark``
        column for faceting.
    output_path : str
        Destination ``.svg`` path.

    Returns
    -------
    str
        Absolute path of the written SVG.
    """
    apply_style()
    df = _to_pandas(metrics_df).copy()
    strat_col = _col(df, "strategy", "model", "model_id", "Model") or "model"
    mrr_col = _col(df, "MRR", "mrr", "mean_reciprocal_rank") or "MRR"
    bench_col = _col(df, "benchmark", "benchmark_id", "dataset")

    # normalise strategy labels
    df[strat_col] = df[strat_col].astype(str).str.lower().str.replace("-", "_").str.replace(" ", "_")
    strategies = [s for s in _STRATEGIES if s in set(df[strat_col].unique())] or sorted(df[strat_col].unique())

    if bench_col:
        benchmarks = sorted(df[bench_col].dropna().unique())
        ncol = min(3, len(benchmarks))
        nrow = int(np.ceil(len(benchmarks) / ncol)) if benchmarks else 1
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow), squeeze=False)
        for ax, bench in zip(axes.ravel(), benchmarks):
            _plot(ax, df[df[bench_col] == bench], strat_col, mrr_col, strategies, title=bench)
        for ax in axes.ravel()[len(benchmarks):]:
            ax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _plot(ax, df, strat_col, mrr_col, strategies, title="All benchmarks (chr held-out)")

    fig.suptitle("Integrated model comparison (chromosome held-out)", y=1.02, fontweight="bold")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_svg(fig, str(out))
    return str(out.resolve())


def _plot(ax, sub, strat_col, mrr_col, strategies, title):
    means, los, his = [], [], []
    for s in strategies:
        vals = sub.loc[sub[strat_col] == s, mrr_col].dropna().to_numpy()
        m, lo, hi = bootstrap_ci(vals)
        means.append(m); los.append(lo); his.append(hi)
    x = np.arange(len(strategies))
    colors = COLORBLIND_PALETTE[: len(strategies)]
    ax.bar(
        x, means,
        yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
        color=colors, edgecolor="black", linewidth=0.5, capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in strategies], fontsize=8)
    ax.set_ylabel("MRR")
    ax.set_ylim(0, max(1.0, (max(his) if any(np.isfinite(his)) else 1.0)) * 1.1)
    ax.set_title(title, fontsize=10)
