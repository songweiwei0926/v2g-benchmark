"""Figure 4 — Context matching.

Compares MRR across context-match levels: exact, closely_matched,
coarse_matched, unmatched. Bar plot with bootstrap error bars, one panel per
model (or a single grouped bar plot when many models are present).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ._style import CONTEXT_COLORS, apply_style, bootstrap_ci, save_svg, _to_pandas

__all__ = ["make_fig4"]

# Canonical ordering of context-match levels (worst -> best).
_LEVEL_ORDER = ["unmatched", "coarse_matched", "closely_matched", "exact"]


def _col(df, *candidates):
    pdf = _to_pandas(df)
    for c in candidates:
        if c in pdf.columns:
            return c
    return None


def make_fig4(stratified_metrics_df, output_path: str) -> str:
    """Generate Figure 4 (context matching) and save as SVG.

    Parameters
    ----------
    stratified_metrics_df : polars.DataFrame | pandas.DataFrame
        Long frame with per-(model, context-match-level) metrics. Expected
        columns: a model identifier, a context-match level column
        (``context_match`` / ``context_level``), and an ``MRR`` column.
    output_path : str
        Destination ``.svg`` path.

    Returns
    -------
    str
        Absolute path of the written SVG.
    """
    apply_style()
    df = _to_pandas(stratified_metrics_df)
    model_col = _col(df, "model", "model_id", "Model") or "model"
    mrr_col = _col(df, "MRR", "mrr", "mean_reciprocal_rank") or "MRR"
    ctx_col = _col(df, "context_match", "context_level", "context")

    if not ctx_col or mrr_col not in df.columns:
        raise ValueError(
            "make_fig4: need a context-match column and an MRR column; "
            f"got columns {list(df.columns)}"
        )

    # normalise level labels
    df = df.copy()
    df[ctx_col] = df[ctx_col].astype(str).str.lower().str.replace(" ", "_")
    levels = [lvl for lvl in _LEVEL_ORDER if lvl in set(df[ctx_col].unique())]
    if not levels:
        levels = sorted(df[ctx_col].unique())

    models = sorted(df[model_col].dropna().unique())
    n_models = len(models)
    # one subplot per model if few, else a single grouped bar plot
    if n_models <= 6:
        ncol = min(3, n_models)
        nrow = int(np.ceil(n_models / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
        for i, (ax, model) in enumerate(zip(axes.ravel(), models)):
            _plot_model(ax, df[df[model_col] == model], model, ctx_col, mrr_col, levels)
        for ax in axes.ravel()[n_models:]:
            ax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_grouped(ax, df, models, ctx_col, mrr_col, levels, model_col)

    fig.suptitle("Context matching improves V2G prediction", y=1.01, fontweight="bold")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_svg(fig, str(out))
    return str(out.resolve())


def _plot_model(ax, sub, model, ctx_col, mrr_col, levels):
    means, los, his = [], [], []
    for lvl in levels:
        vals = sub.loc[sub[ctx_col] == lvl, mrr_col].dropna().to_numpy()
        m, lo, hi = bootstrap_ci(vals)
        means.append(m); los.append(lo); his.append(hi)
    colors = [CONTEXT_COLORS.get(lvl, "#999999") for lvl in levels]
    x = np.arange(len(levels))
    ax.bar(x, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
           color=colors, edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", "\n") for l in levels], fontsize=8)
    ax.set_ylim(0, max(1.0, max(his) * 1.1) if any(np.isfinite(his)) else 1.0)
    ax.set_ylabel("MRR")
    ax.set_title(model, fontsize=9)


def _plot_grouped(ax, df, models, ctx_col, mrr_col, levels, model_col):
    x = np.arange(len(models))
    width = 0.8 / len(levels)
    for i, lvl in enumerate(levels):
        means, errs = [], []
        for model in models:
            vals = df.loc[(df[model_col] == model) & (df[ctx_col] == lvl), mrr_col].dropna().to_numpy()
            m, lo, hi = bootstrap_ci(vals)
            means.append(m)
            errs.append([m - lo, hi - m])
        errs = np.array(errs).T
        ax.bar(x + i * width, means, width, yerr=errs, label=lvl,
               color=CONTEXT_COLORS.get(lvl, "#999999"), edgecolor="black", linewidth=0.4, capsize=2)
    ax.set_xticks(x + width * (len(levels) - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MRR")
    ax.legend(frameon=False, fontsize=8, title="Context match")
