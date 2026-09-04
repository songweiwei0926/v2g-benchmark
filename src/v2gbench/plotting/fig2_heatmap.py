"""Figure 2 — Model x benchmark heatmap.

A heatmap with models on the y-axis, benchmarks on the x-axis, cells coloured
by metric value. Four metrics are shown (MRR, Top1, AUPRC, coverage) as a
grid of sub-heatmaps. Uses a colorblind-friendly sequential palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ._style import METRIC_CMAPS, apply_style, save_svg, _to_pandas

__all__ = ["make_fig2"]

# Metrics displayed, in panel order.
_METRICS = ["MRR", "Top1", "AUPRC", "coverage"]


def _resolve_columns(df):
    """Map canonical metric names to whatever columns are present."""
    pdf = _to_pandas(df)
    aliases = {
        "MRR": ["MRR", "mrr", "mean_reciprocal_rank"],
        "Top1": ["Top1", "top1", "recall_at_1", "Top1_accuracy"],
        "AUPRC": ["AUPRC", "auprc", "avg_precision"],
        "coverage": ["coverage", "Coverage", "frac_covered"],
    }
    model_col = next((c for c in ("model", "model_id", "Model") if c in pdf.columns), "model")
    bench_col = next((c for c in ("benchmark", "benchmark_id", "Benchmark", "dataset") if c in pdf.columns), "benchmark")
    metric_cols = {}
    for m in _METRICS:
        for a in aliases[m]:
            if a in pdf.columns:
                metric_cols[m] = a
                break
    return pdf, model_col, bench_col, metric_cols


def make_fig2(metrics_df, output_path: str) -> str:
    """Generate Figure 2 (model x benchmark heatmap) and save as SVG.

    Parameters
    ----------
    metrics_df : polars.DataFrame | pandas.DataFrame
        Long metrics frame with one row per (model, benchmark) containing
        columns for MRR, Top1, AUPRC and coverage (case-insensitive aliases
        accepted).
    output_path : str
        Destination ``.svg`` path.

    Returns
    -------
    str
        Absolute path of the written SVG.
    """
    apply_style()
    pdf, model_col, bench_col, metric_cols = _resolve_columns(metrics_df)
    available = [m for m in _METRICS if m in metric_cols]
    if not available:
        raise ValueError(
            f"make_fig2: none of {_METRICS} found in columns {list(pdf.columns)}"
        )

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n + 1.5, 0.32 * pdf[model_col].nunique() + 2.5))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        col = metric_cols[metric]
        sub = pdf[[model_col, bench_col, col]].copy()
        mat = sub.pivot_table(index=model_col, columns=bench_col, values=col, aggfunc="mean")
        # sort models by mean metric value (descending) for readability
        order = mat.mean(axis=1).sort_values(ascending=False).index
        mat = mat.reindex(order)
        sns.heatmap(
            mat,
            ax=ax,
            cmap=METRIC_CMAPS.get(metric, "viridis"),
            vmin=0.0,
            vmax=max(1.0, float(np.nanmax(mat.to_numpy()))),
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 6},
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": metric, "shrink": 0.6},
            square=False,
        )
        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("Benchmark")
        ax.set_ylabel("Model" if metric == available[0] else "")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Model x benchmark performance", y=1.02, fontweight="bold")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_svg(fig, str(out))
    return str(out.resolve())
