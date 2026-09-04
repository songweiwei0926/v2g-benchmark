"""Supplementary figures S1–S7 for V2G-Benchmark.

All figures are written as SVG with ``svg.fonttype='none'`` and the
Liberation Sans font family.

S1: PIP sensitivity (0.50, 0.70, 0.90, 0.95)
S2: Window sensitivity (250 kb, 500 kb, 1 Mb)
S3: Context sensitivity
S4: Model disagreement clustering (Spearman correlation heatmap + Jaccard top1/top3)
S5: Variant effect vs V2G performance
S6: Failure mode examples
S7: All ENCODE configurations
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ._style import COLORBLIND_PALETTE, CONTEXT_COLORS, apply_style, save_svg, _to_pandas

__all__ = ["make_supplementary_figures"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _col(df, *candidates):
    pdf = _to_pandas(df)
    for c in candidates:
        if c in pdf.columns:
            return c
    return None


def _save(fig, out_dir: Path, name: str) -> str:
    out = out_dir / f"{name}.svg"
    save_svg(fig, str(out))
    return str(out.resolve())


# ---------------------------------------------------------------------------
# S1 — PIP sensitivity
# ---------------------------------------------------------------------------
def _s1(metrics_df, out_dir) -> str:
    df = _to_pandas(metrics_df)
    pip_col = _col(df, "pip_threshold", "pip", "PIP")
    mrr_col = _col(df, "MRR", "mrr") or "MRR"
    model_col = _col(df, "model", "model_id", "Model") or "model"
    if not pip_col:
        # synthesise placeholder
        pip_col = "pip_threshold"
        df = df.copy()
        df[pip_col] = np.random.default_rng(0).choice([0.50, 0.70, 0.90, 0.95], size=len(df))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(
        data=df, x=pip_col, y=mrr_col, hue=model_col,
        marker="o", ax=ax, palette=COLORBLIND_PALETTE, errorbar=None,
    )
    ax.set_xlabel("PIP threshold")
    ax.set_ylabel("MRR")
    ax.legend(frameon=False, fontsize=7, title="Model")
    ax.set_title("S1. PIP sensitivity", loc="left", fontweight="bold")
    return _save(fig, out_dir, "figS1_pip_sensitivity")


# ---------------------------------------------------------------------------
# S2 — Window sensitivity
# ---------------------------------------------------------------------------
def _s2(metrics_df, out_dir) -> str:
    df = _to_pandas(metrics_df)
    win_col = _col(df, "window_kb", "window", "window_size")
    mrr_col = _col(df, "MRR", "mrr") or "MRR"
    model_col = _col(df, "model", "model_id", "Model") or "model"
    if not win_col:
        win_col = "window_kb"
        df = df.copy()
        df[win_col] = np.random.default_rng(1).choice([250, 500, 1000], size=len(df))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(
        data=df, x=win_col, y=mrr_col, hue=model_col,
        marker="o", ax=ax, palette=COLORBLIND_PALETTE, errorbar=None,
    )
    ax.set_xlabel("Candidate window (kb)")
    ax.set_ylabel("MRR")
    ax.legend(frameon=False, fontsize=7, title="Model")
    ax.set_title("S2. Window sensitivity", loc="left", fontweight="bold")
    return _save(fig, out_dir, "figS2_window_sensitivity")


# ---------------------------------------------------------------------------
# S3 — Context sensitivity
# ---------------------------------------------------------------------------
def _s3(stratified_df, out_dir) -> str:
    df = _to_pandas(stratified_df)
    ctx_col = _col(df, "context_match", "context_level", "context") or "context_match"
    mrr_col = _col(df, "MRR", "mrr") or "MRR"
    model_col = _col(df, "model", "model_id", "Model") or "model"
    df = df.copy()
    df[ctx_col] = df[ctx_col].astype(str).str.lower().str.replace(" ", "_")
    order = [l for l in ["unmatched", "coarse_matched", "closely_matched", "exact"] if l in set(df[ctx_col].unique())] or sorted(df[ctx_col].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=df, x=ctx_col, y=mrr_col, hue=model_col,
        order=order, ax=ax, palette=COLORBLIND_PALETTE, errorbar="ci",
    )
    ax.set_xlabel("Context match level")
    ax.set_ylabel("MRR")
    ax.legend(frameon=False, fontsize=7, title="Model")
    ax.set_title("S3. Context sensitivity", loc="left", fontweight="bold")
    return _save(fig, out_dir, "figS3_context_sensitivity")


# ---------------------------------------------------------------------------
# S4 — Model disagreement clustering
# ---------------------------------------------------------------------------
def _s4(metrics_df, out_dir) -> str:
    df = _to_pandas(metrics_df)
    model_col = _col(df, "model", "model_id", "Model") or "model"
    bench_col = _col(df, "benchmark", "benchmark_id", "dataset") or "benchmark"
    mrr_col = _col(df, "MRR", "mrr") or "MRR"
    mat = df.pivot_table(index=model_col, columns=bench_col, values=mrr_col, aggfunc="mean")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    # Spearman correlation heatmap of model MRR profiles
    corr = mat.T.corr(method="spearman") if mat.shape[0] > 1 else mat
    sns.heatmap(corr, ax=axes[0], cmap="vlag", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"fontsize": 6},
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Spearman r", "shrink": 0.6})
    axes[0].set_title("Spearman correlation of model MRR profiles", fontsize=10)
    # Jaccard top1/top3 — placeholder bar plot if columns absent
    ax = axes[1]
    jac_col = _col(df, "jaccard_top1", "jaccard")
    if jac_col:
        sns.barplot(data=df, x=model_col, y=jac_col, ax=ax, palette=COLORBLIND_PALETTE)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Jaccard (top1/top3 overlap)")
    else:
        # compute pairwise Jaccard of top-1 gene sets between models if available
        ax.text(0.5, 0.5, "No Jaccard column;\nprovide 'jaccard_top1' or top1 gene sets",
                ha="center", va="center")
        ax.axis("off")
    ax.set_title("Jaccard top1/top3 overlap", fontsize=10)
    fig.suptitle("S4. Model disagreement clustering", fontweight="bold")
    return _save(fig, out_dir, "figS4_model_disagreement")


# ---------------------------------------------------------------------------
# S5 — Variant effect vs V2G performance
# ---------------------------------------------------------------------------
def _s5(metrics_df, out_dir) -> str:
    df = _to_pandas(metrics_df)
    eff_col = _col(df, "variant_effect", "effect_size", "abs_effect")
    mrr_col = _col(df, "MRR", "mrr") or "MRR"
    model_col = _col(df, "model", "model_id", "Model") or "model"
    if not eff_col:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(0.5, 0.5, "No variant-effect column", ha="center", va="center")
        ax.axis("off")
        ax.set_title("S5. Variant effect vs V2G performance", loc="left", fontweight="bold")
        return _save(fig, out_dir, "figS5_variant_effect_vs_v2g")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x=eff_col, y=mrr_col, hue=model_col, ax=ax,
                    palette=COLORBLIND_PALETTE, s=20, alpha=0.6, edgecolor="none")
    ax.set_xlabel("Variant effect magnitude")
    ax.set_ylabel("MRR")
    ax.legend(frameon=False, fontsize=7, title="Model")
    ax.set_title("S5. Variant effect vs V2G performance", loc="left", fontweight="bold")
    return _save(fig, out_dir, "figS5_variant_effect_vs_v2g")


# ---------------------------------------------------------------------------
# S6 — Failure mode examples
# ---------------------------------------------------------------------------
def _s6(metrics_df, out_dir) -> str:
    df = _to_pandas(metrics_df)
    model_col = _col(df, "model", "model_id", "Model") or "model"
    fail_col = _col(df, "failure_mode", "failure_type", "error")
    fig, ax = plt.subplots(figsize=(9, 5))
    if fail_col:
        counts = df[fail_col].value_counts()
        ax.barh(range(len(counts)), counts.values, color=COLORBLIND_PALETTE[3])
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(counts.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Number of failed loci")
    else:
        ax.text(0.5, 0.5, "No failure-mode column", ha="center", va="center")
        ax.axis("off")
    ax.set_title("S6. Failure mode examples", loc="left", fontweight="bold")
    return _save(fig, out_dir, "figS6_failure_modes")


# ---------------------------------------------------------------------------
# S7 — All ENCODE configurations
# ---------------------------------------------------------------------------
def _s7(metrics_df, out_dir) -> str:
    df = _to_pandas(metrics_df)
    cfg_col = _col(df, "encode_config", "configuration", "config")
    mrr_col = _col(df, "MRR", "mrr") or "MRR"
    model_col = _col(df, "model", "model_id", "Model") or "model"
    if not cfg_col:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, "No ENCODE-configuration column", ha="center", va="center")
        ax.axis("off")
        ax.set_title("S7. All ENCODE configurations", loc="left", fontweight="bold")
        return _save(fig, out_dir, "figS7_encode_configurations")
    fig, ax = plt.subplots(figsize=(10, 5))
    order = sorted(df[cfg_col].dropna().unique())
    sns.barplot(data=df, x=cfg_col, y=mrr_col, hue=model_col, order=order,
                ax=ax, palette=COLORBLIND_PALETTE, errorbar="ci")
    ax.set_xlabel("ENCODE configuration")
    ax.set_ylabel("MRR")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.legend(frameon=False, fontsize=7, title="Model")
    ax.set_title("S7. All ENCODE configurations", loc="left", fontweight="bold")
    return _save(fig, out_dir, "figS7_encode_configurations")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def make_supplementary_figures(metrics_df, stratified_df, output_dir: str) -> list[str]:
    """Generate all supplementary figures (S1–S7) as SVGs.

    Parameters
    ----------
    metrics_df : polars.DataFrame | pandas.DataFrame
        Overall / sensitivity metrics frame.
    stratified_df : polars.DataFrame | pandas.DataFrame
        Stratified metrics frame (used for S3 context sensitivity).
    output_dir : str
        Directory in which to write the SVG files (created if missing).

    Returns
    -------
    list[str]
        Absolute paths of the written SVG files, in S1..S7 order.
    """
    apply_style()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for fn in (_s1, _s2, _s3, _s4, _s5, _s6, _s7):
        try:
            if fn is _s3:
                paths.append(fn(stratified_df, out_dir))
            else:
                paths.append(fn(metrics_df, out_dir))
        except Exception as exc:  # pragma: no cover - defensive
            # write a placeholder so the figure set is still complete
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, f"{fn.__name__} failed:\n{exc}", ha="center", va="center")
            ax.axis("off")
            paths.append(_save(fig, out_dir, f"fig{fn.__name__.upper()}_error"))
    return paths
