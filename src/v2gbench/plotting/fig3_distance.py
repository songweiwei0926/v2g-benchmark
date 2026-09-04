"""Figure 3 — Distance stratification.

Four panels:

A. MRR vs distance bin (line plot, one line per model family).
B. Nearest vs non-nearest MRR comparison (bar plot).
C. Gold distance rank (1, 2, 3, 4+) MRR (bar plot).
D. Candidate gene number distribution (histogram).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ._style import COLORBLIND_PALETTE, apply_style, save_svg, _to_pandas

__all__ = ["make_fig3"]


def _col(df, *candidates):
    pdf = _to_pandas(df)
    for c in candidates:
        if c in pdf.columns:
            return c
    return None


def make_fig3(stratified_metrics_df, output_path: str) -> str:
    """Generate Figure 3 (distance stratification) and save as SVG.

    Parameters
    ----------
    stratified_metrics_df : polars.DataFrame | pandas.DataFrame
        Long frame with per-(model, stratum) metrics. Expected columns
        include a model identifier, a distance bin / nearest flag /
        gold-distance-rank column, a candidate-gene-count column, and an
        ``MRR`` column. Missing optional columns render the corresponding
        panel as a placeholder.
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
    dist_col = _col(df, "distance_bin", "distance_bin_kb", "bin")
    nearest_col = _col(df, "is_nearest", "nearest")
    rank_col = _col(df, "gold_distance_rank", "distance_rank", "rank")
    n_cand_col = _col(df, "n_candidates", "candidate_gene_number", "n_genes")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # ---- Panel A: MRR vs distance bin ----
    ax = axes[0, 0]
    if dist_col and mrr_col in df.columns:
        families = sorted(df[model_col].dropna().unique())
        palette = sns.color_palette(COLORBLIND_PALETTE, n_colors=len(families))
        for fam, color in zip(families, palette):
            sub = df[df[model_col] == fam].sort_values(dist_col)
            ax.plot(sub[dist_col], sub[mrr_col], marker="o", ms=4, lw=1.5, label=fam, color=color)
        ax.set_xlabel("Distance bin (kb)")
        ax.set_ylabel("MRR")
        ax.legend(frameon=False, fontsize=7, title="Model")
    else:
        ax.text(0.5, 0.5, "No distance-bin data", ha="center", va="center")
        ax.axis("off")
    ax.set_title("A. MRR vs distance bin", loc="left", fontweight="bold")

    # ---- Panel B: nearest vs non-nearest MRR ----
    ax = axes[0, 1]
    if nearest_col and mrr_col in df.columns:
        agg = df.groupby([model_col, nearest_col])[mrr_col].mean().reset_index()
        pivot = agg.pivot(index=model_col, columns=nearest_col, values=mrr_col)
        pivot.columns = [str(c) for c in pivot.columns]
        # ensure two columns named like nearest/non-nearest
        if pivot.shape[1] == 2:
            pivot = pivot.rename(columns={pivot.columns[0]: "Non-nearest", pivot.columns[1]: "Nearest"})
        pivot = pivot.sort_values("Nearest", ascending=True) if "Nearest" in pivot.columns else pivot
        pivot.plot(kind="barh", ax=ax, color=[COLORBLIND_PALETTE[1], COLORBLIND_PALETTE[2]], edgecolor="black", linewidth=0.4)
        ax.set_xlabel("MRR")
        ax.set_ylabel("Model")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No nearest/non-nearest data", ha="center", va="center")
        ax.axis("off")
    ax.set_title("B. Nearest vs non-nearest MRR", loc="left", fontweight="bold")

    # ---- Panel C: gold distance rank MRR ----
    ax = axes[1, 0]
    if rank_col and mrr_col in df.columns:
        df = df.copy()
        df["_rank_grp"] = df[rank_col].astype("Int64").fillna(0)
        df.loc[df["_rank_grp"] >= 4, "_rank_grp"] = 4
        df["_rank_grp"] = df["_rank_grp"].map({1: "1", 2: "2", 3: "3", 4: "4+"})
        order = ["1", "2", "3", "4+"]
        agg = df.groupby(["_rank_grp", model_col])[mrr_col].mean().reset_index()
        sns.barplot(
            data=agg, x="_rank_grp", y=mrr_col, hue=model_col,
            order=order, ax=ax, palette=COLORBLIND_PALETTE, errorbar=None,
        )
        ax.set_xlabel("Gold distance rank")
        ax.set_ylabel("MRR")
        ax.legend(frameon=False, fontsize=7, title="Model")
    else:
        ax.text(0.5, 0.5, "No gold-distance-rank data", ha="center", va="center")
        ax.axis("off")
    ax.set_title("C. MRR by gold distance rank", loc="left", fontweight="bold")

    # ---- Panel D: candidate gene number distribution ----
    ax = axes[1, 1]
    if n_cand_col and n_cand_col in df.columns:
        vals = df[n_cand_col].dropna().to_numpy().astype(float)
        if vals.size:
            bins = np.linspace(0, np.percentile(vals, 99), 30)
            ax.hist(vals, bins=bins, color=COLORBLIND_PALETTE[0], edgecolor="white")
            ax.axvline(np.median(vals), color=COLORBLIND_PALETTE[5], ls="--", lw=1.2, label=f"median={np.median(vals):.0f}")
            ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel("Number of candidate genes per variant")
        ax.set_ylabel("Variants")
    else:
        ax.text(0.5, 0.5, "No candidate-count data", ha="center", va="center")
        ax.axis("off")
    ax.set_title("D. Candidate gene number", loc="left", fontweight="bold")

    fig.suptitle("Distance stratification of V2G performance", y=1.01, fontweight="bold")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_svg(fig, str(out))
    return str(out.resolve())
