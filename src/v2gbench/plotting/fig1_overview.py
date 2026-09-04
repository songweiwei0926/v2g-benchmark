"""Figure 1 — Benchmark overview.

Five panels:

A. Schematic of the benchmark pipeline (text-based annotation).
B. Bar chart of data-source counts (variants, elements, genes, contexts,
   positive pairs, tested negatives).
C. UpSet plot of gold-standard overlaps across evidence sources.
D. Distribution of gold target distances (variant -> gene TSS).
E. Proportion of nearest vs non-nearest gold targets.

Output is written as SVG with ``svg.fonttype='none'`` and the Liberation Sans
font family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

from ._style import (
    COLORBLIND_PALETTE,
    apply_style,
    save_svg,
    _to_pandas,
)

__all__ = ["make_fig1"]


# ---------------------------------------------------------------------------
# Panel A — schematic
# ---------------------------------------------------------------------------
def _panel_a(ax: plt.Axes) -> None:
    """Draw a text-based schematic of the benchmark pipeline."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.04, 0.70, "Datasets\n(CRISPR, eQTL, GWAS,\nL2G, common)"),
        (0.36, 0.70, "Harmonize\n(variant, gene, context,\nevidence)"),
        (0.68, 0.70, "Candidate set\n(distance window +\nexpression)"),
        (0.04, 0.30, "Score models\n(sequence, E2G,\nintegrated)"),
        (0.36, 0.30, "Ranking metrics\n(MRR, Top-k, AUPRC,\ncoverage)"),
        (0.68, 0.30, "Stratify & compare\n(distance, context,\ncomplementarity)"),
    ]
    for x, y, txt in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                0.28,
                0.22,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                linewidth=1.2,
                edgecolor=COLORBLIND_PALETTE[0],
                facecolor=COLORBLIND_PALETTE[4] + "33",
            )
        )
        ax.text(x + 0.14, y + 0.11, txt, ha="center", va="center", fontsize=8)

    # arrows top row -> bottom row
    for x in (0.18, 0.50, 0.82):
        ax.annotate(
            "",
            xy=(x, 0.52),
            xytext=(x, 0.70),
            arrowprops=dict(arrowstyle="->", color=COLORBLIND_PALETTE[1], lw=1.4),
        )
    # horizontal arrows
    for y in (0.81, 0.41):
        ax.annotate(
            "",
            xy=(0.36, y),
            xytext=(0.32, y),
            arrowprops=dict(arrowstyle="->", color=COLORBLIND_PALETTE[0], lw=1.2),
        )
        ax.annotate(
            "",
            xy=(0.68, y),
            xytext=(0.64, y),
            arrowprops=dict(arrowstyle="->", color=COLORBLIND_PALETTE[0], lw=1.2),
        )
    ax.set_title("A. Benchmark overview", loc="left", fontweight="bold")


# ---------------------------------------------------------------------------
# Panel B — data source counts
# ---------------------------------------------------------------------------
def _panel_b(ax: plt.Axes, metrics_df, evidence_df, candidate_df) -> None:
    """Bar chart of data-source counts."""
    pdf = _to_pandas(metrics_df)
    edf = _to_pandas(evidence_df)
    cdf = _to_pandas(candidate_df)

    def _nunique(df, col):
        return int(df[col].nunique()) if col in getattr(df, "columns", []) else 0

    counts = {
        "Variants": _nunique(pdf, "variant_id") or _nunique(edf, "variant_id"),
        "Elements": _nunique(edf, "element_id") if "element_id" in getattr(edf, "columns", []) else 0,
        "Genes": _nunique(edf, "gene_id") or _nunique(cdf, "gene_id"),
        "Contexts": _nunique(edf, "context_id") if "context_id" in getattr(edf, "columns", []) else 0,
        "Positive pairs": int((edf.get("is_gold", 0) == 1).sum()) if "is_gold" in getattr(edf, "columns", []) else 0,
        "Tested negatives": int((cdf.get("is_gold", 0) == 0).sum()) if "is_gold" in getattr(cdf, "columns", []) else 0,
    }
    labels = list(counts.keys())
    vals = list(counts.values())
    colors = COLORBLIND_PALETTE[: len(labels)]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Count")
    ax.set_title("B. Data sources", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


# ---------------------------------------------------------------------------
# Panel C — UpSet plot of gold-standard overlaps
# ---------------------------------------------------------------------------
def _panel_c(ax: plt.Axes, evidence_df) -> None:
    """UpSet plot of gold-standard overlaps across evidence sources.

    Uses the ``upsetplot`` library when available; otherwise falls back to a
    manual bar-chart implementation of the intersection sizes.
    """
    edf = _to_pandas(evidence_df)
    source_col = None
    for c in ("evidence_type", "source", "dataset", "benchmark"):
        if c in getattr(edf, "columns", []):
            source_col = c
            break
    if source_col is None or "variant_id" not in getattr(edf, "columns", []):
        ax.text(0.5, 0.5, "No evidence source data", ha="center", va="center")
        ax.set_title("C. Gold-standard overlaps", loc="left", fontweight="bold")
        ax.axis("off")
        return

    gold = edf
    if "is_gold" in getattr(edf, "columns", []):
        gold = edf[edf["is_gold"] == 1]
    # membership matrix: variant x source
    sources = sorted(gold[source_col].dropna().unique())
    membership = (
        gold.assign(_one=1)
        .pivot_table(index="variant_id", columns=source_col, values="_one", aggfunc="max")
        .fillna(0)
        .astype(int)
        .reindex(columns=sources, fill_value=0)
    )

    try:
        import upsetplot  # type: ignore

        counts = membership.groupby(sources).size().reset_index(name="size")
        # build the upsetplot Series
        idx = [tuple(row[s] for s in sources) for _, row in counts.iterrows()]
        ser = __import__("pandas").Series(counts["size"].values, index=idx)
        ser.index.names = sources
        upsetplot.plot(ser, fig=ax.figure, ax=ax, show_counts=True)
        ax.set_title("C. Gold-standard overlaps", loc="left", fontweight="bold")
        return
    except Exception:
        pass

    # ---- manual fallback ----
    # compute exclusive intersection sizes for the top combinations.
    from itertools import combinations

    combos = []
    for r in range(1, min(len(sources), 4) + 1):
        for combo in combinations(sources, r):
            mask = np.logical_and.reduce([membership[c].to_numpy() == 1 for c in combo])
            # exclusive: not present in any other source
            others = [s for s in sources if s not in combo]
            if others:
                mask = mask & ~np.logical_or.reduce(
                    [membership[s].to_numpy() == 1 for s in others]
                )
            combos.append(("+".join(combo), int(mask.sum())))
    combos = sorted(combos, key=lambda x: x[1], reverse=True)[:12]
    labels = [c[0] for c in combos]
    sizes = [c[1] for c in combos]
    ax.barh(range(len(labels)), sizes, color=COLORBLIND_PALETTE[2])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Intersection size")
    ax.set_title("C. Gold-standard overlaps", loc="left", fontweight="bold")


# ---------------------------------------------------------------------------
# Panel D — gold target distance distribution
# ---------------------------------------------------------------------------
def _panel_d(ax: plt.Axes, evidence_df) -> None:
    """Distribution of gold target distances (variant -> gene TSS)."""
    edf = _to_pandas(evidence_df)
    dist_col = None
    for c in ("distance_to_tss", "distance", "abs_distance"):
        if c in getattr(edf, "columns", []):
            dist_col = c
            break
    if dist_col is None:
        ax.text(0.5, 0.5, "No distance data", ha="center", va="center")
        ax.set_title("D. Gold target distance", loc="left", fontweight="bold")
        ax.axis("off")
        return
    gold = edf
    if "is_gold" in getattr(edf, "columns", []):
        gold = edf[edf["is_gold"] == 1]
    dist = np.abs(gold[dist_col].dropna().to_numpy()).astype(float)
    if dist.size == 0:
        ax.text(0.5, 0.5, "No gold distances", ha="center", va="center")
        ax.set_title("D. Gold target distance", loc="left", fontweight="bold")
        ax.axis("off")
        return
    # convert to kb
    dist_kb = dist / 1e3
    bins = np.linspace(0, min(2000, np.percentile(dist_kb, 99)), 40)
    ax.hist(dist_kb, bins=bins, color=COLORBLIND_PALETTE[0], edgecolor="white")
    ax.axvline(np.median(dist_kb), color=COLORBLIND_PALETTE[5], ls="--", lw=1.2, label=f"median={np.median(dist_kb):.0f} kb")
    ax.set_xlabel("Distance to TSS (kb)")
    ax.set_ylabel("Gold pairs")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("D. Gold target distance", loc="left", fontweight="bold")


# ---------------------------------------------------------------------------
# Panel E — nearest vs non-nearest gold targets
# ---------------------------------------------------------------------------
def _panel_e(ax: plt.Axes, evidence_df) -> None:
    """Proportion of nearest vs non-nearest gold targets."""
    edf = _to_pandas(evidence_df)
    gold = edf
    if "is_gold" in getattr(edf, "columns", []):
        gold = edf[edf["is_gold"] == 1]
    if "is_nearest" in getattr(gold, "columns", []):
        nearest = gold["is_nearest"].astype(int)
    elif "gold_distance_rank" in getattr(gold, "columns", []):
        nearest = (gold["gold_distance_rank"] == 1).astype(int)
    else:
        ax.text(0.5, 0.5, "No nearest/rank data", ha="center", va="center")
        ax.set_title("E. Nearest vs non-nearest", loc="left", fontweight="bold")
        ax.axis("off")
        return
    n_near = int(nearest.sum())
    n_far = int(len(nearest) - n_near)
    total = n_near + n_far
    if total == 0:
        ax.text(0.5, 0.5, "No gold pairs", ha="center", va="center")
        ax.set_title("E. Nearest vs non-nearest", loc="left", fontweight="bold")
        ax.axis("off")
        return
    sizes = [n_near, n_far]
    labels = [f"Nearest\n{n_near} ({100*n_near/total:.1f}%)", f"Non-nearest\n{n_far} ({100*n_far/total:.1f}%)"]
    colors = [COLORBLIND_PALETTE[2], COLORBLIND_PALETTE[1]]
    ax.pie(sizes, labels=labels, colors=colors, startangle=90, wedgeprops=dict(edgecolor="white"))
    ax.set_title("E. Nearest vs non-nearest gold", loc="left", fontweight="bold")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def make_fig1(metrics_df, evidence_df, candidate_df, output_path: str) -> str:
    """Generate Figure 1 (benchmark overview) and save as SVG.

    Parameters
    ----------
    metrics_df : polars.DataFrame | pandas.DataFrame
        Overall metrics frame (used for variant/gene counts).
    evidence_df : polars.DataFrame | pandas.DataFrame
        Long evidence / gold-standard frame with columns such as
        ``variant_id``, ``gene_id``, ``evidence_type``, ``is_gold``,
        ``distance_to_tss``.
    candidate_df : polars.DataFrame | pandas.DataFrame
        Candidate-gene frame with ``variant_id``, ``gene_id``, ``is_gold``.
    output_path : str
        Destination ``.svg`` path.

    Returns
    -------
    str
        The absolute path of the written SVG.
    """
    apply_style()
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[1.1, 1.0, 1.0],
        hspace=0.55, wspace=0.4,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1:])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])

    _panel_a(ax_a)
    _panel_b(ax_b, metrics_df, evidence_df, candidate_df)
    _panel_c(ax_c, evidence_df)
    _panel_d(ax_d, evidence_df)
    _panel_e(ax_e, evidence_df)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_svg(fig, str(out))
    return str(out.resolve())
