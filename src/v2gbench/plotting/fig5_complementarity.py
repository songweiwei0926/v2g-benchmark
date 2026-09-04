"""Figure 5 — Sequence vs E2G complementarity.

A scatter plot with the sequence-model score (AlphaGenome / Borzoi) on x and
the E2G-model score (rE2G / ABC) on y. Points are coloured by outcome:

* both_correct  — both models rank the gold gene first
* sequence_only — only the sequence model is correct
* e2g_only      — only the E2G model is correct
* both_fail     — neither model is correct

A quadrant analysis (counts per outcome) is annotated on the plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ._style import OUTCOME_COLORS, apply_style, save_svg, _to_pandas

__all__ = ["make_fig5"]


def _col(df, *candidates):
    pdf = _to_pandas(df)
    for c in candidates:
        if c in pdf.columns:
            return c
    return None


def make_fig5(predictions_df, output_path: str) -> str:
    """Generate Figure 5 (sequence vs E2G complementarity) and save as SVG.

    Parameters
    ----------
    predictions_df : polars.DataFrame | pandas.DataFrame
        Long predictions frame. Expected columns: a sequence-model score
        (``sequence_score`` / ``alphagenome_score`` / ``borzoi_score``), an
        E2G-model score (``e2g_score`` / ``abc_score`` / ``re2g_score``),
        an ``is_gold`` flag, and ideally per-model correctness flags
        (``sequence_correct``, ``e2g_correct``). When correctness flags are
        absent they are derived by ranking within each variant.
    output_path : str
        Destination ``.svg`` path.

    Returns
    -------
    str
        Absolute path of the written SVG.
    """
    apply_style()
    df = _to_pandas(predictions_df).copy()
    seq_col = _col(df, "sequence_score", "alphagenome_score", "borzoi_score", "seq_score")
    e2g_col = _col(df, "e2g_score", "abc_score", "re2g_score", "e2g_score")
    if not seq_col or not e2g_col:
        raise ValueError(
            "make_fig5: need a sequence-model score column and an E2G-model "
            f"score column; got {list(df.columns)}"
        )

    # restrict to gold pairs (the complementarity question is about causal targets)
    if "is_gold" in df.columns:
        gold = df[df["is_gold"] == 1].copy()
    else:
        gold = df.copy()

    # derive correctness if not provided
    if "sequence_correct" not in gold.columns and "variant_id" in gold.columns:
        gold["sequence_correct"] = _is_top1(gold, seq_col)
    if "e2g_correct" not in gold.columns and "variant_id" in gold.columns:
        gold["e2g_correct"] = _is_top1(gold, e2g_col)

    def _outcome(row):
        s = bool(row.get("sequence_correct", False))
        e = bool(row.get("e2g_correct", False))
        if s and e:
            return "both_correct"
        if s:
            return "sequence_only"
        if e:
            return "e2g_only"
        return "both_fail"

    gold["outcome"] = gold.apply(_outcome, axis=1)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    for outcome, color in OUTCOME_COLORS.items():
        sub = gold[gold["outcome"] == outcome]
        ax.scatter(
            sub[seq_col], sub[e2g_col],
            s=14, alpha=0.55, color=color, edgecolor="none", label=outcome,
        )

    # quadrant lines at median split
    xmed = np.nanmedian(gold[seq_col]) if len(gold) else 0.5
    ymed = np.nanmedian(gold[e2g_col]) if len(gold) else 0.5
    ax.axvline(xmed, color="grey", ls="--", lw=0.8)
    ax.axhline(ymed, color="grey", ls="--", lw=0.8)

    # quadrant counts annotation
    counts = gold["outcome"].value_counts()
    txt = "\n".join(f"{k}: {counts.get(k, 0)}" for k in OUTCOME_COLORS)
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    ax.set_xlabel(f"Sequence model score ({seq_col})")
    ax.set_ylabel(f"E2G model score ({e2g_col})")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Sequence vs E2G complementarity", loc="left", fontweight="bold")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_svg(fig, str(out))
    return str(out.resolve())


def _is_top1(df, score_col) -> "np.ndarray":
    """Return boolean array: is this row the top-ranked gene for its variant?"""
    if "variant_id" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    df = df.copy()
    df["_rank"] = df.groupby("variant_id")[score_col].rank(
        method="first", ascending=False
    )
    return (df["_rank"] == 1).to_numpy()
