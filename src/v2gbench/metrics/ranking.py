"""Ranking metrics for variant-to-gene prediction.

All functions operate on a long-format ``polars.DataFrame`` with (at minimum)
the columns:

    variant_id      : str   – locus identifier
    gene_id         : str   – candidate gene identifier
    ranking_score   : float – higher = more likely causal
    is_gold         : int   – 1 if the (variant, gene) pair is a gold/causal pair, else 0

Optional columns used for deterministic tie-breaking:

    distance_to_tss : int/float – genomic distance to the gene TSS
    gold_gene_set   : list[str] – full set of gold genes for a variant
                                  (used by NDCG for multi-target variants)

The evaluation code is **model-agnostic**: it never reads ``model_id`` or any
model name.  It only consumes ``ranking_score`` and ``is_gold`` (plus optional
tie-break columns).  This keeps the metric implementation decoupled from the
set of models being evaluated.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import polars as pl

__all__ = [
    "rank_candidates",
    "compute_mrr",
    "compute_top1",
    "compute_recall_at_k",
    "compute_ndcg",
    "compute_all_ranking_metrics",
]


# ---------------------------------------------------------------------------
# Ranking helper
# ---------------------------------------------------------------------------
def rank_candidates(df: pl.DataFrame) -> pl.DataFrame:
    """Rank candidate genes per variant by ``ranking_score`` (descending).

    Ties are broken deterministically using, in order:

    1. ``ranking_score``   – descending (primary sort key)
    2. ``distance_to_tss`` – ascending  (closer genes win; optional column,
       missing values are treated as +inf so they sort last)
    3. ``gene_id``         – ascending  (alphabetical, final deterministic tie-break)

    Parameters
    ----------
    df:
        Long-format DataFrame with at least ``variant_id``, ``gene_id`` and
        ``ranking_score``.

    Returns
    -------
    pl.DataFrame
        The input frame with an additional integer ``rank`` column
        (1-indexed, per variant).
    """
    required = {"variant_id", "gene_id", "ranking_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rank_candidates: missing required columns: {missing}")

    # Build sort expressions with deterministic tie-breaking.
    # distance_to_tss is optional; treat missing as +inf (sorts last).
    if "distance_to_tss" in df.columns:
        distance_expr = pl.col("distance_to_tss").fill_null(float("inf"))
    else:
        distance_expr = pl.lit(float("inf"))

    ranked = (
        df.with_columns(distance_expr.alias("_distance_tiebreak"))
        .sort(
            ["variant_id", "ranking_score", "_distance_tiebreak", "gene_id"],
            descending=[False, True, False, False],
        )
        .with_columns(
            pl.col("gene_id")
            .cum_count()
            .over("variant_id")
            .alias("rank")
        )
        .drop("_distance_tiebreak")
    )
    return ranked


# ---------------------------------------------------------------------------
# Core ranking metrics
# ---------------------------------------------------------------------------
def _gold_ranks(df: pl.DataFrame) -> pl.DataFrame:
    """Return one row per variant with the rank of its best (lowest-rank) gold gene."""
    ranked = rank_candidates(df)
    if "is_gold" not in ranked.columns:
        raise ValueError("compute_*: input must contain an 'is_gold' column")
    gold = ranked.filter(pl.col("is_gold") == 1)
    # Best gold gene = minimum rank per variant.
    best = gold.sort("rank").group_by("variant_id", maintain_order=True).first()
    return best


def compute_mrr(df: pl.DataFrame) -> float:
    """Mean Reciprocal Rank.

    For each variant, find the rank of the best gold gene, compute ``1 / rank``
    and average across all variants that have at least one gold gene.

    Variants with no gold gene are excluded from the denominator (they are
    uninformative for ranking quality).
    """
    best = _gold_ranks(df)
    if best.is_empty():
        return float("nan")
    rr = 1.0 / best["rank"].to_numpy()
    return float(np.mean(rr))


def compute_top1(df: pl.DataFrame) -> float:
    """Probability that a gold gene is ranked #1 for its variant."""
    best = _gold_ranks(df)
    if best.is_empty():
        return float("nan")
    return float(np.mean(best["rank"].to_numpy() == 1))


def compute_recall_at_k(df: pl.DataFrame, k: int) -> float:
    """Probability that a gold gene is ranked within the top ``k``.

    For single-gold-gene variants this is simply ``P(rank <= k)``.  For
    multi-gold-gene variants we use the *any-gold-in-top-k* definition
    (recall@k = 1 if at least one gold gene has rank <= k), which matches the
    common variant-to-gene benchmark convention.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    ranked = rank_candidates(df)
    if "is_gold" not in ranked.columns:
        raise ValueError("compute_recall_at_k: input must contain 'is_gold'")
    gold = ranked.filter(pl.col("is_gold") == 1)
    if gold.is_empty():
        return float("nan")
    hit = (
        gold.group_by("variant_id")
        .agg((pl.col("rank") <= k).any().alias("hit"))
    )
    return float(hit["hit"].to_numpy().mean())


def compute_ndcg(df: pl.DataFrame) -> float:
    """Normalised Discounted Cumulative Gain for multi-target variants.

    Relevance per candidate is taken from ``is_gold`` (1 for gold, 0 otherwise).
    If a ``gold_gene_set`` column is present it is used to define the ideal
    ordering (DCG / IDCG); otherwise ``is_gold`` is used directly.

    NDCG = DCG / IDCG, averaged across variants.  Variants with no gold gene
    (IDCG == 0) are skipped.
    """
    ranked = rank_candidates(df)
    if "is_gold" not in ranked.columns:
        raise ValueError("compute_ndcg: input must contain 'is_gold'")

    # Relevance gain: 2^rel - 1 with rel in {0, 1} → {0, 1}.
    ranked = ranked.with_columns(
        pl.when(pl.col("is_gold") == 1)
        .then(1.0)
        .otherwise(0.0)
        .alias("rel")
    )

    def _dcg(rel_sorted: np.ndarray) -> float:
        if rel_sorted.size == 0:
            return 0.0
        discounts = 1.0 / np.log2(np.arange(2, rel_sorted.size + 2))
        return float(np.sum(rel_sorted * discounts))

    ndcgs = []
    for (_vid, sub) in ranked.group_by("variant_id", maintain_order=True):
        rel_sorted = sub.sort("rank")["rel"].to_numpy()
        dcg = _dcg(rel_sorted)
        # Ideal: all gold genes first.
        ideal = np.sort(rel_sorted)[::-1]
        idcg = _dcg(ideal)
        if idcg == 0.0:
            continue
        ndcgs.append(dcg / idcg)
    if not ndcgs:
        return float("nan")
    return float(np.mean(ndcgs))


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def compute_all_ranking_metrics(df: pl.DataFrame) -> Dict[str, float]:
    """Compute the full ranking-metric suite.

    Returns a dict with keys: ``MRR``, ``Top1``, ``Recall@3``, ``Recall@5``,
    ``NDCG``.
    """
    return {
        "MRR": compute_mrr(df),
        "Top1": compute_top1(df),
        "Recall@3": compute_recall_at_k(df, 3),
        "Recall@5": compute_recall_at_k(df, 5),
        "NDCG": compute_ndcg(df),
    }
