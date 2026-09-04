"""Tests for ranking metrics: MRR, Top-1, Recall@K on known input."""

import numpy as np
import polars as pl
import pytest

from v2gbench.metrics.ranking import (
    rank_candidates,
    compute_mrr,
    compute_top1,
    compute_recall_at_k,
    compute_ndcg,
    compute_all_ranking_metrics,
)


class TestRankCandidates:
    def test_rank_descending(self, small_ranking_df):
        ranked = rank_candidates(small_ranking_df)
        # v1: scores 0.9, 0.5, 0.1 → ranks 1, 2, 3
        v1 = ranked.filter(pl.col("variant_id") == "v1").sort("rank")
        assert v1["gene_id"].to_list() == ["g_a", "g_b", "g_c"]
        assert v1["rank"].to_list() == [1, 2, 3]

    def test_tie_break_by_distance(self):
        """Ties in score broken by distance (ascending)."""
        df = pl.DataFrame({
            "variant_id": ["v1", "v1"],
            "gene_id": ["g_far", "g_near"],
            "ranking_score": [0.5, 0.5],
            "distance_to_tss": [10000, 1000],
        })
        ranked = rank_candidates(df)
        near = ranked.filter(pl.col("gene_id") == "g_near")
        assert near["rank"][0] == 1

    def test_tie_break_by_gene_id(self):
        """Final tie-break is gene_id alphabetical."""
        df = pl.DataFrame({
            "variant_id": ["v1", "v1"],
            "gene_id": ["g_z", "g_a"],
            "ranking_score": [0.5, 0.5],
            "distance_to_tss": [1000, 1000],
        })
        ranked = rank_candidates(df)
        a = ranked.filter(pl.col("gene_id") == "g_a")
        assert a["rank"][0] == 1

    def test_missing_columns_raises(self):
        df = pl.DataFrame({"variant_id": ["v1"], "gene_id": ["g1"]})
        with pytest.raises(ValueError):
            rank_candidates(df)


class TestMRR:
    def test_known_mrr(self, small_ranking_df):
        """v1: gold at rank 2 → RR=0.5; v2: gold at rank 1 → RR=1.0; MRR=0.75."""
        mrr = compute_mrr(small_ranking_df)
        assert mrr == pytest.approx(0.75)

    def test_all_top1(self):
        df = pl.DataFrame({
            "variant_id": ["v1", "v2"],
            "gene_id": ["g1", "g2"],
            "ranking_score": [0.9, 0.8],
            "is_gold": [1, 1],
        })
        assert compute_mrr(df) == 1.0

    def test_no_gold_returns_nan(self):
        df = pl.DataFrame({
            "variant_id": ["v1"],
            "gene_id": ["g1"],
            "ranking_score": [0.9],
            "is_gold": [0],
        })
        assert np.isnan(compute_mrr(df))


class TestTop1:
    def test_known_top1(self, small_ranking_df):
        """1 of 2 variants has gold at rank 1 → Top1=0.5."""
        assert compute_top1(small_ranking_df) == pytest.approx(0.5)

    def test_all_top1(self):
        df = pl.DataFrame({
            "variant_id": ["v1", "v2"],
            "gene_id": ["g1", "g2"],
            "ranking_score": [0.9, 0.8],
            "is_gold": [1, 1],
        })
        assert compute_top1(df) == 1.0


class TestRecallAtK:
    def test_recall_at_3(self, small_ranking_df):
        """Both gold genes are within top 3 → Recall@3=1.0."""
        assert compute_recall_at_k(small_ranking_df, 3) == pytest.approx(1.0)

    def test_recall_at_1(self, small_ranking_df):
        """Only v2's gold is at rank 1 → Recall@1=0.5."""
        assert compute_recall_at_k(small_ranking_df, 1) == pytest.approx(0.5)

    def test_k_too_small_raises(self):
        df = pl.DataFrame({"variant_id": ["v1"], "gene_id": ["g1"],
                           "ranking_score": [1.0], "is_gold": [1]})
        with pytest.raises(ValueError):
            compute_recall_at_k(df, 0)


class TestNDCG:
    def test_perfect_ndcg(self):
        df = pl.DataFrame({
            "variant_id": ["v1", "v1"],
            "gene_id": ["g1", "g2"],
            "ranking_score": [0.9, 0.1],
            "is_gold": [1, 0],
        })
        assert compute_ndcg(df) == 1.0

    def test_ndcg_between_zero_and_one(self, small_ranking_df):
        ndcg = compute_ndcg(small_ranking_df)
        assert 0.0 <= ndcg <= 1.0


class TestComputeAllRankingMetrics:
    def test_returns_all_keys(self, small_ranking_df):
        metrics = compute_all_ranking_metrics(small_ranking_df)
        assert set(metrics.keys()) == {"MRR", "Top1", "Recall@3", "Recall@5", "NDCG"}

    def test_values_match_individual(self, small_ranking_df):
        metrics = compute_all_ranking_metrics(small_ranking_df)
        assert metrics["MRR"] == pytest.approx(compute_mrr(small_ranking_df))
        assert metrics["Top1"] == pytest.approx(compute_top1(small_ranking_df))
