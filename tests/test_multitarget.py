"""Tests for multi-gold (multi-target) variant handling."""

import polars as pl
import pytest

from v2gbench.metrics.ranking import (
    compute_mrr,
    compute_top1,
    compute_recall_at_k,
    compute_ndcg,
)


class TestMultiGold:
    """A variant with multiple gold genes should be handled correctly."""

    @pytest.fixture
    def multi_gold_df(self):
        """v1 has 2 gold genes (g1 at rank 1, g2 at rank 3)."""
        return pl.DataFrame({
            "variant_id": ["v1", "v1", "v1", "v1"],
            "gene_id": ["g1", "g2", "g3", "g4"],
            "ranking_score": [0.9, 0.3, 0.5, 0.1],
            "is_gold": [1, 1, 0, 0],
            "distance_to_tss": [1000, 5000, 3000, 10000],
        })

    def test_mrr_uses_best_gold(self, multi_gold_df):
        """MRR uses the best (lowest-rank) gold gene → 1/1 = 1.0."""
        assert compute_mrr(multi_gold_df) == 1.0

    def test_top1_any_gold_at_1(self, multi_gold_df):
        """Top1 = 1 if any gold is at rank 1."""
        assert compute_top1(multi_gold_df) == 1.0

    def test_recall_at_2(self, multi_gold_df):
        """Recall@2: g1 is in top 2 → hit=1 → Recall@2=1.0."""
        assert compute_recall_at_k(multi_gold_df, 2) == 1.0

    def test_recall_at_3_both_gold(self, multi_gold_df):
        """Recall@3: both g1 (rank 1) and g2 (rank 3) in top 3 → hit=1."""
        assert compute_recall_at_k(multi_gold_df, 3) == 1.0

    def test_ndcg_multi_gold(self, multi_gold_df):
        """NDCG should be < 1.0 when gold genes are not all at the top."""
        ndcg = compute_ndcg(multi_gold_df)
        assert 0.0 < ndcg <= 1.0

    def test_multi_gold_different_variants(self):
        """Two variants each with multiple gold genes."""
        df = pl.DataFrame({
            "variant_id": ["v1", "v1", "v1", "v2", "v2", "v2"],
            "gene_id": ["g1", "g2", "g3", "g4", "g5", "g6"],
            "ranking_score": [0.9, 0.8, 0.1, 0.5, 0.3, 0.1],
            "is_gold": [1, 1, 0, 1, 1, 0],
            "distance_to_tss": [100, 200, 300, 100, 200, 300],
        })
        # v1: both gold at top → RR=1.0; v2: gold at rank 1,2 → RR=1.0
        assert compute_mrr(df) == 1.0
        assert compute_top1(df) == 1.0
