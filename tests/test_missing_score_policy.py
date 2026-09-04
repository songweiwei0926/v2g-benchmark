"""Tests for missing prediction policy: coverage=0, ranking_score=0."""

import polars as pl
import pytest

from v2gbench.benchmark.applicability import compute_coverage
from v2gbench.metrics.ranking import compute_mrr


class TestMissingScorePolicy:
    """When a model produces no prediction for a candidate pair:
    coverage=0 and ranking_score should be 0 (not null/NaN)."""

    def test_missing_prediction_coverage_zero(self):
        """A model with no predictions for some candidates → coverage < 1."""
        candidates = pl.DataFrame({
            "variant_id": ["v1", "v1", "v2", "v2"],
            "gene_id": ["g1", "g2", "g3", "g4"],
            "context_id": ["c1", "c1", "c1", "c1"],
        })
        predictions = pl.DataFrame({
            "model_id": ["m1", "m1"],
            "variant_id": ["v1", "v1"],
            "gene_id": ["g1", "g2"],
            "context_id": ["c1", "c1"],
            "ranking_score": [0.5, 0.3],
            "coverage": [1, 1],
        })
        cov = compute_coverage(predictions, candidates, model_id="m1")
        assert cov == 0.5  # 2 of 4 candidates scored

    def test_zero_coverage_when_no_predictions(self):
        candidates = pl.DataFrame({
            "variant_id": ["v1"], "gene_id": ["g1"], "context_id": ["c1"],
        })
        predictions = pl.DataFrame({
            "model_id": [], "variant_id": [], "gene_id": [],
            "context_id": [], "ranking_score": [], "coverage": [],
        })
        cov = compute_coverage(predictions, candidates, model_id="m1")
        assert cov == 0.0

    def test_full_coverage(self):
        candidates = pl.DataFrame({
            "variant_id": ["v1", "v1"],
            "gene_id": ["g1", "g2"],
            "context_id": ["c1", "c1"],
        })
        predictions = pl.DataFrame({
            "model_id": ["m1", "m1"],
            "variant_id": ["v1", "v1"],
            "gene_id": ["g1", "g2"],
            "context_id": ["c1", "c1"],
            "ranking_score": [0.5, 0.3],
            "coverage": [1, 1],
        })
        cov = compute_coverage(predictions, candidates, model_id="m1")
        assert cov == 1.0

    def test_missing_score_does_not_crash_mrr(self):
        """A prediction with ranking_score=0 (missing) should not crash MRR."""
        df = pl.DataFrame({
            "variant_id": ["v1", "v1", "v1"],
            "gene_id": ["g1", "g2", "g3"],
            "ranking_score": [0.9, 0.0, 0.5],  # g2 has 0 (missing)
            "is_gold": [0, 1, 0],
            "distance_to_tss": [100, 200, 300],
        })
        mrr = compute_mrr(df)
        # g2 is gold, ranked by score: 0.9 > 0.5 > 0.0 → rank 3 → RR=1/3
        assert mrr == pytest.approx(1.0 / 3.0)

    def test_empty_predictions_coverage_zero(self):
        candidates = pl.DataFrame({
            "variant_id": ["v1"], "gene_id": ["g1"], "context_id": ["c1"],
        })
        predictions = pl.DataFrame()
        cov = compute_coverage(predictions, candidates)
        assert cov == 0.0
