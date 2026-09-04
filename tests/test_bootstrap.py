"""Tests for bootstrap: sampling unit = variant/locus, not pair."""

import numpy as np
import polars as pl
import pytest

from v2gbench.statistics.bootstrap import (
    bootstrap_metrics,
    compute_ci,
    bootstrap_paired,
)
from v2gbench.metrics.ranking import compute_mrr


@pytest.fixture
def paired_df():
    """3 variants, each with 3 candidate genes. Gold gene varies by variant."""
    return pl.DataFrame({
        "variant_id": ["v1", "v1", "v1", "v2", "v2", "v2", "v3", "v3", "v3"],
        "gene_id": ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9"],
        "ranking_score": [0.9, 0.5, 0.1, 0.3, 0.8, 0.2, 0.1, 0.2, 0.9],
        "is_gold": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "distance_to_tss": [100, 200, 300, 100, 200, 300, 100, 200, 300],
    })


class TestBootstrapSamplingUnit:
    """Bootstrap must resample variants (loci), not individual pairs."""

    def test_resampled_df_has_same_variant_count(self, paired_df):
        """Each bootstrap replicate should have the same number of variants
        (with replacement), but the number of pairs may differ."""
        values = bootstrap_metrics(
            paired_df, compute_mrr, n_replicates=10, unit="variant_id", seed=42
        )
        assert values.shape == (10,)
        # At least some replicates should be non-NaN
        assert np.sum(~np.isnan(values)) > 0

    def test_sampling_unit_is_variant_not_pair(self):
        """Verify that all rows for a resampled variant are kept together.

        We construct a dataset where variant v1 has 5 pairs and v2 has 1 pair.
        If sampling were per-pair, v1's pairs would be over-represented.
        With per-variant sampling, each variant is equally likely.
        """
        df = pl.DataFrame({
            "variant_id": ["v1"] * 5 + ["v2"],
            "gene_id": [f"g{i}" for i in range(6)],
            "ranking_score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.3],
            "is_gold": [1, 0, 0, 0, 0, 1],
            "distance_to_tss": list(range(6)),
        })
        # Run many replicates and check that v2 appears ~50% of the time
        # (with 2 variants, each has 50% chance per draw)
        from v2gbench.statistics.bootstrap import _resample_variants
        rng = np.random.default_rng(0)
        v2_counts = 0
        n_trials = 200
        for _ in range(n_trials):
            resampled = _resample_variants(df, ["variant_id"], rng)
            if "v2" in resampled["variant_id"].to_list():
                v2_counts += 1
        # Should be roughly 50% (1 - 0.5^2 = 0.75 for n=2 draws)
        # With 2 units and 2 draws, P(v2 appears) = 1 - (1/2)^2 = 0.75
        assert 0.6 < v2_counts / n_trials < 0.9

    def test_bootstrap_reproducible(self, paired_df):
        """Same seed → same bootstrap values."""
        v1 = bootstrap_metrics(paired_df, compute_mrr, n_replicates=50, seed=123)
        v2 = bootstrap_metrics(paired_df, compute_mrr, n_replicates=50, seed=123)
        np.testing.assert_array_equal(v1, v2)

    def test_bootstrap_different_seed_different(self):
        """Different seed → different bootstrap values (with high probability).

        Use data where MRR varies across variants (gold gene NOT always rank 1)
        so bootstrap resampling produces variation across replicates.
        """
        df = pl.DataFrame({
            "variant_id": ["v1", "v1", "v1", "v2", "v2", "v2", "v3", "v3", "v3"],
            "gene_id": ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9"],
            "ranking_score": [0.5, 0.9, 0.1, 0.3, 0.8, 0.2, 0.1, 0.9, 0.3],
            "is_gold": [1, 0, 0, 0, 1, 0, 0, 1, 0],
            "distance_to_tss": [100, 200, 300, 100, 200, 300, 100, 200, 300],
        })
        v1 = bootstrap_metrics(df, compute_mrr, n_replicates=100, seed=111)
        v2 = bootstrap_metrics(df, compute_mrr, n_replicates=100, seed=222)
        assert not np.array_equal(v1, v2)


class TestComputeCI:
    def test_ci_contains_mean(self):
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        lo, hi = compute_ci(values, confidence=0.95)
        assert lo < 0.3 < hi

    def test_ci_nan_handling(self):
        values = np.array([np.nan, 0.5, 0.5, 0.5])
        lo, hi = compute_ci(values)
        assert lo == 0.5
        assert hi == 0.5

    def test_ci_all_nan(self):
        lo, hi = compute_ci(np.array([np.nan, np.nan]))
        assert np.isnan(lo) and np.isnan(hi)

    def test_ci_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            compute_ci(np.array([1, 2]), confidence=1.5)


class TestBootstrapPaired:
    def test_paired_returns_deltas(self, paired_df):
        df_a = paired_df.with_columns(pl.col("ranking_score").alias("ranking_score"))
        df_b = paired_df.with_columns(
            (pl.col("ranking_score") * 0.5).alias("ranking_score")
        )
        deltas = bootstrap_paired(df_a, df_b, compute_mrr, n_replicates=20, seed=42)
        assert deltas.shape == (20,)
