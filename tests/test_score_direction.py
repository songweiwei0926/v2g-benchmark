"""Tests for signed score direction conventions."""

import numpy as np
import polars as pl
import pytest

from v2gbench.metrics.direction import (
    compute_direction_accuracy,
    compute_balanced_accuracy,
    compute_direction_mcc,
    compute_spearman,
)
from v2gbench.models.alphagenome import compute_signed_score


class TestSignedScoreConvention:
    """signed_score: positive = up-regulating, negative = down-regulating."""

    def test_positive_signed_means_up(self):
        df = pl.DataFrame({
            "signed_score": [1.0, -1.0, 0.5, -0.5],
            "effect_direction": [1, -1, 1, -1],
        })
        acc = compute_direction_accuracy(df)
        assert acc == 1.0

    def test_negative_signed_means_down(self):
        df = pl.DataFrame({
            "signed_score": [-1.0, 1.0],
            "effect_direction": [1, -1],
        })
        acc = compute_direction_accuracy(df)
        assert acc == 0.0

    def test_mixed_accuracy(self):
        df = pl.DataFrame({
            "signed_score": [1.0, -1.0, 1.0, -1.0],
            "effect_direction": [1, -1, -1, 1],
        })
        acc = compute_direction_accuracy(df)
        assert acc == 0.5

    def test_zero_signed_mapped_to_positive(self):
        """By convention, sign(0) = +1."""
        df = pl.DataFrame({
            "signed_score": [0.0],
            "effect_direction": [1],
        })
        acc = compute_direction_accuracy(df)
        assert acc == 1.0


class TestAlphagenomeSignedScore:
    def test_upregulation_positive(self):
        """ALT > REF → positive signed score."""
        signed = compute_signed_score(ref_expr=1.0, alt_expr=2.0)
        assert signed > 0

    def test_downregulation_negative(self):
        """ALT < REF → negative signed score."""
        signed = compute_signed_score(ref_expr=2.0, alt_expr=1.0)
        assert signed < 0

    def test_no_change_near_zero(self):
        signed = compute_signed_score(ref_expr=1.0, alt_expr=1.0)
        assert abs(signed) < 0.01

    def test_symmetry(self):
        """Swapping ref/alt negates the score."""
        s1 = compute_signed_score(1.0, 3.0)
        s2 = compute_signed_score(3.0, 1.0)
        assert abs(s1 + s2) < 1e-10


class TestBalancedAccuracy:
    def test_perfect_prediction(self):
        df = pl.DataFrame({
            "signed_score": [1.0, -1.0, 1.0, -1.0],
            "effect_direction": [1, -1, 1, -1],
        })
        acc = compute_balanced_accuracy(df)
        assert acc == 1.0

    def test_single_class_returns_nan(self):
        df = pl.DataFrame({
            "signed_score": [1.0, 1.0],
            "effect_direction": [1, 1],
        })
        acc = compute_balanced_accuracy(df)
        assert np.isnan(acc)


class TestSpearman:
    def test_perfect_correlation(self):
        df = pl.DataFrame({
            "signed_score": [1.0, 2.0, 3.0, 4.0, 5.0],
            "effect_size": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        rho = compute_spearman(df)
        assert rho == pytest.approx(1.0)

    def test_too_few_returns_nan(self):
        df = pl.DataFrame({
            "signed_score": [1.0, 2.0],
            "effect_size": [10.0, 20.0],
        })
        rho = compute_spearman(df)
        assert np.isnan(rho)
