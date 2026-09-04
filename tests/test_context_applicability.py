"""Tests for applicability vs coverage distinction."""

import polars as pl
import pytest

from v2gbench.benchmark.applicability import (
    check_model_applicability,
    build_applicability_matrix,
    compute_coverage,
)


@pytest.fixture
def models_config():
    return {
        "models": {
            "model_a": {
                "family": "e2g",
                "mode": "published_prediction",
                "applicability": {
                    "contexts": ["k562", "hepg2"],
                },
            },
            "model_b": {
                "family": "sequence",
                "mode": "local_inference",
            },
            "model_c": {
                "family": "e2g",
                "mode": "published_prediction",
                "applicability": {
                    "excluded_contexts": ["brain"],
                },
            },
        }
    }


class TestApplicability:
    def test_applicable_when_no_restrictions(self, models_config):
        assert check_model_applicability("model_b", "any_ctx", models_config) == "APPLICABLE"

    def test_not_applicable_context_not_in_allowlist(self, models_config):
        assert check_model_applicability("model_a", "brain", models_config) == "NOT_APPLICABLE_CONTEXT"

    def test_applicable_context_in_allowlist(self, models_config):
        assert check_model_applicability("model_a", "k562", models_config) == "APPLICABLE"

    def test_excluded_context(self, models_config):
        assert check_model_applicability("model_c", "brain", models_config) == "NOT_APPLICABLE_CONTEXT"

    def test_not_excluded_context_applicable(self, models_config):
        assert check_model_applicability("model_c", "heart", models_config) == "APPLICABLE"

    def test_unknown_model_returns_missing_data(self, models_config):
        assert check_model_applicability("nonexistent", "k562", models_config) == "NOT_APPLICABLE_MISSING_DATA"


class TestApplicabilityMatrix:
    def test_matrix_shape(self, models_config):
        contexts = pl.DataFrame({
            "context_id": ["k562", "hepg2", "brain"],
            "context_type": ["cell_line", "cell_line", "tissue"],
        })
        matrix = build_applicability_matrix(models_config, contexts)
        assert "model_id" in matrix.columns
        assert "context_id" in matrix.columns
        assert "applicability" in matrix.columns
        # 3 models × 3 contexts = 9 rows
        assert matrix.height == 9

    def test_matrix_values(self, models_config):
        contexts = pl.DataFrame({"context_id": ["k562", "brain"]})
        matrix = build_applicability_matrix(models_config, contexts)
        a_k562 = matrix.filter(
            (pl.col("model_id") == "model_a") & (pl.col("context_id") == "k562")
        )
        assert a_k562["applicability"][0] == "APPLICABLE"
        a_brain = matrix.filter(
            (pl.col("model_id") == "model_a") & (pl.col("context_id") == "brain")
        )
        assert a_brain["applicability"][0] == "NOT_APPLICABLE_CONTEXT"


class TestCoverageVsApplicability:
    """Applicability = can the model score this context?
    Coverage = did the model actually produce a score?"""

    def test_applicable_but_not_covered(self):
        """Model is applicable but produced no predictions → coverage=0."""
        candidates = pl.DataFrame({
            "variant_id": ["v1"], "gene_id": ["g1"], "context_id": ["k562"],
        })
        predictions = pl.DataFrame({
            "model_id": [], "variant_id": [], "gene_id": [],
            "context_id": [], "ranking_score": [], "coverage": [],
        })
        cov = compute_coverage(predictions, candidates, model_id="m1")
        assert cov == 0.0

    def test_applicable_and_covered(self):
        candidates = pl.DataFrame({
            "variant_id": ["v1"], "gene_id": ["g1"], "context_id": ["k562"],
        })
        predictions = pl.DataFrame({
            "model_id": ["m1"], "variant_id": ["v1"], "gene_id": ["g1"],
            "context_id": ["k562"], "ranking_score": [0.5], "coverage": [1],
        })
        cov = compute_coverage(predictions, candidates, model_id="m1")
        assert cov == 1.0
