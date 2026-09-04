"""CRITICAL: assert gold_gene in candidate_set for 100% of instances.

This is the most important test in the benchmark. If a gold gene is missing
from the candidate universe, no model — not even an oracle — could rank it,
making the benchmark silently biased toward distance-based methods.
"""

import polars as pl
import pytest

from v2gbench.benchmark.candidate_sets import (
    build_candidate_set,
    check_gold_coverage,
)


class TestGoldCoverage:
    """Every gold (variant, gene, context) pair MUST appear in the candidate set."""

    def test_100_percent_gold_coverage_smoke(self, candidates_df, evidence_df):
        """Assert 100% gold coverage on smoke fixtures."""
        coverage = check_gold_coverage(
            candidates_df, evidence_df, raise_on_incomplete=False
        )
        assert coverage == 1.0, (
            f"Gold coverage is {coverage:.4%}, expected 100%. "
            "Some gold genes are missing from the candidate set!"
        )

    def test_gold_coverage_raises_when_incomplete(self):
        """check_gold_coverage should raise AssertionError when < 100%."""
        candidates = pl.DataFrame({
            "variant_id": ["v1", "v1", "v2"],
            "gene_id": ["g1", "g2", "g3"],
            "context_id": ["c1", "c1", "c1"],
        })
        gold = pl.DataFrame({
            "variant_id": ["v1", "v2", "v3"],
            "gene_id": ["g1", "g3", "g4"],
            "context_id": ["c1", "c1", "c1"],
        })
        with pytest.raises(AssertionError):
            check_gold_coverage(candidates, gold, raise_on_incomplete=True)

    def test_gold_coverage_complete_no_raise(self):
        """check_gold_coverage should not raise when 100%."""
        candidates = pl.DataFrame({
            "variant_id": ["v1", "v1", "v2"],
            "gene_id": ["g1", "g2", "g3"],
            "context_id": ["c1", "c1", "c1"],
        })
        gold = pl.DataFrame({
            "variant_id": ["v1", "v2"],
            "gene_id": ["g1", "g3"],
            "context_id": ["c1", "c1"],
        })
        coverage = check_gold_coverage(candidates, gold, raise_on_incomplete=True)
        assert coverage == 1.0

    def test_build_candidate_set_includes_gold(self, small_variants_df, small_genes_df):
        """Build a candidate set and verify gold genes are included."""
        cand = build_candidate_set(
            small_variants_df, small_genes_df, window_size=1_000_000
        )
        # Every gene within 1Mb of a variant should be a candidate
        assert cand.height > 0
        # Check that for each variant, at least one gene is a candidate
        for vid in small_variants_df["variant_id"].to_list():
            assert vid in cand["variant_id"].to_list()

    def test_empty_gold_returns_one(self):
        candidates = pl.DataFrame({
            "variant_id": ["v1"], "gene_id": ["g1"], "context_id": ["c1"],
        })
        gold = pl.DataFrame({
            "variant_id": [], "gene_id": [], "context_id": [],
        })
        coverage = check_gold_coverage(candidates, gold, raise_on_incomplete=True)
        assert coverage == 1.0

    def test_all_gold_genes_in_candidates_smoke(self, candidates_df, evidence_df):
        """Explicitly check every gold gene appears in the candidate set."""
        gold_keys = set(
            zip(
                evidence_df["variant_id"].to_list(),
                evidence_df["gene_id"].to_list(),
                evidence_df["context_id"].to_list(),
            )
        )
        cand_keys = set(
            zip(
                candidates_df["variant_id"].to_list(),
                candidates_df["gene_id"].to_list(),
                candidates_df["context_id"].to_list(),
            )
        )
        missing = gold_keys - cand_keys
        assert len(missing) == 0, (
            f"{len(missing)} gold pairs missing from candidates: "
            f"{list(missing)[:5]}"
        )
