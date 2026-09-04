"""Tests: no duplicate (variant, gene, context) pairs in candidate sets."""

import polars as pl
import pytest

from v2gbench.benchmark.candidate_sets import build_candidate_set


class TestNoDuplicates:
    def test_no_duplicate_pairs_smoke(self, candidates_df):
        """The smoke candidate set must have no duplicate (variant, gene, context)."""
        key_cols = ["variant_id", "gene_id", "context_id"]
        dup_count = candidates_df.group_by(key_cols).len().filter(pl.col("len") > 1)
        assert dup_count.height == 0, (
            f"Found {dup_count.height} duplicate (variant, gene, context) pairs"
        )

    def test_no_duplicate_pairs_built(self, small_variants_df, small_genes_df):
        """build_candidate_set must not produce duplicates."""
        cand = build_candidate_set(
            small_variants_df, small_genes_df, window_size=1_000_000
        )
        key_cols = ["variant_id", "gene_id", "context_id"]
        dup_count = cand.group_by(key_cols).len().filter(pl.col("len") > 1)
        assert dup_count.height == 0

    def test_unique_count_matches_row_count(self, candidates_df):
        """Number of unique key combos should equal row count."""
        n_rows = candidates_df.height
        n_unique = candidates_df.select(
            pl.struct(["variant_id", "gene_id", "context_id"]).n_unique()
        ).item()
        assert n_rows == n_unique

    def test_no_duplicate_predictions(self, predictions_df):
        """Predictions should also have no duplicate (model, variant, gene, context)."""
        key_cols = ["model_id", "variant_id", "gene_id", "context_id"]
        dup_count = predictions_df.group_by(key_cols).len().filter(pl.col("len") > 1)
        assert dup_count.height == 0
