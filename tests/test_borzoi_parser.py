"""Tests for Borzoi 4-replicate output aggregation."""

import polars as pl
import pytest

from v2gbench.models.borzoi import (
    ensemble_borzoi_scores,
    BorzoiAdapter,
    DEFAULT_REPLICATES,
)


class TestEnsembleBorzoiScores:
    """4-replicate output should be aggregated by mean + SD."""

    @pytest.fixture
    def replicate_scores(self):
        """4 replicates, each with 2 genes for 1 variant."""
        base = [
            ("v1", "g1", 1.0, 2.0, 1.0, 2.0),
            ("v1", "g2", 0.5, 0.5, 0.5, 0.5),
        ]
        frames = []
        for rep_idx in range(4):
            rows = []
            for vid, gid, s1, s2, a1, a2 in base:
                # Add small per-replicate variation
                signed = s1 + 0.1 * rep_idx
                abs_val = a1 + 0.1 * rep_idx
                rows.append({
                    "variant_id": vid,
                    "gene_id": gid,
                    "ref_rna": 10.0,
                    "alt_rna": 10.0 + signed,
                    "signed": signed,
                    "abs": abs_val,
                })
            frames.append(pl.DataFrame(rows))
        return frames

    def test_ensemble_produces_mean(self, replicate_scores):
        ens = ensemble_borzoi_scores(replicate_scores)
        assert "borzoi_signed" in ens.columns
        assert "borzoi_abs" in ens.columns
        assert "borzoi_sd_across_replicates" in ens.columns
        # g1: signed values are 1.0, 1.1, 1.2, 1.3 → mean = 1.15
        g1 = ens.filter(pl.col("gene_id") == "g1")
        assert g1["borzoi_signed"][0] == pytest.approx(1.15, abs=0.01)

    def test_ensemble_produces_sd(self, replicate_scores):
        ens = ensemble_borzoi_scores(replicate_scores)
        g1 = ens.filter(pl.col("gene_id") == "g1")
        # SD of [1.0, 1.1, 1.2, 1.3] ≈ 0.129
        assert g1["borzoi_sd_across_replicates"][0] > 0

    def test_ensemble_constant_replicates_sd_zero(self):
        """If all replicates agree, SD should be 0 (not null)."""
        frames = []
        for _ in range(4):
            frames.append(pl.DataFrame({
                "variant_id": ["v1"],
                "gene_id": ["g1"],
                "signed": [1.0],
                "abs": [1.0],
            }))
        ens = ensemble_borzoi_scores(frames)
        assert ens["borzoi_sd_across_replicates"][0] == 0.0

    def test_ensemble_empty(self):
        ens = ensemble_borzoi_scores([])
        assert ens.height == 0
        assert "borzoi_signed" in ens.columns

    def test_ensemble_preserves_all_genes(self, replicate_scores):
        ens = ensemble_borzoi_scores(replicate_scores)
        assert ens.height == 2  # 2 genes


class TestBorzoiAdapter:
    def test_adapter_init(self):
        adapter = BorzoiAdapter(config={"replicates": DEFAULT_REPLICATES})
        assert adapter.model_id == "borzoi"
        assert adapter.model_family == "sequence"
        assert len(adapter.replicates) == 4

    def test_default_replicates(self):
        assert DEFAULT_REPLICATES == [
            "replicate_0", "replicate_1", "replicate_2", "replicate_3"
        ]

    def test_normalize_score_uses_abs(self):
        df = pl.DataFrame({
            "variant_id": ["v1"],
            "gene_id": ["g1"],
            "borzoi_abs": [0.7],
            "borzoi_signed": [0.7],
        })
        adapter = BorzoiAdapter()
        out = adapter.normalize_score(df)
        assert "ranking_score" in out.columns
        assert out["ranking_score"][0] == 0.7

    def test_applicability_always_applicable(self):
        adapter = BorzoiAdapter()
        assert adapter.applicability("any_context") == "APPLICABLE"
