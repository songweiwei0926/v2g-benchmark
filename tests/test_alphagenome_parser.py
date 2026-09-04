"""Tests for AlphaGenome API response parsing to canonical prediction schema."""

import math
import polars as pl
import pytest

from v2gbench.models.alphagenome import (
    compute_signed_score,
    _extract_gene_scores,
    _build_frame,
    AlphaGenomeAdapter,
)


class TestComputeSignedScore:
    def test_upregulation_positive(self):
        assert compute_signed_score(1.0, 2.0) > 0

    def test_downregulation_negative(self):
        assert compute_signed_score(2.0, 1.0) < 0

    def test_pseudocount_eps(self):
        """Very small values should not cause log(0)."""
        score = compute_signed_score(0.0, 0.0)
        assert math.isfinite(score)

    def test_symmetric(self):
        s1 = compute_signed_score(1.0, 3.0)
        s2 = compute_signed_score(3.0, 1.0)
        assert abs(s1 + s2) < 1e-10


class TestExtractGeneScores:
    def test_parse_dict_records(self):
        class MockOutput:
            gene_scores = [
                {"gene_id": "ENSG00000000001", "reference": 1.0, "alternate": 2.0},
                {"gene_id": "ENSG00000000002", "reference": 3.0, "alternate": 1.5},
            ]

        raw = _extract_gene_scores(MockOutput(), "v1", "c1")
        assert "ENSG00000000001" in raw
        assert raw["ENSG00000000001"] == (1.0, 2.0)
        assert raw["ENSG00000000002"] == (3.0, 1.5)

    def test_parse_alt_key_names(self):
        class MockOutput:
            gene_scores = [
                {"gene": "ENSG1", "ref": 1.0, "alt": 2.0},
            ]

        raw = _extract_gene_scores(MockOutput(), "v1", "c1")
        assert "ENSG1" in raw

    def test_empty_output(self):
        class MockOutput:
            gene_scores = []

        raw = _extract_gene_scores(MockOutput(), "v1", "c1")
        assert raw == {}


class TestBuildFrame:
    def test_builds_canonical_columns(self):
        results = [
            {"variant_id": "v1", "gene_id": "g1", "context_id": "c1",
             "alphagenome_signed": 0.5, "alphagenome_abs": 0.5},
            {"variant_id": "v1", "gene_id": "g2", "context_id": "c1",
             "alphagenome_signed": -0.3, "alphagenome_abs": 0.3},
        ]
        df = _build_frame(results)
        assert "variant_id" in df.columns
        assert "gene_id" in df.columns
        assert "alphagenome_signed" in df.columns
        assert "alphagenome_abs" in df.columns
        assert "alphagenome_quantile" in df.columns
        assert df.height == 2

    def test_quantile_within_unit_interval(self):
        results = [
            {"variant_id": "v1", "gene_id": f"g{i}", "context_id": "c1",
             "alphagenome_signed": float(i), "alphagenome_abs": float(i)}
            for i in range(5)
        ]
        df = _build_frame(results)
        q = df["alphagenome_quantile"].to_list()
        assert all(0.0 <= x <= 1.0 for x in q)

    def test_empty_results(self):
        df = _build_frame([])
        assert df.height == 0
        assert "alphagenome_signed" in df.columns


class TestAlphaGenomeAdapter:
    def test_adapter_init(self):
        adapter = AlphaGenomeAdapter(config={"scorer": "RNA_SEQ"})
        assert adapter.model_id == "alphagenome"
        assert adapter.model_family == "sequence"

    def test_normalize_score_uses_quantile(self):
        df = pl.DataFrame({
            "variant_id": ["v1", "v1"],
            "gene_id": ["g1", "g2"],
            "alphagenome_quantile": [0.9, 0.1],
            "alphagenome_abs": [0.5, 0.1],
            "alphagenome_signed": [0.5, -0.1],
        })
        adapter = AlphaGenomeAdapter()
        out = adapter.normalize_score(df)
        assert "ranking_score" in out.columns
        assert out["ranking_score"][0] == 0.9

    def test_normalize_score_fallback_to_abs(self):
        df = pl.DataFrame({
            "variant_id": ["v1"],
            "gene_id": ["g1"],
            "alphagenome_abs": [0.5],
            "alphagenome_signed": [0.5],
        })
        adapter = AlphaGenomeAdapter()
        out = adapter.normalize_score(df)
        assert out["ranking_score"][0] == 0.5

    def test_applicability_always_applicable(self):
        adapter = AlphaGenomeAdapter()
        assert adapter.applicability("any_context") == "APPLICABLE"
