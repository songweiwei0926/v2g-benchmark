"""Tests for Enformer CAGE-track score mapping to genes."""

import numpy as np
import polars as pl
import pytest

from v2gbench.models.enformer import (
    match_cage_tracks,
    _extract_tss_cage,
    EnformerAdapter,
    ENFORMER_BIN_SIZE,
    TSS_HALF_WINDOW,
)


@pytest.fixture
def track_metadata():
    return pl.DataFrame({
        "track_index": [0, 1, 2, 3, 4],
        "species": ["human", "human", "human", "mouse", "human"],
        "description": ["K562 CAGE", "HepG2 CAGE", "brain CAGE", "heart CAGE", "GM12878 CAGE"],
        "ontology_id": ["CL:CCL_243", "CL:0000372", "UBERON:0000955", None, "CL:0000477"],
        "track_type": ["CAGE", "CAGE", "CAGE", "CAGE", "CAGE"],
    })


class TestMatchCageTracks:
    def test_ontology_id_match(self, track_metadata):
        result = match_cage_tracks("CL:CCL_243", track_metadata)
        assert 0 in result

    def test_description_match(self, track_metadata):
        result = match_cage_tracks("hepg2", track_metadata)
        assert 1 in result

    def test_fallback_all_human_cage(self, track_metadata):
        result = match_cage_tracks("nonexistent", track_metadata)
        # Should fall back to all human CAGE tracks
        assert len(result) > 0
        assert all(r in [0, 1, 2, 4] for r in result)  # not mouse (3)

    def test_empty_metadata(self):
        result = match_cage_tracks("k562", pl.DataFrame())
        assert result == []


class TestExtractTssCage:
    def test_sum_within_window(self):
        """CAGE signal in ±1kb around TSS should be summed."""
        # Create a track with signal at specific bins
        tracks = np.zeros(1000)
        # TSS at genomic pos 100000, window_start at 0, bin_size=128
        # TSS bin = 100000 / 128 ≈ 781
        # ±1000bp → ±8 bins → bins 773 to 789
        tracks[780:790] = 1.0
        val = _extract_tss_cage(tracks, tss_genomic=100000, window_start=0)
        assert val > 0

    def test_zero_outside_window(self):
        tracks = np.zeros(1000)
        tracks[0:10] = 5.0  # signal far from TSS
        val = _extract_tss_cage(tracks, tss_genomic=100000, window_start=0)
        assert val == 0.0

    def test_handles_edge_positions(self):
        tracks = np.zeros(100)
        tracks[0:5] = 1.0
        val = _extract_tss_cage(tracks, tss_genomic=0, window_start=0)
        assert val >= 0  # should not crash


class TestEnformerAdapter:
    def test_adapter_init(self):
        adapter = EnformerAdapter()
        assert adapter.model_id == "enformer"
        assert adapter.model_family == "sequence"

    def test_normalize_score_uses_abs(self):
        df = pl.DataFrame({
            "variant_id": ["v1"],
            "gene_id": ["g1"],
            "enformer_abs": [0.6],
            "enformer_signed": [0.6],
        })
        adapter = EnformerAdapter()
        out = adapter.normalize_score(df)
        assert "ranking_score" in out.columns
        assert out["ranking_score"][0] == 0.6

    def test_applicability_always_applicable(self):
        adapter = EnformerAdapter()
        assert adapter.applicability("any_context") == "APPLICABLE"

    def test_constants(self):
        assert ENFORMER_BIN_SIZE == 128
        assert TSS_HALF_WINDOW == 1_000
