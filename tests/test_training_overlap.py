"""Tests for training-overlap (leakage) label assignment."""

import polars as pl
import pytest

from v2gbench.benchmark.leakage import (
    build_leakage_registry,
    assign_leakage_type,
    filter_strict_no_leakage,
)


@pytest.fixture
def leakage_registry():
    return pl.DataFrame({
        "model_id": ["model_a", "model_b"],
        "family": ["e2g", "sequence"],
        "mode": ["published_prediction", "local_inference"],
        "training_dataset": ["gtex_v8", None],
        "training_celltype": [["k562", "hepg2"], None],
        "training_label_type": ["eQTL", None],
        "training_assay": ["rna_seq", None],
        "training_variants": [None, None],
        "training_genes": [None, None],
        "training_pairs": [["GRCh38:chr1:100:A:G|ENSG00000000001"], None],
        "benchmark_overlap": ["DECLARED", "UNKNOWN"],
    })


class TestAssignLeakageType:
    def test_pair_label_seen(self, leakage_registry):
        result = assign_leakage_type(
            "model_a",
            "GRCh38:chr1:100:A:G",
            "ENSG00000000001",
            "k562",
            leakage_registry,
        )
        assert result == "PAIR_LABEL_SEEN"

    def test_celltype_seen(self, leakage_registry):
        result = assign_leakage_type(
            "model_a",
            "GRCh38:chr1:999:A:G",  # not in training_pairs
            "ENSG00000099999",
            "k562",
            leakage_registry,
        )
        assert result == "CELLTYPE_SEEN"

    def test_dataset_seen(self, leakage_registry):
        """model_a has training_dataset but celltype not matching."""
        result = assign_leakage_type(
            "model_a",
            "GRCh38:chr1:999:A:G",
            "ENSG00000099999",
            "unknown_ctx",
            leakage_registry,
        )
        # celltype not in training, but training_assay exists → ASSAY_TRACK_SEEN
        assert result == "ASSAY_TRACK_SEEN"

    def test_unknown_model(self, leakage_registry):
        result = assign_leakage_type(
            "nonexistent", "v1", "g1", "c1", leakage_registry
        )
        assert result == "UNKNOWN"

    def test_no_known_overlap(self, leakage_registry):
        """model_b has no training info → UNKNOWN."""
        result = assign_leakage_type(
            "model_b", "v1", "g1", "c1", leakage_registry
        )
        assert result == "UNKNOWN"


class TestFilterStrictNoLeakage:
    def test_filters_pair_label_seen(self, leakage_registry):
        evidence = pl.DataFrame({
            "variant_id": [
                "GRCh38:chr1:100:A:G",
                "GRCh38:chr1:200:A:G",
            ],
            "gene_id": ["ENSG00000000001", "ENSG00000000002"],
            "context_id": ["k562", "hepg2"],
        })
        filtered = filter_strict_no_leakage(
            evidence, leakage_registry, model_id="model_a"
        )
        # First pair is PAIR_LABEL_SEEN → filtered out
        assert filtered.height == 1
        assert "ENSG00000000002" in filtered["gene_id"].to_list()

    def test_existing_training_overlap_column(self, leakage_registry):
        evidence = pl.DataFrame({
            "variant_id": ["v1", "v2"],
            "gene_id": ["g1", "g2"],
            "context_id": ["c1", "c2"],
            "training_overlap": ["PAIR_LABEL_SEEN", "NO_KNOWN_OVERLAP"],
        })
        filtered = filter_strict_no_leakage(evidence, leakage_registry)
        assert filtered.height == 1
        assert filtered["training_overlap"][0] == "NO_KNOWN_OVERLAP"


class TestBuildLeakageRegistry:
    def test_from_config_dict(self):
        config = {
            "models": {
                "m1": {
                    "family": "e2g",
                    "mode": "published_prediction",
                    "training": {"dataset": "gtex", "celltype": ["k562"]},
                },
            }
        }
        registry = build_leakage_registry(config)
        assert registry.height == 1
        assert registry["model_id"][0] == "m1"
        assert registry["training_dataset"][0] == "gtex"
