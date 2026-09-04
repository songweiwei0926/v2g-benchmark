#!/usr/bin/env python3
"""Tests for the fixed context mapping system (PART 1).

Tests cover:
- GTEx tissue mapping (all 50 tissues, 100% mapped)
- CRISPR cell type mapping (5 cell types, 100% mapped)
- GWAS trait/context separation (traits → GLOBAL_GWAS)
- GLOBAL_GWAS fallback
- eQTL Catalogue study mapping
- Zero unresolved GTEx tissues
- Mapping provenance fields
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.harmonize.build_context_inventory import (
    GTEX_TISSUE_MAP,
    CRISPR_CELL_TYPES,
    GWAS_TRAITS,
    EQTL_CATALOGUE_STUDIES,
    resolve_context,
    make_context_id,
)


# ---------------------------------------------------------------------------
# GTEx tissue mapping tests
# ---------------------------------------------------------------------------

class TestGTexTissueMapping:
    """Test that all 50 GTEx tissues are correctly mapped."""

    @pytest.mark.parametrize("raw_context", [
        "Whole_Blood",
        "Liver",
        "Brain_Cortex",
        "Brain_Hippocampus",
        "Adipose_Subcutaneous",
        "Artery_Aorta",
        "Skin_Sun_Exposed_Lower_leg",
        "Cells_Cultured_fibroblasts",
        "Cells_EBV-transformed_lymphocytes",
    ])
    def test_gtex_tissue_mapped(self, raw_context):
        """Each GTEx tissue must map to its official canonical name."""
        result = resolve_context(raw_context, "GTEx")
        assert result["mapping_confidence"] == 1.0
        assert result["mapping_method"] == "gtex_official_mapping"
        assert result["context_name"] is not None
        assert result["context_type"] in ("tissue", "cell_line")
        assert result["ontology_id"] is not None

    def test_whole_blood_mapping(self):
        result = resolve_context("Whole_Blood", "GTEx")
        assert result["context_name"] == "Whole Blood"
        assert result["ontology_id"] == "UBERON:0000178"
        assert result["context_supergroup"] == "blood"

    def test_brain_cortex_mapping(self):
        result = resolve_context("Brain_Cortex", "GTEx")
        assert result["context_name"] == "Brain - Cortex"
        assert result["ontology_id"] == "UBERON:0000956"

    def test_brain_hippocampus_mapping(self):
        result = resolve_context("Brain_Hippocampus", "GTEx")
        assert result["context_name"] == "Brain - Hippocampus"
        assert result["ontology_id"] == "UBERON:0002421"

    def test_adipose_subcutaneous_mapping(self):
        result = resolve_context("Adipose_Subcutaneous", "GTEx")
        assert result["context_name"] == "Adipose - Subcutaneous"
        assert result["ontology_id"] == "UBERON:0002190"

    def test_artery_aorta_mapping(self):
        result = resolve_context("Artery_Aorta", "GTEx")
        assert result["context_name"] == "Artery - Aorta"

    def test_skin_sun_exposed_mapping(self):
        result = resolve_context("Skin_Sun_Exposed_Lower_leg", "GTEx")
        assert result["context_name"] == "Skin - Sun Exposed (Lower leg)"

    def test_cells_cultured_fibroblasts_mapping(self):
        result = resolve_context("Cells_Cultured_fibroblasts", "GTEx")
        assert result["context_name"] == "Cells - Cultured fibroblasts"
        assert result["context_type"] == "cell_line"

    def test_cells_ebv_transformed_mapping(self):
        result = resolve_context("Cells_EBV-transformed_lymphocytes", "GTEx")
        assert result["context_name"] == "Cells - EBV-transformed lymphocytes"
        assert result["context_type"] == "cell_line"

    def test_all_50_gtex_tissues_mapped(self):
        """All 50 GTEx tissues must be mapped with confidence 1.0."""
        for raw in GTEX_TISSUE_MAP:
            result = resolve_context(raw, "GTEx")
            assert result["mapping_confidence"] == 1.0, f"GTEx tissue {raw} not mapped"
            assert result["mapping_method"] == "gtex_official_mapping"
            assert result["context_name"] is not None

    def test_zero_unresolved_gtex_tissues(self):
        """No GTEx tissue should be unresolved."""
        for raw in GTEX_TISSUE_MAP:
            result = resolve_context(raw, "GTEx")
            assert result["mapping_method"] != "unresolved", f"GTEx tissue {raw} unresolved"


# ---------------------------------------------------------------------------
# CRISPR cell type mapping tests
# ---------------------------------------------------------------------------

class TestCRISPRCellTypeMapping:
    """Test that all CRISPR cell types are correctly mapped."""

    @pytest.mark.parametrize("cell_type", list(CRISPR_CELL_TYPES.keys()))
    def test_crispr_celltype_mapped(self, cell_type):
        """Each CRISPR cell type must map with confidence 1.0."""
        result = resolve_context(cell_type, "CRISPR")
        assert result["mapping_confidence"] == 1.0
        assert result["mapping_method"] == "crispr_celltype_mapping"
        assert result["context_name"] is not None
        assert result["context_type"] == "cell_line"

    def test_k562_mapping(self):
        result = resolve_context("K562", "CRISPR")
        assert result["context_name"] == "K562"
        assert result["ontology_id"] == "CL:C00002"

    def test_gm12878_mapping(self):
        result = resolve_context("GM12878", "CRISPR")
        assert result["context_name"] == "GM12878"

    def test_all_crispr_celltypes_mapped(self):
        """All 5 CRISPR cell types must be mapped."""
        for raw in CRISPR_CELL_TYPES:
            result = resolve_context(raw, "CRISPR")
            assert result["mapping_confidence"] == 1.0


# ---------------------------------------------------------------------------
# GWAS trait/context separation tests
# ---------------------------------------------------------------------------

class TestGWASTraitSeparation:
    """Test that GWAS traits are separated from biological contexts."""

    @pytest.mark.parametrize("trait", ["Height", "CAD", "T2D", "BrC", "Asthma"])
    def test_gwas_trait_to_global(self, trait):
        """GWAS traits must map to GLOBAL_GWAS, not to a biological context."""
        result = resolve_context(trait, "GWAS")
        assert result["context_id"] == "CTX_GLOBAL_GWAS"
        assert result["context_name"] == "global_or_unspecified"
        assert result["context_type"] == "global"
        assert result["mapping_method"] == "predefined_global_context"
        assert result["mapping_confidence"] == 1.0

    def test_all_gwas_traits_to_global(self):
        """All 94 GWAS traits must map to GLOBAL_GWAS."""
        for trait in GWAS_TRAITS:
            result = resolve_context(trait, "GWAS")
            assert result["context_id"] == "CTX_GLOBAL_GWAS", f"Trait {trait} not global"
            assert result["context_type"] == "global"

    def test_gwas_trait_not_treated_as_context(self):
        """A GWAS trait should not be mapped as a biological context."""
        result = resolve_context("Height", "GWAS")
        assert result["context_type"] != "tissue"
        assert result["context_type"] != "cell_type"
        assert result["context_type"] != "cell_line"


# ---------------------------------------------------------------------------
# GLOBAL_GWAS fallback tests
# ---------------------------------------------------------------------------

class TestGlobalGwasFallback:
    """Test the GLOBAL_GWAS fallback mechanism."""

    def test_opentargets_gwas_label(self):
        """OpenTargets 'GWAS' label should map to GLOBAL_GWAS."""
        result = resolve_context("GWAS", "OpenTargets")
        assert result["context_id"] == "CTX_GLOBAL_GWAS"
        assert result["mapping_method"] == "predefined_global_context"

    def test_unknown_gwas_source_to_global(self):
        """Any context from GWAS source should go to GLOBAL_GWAS."""
        result = resolve_context("SomeUnknownTrait", "GWAS")
        assert result["context_id"] == "CTX_GLOBAL_GWAS"
        assert result["context_type"] == "global"


# ---------------------------------------------------------------------------
# eQTL Catalogue mapping tests
# ---------------------------------------------------------------------------

class TestEQTLCatalogueMapping:
    """Test eQTL Catalogue study mapping."""

    def test_qts000001_mapping(self):
        """QTS000001 should map to a biological context."""
        result = resolve_context("QTS000001", "eQTL_Catalogue")
        assert result["mapping_confidence"] >= 0.9
        assert result["context_name"] is not None
        assert result["context_type"] in ("cell_type", "tissue", "cell_line")


# ---------------------------------------------------------------------------
# Mapping provenance tests
# ---------------------------------------------------------------------------

class TestMappingProvenance:
    """Test that mapping provenance fields are complete."""

    def test_provenance_fields_present(self):
        """Every resolved context must have all provenance fields."""
        result = resolve_context("Whole_Blood", "GTEx")
        required_fields = [
            "raw_context", "context_id", "context_name", "context_type",
            "ontology_id", "mapping_method", "mapping_confidence",
            "context_supergroup",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_context_id_deterministic(self):
        """Same context should produce same context_id."""
        r1 = resolve_context("Liver", "GTEx")
        r2 = resolve_context("Liver", "GTEx")
        assert r1["context_id"] == r2["context_id"]

    def test_different_contexts_different_ids(self):
        """Different contexts should have different context_ids."""
        r1 = resolve_context("Liver", "GTEx")
        r2 = resolve_context("Lung", "GTEx")
        assert r1["context_id"] != r2["context_id"]


# ---------------------------------------------------------------------------
# Integration test: verify output files
# ---------------------------------------------------------------------------

class TestContextMappingOutputs:
    """Verify that the context mapping output files exist and are valid."""

    @pytest.fixture(scope="class")
    def project_root(self):
        return Path("/workspace/v2g-benchmark")

    def test_context_inventory_exists(self, project_root):
        p = project_root / "data" / "interim" / "context_inventory.parquet"
        assert p.exists(), "context_inventory.parquet not found"
        df = pl.read_parquet(str(p))
        assert df.height > 0
        assert "source_dataset" in df.columns
        assert "raw_context" in df.columns
        assert "mapping_status" in df.columns

    def test_gtex_contexts_reference_exists(self, project_root):
        p = project_root / "data" / "reference" / "gtex_contexts.parquet"
        assert p.exists(), "gtex_contexts.parquet not found"
        df = pl.read_parquet(str(p))
        assert df.height == 50

    def test_generated_config_exists(self, project_root):
        p = project_root / "config" / "generated_context_mapping.yaml"
        assert p.exists(), "generated_context_mapping.yaml not found"

    def test_context_mapping_qc_exists(self, project_root):
        p = project_root / "results" / "tables" / "context_mapping_qc.tsv"
        assert p.exists(), "context_mapping_qc.tsv not found"
        df = pl.read_csv(str(p), separator="\t")
        # All mandatory sources must PASS
        for source in ["GTEx", "CRISPR", "eQTL_Catalogue", "GWAS", "OpenTargets"]:
            row = df.filter(pl.col("source") == source)
            assert row.height == 1, f"Source {source} not in QC report"
            assert row["status"][0] == "PASS", f"Source {source} not PASS"

    def test_unmapped_contexts_report_exists(self, project_root):
        p = project_root / "results" / "tables" / "unmapped_contexts.tsv"
        assert p.exists(), "unmapped_contexts.tsv not found"

    def test_context_mapping_parquet_updated(self, project_root):
        p = project_root / "data" / "processed" / "context_mapping.parquet"
        assert p.exists()
        df = pl.read_parquet(str(p))
        assert df.height > 0
        # All should have confidence >= 0.8 (no unmapped)
        unmapped = df.filter(pl.col("mapping_confidence") < 0.8)
        assert unmapped.height == 0, f"{unmapped.height} unmapped contexts remain"
