"""Tests for GENCODE gene mapping: ID de-versioning and TSS extraction."""

import polars as pl
import pytest

from v2gbench.schemas.gene import deversion_gene_id, compute_tss
from v2gbench.harmonize.genes import deversion_gene_ids, parse_gencode_gtf


class TestDeversionGeneId:
    def test_removes_version_suffix(self):
        assert deversion_gene_id("ENSG00000123456.12") == "ENSG00000123456"

    def test_no_suffix_unchanged(self):
        assert deversion_gene_id("ENSG00000123456") == "ENSG00000123456"

    def test_multiple_dots(self):
        # Ensembl IDs only have one version suffix; regex removes only the last .N
        assert deversion_gene_id("ENSG00000123456.12.3") == "ENSG00000123456.12"

    def test_empty_string(self):
        assert deversion_gene_id("") == ""


class TestComputeTss:
    def test_plus_strand(self):
        assert compute_tss(1000, 5000, "+") == 1000

    def test_minus_strand(self):
        assert compute_tss(1000, 5000, "-") == 5000

    def test_single_base_gene_plus(self):
        assert compute_tss(100, 100, "+") == 100

    def test_single_base_gene_minus(self):
        assert compute_tss(100, 100, "-") == 100


class TestDeversionGeneIds:
    def test_dataframe_deversion(self):
        df = pl.DataFrame({
            "gene_id": ["ENSG00000000001.5", "ENSG00000000002.10"],
            "other": [1, 2],
        })
        out = deversion_gene_ids(df)
        assert out["gene_id"].to_list() == ["ENSG00000000001", "ENSG00000000002"]
        assert out["other"].to_list() == [1, 2]

    def test_missing_column_raises(self):
        df = pl.DataFrame({"x": [1]})
        with pytest.raises(ValueError):
            deversion_gene_ids(df)


class TestParseGencodeGtf:
    def test_parse_minimal_gtf(self, tmp_path):
        gtf = tmp_path / "test.gtf"
        gtf.write_text(
            'chr22\tGENCODE\tgene\t17000100\t17050000\t.\t+\t.\t'
            'gene_id "ENSG00000000001.5"; gene_name "G1"; gene_type "protein_coding";\n'
            'chr22\tGENCODE\tgene\t17900000\t18100000\t.\t-\t.\t'
            'gene_id "ENSG00000000002.10"; gene_name "G2"; gene_type "lncRNA";\n'
            'chr22\tGENCODE\texon\t17000100\t17000200\t.\t+\t.\t'
            'gene_id "ENSG00000000001.5"; transcript_id "ENST00000000001.1"; exon_number "1";\n'
        )
        df = parse_gencode_gtf(gtf)
        assert df.height == 2
        assert "gene_id" in df.columns
        assert "tss" in df.columns
        # De-versioned
        assert "ENSG00000000001" in df["gene_id"].to_list()
        # TSS: + strand → start, - strand → end
        g1 = df.filter(pl.col("gene_id") == "ENSG00000000001")
        assert g1["tss"][0] == 17000100
        g2 = df.filter(pl.col("gene_id") == "ENSG00000000002")
        assert g2["tss"][0] == 18100000

    def test_empty_gtf(self, tmp_path):
        gtf = tmp_path / "empty.gtf"
        gtf.write_text("##gff-version 3\n")
        df = parse_gencode_gtf(gtf)
        assert df.is_empty()

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_gencode_gtf("/nonexistent/path.gtf")


class TestSmokeGeneMaster:
    def test_gene_ids_deversioned(self, gene_master_df):
        for gid in gene_master_df["gene_id"].to_list():
            assert "." not in gid
            assert gid.startswith("ENSG")

    def test_tss_present(self, gene_master_df):
        assert gene_master_df["tss"].null_count() == 0
        assert (gene_master_df["tss"] >= 1).all()

    def test_strand_valid(self, gene_master_df):
        strands = gene_master_df["strand"].unique().to_list()
        assert all(s in ["+", "-"] for s in strands)
