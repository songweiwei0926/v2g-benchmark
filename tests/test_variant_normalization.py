"""Tests for variant normalization and canonical ID generation."""

import polars as pl
import pytest

from v2gbench.schemas.variant import make_variant_id, parse_variant_id
from v2gbench.harmonize.variants import generate_variant_id, normalize_variant_df


class TestMakeVariantId:
    def test_canonical_format(self):
        vid = make_variant_id("chr1", 123456, "A", "G")
        assert vid == "GRCh38:chr1:123456:A:G"

    def test_custom_build(self):
        vid = make_variant_id("chrX", 100, "C", "T", build="GRCh37")
        assert vid.startswith("GRCh37:")

    def test_parse_roundtrip(self):
        vid = "GRCh38:chr22:17000000:A:G"
        parsed = parse_variant_id(vid)
        assert parsed["chrom"] == "chr22"
        assert parsed["pos"] == 17000000
        assert parsed["ref"] == "A"
        assert parsed["alt"] == "G"
        assert parsed["genome_build"] == "GRCh38"

    def test_parse_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_variant_id("invalid:id")


class TestGenerateVariantId:
    def test_chr_prefix_added(self):
        vid = generate_variant_id("1", 100, "a", "g")
        assert vid == "GRCh38:chr1:100:A:G"

    def test_chr_prefix_preserved(self):
        vid = generate_variant_id("chr1", 100, "A", "G")
        assert vid == "GRCh38:chr1:100:A:G"

    def test_alleles_uppercased(self):
        vid = generate_variant_id("chr1", 100, "a", "t")
        assert ":A:T" in vid


class TestNormalizeVariantDf:
    def test_adds_variant_id(self, small_variants_df):
        df = small_variants_df.drop("variant_id")
        out = normalize_variant_df(df, assign_qc=False)
        assert "variant_id" in out.columns
        assert out.height == df.height
        first_id = out["variant_id"][0]
        assert first_id.startswith("GRCh38:chr22:")

    def test_genome_build_column(self, small_variants_df):
        df = small_variants_df.drop("variant_id")
        out = normalize_variant_df(df, assign_qc=False)
        assert "genome_build" in out.columns
        assert (out["genome_build"] == "GRCh38").all()

    def test_qc_status_pass(self, small_variants_df):
        df = small_variants_df.drop("variant_id")
        out = normalize_variant_df(df, assign_qc=True)
        assert "qc_status" in out.columns
        # Without ref_match check (no fasta), all should be PASS
        assert (out["qc_status"] == "PASS").all()

    def test_missing_columns_raises(self):
        df = pl.DataFrame({"chrom": ["chr1"], "pos": [100]})
        with pytest.raises(ValueError):
            normalize_variant_df(df)

    def test_smoke_variants_have_valid_ids(self, variants_df):
        for vid in variants_df["variant_id"].to_list():
            parsed = parse_variant_id(vid)
            assert parsed["genome_build"] == "GRCh38"
            assert parsed["chrom"] == "chr22"
            assert parsed["pos"] >= 1
