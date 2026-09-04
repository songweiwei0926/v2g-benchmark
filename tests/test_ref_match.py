"""Tests for REF allele matching against GRCh38 FASTA."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import polars as pl
import pytest

from v2gbench.harmonize.variants import check_ref_match, assign_qc_status


class TestCheckRefMatch:
    def _make_fasta_mock(self, base_map):
        """Create a mock that returns bases for (chrom, pos) lookups."""
        mock_fasta = MagicMock()
        # We patch _read_fasta_base instead
        return mock_fasta

    def test_ref_match_with_mocked_samtools(self, tmp_path):
        """Test ref_match using a mocked samtools faidx."""
        fasta = tmp_path / "ref.fa"
        fasta.write_text(">chr22\nACGTACGTACGT\n")
        fasta_fai = tmp_path / "ref.fa.fai"
        fasta_fai.write_text("chr22\t12\t7\t12\t13\n")

        df = pl.DataFrame({
            "chrom": ["chr22"],
            "pos": [1],
            "ref": ["A"],
        })

        with patch("v2gbench.harmonize.variants._read_fasta_base") as mock_read:
            mock_read.return_value = "A"
            out = check_ref_match(df, fasta)
            assert "ref_match" in out.columns
            assert out["ref_match"][0] is True

    def test_ref_mismatch_detected(self, tmp_path):
        fasta = tmp_path / "ref.fa"
        fasta.write_text(">chr22\nACGT\n")
        (tmp_path / "ref.fa.fai").write_text("chr22\t4\t7\t4\t5\n")

        df = pl.DataFrame({
            "chrom": ["chr22"],
            "pos": [1],
            "ref": ["G"],
        })

        with patch("v2gbench.harmonize.variants._read_fasta_base") as mock_read:
            mock_read.return_value = "A"
            out = check_ref_match(df, fasta)
            assert out["ref_match"][0] is False

    def test_missing_fasta_raises(self, tmp_path):
        df = pl.DataFrame({"chrom": ["chr1"], "pos": [1], "ref": ["A"]})
        with pytest.raises(FileNotFoundError):
            check_ref_match(df, tmp_path / "nonexistent.fa")


class TestAssignQcStatus:
    def test_ref_mismatch_flag(self):
        df = pl.DataFrame({
            "chrom": ["chr1"], "pos": [100],
            "ref": ["A"], "alt": ["G"],
            "ref_match": [False],
        })
        out = assign_qc_status(df)
        assert out["qc_status"][0] == "REF_MISMATCH"

    def test_pass_when_ref_matches(self):
        df = pl.DataFrame({
            "chrom": ["chr1"], "pos": [100],
            "ref": ["A"], "alt": ["G"],
            "ref_match": [True],
        })
        out = assign_qc_status(df)
        assert out["qc_status"][0] == "PASS"

    def test_multiallelic_split(self):
        df = pl.DataFrame({
            "chrom": ["chr1"], "pos": [100],
            "ref": ["A"], "alt": ["G,T"],
        })
        out = assign_qc_status(df)
        assert out["qc_status"][0] == "MULTIALLELIC_SPLIT"

    def test_invalid_allele(self):
        df = pl.DataFrame({
            "chrom": ["chr1"], "pos": [100],
            "ref": [""], "alt": ["G"],
        })
        out = assign_qc_status(df)
        assert out["qc_status"][0] == "INVALID_ALLELE"
