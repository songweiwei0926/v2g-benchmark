"""Canonical gene schema."""
import pandera.polars as pa
import polars as pl
import re

gene_schema = pa.DataFrameSchema({
    "gene_id": pa.Column(str, pa.Check.str_matches(r"^ENSG\d{11}$"), nullable=False),
    "gene_symbol": pa.Column(str, nullable=True),
    "chrom": pa.Column(str, nullable=False),
    "start": pa.Column(pl.Int64, pa.Check.ge(1), nullable=False),
    "end": pa.Column(pl.Int64, pa.Check.ge(1), nullable=False),
    "strand": pa.Column(str, pa.Check.isin(["+", "-"]), nullable=False),
    "tss": pa.Column(pl.Int64, pa.Check.ge(1), nullable=False),
    "gene_type": pa.Column(str, nullable=True),
    "canonical_transcript": pa.Column(str, nullable=True),
    "exon_intervals": pa.Column(str, nullable=True),  # JSON-encoded list of [start, end]
}, strict=True, coerce=True)


def deversion_gene_id(gene_id: str) -> str:
    """Remove version suffix from Ensembl gene ID.
    
    ENSG00000123456.12 → ENSG00000123456
    """
    return re.sub(r"\.\d+$", "", gene_id)


def compute_tss(start: int, end: int, strand: str) -> int:
    """Compute TSS from gene coordinates and strand."""
    if strand == "+":
        return start
    else:
        return end
