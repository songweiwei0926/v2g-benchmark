"""Canonical variant schema."""
import pandera.polars as pa
import polars as pl

variant_schema = pa.DataFrameSchema({
    "variant_id": pa.Column(str, pa.Check.str_startswith("GRCh38:"), nullable=False),
    "chrom": pa.Column(str, pa.Check.isin([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]), nullable=False),
    "pos": pa.Column(pl.Int64, pa.Check.ge(1), nullable=False),
    "ref": pa.Column(str, pa.Check.str_length(min_value=1, max_value=100), nullable=False),
    "alt": pa.Column(str, pa.Check.str_length(min_value=1, max_value=100), nullable=False),
    "genome_build": pa.Column(str, pa.Check.equal_to("GRCh38"), nullable=False),
    "rsid": pa.Column(str, nullable=True),
}, strict=True, coerce=True)


def make_variant_id(chrom: str, pos: int, ref: str, alt: str, build: str = "GRCh38") -> str:
    """Generate canonical variant ID: GRCh38:chr1:123456:A:G"""
    return f"{build}:{chrom}:{pos}:{ref}:{alt}"


def parse_variant_id(variant_id: str) -> dict:
    """Parse a canonical variant ID into components."""
    parts = variant_id.split(":")
    if len(parts) != 5:
        raise ValueError(f"Invalid variant_id: {variant_id}")
    return {
        "genome_build": parts[0],
        "chrom": parts[1],
        "pos": int(parts[2]),
        "ref": parts[3],
        "alt": parts[4],
    }


# QC status enum
VARIANT_QC_STATUS = ["PASS", "REF_MISMATCH", "MULTIALLELIC_SPLIT", "LIFTOVER_FAILED", "INVALID_ALLELE"]
