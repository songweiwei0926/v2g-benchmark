"""Canonical candidate pair schema."""
import pandera.polars as pa
import polars as pl

candidate_schema = pa.DataFrameSchema({
    "candidate_set_id": pa.Column(str, nullable=False),
    "variant_id": pa.Column(str, nullable=True),
    "gene_id": pa.Column(str, nullable=False),
    "context_id": pa.Column(str, nullable=False),
    "distance_to_tss": pa.Column(pl.Int64, nullable=True),
    "distance_rank": pa.Column(pl.Int64, pa.Check.ge(1), nullable=True),
    "is_nearest": pa.Column(bool, nullable=True),
    "is_gold": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    "gold_confidence": pa.Column(float, nullable=True),
    "candidate_basis": pa.Column(str, nullable=False),
}, strict=True, coerce=True)

# Candidate basis enum
CANDIDATE_BASIS = ["CONTEXT_TESTED", "CONTEXT_EXPRESSED", "GENCODE_FALLBACK"]
