"""Canonical model prediction schema."""
import pandera.polars as pa
import polars as pl

# Source mode enum
SOURCE_MODES = ["published_prediction", "local_inference", "remote_inference", "derived_baseline", "derived_ensemble"]

# Applicability enum
APPLICABILITY_STATUS = [
    "APPLICABLE", "NOT_APPLICABLE_CONTEXT", "NOT_APPLICABLE_VARIANT",
    "NOT_APPLICABLE_ELEMENT", "NOT_APPLICABLE_MISSING_DATA"
]

prediction_schema = pa.DataFrameSchema({
    "model_id": pa.Column(str, nullable=False),
    "model_family": pa.Column(str, nullable=False),
    "benchmark_id": pa.Column(str, nullable=False),
    "variant_id": pa.Column(str, nullable=True),
    "element_id": pa.Column(str, nullable=True),
    "gene_id": pa.Column(str, nullable=False),
    "context_id": pa.Column(str, nullable=False),
    "raw_score": pa.Column(float, nullable=True),
    "ranking_score": pa.Column(float, nullable=False),
    "signed_score": pa.Column(float, nullable=True),
    "coverage": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    "applicability": pa.Column(str, pa.Check.isin(APPLICABILITY_STATUS), nullable=False),
    "source_mode": pa.Column(str, pa.Check.isin(SOURCE_MODES), nullable=False),
}, strict=True, coerce=True)
