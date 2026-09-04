"""Canonical gold evidence schema."""
import pandera.polars as pa
import polars as pl

# Evidence type enum
EVIDENCE_TYPES = ["CRISPRi", "CRISPRa", "PerturbSeq", "FlowFISH", "eQTL", "GWAS", "curated_L2G"]

# Leakage types
LEAKAGE_TYPES = [
    "PAIR_LABEL_SEEN", "CELLTYPE_SEEN", "ASSAY_TRACK_SEEN",
    "DATASET_SEEN", "NO_KNOWN_OVERLAP", "UNKNOWN"
]

evidence_schema = pa.DataFrameSchema({
    "benchmark_id": pa.Column(str, nullable=False),
    "evidence_id": pa.Column(str, nullable=False),
    "variant_id": pa.Column(str, nullable=True),
    "element_id": pa.Column(str, nullable=True),
    "gene_id": pa.Column(str, nullable=False),
    "context_id": pa.Column(str, nullable=False),
    "trait_id": pa.Column(str, nullable=True),
    "evidence_type": pa.Column(str, pa.Check.isin(EVIDENCE_TYPES), nullable=False),
    "label": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    "effect_size": pa.Column(float, nullable=True),
    "effect_direction": pa.Column(str, pa.Check.isin(["up", "down", "none", "unknown"]), nullable=True),
    "pip": pa.Column(float, checks=[pa.Check.ge(0), pa.Check.le(1)], nullable=True),
    "pvalue": pa.Column(float, nullable=True),
    "source_dataset": pa.Column(str, nullable=False),
    "source_publication": pa.Column(str, nullable=True),
    "confidence": pa.Column(float, checks=[pa.Check.ge(0), pa.Check.le(1)], nullable=True),
    "training_overlap": pa.Column(str, pa.Check.isin(LEAKAGE_TYPES), nullable=True),
}, strict=True, coerce=True)


# Canonical pairs schema (aggregated)
canonical_pairs_schema = pa.DataFrameSchema({
    "variant_id": pa.Column(str, nullable=True),
    "element_id": pa.Column(str, nullable=True),
    "gene_id": pa.Column(str, nullable=False),
    "context_id": pa.Column(str, nullable=False),
    "n_evidence_sources": pa.Column(pl.Int64, pa.Check.ge(1), nullable=False),
    "evidence_sources": pa.Column(str, nullable=False),
    "max_confidence": pa.Column(float, nullable=True),
}, strict=True, coerce=True)
