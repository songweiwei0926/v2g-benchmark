"""Benchmark building modules.

Constructs the gold evidence registry, candidate gene universe, training-
overlap (leakage) registry, model-context applicability matrix, and the
deterministic SEQ_CORE sampling subset that together define the benchmark
universe prior to model scoring.
"""

from .gold_registry import (
    EVIDENCE_COLUMNS,
    build_evidence_long,
    build_canonical_pairs,
    assign_eqtl_labels,
    assign_crispr_labels,
    deduplicate_gtex_eqtl,
)
from .candidate_sets import (
    build_candidate_set,
    build_all_candidate_sets,
    check_gold_coverage,
    compute_gold_distance_rank,
)
from .leakage import (
    build_leakage_registry,
    assign_leakage_type,
    filter_strict_no_leakage,
)
from .applicability import (
    build_applicability_matrix,
    check_model_applicability,
    compute_coverage,
)
from .seq_core import (
    STRATA_COLUMNS,
    build_seq_core,
    hash_variant_context,
    assign_strata,
)

__all__ = [
    # gold_registry
    "EVIDENCE_COLUMNS",
    "build_evidence_long",
    "build_canonical_pairs",
    "assign_eqtl_labels",
    "assign_crispr_labels",
    "deduplicate_gtex_eqtl",
    # candidate_sets
    "build_candidate_set",
    "build_all_candidate_sets",
    "check_gold_coverage",
    "compute_gold_distance_rank",
    # leakage
    "build_leakage_registry",
    "assign_leakage_type",
    "filter_strict_no_leakage",
    # applicability
    "build_applicability_matrix",
    "check_model_applicability",
    "compute_coverage",
    # seq_core
    "STRATA_COLUMNS",
    "build_seq_core",
    "hash_variant_context",
    "assign_strata",
]
