"""Harmonization package — normalize variants, genes, and contexts to canonical forms.

This package provides utilities to bring heterogeneous input data onto the
canonical schemas defined in :mod:`v2gbench.schemas`:

* :mod:`v2gbench.harmonize.variants`  — bcftools-based variant normalization
  and canonical ``variant_id`` generation.
* :mod:`v2gbench.harmonize.genes`     — GENCODE GTF parsing and gene master
  table construction.
* :mod:`v2gbench.harmonize.contexts`  — multi-level cell-type / tissue context
  ontology matching.
* :mod:`v2gbench.harmonize.map_contexts` — end-to-end context mapping pipeline.
"""

from .variants import (
    normalize_variants,
    generate_variant_id,
    check_ref_match,
    assign_qc_status,
    normalize_variant_df,
)
from .genes import (
    parse_gencode_gtf,
    deversion_gene_ids,
    build_gene_master_table,
    extract_exon_intervals,
)
from .contexts import (
    normalize_context,
    load_context_aliases,
    map_context,
    map_context_level1,
    map_context_level2,
    map_context_level3,
    map_context_level4,
    map_context_level5,
    map_context_level6,
)
from .map_contexts import (
    build_context_mapping_table,
    filter_primary_mapping,
    write_context_mapping,
)

__all__ = [
    # variants
    "normalize_variants",
    "generate_variant_id",
    "check_ref_match",
    "assign_qc_status",
    "normalize_variant_df",
    # genes
    "parse_gencode_gtf",
    "deversion_gene_ids",
    "build_gene_master_table",
    "extract_exon_intervals",
    # contexts
    "normalize_context",
    "load_context_aliases",
    "map_context",
    "map_context_level1",
    "map_context_level2",
    "map_context_level3",
    "map_context_level4",
    "map_context_level5",
    "map_context_level6",
    # map_contexts
    "build_context_mapping_table",
    "filter_primary_mapping",
    "write_context_mapping",
]
