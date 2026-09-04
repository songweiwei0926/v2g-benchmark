"""End-to-end context mapping pipeline.

Builds a full mapping table from a list of source contexts to canonical
contexts, with confidence scores and mapping methods, then filters and
persists it. This orchestrates the per-context cascade defined in
:mod:`v2gbench.harmonize.contexts`.

All DataFrame operations use :mod:`polars`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import polars as pl
import yaml

from ..io.parquet import write_parquet
from .contexts import load_context_aliases, map_context

logger = logging.getLogger(__name__)

# Columns emitted by the mapping table.
_MAPPING_COLUMNS = [
    "source_context",
    "matched_context",
    "matched_ontology_id",
    "mapping_method",
    "mapping_level",
    "ontology_distance",
    "mapping_confidence",
]


def build_context_mapping_table(
    source_contexts: Iterable[str],
    config_path: str | Path,
    *,
    target_contexts: Optional[list[str]] = None,
) -> pl.DataFrame:
    """Build a table mapping all source contexts to canonical contexts.

    For each source context, runs
    :func:`v2gbench.harmonize.contexts.map_context` against the alias
    configuration loaded from ``config_path``. Unmapped contexts are retained
    with null match fields and ``mapping_confidence = 0.0``.

    Parameters
    ----------
    source_contexts
        Iterable of source context labels to map.
    config_path
        Path to the context-mapping YAML (see
        :func:`v2gbench.harmonize.contexts.load_context_aliases`).
    target_contexts
        Optional allow-list of canonical context names to restrict matching.

    Returns
    -------
    pl.DataFrame
        One row per source context with columns: ``source_context``,
        ``matched_context``, ``matched_ontology_id``, ``mapping_method``,
        ``mapping_level``, ``ontology_distance``, ``mapping_confidence``.
    """
    aliases = load_context_aliases(config_path)
    rows: list[dict[str, Any]] = []
    for source in source_contexts:
        source = str(source)
        result = map_context(source, target_contexts=target_contexts, aliases=aliases)
        if result is None:
            rows.append(
                {
                    "source_context": source,
                    "matched_context": None,
                    "matched_ontology_id": None,
                    "mapping_method": "unmapped",
                    "mapping_level": None,
                    "ontology_distance": None,
                    "mapping_confidence": 0.0,
                }
            )
        else:
            rows.append(
                {
                    "source_context": result.get("source_context", source),
                    "matched_context": result.get("matched_context"),
                    "matched_ontology_id": result.get("matched_ontology_id"),
                    "mapping_method": result.get("mapping_method"),
                    "mapping_level": result.get("mapping_level"),
                    "ontology_distance": result.get("ontology_distance"),
                    "mapping_confidence": result.get("mapping_confidence", 0.0),
                }
            )

    schema = {
        "source_context": pl.Utf8,
        "matched_context": pl.Utf8,
        "matched_ontology_id": pl.Utf8,
        "mapping_method": pl.Utf8,
        "mapping_level": pl.Int64,
        "ontology_distance": pl.Int64,
        "mapping_confidence": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def filter_primary_mapping(
    mapping_df: pl.DataFrame,
    min_confidence: float = 0.8,
) -> pl.DataFrame:
    """Filter the mapping table for the primary benchmark.

    Keeps only rows with ``mapping_confidence >= min_confidence`` and a
    non-null ``matched_context``. Rows flagged ``unmapped`` are dropped.

    Parameters
    ----------
    mapping_df
        Output of :func:`build_context_mapping_table`.
    min_confidence
        Minimum confidence threshold (default 0.8).

    Returns
    -------
    pl.DataFrame
        Filtered mapping table.
    """
    if mapping_df.is_empty():
        return mapping_df
    return mapping_df.filter(
        (pl.col("mapping_confidence") >= min_confidence)
        & pl.col("matched_context").is_not_null()
        & (pl.col("mapping_method") != "unmapped")
    )


def write_context_mapping(
    mapping_df: pl.DataFrame,
    output_path: str | Path,
) -> str:
    """Write a context mapping table to parquet (ZSTD-compressed).

    Parameters
    ----------
    mapping_df
        Mapping table from :func:`build_context_mapping_table` (or a filtered
        subset).
    output_path
        Destination parquet path. Parent directories are created.

    Returns
    -------
    str
        The path written.
    """
    write_parquet(mapping_df, output_path)
    logger.info("Wrote context mapping (%d rows) to %s", mapping_df.height, output_path)
    return str(output_path)


__all__ = [
    "build_context_mapping_table",
    "filter_primary_mapping",
    "write_context_mapping",
]
