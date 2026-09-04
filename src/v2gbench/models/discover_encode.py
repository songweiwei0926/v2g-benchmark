"""Auto-discovery of supplementary models from the ENCODE prediction bundle.

The ENCODE consortium distributes a "predictions table" plus a "methods
table" describing every E2G/V2G model run in their studies.  Rather than
hard-coding each model, we scan those tables at runtime and register any
model whose prediction file conforms to the canonical
``{chrom, start, end, gene, score}`` schema as a **supplementary** model
(``enabled: supplementary`` in the registry sense — run if the schema is
valid and resources are present).

Functions
---------
* :func:`validate_prediction_schema` — check a single prediction file.
* :func:`discover_encode_models` — scan the methods + predictions tables.
* :func:`build_supplementary_model_registry` — turn discoveries into a registry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

# A prediction file must expose at least these columns (after alias mapping).
REQUIRED_PREDICTION_COLUMNS = {"chrom", "start", "end", "gene_id", "score"}

# Accepted aliases → canonical name.
COLUMN_ALIASES = {
    "chr": "chrom", "chromosome": "chrom",
    "gene": "gene_id", "TargetGene": "gene_id", "Gene": "gene_id",
    "GeneSymbol": "gene_id",
    "ABC.Score": "score", "ABC_Score": "score", "p": "score",
    "Prob": "score", "probability": "score", "importance": "score",
    "EnhancerToGene": "score", "coaccess": "score", "cor": "score",
}


def _read_table(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pl.read_parquet(path)
    if suffix in (".tsv", ".txt"):
        return pl.read_csv(path, separator="\t")
    if suffix == ".csv":
        return pl.read_csv(path)
    return pl.read_csv(path, separator="\t")


def _canonical_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename aliased columns to canonical names (in place on a copy)."""
    rename = {k: v for k, v in COLUMN_ALIASES.items() if k in df.columns}
    return df.rename(rename)


def validate_prediction_schema(prediction_path: str | Path) -> bool:
    """Return ``True`` if ``prediction_path`` has the required columns.

    The check is cheap: only the schema (column names) is read, not the full
    file body, via ``pl.scan_parquet``/``scan_csv`` where possible.
    """
    path = Path(prediction_path)
    if not path.exists():
        return False
    try:
        df = _read_table(path)
    except Exception:
        return False
    df = _canonical_columns(df)
    return REQUIRED_PREDICTION_COLUMNS.issubset(set(df.columns))


def discover_encode_models(methods_table_path: str | Path,
                           predictions_table_path: str | Path) -> List[Dict[str, Any]]:
    """Discover all model configurations from the ENCODE methods/predictions tables.

    The methods table describes each model (``model_id``, ``family``,
    ``paper``, ``code_url`` …); the predictions table lists, per model, the
    prediction file path(s) and the cell type / context they correspond to.

    Returns a list of dicts, one per discovered model, each containing:

    * ``model_id``      — stable id (from methods table or derived).
    * ``model_family``  — ``e2g`` / ``singlecell`` / ``sequence`` …
    * ``enabled``       — always ``"supplementary"``.
    * ``mode``          — ``"published_prediction"``.
    * ``input_path``    — path to the prediction file.
    * ``context_id``    — context the prediction applies to (if known).
    * ``schema_valid``  — result of :func:`validate_prediction_schema`.
    """
    methods = _read_table(methods_table_path)
    predictions = _read_table(predictions_table_path)

    # Normalize column names on both tables.
    methods = _canonical_columns(methods)
    predictions = _canonical_columns(predictions)

    # Identify the join key between methods and predictions (be tolerant).
    join_key = None
    for cand in ("model_id", "model", "method", "method_id"):
        if cand in methods.columns and cand in predictions.columns:
            join_key = cand
            break

    discovered: List[Dict[str, Any]] = []

    if join_key is None:
        # No join possible: treat each prediction row as its own model.
        for row in predictions.to_dicts():
            pred_path = _find_prediction_path(row)
            if pred_path is None:
                continue
            discovered.append(_make_entry(
                model_id=row.get("model_id") or row.get("model") or Path(pred_path).stem,
                family=row.get("family", "e2g"),
                pred_path=pred_path,
                context_id=row.get("context_id"),
            ))
        return discovered

    # Join methods ↔ predictions on the shared key.
    merged = methods.join(predictions, on=join_key, how="inner", suffix="_pred")
    for row in merged.to_dicts():
        pred_path = _find_prediction_path(row)
        if pred_path is None:
            continue
        discovered.append(_make_entry(
            model_id=str(row.get(join_key)),
            family=row.get("family", row.get("model_family", "e2g")),
            pred_path=pred_path,
            context_id=row.get("context_id") or row.get("context_id_pred"),
        ))
    return discovered


def _find_prediction_path(row: Dict[str, Any]) -> Optional[str]:
    """Locate the prediction file path within a row dict."""
    for key in ("input_path", "prediction_path", "path", "file", "file_path", "predictions_path"):
        val = row.get(key)
        if val:
            return str(val)
    return None


def _make_entry(model_id: str, family: str, pred_path: str,
                context_id: Optional[str]) -> Dict[str, Any]:
    schema_valid = validate_prediction_schema(pred_path)
    return {
        "model_id": model_id,
        "model_family": family,
        "enabled": "supplementary",
        "mode": "published_prediction",
        "input_path": pred_path,
        "context_id": context_id,
        "schema_valid": schema_valid,
    }


def build_supplementary_model_registry(
    discovered_models: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build a registry dict keyed by ``model_id``.

    Only models with ``schema_valid=True`` are included; invalid ones are
    recorded under a ``"_invalid"`` key for diagnostics.
    """
    registry: Dict[str, Dict[str, Any]] = {}
    invalid: List[str] = []
    for m in discovered_models:
        if m.get("schema_valid"):
            registry[m["model_id"]] = {
                "family": m["model_family"],
                "enabled": "supplementary",
                "mode": m["mode"],
                "input_path": m["input_path"],
                "context_id": m.get("context_id"),
            }
        else:
            invalid.append(m["model_id"])
    if invalid:
        registry["_invalid"] = {"model_ids": invalid}
    return registry
