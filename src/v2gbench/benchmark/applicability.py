"""Model-context applicability matrix.

Not every model can produce a score for every (variant/element, gene,
context) triple. A sequence model trained on RNA tracks cannot score a
context for which it has no matching output track; a published E2G model may
only cover a subset of cell types. This module encodes those constraints as
an explicit *applicability matrix* (model x context) and provides helpers to
compute prediction coverage against a candidate universe.

Applicability statuses follow
:data:`v2gbench.schemas.prediction.APPLICABILITY_STATUS`:

* ``APPLICABLE``
* ``NOT_APPLICABLE_CONTEXT``
* ``NOT_APPLICABLE_VARIANT``
* ``NOT_APPLICABLE_ELEMENT``
* ``NOT_APPLICABLE_MISSING_DATA``

All functions use :mod:`polars` (never pandas) and read model config from
YAML via :mod:`yaml`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import polars as pl
import yaml

from ..io.parquet import read_parquet, write_parquet
from ..schemas.prediction import APPLICABILITY_STATUS

PathLike = Union[str, Path]


def _load_models_config(
    models_config: Union[PathLike, Dict[str, Any]],
) -> Dict[str, Any]:
    """Accept either a path to a YAML file or an in-memory config dict."""
    if isinstance(models_config, (str, Path)):
        with open(models_config) as f:
            return yaml.safe_load(f)
    return models_config


def _model_context_constraints(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract context/variant/element applicability constraints.

    Recognised config keys (under ``applicability`` sub-dict or top level):

    * ``applicable_contexts``   -- allow-list of context ids (None = all).
    * ``excluded_contexts``     -- deny-list of context ids.
    * ``applicable_context_types`` -- allow-list of context types.
    * ``applicable_variants``   -- allow-list of variant ids (None = all).
    * ``applicable_elements``   -- allow-list of element ids (None = all).
    * ``requires_context_data`` -- bool; if True, missing context -> NOT_APPLICABLE.
    """
    appl = model_cfg.get("applicability", {}) if isinstance(model_cfg, dict) else {}
    if not isinstance(appl, dict):
        appl = {}

    def _get(key: str, top_key: str) -> Any:
        if key in appl:
            return appl[key]
        if isinstance(model_cfg, dict) and top_key in model_cfg:
            return model_cfg[top_key]
        return None

    return {
        "applicable_contexts": _get("contexts", "applicable_contexts"),
        "excluded_contexts": _get("excluded_contexts", "excluded_contexts"),
        "applicable_context_types": _get(
            "context_types", "applicable_context_types"
        ),
        "applicable_variants": _get("variants", "applicable_variants"),
        "applicable_elements": _get("elements", "applicable_elements"),
        "requires_context_data": _get(
            "requires_context_data", "requires_context_data"
        ),
    }


def check_model_applicability(
    model_id: str,
    context_id: str,
    models_config: Union[PathLike, Dict[str, Any]],
    context_type: Optional[str] = None,
) -> str:
    """Check whether a model can predict for a given context.

    Returns one of :data:`APPLICABILITY_STATUS`. The check is context-centric
    (variant/element allow-lists are evaluated elsewhere when a specific pair
    is scored); here we only resolve context-level applicability, so the
    possible returns are ``APPLICABLE``, ``NOT_APPLICABLE_CONTEXT``, or
    ``NOT_APPLICABLE_MISSING_DATA``.

    Parameters
    ----------
    model_id
        Model identifier.
    context_id
        Context identifier to test.
    models_config
        Path to ``models.yaml`` or in-memory config dict.
    context_type
        Optional context type (e.g. ``"cell_line"``) for type-level checks.

    Returns
    -------
    str
        Applicability status.
    """
    cfg = _load_models_config(models_config)
    models = cfg.get("models", cfg) if isinstance(cfg, dict) else {}

    model_cfg = models.get(model_id)
    if not isinstance(model_cfg, dict):
        return "NOT_APPLICABLE_MISSING_DATA"

    c = _model_context_constraints(model_cfg)

    def _as_list(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v]
        return [str(v)]

    excluded = _as_list(c["excluded_contexts"])
    if context_id in excluded:
        return "NOT_APPLICABLE_CONTEXT"

    allowed = _as_list(c["applicable_contexts"])
    if allowed and context_id not in allowed:
        return "NOT_APPLICABLE_CONTEXT"

    if context_type is not None:
        allowed_types = _as_list(c["applicable_context_types"])
        if allowed_types and context_type not in allowed_types:
            return "NOT_APPLICABLE_CONTEXT"

    if c.get("requires_context_data") and not allowed and not context_type:
        # Model declares it needs context data but none is mapped.
        return "NOT_APPLICABLE_MISSING_DATA"

    return "APPLICABLE"


def build_applicability_matrix(
    models_config: Union[PathLike, Dict[str, Any]],
    contexts: Union[pl.DataFrame, PathLike, Sequence[str], Sequence[Dict[str, Any]]],
    output_path: Optional[PathLike] = None,
) -> pl.DataFrame:
    """Build the model x context applicability matrix.

    Parameters
    ----------
    models_config
        Path to ``models.yaml`` or in-memory config dict.
    contexts
        Either a contexts frame (with ``context_id`` and optional
        ``context_type``), a path to its Parquet file, a list of context ids,
        or a list of context dicts.
    output_path
        Optional destination Parquet path.

    Returns
    -------
    pl.DataFrame
        Long-format matrix with columns
        ``model_id, context_id, applicability``.
    """
    cfg = _load_models_config(models_config)
    models = cfg.get("models", cfg) if isinstance(cfg, dict) else {}

    # Normalise contexts into a frame with context_id + context_type.
    if isinstance(contexts, (str, Path)):
        contexts_df = read_parquet(contexts)
    elif isinstance(contexts, pl.DataFrame):
        contexts_df = contexts
    elif isinstance(contexts, Sequence) and len(contexts) > 0 and isinstance(
        contexts[0], dict
    ):
        contexts_df = pl.DataFrame(list(contexts))
    elif isinstance(contexts, Sequence):
        contexts_df = pl.DataFrame(
            {"context_id": list(contexts)}
        )
    else:
        contexts_df = pl.DataFrame({"context_id": []})

    if "context_id" not in contexts_df.columns:
        raise ValueError("contexts must expose a context_id column.")

    context_ids = contexts_df["context_id"].to_list()
    context_types = (
        contexts_df["context_type"].to_list()
        if "context_type" in contexts_df.columns
        else [None] * len(context_ids)
    )

    rows: List[Dict[str, Any]] = []
    for model_id, model_cfg in models.items():
        if not isinstance(model_cfg, dict):
            continue
        for ctx_id, ctx_type in zip(context_ids, context_types):
            status = check_model_applicability(
                model_id, ctx_id, cfg, context_type=ctx_type
            )
            rows.append(
                {
                    "model_id": model_id,
                    "context_id": ctx_id,
                    "applicability": status,
                }
            )

    matrix = pl.DataFrame(
        rows,
        schema={
            "model_id": pl.Utf8,
            "context_id": pl.Utf8,
            "applicability": pl.Utf8,
        },
    )
    if output_path is not None:
        write_parquet(matrix, output_path)
    return matrix


def compute_coverage(
    predictions_df: pl.DataFrame,
    candidate_df: pl.DataFrame,
    model_id: Optional[str] = None,
) -> float:
    """Compute prediction coverage over the applicable candidate universe.

    Coverage = scored applicable pairs / all applicable pairs, where
    "applicable" means the candidate pair's (variant/element, gene, context)
    is within the model's scope and "scored" means a prediction row exists
    with a non-null ``ranking_score`` and ``coverage == 1``.

    Parameters
    ----------
    predictions_df
        Predictions frame conforming to ``prediction_schema``.
    candidate_df
        Candidate universe frame.
    model_id
        Optional model id to restrict both sides to a single model.

    Returns
    -------
    float
        Coverage fraction in ``[0, 1]``. Returns ``1.0`` when the applicable
        universe is empty (nothing to cover).
    """
    if predictions_df.height == 0 or candidate_df.height == 0:
        return 0.0

    preds = predictions_df
    if model_id is not None and "model_id" in preds.columns:
        preds = preds.filter(pl.col("model_id") == model_id)

    # Scored = prediction exists with coverage flag == 1 (and a real score).
    score_cols = [c for c in ("ranking_score", "coverage") if c in preds.columns]
    if "coverage" in preds.columns:
        scored = preds.filter(pl.col("coverage") == 1)
    else:
        scored = preds.filter(pl.col("ranking_score").is_not_null())

    # Join keys: prefer variant_id, fall back to element_id.
    key_cols: List[str] = []
    for k in ("variant_id", "element_id", "gene_id", "context_id"):
        if k in scored.columns and k in candidate_df.columns:
            key_cols.append(k)
    if len(key_cols) < 2:
        raise ValueError(
            "Cannot compute coverage: predictions and candidates share < 2 keys."
        )

    scored_keys = scored.select(key_cols).unique()
    cand_keys = candidate_df.select(key_cols).unique()

    all_applicable = cand_keys.height
    if all_applicable == 0:
        return 1.0

    scored_applicable = scored_keys.join(cand_keys, on=key_cols, how="inner").height
    return scored_applicable / all_applicable


__all__ = [
    "build_applicability_matrix",
    "check_model_applicability",
    "compute_coverage",
]
