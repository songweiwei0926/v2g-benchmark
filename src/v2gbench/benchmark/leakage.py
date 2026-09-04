"""Training-overlap (leakage) registry.

A model is only a fair *zero-shot* benchmark participant if its training
data did not see the exact (variant, gene, context) pair, the cell type, the
assay track, or -- for the strictest comparison -- the dataset under
evaluation. This module builds a per-model registry of training provenance
and assigns each evaluated pair a leakage category from
:data:`v2gbench.schemas.evidence.LEAKAGE_TYPES`:

* ``PAIR_LABEL_SEEN``    -- the exact pair's label was in training (strictest).
* ``CELLTYPE_SEEN``      -- the context/cell type was in training.
* ``ASSAY_TRACK_SEEN``   -- the assay/track type was in training.
* ``DATASET_SEEN``       -- the source dataset was in training.
* ``NO_KNOWN_OVERLAP``   -- no known overlap with training.
* ``UNKNOWN``            -- training provenance not declared.

All functions use :mod:`polars` (never pandas) and read model provenance
from YAML via :mod:`yaml`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import polars as pl
import yaml

from ..io.parquet import read_parquet, write_parquet
from ..schemas.evidence import LEAKAGE_TYPES

PathLike = Union[str, Path]


def _load_models_config(
    models_config: Union[PathLike, Dict[str, Any]],
) -> Dict[str, Any]:
    """Accept either a path to a YAML file or an in-memory config dict."""
    if isinstance(models_config, (str, Path)):
        with open(models_config) as f:
            return yaml.safe_load(f)
    return models_config


def _model_training_fields(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training-provenance fields from a single model's config.

    Looks for a ``training`` sub-dict first, then falls back to top-level
    ``training_*`` keys, then to safe defaults.
    """
    training = model_cfg.get("training", {}) if isinstance(model_cfg, dict) else {}
    if not isinstance(training, dict):
        training = {}

    def _get(key: str, top_key: str) -> Any:
        if key in training:
            return training[key]
        if isinstance(model_cfg, dict) and top_key in model_cfg:
            return model_cfg[top_key]
        return None

    return {
        "training_dataset": _get("dataset", "training_dataset"),
        "training_celltype": _get("celltype", "training_celltype"),
        "training_label_type": _get("label_type", "training_label_type"),
        "training_assay": _get("assay", "training_assay"),
        "training_variants": _get("variants", "training_variants"),
        "training_genes": _get("genes", "training_genes"),
        "training_pairs": _get("pairs", "training_pairs"),
    }


def build_leakage_registry(
    models_config: Union[PathLike, Dict[str, Any]],
    output_path: Optional[PathLike] = None,
) -> pl.DataFrame:
    """Build a per-model training-provenance registry.

    For each model declared in ``models_config`` (under the ``models`` key,
    or at top level if the config *is* the models mapping), record:

    * ``model_id``
    * ``family``
    * ``mode``
    * ``training_dataset``
    * ``training_celltype``
    * ``training_label_type``
    * ``training_assay``
    * ``benchmark_overlap`` -- a coarse flag summarising whether the model
      declares *any* overlap with the benchmark datasets/cell types. This is
      a conservative ``UNKNOWN`` when provenance is missing.

    Parameters
    ----------
    models_config
        Path to ``models.yaml`` or an in-memory config dict.
    output_path
        Optional destination Parquet path.

    Returns
    -------
    pl.DataFrame
        Leakage registry frame.
    """
    cfg = _load_models_config(models_config)
    models = cfg.get("models", cfg) if isinstance(cfg, dict) else {}

    rows: List[Dict[str, Any]] = []
    for model_id, model_cfg in models.items():
        if not isinstance(model_cfg, dict):
            continue
        fields = _model_training_fields(model_cfg)

        # Coarse overlap flag: any declared training field that is non-null.
        declared = any(v is not None for v in fields.values())
        benchmark_overlap = "DECLARED" if declared else "UNKNOWN"

        rows.append(
            {
                "model_id": model_id,
                "family": model_cfg.get("family"),
                "mode": model_cfg.get("mode"),
                "training_dataset": fields["training_dataset"],
                "training_celltype": fields["training_celltype"],
                "training_label_type": fields["training_label_type"],
                "training_assay": fields["training_assay"],
                "training_variants": fields["training_variants"],
                "training_genes": fields["training_genes"],
                "training_pairs": fields["training_pairs"],
                "benchmark_overlap": benchmark_overlap,
            }
        )

    registry = pl.DataFrame(rows)
    if output_path is not None:
        write_parquet(registry, output_path)
    return registry


def _as_list(value: Any) -> List[str]:
    """Normalise a training-field value into a list of lowercased strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).lower() for v in value]
    return [str(value).lower()]


def assign_leakage_type(
    model_id: str,
    variant_id: Optional[str],
    gene_id: Optional[str],
    context_id: Optional[str],
    leakage_registry: Union[pl.DataFrame, PathLike],
) -> str:
    """Assign a single leakage category to one (model, pair, context) tuple.

    The category is the *most specific* overlap that applies, checked in
    priority order:

    1. ``PAIR_LABEL_SEEN``    -- (variant, gene) [and context] in training pairs.
    2. ``CELLTYPE_SEEN``      -- context in training cell types.
    3. ``ASSAY_TRACK_SEEN``   -- assay/track in training assays.
    4. ``DATASET_SEEN``       -- source dataset in training datasets.
    5. ``NO_KNOWN_OVERLAP``   -- provenance declared, no overlap found.
    6. ``UNKNOWN``            -- model not in registry or provenance missing.

    Parameters
    ----------
    model_id
        Model identifier.
    variant_id, gene_id, context_id
        The evaluated pair / context (any may be None).
    leakage_registry
        Registry frame (in-memory or Parquet path).

    Returns
    -------
    str
        One of :data:`LEAKAGE_TYPES`.
    """
    if isinstance(leakage_registry, (str, Path)):
        leakage_registry = read_parquet(leakage_registry)

    if "model_id" not in leakage_registry.columns:
        return "UNKNOWN"

    model_rows = leakage_registry.filter(pl.col("model_id") == model_id)
    if model_rows.height == 0:
        return "UNKNOWN"

    row = model_rows.row(0, named=True)

    # 1. Pair-level label seen.
    training_pairs = _as_list(row.get("training_pairs"))
    if training_pairs and variant_id is not None and gene_id is not None:
        pair_key = f"{variant_id}|{gene_id}".lower()
        pair_key_ctx = f"{variant_id}|{gene_id}|{context_id}".lower()
        if pair_key_ctx in training_pairs or pair_key in training_pairs:
            return "PAIR_LABEL_SEEN"

    # Also check explicit variant/gene training lists.
    training_variants = _as_list(row.get("training_variants"))
    training_genes = _as_list(row.get("training_genes"))
    if training_variants and training_genes:
        if (
            variant_id is not None
            and gene_id is not None
            and variant_id.lower() in training_variants
            and gene_id.lower() in training_genes
        ):
            return "PAIR_LABEL_SEEN"

    # 2. Cell type seen.
    training_celltypes = _as_list(row.get("training_celltype"))
    if training_celltypes and context_id is not None:
        if context_id.lower() in training_celltypes:
            return "CELLTYPE_SEEN"

    # 3. Assay track seen.
    training_assays = _as_list(row.get("training_assay"))
    if training_assays:
        return "ASSAY_TRACK_SEEN"

    # 4. Dataset seen.
    training_datasets = _as_list(row.get("training_dataset"))
    if training_datasets:
        return "DATASET_SEEN"

    # 5. Provenance declared but no overlap.
    declared = any(
        row.get(k) is not None
        for k in (
            "training_dataset",
            "training_celltype",
            "training_label_type",
            "training_assay",
            "training_pairs",
            "training_variants",
            "training_genes",
        )
    )
    return "NO_KNOWN_OVERLAP" if declared else "UNKNOWN"


def filter_strict_no_leakage(
    evidence_df: pl.DataFrame,
    leakage_registry: Union[pl.DataFrame, PathLike],
    model_id: Optional[str] = None,
) -> pl.DataFrame:
    """Filter evidence to pairs with no strict pair-label leakage.

    "Strict" here means the exact (variant, gene[, context]) pair's label was
    *not* seen in training (``PAIR_LABEL_SEEN`` is excluded). Cell-type /
    assay / dataset overlap is *not* filtered out by this function -- those
    are softer overlaps handled by subset definitions elsewhere.

    Parameters
    ----------
    evidence_df
        Evidence-long frame. If a ``training_overlap`` column already exists
        it is used directly; otherwise each row is classified via
        :func:`assign_leakage_type` (requires ``model_id`` when multiple
        models are in scope, or applies the single-model registry).
    leakage_registry
        Registry frame (in-memory or Parquet path).
    model_id
        Model id to classify rows for, when ``training_overlap`` is absent.

    Returns
    -------
    pl.DataFrame
        Subset of ``evidence_df`` with ``PAIR_LABEL_SEEN`` rows removed.
    """
    if evidence_df.height == 0:
        return evidence_df

    if isinstance(leakage_registry, (str, Path)):
        leakage_registry = read_parquet(leakage_registry)

    if "training_overlap" in evidence_df.columns:
        return evidence_df.filter(pl.col("training_overlap") != "PAIR_LABEL_SEEN")

    if model_id is None:
        # If registry holds a single model, use it; otherwise we cannot classify.
        model_ids = (
            leakage_registry["model_id"].unique().to_list()
            if "model_id" in leakage_registry.columns
            else []
        )
        if len(model_ids) == 1:
            model_id = model_ids[0]
        else:
            raise ValueError(
                "model_id is required to classify leakage when "
                "training_overlap is absent and the registry has != 1 model."
            )

    labels = [
        assign_leakage_type(
            model_id,
            row.get("variant_id"),
            row.get("gene_id"),
            row.get("context_id"),
            leakage_registry,
        )
        for row in evidence_df.iter_rows(named=True)
    ]
    evidence_df = evidence_df.with_columns(
        pl.Series("training_overlap", labels, dtype=pl.Utf8)
    )
    return evidence_df.filter(pl.col("training_overlap") != "PAIR_LABEL_SEEN")


__all__ = [
    "build_leakage_registry",
    "assign_leakage_type",
    "filter_strict_no_leakage",
]
