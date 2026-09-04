"""Table generation for V2G-Benchmark.

Produces the main results TSV, the supplementary ``SupplementaryTables.xlsx``
workbook (17 sheets), the model-exclusions TSV, and the mandatory-completion
matrix TSV. Uses polars for data handling and openpyxl (via pandas ExcelWriter)
for the xlsx workbook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl

from ._style import _to_pandas

__all__ = [
    "make_main_results_table",
    "make_supplementary_tables",
    "make_model_exclusions_table",
    "make_mandatory_completion_matrix",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _col(df, *candidates):
    pdf = _to_pandas(df)
    for c in candidates:
        if c in pdf.columns:
            return c
    return None


def _to_polars(df) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    return pl.from_pandas(_to_pandas(df))


# ---------------------------------------------------------------------------
# Main results table
# ---------------------------------------------------------------------------
def make_main_results_table(metrics_df, output_path: str) -> str:
    """Write the main results TSV.

    Output columns: ``Model, Family, MRR, Top1, Top3, Top5, AUPRC, Coverage,
    CI_low, CI_high``.

    Parameters
    ----------
    metrics_df : polars.DataFrame | pandas.DataFrame
        Overall metrics frame with one row per model. Recognised column
        aliases are mapped to the canonical output names; missing columns are
        filled with nulls.
    output_path : str
        Destination ``.tsv`` path.

    Returns
    -------
    str
        Absolute path of the written TSV.
    """
    df = _to_polars(metrics_df)
    aliases = {
        "Model": ["model", "model_id", "Model"],
        "Family": ["family", "model_family", "Family"],
        "MRR": ["MRR", "mrr", "mean_reciprocal_rank"],
        "Top1": ["Top1", "top1", "recall_at_1"],
        "Top3": ["Top3", "top3", "recall_at_3"],
        "Top5": ["Top5", "top5", "recall_at_5"],
        "AUPRC": ["AUPRC", "auprc", "avg_precision"],
        "Coverage": ["coverage", "Coverage", "frac_covered"],
        "CI_low": ["CI_low", "ci_low", "mrr_ci_low"],
        "CI_high": ["CI_high", "ci_high", "mrr_ci_high"],
    }
    cols = {}
    for out_name, cands in aliases.items():
        src = next((c for c in cands if c in df.columns), None)
        cols[out_name] = pl.col(src) if src else pl.lit(None)
    out = df.select(**cols)
    # sort by MRR descending
    if "MRR" in out.columns:
        out = out.sort("MRR", descending=True)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(str(p), separator="\t")
    return str(p.resolve())


# ---------------------------------------------------------------------------
# Model exclusions table
# ---------------------------------------------------------------------------
def make_model_exclusions_table(excluded_models, output_path: str) -> str:
    """Write the model-exclusions TSV.

    Output columns: ``model, reason, reference, code_available,
    weights_available, gene_specific, included``.

    Parameters
    ----------
    excluded_models : polars.DataFrame | pandas.DataFrame | list[dict]
        Frame or list of records describing excluded models.
    output_path : str
        Destination ``.tsv`` path.

    Returns
    -------
    str
        Absolute path of the written TSV.
    """
    if isinstance(excluded_models, (list, tuple)):
        df = pl.DataFrame(excluded_models)
    else:
        df = _to_polars(excluded_models)
    aliases = {
        "model": ["model", "model_id", "Model"],
        "reason": ["reason", "exclusion_reason"],
        "reference": ["reference", "citation", "url"],
        "code_available": ["code_available", "code"],
        "weights_available": ["weights_available", "weights"],
        "gene_specific": ["gene_specific", "gene_specificity"],
        "included": ["included", "included_in_benchmark"],
    }
    cols = {}
    for out_name, cands in aliases.items():
        src = next((c for c in cands if c in df.columns), None)
        cols[out_name] = pl.col(src) if src else pl.lit(None)
    out = df.select(**cols)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(str(p), separator="\t")
    return str(p.resolve())


# ---------------------------------------------------------------------------
# Mandatory completion matrix
# ---------------------------------------------------------------------------
def make_mandatory_completion_matrix(completion_data, output_path: str) -> str:
    """Write the mandatory-completion matrix TSV.

    Output columns: ``component, type, required, download, parse, harmonize,
    score, qc, benchmark, final_status``.

    Parameters
    ----------
    completion_data : polars.DataFrame | pandas.DataFrame | list[dict]
        Frame or list of records describing per-component completion status.
    output_path : str
        Destination ``.tsv`` path.

    Returns
    -------
    str
        Absolute path of the written TSV.
    """
    if isinstance(completion_data, (list, tuple)):
        df = pl.DataFrame(completion_data)
    else:
        df = _to_polars(completion_data)
    canonical = [
        "component", "type", "required", "download", "parse", "harmonize",
        "score", "qc", "benchmark", "final_status",
    ]
    cols = {}
    for name in canonical:
        src = next((c for c in (name, name.lower(), name.title()) if c in df.columns), None)
        cols[name] = pl.col(src) if src else pl.lit(None)
    out = df.select(**cols)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(str(p), separator="\t")
    return str(p.resolve())


# ---------------------------------------------------------------------------
# Supplementary tables workbook (17 sheets)
# ---------------------------------------------------------------------------
# Sheet name -> key expected in the all_data mapping (or a callable builder).
_SHEET_SPEC: list[tuple[str, str]] = [
    ("S1_DatasetRegistry", "dataset_registry"),
    ("S2_GoldStandardSummary", "gold_standard_summary"),
    ("S3_ContextMapping", "context_mapping"),
    ("S4_DatasetOverlaps", "dataset_overlaps"),
    ("S5_ModelRegistry", "model_registry"),
    ("S6_ModelVersions", "model_versions"),
    ("S7_ModelApplicability", "model_applicability"),
    ("S8_TrainingLeakage", "training_leakage"),
    ("S9_AllModelConfigurations", "all_model_configurations"),
    ("S10_OverallMetrics", "overall_metrics"),
    ("S11_StratifiedMetrics", "stratified_metrics"),
    ("S12_PairwiseBootstrap", "pairwise_bootstrap"),
    ("S13_FailureModeLoci", "failure_mode_loci"),
    ("S14_SequenceModelScores", "sequence_model_scores"),
    ("S15_IntegratedFeatureImportance", "integrated_feature_importance"),
    ("S16_QCReport", "qc_report"),
    ("S17_ExcludedModels", "excluded_models"),
]


def make_supplementary_tables(all_data: Mapping[str, Any], output_path: str) -> str:
    """Generate ``SupplementaryTables.xlsx`` with 17 sheets.

    Parameters
    ----------
    all_data : Mapping[str, Any]
        Mapping from logical key (e.g. ``"dataset_registry"``) to a
        polars/pandas DataFrame or list of records. Keys not present produce
        an empty placeholder sheet so the workbook always has all 17 sheets.
    output_path : str
        Destination ``.xlsx`` path.

    Returns
    -------
    str
        Absolute path of the written xlsx.
    """
    import pandas as pd

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    sheets: dict[str, "pd.DataFrame"] = {}
    for sheet_name, key in _SHEET_SPEC:
        frame = all_data.get(key)
        if frame is None:
            sheets[sheet_name] = pd.DataFrame()
        elif isinstance(frame, pl.DataFrame):
            sheets[sheet_name] = frame.to_pandas()
        elif isinstance(frame, pd.DataFrame):
            sheets[sheet_name] = frame
        elif isinstance(frame, (list, tuple)):
            sheets[sheet_name] = pd.DataFrame(list(frame))
        else:
            sheets[sheet_name] = pd.DataFrame()

    with pd.ExcelWriter(str(p), engine="openpyxl") as writer:
        for sheet_name, pdf in sheets.items():
            # openpyxl sheet name limit is 31 chars
            name = sheet_name[:31]
            pdf.to_excel(writer, sheet_name=name, index=False)
            # light formatting: bold header row, auto-ish column width
            ws = writer.sheets[name]
            from openpyxl.styles import Font

            for cell in ws[1]:
                cell.font = Font(bold=True)
            for col in ws.columns:
                length = max(
                    (len(str(cell.value)) for cell in col if cell.value is not None),
                    default=8,
                )
                ws.column_dimensions[col[0].column_letter].width = min(max(length + 2, 10), 60)

    return str(p.resolve())
