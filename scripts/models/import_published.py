#!/usr/bin/env python3
"""Import published predictions (Family 1, 2 & 4).

Reads pre-computed E2G / V2G predictions from external tools and maps them
onto the canonical prediction schema. Covered models:

    Family 1 (classical E2G): abc, encode_re2g, sce2g_atac, sce2g_multiome,
                              epimap, graphreg
    Family 2 (single-cell):   pgboost, scent, signac, archr, cicero
    Family 4 (disease):       opentargets_l2g_2021, opentargets_l2g_current

E2G scores are lifted onto overlapping variants via e2g_to_v2g. Candidate
pairs a model did not cover receive coverage=0, ranking_score=0.

CLI (Snakemake interface)
-------------------------
    import_published.py
        --candidates <parquet> --gene-master <parquet>
        --synapse-dir <path> --zenodo-dir <path>
        --variants <parquet> --output <parquet>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from v2gbench.io.parquet import read_parquet, write_parquet
from v2gbench.models.published import (
    PublishedPredictionAdapter,
    IMPORTER_REGISTRY,
    handle_missing_prediction,
)
from v2gbench.models.opentargets_l2g import OpenTargetsL2GAdapter
from v2gbench.utils.config import load_config


def _find_root() -> Path:
    cur = Path.cwd()
    while cur != cur.parent:
        if (cur / "config" / "project.yaml").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def _discover_prediction_file(base_dir: Path, model_id: str, extensions=("parquet", "tsv", "tsv.gz", "csv")) -> Optional[Path]:
    """Search base_dir for a prediction file matching the model_id."""
    if not base_dir.exists():
        return None
    # Direct name match.
    for ext in extensions:
        candidate = base_dir / f"{model_id}.{ext}"
        if candidate.exists():
            return candidate
    # Subdirectory match.
    sub = base_dir / model_id
    if sub.is_dir():
        for ext in extensions:
            for hit in sub.glob(f"*.{ext}"):
                return hit
    # Loose glob match.
    for ext in extensions:
        for hit in base_dir.glob(f"*{model_id}*{ext}"):
            return hit
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import published predictions (ABC, rE2G, scE2G, EpiMap, GraphReg, pgBoost, SCENT, Signac, ArchR, Cicero, OpenTargets L2G)."
    )
    parser.add_argument("--candidates", required=True, help="Candidate pairs parquet path")
    parser.add_argument("--gene-master", required=True, help="GENCODE gene master parquet path")
    parser.add_argument("--synapse-dir", required=True, help="Synapse prediction bundle directory")
    parser.add_argument("--zenodo-dir", required=True, help="Zenodo prediction bundle directory")
    parser.add_argument("--variants", required=True, help="Variants parquet path")
    parser.add_argument("--output", required=True, help="Output predictions parquet path")
    args = parser.parse_args()

    root = _find_root()
    models_cfg = load_config(root / "config" / "models.yaml") if (root / "config" / "models.yaml").exists() else {}
    models = models_cfg.get("models", {})

    candidate_df = read_parquet(args.candidates)
    gene_master_df = read_parquet(args.gene_master)
    variants_df = read_parquet(args.variants)
    synapse_dir = Path(args.synapse_dir)
    zenodo_dir = Path(args.zenodo_dir)
    print(f"[import_published] candidates={candidate_df.height} variants={variants_df.height}")
    print(f"[import_published] synapse_dir={synapse_dir} zenodo_dir={zenodo_dir}")

    frames: List[pl.DataFrame] = []

    # --- Family 1 & 2: E2G published predictions ---
    e2g_model_ids = [
        "abc", "encode_re2g", "sce2g_atac", "sce2g_multiome",
        "epimap", "graphreg", "pgboost", "scent", "signac", "archr", "cicero",
    ]
    for model_id in e2g_model_ids:
        if model_id not in IMPORTER_REGISTRY:
            print(f"[import_published] SKIP {model_id}: no importer registered")
            continue
        # Search both bundle dirs for the prediction file.
        pred_path = _discover_prediction_file(synapse_dir, model_id) or _discover_prediction_file(zenodo_dir, model_id)
        if pred_path is None:
            print(f"[import_published] SKIP {model_id}: no prediction file found")
            # Emit missing-prediction rows so the candidate universe is covered.
            missing = handle_missing_prediction(candidate_df, model_id)
            frames.append(missing)
            continue

        cfg = dict(models.get(model_id, {}))
        cfg["input_path"] = str(pred_path)
        cfg.setdefault("model_family", "e2g")
        adapter = PublishedPredictionAdapter(model_id=model_id, config=cfg)
        if not adapter.validate_resources():
            print(f"[import_published] SKIP {model_id}: resource validation failed for {pred_path}")
            frames.append(handle_missing_prediction(candidate_df, model_id))
            continue

        print(f"[import_published] Importing {model_id} from {pred_path} ...")
        inputs: Dict[str, Any] = {
            "candidate_df": candidate_df,
            "variants_df": variants_df,
        }
        preds = adapter.score(inputs)
        qc = adapter.qc(preds)
        print(f"  {model_id}: rows={preds.height} qc={qc}")
        frames.append(preds)

    # --- Family 4: OpenTargets L2G (Track C only) ---
    for model_id in ("opentargets_l2g_2021", "opentargets_l2g_current"):
        cfg = dict(models.get(model_id, {}))
        version = cfg.get("version", "2021" if "2021" in model_id else "current")
        pred_path = _discover_prediction_file(synapse_dir, model_id) or _discover_prediction_file(zenodo_dir, model_id)
        if pred_path is None:
            print(f"[import_published] SKIP {model_id}: no prediction file found")
            frames.append(handle_missing_prediction(candidate_df, model_id))
            continue
        cfg["input_path"] = str(pred_path)
        cfg["model_id"] = model_id
        adapter = OpenTargetsL2GAdapter(config=cfg)
        if not adapter.validate_resources():
            print(f"[import_published] SKIP {model_id}: resource validation failed")
            frames.append(handle_missing_prediction(candidate_df, model_id))
            continue
        print(f"[import_published] Importing {model_id} from {pred_path} ...")
        preds = adapter.score({"candidate_df": candidate_df})
        qc = adapter.qc(preds)
        print(f"  {model_id}: rows={preds.height} qc={qc}")
        frames.append(preds)

    combined = pl.concat(frames, how="vertical_relaxed")
    write_parquet(combined, args.output)
    print(f"[import_published] Wrote {combined.height} prediction rows -> {args.output}")


if __name__ == "__main__":
    main()
