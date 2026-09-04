#!/usr/bin/env python3
"""Score the 7 baseline (Family 0) models.

Baselines are training-free and derive their ranking_score directly from the
candidate table (distance_to_tss, is_expressed). The seven baselines are:

    random, nearest_tss, inverse_distance,
    exp_distance_50kb, exp_distance_100kb, exp_distance_250kb,
    nearest_expressed

CLI (Snakemake interface)
-------------------------
    score_baselines.py
        --candidates <parquet> --gene-master <parquet> --output <parquet>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import polars as pl

from v2gbench.io.parquet import read_parquet, write_parquet
from v2gbench.models.baselines import (
    RandomAdapter,
    NearestTSSAdapter,
    InverseDistanceAdapter,
    ExponentialDistanceAdapter,
    NearestExpressedAdapter,
)
from v2gbench.utils.config import load_config


def _find_root() -> Path:
    cur = Path.cwd()
    while cur != cur.parent:
        if (cur / "config" / "project.yaml").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def _build_baseline_adapters(models_cfg: Dict[str, Any], seed: int) -> List:
    """Instantiate the 7 baseline adapters from the model registry."""
    models = models_cfg.get("models", {}) if models_cfg else {}
    adapters = []

    # random
    rand_cfg = models.get("random", {})
    adapters.append(RandomAdapter(config={**rand_cfg, "seed": seed}))

    # nearest_tss
    adapters.append(NearestTSSAdapter(config=models.get("nearest_tss", {})))

    # inverse_distance
    adapters.append(InverseDistanceAdapter(config=models.get("inverse_distance", {})))

    # exp_distance variants
    for mid in ("exp_distance_50kb", "exp_distance_100kb", "exp_distance_250kb"):
        cfg = dict(models.get(mid, {}))
        # Ensure lambda is set from the registry or derived from the model_id.
        if "lambda" not in cfg:
            kb = int(mid.replace("exp_distance_", "").replace("kb", ""))
            cfg["lambda"] = kb * 1000
        cfg["model_id"] = mid
        adapters.append(ExponentialDistanceAdapter(config=cfg))

    # nearest_expressed
    adapters.append(NearestExpressedAdapter(config=models.get("nearest_expressed", {})))

    return adapters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score 7 baseline models (random, nearest_tss, inverse_distance, exp_distance_*, nearest_expressed)."
    )
    parser.add_argument("--candidates", required=True, help="Candidate pairs parquet path")
    parser.add_argument("--gene-master", required=True, help="GENCODE gene master parquet path")
    parser.add_argument("--output", required=True, help="Output predictions parquet path")
    args = parser.parse_args()

    root = _find_root()
    project_cfg = load_config(root / "config" / "project.yaml") if (root / "config" / "project.yaml").exists() else {}
    seed = int(project_cfg.get("project", {}).get("seed", 20260904))
    models_cfg = load_config(root / "config" / "models.yaml") if (root / "config" / "models.yaml").exists() else {}

    candidate_df = read_parquet(args.candidates)
    gene_master_df = read_parquet(args.gene_master)
    print(f"[score_baselines] candidates={candidate_df.height} genes={gene_master_df.height}")

    adapters = _build_baseline_adapters(models_cfg, seed)
    print(f"[score_baselines] {len(adapters)} baseline adapters: {[a.model_id for a in adapters]}")

    frames: List[pl.DataFrame] = []
    for adapter in adapters:
        print(f"[score_baselines] Scoring {adapter.model_id} ...")
        inputs: Dict[str, Any] = {
            "candidate_df": candidate_df,
            "gene_master_df": gene_master_df,
        }
        preds = adapter.score(inputs)
        qc = adapter.qc(preds)
        print(f"  {adapter.model_id}: rows={preds.height} qc={qc}")
        frames.append(preds)

    combined = pl.concat(frames, how="vertical_relaxed")
    write_parquet(combined, args.output)
    print(f"[score_baselines] Wrote {combined.height} prediction rows -> {args.output}")


if __name__ == "__main__":
    main()
