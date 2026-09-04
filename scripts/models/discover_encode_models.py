#!/usr/bin/env python3
"""Auto-discover ENCODE models from the Synapse prediction bundle.

Scans the ENCODE methods + predictions tables at runtime and registers any
model whose prediction file conforms to the canonical
{chrom, start, end, gene_id, score} schema as a supplementary model. Only
schema-valid models are included in the output registry; invalid ones are
recorded for diagnostics.

CLI (Snakemake interface)
-------------------------
    discover_encode_models.py
        --synapse-dir <path> --candidates <parquet>
        --output <parquet> --configurations-output <tsv>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import polars as pl

from v2gbench.io.parquet import read_parquet, write_parquet, write_tsv
from v2gbench.models.discover_encode import (
    discover_encode_models,
    build_supplementary_model_registry,
    validate_prediction_schema,
)
from v2gbench.models.published import PublishedPredictionAdapter, handle_missing_prediction
from v2gbench.utils.config import load_config


def _find_methods_and_predictions(synapse_dir: Path):
    """Locate the methods and predictions tables within the Synapse bundle."""
    methods_path = None
    predictions_path = None

    # Common file names.
    for name in ("methods.tsv", "methods_table.tsv", "methods.csv", "methods.parquet"):
        cand = synapse_dir / name
        if cand.exists():
            methods_path = cand
            break
    for name in ("predictions.tsv", "predictions_table.tsv", "predictions.csv", "predictions.parquet"):
        cand = synapse_dir / name
        if cand.exists():
            predictions_path = cand
            break

    # Fallback: glob for files containing 'method' / 'prediction'.
    if methods_path is None:
        for hit in synapse_dir.rglob("*method*"):
            if hit.is_file() and hit.suffix in (".tsv", ".csv", ".parquet", ".txt"):
                methods_path = hit
                break
    if predictions_path is None:
        for hit in synapse_dir.rglob("*prediction*"):
            if hit.is_file() and hit.suffix in (".tsv", ".csv", ".parquet", ".txt"):
                predictions_path = hit
                break

    return methods_path, predictions_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-discover ENCODE models from the Synapse prediction bundle."
    )
    parser.add_argument("--synapse-dir", required=True, help="Synapse prediction bundle directory")
    parser.add_argument("--candidates", required=True, help="Candidate pairs parquet path")
    parser.add_argument("--output", required=True, help="Output predictions parquet path")
    parser.add_argument("--configurations-output", required=True, help="Output model configurations TSV path")
    args = parser.parse_args()

    synapse_dir = Path(args.synapse_dir)
    candidate_df = read_parquet(args.candidates)
    print(f"[discover_encode_models] synapse_dir={synapse_dir} candidates={candidate_df.height}")

    methods_path, predictions_path = _find_methods_and_predictions(synapse_dir)

    discovered: List[Dict[str, Any]] = []
    if methods_path is not None and predictions_path is not None:
        print(f"[discover_encode_models] methods={methods_path} predictions={predictions_path}")
        discovered = discover_encode_models(methods_path, predictions_path)
    else:
        print("[discover_encode_models] No methods/predictions tables found; scanning directory for prediction files ...")
        # Fallback: treat each prediction file in the bundle as a candidate model.
        for hit in sorted(synapse_dir.rglob("*")):
            if hit.is_file() and hit.suffix in (".tsv", ".csv", ".parquet"):
                if validate_prediction_schema(hit):
                    discovered.append({
                        "model_id": hit.stem,
                        "model_family": "e2g",
                        "enabled": "supplementary",
                        "mode": "published_prediction",
                        "input_path": str(hit),
                        "context_id": None,
                        "schema_valid": True,
                    })

    print(f"[discover_encode_models] Discovered {len(discovered)} model configurations")
    valid = [m for m in discovered if m.get("schema_valid")]
    invalid = [m for m in discovered if not m.get("schema_valid")]
    print(f"  schema_valid={len(valid)} schema_invalid={len(invalid)}")

    # Write configurations TSV.
    config_df = pl.DataFrame(discovered) if discovered else pl.DataFrame(schema={
        "model_id": pl.Utf8, "model_family": pl.Utf8, "enabled": pl.Utf8,
        "mode": pl.Utf8, "input_path": pl.Utf8, "context_id": pl.Utf8,
        "schema_valid": pl.Boolean,
    })
    write_tsv(config_df, args.configurations_output)
    print(f"[discover_encode_models] Wrote configurations -> {args.configurations_output}")

    # Build registry and score each valid model.
    registry = build_supplementary_model_registry(discovered)
    frames: List[pl.DataFrame] = []

    for model_id, cfg in registry.items():
        if model_id == "_invalid":
            continue
        input_path = cfg.get("input_path")
        if not input_path or not Path(input_path).exists():
            print(f"[discover_encode_models] SKIP {model_id}: input_path missing")
            frames.append(handle_missing_prediction(candidate_df, model_id))
            continue

        adapter_cfg = {
            "input_path": input_path,
            "model_family": cfg.get("family", "e2g"),
            "context_id": cfg.get("context_id"),
        }
        try:
            adapter = PublishedPredictionAdapter(model_id=model_id, config=adapter_cfg)
            # PublishedPredictionAdapter needs variants_df for e2g_to_v2g.
            # Derive a minimal variants frame from the candidate set.
            from v2gbench.schemas.variant import parse_variant_id
            if all(c in candidate_df.columns for c in ("chrom", "pos")):
                variants_df = candidate_df.select(["variant_id", "chrom", "pos"]).unique()
            elif "variant_id" in candidate_df.columns:
                variants_df = candidate_df.select("variant_id").unique().with_columns(
                    pl.col("variant_id").map_elements(lambda v: parse_variant_id(v)["chrom"], return_dtype=pl.Utf8).alias("chrom"),
                    pl.col("variant_id").map_elements(lambda v: parse_variant_id(v)["pos"], return_dtype=pl.Int64).alias("pos"),
                )
            else:
                variants_df = pl.DataFrame(schema={"variant_id": pl.Utf8, "chrom": pl.Utf8, "pos": pl.Int64})

            preds = adapter.score({"candidate_df": candidate_df, "variants_df": variants_df})
            qc = adapter.qc(preds)
            print(f"  {model_id}: rows={preds.height} qc={qc}")
            frames.append(preds)
        except Exception as exc:
            print(f"[discover_encode_models] ERROR scoring {model_id}: {exc}")
            frames.append(handle_missing_prediction(candidate_df, model_id))

    if frames:
        combined = pl.concat(frames, how="vertical_relaxed")
    else:
        combined = pl.DataFrame()
    write_parquet(combined, args.output)
    print(f"[discover_encode_models] Wrote {combined.height} prediction rows -> {args.output}")


if __name__ == "__main__":
    main()
