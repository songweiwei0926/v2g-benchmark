#!/usr/bin/env python3
"""Verify published prediction downloads and build a manifest.

Checks that the Synapse (ENCODE-rE2G) and Zenodo (pgBoost) download
directories exist and contain files, then writes a manifest TSV listing
all available published prediction files.

CLI (Snakemake):
    python scripts/download/download_published_predictions.py \
        --output-dir data/raw --config config/datasets.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from v2gbench.utils.config import load_config


def _scan_dir(d: Path) -> list[Path]:
    """Recursively list all regular files under *d*."""
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and organize published prediction downloads"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory (e.g. data/raw)")
    parser.add_argument("--config", required=True, help="Path to datasets.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg.get("datasets", {})
    out_dir = Path(args.output_dir)

    # Expected download directories.
    synapse_dir = out_dir / "encode_predictions_bundle"
    zenodo_dir = out_dir / "pgboost_zenodo"

    print("Verifying published prediction downloads ...")

    # --- Synapse (ENCODE-rE2G) ---
    synapse_files = _scan_dir(synapse_dir)
    synapse_cfg = datasets.get("ENCODE_Predictions_Bundle", {})
    print(f"\n  Synapse (ENCODE-rE2G):")
    print(f"    Directory: {synapse_dir}")
    print(f"    Files found: {len(synapse_files)}")
    if not synapse_files:
        print(f"    WARNING: No files found — Synapse download may have failed.",
              file=sys.stderr)
    else:
        total_gb = sum(f.stat().st_size for f in synapse_files) / 1e9
        print(f"    Total size: {total_gb:.2f} GB (expected ~{synapse_cfg.get('size_gb', 76)} GB)")

    # --- Zenodo (pgBoost) ---
    zenodo_files = _scan_dir(zenodo_dir)
    zenodo_cfg = datasets.get("pgBoost_Zenodo", {})
    expected_zenodo = zenodo_cfg.get("files", [
        "pgboost_scores.tsv.gz",
        "constituent_method_scores.tsv.gz",
        "gwas_evaluation.tsv",
    ])
    print(f"\n  Zenodo (pgBoost):")
    print(f"    Directory: {zenodo_dir}")
    print(f"    Files found: {len(zenodo_files)}")
    found_names = {f.name for f in zenodo_files}
    missing = [fn for fn in expected_zenodo if fn not in found_names]
    if missing:
        print(f"    WARNING: Missing expected files: {missing}", file=sys.stderr)
    else:
        print(f"    All expected files present: {expected_zenodo}")

    # --- Build manifest ---
    manifest_path = out_dir / "published_predictions_manifest.tsv"
    rows = []
    now = datetime.now().isoformat()

    for f in synapse_files:
        rows.append({
            "source": "synapse_encode_re2g",
            "filename": f.name,
            "relative_path": str(f.relative_to(out_dir)),
            "size_bytes": f.stat().st_size,
            "verified_at": now,
        })

    for f in zenodo_files:
        rows.append({
            "source": "zenodo_pgboost",
            "filename": f.name,
            "relative_path": str(f.relative_to(out_dir)),
            "size_bytes": f.stat().st_size,
            "verified_at": now,
        })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source", "filename", "relative_path", "size_bytes", "verified_at"]
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest written: {manifest_path}")
    print(f"  Total files: {len(rows)}")

    # Non-fatal if some downloads are missing — the manifest records what exists.
    if not synapse_files and not zenodo_files:
        print("ERROR: No published prediction files found at all.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
