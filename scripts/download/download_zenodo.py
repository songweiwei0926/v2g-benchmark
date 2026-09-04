#!/usr/bin/env python3
"""Download pgBoost benchmark files from Zenodo record 11211925.

Downloads the three key files from the pgBoost Zenodo deposit:
  - pgboost_scores.tsv.gz
  - constituent_method_scores.tsv.gz
  - gwas_evaluation.tsv

Uses the Zenodo REST API to resolve direct download URLs.

CLI (Snakemake):
    python scripts/download/download_zenodo.py \
        --output-dir data/raw --config config/datasets.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import requests

from v2gbench.io.download import download_with_aria2c, verify_sha256
from v2gbench.utils.config import load_config


def _get_zenodo_files(zenodo_id: str, max_retries: int = 3) -> list[dict]:
    """Fetch the file listing from the Zenodo API for *zenodo_id*."""
    url = f"https://zenodo.org/api/records/{zenodo_id}"
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                data = r.json()
                return data.get("files", [])
            print(f"  Zenodo API returned {r.status_code}", file=sys.stderr)
        except requests.RequestException as exc:
            print(f"  Zenodo API request failed (attempt {attempt + 1}): {exc}", file=sys.stderr)
        time.sleep(5 * (attempt + 1))
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download pgBoost files from Zenodo 11211925"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory (e.g. data/raw)")
    parser.add_argument("--config", required=True, help="Path to datasets.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg.get("datasets", {})
    zenodo_cfg = datasets.get("pgBoost_Zenodo", {})
    zenodo_id = str(zenodo_cfg.get("zenodo_id", "11211925"))
    expected_files = zenodo_cfg.get("files", [
        "pgboost_scores.tsv.gz",
        "constituent_method_scores.tsv.gz",
        "gwas_evaluation.tsv",
    ])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zenodo_dir = out_dir / "pgboost_zenodo"
    zenodo_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching Zenodo record {zenodo_id} file listing ...")
    zenodo_files = _get_zenodo_files(zenodo_id)

    # Build a map of filename -> download URL.
    file_map: dict[str, str] = {}
    for f in zenodo_files:
        fname = f.get("key", f.get("filename", ""))
        dl_url = f.get("links", {}).get("self") or f.get("download_url", "")
        if fname and dl_url:
            file_map[fname] = dl_url

    if not file_map:
        print(f"  WARNING: Zenodo API returned no files. Using direct URL fallback.", file=sys.stderr)
        for fname in expected_files:
            file_map[fname] = f"https://zenodo.org/records/{zenodo_id}/files/{fname}?download=1"

    print(f"  Available files: {list(file_map.keys())}")

    ok_count = 0
    for fname in expected_files:
        out_path = zenodo_dir / fname
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  SKIP (exists): {fname}")
            ok_count += 1
            continue

        dl_url = file_map.get(fname)
        if not dl_url:
            print(f"  ERROR: '{fname}' not found in Zenodo record {zenodo_id}", file=sys.stderr)
            continue

        print(f"  Downloading {fname} ...")
        if download_with_aria2c(dl_url, out_path):
            actual_hash = verify_sha256(out_path)
            print(f"    OK: {out_path.stat().st_size / 1e6:.2f} MB  SHA256: {actual_hash}")
            ok_count += 1
        else:
            print(f"    FAILED: {fname}", file=sys.stderr)

    print(f"\npgBoost Zenodo download: {ok_count}/{len(expected_files)} files.")
    if ok_count == len(expected_files):
        print(f"  Output: {zenodo_dir}")
        return 0
    else:
        print(f"  Some files missing — check errors above.", file=sys.stderr)
        return 1 if ok_count == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
