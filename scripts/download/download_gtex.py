#!/usr/bin/env python3
"""Download the GTEx v11 SuSiE cis-eQTL tar archive.

Downloads the single-tissue SuSiE fine-mapping tar from Google Cloud Storage.

CLI (Snakemake):
    python scripts/download/download_gtex.py \
        --output-dir data/raw --config config/datasets.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from v2gbench.io.download import download_with_aria2c, verify_sha256
from v2gbench.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download GTEx v11 SuSiE eQTL tar"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory (e.g. data/raw)")
    parser.add_argument("--config", required=True, help="Path to datasets.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg.get("datasets", {})
    gtex_cfg = datasets.get("GTEx_V11", {})
    url = gtex_cfg.get("url")
    if not url:
        print("ERROR: GTEx_V11 url not found in config", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gtex_v11_susie.tar"

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"GTEx tar already exists: {out_path} — skipping download.")
        return 0

    print(f"Downloading GTEx v11 SuSiE eQTL tar ...")
    print(f"  URL: {url}")
    if not download_with_aria2c(url, out_path):
        print(f"ERROR: Failed to download GTEx tar from {url}", file=sys.stderr)
        return 1

    actual_hash = verify_sha256(out_path)
    print(f"Download complete: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1e9:.3f} GB")
    print(f"  SHA256: {actual_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
