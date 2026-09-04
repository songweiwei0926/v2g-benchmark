#!/usr/bin/env python3
"""Clone the EngreitzLab/CRISPR_comparison repo and extract the key TSV.

Clones the repository with ``--depth 1`` and verifies the presence of the
benchmark ensemble data file ``EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz``.

CLI (Snakemake):
    python scripts/download/download_crispr.py \
        --output-dir data/raw --config config/datasets.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from v2gbench.utils.config import load_config


def _git_clone(repo_url: str, dest: Path) -> bool:
    """Shallow-clone *repo_url* into *dest*."""
    if dest.exists() and (dest / ".git").exists():
        print(f"  Repo already cloned: {dest}")
        return True
    if dest.exists():
        # Partial / stale — remove and re-clone.
        import shutil
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", repo_url, str(dest)]
    print(f"  Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  git clone failed:\n{exc.stderr}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("  ERROR: git not found on PATH", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clone CRISPR_comparison repo and extract key TSV"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory (e.g. data/raw)")
    parser.add_argument("--config", required=True, help="Path to datasets.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg.get("datasets", {})
    crispr_cfg = datasets.get("ENCODE_CRISPR", {})
    repo = crispr_cfg.get("repo", "EngreitzLab/CRISPR_comparison")
    repo_url = crispr_cfg.get("url", f"https://github.com/{repo}")
    key_file = crispr_cfg.get("key_file", "resources/crispr_data/EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clone_dest = out_dir / "crispr_comparison"

    print(f"Cloning {repo} ...")
    if not _git_clone(repo_url, clone_dest):
        print(f"ERROR: Failed to clone {repo_url}", file=sys.stderr)
        return 1

    key_path = clone_dest / key_file
    if not key_path.exists():
        print(f"ERROR: Key file not found after clone: {key_path}", file=sys.stderr)
        print(f"  Searched in: {clone_dest}", file=sys.stderr)
        return 1

    print(f"CRISPR comparison repo cloned successfully.")
    print(f"  Repo:     {clone_dest}")
    print(f"  Key file: {key_path}")
    print(f"  Size:     {key_path.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
