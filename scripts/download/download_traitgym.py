#!/usr/bin/env python3
"""Clone the songlab-cal/TraitGym repo.

Shallow-clones the TraitGym variant-effect benchmark repository.

CLI (Snakemake):
    python scripts/download/download_traitgym.py \
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
        description="Clone songlab-cal/TraitGym repo"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory (e.g. data/raw)")
    parser.add_argument("--config", required=True, help="Path to datasets.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg.get("datasets", {})
    tg_cfg = datasets.get("TraitGym", {})
    repo = tg_cfg.get("repo", "songlab-cal/TraitGym")
    repo_url = tg_cfg.get("url", f"https://github.com/{repo}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clone_dest = out_dir / "traitgym"

    print(f"Cloning {repo} ...")
    if not _git_clone(repo_url, clone_dest):
        print(f"ERROR: Failed to clone {repo_url}", file=sys.stderr)
        return 1

    print(f"TraitGym repo cloned successfully.")
    print(f"  Repo: {clone_dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
