#!/usr/bin/env python3
"""Clone the opentargets-archive/genetics-gold-standards repo.

Shallow-clones the OpenTargets genetics gold-standards repository.

CLI (Snakemake):
    python scripts/download/download_opentargets.py \
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
        description="Clone opentargets-archive/genetics-gold-standards repo"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory (e.g. data/raw)")
    parser.add_argument("--config", required=True, help="Path to datasets.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg.get("datasets", {})
    ot_cfg = datasets.get("OpenTargets_GoldStandard", {})
    repo = ot_cfg.get("repo", "opentargets-archive/genetics-gold-standards")
    repo_url = ot_cfg.get("url", f"https://github.com/{repo}")
    key_path = ot_cfg.get("key_path", "gold_standards/processed/")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clone_dest = out_dir / "opentargets_gold_standards"

    print(f"Cloning {repo} ...")
    if not _git_clone(repo_url, clone_dest):
        print(f"ERROR: Failed to clone {repo_url}", file=sys.stderr)
        return 1

    gold_dir = clone_dest / key_path
    if gold_dir.exists():
        n_files = sum(1 for _ in gold_dir.rglob("*") if _.is_file())
        print(f"OpenTargets gold-standards repo cloned successfully.")
        print(f"  Repo:           {clone_dest}")
        print(f"  Gold standards: {gold_dir} ({n_files} files)")
    else:
        print(f"OpenTargets gold-standards repo cloned successfully.")
        print(f"  Repo: {clone_dest}")
        print(f"  Note: key_path '{key_path}' not found — check repo structure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
