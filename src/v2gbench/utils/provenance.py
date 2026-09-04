"""Provenance lock file generation."""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_info(repo_path: str | Path = ".") -> Dict[str, str]:
    """Get git commit and dirty status."""
    info = {"commit": "unknown", "tree_hash": "unknown", "dirty": "unknown"}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_path, text=True
        ).strip()
        info["tree_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repo_path, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_path, text=True
        ).strip()
        info["dirty"] = "true" if dirty else "false"
    except Exception:
        pass
    return info


def generate_provenance_lock(
    output_path: str | Path,
    datasets: Dict[str, Any],
    models: Dict[str, Any],
    config_hash: str,
    git_info: Optional[Dict[str, str]] = None,
) -> None:
    """Generate provenance.lock.yaml file."""
    if git_info is None:
        git_info = get_git_info()

    lock = {
        "timestamp": datetime.now().isoformat(),
        "config_hash": config_hash,
        "git": git_info,
        "datasets": {},
        "models": {},
    }

    for ds_id, ds_info in datasets.items():
        lock["datasets"][ds_id] = {
            "source": ds_info.get("source", "unknown"),
            "url": ds_info.get("url", "unknown"),
            "release": ds_info.get("release", "unknown"),
            "adapter": ds_info.get("adapter", "unknown"),
        }

    for model_id, model_info in models.items():
        if model_info.get("enabled") == "true" or model_info.get("enabled") is True:
            lock["models"][model_id] = {
                "family": model_info.get("family", "unknown"),
                "mode": model_info.get("mode", "unknown"),
                "version": model_info.get("paper_version", model_info.get("version", "unknown")),
            }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write as YAML-like format
    with open(output_path, "w") as f:
        f.write("# Provenance lock — generated at run start, immutable\n")
        import yaml
        yaml.dump(lock, f, default_flow_style=False, sort_keys=True)


def generate_run_manifest(
    output_path: str | Path,
    run_id: str,
    start: str,
    end: str,
    hostname: str,
    config_hash: str,
    git_info: Dict[str, str],
    n_jobs: int,
    n_failed: int,
    n_retries: int,
) -> None:
    """Generate run_manifest.json."""
    manifest = {
        "run_id": run_id,
        "start": start,
        "end": end,
        "hostname": hostname,
        "config_hash": config_hash,
        "git_commit": git_info.get("commit", "unknown"),
        "git_tree_hash": git_info.get("tree_hash", "unknown"),
        "git_dirty": git_info.get("dirty", "unknown"),
        "number_jobs": n_jobs,
        "failed_jobs": n_failed,
        "resolved_retries": n_retries,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
