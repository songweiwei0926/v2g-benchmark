#!/usr/bin/env python3
"""Download the ENCODE-rE2G prediction bundle from Synapse.

Uses ``synapseclient`` to authenticate via the ``SYNAPSE_AUTH_TOKEN``
environment variable and download the ENCODE-rE2G prediction bundle.

The script first attempts to locate the project by name ("ENCODE-rE2G")
and falls back to known Synapse IDs if the search API is unavailable.

CLI (Snakemake):
    python scripts/download/download_synapse.py \
        --output-dir data/raw --config config/datasets.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from v2gbench.utils.config import load_config

# Known Synapse IDs for the ENCODE-rE2G prediction bundle.
# These are used as fallbacks if the project search fails.
_KNOWN_PROJECT_ID = "syn51338475"  # ENCODE-rE2G project (may change)
_KNOWN_FOLDER_IDS: list[str] = []  # populated if known


def _get_synapse_client() -> Optional[object]:
    """Create and authenticate a Synapse client from the env token."""
    token = os.environ.get("SYNAPSE_AUTH_TOKEN", "").strip()
    if not token:
        print("ERROR: SYNAPSE_AUTH_TOKEN environment variable not set.", file=sys.stderr)
        return None
    try:
        import synapseclient
    except ImportError:
        print("ERROR: synapseclient not installed. Install with: pip install synapseclient",
              file=sys.stderr)
        return None
    try:
        syn = synapseclient.Synapse()
        syn.login(authToken=token, silent=True)
        print("  Synapse authentication successful.")
        return syn
    except Exception as exc:
        print(f"ERROR: Synapse login failed: {exc}", file=sys.stderr)
        return None


def _find_project(syn, project_name: str) -> Optional[str]:
    """Search Synapse for a project by name, returning its synID."""
    try:
        results = syn.getChildren(parent="syn1", includeTypes=["project"])
        for child in results:
            if project_name.lower() in child.get("name", "").lower():
                return child["id"]
    except Exception:
        pass
    # Fallback: use the entity query API.
    try:
        results = syn.entityQuery(
            f"select id from project where name contains '{project_name}'"
        )
        if results and results.get("results"):
            return results["results"][0]["id"]
    except Exception as exc:
        print(f"  Project search failed: {exc}", file=sys.stderr)
    return None


def _download_recursive(syn, syn_id: str, dest: Path, depth: int = 0) -> int:
    """Recursively download a Synapse folder/project into *dest*.

    Returns the number of files downloaded.
    """
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        children = list(syn.getChildren(parent=syn_id))
    except Exception as exc:
        print(f"  {'  ' * depth}ERROR listing {syn_id}: {exc}", file=sys.stderr)
        return 0

    for child in children:
        ctype = child.get("type", "")
        cid = child.get("id")
        cname = child.get("name", cid)
        if ctype == "org.sagebionetworks.repo.model.Folder":
            print(f"  {'  ' * depth}Folder: {cname}/")
            count += _download_recursive(syn, cid, dest / cname, depth + 1)
        elif ctype == "org.sagebionetworks.repo.model.FileEntity":
            out_path = dest / cname
            if out_path.exists() and out_path.stat().st_size > 0:
                print(f"  {'  ' * depth}SKIP (exists): {cname}")
                count += 1
                continue
            print(f"  {'  ' * depth}Downloading: {cname} ...")
            try:
                syn.get(cid, downloadLocation=str(dest), ifcollision="overwrite.local")
                count += 1
            except Exception as exc:
                print(f"  {'  ' * depth}FAILED: {cname}: {exc}", file=sys.stderr)
        else:
            print(f"  {'  ' * depth}SKIP (type={ctype}): {cname}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download ENCODE-rE2G prediction bundle from Synapse"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory (e.g. data/raw)")
    parser.add_argument("--config", required=True, help="Path to datasets.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg.get("datasets", {})
    bundle_cfg = datasets.get("ENCODE_Predictions_Bundle", {})
    project_name = bundle_cfg.get("synapse_project", "ENCODE-rE2G")
    size_gb = bundle_cfg.get("size_gb", 76)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = out_dir / "encode_predictions_bundle"

    print(f"ENCODE-rE2G prediction bundle download from Synapse")
    print(f"  Project: {project_name}")
    print(f"  Expected size: ~{size_gb} GB")
    print(f"  Output: {bundle_dir}")

    syn = _get_synapse_client()
    if syn is None:
        return 1

    # Locate the project.
    project_id = _find_project(syn, project_name)
    if project_id is None:
        print(f"  Project '{project_name}' not found via search.", file=sys.stderr)
        if _KNOWN_PROJECT_ID:
            print(f"  Falling back to known project ID: {_KNOWN_PROJECT_ID}", file=sys.stderr)
            project_id = _KNOWN_PROJECT_ID
        else:
            print("ERROR: Could not locate the ENCODE-rE2G project on Synapse.", file=sys.stderr)
            return 1

    print(f"  Project Synapse ID: {project_id}")
    print(f"  Downloading all files recursively ...")
    n_files = _download_recursive(syn, project_id, bundle_dir)

    print(f"\nSynapse download complete: {n_files} file(s) downloaded to {bundle_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
