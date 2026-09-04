#!/usr/bin/env python3
"""Download eQTL Catalogue files (stable + r8_beta releases) via the EBI API.

Queries the eQTL Catalogue REST API to enumerate available files for each
release, then downloads them into separate subdirectories.

CLI (Snakemake):
    python scripts/download/download_eqtl_catalogue.py \
        --output-dir data/raw --config config/datasets.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import requests

from v2gbench.io.download import download_with_aria2c
from v2gbench.utils.config import load_config

# Base API URL (trailing slash kept for clean joins).
_API_BASE = "https://www.ebi.ac.uk/eqtl/api/"
# Known file-listing endpoints per release.
_RELEASE_ENDPOINTS = {
    "stable": "files/stable",
    "r8_beta": "files/r8_beta",
}


def _api_get(url: str, max_retries: int = 3, timeout: int = 60) -> Optional[dict]:
    """GET *url* from the eQTL Catalogue API with retries."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
            if r.status_code == 200:
                return r.json()
            print(f"  API returned {r.status_code} for {url}", file=sys.stderr)
        except requests.RequestException as exc:
            print(f"  API request failed (attempt {attempt + 1}): {exc}", file=sys.stderr)
        time.sleep(5 * (attempt + 1))
    return None


def _list_files_paginated(endpoint: str) -> list[dict]:
    """Paginate through the eQTL Catalogue ``/files`` endpoint.

    The API returns ``{"_embedded": {"files": [...]}, "_links": {"next": ...}}``
    or a plain list.  We follow ``next`` links until exhausted.
    """
    url = f"{_API_BASE}{endpoint}"
    all_files: list[dict] = []
    visited: set[str] = set()
    while url and url not in visited:
        visited.add(url)
        data = _api_get(url)
        if data is None:
            break
        embedded = data.get("_embedded", data)
        page_files = embedded.get("files", []) if isinstance(embedded, dict) else embedded
        if isinstance(page_files, list):
            all_files.extend(page_files)
        links = data.get("_links", {})
        next_link = links.get("next", {})
        url = next_link.get("href") if isinstance(next_link, dict) else None
    return all_files


def _resolve_download_url(file_entry: dict, release: str) -> Optional[str]:
    """Extract a download URL from a file-listing entry."""
    for key in ("downloadLink", "download_url", "url", "href"):
        val = file_entry.get(key)
        if isinstance(val, str) and val.startswith("http"):
            return val
    rel = file_entry.get("relativePath") or file_entry.get("path")
    fname = file_entry.get("fileName") or file_entry.get("name")
    if rel:
        return f"{_API_BASE}{rel.lstrip('/')}"
    if fname:
        return f"{_API_BASE}files/{release}/{fname}"
    return None


def _download_release(release: str, endpoint: str, out_dir: Path) -> bool:
    """Download all files for a single release."""
    release_dir = out_dir / f"eqtl_catalogue_{release}"
    release_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Querying API for release '{release}' ...")
    file_entries = _list_files_paginated(endpoint)

    if not file_entries:
        print(f"  WARNING: No files listed by API for release '{release}'.", file=sys.stderr)
        print(f"  Directory created: {release_dir}")
        return True  # not fatal — directory created

    print(f"  Found {len(file_entries)} file(s) for release '{release}'.")
    ok_count = 0
    for entry in file_entries:
        dl_url = _resolve_download_url(entry, release)
        if not dl_url:
            print(f"    SKIP: no download URL for {entry}")
            continue
        fname = entry.get("fileName") or entry.get("name") or dl_url.rsplit("/", 1)[-1]
        out_path = release_dir / fname
        if out_path.exists() and out_path.stat().st_size > 0:
            ok_count += 1
            continue
        print(f"    Downloading {fname} ...")
        if download_with_aria2c(dl_url, out_path):
            ok_count += 1
        else:
            print(f"    FAILED: {fname}", file=sys.stderr)

    print(f"  Release '{release}': {ok_count}/{len(file_entries)} files downloaded.")
    return ok_count > 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download eQTL Catalogue (stable + r8_beta) via EBI API"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory (e.g. data/raw)")
    parser.add_argument("--config", required=True, help="Path to datasets.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg.get("datasets", {})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    releases = []
    for key in ("eQTLCatalogue_STABLE", "eQTLCatalogue_R8_BETA"):
        d = datasets.get(key, {})
        rel = d.get("release")
        if rel:
            releases.append(rel)

    if not releases:
        releases = ["stable", "r8_beta"]

    print("eQTL Catalogue download")
    print(f"  Releases: {releases}")
    print(f"  API base: {_API_BASE}")

    all_ok = True
    for rel in releases:
        endpoint = _RELEASE_ENDPOINTS.get(rel, f"files/{rel}")
        ok = _download_release(rel, endpoint, out_dir)
        if not ok:
            all_ok = False

    if all_ok:
        print("\neQTL Catalogue download complete.")
    else:
        print("\nWARNING: Some releases had no downloadable files.", file=sys.stderr)
    return 0  # non-fatal — directories created


if __name__ == "__main__":
    sys.exit(main())
