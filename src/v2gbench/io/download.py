"""Download utilities with aria2c and fallback to requests."""
import hashlib
import os
import subprocess
import requests
from pathlib import Path
from typing import Optional


def download_with_aria2c(
    url: str,
    output_path: str | Path,
    max_retries: int = 5,
    connections: int = 8,
    timeout: int = 600,
) -> bool:
    """Download a file using aria2c with resume and retry support."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "aria2c",
        "--max-tries", str(max_retries),
        "--retry-wait", "5",
        "--connect-timeout", "30",
        "--timeout", str(timeout),
        "--max-connection-per-server", str(connections),
        "--split", str(connections),
        "--continue", "true",
        "--dir", str(output_path.parent),
        "--out", output_path.name,
        url,
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except FileNotFoundError:
        # aria2c not installed, fall back to requests
        return download_with_requests(url, output_path, max_retries)
    except subprocess.CalledProcessError as e:
        print(f"aria2c failed: {e.stderr}")
        return download_with_requests(url, output_path, max_retries)


def download_with_requests(
    url: str,
    output_path: str | Path,
    max_retries: int = 5,
    chunk_size: int = 1024 * 1024,
) -> bool:
    """Download a file using requests with retry and resume support."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for partial download
    existing_size = output_path.stat().st_size if output_path.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=600)
            r.raise_for_status()

            mode = "ab" if existing_size > 0 else "wb"
            total = int(r.headers.get("content-length", 0)) + existing_size

            with open(output_path, mode) as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)

            return True
        except Exception as e:
            print(f"Download attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(5 * (attempt + 1))
    return False


def verify_sha256(path: str | Path, expected_sha256: Optional[str] = None) -> str | bool:
    """Verify SHA256 checksum of a file. Returns hash or False if mismatch."""
    path = Path(path)
    if not path.exists():
        return False

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()

    if expected_sha256:
        return actual == expected_sha256
    return actual
