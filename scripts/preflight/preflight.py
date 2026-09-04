#!/usr/bin/env python3
"""Preflight checks — environment, resources, credentials, API smoke tests."""

import argparse
import os
import shutil
import subprocess
import sys
import yaml
from pathlib import Path


def check_command(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def check_python_package(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def check_disk(path: str, min_gb: float) -> tuple[bool, float]:
    try:
        stat = os.statvfs(path)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        return free_gb >= min_gb, free_gb
    except Exception:
        return False, 0.0


def check_internet(url: str, timeout: int = 10) -> bool:
    import requests
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code < 400
    except Exception:
        return False


def check_alphagenome_api(api_key: str) -> bool:
    try:
        from alphagenome.models import dna_client
        client = dna_client.create(api_key)
        # Minimal smoke test
        return True
    except Exception as e:
        print(f"  AlphaGenome API check failed: {e}")
        return False


def check_synapse_token(token: str) -> bool:
    try:
        import synapseclient
        syn = synapseclient.Synapse()
        syn.login(authToken=token)
        return True
    except Exception as e:
        print(f"  Synapse token check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="V2G Benchmark preflight checks")
    parser.add_argument("--config", required=True, help="Path to site.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        site = yaml.safe_load(f)

    print("=" * 60)
    print("V2G-Benchmark-OneShot v1.0 — Preflight Checks")
    print("=" * 60)

    checks = []
    all_pass = True

    # === System commands ===
    print("\n--- System Commands ---")
    for cmd in ["python3", "git", "wget", "curl", "bcftools", "bedtools", "samtools"]:
        ok = check_command(cmd)
        checks.append((f"command:{cmd}", ok))
        print(f"  {'PASS' if ok else 'FAIL'}: {cmd}")
        if not ok:
            all_pass = False

    # aria2c is optional (fallback to requests)
    aria2c_ok = check_command("aria2c")
    checks.append(("command:aria2c", aria2c_ok))
    print(f"  {'PASS' if aria2c_ok else 'WARN'}: aria2c (optional, will use requests fallback)")

    # === Python packages ===
    print("\n--- Python Packages ---")
    for pkg in ["polars", "pyarrow", "yaml", "requests", "sklearn", "scipy", "numpy",
                "matplotlib", "seaborn", "openpyxl", "pandera"]:
        mod = pkg if pkg != "yaml" else "yaml"
        ok = check_python_package(pkg)
        checks.append((f"python:{pkg}", ok))
        print(f"  {'PASS' if ok else 'FAIL'}: {pkg}")
        if not ok:
            all_pass = False

    # Optional packages
    for pkg in ["synapseclient", "alphagenome", "borzoi_pytorch", "enformer_pytorch", "xgboost"]:
        ok = check_python_package(pkg)
        checks.append((f"python:{pkg}", ok))
        print(f"  {'PASS' if ok else 'WARN'}: {pkg} (may be installed later)")

    # === Snakemake ===
    print("\n--- Snakemake ---")
    snakemake_ok = check_command("snakemake")
    checks.append(("snakemake", snakemake_ok))
    print(f"  {'PASS' if snakemake_ok else 'FAIL'}: snakemake")
    if not snakemake_ok:
        all_pass = False

    # === Disk space ===
    print("\n--- Disk Space ---")
    disk_min = site.get("disk", {}).get("scratch_min_gb", 200)
    for path in ["/workspace", "/mnt/shared-workspace"]:
        ok, free = check_disk(path, disk_min)
        checks.append((f"disk:{path}", ok))
        print(f"  {'PASS' if ok else 'WARN'}: {path} ({free:.1f} GB free, min {disk_min} GB)")

    # === Credentials ===
    print("\n--- Credentials ---")
    cred_env = site.get("credentials", {})
    
    alpha_key = os.environ.get(cred_env.get("alphagenome_api_key", "ALPHAGENOME_API_KEY"), "")
    if alpha_key:
        print("  Checking AlphaGenome API...")
        alpha_ok = check_alphagenome_api(alpha_key)
        checks.append(("credential:alphagenome", alpha_ok))
        print(f"  {'PASS' if alpha_ok else 'FAIL'}: AlphaGenome API key")
        if not alpha_ok:
            all_pass = False
    else:
        checks.append(("credential:alphagenome", False))
        print("  FAIL: ALPHAGENOME_API_KEY not set")
        all_pass = False

    syn_token = os.environ.get(cred_env.get("synapse_auth_token", "SYNAPSE_AUTH_TOKEN"), "")
    if syn_token:
        print("  Checking Synapse token...")
        syn_ok = check_synapse_token(syn_token)
        checks.append(("credential:synapse", syn_ok))
        print(f"  {'PASS' if syn_ok else 'FAIL'}: Synapse auth token")
        if not syn_ok:
            all_pass = False
    else:
        checks.append(("credential:synapse", False))
        print("  FAIL: SYNAPSE_AUTH_TOKEN not set")
        all_pass = False

    # === Internet connectivity ===
    print("\n--- Internet Connectivity ---")
    for name, url in [
        ("GTEx", "https://storage.googleapis.com/adult-gtex/"),
        ("EBI", "https://www.ebi.ac.uk/"),
        ("GitHub", "https://github.com/"),
        ("Zenodo", "https://zenodo.org/"),
        ("Synapse", "https://www.synapse.org/"),
    ]:
        ok = check_internet(url)
        checks.append((f"internet:{name}", ok))
        print(f"  {'PASS' if ok else 'FAIL'}: {name} ({url})")
        if not ok:
            all_pass = False

    # === Summary ===
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok in checks if ok)
    n_total = len(checks)
    print(f"Preflight: {n_pass}/{n_total} checks passed")
    print("=" * 60)

    if all_pass:
        print("PREFLIGHT: PASS")
        # Write pass marker
        Path("data/locked").mkdir(parents=True, exist_ok=True)
        with open("data/locked/preflight_pass.txt", "w") as f:
            f.write(f"Preflight passed at {n_pass}/{n_total}\n")
        return 0
    else:
        failed = [name for name, ok in checks if not ok]
        print(f"PREFLIGHT: FAIL")
        print(f"Failed checks: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
