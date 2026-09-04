#!/usr/bin/env python3
"""Resource discovery — discover and lock all external resource versions."""

import argparse
import csv
import os
import subprocess
import sys
import yaml
import requests
from pathlib import Path
from datetime import datetime


def get_github_latest_commit(repo: str, token: str = None) -> dict:
    """Get latest commit SHA and date for a GitHub repo."""
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    try:
        r = requests.get(f"https://api.github.com/repos/{repo}/commits", headers=headers, timeout=30)
        if r.status_code == 200:
            data = r.json()[0]
            return {
                "commit": data["sha"],
                "date": data["commit"]["committer"]["date"],
            }
    except Exception as e:
        print(f"  Warning: could not get commit for {repo}: {e}")
    return {"commit": "unknown", "date": "unknown"}


def get_zenodo_record(zenodo_id: str) -> dict:
    """Get Zenodo record metadata."""
    try:
        r = requests.get(f"https://zenodo.org/api/records/{zenodo_id}", timeout=30)
        if r.status_code == 200:
            data = r.json()
            return {
                "id": str(data.get("id", zenodo_id)),
                "doi": data.get("doi", "unknown"),
                "version": data.get("metadata", {}).get("version", "unknown"),
                "files": len(data.get("files", [])),
            }
    except Exception as e:
        print(f"  Warning: could not get Zenodo record {zenodo_id}: {e}")
    return {"id": zenodo_id, "doi": "unknown", "version": "unknown", "files": 0}


def main():
    parser = argparse.ArgumentParser(description="Discover and lock external resources")
    parser.add_argument("--config", required=True, help="Path to site.yaml")
    parser.add_argument("--output", required=True, help="Output TSV path")
    args = parser.parse_args()

    with open(args.config) as f:
        site = yaml.safe_load(f)

    github_token = os.environ.get("GITHUB_TOKEN", "")

    print("Discovering external resources...")

    rows = []

    # GitHub repos
    github_repos = [
        ("EngreitzLab/CRISPR_comparison", "dataset", "ENCODE_CRISPR"),
        ("EngreitzLab/ENCODE_Test_Dataset_Analysis", "dataset", "ENCODE_HeldOut"),
        ("EngreitzLab/GWAS_E2G_benchmarking", "dataset", "GWAS_E2G"),
        ("opentargets-archive/genetics-gold-standards", "dataset", "OpenTargets_GoldStandard"),
        ("songlab-cal/TraitGym", "dataset", "TraitGym"),
    ]

    for repo, rtype, rid in github_repos:
        print(f"  GitHub: {repo}")
        info = get_github_latest_commit(repo, github_token)
        rows.append({
            "resource_id": rid,
            "type": rtype,
            "source": "GitHub",
            "repo": repo,
            "version": info["commit"],
            "date": info["date"],
            "url": f"https://github.com/{repo}",
            "discovered_at": datetime.now().isoformat(),
        })

    # Zenodo
    print("  Zenodo: 11211925 (pgBoost)")
    zenodo = get_zenodo_record("11211925")
    rows.append({
        "resource_id": "pgBoost_Zenodo",
        "type": "dataset",
        "source": "Zenodo",
        "repo": "N/A",
        "version": zenodo["doi"],
        "date": zenodo["id"],
        "url": f"https://zenodo.org/record/{zenodo['id']}",
        "discovered_at": datetime.now().isoformat(),
    })

    # GTEx
    rows.append({
        "resource_id": "GTEx_V11",
        "type": "dataset",
        "source": "GTEx Portal",
        "repo": "N/A",
        "version": "v11",
        "date": "2024",
        "url": "https://gtexportal.org/home/downloads/adult-gtex",
        "discovered_at": datetime.now().isoformat(),
    })

    # eQTL Catalogue
    rows.append({
        "resource_id": "eQTLCatalogue_STABLE",
        "type": "dataset",
        "source": "EBI",
        "repo": "N/A",
        "version": "stable",
        "date": "2024",
        "url": "https://www.ebi.ac.uk/eqtl/",
        "discovered_at": datetime.now().isoformat(),
    })

    # Reference genome
    rows.append({
        "resource_id": "GRCh38_FASTA",
        "type": "reference",
        "source": "NCBI",
        "repo": "N/A",
        "version": "GRCh38",
        "date": "2013",
        "url": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/",
        "discovered_at": datetime.now().isoformat(),
    })

    rows.append({
        "resource_id": "GENCODE_V47",
        "type": "reference",
        "source": "GENCODE",
        "repo": "N/A",
        "version": "v47",
        "date": "2024",
        "url": "https://www.gencodegenes.org/human/release_47.html",
        "discovered_at": datetime.now().isoformat(),
    })

    # AlphaGenome
    rows.append({
        "resource_id": "AlphaGenome_API",
        "type": "model",
        "source": "Google DeepMind",
        "repo": "N/A",
        "version": "alphagenome-0.8.0",
        "date": "2025",
        "url": "https://alphagenome.deepmind.com/",
        "discovered_at": datetime.now().isoformat(),
    })

    # Borzoi
    rows.append({
        "resource_id": "Borzoi_weights",
        "type": "model_weights",
        "source": "Calico",
        "repo": "N/A",
        "version": "borzoi-pytorch-0.5.1",
        "date": "2024",
        "url": "https://github.com/calico/borzoi",
        "discovered_at": datetime.now().isoformat(),
    })

    # Enformer
    rows.append({
        "resource_id": "Enformer_weights",
        "type": "model_weights",
        "source": "Google DeepMind",
        "repo": "N/A",
        "version": "enformer-pytorch",
        "date": "2021",
        "url": "https://github.com/google-deepmind/deepmind-research/tree/master/enformer",
        "discovered_at": datetime.now().isoformat(),
    })

    # Write TSV
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResource manifest written to {output}")
    print(f"Total resources: {len(rows)}")


if __name__ == "__main__":
    main()
