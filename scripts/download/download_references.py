#!/usr/bin/env python3
"""Download GRCh38 reference FASTA and GENCODE v47 GTF.

Downloads the NCBI GRCh38 no-alt analysis set FASTA, decompresses it,
runs ``samtools faidx``, and downloads + decompresses the GENCODE v47
annotation GTF.

CLI (Snakemake):
    python scripts/download/download_references.py \
        --output-dir data/reference --config config/datasets.yaml
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
from pathlib import Path

from v2gbench.io.download import download_with_aria2c
from v2gbench.utils.config import load_config


def _gunzip(src: Path, dst: Path) -> None:
    """Decompress a gzip file to *dst*."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    print(f"  Decompressed {src.name} -> {dst.name}")


def _samtools_faidx(fasta: Path) -> Path:
    """Run ``samtools faidx`` on *fasta*, returning the .fai path."""
    fai = fasta.with_suffix(fasta.suffix + ".fai")
    if fai.exists():
        print(f"  FASTA index already exists: {fai}")
        return fai
    if shutil.which("samtools") is None:
        print("  WARNING: samtools not found — skipping faidx. "
              "Index manually before running bcftools/samtools-based steps.",
              file=sys.stderr)
        return fai
    print(f"  Running samtools faidx on {fasta.name} ...")
    subprocess.run(["samtools", "faidx", str(fasta)], check=True)
    print(f"  Created index: {fai}")
    return fai


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download GRCh38 FASTA + GENCODE v47 GTF"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory (e.g. data/reference)")
    parser.add_argument("--config", required=True, help="Path to datasets.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = cfg.get("datasets", {})
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- GRCh38 FASTA ---
    fasta_cfg = datasets.get("GRCh38_FASTA", {})
    fasta_url = fasta_cfg.get("url")
    if not fasta_url:
        print("ERROR: GRCh38_FASTA url not found in config", file=sys.stderr)
        return 1

    fasta_gz = out_dir / "GRCh38.fa.gz"
    fasta_fa = out_dir / "GRCh38.fa"

    print(f"[1/2] Downloading GRCh38 FASTA from NCBI ...")
    if fasta_fa.exists():
        print(f"  {fasta_fa} already exists — skipping download.")
    else:
        if not download_with_aria2c(fasta_url, fasta_gz):
            print(f"ERROR: Failed to download FASTA from {fasta_url}", file=sys.stderr)
            return 1
        _gunzip(fasta_gz, fasta_fa)
        fasta_gz.unlink(missing_ok=True)

    # Index
    _samtools_faidx(fasta_fa)

    # --- GENCODE v47 GTF ---
    gtf_cfg = datasets.get("GENCODE_V47", {})
    gtf_url = gtf_cfg.get("url")
    if not gtf_url:
        print("ERROR: GENCODE_V47 url not found in config", file=sys.stderr)
        return 1

    gtf_gz = out_dir / "gencode.v47.annotation.gtf.gz"
    gtf_out = out_dir / "gencode.v47.genes.gtf"

    print(f"[2/2] Downloading GENCODE v47 GTF from EBI ...")
    if gtf_out.exists():
        print(f"  {gtf_out} already exists — skipping download.")
    else:
        if not download_with_aria2c(gtf_url, gtf_gz):
            print(f"ERROR: Failed to download GTF from {gtf_url}", file=sys.stderr)
            return 1
        _gunzip(gtf_gz, gtf_out)
        gtf_gz.unlink(missing_ok=True)

    print("Reference download complete.")
    print(f"  FASTA:  {fasta_fa}")
    print(f"  Index:  {fasta_fa}.fai")
    print(f"  GTF:    {gtf_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
