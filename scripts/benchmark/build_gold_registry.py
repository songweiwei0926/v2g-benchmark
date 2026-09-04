#!/usr/bin/env python3
"""Build the gold evidence registry (evidence_long + canonical_pairs).

Combines all per-dataset adapter outputs into a single long-format evidence
table, assigns eQTL / CRISPR labels, de-duplicates GTEx-in-Catalogue rows,
and aggregates into canonical (locus, gene, context) pairs.

CLI (Snakemake interface)
-------------------------
    build_gold_registry.py
        --variants <parquet> --contexts <parquet> --gene-master <parquet>
        --gtex <path> --eqtl <path> --crispr <path> --gwas <path>
        --opentargets <path> --pgboost <path>
        --output <parquet> --canonical-output <parquet>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import polars as pl

from v2gbench.benchmark.gold_registry import (
    build_evidence_long,
    build_canonical_pairs,
    assign_eqtl_labels,
    assign_crispr_labels,
    deduplicate_gtex_eqtl,
)
from v2gbench.io.parquet import read_parquet, write_parquet
from v2gbench.utils.config import load_config


def _find_root() -> Path:
    cur = Path.cwd()
    while cur != cur.parent:
        if (cur / "config" / "project.yaml").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def _load_project_config() -> Dict:
    root = _find_root()
    cfg_path = root / "config" / "project.yaml"
    if cfg_path.exists():
        return load_config(cfg_path)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build evidence_long + canonical_pairs from gold-standard datasets."
    )
    parser.add_argument("--variants", required=True, help="Variants parquet path")
    parser.add_argument("--contexts", required=True, help="Contexts parquet path")
    parser.add_argument("--gene-master", required=True, help="GENCODE gene master parquet path")
    parser.add_argument("--gtex", required=True, help="GTEx adapter output path")
    parser.add_argument("--eqtl", required=True, help="eQTL Catalogue adapter output path")
    parser.add_argument("--crispr", required=True, help="CRISPR adapter output path")
    parser.add_argument("--gwas", required=True, help="GWAS adapter output path")
    parser.add_argument("--opentargets", required=True, help="OpenTargets adapter output path")
    parser.add_argument("--pgboost", default=None, help="pgBoost adapter output path (optional, model not gold standard)")
    parser.add_argument("--output", required=True, help="Output evidence_long parquet path")
    parser.add_argument("--canonical-output", required=True, help="Output canonical_pairs parquet path")
    args = parser.parse_args()

    project_cfg = _load_project_config()
    eqtl_cfg = project_cfg.get("eqtl", {})
    primary_pip = float(eqtl_cfg.get("primary_pip", 0.90))
    sensitivity_pips = tuple(eqtl_cfg.get("sensitivity_pip", [0.50, 0.70, 0.90, 0.95]))
    negative_pip = float(eqtl_cfg.get("negative_pip", 0.01))

    # Assemble the adapter-output mapping. Each path may be a parquet file or a
    # directory of per-dataset parquet files; build_evidence_long handles both.
    adapters_output: Dict[str, str] = {
        "GTEx_V11": args.gtex,
        "eQTLCatalogue": args.eqtl,
        "ENCODE_CRISPR": args.crispr,
        "GWAS_E2G": args.gwas,
        "OpenTargets_GoldStandard": args.opentargets,
    }
    if args.pgboost is not None and Path(args.pgboost).exists():
        adapters_output["pgBoost_Zenodo"] = args.pgboost

    print("[build_gold_registry] Combining adapter outputs into evidence_long ...")
    evidence = build_evidence_long(adapters_output, args.output)
    print(f"[build_gold_registry] evidence_long rows: {evidence.height}")

    # De-duplicate GTEx rows embedded in the eQTL Catalogue.
    print("[build_gold_registry] De-duplicating GTEx-in-Catalogue rows ...")
    evidence = deduplicate_gtex_eqtl(evidence)

    # Assign labels by evidence type.
    print("[build_gold_registry] Assigning eQTL labels (PIP thresholds) ...")
    evidence = assign_eqtl_labels(
        evidence,
        primary_pip=primary_pip,
        sensitivity_pips=sensitivity_pips,
        negative_pip=negative_pip,
    )
    print("[build_gold_registry] Assigning CRISPR labels (author calls) ...")
    evidence = assign_crispr_labels(evidence)

    # Persist the labelled evidence_long.
    write_parquet(evidence, args.output)
    print(f"[build_gold_registry] Wrote labelled evidence_long -> {args.output}")

    # Aggregate into canonical pairs.
    print("[build_gold_registry] Aggregating canonical pairs ...")
    canonical = build_canonical_pairs(evidence, args.canonical_output)
    print(f"[build_gold_registry] canonical_pairs rows: {canonical.height}")
    print(f"[build_gold_registry] Wrote canonical_pairs -> {args.canonical_output}")

    # Quick label summary.
    if "label" in evidence.columns and evidence.height > 0:
        summary = evidence.group_by("label").len().sort("label")
        print("[build_gold_registry] Label counts:")
        print(summary)


if __name__ == "__main__":
    main()
