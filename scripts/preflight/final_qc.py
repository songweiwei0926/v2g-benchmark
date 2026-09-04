#!/usr/bin/env python3
"""Final QC — verify all mandatory components are complete before SUCCESS."""

import argparse
import csv
import hashlib
import json
import os
import sys
import yaml
from pathlib import Path
import polars as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def check_file_exists(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def check_parquet_nonempty(path: str | Path) -> tuple[bool, int]:
    if not check_file_exists(path):
        return False, 0
    try:
        df = pl.read_parquet(path)
        return len(df) > 0, len(df)
    except Exception:
        return False, 0


def check_all_metrics_finite(metrics_path: str | Path) -> bool:
    if not check_file_exists(metrics_path):
        return False
    try:
        df = pl.read_csv(metrics_path, separator="\t")
        numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float64, pl.Float32, pl.Int64]]
        for col in numeric_cols:
            if df[col].null_count() == len(df):
                return False
            if df[col].is_infinite().any():
                return False
        return True
    except Exception:
        return False


def check_gold_coverage(report_path: str | Path) -> bool:
    if not check_file_exists(report_path):
        return False
    try:
        df = pl.read_csv(report_path, separator="\t")
        coverage = df["coverage_percentage"].item() if "coverage_percentage" in df.columns else 0
        return coverage == 100.0
    except Exception:
        return False


def check_config_hash_unchanged(run_lock_path: str | Path, config_dir: str = "config") -> bool:
    if not check_file_exists(run_lock_path):
        return False
    with open(run_lock_path) as f:
        lock = json.load(f)
    original_hash = lock.get("config_hash", "")

    # Recompute
    hasher = hashlib.sha256()
    config_dir = Path(config_dir)
    for yml in sorted(config_dir.glob("*.yaml")):
        hasher.update(yml.read_bytes())
    for yml in sorted(config_dir.glob("*.yml")):
        hasher.update(yml.read_bytes())
    for sub in sorted(config_dir.iterdir()):
        if sub.is_dir():
            for yml in sorted(sub.glob("*.yaml")):
                hasher.update(yml.read_bytes())
    current_hash = hasher.hexdigest()

    return original_hash == current_hash


def check_mandatory_completion_matrix(path: str | Path) -> bool:
    if not check_file_exists(path):
        return False
    try:
        df = pl.read_csv(path, separator="\t")
        if "required" not in df.columns or "final_status" not in df.columns:
            return False
        mandatory = df.filter(pl.col("required") == "TRUE")
        if len(mandatory) == 0:
            return False
        return (mandatory["final_status"] == "PASS").all()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Final QC for V2G Benchmark")
    parser.add_argument("--config", required=True, help="Path to site.yaml")
    parser.add_argument("--run-lock", required=True, help="Path to run_lock.json")
    parser.add_argument("--output", required=True, help="Output directory for release")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("V2G-Benchmark-OneShot v1.0 — Final QC")
    print("=" * 60)

    checks = []
    all_pass = True

    # === Mandatory datasets ===
    print("\n--- Mandatory Datasets ---")
    dataset_checks = [
        ("data/raw/gtex_v11_susie.tar", "GTEx v11"),
        ("data/raw/eqtl_catalogue_stable/", "eQTL Catalogue STABLE"),
        ("data/raw/crispr_comparison/", "CRISPR comparison"),
        ("data/raw/gwas_e2g_benchmarking/", "GWAS E2G"),
        ("data/raw/opentargets_gold_standards/", "Open Targets"),
        ("data/raw/pgboost_zenodo/", "pgBoost Zenodo"),
        ("data/raw/traitgym/", "TraitGym"),
    ]
    for path, name in dataset_checks:
        ok = check_file_exists(path)
        checks.append((f"dataset:{name}", ok))
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
        if not ok:
            all_pass = False

    # === Mandatory models ===
    print("\n--- Mandatory Models ---")
    model_checks = [
        ("predictions/baseline/baseline_predictions.parquet", "Baselines"),
        ("predictions/published/published_predictions.parquet", "Published models"),
        ("predictions/sequence/borzoi_predictions.parquet", "Borzoi"),
        ("predictions/sequence/enformer_predictions.parquet", "Enformer"),
        ("predictions/sequence/alphagenome_predictions.parquet", "AlphaGenome"),
        ("predictions/integrated/integrated_rank_predictions.parquet", "Integrated-Rank"),
        ("predictions/integrated/integrated_logistic_predictions.parquet", "Integrated-Logistic"),
        ("predictions/integrated/integrated_xgboost_predictions.parquet", "Integrated-XGBoost"),
    ]
    for path, name in model_checks:
        ok, n = check_parquet_nonempty(path)
        checks.append((f"model:{name}", ok))
        print(f"  {'PASS' if ok else 'FAIL'}: {name} ({n} rows)")
        if not ok:
            all_pass = False

    # === Core data tables ===
    print("\n--- Core Data Tables ---")
    core_checks = [
        ("data/processed/variants_normalized.parquet", "Variants"),
        ("data/reference/gene_master.parquet", "Gene master"),
        ("data/processed/evidence_long.parquet", "Evidence long"),
        ("data/processed/canonical_pairs.parquet", "Canonical pairs"),
        ("data/processed/candidate_1m.parquet", "Candidate 1Mb"),
        ("data/processed/candidate_v2g.parquet", "Candidate V2G"),
        ("data/processed/all_model_predictions.parquet", "All predictions"),
        ("data/processed/seq_core.parquet", "SEQ_CORE"),
    ]
    for path, name in core_checks:
        ok, n = check_parquet_nonempty(path)
        checks.append((f"core:{name}", ok))
        print(f"  {'PASS' if ok else 'FAIL'}: {name} ({n} rows)")
        if not ok:
            all_pass = False

    # === Figures ===
    print("\n--- Figures ---")
    fig_checks = [
        "results/figures/fig1_overview.svg",
        "results/figures/fig2_heatmap.svg",
        "results/figures/fig3_distance.svg",
        "results/figures/fig4_context.svg",
        "results/figures/fig5_complementarity.svg",
        "results/figures/fig6_integrated.svg",
    ]
    for path in fig_checks:
        ok = check_file_exists(path)
        checks.append((f"figure:{Path(path).stem}", ok))
        print(f"  {'PASS' if ok else 'FAIL'}: {Path(path).name}")
        if not ok:
            all_pass = False

    # === Tables ===
    print("\n--- Tables ---")
    table_checks = [
        "results/tables/main_results.tsv",
        "results/tables/model_exclusions.tsv",
        "results/tables/mandatory_completion_matrix.tsv",
        "results/tables/model_score_qc.tsv",
        "results/tables/model_context_matrix.tsv",
        "results/tables/SupplementaryTables.xlsx",
    ]
    for path in table_checks:
        ok = check_file_exists(path)
        checks.append((f"table:{Path(path).stem}", ok))
        print(f"  {'PASS' if ok else 'FAIL'}: {Path(path).name}")
        if not ok:
            all_pass = False

    # === Metrics ===
    print("\n--- Metrics ---")
    metric_checks = [
        "results/metrics/ranking_metrics.tsv",
        "results/metrics/classification_metrics.tsv",
    ]
    for path in metric_checks:
        ok = check_all_metrics_finite(path)
        checks.append((f"metrics:{Path(path).stem}", ok))
        print(f"  {'PASS' if ok else 'FAIL'}: {Path(path).name} (finite)")
        if not ok:
            all_pass = False

    # === Gold coverage ===
    print("\n--- Gold Coverage ---")
    ok = check_gold_coverage("data/processed/gold_coverage_report.tsv")
    checks.append(("gold_coverage_100pct", ok))
    print(f"  {'PASS' if ok else 'FAIL'}: Gold candidate coverage 100%")
    if not ok:
        all_pass = False

    # === Config hash ===
    print("\n--- Config Integrity ---")
    ok = check_config_hash_unchanged(args.run_lock)
    checks.append(("config_hash_unchanged", ok))
    print(f"  {'PASS' if ok else 'FAIL'}: Config hash unchanged")
    if not ok:
        all_pass = False

    # === Mandatory completion matrix ===
    print("\n--- Mandatory Completion Matrix ---")
    ok = check_mandatory_completion_matrix("results/tables/mandatory_completion_matrix.tsv")
    checks.append(("mandatory_matrix_all_pass", ok))
    print(f"  {'PASS' if ok else 'FAIL'}: All mandatory components PASS")
    if not ok:
        all_pass = False

    # === Bootstrap ===
    print("\n--- Bootstrap ---")
    ok, n = check_parquet_nonempty("results/bootstrap/bootstrap_results.parquet")
    checks.append(("bootstrap_complete", ok))
    print(f"  {'PASS' if ok else 'FAIL'}: Bootstrap results ({n} rows)")
    if not ok:
        all_pass = False

    # === Stratified ===
    print("\n--- Stratified Analysis ---")
    ok, n = check_parquet_nonempty("results/stratified/stratified_metrics.parquet")
    checks.append(("stratified_complete", ok))
    print(f"  {'PASS' if ok else 'FAIL'}: Stratified metrics ({n} rows)")
    if not ok:
        all_pass = False

    # === Failure analysis ===
    print("\n--- Failure Analysis ---")
    ok, n = check_parquet_nonempty("results/failures/error_cases.parquet")
    checks.append(("failure_analysis_complete", ok))
    print(f"  {'PASS' if ok else 'FAIL'}: Failure analysis ({n} rows)")
    if not ok:
        all_pass = False

    # === Summary ===
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok in checks if ok)
    n_total = len(checks)
    print(f"Final QC: {n_pass}/{n_total} checks passed")
    print("=" * 60)

    if all_pass:
        print("FINAL QC: PASS")
        
        # Create SUCCESS file
        success_path = output_dir / "SUCCESS"
        with open(success_path, "w") as f:
            f.write("V2G-Benchmark-OneShot completed successfully.\n")
        
        # Copy core tables to release
        import shutil
        for src, dst in [
            ("data/processed/candidate_v2g.parquet", "benchmark_registry.parquet"),
            ("data/processed/all_model_predictions.parquet", "model_predictions.parquet"),
            ("results/metrics/ranking_metrics.tsv", "metrics.tsv"),
            ("results/tables/SupplementaryTables.xlsx", "SupplementaryTables.xlsx"),
        ]:
            if Path(src).exists():
                shutil.copy2(src, output_dir / dst)

        # Write model registry
        with open("config/models.yaml") as f:
            models = yaml.safe_load(f)
        with open(output_dir / "model_registry.tsv", "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["model_id", "family", "mode", "enabled"])
            for mid, minfo in models.get("models", {}).items():
                writer.writerow([mid, minfo.get("family", ""), minfo.get("mode", ""), minfo.get("enabled", "")])

        # Write dataset registry
        with open("config/datasets.yaml") as f:
            datasets = yaml.safe_load(f)
        with open(output_dir / "dataset_registry.tsv", "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["dataset_id", "track", "type", "source", "adapter"])
            for did, dinfo in datasets.get("datasets", {}).items():
                writer.writerow([did, dinfo.get("track", ""), dinfo.get("type", ""), dinfo.get("source", ""), dinfo.get("adapter", "")])

        # Write software versions
        import subprocess
        with open(output_dir / "software_versions.tsv", "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["software", "version"])
            for cmd in ["python3", "snakemake", "bcftools", "bedtools", "samtools"]:
                try:
                    v = subprocess.check_output([cmd, "--version"], text=True, stderr=subprocess.DEVNULL).split("\n")[0]
                    writer.writerow([cmd, v])
                except Exception:
                    writer.writerow([cmd, "unknown"])
            for pkg in ["polars", "pyarrow", "numpy", "scipy", "sklearn", "xgboost"]:
                try:
                    mod = __import__(pkg)
                    writer.writerow([pkg, getattr(mod, "__version__", "unknown")])
                except Exception:
                    writer.writerow([pkg, "not_installed"])

        # Write data checksums
        with open(output_dir / "data_checksums.tsv", "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["file", "sha256"])
            for p in Path("data/processed").glob("*.parquet"):
                hasher = hashlib.sha256()
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(8192), b""):
                        hasher.update(chunk)
                writer.writerow([str(p), hasher.hexdigest()])

        # Copy provenance lock
        if Path("data/locked/provenance.lock.yaml").exists():
            shutil.copy2("data/locked/provenance.lock.yaml", output_dir / "provenance.lock.yaml")

        # Write README
        with open(output_dir / "README.txt", "w") as f:
            f.write("V2G-Benchmark-OneShot v1.0\n")
            f.write("=" * 40 + "\n\n")
            f.write("This release contains the complete benchmark results.\n\n")
            try:
                preds = pl.read_parquet("data/processed/all_model_predictions.parquet")
                evidence = pl.read_parquet("data/processed/evidence_long.parquet")
                candidates = pl.read_parquet("data/processed/candidate_1m.parquet")
                f.write(f"N variants: {candidates['variant_id'].n_unique()}\n")
                f.write(f"N enhancer-gene pairs: {len(candidates)}\n")
                f.write(f"N genes: {candidates['gene_id'].n_unique()}\n")
                f.write(f"N contexts: {candidates['context_id'].n_unique()}\n")
                f.write(f"N models: {preds['model_id'].n_unique()}\n")
            except Exception:
                f.write("Statistics: (could not read)\n")

        print(f"\nSUCCESS file created: {success_path}")
        return 0
    else:
        print("FINAL QC: FAIL")
        failed = [name for name, ok in checks if not ok]
        print(f"Failed checks: {', '.join(failed)}")
        print("PROJECT STATUS = INCOMPLETE")
        return 1


if __name__ == "__main__":
    sys.exit(main())
