# V2G-Benchmark-OneShot v1.0 — Main Snakefile
# Orchestrates the complete pipeline from preflight to release.

import yaml
from pathlib import Path

# Load project config
with open("config/project.yaml") as f:
    PROJECT = yaml.safe_load(f)

with open("config/site.yaml") as f:
    SITE = yaml.safe_load(f)

with open("config/models.yaml") as f:
    MODELS = yaml.safe_load(f)

with open("config/datasets.yaml") as f:
    DATASETS = yaml.safe_load(f)

# Seed
SEED = PROJECT["project"]["seed"]

# Include all rule modules
include: "workflow/rules/preflight.smk"
include: "workflow/rules/download.smk"
include: "workflow/rules/reference.smk"
include: "workflow/rules/harmonize.smk"
include: "workflow/rules/benchmarks.smk"
include: "workflow/rules/models_published.smk"
include: "workflow/rules/models_sequence.smk"
include: "workflow/rules/evaluate.smk"
include: "workflow/rules/ensemble.smk"
include: "workflow/rules/figures.smk"

# === Top-level rule ===
rule all:
    input:
        # Final release outputs
        "results/release/SUCCESS",
        "results/release/run_manifest.json",
        "results/release/provenance.lock.yaml",
        "results/release/software_versions.tsv",
        "results/release/data_checksums.tsv",
        "results/release/model_registry.tsv",
        "results/release/dataset_registry.tsv",
        "results/release/benchmark_registry.parquet",
        "results/release/model_predictions.parquet",
        "results/release/metrics.tsv",
        "results/release/SupplementaryTables.xlsx",
        # Figures
        "results/figures/fig1_overview.svg",
        "results/figures/fig2_heatmap.svg",
        "results/figures/fig3_distance.svg",
        "results/figures/fig4_context.svg",
        "results/figures/fig5_complementarity.svg",
        "results/figures/fig6_integrated.svg",
        # Tables
        "results/tables/main_results.tsv",
        "results/tables/model_exclusions.tsv",
        "results/tables/mandatory_completion_matrix.tsv",
        "results/tables/model_score_qc.tsv",
        "results/tables/model_context_matrix.tsv",
        # Core data tables
        "results/release/benchmark_registry.parquet",
        "data/processed/candidate_v2g.parquet",
        "data/processed/all_model_predictions.parquet",
