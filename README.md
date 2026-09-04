# V2G-Benchmark: Variant-to-Gene Prediction Benchmark

A comprehensive benchmark for evaluating variant-to-gene (V2G) prediction methods
across multiple gold-standard datasets, model categories, and evaluation tracks.

## Overview

V2G-Benchmark systematically evaluates methods that predict which gene(s) a given
non-coding variant regulates. It integrates six gold-standard datasets, ~25 models
(baselines, published methods, sequence models, and integrated ensembles), and five
evaluation tracks (A–E) with rigorous statistics.

### Gold-standard datasets
- **GTEx eQTL SuSiE** — fine-mapped cis-eQTLs across 49 tissues
- **eQTL Catalogue** — community-aggregated eQTL fine-mapping
- **CRISPR comparison** (Engreitz Lab) — direct enhancer-gene CRISPR validations
- **GWAS E2G benchmarking** (Engreitz Lab) — GWAS-locus CRISPR screens
- **Open Targets genetics gold standards** — curated causal gene assignments
- **pgBoost** — GWAS fine-mapping with constituent method scores

### Model categories
- **Baselines**: random, nearest TSS, inverse/distance-decay, nearest expressed
- **Published**: ABC, rE2G, scE2G, EpiMap, GraphReg, pgBoost, SCENT, Signac, ArchR,
  Cicero, OpenTargets L2G
- **Sequence models**: Borzoi (4-replicate ensemble), Enformer (CAGE tracks),
  AlphaGenome (API)
- **Integrated**: rank-mean, logistic regression, XGBoost (chromosome-fold CV)

### Evaluation tracks
- **Track A**: Full benchmark — all models, all gold-standard pairs
- **Track B**: CRISPR-only — subset restricted to CRISPR-validated loci
- **Track C**: eQTL-only — subset restricted to eQTL fine-mapping loci
- **Track D**: SEQ_CORE — ≤5000 variant-context pairs, deterministic sampling
- **Track E**: Cross-context — model generalization across cell types

## Installation

```bash
# Clone
git clone https://github.com/songweiwei0926/v2g-benchmark.git
cd v2g-benchmark

# Create environment
conda env create -f environment.yml
conda activate v2gbench

# Install package
pip install -e .

# Set up credentials
cp .env.example .env
# Edit .env with your API keys:
#   ALPHAGENOME_API_KEY=...
#   SYNAPSE_AUTH_TOKEN=...
#   GITHUB_TOKEN=...
```

## Usage

### Full pipeline (one command)

```bash
bash run_once.sh
```

This runs the complete pipeline:
1. **Preflight** — environment check, config hash, resource discovery
2. **Download** — all reference and gold-standard datasets
3. **Harmonize** — variant normalization, gene mapping, context mapping
4. **Benchmark** — gold registry, candidate sets, leakage, applicability, SEQ_CORE
5. **Models** — score all baselines, published, sequence, and integrated models
6. **Evaluate** — ranking, classification, direction, bootstrap, stratified, failures
7. **Figures** — 6 main figures + supplementary figures
8. **Tables** — main results, exclusions, completion matrix, supplementary tables
9. **Final QC** — generates `results/release/SUCCESS`

### Individual steps

```bash
# Dry run (check all rules resolve)
snakemake -n

# Run a specific stage
snakemake --cores 8 results/metrics/ranking_metrics.tsv

# Run preflight only
python scripts/preflight/preflight.py --config config/site.yaml
```

## Configuration

All parameters are in `config/`:
- `project.yaml` — seed, windows, PIP thresholds, bootstrap, folds, integrated features
- `site.yaml` — paths, credentials, execution mode, GPU config
- `datasets.yaml` — 12 dataset entries with URLs and adapters
- `models.yaml` — ~25 models, figure order, exclusions
- `benchmarks.yaml` — 5 tracks (A–E), 3 subsets
- `context_mapping.yaml` — 6-level context cascade
- `resources.yaml` — per-rule CPU/memory/time/GPU
- `plotting.yaml` — 6 main figures, 7 supplementary, 17 supplementary tables

Config files are hash-locked at run start. Any modification after run start = FAIL.

## Key design decisions

- **Seed**: 20260904 (all deterministic sampling, random baseline, bootstrap)
- **Variant ID**: `GRCh38:chr1:123456:A:G`
- **Gene ID**: Ensembl ID de-versioned (ENSG00000123456)
- **GENCODE**: v47; **Genome build**: GRCh38
- **PIP thresholds**: primary=0.90, sensitivity=[0.50, 0.70, 0.90, 0.95], negative=0.01
- **Candidate windows**: 250kb, 500kb, 1Mb (primary=1Mb)
- **SEQ_CORE**: max 5000 variant-context pairs, SHA256 deterministic sampling
- **Bootstrap**: 2000 replicates, sampling unit = variant/locus (not pair)
- **Tie-breaking**: score desc, distance asc, gene_id alphabetical
- **Missing predictions**: coverage=0, ranking_score=0 (not deleted)
- **Evaluation code is model-agnostic** — reads only model_id, ranking_score, gold

## Project structure

```
v2g-benchmark/
├── config/              # YAML configuration files
├── workflow/rules/      # Snakemake rules (.smk)
├── src/v2gbench/        # Core Python package
│   ├── schemas/         # Data schemas (variant, gene, context, evidence, ...)
│   ├── io/              # Parquet I/O, download utilities
│   ├── utils/           # Config, hashing, provenance, logging
│   ├── harmonize/       # Variant/gene/context normalization
│   ├── benchmark/       # Gold registry, candidates, leakage, applicability
│   ├── metrics/         # Ranking, classification, direction, effect size
│   ├── statistics/      # Bootstrap, paired tests, sampling
│   ├── models/          # Model adapters (baselines, published, sequence, integrated)
│   └── plotting/        # Figure and table generation
├── scripts/             # CLI scripts (preflight, download, harmonize, benchmark, models, evaluate, figures)
├── tests/               # Unit tests + smoke data fixtures
├── environment.yml      # Conda environment
├── pyproject.toml       # Package metadata
├── Snakefile            # Main workflow entry point
└── run_once.sh          # One-command pipeline execution
```

## License

MIT
