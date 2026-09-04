"""Generate small synthetic test fixtures for V2G-Benchmark smoke tests.

Produces deterministic Parquet files under ``tests/data/smoke/``:

* ``variants.parquet``       — 20 variants on chr22 (GRCh38:chr22:pos:ref:alt)
* ``gene_master.parquet``    — ~50 genes on chr22 with ENSG IDs and TSS
* ``candidates.parquet``     — ~100 candidate (variant, gene, context) pairs
* ``evidence.parquet``       — ~30 gold evidence rows (CRISPR, eQTL, GWAS mix)
* ``predictions.parquet``    — ~200 prediction rows (3–4 models)
* ``contexts.parquet``       — ~10 contexts

All generation uses seed **20260904** for full reproducibility.

Usage::

    python tests/data/smoke/generate_fixtures.py
    python tests/data/smoke/generate_fixtures.py --output-dir /tmp/smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

SEED = 20260904
CHROM = "chr22"
BUILD = "GRCh38"

POS_MIN = 17_000_000
POS_MAX = 50_000_000


def _make_variants(rng: np.random.Generator, n: int = 20) -> pl.DataFrame:
    """Generate ``n`` variants on chr22 with canonical variant IDs."""
    positions = sorted(rng.integers(POS_MIN, POS_MAX, size=n).tolist())
    bases = ["A", "C", "G", "T"]
    rows = []
    for pos in positions:
        ref = bases[rng.integers(0, 4)]
        alt = bases[rng.integers(0, 4)]
        while alt == ref:
            alt = bases[rng.integers(0, 4)]
        variant_id = f"{BUILD}:{CHROM}:{pos}:{ref}:{alt}"
        rows.append(
            {
                "variant_id": variant_id,
                "chrom": CHROM,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "genome_build": BUILD,
                "rsid": f"rs{rng.integers(1, 99_999_999)}",
            }
        )
    return pl.DataFrame(rows)


def _make_genes(rng: np.random.Generator, n: int = 50) -> pl.DataFrame:
    """Generate ``n`` genes on chr22 with de-versioned ENSG IDs and TSS."""
    rows = []
    used_positions = set()
    for i in range(n):
        gene_id = f"ENSG{i:011d}"
        start = int(rng.integers(POS_MIN, POS_MAX))
        while start in used_positions:
            start = int(rng.integers(POS_MIN, POS_MAX))
        used_positions.add(start)
        end = start + int(rng.integers(5_000, 200_000))
        strand = "+" if rng.random() > 0.5 else "-"
        tss = start if strand == "+" else end
        gene_type = str(rng.choice(
            ["protein_coding", "lncRNA", "miRNA", "pseudogene", "snoRNA"]
        ))
        rows.append(
            {
                "gene_id": gene_id,
                "gene_symbol": f"G{i}",
                "chrom": CHROM,
                "start": start,
                "end": end,
                "strand": strand,
                "tss": tss,
                "gene_type": gene_type,
                "canonical_transcript": f"ENST{i:011d}",
                "exon_intervals": None,
            }
        )
    return pl.DataFrame(rows).sort("start")


def _make_contexts(rng: np.random.Generator, n: int = 10) -> pl.DataFrame:
    """Generate ``n`` contexts with ontology IDs."""
    cell_types = [
        ("K562", "cell_line", "CL:CCL_243"),
        ("HepG2", "cell_line", "CL:0000372"),
        ("GM12878", "cell_line", "CL:0000477"),
        ("heart", "tissue", "UBERON:0000948"),
        ("liver", "tissue", "UBERON:0002107"),
        ("brain", "tissue", "UBERON:0000955"),
        ("CD4 T cell", "primary_cell", "CL:0000624"),
        ("CD8 T cell", "primary_cell", "CL:0000625"),
        ("iPSC", "in_vitro", "CL:0002364"),
        ("pancreas", "organoid", "UBERON:0001264"),
    ]
    rows = []
    for i in range(n):
        name, ctx_type, ont = cell_types[i % len(cell_types)]
        ctx_id = name.strip().lower().replace(" ", "_").replace("-", "_")
        rows.append(
            {
                "context_id": ctx_id,
                "context_name": name,
                "context_type": ctx_type,
                "ontology_id": ont,
                "parent_context": None,
                "supergroup": str(rng.choice(["blood", "tissue", "cell_line"])),
            }
        )
    return pl.DataFrame(rows)


def _make_candidates(
    rng: np.random.Generator,
    variants_df: pl.DataFrame,
    genes_df: pl.DataFrame,
    contexts_df: pl.DataFrame,
    window: int = 1_000_000,
    n_target: int = 100,
) -> pl.DataFrame:
    """Generate ~100 candidate (variant, gene, context) pairs within window."""
    variants = variants_df.to_dicts()
    genes = genes_df.to_dicts()
    contexts = contexts_df["context_id"].to_list()

    rows = []
    seen = set()
    attempts = 0
    while len(rows) < n_target and attempts < n_target * 20:
        attempts += 1
        v = variants[rng.integers(0, len(variants))]
        g = genes[rng.integers(0, len(genes))]
        ctx = contexts[rng.integers(0, len(contexts))]
        dist = abs(v["pos"] - g["tss"])
        if dist > window:
            continue
        key = (v["variant_id"], g["gene_id"], ctx)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "candidate_set_id": f"candidate_{window}_{v['variant_id']}",
                "variant_id": v["variant_id"],
                "gene_id": g["gene_id"],
                "context_id": ctx,
                "distance_to_tss": dist,
                "distance_rank": None,
                "is_nearest": False,
                "is_gold": 0,
                "gold_confidence": None,
                "candidate_basis": "CONTEXT_TESTED",
            }
        )

    schema = {
        "candidate_set_id": pl.Utf8, "variant_id": pl.Utf8, "gene_id": pl.Utf8,
        "context_id": pl.Utf8, "distance_to_tss": pl.Int64, "distance_rank": pl.Int64,
        "is_nearest": pl.Boolean, "is_gold": pl.Int64, "gold_confidence": pl.Float64,
        "candidate_basis": pl.Utf8,
    }
    df = pl.DataFrame(rows, schema=schema, infer_schema_length=None)
    if df.height == 0:
        return df

    df = df.sort(["variant_id", "context_id", "distance_to_tss", "gene_id"])
    df = df.with_columns(
        pl.col("distance_to_tss")
        .rank("ordinal")
        .over("variant_id", "context_id")
        .cast(pl.Int64)
        .alias("distance_rank")
    )
    df = df.with_columns(
        (pl.col("distance_rank") == 1).alias("is_nearest")
    )
    return df


def _make_evidence(
    rng: np.random.Generator,
    variants_df: pl.DataFrame,
    genes_df: pl.DataFrame,
    contexts_df: pl.DataFrame,
    candidates_df: pl.DataFrame,
    n: int = 30,
) -> pl.DataFrame:
    """Generate ~30 gold evidence rows with a mix of CRISPR, eQTL, GWAS."""
    cand = candidates_df.to_dicts()
    evidence_types = ["CRISPRi", "eQTL", "GWAS", "PerturbSeq", "curated_L2G"]
    rows = []
    for i in range(n):
        c = cand[rng.integers(0, len(cand))]
        etype = evidence_types[i % len(evidence_types)]
        pip = float(rng.uniform(0.5, 1.0)) if etype == "eQTL" else None
        effect_dir = str(rng.choice(["up", "down", "none", "unknown"]))
        rows.append(
            {
                "benchmark_id": "smoke_benchmark",
                "evidence_id": f"ev_{i:04d}",
                "variant_id": c["variant_id"],
                "element_id": None,
                "gene_id": c["gene_id"],
                "context_id": c["context_id"],
                "trait_id": f"trait_{i % 5}",
                "evidence_type": etype,
                "label": 1,
                "effect_size": float(rng.uniform(-2, 2)) if etype != "GWAS" else None,
                "effect_direction": effect_dir,
                "pip": pip,
                "pvalue": float(rng.uniform(0, 1e-5)) if etype == "GWAS" else None,
                "source_dataset": f"dataset_{etype.lower()}",
                "source_publication": f"pub_{i % 3}",
                "confidence": float(rng.uniform(0.8, 1.0)),
                "training_overlap": "NO_KNOWN_OVERLAP",
            }
        )
    schema = {
        "benchmark_id": pl.Utf8, "evidence_id": pl.Utf8, "variant_id": pl.Utf8,
        "element_id": pl.Utf8, "gene_id": pl.Utf8, "context_id": pl.Utf8,
        "trait_id": pl.Utf8, "evidence_type": pl.Utf8, "label": pl.Int64,
        "effect_size": pl.Float64, "effect_direction": pl.Utf8, "pip": pl.Float64,
        "pvalue": pl.Float64, "source_dataset": pl.Utf8, "source_publication": pl.Utf8,
        "confidence": pl.Float64, "training_overlap": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema, infer_schema_length=None)


def _make_predictions(
    rng: np.random.Generator,
    candidates_df: pl.DataFrame,
    evidence_df: pl.DataFrame,
    n_models: int = 4,
) -> pl.DataFrame:
    """Generate ~200 prediction rows for 3-4 models with ranking_score + is_gold."""
    models = [
        ("random", "baseline", "derived_baseline"),
        ("nearest_tss", "baseline", "derived_baseline"),
        ("abc", "e2g", "published_prediction"),
        ("borzoi", "sequence", "local_inference"),
    ][:n_models]

    gold_keys = set()
    for row in evidence_df.iter_rows(named=True):
        gold_keys.add((row["variant_id"], row["gene_id"], row["context_id"]))

    cand = candidates_df.to_dicts()
    rows = []
    for model_id, family, mode in models:
        for c in cand:
            key = (c["variant_id"], c["gene_id"], c["context_id"])
            is_gold = 1 if key in gold_keys else 0
            base = rng.uniform(0.0, 0.5)
            if is_gold:
                base += rng.uniform(0.2, 0.5)
            ranking_score = float(min(base, 1.0))
            signed = float(rng.uniform(-1, 1)) if family == "sequence" else None
            rows.append(
                {
                    "model_id": model_id,
                    "model_family": family,
                    "benchmark_id": "smoke_benchmark",
                    "variant_id": c["variant_id"],
                    "element_id": None,
                    "gene_id": c["gene_id"],
                    "context_id": c["context_id"],
                    "raw_score": ranking_score,
                    "ranking_score": ranking_score,
                    "signed_score": signed,
                    "coverage": 1,
                    "applicability": "APPLICABLE",
                    "source_mode": mode,
                    "is_gold": is_gold,
                    "distance_to_tss": c["distance_to_tss"],
                }
            )
    schema = {
        "model_id": pl.Utf8, "model_family": pl.Utf8, "benchmark_id": pl.Utf8,
        "variant_id": pl.Utf8, "element_id": pl.Utf8, "gene_id": pl.Utf8,
        "context_id": pl.Utf8, "raw_score": pl.Float64, "ranking_score": pl.Float64,
        "signed_score": pl.Float64, "coverage": pl.Int64, "applicability": pl.Utf8,
        "source_mode": pl.Utf8, "is_gold": pl.Int64, "distance_to_tss": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema, infer_schema_length=None)


def generate_all(output_dir: Path) -> dict[str, Path]:
    """Generate all smoke fixtures and return a mapping of name -> path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    variants_df = _make_variants(rng, n=20)
    genes_df = _make_genes(rng, n=50)
    contexts_df = _make_contexts(rng, n=10)
    candidates_df = _make_candidates(rng, variants_df, genes_df, contexts_df, n_target=100)
    evidence_df = _make_evidence(rng, variants_df, genes_df, contexts_df, candidates_df, n=30)
    predictions_df = _make_predictions(rng, candidates_df, evidence_df, n_models=4)

    paths = {}
    for name, df in [
        ("variants", variants_df),
        ("gene_master", genes_df),
        ("candidates", candidates_df),
        ("evidence", evidence_df),
        ("predictions", predictions_df),
        ("contexts", contexts_df),
    ]:
        p = output_dir / f"{name}.parquet"
        df.write_parquet(p, compression="zstd")
        paths[name] = p

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V2G-Benchmark smoke fixtures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to write fixture Parquet files (default: this directory)",
    )
    args = parser.parse_args()

    paths = generate_all(args.output_dir)
    print(f"Generated {len(paths)} smoke fixtures in {args.output_dir}:")
    for name, p in paths.items():
        df = pl.read_parquet(p)
        print(f"  {name:15s} -> {p}  ({df.height} rows, {df.width} cols)")


if __name__ == "__main__":
    main()
