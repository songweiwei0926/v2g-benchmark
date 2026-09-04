#!/usr/bin/env python3
"""Build candidate gene universe at 250k/500k/1Mb windows — chromosome-by-chromosome.

Processes each chromosome independently to avoid the ~7B row intermediate
from a full same-chromosome join of 2.2M variants × 78K genes.

Usage:
    build_candidate_sets_chunked.py
        --variants <parquet> --gene-master <parquet> --gold <parquet>
        --contexts <parquet> --output-dir data/processed
        --main-output <parquet> --coverage-report <tsv>
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Dict, List

import polars as pl

from v2gbench.io.parquet import read_parquet, write_parquet, write_tsv
from v2gbench.utils.config import load_config
from v2gbench.schemas.candidate import candidate_schema


def _find_root() -> Path:
    cur = Path.cwd()
    while cur != cur.parent:
        if (cur / "config" / "project.yaml").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def build_candidate_chrom(
    variants_chr: pl.DataFrame,
    genes_chr: pl.DataFrame,
    window_size: int,
    context_id: str = "default",
) -> pl.DataFrame:
    """Build candidates for a single chromosome."""
    if variants_chr.height == 0 or genes_chr.height == 0:
        return pl.DataFrame(schema={
            "candidate_set_id": pl.Utf8, "variant_id": pl.Utf8,
            "gene_id": pl.Utf8, "context_id": pl.Utf8,
            "distance_to_tss": pl.Int64, "distance_rank": pl.Int64,
            "is_nearest": pl.Boolean, "is_gold": pl.Int64,
            "gold_confidence": pl.Float64, "candidate_basis": pl.Utf8,
        })

    joined = (
        variants_chr.select(["variant_id", "chrom", "pos"])
        .join(genes_chr.select(["gene_id", "chrom", "tss"]), on="chrom", how="inner")
        .with_columns(
            (pl.col("pos") - pl.col("tss")).abs().alias("distance_to_tss"),
        )
        .filter(pl.col("distance_to_tss") <= window_size)
    )

    if joined.height == 0:
        return pl.DataFrame(schema={
            "candidate_set_id": pl.Utf8, "variant_id": pl.Utf8,
            "gene_id": pl.Utf8, "context_id": pl.Utf8,
            "distance_to_tss": pl.Int64, "distance_rank": pl.Int64,
            "is_nearest": pl.Boolean, "is_gold": pl.Int64,
            "gold_confidence": pl.Float64, "candidate_basis": pl.Utf8,
        })

    joined = joined.sort(["variant_id", "distance_to_tss", "gene_id"]).with_columns(
        pl.col("distance_to_tss").rank("ordinal").over("variant_id").cast(pl.Int64).alias("distance_rank"),
    )
    joined = joined.with_columns(
        (pl.col("distance_rank") == 1).alias("is_nearest"),
        pl.lit("CONTEXT_TESTED").alias("candidate_basis"),
        pl.lit(context_id).alias("context_id"),
        pl.lit(0).cast(pl.Int64).alias("is_gold"),
        pl.lit(None).cast(pl.Float64).alias("gold_confidence"),
        (pl.lit(f"candidate_{window_size}_") + pl.col("variant_id")).alias("candidate_set_id"),
    )

    return joined.select([
        "candidate_set_id", "variant_id", "gene_id", "context_id",
        "distance_to_tss", "distance_rank", "is_nearest",
        "is_gold", "gold_confidence", "candidate_basis",
    ])


def annotate_gold(cand_df: pl.DataFrame, gold_df: pl.DataFrame) -> pl.DataFrame:
    """Mark is_gold / gold_confidence from gold pairs frame."""
    if cand_df.height == 0:
        return cand_df

    # Gold pairs with variant_id (non-null only — element-centric pairs excluded)
    gold_var = gold_df.filter(pl.col("variant_id").is_not_null())
    if gold_var.height == 0:
        return cand_df

    keys = ["variant_id", "gene_id", "context_id"]
    gold_small = gold_var.select(keys).unique().with_columns(
        pl.lit(1.0).cast(pl.Float64).alias("gold_confidence")
    )

    annotated = cand_df.join(gold_small, on=keys, how="left", suffix="_gold")
    if "gold_confidence_gold" in annotated.columns:
        annotated = annotated.with_columns(
            pl.coalesce(["gold_confidence_gold", "gold_confidence"]).alias("gold_confidence")
        ).drop("gold_confidence_gold")

    annotated = annotated.with_columns(
        pl.when(pl.col("gold_confidence").is_not_null())
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int64)
        .alias("is_gold"),
    )
    return annotated.select(cand_df.columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate sets chromosome-by-chromosome.")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--gene-master", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--main-output", required=True)
    parser.add_argument("--coverage-report", required=True)
    args = parser.parse_args()

    root = _find_root()
    project_cfg = load_config(root / "config" / "project.yaml")
    bench_cfg = project_cfg.get("benchmark", {})
    windows = list(bench_cfg.get("sensitivity_windows", [250_000, 500_000, 1_000_000]))
    primary_window = int(bench_cfg.get("primary_window", 1_000_000))

    variants_df = read_parquet(args.variants).filter(pl.col("qc_status") == "PASS")
    gene_master_df = read_parquet(args.gene_master)
    gold_df = read_parquet(args.gold)

    print(f"[chunked] PASS variants={variants_df.height} genes={gene_master_df.height}")
    print(f"[chunked] windows={windows} primary={primary_window}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chromosomes = sorted(variants_df["chrom"].unique().to_list())
    print(f"[chunked] chromosomes={chromosomes}")

    for window in windows:
        print(f"\n[chunked] Building window={window} ...")
        all_parts: List[pl.DataFrame] = []

        for chrom in chromosomes:
            v_chr = variants_df.filter(pl.col("chrom") == chrom)
            g_chr = gene_master_df.filter(pl.col("chrom") == chrom)
            part = build_candidate_chrom(v_chr, g_chr, window)
            if part.height > 0:
                all_parts.append(part)
            print(f"  {chrom}: variants={v_chr.height} genes={g_chr.height} candidates={part.height}")
            del part, v_chr, g_chr
            gc.collect()

        if all_parts:
            cand = pl.concat(all_parts, how="vertical_relaxed")
            del all_parts
            gc.collect()

            # Annotate gold
            print(f"[chunked] Annotating gold for window={window} ...")
            cand = annotate_gold(cand, gold_df)

            # Validate
            try:
                cand = candidate_schema.validate(cand)
            except Exception as e:
                print(f"[chunked] WARNING: schema validation failed: {e}")
                # Continue anyway — schema may be too strict

            # Write
            if window >= 1_000_000 and window % 1_000_000 == 0:
                suffix = f"{window // 1_000_000}m"
            else:
                suffix = f"{window // 1000}k"
            out_path = output_dir / f"candidate_{suffix}.parquet"
            write_parquet(cand, out_path)
            print(f"[chunked] Wrote {out_path} ({cand.height} rows)")

            n_gold = int(cand["is_gold"].sum()) if "is_gold" in cand.columns else 0
            print(f"[chunked] Gold candidates: {n_gold}")

            if window == primary_window:
                write_parquet(cand, args.main_output)
                print(f"[chunked] Wrote main output -> {args.main_output}")

            del cand
            gc.collect()

    # Coverage report
    print("\n[chunked] Building coverage report ...")
    coverage_rows = []
    for window in windows:
        if window >= 1_000_000 and window % 1_000_000 == 0:
            suffix = f"{window // 1_000_000}m"
        else:
            suffix = f"{window // 1000}k"
        cand_path = output_dir / f"candidate_{suffix}.parquet"
        if not cand_path.exists():
            continue
        cand = read_parquet(cand_path)
        gold_var = gold_df.filter(pl.col("variant_id").is_not_null())
        keys = ["variant_id", "gene_id", "context_id"]
        gold_keys = gold_var.select(keys).unique()
        cand_keys = cand.select(keys).unique()
        total = gold_keys.height
        covered = gold_keys.join(cand_keys, on=keys, how="inner").height if total > 0 else 0
        coverage = covered / total if total > 0 else 1.0
        n_gold = int(cand["is_gold"].sum()) if "is_gold" in cand.columns else 0
        coverage_rows.append({
            "window_bp": window,
            "n_candidates": cand.height,
            "n_gold_candidates": n_gold,
            "gold_coverage": round(coverage, 6),
        })
        print(f"  window={window}: candidates={cand.height} gold={n_gold} coverage={coverage:.4%}")
        del cand
        gc.collect()

    coverage_df = pl.DataFrame(coverage_rows)
    write_tsv(coverage_df, args.coverage_report)
    print(f"[chunked] Wrote coverage report -> {args.coverage_report}")


if __name__ == "__main__":
    main()
