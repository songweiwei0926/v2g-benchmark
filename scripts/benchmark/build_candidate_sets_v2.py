#!/usr/bin/env python3
"""Build candidate sets — chromosome-by-chromosome, batch-by-batch.

Memory-safe: processes at most `batch_size` variants at a time per chromosome,
so peak memory is bounded by batch_size × genes_per_chrom × row_size.

Gold annotation is context-agnostic: a (variant_id, gene_id) pair is marked
is_gold=1 if it appears in the gold registry in ANY context.

Output: one parquet per window (candidate_250k, candidate_500k, candidate_1m).
The primary window (1Mb) is also copied to --main-output.
"""

from __future__ import annotations
import argparse, gc, shutil
from pathlib import Path
import polars as pl
from v2gbench.io.parquet import read_parquet, write_parquet, write_tsv
from v2gbench.utils.config import load_config

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def _find_root():
    cur = Path.cwd()
    while cur != cur.parent:
        if (cur / "config" / "project.yaml").exists():
            return cur
        cur = cur.parent
    return Path.cwd()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", required=True)
    parser.add_argument("--gene-master", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--main-output", required=True)
    parser.add_argument("--coverage-report", required=True)
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()

    root = _find_root()
    project_cfg = load_config(root / "config" / "project.yaml")
    bench_cfg = project_cfg.get("benchmark", {})
    windows = list(bench_cfg.get("sensitivity_windows", [250_000, 500_000, 1_000_000]))
    primary_window = int(bench_cfg.get("primary_window", 1_000_000))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_candidate_tmp"
    tmp_dir.mkdir(exist_ok=True)

    # --- Gold keys (context-agnostic) ---
    gold_df = read_parquet(args.gold)
    gold_keys = (
        gold_df.filter(pl.col("variant_id").is_not_null())
        .select(["variant_id", "gene_id"])
        .unique()
        .with_columns(pl.lit(1).cast(pl.Int64).alias("is_gold"))
    )
    n_gold_pairs = gold_keys.height
    print(f"[v2] Gold variant-gene pairs (context-agnostic): {n_gold_pairs}")
    del gold_df
    gc.collect()

    # --- Genes ---
    genes_df = read_parquet(args.gene_master).select(["gene_id", "chrom", "tss"])
    print(f"[v2] Genes: {genes_df.height}")

    # --- Variants (PASS only) ---
    variants_df = read_parquet(args.variants).filter(pl.col("qc_status") == "PASS")
    print(f"[v2] PASS variants: {variants_df.height}")

    batch_size = args.batch_size

    for window in windows:
        suffix = (
            f"{window // 1_000_000}m"
            if window >= 1_000_000 and window % 1_000_000 == 0
            else f"{window // 1000}k"
        )
        window_tmp = tmp_dir / f"win_{suffix}"
        window_tmp.mkdir(exist_ok=True)

        print(f"\n[v2] === window={window} ({suffix}) ===")

        for chrom in CHROMS:
            chrom_vars = variants_df.filter(pl.col("chrom") == chrom)
            if chrom_vars.height == 0:
                continue
            chrom_genes = genes_df.filter(pl.col("chrom") == chrom)
            if chrom_genes.height == 0:
                continue

            n_batches = (chrom_vars.height + batch_size - 1) // batch_size
            chrom_parts = []

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, chrom_vars.height)
                batch = chrom_vars[start:end]

                # Join with genes on same chrom, filter by distance
                joined = (
                    batch.select(["variant_id", "chrom", "pos"])
                    .join(chrom_genes, on="chrom", how="inner")
                    .with_columns(
                        (pl.col("pos") - pl.col("tss")).abs().alias("distance_to_tss")
                    )
                    .filter(pl.col("distance_to_tss") <= window)
                )

                if joined.height == 0:
                    continue

                # Rank by distance within each variant
                joined = joined.sort(["variant_id", "distance_to_tss", "gene_id"]).with_columns(
                    pl.col("distance_to_tss")
                    .rank("ordinal")
                    .over("variant_id")
                    .cast(pl.Int64)
                    .alias("distance_rank"),
                )

                # Gold annotation (context-agnostic: variant_id + gene_id)
                joined = joined.join(
                    gold_keys, on=["variant_id", "gene_id"], how="left"
                ).with_columns(
                    pl.when(pl.col("is_gold").is_null())
                    .then(0)
                    .otherwise(pl.col("is_gold"))
                    .cast(pl.Int64)
                    .alias("is_gold"),
                )

                # Add remaining columns
                joined = joined.with_columns(
                    (pl.col("distance_rank") == 1).alias("is_nearest"),
                    pl.lit("CONTEXT_TESTED").alias("candidate_basis"),
                    pl.lit("default").alias("context_id"),
                    pl.lit(1.0).cast(pl.Float64).alias("gold_confidence"),
                    (pl.lit(f"candidate_{window}_") + pl.col("variant_id")).alias(
                        "candidate_set_id"
                    ),
                ).select([
                    "candidate_set_id", "variant_id", "gene_id", "context_id",
                    "distance_to_tss", "distance_rank", "is_nearest",
                    "is_gold", "gold_confidence", "candidate_basis",
                ])

                # Write batch to file
                batch_file = window_tmp / f"{chrom}_{batch_idx:05d}.parquet"
                write_parquet(joined, batch_file)
                chrom_parts.append(batch_file)

                del joined
                gc.collect()

            n_chrom_cand = (
                sum(
                    pl.scan_parquet(f).select(pl.len()).collect().item()
                    for f in chrom_parts
                )
                if chrom_parts
                else 0
            )
            n_chrom_gold = (
                sum(
                    pl.scan_parquet(f).select(pl.col("is_gold").sum()).collect().item()
                    for f in chrom_parts
                )
                if chrom_parts
                else 0
            )
            print(
                f"  {chrom}: {chrom_vars.height} vars, {chrom_genes.height} genes, "
                f"{n_chrom_cand} candidates, {n_chrom_gold} gold, {len(chrom_parts)} batches"
            )

        # Merge all batch files for this window
        all_files = sorted(window_tmp.glob("*.parquet"))
        print(f"[v2] Merging {len(all_files)} batch files for window {suffix}...")

        out_path = output_dir / f"candidate_{suffix}.parquet"
        if all_files:
            merged_lf = pl.scan_parquet([str(f) for f in all_files])
            merged_lf.sink_parquet(str(out_path))
        else:
            pl.DataFrame(
                schema={
                    "candidate_set_id": pl.Utf8,
                    "variant_id": pl.Utf8,
                    "gene_id": pl.Utf8,
                    "context_id": pl.Utf8,
                    "distance_to_tss": pl.Int64,
                    "distance_rank": pl.Int64,
                    "is_nearest": pl.Boolean,
                    "is_gold": pl.Int64,
                    "gold_confidence": pl.Float64,
                    "candidate_basis": pl.Utf8,
                }
            ).write_parquet(out_path)

        # Verify
        stats = pl.scan_parquet(out_path).select(
            pl.len().alias("n_candidates"),
            pl.col("is_gold").sum().alias("n_gold"),
        ).collect()
        n_cand = stats["n_candidates"][0]
        n_gold = stats["n_gold"][0]
        print(f"[v2] {out_path.name}: {n_cand} candidates, {n_gold} gold")

        if window == primary_window:
            shutil.copy2(out_path, args.main_output)
            print(f"[v2] Copied to main: {args.main_output}")

        # Clean up batch files
        for f in all_files:
            f.unlink(missing_ok=True)
        gc.collect()

    # Clean up temp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # --- Coverage report ---
    print("\n[v2] Coverage report ...")
    gold_all = read_parquet(args.gold)
    gold_var_agg = (
        gold_all.filter(pl.col("variant_id").is_not_null())
        .select(["variant_id", "gene_id"])
        .unique()
    )
    total_gold = gold_var_agg.height
    del gold_all
    gc.collect()

    coverage_rows = []
    for window in windows:
        suffix = (
            f"{window // 1_000_000}m"
            if window >= 1_000_000 and window % 1_000_000 == 0
            else f"{window // 1000}k"
        )
        cand_path = output_dir / f"candidate_{suffix}.parquet"
        if not cand_path.exists():
            continue
        cov = (
            pl.scan_parquet(cand_path)
            .filter(pl.col("is_gold") == 1)
            .select(["variant_id", "gene_id"])
            .unique()
            .collect()
        )
        covered = cov.height
        coverage = covered / total_gold if total_gold > 0 else 1.0
        n_cand = pl.scan_parquet(cand_path).select(pl.len()).collect().item()
        n_gold = pl.scan_parquet(cand_path).select(pl.col("is_gold").sum()).collect().item()
        coverage_rows.append({
            "window_bp": window,
            "n_candidates": n_cand,
            "n_gold_candidates": n_gold,
            "gold_coverage": round(coverage, 6),
        })
        print(
            f"  window={window}: candidates={n_cand} gold={n_gold} "
            f"coverage={coverage:.4%} ({covered}/{total_gold})"
        )

    coverage_df = pl.DataFrame(coverage_rows)
    Path(args.coverage_report).parent.mkdir(parents=True, exist_ok=True)
    write_tsv(coverage_df, args.coverage_report)
    print(f"[v2] Wrote coverage report -> {args.coverage_report}")
    print("[v2] Done.")


if __name__ == "__main__":
    main()
