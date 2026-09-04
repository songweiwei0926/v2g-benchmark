#!/usr/bin/env python3
"""Normalize variants from all raw datasets into a canonical parquet.

Reads variant-level data from GTEx, eQTL Catalogue, CRISPR, GWAS,
OpenTargets, and pgBoost datasets, extracts the canonical variant columns
(chrom, pos, ref, alt), and normalizes them using
``v2gbench.harmonize.variants.normalize_variant_df``.

Outputs:
  - ``variants_normalized.parquet`` — deduplicated canonical variants with
    ``variant_id``, ``chrom``, ``pos``, ``ref``, ``alt``, ``genome_build``,
    ``ref_match``, ``qc_status``, and ``source`` columns.
  - ``variant_qc_report.tsv`` — per-source QC summary.

CLI (Snakemake):
    python scripts/harmonize/normalize_variants.py \
        --gtex <path> --eqtl <path> --crispr <path> --gwas <path> \
        --opentargets <path> --pgboost <path> --fasta <path> \
        --output data/processed/variants_normalized.parquet \
        --qc-output data/processed/variant_qc_report.tsv
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import sys
import tarfile
from pathlib import Path
from typing import Optional

import polars as pl

from v2gbench.harmonize.variants import normalize_variant_df
from v2gbench.io.parquet import write_parquet, write_tsv, read_tsv

# ---------------------------------------------------------------------------
# Variant-ID parsing helpers
# ---------------------------------------------------------------------------

# GTEx variant_id format: chr1_12345_A_G_b38  (or without _b38 suffix)
_GTEX_VID_RE = re.compile(r"^(chr\w+)_(\d+)_([ACGTN]+)_([ACGTN]+)(?:_b\d+)?$")

# rsID pattern (used as fallback when no chr/pos/ref/alt available)
_RSID_RE = re.compile(r"^rs\d+$")


def _parse_gtex_variant_id(vid: str) -> Optional[dict]:
    """Parse a GTEx-style variant_id into chrom/pos/ref/alt."""
    m = _GTEX_VID_RE.match(str(vid))
    if not m:
        return None
    return {
        "chrom": m.group(1),
        "pos": int(m.group(2)),
        "ref": m.group(3),
        "alt": m.group(4),
    }


def _coerce_variant_cols(
    df: pl.DataFrame,
    chrom_col: str = "chrom",
    pos_col: str = "pos",
    ref_col: str = "ref",
    alt_col: str = "alt",
) -> pl.DataFrame:
    """Select and rename variant columns to canonical names."""
    rename_map = {chrom_col: "chrom", pos_col: "pos", ref_col: "ref", alt_col: "alt"}
    cols_present = [c for c in rename_map if c in df.columns]
    if len(cols_present) < 4:
        return pl.DataFrame()
    return df.select(cols_present).rename({c: rename_map[c] for c in cols_present})


# ---------------------------------------------------------------------------
# Per-dataset extractors
# ---------------------------------------------------------------------------

def _extract_gtex(gtex_path: Path) -> pl.DataFrame:
    """Extract variants from the GTEx v11 SuSiE tar archive.

    The tar contains per-tissue ``.parquet`` files with a ``variant_id``
    column.  We scan each member and collect unique variants.
    """
    if not gtex_path.exists():
        print(f"  GTEx: path not found — {gtex_path}", file=sys.stderr)
        return pl.DataFrame()

    print(f"  GTEx: extracting variants from tar ...")
    chunks: list[pl.DataFrame] = []

    with tarfile.open(gtex_path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            fname = member.name
            if not (fname.endswith(".parquet") or fname.endswith(".txt")
                    or fname.endswith(".tsv") or fname.endswith(".txt.gz")
                    or fname.endswith(".tsv.gz")):
                continue
            try:
                fobj = tar.extractfile(member)
                if fobj is None:
                    continue
                raw = fobj.read()
                if fname.endswith(".parquet"):
                    df = pl.read_parquet(io.BytesIO(raw))
                    if "variant_id" not in df.columns:
                        continue
                    # Extract tissue name from filename: Adipose_Subcutaneous.v11.eQTLs.SuSiE_summary.parquet
                    tissue = fname.split("/")[-1].split(".")[0]
                    # Vectorized parsing of variant_id: chr1_285155_A_C_b38
                    vids = df["variant_id"].unique()
                    parsed = vids.str.extract_groups(r"^(chr\w+)_(\d+)_([ACGTN]+)_([ACGTN]+)(?:_b\d+)?$")
                    if parsed is not None and parsed.len() > 0:
                        parsed_df = parsed.to_frame().unnest("variant_id")
                        parsed_df = parsed_df.rename({"1": "chrom", "2": "pos", "3": "ref", "4": "alt"})
                        parsed_df = parsed_df.with_columns(
                            pl.col("pos").cast(pl.Int64),
                            pl.lit(tissue).alias("context"),
                        ).filter(pl.col("chrom").is_not_null())
                        if parsed_df.height > 0:
                            chunks.append(parsed_df)
                else:
                    if fname.endswith(".gz"):
                        raw = gzip.decompress(raw)
                    text = raw.decode("utf-8", errors="replace")
                    lines = text.strip().splitlines()
                    if not lines:
                        continue
                    header = lines[0].split("\t")
                    if "variant_id" not in header:
                        continue
                    vid_idx = header.index("variant_id")
                    vids = set()
                    for line in lines[1:]:
                        fields = line.split("\t")
                        if len(fields) <= vid_idx:
                            continue
                        vids.add(fields[vid_idx])
                    if vids:
                        df_vids = pl.DataFrame({"vid": list(vids)})
                        parsed = df_vids["vid"].str.extract_groups(r"^(chr\w+)_(\d+)_([ACGTN]+)_([ACGTN]+)(?:_b\d+)?$")
                        if parsed is not None and parsed.len() > 0:
                            parsed_df = parsed.to_frame().unnest("vid")
                            parsed_df = parsed_df.rename({"1": "chrom", "2": "pos", "3": "ref", "4": "alt"})
                            parsed_df = parsed_df.with_columns(
                                pl.col("pos").cast(pl.Int64),
                            ).filter(pl.col("chrom").is_not_null())
                            if parsed_df.height > 0:
                                chunks.append(parsed_df)
            except Exception as exc:
                print(f"    Warning: failed to read {fname}: {exc}", file=sys.stderr)

    if not chunks:
        print(f"  GTEx: no variants extracted", file=sys.stderr)
        return pl.DataFrame()
    df = pl.concat(chunks).unique().with_columns(pl.lit("GTEx").alias("source"))
    print(f"  GTEx: {df.height} unique variants")
    return df


def _extract_eqtl_catalogue(eqtl_dir: Path) -> pl.DataFrame:
    """Extract variants from eQTL Catalogue TSV files."""
    if not eqtl_dir.exists():
        print(f"  eQTL Catalogue: directory not found — {eqtl_dir}", file=sys.stderr)
        return pl.DataFrame()

    print(f"  eQTL Catalogue: scanning {eqtl_dir} ...")
    chunks: list[pl.DataFrame] = []

    tsv_files = sorted(
        p for p in eqtl_dir.rglob("*")
        if p.is_file() and (p.name.endswith(".tsv") or p.name.endswith(".tsv.gz"))
    )

    for tf in tsv_files:
        try:
            kwargs = {}
            df = read_tsv(tf, n_rows=5, infer_schema_length=10000, **kwargs)
            cols = set(df.columns)
            chrom_col = next((c for c in ["chr", "chrom", "chromosome"] if c in cols), None)
            pos_col = next((c for c in ["position", "pos", "bp"] if c in cols), None)
            ref_col = next((c for c in ["ref", "reference"] if c in cols), None)
            alt_col = next((c for c in ["alt", "alternate"] if c in cols), None)
            if not all([chrom_col, pos_col, ref_col, alt_col]):
                # Try parsing from a variant column (format: chr1_108004887_G_T)
                vid_col = next((c for c in ["variant", "variant_id", "SNP"] if c in cols), None)
                if vid_col:
                    df_full = read_tsv(tf, columns=[vid_col], infer_schema_length=10000, **kwargs)
                    vids = df_full[vid_col].unique()
                    parsed = vids.str.extract_groups(r"^(chr\w+)_(\d+)_([ACGTN]+)_([ACGTN]+)$")
                    if parsed is not None and parsed.len() > 0:
                        parsed_df = parsed.to_frame().unnest(vid_col)
                        parsed_df = parsed_df.rename({"1": "chrom", "2": "pos", "3": "ref", "4": "alt"})
                        parsed_df = parsed_df.with_columns(pl.col("pos").cast(pl.Int64))
                        parsed_df = parsed_df.filter(pl.col("chrom").is_not_null())
                        if parsed_df.height > 0:
                            study = tf.parent.parent.name  # QTS000001
                            parsed_df = parsed_df.with_columns(pl.lit(study).alias("context"))
                            chunks.append(parsed_df)
                continue
            df_full = read_tsv(tf, columns=[chrom_col, pos_col, ref_col, alt_col], infer_schema_length=10000, **kwargs)
            df_full = df_full.rename({chrom_col: "chrom", pos_col: "pos", ref_col: "ref", alt_col: "alt"})
            study = tf.parent.parent.name
            df_full = df_full.with_columns(pl.lit(study).alias("context"))
            chunks.append(df_full)
        except Exception as exc:
            print(f"    Warning: failed to read {tf.name}: {exc}", file=sys.stderr)

    if not chunks:
        print(f"  eQTL Catalogue: no variants extracted", file=sys.stderr)
        return pl.DataFrame()
    df = pl.concat(chunks).unique().with_columns(pl.lit("eQTL_Catalogue").alias("source"))
    print(f"  eQTL Catalogue: {df.height} unique variants")
    return df


def _extract_crispr(crispr_dir: Path) -> pl.DataFrame:
    """Extract variants from the CRISPR comparison ensemble TSV."""
    if not crispr_dir.exists():
        print(f"  CRISPR: directory not found — {crispr_dir}", file=sys.stderr)
        return pl.DataFrame()

    # Find the key file.
    key_file = crispr_dir / "resources" / "crispr_data" / "EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"
    if not key_file.exists():
        # Search for any EPCrisprBenchmark TSV.
        candidates = list(crispr_dir.rglob("EPCrisprBenchmark_*.tsv.gz"))
        if not candidates:
            print(f"  CRISPR: key TSV not found under {crispr_dir}", file=sys.stderr)
            return pl.DataFrame()
        key_file = candidates[0]

    print(f"  CRISPR: reading {key_file.name} ...")
    try:
        kwargs = {}
        df = read_tsv(key_file, infer_schema_length=10000, **kwargs)
    except Exception as exc:
        print(f"  CRISPR: failed to read {key_file}: {exc}", file=sys.stderr)
        return pl.DataFrame()

    cols = set(df.columns)
    # CRISPR ensemble data may have variant columns under various names.
    chrom_col = next((c for c in ["chr", "chrom", "chromosome", "Chr"] if c in cols), None)
    pos_col = next((c for c in ["pos", "position", "Pos", "variantPos", "chromStart"] if c in cols), None)
    ref_col = next((c for c in ["ref", "Ref", "reference"] if c in cols), None)
    alt_col = next((c for c in ["alt", "Alt", "alternate"] if c in cols), None)

    if not all([chrom_col, pos_col, ref_col, alt_col]):
        # Try parsing from a variant_id column.
        vid_col = next((c for c in ["variant_id", "variant", "SNP"] if c in cols), None)
        if vid_col:
            parsed = df[vid_col].to_list()
            rows = []
            for vid in parsed:
                p = _parse_gtex_variant_id(vid)
                if p:
                    rows.append(p)
            if rows:
                df = pl.DataFrame(rows).with_columns(pl.lit("CRISPR").alias("source"))
                print(f"  CRISPR: {len(rows)} variants from variant_id column")
                return df
        print(f"  CRISPR: no variant columns found in {key_file.name}", file=sys.stderr)
        return pl.DataFrame()

    sub = _coerce_variant_cols(df, chrom_col, pos_col, ref_col, alt_col)
    sub = sub.unique().with_columns(pl.lit("CRISPR").alias("source"))
    print(f"  CRISPR: {sub.height} unique variants")
    return sub


def _extract_gwas(gwas_dir: Path) -> pl.DataFrame:
    """Extract variants from the GWAS E2G benchmarking repo."""
    if not gwas_dir.exists():
        print(f"  GWAS: directory not found — {gwas_dir}", file=sys.stderr)
        return pl.DataFrame()

    print(f"  GWAS: scanning {gwas_dir} for variant TSVs ...")
    chunks: list[pl.DataFrame] = []

    tsv_files = sorted(
        p for p in gwas_dir.rglob("*")
        if p.is_file() and (p.name.endswith(".tsv") or p.name.endswith(".tsv.gz")
                            or p.name.endswith(".txt") or p.name.endswith(".txt.gz"))
    )

    for tf in tsv_files:
        try:
            kwargs = {}
            df = read_tsv(tf, n_rows=5, infer_schema_length=10000, **kwargs)
            cols = set(df.columns)
            chrom_col = next((c for c in ["chr", "chrom", "chromosome", "Chr", "#Chr"] if c in cols), None)
            # Prefer 'end' (1-based) when 'start' also exists (BED format);
            # GWAS variant.list.txt files have 0-based start/position and 1-based end.
            if "end" in cols and "start" in cols:
                pos_col = "end"
            else:
                pos_col = next((c for c in ["pos", "position", "Pos", "bp", "BP", "end", "start"] if c in cols), None)
            ref_col = next((c for c in ["ref", "Ref", "reference", "A1", "allele1"] if c in cols), None)
            alt_col = next((c for c in ["alt", "Alt", "alternate", "A2", "allele2"] if c in cols), None)
            if not all([chrom_col, pos_col, ref_col, alt_col]):
                continue
            df_full = read_tsv(tf, columns=[chrom_col, pos_col, ref_col, alt_col], infer_schema_length=10000, **kwargs)
            df_full = df_full.rename({chrom_col: "chrom", pos_col: "pos", ref_col: "ref", alt_col: "alt"})
            # If position came from 'start' (0-based BED), add 1 to make it 1-based
            if pos_col == "start":
                df_full = df_full.with_columns((pl.col("pos") + 1).alias("pos"))
            # Extract trait name from parent directory
            trait = tf.parent.name
            df_full = df_full.with_columns(pl.lit(trait).alias("context"))
            chunks.append(df_full)
        except Exception:
            continue

    if not chunks:
        print(f"  GWAS: no variants extracted", file=sys.stderr)
        return pl.DataFrame()
    df = pl.concat(chunks).unique().with_columns(pl.lit("GWAS").alias("source"))
    print(f"  GWAS: {df.height} unique variants")
    return df


def _extract_opentargets(ot_dir: Path) -> pl.DataFrame:
    """Extract variants from the OpenTargets gold-standards repo."""
    if not ot_dir.exists():
        print(f"  OpenTargets: directory not found — {ot_dir}", file=sys.stderr)
        return pl.DataFrame()

    print(f"  OpenTargets: scanning {ot_dir} ...")
    chunks: list[pl.DataFrame] = []

    data_files = sorted(
        p for p in ot_dir.rglob("*")
        if p.is_file() and (p.name.endswith(".tsv") or p.name.endswith(".tsv.gz")
                            or p.name.endswith(".csv") or p.name.endswith(".csv.gz"))
    )

    for tf in data_files:
        try:
            kwargs = {}
            sep = "\t"
            if tf.name.endswith(".csv") or tf.name.endswith(".csv.gz"):
                sep = ","
            df = pl.read_csv(tf, separator=sep, n_rows=5, infer_schema_length=10000, **kwargs)
            cols = set(df.columns)
            chrom_col = next((c for c in ["chr", "chrom", "chromosome", "Chr",
                                          "sentinel_variant.locus_GRCh38.chromosome"] if c in cols), None)
            pos_col = next((c for c in ["pos", "position", "Pos", "bp", "BP",
                                        "sentinel_variant.locus_GRCh38.position"] if c in cols), None)
            ref_col = next((c for c in ["ref", "Ref", "reference",
                                        "sentinel_variant.alleles.reference"] if c in cols), None)
            alt_col = next((c for c in ["alt", "Alt", "alternate",
                                        "sentinel_variant.alleles.alternative"] if c in cols), None)
            if not all([chrom_col, pos_col, ref_col, alt_col]):
                continue
            df_full = pl.read_csv(tf, separator=sep, columns=[chrom_col, pos_col, ref_col, alt_col],
                                  infer_schema_length=10000, **kwargs)
            df_full = df_full.rename({chrom_col: "chrom", pos_col: "pos", ref_col: "ref", alt_col: "alt"})
            df_full = df_full.with_columns(pl.lit("GWAS").alias("context"))
            chunks.append(df_full)
        except Exception:
            continue

    if not chunks:
        print(f"  OpenTargets: no variants extracted", file=sys.stderr)
        return pl.DataFrame()
    df = pl.concat(chunks).unique().with_columns(pl.lit("OpenTargets").alias("source"))
    print(f"  OpenTargets: {df.height} unique variants")
    return df


def _extract_pgboost(pgboost_dir: Path) -> pl.DataFrame:
    """Extract variants from the pgBoost Zenodo files."""
    if not pgboost_dir.exists():
        print(f"  pgBoost: directory not found — {pgboost_dir}", file=sys.stderr)
        return pl.DataFrame()

    print(f"  pgBoost: scanning {pgboost_dir} ...")
    variants: list[dict] = []
    seen: set[str] = set()

    data_files = sorted(
        p for p in pgboost_dir.rglob("*")
        if p.is_file() and (p.name.endswith(".tsv") or p.name.endswith(".tsv.gz"))
    )

    for tf in data_files:
        try:
            kwargs = {}
            df = read_tsv(tf, n_rows=5, infer_schema_length=10000, **kwargs)
            cols = set(df.columns)
            chrom_col = next((c for c in ["chr", "chrom", "chromosome", "Chr"] if c in cols), None)
            pos_col = next((c for c in ["pos", "position", "Pos", "bp"] if c in cols), None)
            ref_col = next((c for c in ["ref", "Ref", "reference"] if c in cols), None)
            alt_col = next((c for c in ["alt", "Alt", "alternate"] if c in cols), None)
            if not all([chrom_col, pos_col, ref_col, alt_col]):
                # Try variant_id column.
                vid_col = next((c for c in ["variant_id", "variant"] if c in cols), None)
                if vid_col:
                    df_full = read_tsv(tf, columns=[vid_col], **kwargs)
                    for vid in df_full[vid_col].to_list():
                        p = _parse_gtex_variant_id(vid)
                        if p:
                            key = f"{p['chrom']}:{p['pos']}:{p['ref']}:{p['alt']}"
                            if key not in seen:
                                seen.add(key)
                                variants.append(p)
                continue
            df_full = read_tsv(tf, columns=[chrom_col, pos_col, ref_col, alt_col], **kwargs)
            for row in df_full.iter_rows(named=True):
                key = f"{row[chrom_col]}:{row[pos_col]}:{row[ref_col]}:{row[alt_col]}"
                if key in seen:
                    continue
                seen.add(key)
                variants.append({
                    "chrom": row[chrom_col],
                    "pos": row[pos_col],
                    "ref": row[ref_col],
                    "alt": row[alt_col],
                })
        except Exception:
            continue

    if not variants:
        print(f"  pgBoost: no variants extracted", file=sys.stderr)
        return pl.DataFrame()
    df = pl.DataFrame(variants).with_columns(pl.lit("pgBoost").alias("source"))
    print(f"  pgBoost: {len(variants)} unique variants")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize variants from all raw datasets"
    )
    parser.add_argument("--gtex", required=True, type=Path, help="GTEx v11 SuSiE tar path")
    parser.add_argument("--eqtl", required=True, type=Path, help="eQTL Catalogue directory")
    parser.add_argument("--crispr", required=True, type=Path, help="CRISPR comparison directory")
    parser.add_argument("--gwas", required=True, type=Path, help="GWAS E2G benchmarking directory")
    parser.add_argument("--opentargets", required=True, type=Path, help="OpenTargets gold-standards directory")
    parser.add_argument("--pgboost", required=True, type=Path, help="pgBoost Zenodo directory")
    parser.add_argument("--fasta", required=True, type=Path, help="GRCh38 reference FASTA path")
    parser.add_argument("--output", required=True, type=Path, help="Output parquet path")
    parser.add_argument("--qc-output", required=True, type=Path, help="QC report TSV path")
    args = parser.parse_args()

    print("Normalizing variants from all datasets ...")

    # Extract variants from each source.
    extractors = [
        ("GTEx", _extract_gtex, args.gtex),
        ("eQTL_Catalogue", _extract_eqtl_catalogue, args.eqtl),
        ("CRISPR", _extract_crispr, args.crispr),
        ("GWAS", _extract_gwas, args.gwas),
        ("OpenTargets", _extract_opentargets, args.opentargets),
        ("pgBoost", _extract_pgboost, args.pgboost),
    ]

    frames: list[pl.DataFrame] = []
    source_counts: dict[str, int] = {}

    for name, fn, path in extractors:
        df = fn(path)
        if not df.is_empty():
            # Ensure consistent column order
            cols = [c for c in ["chrom", "pos", "ref", "alt", "context", "source"] if c in df.columns]
            df = df.select(cols)
            frames.append(df)
            source_counts[name] = df.height

    if not frames:
        print("ERROR: No variants extracted from any dataset.", file=sys.stderr)
        return 1

    combined = pl.concat(frames, how="vertical_relaxed")
    print(f"\nTotal variants before dedup: {combined.height}")

    # Deduplicate on chrom/pos/ref/alt, keeping all source labels.
    combined = combined.unique(subset=["chrom", "pos", "ref", "alt"], keep="first")
    print(f"Total unique variants: {combined.height}")

    # Normalize using the harmonize module.
    print(f"\nNormalizing with bcftools-compatible canonical schema ...")
    normalized = normalize_variant_df(
        combined,
        reference_fasta=args.fasta,
        chrom_col="chrom",
        pos_col="pos",
        ref_col="ref",
        alt_col="alt",
        build="GRCh38",
        assign_qc=True,
    )

    # Write parquet.
    write_parquet(normalized, args.output)
    print(f"\nWrote normalized variants: {args.output} ({normalized.height} rows)")

    # Build QC report.
    qc_rows = []
    for name, count in source_counts.items():
        qc_rows.append({"source": name, "variants_extracted": count})

    # Add QC status breakdown.
    if "qc_status" in normalized.columns:
        status_counts = (
            normalized.group_by("qc_status")
            .len()
            .rename({"len": "count"})
            .to_dicts()
        )
        for sc in status_counts:
            qc_rows.append({
                "source": f"QC:{sc['qc_status']}",
                "variants_extracted": sc["count"],
            })

    if "ref_match" in normalized.columns:
        ref_match_count = normalized.filter(pl.col("ref_match") == True).height  # noqa: E712
        ref_mismatch = normalized.filter(pl.col("ref_match") == False).height  # noqa: E712
        qc_rows.append({"source": "ref_match:True", "variants_extracted": ref_match_count})
        qc_rows.append({"source": "ref_match:False", "variants_extracted": ref_mismatch})

    qc_df = pl.DataFrame(qc_rows)
    write_tsv(qc_df, args.qc_output)
    print(f"Wrote QC report: {args.qc_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
