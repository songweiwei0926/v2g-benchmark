#!/usr/bin/env python3
"""Fix allele orientation for variants where REF doesn't match GRCh38.

For variants flagged as REF_MISMATCH, tries swapping REF/ALT and re-checking
against the reference FASTA. If the swapped REF matches, updates the variant
with the correct orientation.

This resolves the common issue where GWAS allele1/allele2 are not
ref/alt-ordered (allele1 is the effect allele, not necessarily the reference).
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl


def fix_allele_orientation(
    variants_df: pl.DataFrame,
    reference_fasta: str | Path,
) -> pl.DataFrame:
    """Fix allele orientation for REF_MISMATCH variants.

    For each variant where ref_match=False:
    1. Swap ref and alt
    2. Check if the swapped ref matches the FASTA
    3. If yes, update ref, alt, variant_id, ref_match=True, qc_status=PASS

    Parameters
    ----------
    variants_df
        Normalized variants with ref_match and qc_status columns.
    reference_fasta
        Path to indexed GRCh38 FASTA.

    Returns
    -------
    pl.DataFrame
        Updated variants with fixed allele orientation.
    """
    import pysam

    fasta = pysam.FastaFile(str(reference_fasta))
    chroms_available = set(fasta.references)

    df = variants_df.clone()

    # Get mismatched variants
    mismatched = df.filter(pl.col("ref_match") == False)
    n_mismatch = mismatched.height
    print(f"  REF_MISMATCH variants to check: {n_mismatch}")

    if n_mismatch == 0:
        fasta.close()
        return df

    # For each mismatched variant, try swapping ref/alt
    rows = mismatched.select(["chrom", "pos", "ref", "alt"]).to_dicts()
    swap_indices = []
    new_refs = []
    new_alts = []

    mismatched_idx = mismatched.to_pandas().index.tolist()

    for i, row in enumerate(rows):
        ref_allele = str(row["ref"]).upper()
        alt_allele = str(row["alt"]).upper()
        swapped_ref = alt_allele
        swapped_alt = ref_allele

        chrom = str(row["chrom"])
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"
        if chrom not in chroms_available:
            continue

        pos = int(row["pos"])
        end = pos + len(swapped_ref) - 1
        try:
            ref_base = fasta.fetch(chrom, pos - 1, end).upper()
            if ref_base == swapped_ref:
                swap_indices.append(mismatched_idx[i])
                new_refs.append(swapped_ref)
                new_alts.append(swapped_alt)
        except (ValueError, KeyError, IOError):
            continue

    fasta.close()

    n_fixed = len(swap_indices)
    n_remaining = n_mismatch - n_fixed
    print(f"  Fixed by allele swap: {n_fixed}")
    print(f"  Remaining REF_MISMATCH: {n_remaining}")

    if n_fixed == 0:
        return df

    # Apply swaps
    df_pd = df.to_pandas()
    for idx, new_ref, new_alt in zip(swap_indices, new_refs, new_alts):
        df_pd.at[idx, "ref"] = new_ref
        df_pd.at[idx, "alt"] = new_alt
        df_pd.at[idx, "ref_match"] = True
        df_pd.at[idx, "qc_status"] = "PASS"
        # Update variant_id
        chrom = df_pd.at[idx, "chrom"]
        pos = df_pd.at[idx, "pos"]
        df_pd.at[idx, "variant_id"] = f"GRCh38:{chrom}:{pos}:{new_ref}:{new_alt}"

    result = pl.from_pandas(df_pd)

    # Re-deduplicate after allele fixes (swapped variants might now duplicate)
    before = result.height
    result = result.unique(subset=["chrom", "pos", "ref", "alt"], keep="first")
    after = result.height
    if before != after:
        print(f"  Dedup after swap: {before} -> {after} (removed {before - after} duplicates)")

    return result


def main() -> int:
    project_root = Path("/workspace/v2g-benchmark")

    print("=" * 70)
    print("Fixing allele orientation for REF_MISMATCH variants")
    print("=" * 70)

    # Load variants
    variants_path = project_root / "data" / "processed" / "variants_normalized.parquet"
    fasta_path = project_root / "data" / "reference" / "GRCh38.fa"

    print(f"\nLoading variants: {variants_path}")
    df = pl.read_parquet(str(variants_path))
    print(f"  Loaded {df.height} variants")

    # Show before stats
    print(f"\nBefore fix:")
    print(f"  PASS: {df.filter(pl.col('qc_status') == 'PASS').height}")
    print(f"  REF_MISMATCH: {df.filter(pl.col('qc_status') == 'REF_MISMATCH').height}")

    # Fix allele orientation
    print(f"\nFixing allele orientation...")
    fixed_df = fix_allele_orientation(df, fasta_path)

    # Show after stats
    print(f"\nAfter fix:")
    print(f"  PASS: {fixed_df.filter(pl.col('qc_status') == 'PASS').height}")
    print(f"  REF_MISMATCH: {fixed_df.filter(pl.col('qc_status') == 'REF_MISMATCH').height}")

    # Per-source breakdown
    print(f"\nPer-source QC after fix:")
    qc = fixed_df.group_by(["source", "qc_status"]).len().sort(["source", "qc_status"])
    print(qc.to_pandas().to_string())

    # Write fixed variants
    output_path = project_root / "data" / "processed" / "variants_normalized.parquet"
    fixed_df.write_parquet(str(output_path))
    print(f"\nWrote fixed variants: {output_path} ({fixed_df.height} variants)")

    # Update QC report
    qc_report_path = project_root / "data" / "processed" / "variant_qc_report.tsv"
    qc_rows = []
    for source in fixed_df["source"].unique().sort().to_list():
        n = fixed_df.filter(pl.col("source") == source).height
        qc_rows.append({"source": source, "variants_extracted": n})
    qc_rows.append({"source": "QC:REF_MISMATCH", "variants_extracted": fixed_df.filter(pl.col("ref_match") == False).height})
    qc_rows.append({"source": "QC:PASS", "variants_extracted": fixed_df.filter(pl.col("ref_match") == True).height})
    qc_rows.append({"source": "ref_match:True", "variants_extracted": fixed_df.filter(pl.col("ref_match") == True).height})
    qc_rows.append({"source": "ref_match:False", "variants_extracted": fixed_df.filter(pl.col("ref_match") == False).height})
    qc_df = pl.DataFrame(qc_rows)
    qc_df.write_csv(str(qc_report_path), separator="\t")
    print(f"Wrote QC report: {qc_report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
