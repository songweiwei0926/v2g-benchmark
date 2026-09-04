"""Variant normalization using ``bcftools norm``.

This module brings VCF records onto the canonical GRCh38 variant schema defined
in :mod:`v2gbench.schemas.variant`. It wraps ``bcftools norm`` for
left-alignment / splitting of multiallelics and provides pure-Python helpers
for canonical ``variant_id`` generation, REF-allele verification against the
reference FASTA, and QC-status assignment.

All DataFrame operations use :mod:`polars`.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import polars as pl

from ..io.parquet import write_parquet
from ..schemas.variant import (
    VARIANT_QC_STATUS,
    make_variant_id,
    variant_schema,
)

logger = logging.getLogger(__name__)

# Canonical chromosome set for GRCh38 (used for validation / coercion).
_VALID_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]


def normalize_variants(
    input_vcf: str | Path,
    output_vcf: str | Path,
    reference_fasta: str | Path,
    *,
    multiallelic_mode: str = "-",
    extra_args: Optional[list[str]] = None,
) -> str:
    """Normalize a VCF with ``bcftools norm``.

    Runs::

        bcftools norm -f <reference_fasta> -m <multiallelic_mode> -any <input_vcf> -o <output_vcf>

    which left-aligns indels, splits/merges multiallelic records, and drops
    duplicate records. The reference FASTA must be GRCh38 and indexed
    (``.fai``).

    Parameters
    ----------
    input_vcf
        Path to the input VCF (may be bgzipped ``.vcf.gz``).
    output_vcf
        Path to write the normalized VCF. Parent directories are created.
    reference_fasta
        Path to the GRCh38 reference FASTA (with ``.fai`` index).
    multiallelic_mode
        ``bcftools -m`` mode: ``-`` to split multiallelics (default) or ``+``
        to merge them.
    extra_args
        Additional arguments appended to the ``bcftools norm`` invocation.

    Returns
    -------
    str
        The path to the normalized output VCF (as a string).

    Raises
    ------
    FileNotFoundError
        If ``bcftools`` is not on ``PATH`` or the input/reference are missing.
    RuntimeError
        If ``bcftools norm`` exits with a non-zero status.
    """
    if shutil.which("bcftools") is None:
        raise FileNotFoundError("bcftools not found on PATH; install htslib/bcftools.")
    input_vcf = Path(input_vcf)
    output_vcf = Path(output_vcf)
    reference_fasta = Path(reference_fasta)
    if not input_vcf.exists():
        raise FileNotFoundError(f"Input VCF not found: {input_vcf}")
    if not reference_fasta.exists():
        raise FileNotFoundError(f"Reference FASTA not found: {reference_fasta}")

    output_vcf.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bcftools",
        "norm",
        "-f", str(reference_fasta),
        "-m", multiallelic_mode,
        "-any",
        "-O", "z",
        "-o", str(output_vcf),
        str(input_vcf),
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Running bcftools norm: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"bcftools norm failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    if result.stderr.strip():
        logger.debug("bcftools norm stderr: %s", result.stderr.strip())

    # Index the output for downstream tooling.
    tbi = output_vcf.with_suffix(output_vcf.suffix + ".tbi")
    if not tbi.exists():
        try:
            subprocess.run(
                ["bcftools", "index", "-t", str(output_vcf)],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("Failed to index %s: %s", output_vcf, exc)

    return str(output_vcf)


def generate_variant_id(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    build: str = "GRCh38",
) -> str:
    """Generate a canonical variant identifier.

    The canonical form is ``GRCh38:chr1:123456:A:G`` — genome build, chromosome,
    1-based position, reference allele, alternate allele. This is a thin
    wrapper around :func:`v2gbench.schemas.variant.make_variant_id` that also
    coerces ``chrom`` to a ``chr``-prefixed form and uppercases alleles.

    Parameters
    ----------
    chrom
        Chromosome name (e.g. ``"1"`` or ``"chr1"``).
    pos
        1-based genomic position.
    ref, alt
        Reference and alternate alleles (IUPAC bases).
    build
        Genome build label, default ``"GRCh38"``.

    Returns
    -------
    str
        Canonical variant ID.
    """
    chrom = str(chrom)
    if not chrom.startswith("chr"):
        chrom = f"chr{chrom}"
    return make_variant_id(chrom, int(pos), str(ref).upper(), str(alt).upper(), build)


def _read_fasta_base(reference_fasta: Path, chrom: str, pos: int) -> Optional[str]:
    """Fetch a single base from the reference FASTA at (chrom, pos).

    Uses ``samtools faidx`` when available (fast); returns ``None`` if the
    region cannot be resolved.
    """
    chrom = str(chrom)
    if not chrom.startswith("chr"):
        chrom = f"chr{chrom}"
    region = f"{chrom}:{pos}-{pos}"

    if shutil.which("samtools") is not None:
        try:
            result = subprocess.run(
                ["samtools", "faidx", str(reference_fasta), region],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = result.stdout.strip().splitlines()
            if len(lines) >= 2:
                return lines[1].strip().upper() or None
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("samtools faidx failed for %s", region)
    return None


def _read_fasta_window(reference_fasta: Path, chrom: str, start: int, length: int) -> Optional[str]:
    """Fetch a multi-base window from the reference FASTA."""
    chrom = str(chrom)
    if not chrom.startswith("chr"):
        chrom = f"chr{chrom}"
    end = start + length - 1
    region = f"{chrom}:{start}-{end}"
    if shutil.which("samtools") is not None:
        try:
            result = subprocess.run(
                ["samtools", "faidx", str(reference_fasta), region],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = result.stdout.strip().splitlines()
            if len(lines) >= 2:
                return "".join(line.strip() for line in lines[1:]).upper() or None
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("samtools faidx failed for %s", region)
    return None


def check_ref_match(
    variants_df: pl.DataFrame,
    reference_fasta: str | Path,
    *,
    chrom_col: str = "chrom",
    pos_col: str = "pos",
    ref_col: str = "ref",
) -> pl.DataFrame:
    """Verify that each variant's REF allele matches the GRCh38 reference.

    Adds a boolean column ``ref_match`` (``True`` when the recorded REF equals
    the base(s) in the reference FASTA at the same position). Lookup is
    performed via ``samtools faidx``; variants whose region cannot be resolved
    are marked ``False``.

    Parameters
    ----------
    variants_df
        Polars DataFrame with chromosome / position / REF columns.
    reference_fasta
        Path to the indexed GRCh38 FASTA.
    chrom_col, pos_col, ref_col
        Column names in ``variants_df``.

    Returns
    -------
    pl.DataFrame
        A copy of ``variants_df`` with an added ``ref_match`` column.
    """
    reference_fasta = Path(reference_fasta)
    if not reference_fasta.exists():
        raise FileNotFoundError(f"Reference FASTA not found: {reference_fasta}")

    df = variants_df.clone()
    rows = df.select([chrom_col, pos_col, ref_col]).to_dicts()
    matches: list[bool] = []

    try:
        import pysam
        fasta = pysam.FastaFile(str(reference_fasta))
        chroms_available = set(fasta.references)
        for row in rows:
            ref_allele = str(row[ref_col]).upper()
            if not ref_allele:
                matches.append(False)
                continue
            chrom = str(row[chrom_col])
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"
            if chrom not in chroms_available:
                matches.append(False)
                continue
            pos = int(row[pos_col])
            end = pos + len(ref_allele) - 1
            try:
                ref_base = fasta.fetch(chrom, pos - 1, end).upper()
                matches.append(ref_base == ref_allele)
            except (ValueError, KeyError, IOError):
                matches.append(False)
        fasta.close()
    except ImportError:
        for row in rows:
            ref_allele = str(row[ref_col]).upper()
            if not ref_allele:
                matches.append(False)
                continue
            if len(ref_allele) == 1:
                ref_base = _read_fasta_base(reference_fasta, str(row[chrom_col]), int(row[pos_col]))
                matches.append(ref_base is not None and ref_base == ref_allele)
            else:
                ref_base = _read_fasta_window(
                    reference_fasta,
                    str(row[chrom_col]),
                    int(row[pos_col]),
                    len(ref_allele),
                )
                matches.append(ref_base is not None and ref_base == ref_allele)

    return df.with_columns(pl.Series(name="ref_match", values=matches, dtype=pl.Boolean))


def assign_qc_status(variants_df: pl.DataFrame) -> pl.DataFrame:
    """Assign a QC status to each variant row.

    The status column ``qc_status`` takes one of the values in
    :data:`v2gbench.schemas.variant.VARIANT_QC_STATUS`:

    * ``PASS`` — variant is well-formed and REF matches the reference.
    * ``REF_MISMATCH`` — recorded REF does not match GRCh38.
    * ``MULTIALLELIC_SPLIT`` — flagged via a ``multiallelic_split`` column
      (set by upstream splitting) or an ``ALT`` containing a comma.
    * ``INVALID_ALLELE`` — empty REF/ALT or non-IUPAC characters.
    * ``LIFTOVER_FAILED`` — flagged via a ``liftover_failed`` column.

    Parameters
    ----------
    variants_df
        Polars DataFrame. Recognized optional columns: ``ref_match``,
        ``alt``, ``ref``, ``multiallelic_split``, ``liftover_failed``.

    Returns
    -------
    pl.DataFrame
        A copy of ``variants_df`` with an added ``qc_status`` column.
    """
    df = variants_df.clone()
    cols = set(df.columns)

    has_ref_match = "ref_match" in cols
    has_multi = "multiallelic_split" in cols
    has_liftover = "liftover_failed" in cols
    has_ref = "ref" in cols
    has_alt = "alt" in cols

    # Build the conditional expression from highest-priority to lowest.
    # Default PASS.
    expr = pl.lit("PASS")

    if has_liftover:
        expr = pl.when(pl.col("liftover_failed") == True).then(pl.lit("LIFTOVER_FAILED")).otherwise(expr)  # noqa: E712

    if has_ref and has_alt:
        invalid = (
            (pl.col("ref").str.len_chars() < 1)
            | (pl.col("alt").str.len_chars() < 1)
            | pl.col("ref").str.to_uppercase().str.contains(r"^[ACGTNRYKMSWBDHV]+$").not_()
            | pl.col("alt").str.to_uppercase().str.contains(r"^[ACGTNRYKMSWBDHV,*]+$").not_()
        )
        expr = pl.when(invalid).then(pl.lit("INVALID_ALLELE")).otherwise(expr)

    if has_multi:
        expr = pl.when(pl.col("multiallelic_split") == True).then(pl.lit("MULTIALLELIC_SPLIT")).otherwise(expr)  # noqa: E712
    elif has_alt:
        expr = pl.when(pl.col("alt").str.contains(",")).then(pl.lit("MULTIALLELIC_SPLIT")).otherwise(expr)

    if has_ref_match:
        expr = pl.when(pl.col("ref_match") == False).then(pl.lit("REF_MISMATCH")).otherwise(expr)  # noqa: E712

    return df.with_columns(expr.alias("qc_status"))


def normalize_variant_df(
    df: pl.DataFrame,
    reference_fasta: Optional[str | Path] = None,
    *,
    chrom_col: str = "chrom",
    pos_col: str = "pos",
    ref_col: str = "ref",
    alt_col: str = "alt",
    build: str = "GRCh38",
    assign_qc: bool = True,
) -> pl.DataFrame:
    """Normalize a polars DataFrame of variants to the canonical schema.

    Produces a canonical ``variant_id`` for every row, coerces chromosome names
    to ``chr``-prefixed form, uppercases alleles, optionally verifies REF
    alleles against the reference FASTA, and assigns a QC status.

    Parameters
    ----------
    df
        Polars DataFrame with chromosome / position / REF / ALT columns.
    reference_fasta
        Optional path to an indexed GRCh38 FASTA. When provided, a
        ``ref_match`` column is added via :func:`check_ref_match`.
    chrom_col, pos_col, ref_col, alt_col
        Names of the input columns.
    build
        Genome build label for the canonical ID.
    assign_qc
        Whether to add a ``qc_status`` column via :func:`assign_qc_status`.

    Returns
    -------
    pl.DataFrame
        Normalized DataFrame with ``variant_id`` (and optionally ``ref_match``,
        ``qc_status``) columns.
    """
    required = {chrom_col, pos_col, ref_col, alt_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"normalize_variant_df: missing required columns: {sorted(missing)}")

    out = df.clone()

    # Coerce chromosome to chr-prefixed string.
    out = out.with_columns(
        pl.col(chrom_col).cast(pl.Utf8).map_elements(
            lambda c: c if str(c).startswith("chr") else f"chr{c}",
            return_dtype=pl.Utf8,
        ).alias(chrom_col),
        pl.col(pos_col).cast(pl.Int64).alias(pos_col),
        pl.col(ref_col).cast(pl.Utf8).str.to_uppercase().alias(ref_col),
        pl.col(alt_col).cast(pl.Utf8).str.to_uppercase().alias(alt_col),
    )

    # Build canonical variant_id.
    out = out.with_columns(
        pl.struct([chrom_col, pos_col, ref_col, alt_col]).map_elements(
            lambda r: generate_variant_id(
                r[chrom_col], r[pos_col], r[ref_col], r[alt_col], build=build
            ),
            return_dtype=pl.Utf8,
        ).alias("variant_id"),
        pl.lit(build).alias("genome_build"),
    )

    # Optional REF verification.
    if reference_fasta is not None:
        out = check_ref_match(
            out,
            reference_fasta,
            chrom_col=chrom_col,
            pos_col=pos_col,
            ref_col=ref_col,
        )

    if assign_qc:
        out = assign_qc_status(out)

    return out


__all__ = [
    "normalize_variants",
    "generate_variant_id",
    "check_ref_match",
    "assign_qc_status",
    "normalize_variant_df",
]
