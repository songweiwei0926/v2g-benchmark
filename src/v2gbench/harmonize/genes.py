"""Gene normalization from GENCODE GTF.

Parses GENCODE (v47) GTF files, extracts gene- and exon-level records, and
builds a canonical gene master table conforming to
:mod:`v2gbench.schemas.gene`. Ensembl gene IDs are de-versioned
(``ENSG00000123456.12`` → ``ENSG00000123456``) and exon intervals are stored
as JSON-encoded strings.

All DataFrame operations use :mod:`polars`; GTF files may be gzip-compressed
(``*.gtf.gz``).
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from pathlib import Path
from typing import Optional

import polars as pl

from ..io.parquet import write_parquet
from ..schemas.gene import compute_tss, deversion_gene_id, gene_schema

logger = logging.getLogger(__name__)

# Regex for the GTF ``gene_id`` attribute and other attributes of interest.
_ATTR_GENE_ID = re.compile(r'gene_id "([^"]+)"')
_ATTR_GENE_NAME = re.compile(r'gene_name "([^"]+)"')
_ATTR_GENE_TYPE = re.compile(r'gene_type "([^"]+)"')
_ATTR_GENE_BIOTYPE = re.compile(r'gene_biotype "([^"]+)"')
_ATTR_TRANSCRIPT_ID = re.compile(r'transcript_id "([^"]+)"')
_ATTR_EXON_NUMBER = re.compile(r'exon_number "(\d+)"')

# Canonical transcript priority: MANE Select > MANE Plus Clinical > Ensembl canonical > longest.
_CANONICAL_PRIORITY = ("mane_select", "mane_plus_clinical", "ensembl_canonical")


def _open_gtf(gtf_path: str | Path):
    """Open a GTF file, transparently handling gzip."""
    gtf_path = Path(gtf_path)
    if str(gtf_path).endswith(".gz"):
        return gzip.open(gtf_path, "rt")
    return open(gtf_path, "rt")


def _parse_attributes(attr_field: str) -> dict[str, str]:
    """Parse the GTF 9th-column attribute string into a dict.

    Uses regex extraction for robustness against minor formatting variations
    in the standard ``key "value";`` syntax.
    """
    attrs: dict[str, str] = {}
    for match in re.finditer(r'(\w+)\s+"([^"]*)"', attr_field):
        attrs[match.group(1)] = match.group(2)
    return attrs


def parse_gencode_gtf(gtf_path: str | Path) -> pl.DataFrame:
    """Parse a GENCODE GTF and extract gene-level records.

    Only ``feature == "gene"`` rows are returned, with columns:
    ``gene_id``, ``gene_symbol``, ``chrom``, ``start``, ``end``, ``strand``,
    ``gene_type``, ``tss``. Gene IDs are de-versioned.

    Parameters
    ----------
    gtf_path
        Path to a GENCODE GTF (optionally gzip-compressed).

    Returns
    -------
    pl.DataFrame
        Gene-level records with the columns listed above.

    Raises
    ------
    FileNotFoundError
        If the GTF does not exist.
    """
    gtf_path = Path(gtf_path)
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF not found: {gtf_path}")

    records: list[dict] = []
    with _open_gtf(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            if fields[2] != "gene":
                continue
            chrom, source, feature, start, end, score, strand, frame, attr = fields[:9]
            attrs = _parse_attributes(attr)
            gene_id = attrs.get("gene_id", "")
            if not gene_id:
                continue
            gene_id = deversion_gene_id(gene_id)
            gene_symbol = attrs.get("gene_name", attrs.get("Name"))
            gene_type = attrs.get("gene_type", attrs.get("gene_biotype"))
            start_i, end_i = int(start), int(end)
            records.append(
                {
                    "gene_id": gene_id,
                    "gene_symbol": gene_symbol,
                    "chrom": chrom,
                    "start": start_i,
                    "end": end_i,
                    "strand": strand,
                    "gene_type": gene_type,
                    "tss": compute_tss(start_i, end_i, strand),
                }
            )

    if not records:
        logger.warning("No gene records parsed from %s", gtf_path)
        return pl.DataFrame(
            schema={
                "gene_id": pl.Utf8,
                "gene_symbol": pl.Utf8,
                "chrom": pl.Utf8,
                "start": pl.Int64,
                "end": pl.Int64,
                "strand": pl.Utf8,
                "gene_type": pl.Utf8,
                "tss": pl.Int64,
            }
        )

    return pl.DataFrame(records)


def deversion_gene_ids(df: pl.DataFrame, *, gene_id_col: str = "gene_id") -> pl.DataFrame:
    """Remove the ``.N`` version suffix from Ensembl gene IDs in a DataFrame.

    Parameters
    ----------
    df
        Polars DataFrame containing a gene-ID column.
    gene_id_col
        Name of the column to de-version.

    Returns
    -------
    pl.DataFrame
        A copy of ``df`` with the version suffix stripped from ``gene_id_col``.
    """
    if gene_id_col not in df.columns:
        raise ValueError(f"deversion_gene_ids: column '{gene_id_col}' not found")
    return df.with_columns(
        pl.col(gene_id_col).map_elements(deversion_gene_id, return_dtype=pl.Utf8).alias(gene_id_col)
    )


def extract_exon_intervals(
    gtf_path: str | Path,
    gene_id: str,
) -> list[list[int]]:
    """Extract exon intervals (1-based, inclusive) for a single gene.

    Scans the GTF for ``exon`` records whose ``gene_id`` attribute matches
    ``gene_id`` (after de-versioning both sides). Returns a list of
    ``[start, end]`` pairs sorted by genomic coordinate.

    Parameters
    ----------
    gtf_path
        Path to a GENCODE GTF (optionally gzip-compressed).
    gene_id
        Ensembl gene ID, with or without a version suffix.

    Returns
    -------
    list[list[int]]
        Sorted list of ``[start, end]`` exon intervals.
    """
    gtf_path = Path(gtf_path)
    target = deversion_gene_id(gene_id)
    intervals: list[list[int]] = []
    with _open_gtf(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue
            attrs = _parse_attributes(fields[8])
            gid = deversion_gene_id(attrs.get("gene_id", ""))
            if gid != target:
                continue
            intervals.append([int(fields[3]), int(fields[4])])
    intervals.sort(key=lambda iv: (iv[0], iv[1]))
    return intervals


def _pick_canonical_transcript(
    gtf_path: str | Path,
    gene_id: str,
) -> Optional[str]:
    """Pick a canonical transcript for a gene.

    Priority: MANE Select > MANE Plus Clinical > Ensembl canonical tag >
    longest transcript (by summed exon length). Returns ``None`` if no
    transcript is found.
    """
    gtf_path = Path(gtf_path)
    target = deversion_gene_id(gene_id)
    mane_select: Optional[str] = None
    mane_plus: Optional[str] = None
    ensembl_canon: Optional[str] = None
    transcript_lengths: dict[str, int] = {}

    with _open_gtf(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue
            attrs = _parse_attributes(fields[8])
            gid = deversion_gene_id(attrs.get("gene_id", ""))
            if gid != target:
                continue
            tx_id = attrs.get("transcript_id")
            if not tx_id:
                continue
            tx_id = re.sub(r"\.\d+$", "", tx_id)
            transcript_lengths[tx_id] = transcript_lengths.get(tx_id, 0) + (
                int(fields[4]) - int(fields[3]) + 1
            )
            if attrs.get("mane_select") and mane_select is None:
                mane_select = tx_id
            if attrs.get("mane_plus_clinical") and mane_plus is None:
                mane_plus = tx_id
            if attrs.get("ensembl_canonical") and ensembl_canon is None:
                ensembl_canon = tx_id

    for candidate in (mane_select, mane_plus, ensembl_canon):
        if candidate:
            return candidate
    if transcript_lengths:
        return max(transcript_lengths, key=transcript_lengths.get)
    return None


def build_gene_master_table(
    gtf_path: str | Path,
    output_path: str | Path,
) -> str:
    """Build the canonical gene master table from a GENCODE GTF.

    Produces a parquet file with columns matching
    :data:`v2gbench.schemas.gene.gene_schema`:
    ``gene_id``, ``gene_symbol``, ``chrom``, ``start``, ``end``, ``strand``,
    ``tss``, ``gene_type``, ``canonical_transcript``, ``exon_intervals``
    (a JSON-encoded list of ``[start, end]`` pairs).

    Parameters
    ----------
    gtf_path
        Path to a GENCODE GTF (optionally gzip-compressed).
    output_path
        Destination parquet path (e.g.
        ``data/reference/gene_master.parquet``). Parent directories are
        created.

    Returns
    -------
    str
        The path to the written parquet file.
    """
    gtf_path = Path(gtf_path)
    output_path = Path(output_path)

    genes_df = parse_gencode_gtf(gtf_path)
    if genes_df.is_empty():
        logger.warning("No genes parsed; writing empty gene master table to %s", output_path)
        empty = pl.DataFrame(
            schema={
                "gene_id": pl.Utf8,
                "gene_symbol": pl.Utf8,
                "chrom": pl.Utf8,
                "start": pl.Int64,
                "end": pl.Int64,
                "strand": pl.Utf8,
                "tss": pl.Int64,
                "gene_type": pl.Utf8,
                "canonical_transcript": pl.Utf8,
                "exon_intervals": pl.Utf8,
            }
        )
        write_parquet(empty, output_path)
        return str(output_path)

    gene_ids = genes_df["gene_id"].to_list()
    canonical_txs: list[Optional[str]] = []
    exon_jsons: list[Optional[str]] = []
    logger.info("Building exon intervals + canonical transcripts for %d genes", len(gene_ids))
    for gid in gene_ids:
        tx = _pick_canonical_transcript(gtf_path, gid)
        canonical_txs.append(tx)
        intervals = extract_exon_intervals(gtf_path, gid)
        exon_jsons.append(json.dumps(intervals) if intervals else None)

    genes_df = genes_df.with_columns(
        pl.Series(name="canonical_transcript", values=canonical_txs, dtype=pl.Utf8),
        pl.Series(name="exon_intervals", values=exon_jsons, dtype=pl.Utf8),
    )

    # Order columns to match the schema.
    ordered = genes_df.select(
        [
            "gene_id",
            "gene_symbol",
            "chrom",
            "start",
            "end",
            "strand",
            "tss",
            "gene_type",
            "canonical_transcript",
            "exon_intervals",
        ]
    )

    write_parquet(ordered, output_path)
    logger.info("Wrote gene master table (%d genes) to %s", ordered.height, output_path)
    return str(output_path)


__all__ = [
    "parse_gencode_gtf",
    "deversion_gene_ids",
    "build_gene_master_table",
    "extract_exon_intervals",
]
