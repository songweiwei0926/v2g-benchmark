"""Parquet I/O utilities with ZSTD compression."""
import polars as pl
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def write_parquet(
    df: pl.DataFrame,
    path: str | Path,
    compression: str = "zstd",
    compression_level: int = 3,
) -> None:
    """Write a DataFrame to Parquet with ZSTD compression."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression=compression, compression_level=compression_level)


def read_parquet(path: str | Path, columns: Optional[List[str]] = None) -> pl.DataFrame:
    """Read a Parquet file, optionally selecting columns."""
    return pl.read_parquet(path, columns=columns)


def write_tsv(df: pl.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to TSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path, separator="\t")


def read_tsv(path: str | Path, **kwargs) -> pl.DataFrame:
    """Read a TSV file."""
    return pl.read_csv(path, separator="\t", **kwargs)


def append_parquet(
    df: pl.DataFrame,
    path: str | Path,
) -> None:
    """Append rows to an existing Parquet file (read + concat + write)."""
    path = Path(path)
    if path.exists():
        existing = read_parquet(path)
        combined = pl.concat([existing, df], how="vertical_relaxed")
        write_parquet(combined, path)
    else:
        write_parquet(df, path)
