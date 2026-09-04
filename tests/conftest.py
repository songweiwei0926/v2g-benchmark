"""Shared pytest fixtures and smoke-data paths for V2G-Benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SMOKE_DIR = PROJECT_ROOT / "tests" / "data" / "smoke"
CONFIG_DIR = PROJECT_ROOT / "config"


def _smoke_path(name: str) -> Path:
    """Return the path to a smoke fixture parquet, generating if needed."""
    p = SMOKE_DIR / f"{name}.parquet"
    if not p.exists():
        # Lazily generate fixtures if they don't exist.
        from tests.data.smoke.generate_fixtures import generate_all
        generate_all(SMOKE_DIR)
    return p


# ---------------------------------------------------------------------------
# Fixtures: smoke data paths
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def smoke_dir() -> Path:
    """Directory containing smoke fixture parquet files."""
    return SMOKE_DIR


@pytest.fixture(scope="session")
def variants_parquet() -> Path:
    return _smoke_path("variants")


@pytest.fixture(scope="session")
def gene_master_parquet() -> Path:
    return _smoke_path("gene_master")


@pytest.fixture(scope="session")
def candidates_parquet() -> Path:
    return _smoke_path("candidates")


@pytest.fixture(scope="session")
def evidence_parquet() -> Path:
    return _smoke_path("evidence")


@pytest.fixture(scope="session")
def predictions_parquet() -> Path:
    return _smoke_path("predictions")


@pytest.fixture(scope="session")
def contexts_parquet() -> Path:
    return _smoke_path("contexts")


# ---------------------------------------------------------------------------
# Fixtures: loaded smoke DataFrames
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def variants_df(variants_parquet) -> pl.DataFrame:
    return pl.read_parquet(variants_parquet)


@pytest.fixture(scope="session")
def gene_master_df(gene_master_parquet) -> pl.DataFrame:
    return pl.read_parquet(gene_master_parquet)


@pytest.fixture(scope="session")
def candidates_df(candidates_parquet) -> pl.DataFrame:
    return pl.read_parquet(candidates_parquet)


@pytest.fixture(scope="session")
def evidence_df(evidence_parquet) -> pl.DataFrame:
    return pl.read_parquet(evidence_parquet)


@pytest.fixture(scope="session")
def predictions_df(predictions_parquet) -> pl.DataFrame:
    return pl.read_parquet(predictions_parquet)


@pytest.fixture(scope="session")
def contexts_df(contexts_parquet) -> pl.DataFrame:
    return pl.read_parquet(contexts_parquet)


# ---------------------------------------------------------------------------
# Fixtures: small inline synthetic data for unit tests
# ---------------------------------------------------------------------------
@pytest.fixture
def small_variants_df() -> pl.DataFrame:
    """5 variants on chr22 for inline unit tests."""
    return pl.DataFrame(
        {
            "variant_id": [
                "GRCh38:chr22:17000000:A:G",
                "GRCh38:chr22:18000000:C:T",
                "GRCh38:chr22:19000000:G:A",
                "GRCh38:chr22:20000000:T:C",
                "GRCh38:chr22:21000000:A:T",
            ],
            "chrom": ["chr22"] * 5,
            "pos": [17_000_000, 18_000_000, 19_000_000, 20_000_000, 21_000_000],
            "ref": ["A", "C", "G", "T", "A"],
            "alt": ["G", "T", "A", "C", "T"],
            "genome_build": ["GRCh38"] * 5,
            "rsid": [None] * 5,
        }
    )


@pytest.fixture
def small_genes_df() -> pl.DataFrame:
    """5 genes on chr22 for inline unit tests."""
    return pl.DataFrame(
        {
            "gene_id": [
                "ENSG00000000001",
                "ENSG00000000002",
                "ENSG00000000003",
                "ENSG00000000004",
                "ENSG00000000005",
            ],
            "gene_symbol": ["G1", "G2", "G3", "G4", "G5"],
            "chrom": ["chr22"] * 5,
            "start": [17_000_100, 17_900_000, 19_100_000, 19_900_000, 21_100_000],
            "end": [17_050_000, 18_100_000, 19_200_000, 20_100_000, 21_200_000],
            "strand": ["+", "-", "+", "-", "+"],
            "tss": [17_000_100, 18_100_000, 19_100_000, 20_100_000, 21_100_000],
            "gene_type": ["protein_coding"] * 5,
            "canonical_transcript": [None] * 5,
            "exon_intervals": [None] * 5,
        }
    )


@pytest.fixture
def small_ranking_df() -> pl.DataFrame:
    """A small DataFrame with known ranking for metric verification.

    Variant v1 has 3 candidates; gold gene is ranked 2nd → RR=0.5.
    Variant v2 has 3 candidates; gold gene is ranked 1st → RR=1.0.
    MRR = (0.5 + 1.0) / 2 = 0.75
    Top1 = 1/2 = 0.5
    Recall@3 = 2/2 = 1.0
    """
    return pl.DataFrame(
        {
            "variant_id": [
                "v1", "v1", "v1",
                "v2", "v2", "v2",
            ],
            "gene_id": [
                "g_a", "g_b", "g_c",
                "g_d", "g_e", "g_f",
            ],
            "ranking_score": [
                0.9, 0.5, 0.1,
                0.8, 0.3, 0.2,
            ],
            "is_gold": [
                0, 1, 0,
                1, 0, 0,
            ],
            "distance_to_tss": [
                1000, 5000, 10000,
                2000, 8000, 15000,
            ],
        }
    )


@pytest.fixture
def rng_factory():
    """Return a factory that creates a deterministic RNG from a seed."""
    def _make(seed: int = 20260904) -> np.random.Generator:
        return np.random.default_rng(seed)
    return _make
