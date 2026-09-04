"""Tests for SHA256-based deterministic sampling reproducibility."""

import polars as pl
import pytest

from v2gbench.utils.hashing import (
    stable_hash,
    hash_to_float,
    hash_to_int,
    deterministic_rank,
    deterministic_sample,
)
from v2gbench.statistics.sampling import (
    deterministic_stratified_sample,
    assign_distance_bin,
    assign_pip_bin,
    assign_nearest_rank,
)
from v2gbench.benchmark.seq_core import hash_variant_context


class TestStableHash:
    def test_same_input_same_hash(self):
        assert stable_hash("abc") == stable_hash("abc")

    def test_different_input_different_hash(self):
        assert stable_hash("abc") != stable_hash("abd")

    def test_multiple_args(self):
        assert stable_hash("a", "b") == stable_hash("a", "b")
        assert stable_hash("a", "b") != stable_hash("b", "a")

    def test_returns_hex_string(self):
        h = stable_hash("test")
        assert len(h) == 64  # SHA256 hex
        assert all(c in "0123456789abcdef" for c in h)


class TestHashToFloat:
    def test_in_unit_interval(self):
        f = hash_to_float("test")
        assert 0.0 <= f < 1.0

    def test_deterministic(self):
        assert hash_to_float("abc") == hash_to_float("abc")

    def test_different_inputs_different_floats(self):
        floats = [hash_to_float(f"item_{i}") for i in range(100)]
        assert len(set(floats)) > 90  # mostly unique


class TestHashToInt:
    def test_in_range(self):
        assert 0 <= hash_to_int("test", max_val=10) < 10

    def test_deterministic(self):
        assert hash_to_int("abc", max_val=100) == hash_to_int("abc", max_val=100)


class TestDeterministicRank:
    def test_same_seed_same_order(self):
        items = ["a", "b", "c", "d", "e"]
        r1 = deterministic_rank(items, seed=42)
        r2 = deterministic_rank(items, seed=42)
        assert r1 == r2

    def test_different_seed_different_order(self):
        items = ["a", "b", "c", "d", "e"]
        r1 = deterministic_rank(items, seed=42)
        r2 = deterministic_rank(items, seed=99)
        assert r1 != r2

    def test_preserves_all_items(self):
        items = ["a", "b", "c"]
        ranked = deterministic_rank(items, seed=1)
        assert set(ranked) == set(items)


class TestDeterministicSample:
    def test_same_seed_same_sample(self):
        items = list(range(100))
        s1 = deterministic_sample(items, 10, seed=42)
        s2 = deterministic_sample(items, 10, seed=42)
        assert s1 == s2

    def test_different_seed_different_sample(self):
        items = list(range(100))
        s1 = deterministic_sample(items, 10, seed=42)
        s2 = deterministic_sample(items, 10, seed=99)
        assert s1 != s2

    def test_sample_size(self):
        items = list(range(50))
        s = deterministic_sample(items, 10, seed=1)
        assert len(s) == 10


class TestDeterministicStratifiedSample:
    def test_reproducible(self):
        df = pl.DataFrame({
            "variant_id": [f"v{i}" for i in range(100)],
            "stratum": ["A"] * 50 + ["B"] * 50,
            "value": list(range(100)),
        })
        s1 = deterministic_stratified_sample(df, ["stratum"], 20, seed=42)
        s2 = deterministic_stratified_sample(df, ["stratum"], 20, seed=42)
        assert s1.equals(s2)

    def test_different_seed_different(self):
        df = pl.DataFrame({
            "variant_id": [f"v{i}" for i in range(100)],
            "stratum": ["A"] * 50 + ["B"] * 50,
            "value": list(range(100)),
        })
        s1 = deterministic_stratified_sample(df, ["stratum"], 20, seed=42)
        s2 = deterministic_stratified_sample(df, ["stratum"], 20, seed=99)
        assert not s1.equals(s2)

    def test_sample_size_correct(self):
        df = pl.DataFrame({
            "variant_id": [f"v{i}" for i in range(100)],
            "stratum": ["A"] * 50 + ["B"] * 50,
        })
        s = deterministic_stratified_sample(df, ["stratum"], 20, seed=1)
        assert s.height == 20

    def test_preserves_strata_proportions(self):
        df = pl.DataFrame({
            "variant_id": [f"v{i}" for i in range(100)],
            "stratum": ["A"] * 80 + ["B"] * 20,
        })
        s = deterministic_stratified_sample(df, ["stratum"], 10, seed=1)
        n_a = s.filter(pl.col("stratum") == "A").height
        n_b = s.filter(pl.col("stratum") == "B").height
        # Proportional: A should get ~8, B should get ~2
        assert n_a >= 6 and n_b >= 1


class TestHashVariantContext:
    def test_deterministic(self):
        h1 = hash_variant_context("v1", "c1")
        h2 = hash_variant_context("v1", "c1")
        assert h1 == h2

    def test_different_pairs_different_hash(self):
        assert hash_variant_context("v1", "c1") != hash_variant_context("v1", "c2")
        assert hash_variant_context("v1", "c1") != hash_variant_context("v2", "c1")


class TestAssignBins:
    def test_distance_bin(self):
        bins = [1e4, 1e5, 1e6]
        assert assign_distance_bin(500, bins) == 1
        assert assign_distance_bin(50_000, bins) == 2
        assert assign_distance_bin(500_000, bins) == 3
        assert assign_distance_bin(2_000_000, bins) == 3

    def test_distance_bin_none(self):
        assert assign_distance_bin(None, [1e4]) is None

    def test_pip_bin(self):
        bins = [0.7, 0.9, 0.95]
        assert assign_pip_bin(0.5, bins) == 1
        assert assign_pip_bin(0.8, bins) == 2
        assert assign_pip_bin(0.92, bins) == 3
        assert assign_pip_bin(0.99, bins) == 3

    def test_nearest_rank(self):
        assert assign_nearest_rank(1) == "1"
        assert assign_nearest_rank(2) == "2"
        assert assign_nearest_rank(3) == "3"
        assert assign_nearest_rank(5) == "4+"
        assert assign_nearest_rank(None) is None
