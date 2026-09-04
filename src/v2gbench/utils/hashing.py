"""SHA256-based deterministic hashing for reproducible sampling and ranking."""

import hashlib
import numpy as np


def stable_hash(*args) -> str:
    """Compute a deterministic SHA256 hash from string arguments."""
    s = "|".join(str(a) for a in args)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hash_to_float(*args) -> float:
    """Convert a stable hash to a float in [0, 1)."""
    h = stable_hash(*args)
    # Use first 16 hex chars → 64-bit int → [0, 1)
    return int(h[:16], 16) / float(2**64)


def hash_to_int(*args, max_val: int) -> int:
    """Convert a stable hash to an int in [0, max_val)."""
    return int(hash_to_float(*args) * max_val)


def deterministic_rank(items, seed: int = 20260904) -> list:
    """Deterministically shuffle a list using SHA256 hashing.
    
    Returns items sorted by their stable hash with the seed.
    """
    return sorted(items, key=lambda x: stable_hash(str(seed), str(x)))


def deterministic_sample(items, n: int, seed: int = 20260904) -> list:
    """Deterministically sample n items from a list."""
    ranked = deterministic_rank(items, seed)
    return ranked[:n]
