"""Context normalization and multi-level ontology matching.

A *context* is a cell type, tissue, or cell-line label associated with a
dataset. Different sources use different naming conventions, so this module
maps a *source* context onto a *canonical* context via a cascade of six
matching levels of decreasing stringency:

1. **Exact ontology ID match** — e.g. ``CL:0000057`` matches a target whose
   ``ontology_id`` equals it.
2. **Normalized exact string match** — normalized names are equal.
3. **Synonym match** — the source name appears in a target's synonym list.
4. **Ontology parent/child distance ≤ 1** — direct parent/child in the
   ontology.
5. **Ontology distance ≤ 2** — grandparent/grandchild or sibling.
6. **Published scE2G mapping** — a curated crosswalk from the scE2G project.

Each level returns a match with a ``mapping_method``, ``ontology_distance``,
and ``mapping_confidence`` (1.0 for level 1, decaying for lower levels).
:func:`map_context` tries the levels in order and returns the first (best)
match.

All DataFrame operations use :mod:`polars`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import polars as pl
import yaml

from ..schemas.context import normalize_context_name

logger = logging.getLogger(__name__)

# Confidence assigned to each mapping level (higher = more trustworthy).
_LEVEL_CONFIDENCE = {
    1: 1.00,
    2: 0.95,
    3: 0.90,
    4: 0.80,
    5: 0.70,
    6: 0.60,
}

# Human-readable method labels.
_LEVEL_METHOD = {
    1: "ontology_id_exact",
    2: "name_exact",
    3: "synonym",
    4: "ontology_distance_1",
    5: "ontology_distance_2",
    6: "scE2G_published",
}

# Ontology prefixes we recognize as bare IDs.
_ONTOLOGY_PREFIXES = {"CL", "UBERON", "EFO", "CLAO", "CVCL", "MONDO", "DOID"}


def normalize_context(context_name: str) -> str:
    """Normalize a context name for matching.

    Lowercases, strips whitespace, and replaces ``_`` and ``-`` separators with
    spaces. Mirrors :func:`v2gbench.schemas.context.normalize_context_name`.

    Parameters
    ----------
    context_name
        Raw context label.

    Returns
    -------
    str
        Normalized label.
    """
    if context_name is None:
        return ""
    return normalize_context_name(str(context_name))


def load_context_aliases(yaml_path: str | Path) -> dict[str, Any]:
    """Load a context-mapping YAML configuration.

    The YAML is expected to describe canonical contexts, their ontology IDs,
    synonyms, ontology parents/children, and an optional scE2G crosswalk. The
    exact schema is project-specific; this function returns the parsed dict
    unchanged and performs light validation.

    Expected top-level keys (all optional but recommended):

    * ``canonical_contexts`` — list of dicts with at least ``name`` and
      ``ontology_id``; may include ``synonyms``, ``parents``, ``children``.
    * ``ontology_distances`` — optional dict mapping
      ``"source_oid|target_oid"`` → integer distance.
    * ``sce2g_mapping`` — optional dict mapping source context → canonical
      context.

    Parameters
    ----------
    yaml_path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the YAML does not exist.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Context aliases YAML not found: {yaml_path}")
    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        logger.warning("Context aliases YAML %s is empty", yaml_path)
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Context aliases YAML {yaml_path} must parse to a mapping")
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_contexts(aliases: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the list of canonical context dicts from the aliases config."""
    ctxs = aliases.get("canonical_contexts", aliases.get("contexts", []))
    return ctxs or []


def _is_ontology_id(token: str) -> bool:
    """Heuristic: does ``token`` look like a bare ontology ID (e.g. CL:0000057)?"""
    token = token.strip()
    if ":" not in token:
        return False
    prefix = token.split(":", 1)[0].upper()
    return prefix in _ONTOLOGY_PREFIXES


def _ontology_distance(
    aliases: dict[str, Any],
    source_oid: Optional[str],
    target_oid: Optional[str],
) -> Optional[int]:
    """Look up the ontology distance between two ontology IDs.

    Uses an explicit ``ontology_distances`` table when present; otherwise
    infers distance 1 for direct parent/child links and 2 for shared parents
    (siblings) or grandparent/grandchild links declared in the config.
    """
    if not source_oid or not target_oid:
        return None
    if source_oid == target_oid:
        return 0

    explicit = aliases.get("ontology_distances", {})
    if isinstance(explicit, dict):
        key = f"{source_oid}|{target_oid}"
        rev = f"{target_oid}|{source_oid}"
        if key in explicit:
            return int(explicit[key])
        if rev in explicit:
            return int(explicit[rev])

    # Infer from parent/child adjacency declared per context.
    ctxs = _canonical_contexts(aliases)
    by_oid = {c.get("ontology_id"): c for c in ctxs if c.get("ontology_id")}
    src = by_oid.get(source_oid, {})
    tgt = by_oid.get(target_oid, {})
    src_parents = set(src.get("parents", []) or [])
    tgt_parents = set(tgt.get("parents", []) or [])
    src_children = set(src.get("children", []) or [])
    tgt_children = set(tgt.get("children", []) or [])

    if target_oid in src_parents or target_oid in src_children:
        return 1
    if source_oid in tgt_parents or source_oid in tgt_children:
        return 1
    # Siblings share a parent.
    if src_parents & tgt_parents:
        return 2
    # Grandparent / grandchild.
    for p in src_parents:
        gp = by_oid.get(p, {})
        if target_oid in (gp.get("parents", []) or []) or target_oid in (gp.get("children", []) or []):
            return 2
    return None


def _resolve_source_ontology_id(
    context: str,
    aliases: dict[str, Any],
) -> Optional[str]:
    """Resolve a source context to its ontology ID via name/synonym lookup."""
    if not context:
        return None
    token = context.strip()
    if _is_ontology_id(token):
        return token
    ctxs = _canonical_contexts(aliases)
    norm = normalize_context(context)
    for target in ctxs:
        if normalize_context(target.get("name", "")) == norm:
            return target.get("ontology_id")
        for syn in target.get("synonyms", []) or []:
            if normalize_context(syn) == norm:
                return target.get("ontology_id")
    return None


def _format_match(
    level: int,
    target: dict[str, Any],
    ontology_distance: Optional[int] = None,
) -> dict[str, Any]:
    """Build a standardized match-result dict."""
    return {
        "matched_context": target.get("name"),
        "matched_ontology_id": target.get("ontology_id"),
        "mapping_method": _LEVEL_METHOD[level],
        "mapping_level": level,
        "ontology_distance": ontology_distance,
        "mapping_confidence": _LEVEL_CONFIDENCE[level],
    }


# ---------------------------------------------------------------------------
# Level matchers
# ---------------------------------------------------------------------------

def map_context_level1(
    context: str,
    aliases: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Level 1: exact ontology-ID match.

    ``context`` may be either a bare ontology ID (e.g. ``CL:0000057``) or a
    name. If it is an ontology ID, it is matched directly against each
    canonical context's ``ontology_id``.

    Returns
    -------
    dict or None
        Match result, or ``None`` if no exact ontology-ID match is found.
    """
    ctxs = _canonical_contexts(aliases)
    oid = context.strip() if context else ""
    for target in ctxs:
        if target.get("ontology_id") and target["ontology_id"] == oid:
            return _format_match(1, target, ontology_distance=0)
    return None


def map_context_level2(
    context: str,
    aliases: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Level 2: normalized exact string match on context name."""
    ctxs = _canonical_contexts(aliases)
    norm = normalize_context(context)
    if not norm:
        return None
    for target in ctxs:
        if normalize_context(target.get("name", "")) == norm:
            return _format_match(2, target, ontology_distance=0)
    return None


def map_context_level3(
    context: str,
    aliases: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Level 3: synonym match.

    The normalized source name is compared against each canonical context's
    ``synonyms`` list (also normalized).
    """
    ctxs = _canonical_contexts(aliases)
    norm = normalize_context(context)
    if not norm:
        return None
    for target in ctxs:
        synonyms = target.get("synonyms", []) or []
        for syn in synonyms:
            if normalize_context(syn) == norm:
                return _format_match(3, target, ontology_distance=0)
    return None


def map_context_level4(
    context: str,
    aliases: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Level 4: ontology parent/child distance ≤ 1.

    Requires the source context to resolve to an ontology ID (either directly
    or via name/synonym lookup) and finds a canonical context at ontology
    distance exactly 1.
    """
    ctxs = _canonical_contexts(aliases)
    source_oid = _resolve_source_ontology_id(context, aliases)
    if not source_oid:
        return None
    for target in ctxs:
        target_oid = target.get("ontology_id")
        dist = _ontology_distance(aliases, source_oid, target_oid)
        if dist is not None and dist == 1:
            return _format_match(4, target, ontology_distance=1)
    return None


def map_context_level5(
    context: str,
    aliases: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Level 5: ontology distance ≤ 2 (and > 1)."""
    ctxs = _canonical_contexts(aliases)
    source_oid = _resolve_source_ontology_id(context, aliases)
    if not source_oid:
        return None
    for target in ctxs:
        target_oid = target.get("ontology_id")
        dist = _ontology_distance(aliases, source_oid, target_oid)
        if dist is not None and 0 < dist <= 2:
            return _format_match(5, target, ontology_distance=dist)
    return None


def map_context_level6(
    context: str,
    aliases: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Level 6: published scE2G mapping.

    Uses the ``sce2g_mapping`` crosswalk in the aliases config, which maps
    source context names (normalized) to canonical context names.
    """
    sce2g = aliases.get("sce2g_mapping", aliases.get("scE2G_mapping", {})) or {}
    if not isinstance(sce2g, dict):
        return None
    norm = normalize_context(context)
    if not norm:
        return None
    # Try both raw and normalized keys.
    canonical_name = sce2g.get(context) or sce2g.get(norm)
    if not canonical_name:
        for k, v in sce2g.items():
            if normalize_context(k) == norm:
                canonical_name = v
                break
    if not canonical_name:
        return None
    ctxs = _canonical_contexts(aliases)
    for target in ctxs:
        if normalize_context(target.get("name", "")) == normalize_context(canonical_name):
            return _format_match(6, target, ontology_distance=None)
    # If the canonical name isn't in the context list, still report the match.
    return {
        "matched_context": canonical_name,
        "matched_ontology_id": None,
        "mapping_method": _LEVEL_METHOD[6],
        "mapping_level": 6,
        "ontology_distance": None,
        "mapping_confidence": _LEVEL_CONFIDENCE[6],
    }


# ---------------------------------------------------------------------------
# Top-level cascade
# ---------------------------------------------------------------------------

def map_context(
    source_context: str,
    target_contexts: Optional[list[str]] = None,
    aliases: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Map a source context to a canonical context via the 6-level cascade.

    Levels are tried in order 1 → 6; the first non-``None`` result is
    returned. If ``target_contexts`` is provided, matching is restricted to
    canonical contexts whose name appears in that list (results whose
    ``matched_context`` is not in ``target_contexts`` are skipped).

    Parameters
    ----------
    source_context
        The context label to map.
    target_contexts
        Optional allow-list of canonical context names to restrict matching.
    aliases
        Parsed alias configuration (from :func:`load_context_aliases`).

    Returns
    -------
    dict or None
        Best match result with keys ``matched_context``, ``matched_ontology_id``,
        ``mapping_method``, ``mapping_level``, ``ontology_distance``, and
        ``mapping_confidence``; or ``None`` if no level matches.
    """
    if aliases is None:
        raise ValueError("map_context: 'aliases' config is required")
    if not source_context:
        return None

    level_fns = (
        map_context_level1,
        map_context_level2,
        map_context_level3,
        map_context_level4,
        map_context_level5,
        map_context_level6,
    )

    target_set = (
        {normalize_context(t) for t in target_contexts}
        if target_contexts
        else None
    )

    for fn in level_fns:
        try:
            result = fn(source_context, aliases)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Context mapping level %s raised: %s", fn.__name__, exc)
            continue
        if result is None:
            continue
        if target_set is not None:
            matched_norm = normalize_context(result.get("matched_context") or "")
            if matched_norm not in target_set:
                continue
        result["source_context"] = source_context
        return result

    return None


__all__ = [
    "normalize_context",
    "load_context_aliases",
    "map_context",
    "map_context_level1",
    "map_context_level2",
    "map_context_level3",
    "map_context_level4",
    "map_context_level5",
    "map_context_level6",
]
