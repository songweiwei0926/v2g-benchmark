"""AlphaGenome API adapter (Family 3 — sequence models, remote inference).

AlphaGenome is accessed through DeepMind's hosted API.  For each
(variant, context) pair we issue one request that returns gene-centric RNA
scores; we convert these into a signed effect
``log(ALT + eps) - log(REF + eps)`` and an absolute value, plus a quantile
score across all genes for the variant.

Because the API is rate-limited and occasionally flaky, the inference loop
implements:

* exponential backoff with jitter on retryable errors,
* a persistent on-disk cache keyed by ``(model_version, variant_id,
  context_ontology, scorer)`` so partial runs can be resumed,
* periodic checkpointing of accumulated results.

Output columns: ``alphagenome_signed``, ``alphagenome_abs``,
``alphagenome_quantile``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from .base import ModelAdapter

logger = logging.getLogger(__name__)

EPS = 0.001  # pseudocount for log-ratio
DEFAULT_MODEL_VERSION = "alpha_genome_2025"
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0  # seconds


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
def create_client(api_key: str):
    """Create an AlphaGenome ``dna_client``.

    Wraps ``alphagenome.models.dna_client.Client`` so the import is lazy and
    the module remains importable without the package installed.
    """
    try:
        from alphagenome.models import dna_client  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "alphagenome is required for AlphaGenome inference. "
            "Install with `pip install alphagenome`."
        ) from exc
    return dna_client.Client(api_key=api_key)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def compute_signed_score(ref_expr: float, alt_expr: float) -> float:
    """Signed log-fold-change: ``log(ALT + eps) - log(REF + eps)``."""
    import math
    return math.log(alt_expr + EPS) - math.log(ref_expr + EPS)


def score_variant_alphagenome(client, variant: Dict[str, Any], context: Dict[str, Any],
                              gene_master_df: pl.DataFrame) -> pl.DataFrame:
    """Score one variant in one context via the AlphaGenome API.

    Returns a frame with ``variant_id, gene_id, context_id,
    alphagenome_signed, alphagenome_abs``.

    ``variant`` must contain ``variant_id, chrom, pos, ref, alt``.
    ``context`` must contain ``context_id`` and ``ontology_id`` (UBERON/CL).
    """
    from alphagenome.models import dna_client  # type: ignore

    interval = dna_client.Interval(
        chromosome=variant["chrom"],
        start=int(variant["pos"]),
        end=int(variant["pos"]) + len(variant["ref"]),
    )
    variant_obj = dna_client.Variant(
        chromosome=variant["chrom"],
        start=int(variant["pos"]),
        reference_bases=variant["ref"],
        alternate_bases=variant["alt"],
    )

    scorer = context.get("scorer", "RNA_SEQ")
    gene_scorer = context.get("gene_scorer", "GeneMaskLFC")
    ontology = context.get("ontology_id", "")

    output = client.score_variants(
        interval=interval,
        variants=[variant_obj],
        ontology_terms=[ontology] if ontology else None,
        scoring=scorer,
        gene_scoring=gene_scorer,
    )

    # The API returns gene-centric scores; normalize to a dict gene_id → score.
    raw = _extract_gene_scores(output, variant["variant_id"], context["context_id"])

    if not raw:
        return pl.DataFrame(schema={
            "variant_id": pl.Utf8, "gene_id": pl.Utf8, "context_id": pl.Utf8,
            "alphagenome_signed": pl.Float64, "alphagenome_abs": pl.Float64,
        })

    rows: List[Dict[str, Any]] = []
    for gene_id, (ref_val, alt_val) in raw.items():
        signed = compute_signed_score(ref_val, alt_val)
        rows.append({
            "variant_id": variant["variant_id"],
            "gene_id": gene_id,
            "context_id": context["context_id"],
            "alphagenome_signed": float(signed),
            "alphagenome_abs": float(abs(signed)),
        })
    return pl.DataFrame(rows, schema={
        "variant_id": pl.Utf8, "gene_id": pl.Utf8, "context_id": pl.Utf8,
        "alphagenome_signed": pl.Float64, "alphagenome_abs": pl.Float64,
    })


def _extract_gene_scores(output, variant_id: str, context_id: str) -> Dict[str, tuple]:
    """Best-effort extraction of (ref, alt) gene scores from an API response.

    The AlphaGenome SDK response object exposes ``.values`` / ``.genes``; we
    handle both dict-like and attribute-like shapes defensively.
    """
    raw: Dict[str, tuple] = {}
    try:
        records = output.gene_scores if hasattr(output, "gene_scores") else output
        if hasattr(records, "to_dict"):
            records = records.to_dict(orient="records")
        for rec in records:
            gene_id = rec.get("gene_id") or rec.get("gene")
            ref_val = float(rec.get("reference") or rec.get("ref") or rec.get("ref_value") or 0.0)
            alt_val = float(rec.get("alternate") or rec.get("alt") or rec.get("alt_value") or 0.0)
            if gene_id:
                raw[str(gene_id)] = (ref_val, alt_val)
    except Exception as exc:  # pragma: no cover - depends on SDK shape
        logger.warning("AlphaGenome: could not parse gene scores for %s: %s", variant_id, exc)
    return raw


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def cache_key(model_version: str, variant_id: str, context_ontology: str,
              scorer: str) -> str:
    """SHA256 cache key for a single (variant, context, scorer) request."""
    s = "|".join([model_version, variant_id, context_ontology, scorer])
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.json"


def load_cache(cache_dir: str | Path) -> Dict[str, Any]:
    """Load the entire cache directory into a dict keyed by cache key."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return {}
    out: Dict[str, Any] = {}
    for f in cache_dir.glob("*.json"):
        try:
            out[f.stem] = json.loads(f.read_text())
        except Exception:
            logger.warning("AlphaGenome: corrupt cache file %s — skipping", f)
    return out


def save_cache(cache_dir: str | Path, key: str, result: Any) -> None:
    """Persist a single cache entry as JSON."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(cache_dir, key).write_text(json.dumps(result, default=str))


# --------------------------------------------------------------------------- #
# Main inference loop
# --------------------------------------------------------------------------- #
def run_alphagenome_inference(variants_df: pl.DataFrame, gene_master_df: pl.DataFrame,
                              output_dir: str | Path, api_key: str,
                              batch_size: int = 50,
                              model_version: str = DEFAULT_MODEL_VERSION,
                              cache_dir: Optional[str | Path] = None,
                              contexts: Optional[List[Dict[str, Any]]] = None,
                              scorer: str = "RNA_SEQ") -> pl.DataFrame:
    """Run AlphaGenome inference with retry/backoff and checkpointing.

    ``contexts`` is a list of context dicts (``context_id``, ``ontology_id``).
    If omitted, a single default context is used per variant.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else output_dir / "cache"
    cache = load_cache(cache_dir)
    client = create_client(api_key)

    variants = variants_df.to_dicts()
    if contexts is None:
        contexts = [{"context_id": "default", "ontology_id": "", "scorer": scorer}]

    results: List[Dict[str, Any]] = []
    n_done = 0

    for v in variants:
        for ctx in contexts:
            key = cache_key(model_version, v["variant_id"],
                            ctx.get("ontology_id", ""), ctx.get("scorer", scorer))
            if key in cache:
                cached = cache[key]
                if isinstance(cached, list):
                    results.extend(cached)
                continue

            attempt = 0
            backoff = INITIAL_BACKOFF
            while True:
                try:
                    df = score_variant_alphagenome(client, v, ctx, gene_master_df)
                    rows = df.to_dicts()
                    save_cache(cache_dir, key, rows)
                    results.extend(rows)
                    break
                except Exception as exc:
                    attempt += 1
                    if attempt > MAX_RETRIES:
                        logger.error("AlphaGenome: giving up on %s/%s after %d tries: %s",
                                     v["variant_id"], ctx["context_id"], attempt, exc)
                        break
                    sleep = backoff * (1 + (hash(v["variant_id"]) % 100) / 100.0)  # jitter
                    logger.warning("AlphaGenome: retry %d for %s/%s (sleep %.1fs): %s",
                                   attempt, v["variant_id"], ctx["context_id"], sleep, exc)
                    time.sleep(sleep)
                    backoff *= 2

            n_done += 1
            if n_done % batch_size == 0:
                _checkpoint(results, output_dir)

    final = _build_frame(results)
    final.write_parquet(output_dir / "alphagenome_predictions.parquet")
    return final


def _checkpoint(results: List[Dict[str, Any]], output_dir: Path) -> None:
    if not results:
        return
    _build_frame(results).write_parquet(output_dir / "alphagenome_checkpoint.parquet")


def _build_frame(results: List[Dict[str, Any]]) -> pl.DataFrame:
    schema = {
        "variant_id": pl.Utf8, "gene_id": pl.Utf8, "context_id": pl.Utf8,
        "alphagenome_signed": pl.Float64, "alphagenome_abs": pl.Float64,
    }
    if not results:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(results, schema=schema, infer_schema_length=None)
    # Add per-variant quantile of |signed| (rank within variant).
    df = df.with_columns(
        pl.col("alphagenome_abs").rank("average").over("variant_id")
        .alias("_rank")
    ).with_columns(
        (pl.col("_rank") / pl.col("alphagenome_abs").count().over("variant_id"))
        .alias("alphagenome_quantile")
    ).drop("_rank")
    return df


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class AlphaGenomeAdapter(ModelAdapter):
    """ModelAdapter wrapping the AlphaGenome remote inference API."""

    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        super().__init__(model_id="alphagenome", model_family="sequence", config=config)
        self.model_version = config.get("model_version", DEFAULT_MODEL_VERSION)
        self.scorer = config.get("scorer", "RNA_SEQ")
        self.gene_scorer = config.get("gene_scorer", "GeneMaskLFC")
        self.sequence_length = int(config.get("sequence_length", 262_144))

    def validate_resources(self) -> bool:
        api_key = self.config.get("api_key")
        if not api_key:
            return False
        try:
            import alphagenome  # noqa: F401
        except ImportError:
            return False
        return True

    def applicability(self, context_id: str) -> str:
        return "APPLICABLE"

    def score(self, inputs: Dict[str, Any]) -> pl.DataFrame:
        variants_df: pl.DataFrame = inputs["variants_df"]
        gene_master_df: pl.DataFrame = inputs["gene_master_df"]
        output_dir = inputs.get("output_dir", "./predictions/alphagenome")
        api_key = inputs.get("api_key") or self.config.get("api_key")
        contexts = inputs.get("contexts")
        cache_dir = inputs.get("cache_dir")

        df = run_alphagenome_inference(
            variants_df, gene_master_df, output_dir, api_key,
            batch_size=int(self.config.get("batch_size", 50)),
            model_version=self.model_version,
            cache_dir=cache_dir, contexts=contexts, scorer=self.scorer,
        )
        return self.normalize_score(df)

    def normalize_score(self, df: pl.DataFrame) -> pl.DataFrame:
        # ranking_score = quantile of |signed| within each variant.
        if "alphagenome_quantile" in df.columns:
            df = df.with_columns(pl.col("alphagenome_quantile").alias("ranking_score"))
        elif "alphagenome_abs" in df.columns:
            df = df.with_columns(pl.col("alphagenome_abs").alias("ranking_score"))
        return df
