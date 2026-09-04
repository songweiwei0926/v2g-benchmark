#!/usr/bin/env python3
"""Build evidence adapter outputs from raw data sources.

Parses raw data from all benchmark sources into the canonical evidence schema
and writes one parquet file per source. These are then combined by
build_gold_registry.py into evidence_long.parquet.

Outputs:
  - data/interim/adapter_gtex.parquet
  - data/interim/adapter_eqtl_catalogue.parquet
  - data/interim/adapter_crispr.parquet
  - data/interim/adapter_gwas.parquet
  - data/interim/adapter_opentargets.parquet
"""

from __future__ import annotations

import gzip
import hashlib
import io
import re
import sys
import tarfile
from pathlib import Path

import polars as pl

PROJECT = Path("/workspace/v2g-benchmark")

# Evidence columns (must match EVIDENCE_COLUMNS in gold_registry.py)
EVIDENCE_COLS = [
    "benchmark_id", "evidence_id", "variant_id", "element_id", "gene_id",
    "context_id", "trait_id", "evidence_type", "label", "effect_size",
    "effect_direction", "pip", "pvalue", "source_dataset",
    "source_publication", "confidence", "training_overlap",
]


def _deversion_gene_id(gid: str) -> str:
    """Remove version suffix from Ensembl gene ID."""
    return re.sub(r"\.\d+$", "", gid) if gid else gid


def _make_evidence_id(*parts) -> str:
    """Create a deterministic evidence ID."""
    raw = "|".join(str(p) for p in parts)
    return "EV_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_variant_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Create canonical variant ID."""
    return f"GRCh38:{chrom}:{pos}:{ref}:{alt}"


def _load_context_mapping() -> dict[str, dict]:
    """Load context mapping from parquet."""
    ctx_map = pl.read_parquet(str(PROJECT / "data/processed/context_mapping.parquet"))
    mapping = {}
    for row in ctx_map.iter_rows(named=True):
        mapping[row["raw_context"]] = row
    return mapping


# ---------------------------------------------------------------------------
# GTEx v11 SuSiE adapter
# ---------------------------------------------------------------------------

def adapt_gtex(output_path: Path) -> pl.DataFrame:
    """Parse GTEx v11 SuSiE tar into evidence schema."""
    tar_path = PROJECT / "data/raw/gtex_v11_susie.tar"
    ctx_map = _load_context_mapping()

    print("  GTEx: extracting from tar...")
    rows = []
    with tarfile.open(str(tar_path), "r") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".parquet"):
                continue
            # Extract tissue name from filename
            tissue = member.name.split("/")[-1].replace(".parquet", "")
            # Map to context_id
            ctx_info = ctx_map.get(tissue, {})
            context_id = ctx_info.get("context_id", "CTX_UNKNOWN")
            
            # Read parquet from tar
            f = tar.extractfile(member)
            if f is None:
                continue
            data = pl.read_parquet(io.BytesIO(f.read()))
            
            for row in data.iter_rows(named=True):
                gene_id = _deversion_gene_id(row.get("phenotype_id", ""))
                variant_id_raw = row.get("variant_id", "")
                pip = float(row.get("pip", 0.0))
                afc = row.get("afc")
                afc_se = row.get("afc_se")
                
                # Parse variant_id: chr1_285155_A_C_b38
                m = re.match(r"^(chr\w+)_(\d+)_([ACGTN]+)_([ACGTN]+)(?:_b\d+)?$", variant_id_raw)
                if not m:
                    continue
                chrom, pos, ref, alt = m.group(1), int(m.group(2)), m.group(3), m.group(4)
                variant_id = _make_variant_id(chrom, pos, ref, alt)
                
                # Label: 1 if pip >= 0.90, 0 if pip < 0.01, skip otherwise
                if pip >= 0.90:
                    label = 1
                elif pip < 0.01:
                    label = 0
                else:
                    continue  # Skip intermediate PIP for gold registry
                
                effect_size = float(afc) if afc is not None else None
                if effect_size is not None:
                    direction = "up" if effect_size > 0 else "down" if effect_size < 0 else "none"
                else:
                    direction = "unknown"
                
                rows.append({
                    "benchmark_id": "GTEx_V11",
                    "evidence_id": _make_evidence_id(variant_id, gene_id, tissue),
                    "variant_id": variant_id,
                    "element_id": None,
                    "gene_id": gene_id,
                    "context_id": context_id,
                    "trait_id": None,
                    "evidence_type": "eQTL",
                    "label": label,
                    "effect_size": effect_size,
                    "effect_direction": direction,
                    "pip": pip,
                    "pvalue": None,
                    "source_dataset": "GTEx_V11",
                    "source_publication": "GTEx Consortium 2023",
                    "confidence": pip,
                    "training_overlap": "UNKNOWN",
                })
    
    df = pl.DataFrame(rows, schema={c: pl.Utf8 if c in ("benchmark_id","evidence_id","variant_id","element_id","gene_id","context_id","trait_id","evidence_type","effect_direction","source_dataset","source_publication","training_overlap") else pl.Float64 if c in ("effect_size","pip","pvalue","confidence") else pl.Int64 for c in EVIDENCE_COLS})
    df.write_parquet(str(output_path))
    print(f"  GTEx: {df.height} evidence rows -> {output_path}")
    return df


# ---------------------------------------------------------------------------
# eQTL Catalogue adapter
# ---------------------------------------------------------------------------

def adapt_eqtl_catalogue(output_path: Path) -> pl.DataFrame:
    """Parse eQTL Catalogue credible sets into evidence schema."""
    eqtl_dir = PROJECT / "data/raw/eqtl_catalogue_stable"
    ctx_map = _load_context_mapping()

    print("  eQTL Catalogue: scanning...")
    rows = []
    
    for study_dir in sorted(eqtl_dir.rglob("QTD000001")):
        study_id = study_dir.parent.name  # QTS000001, etc.
        ctx_info = ctx_map.get(study_id, {})
        context_id = ctx_info.get("context_id", "CTX_UNKNOWN")
        
        for tsv_file in study_dir.glob("*.tsv.gz"):
            df = pl.read_csv(str(tsv_file), separator="\t", infer_schema_length=10000)
            if "variant" not in df.columns:
                continue
            
            for row in df.iter_rows(named=True):
                variant_raw = row.get("variant", "")
                # Parse: chr1_108004887_G_T
                m = re.match(r"^(chr\w+)_(\d+)_([ACGTN]+)_([ACGTN]+)$", variant_raw)
                if not m:
                    continue
                chrom, pos, ref, alt = m.group(1), int(m.group(2)), m.group(3), m.group(4)
                variant_id = _make_variant_id(chrom, pos, ref, alt)
                
                gene_id = _deversion_gene_id(row.get("gene_id", row.get("phenotype_id", "")))
                pip = float(row.get("pip", 0.0))
                
                if pip >= 0.90:
                    label = 1
                elif pip < 0.01:
                    label = 0
                else:
                    continue
                
                rows.append({
                    "benchmark_id": "eQTLCatalogue",
                    "evidence_id": _make_evidence_id(variant_id, gene_id, study_id),
                    "variant_id": variant_id,
                    "element_id": None,
                    "gene_id": gene_id,
                    "context_id": context_id,
                    "trait_id": None,
                    "evidence_type": "eQTL",
                    "label": label,
                    "effect_size": float(row.get("beta", row.get("afc", 0.0))) if row.get("beta") or row.get("afc") else None,
                    "effect_direction": "unknown",
                    "pip": pip,
                    "pvalue": None,
                    "source_dataset": "eQTLCatalogue",
                    "source_publication": "eQTL Catalogue 2022",
                    "confidence": pip,
                    "training_overlap": "UNKNOWN",
                })
    
    if rows:
        df = pl.DataFrame(rows)
        df = df.select(EVIDENCE_COLS)
    else:
        df = pl.DataFrame(schema={c: pl.Utf8 for c in EVIDENCE_COLS})
    df.write_parquet(str(output_path))
    print(f"  eQTL Catalogue: {df.height} evidence rows -> {output_path}")
    return df


# ---------------------------------------------------------------------------
# CRISPR adapter
# ---------------------------------------------------------------------------

def adapt_crispr(output_path: Path) -> pl.DataFrame:
    """Parse EPCrisprBenchmark data into evidence schema."""
    crispr_path = PROJECT / "data/raw/crispr_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"
    ctx_map = _load_context_mapping()

    print("  CRISPR: parsing EPCrisprBenchmark...")
    df = pl.read_csv(str(crispr_path), separator="\t", infer_schema_length=10000)
    
    rows = []
    for row in df.iter_rows(named=True):
        chrom = row.get("chrom", "")
        start = int(row.get("chromStart", 0))
        end = int(row.get("chromEnd", 0))
        gene_symbol = row.get("measuredGeneSymbol", "")
        gene_id = _deversion_gene_id(row.get("measuredGeneEnsemblId", ""))
        effect_size = float(row.get("EffectSize", 0.0))
        regulated = row.get("Regulated", False)
        cell_type = row.get("CellType", "")
        
        ctx_info = ctx_map.get(cell_type, {})
        context_id = ctx_info.get("context_id", "CTX_UNKNOWN")
        
        element_id = f"{chrom}:{start}-{end}"
        
        # Label from author-curated Regulated column (true positive vs true negative).
        # Regulated distinguishes powered negatives from mere P>0.05 non-significance.
        label = 1 if regulated else 0
        
        direction = "up" if effect_size > 0 else "down" if effect_size < 0 else "none"
        
        rows.append({
            "benchmark_id": "ENCODE_CRISPR",
            "evidence_id": _make_evidence_id(element_id, gene_id, cell_type),
            "variant_id": None,
            "element_id": element_id,
            "gene_id": gene_id,
            "context_id": context_id,
            "trait_id": None,
            "evidence_type": "CRISPRi",
            "label": label,
            "effect_size": effect_size,
            "effect_direction": direction,
            "pip": None,
            "pvalue": float(row.get("pValueAdjusted", 0.0)) if row.get("pValueAdjusted") else None,
            "source_dataset": "EPCrisprBenchmark",
            "source_publication": "Fulco et al. 2019, Gasperini et al. 2019, Schraivogel et al. 2020",
            "confidence": 1.0 if label == 1 else 0.5,
            "training_overlap": "UNKNOWN",
        })
    
    df_out = pl.DataFrame(rows)
    df_out = df_out.select(EVIDENCE_COLS)
    df_out.write_parquet(str(output_path))
    print(f"  CRISPR: {df_out.height} evidence rows -> {output_path}")
    return df_out


# ---------------------------------------------------------------------------
# GWAS E2G adapter
# ---------------------------------------------------------------------------

def adapt_gwas(output_path: Path) -> pl.DataFrame:
    """Parse GWAS E2G benchmarking data into evidence schema."""
    gwas_dir = PROJECT / "data/raw/gwas_e2g_benchmarking/resources/191010_UKBB_SuSiE_hg38_liftover"
    ctx_map = _load_context_mapping()

    print("  GWAS: scanning variant lists...")
    rows = []
    
    for trait_dir in sorted(gwas_dir.iterdir()):
        if not trait_dir.is_dir():
            continue
        trait = trait_dir.name
        # GWAS traits map to GLOBAL_GWAS
        ctx_info = ctx_map.get(trait, {})
        context_id = ctx_info.get("context_id", "CTX_GLOBAL_GWAS")
        
        variant_file = trait_dir / "variant.list.txt"
        if not variant_file.exists():
            continue
        
        df = pl.read_csv(str(variant_file), separator="\t", infer_schema_length=10000)
        if "chromosome" not in df.columns or "end" not in df.columns:
            continue
        
        for row in df.iter_rows(named=True):
            chrom = row.get("chromosome", "")
            pos = int(row.get("end", 0))  # end is 1-based position
            ref = str(row.get("allele1", ""))
            alt = str(row.get("allele2", ""))
            pip = float(row.get("pip", 0.0))
            
            if not ref or not alt or not chrom:
                continue
            
            variant_id = _make_variant_id(chrom, pos, ref, alt)
            
            if pip >= 0.90:
                label = 1
            elif pip < 0.01:
                label = 0
            else:
                continue
            
            rows.append({
                "benchmark_id": "GWAS_E2G",
                "evidence_id": _make_evidence_id(variant_id, trait),
                "variant_id": variant_id,
                "element_id": None,
                "gene_id": "UNKNOWN",  # GWAS variants don't have direct gene associations
                "context_id": context_id,
                "trait_id": trait,
                "evidence_type": "GWAS",
                "label": label,
                "effect_size": float(row.get("beta_posterior", 0.0)) if row.get("beta_posterior") else None,
                "effect_direction": "up" if float(row.get("beta_posterior", 0)) > 0 else "down" if float(row.get("beta_posterior", 0)) < 0 else "unknown",
                "pip": pip,
                "pvalue": None,
                "source_dataset": "GWAS_E2G",
                "source_publication": "UKBB SuSiE 2019",
                "confidence": pip,
                "training_overlap": "UNKNOWN",
            })
    
    if rows:
        df = pl.DataFrame(rows)
        df = df.select(EVIDENCE_COLS)
    else:
        df = pl.DataFrame(schema={c: pl.Utf8 for c in EVIDENCE_COLS})
    df.write_parquet(str(output_path))
    print(f"  GWAS: {df.height} evidence rows -> {output_path}")
    return df


# ---------------------------------------------------------------------------
# OpenTargets adapter
# ---------------------------------------------------------------------------

def adapt_opentargets(output_path: Path) -> pl.DataFrame:
    """Parse OpenTargets gold standards into evidence schema."""
    ot_path = PROJECT / "data/raw/opentargets_gold_standards/gold_standards/processed/gwas_gold_standards.191108.tsv"

    print("  OpenTargets: parsing gold standards...")
    df = pl.read_csv(str(ot_path), separator="\t", infer_schema_length=10000)
    
    rows = []
    for row in df.iter_rows(named=True):
        chrom = row.get("sentinel_variant.locus_GRCh38.chromosome", "")
        pos = int(row.get("sentinel_variant.locus_GRCh38.position", 0))
        ref = str(row.get("sentinel_variant.alleles.reference", ""))
        alt = str(row.get("sentinel_variant.alleles.alternative", ""))
        
        if not chrom or not ref or not alt:
            continue
        
        variant_id = _make_variant_id(chrom, pos, ref, alt)
        gene_id = _deversion_gene_id(row.get("gene_id", row.get("target_id", "")))
        
        if not gene_id or gene_id == "None":
            continue
        
        rows.append({
            "benchmark_id": "OpenTargets_GoldStandard",
            "evidence_id": _make_evidence_id(variant_id, gene_id, "OT"),
            "variant_id": variant_id,
            "element_id": None,
            "gene_id": gene_id,
            "context_id": "CTX_GLOBAL_GWAS",
            "trait_id": row.get("disease_id", row.get("trait_id", "")),
            "evidence_type": "curated_L2G",
            "label": 1,
            "effect_size": None,
            "effect_direction": "unknown",
            "pip": None,
            "pvalue": None,
            "source_dataset": "OpenTargets_GoldStandard",
            "source_publication": "OpenTargets 2019",
            "confidence": 0.9,
            "training_overlap": "UNKNOWN",
        })
    
    if rows:
        df_out = pl.DataFrame(rows)
        df_out = df_out.select(EVIDENCE_COLS)
    else:
        df_out = pl.DataFrame(schema={c: pl.Utf8 for c in EVIDENCE_COLS})
    df_out.write_parquet(str(output_path))
    print(f"  OpenTargets: {df_out.height} evidence rows -> {output_path}")
    return df_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    interim = PROJECT / "data/interim"
    interim.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Building evidence adapter outputs")
    print("=" * 70)

    # Build all adapters
    adapt_gtex(interim / "adapter_gtex.parquet")
    adapt_eqtl_catalogue(interim / "adapter_eqtl_catalogue.parquet")
    adapt_crispr(interim / "adapter_crispr.parquet")
    adapt_gwas(interim / "adapter_gwas.parquet")
    adapt_opentargets(interim / "adapter_opentargets.parquet")

    print("\nAll adapter outputs built successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
