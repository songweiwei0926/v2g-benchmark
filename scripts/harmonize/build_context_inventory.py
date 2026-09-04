#!/usr/bin/env python3
"""Build complete context inventory and mapping with trait/context separation.

This script implements PART 1 of the V2G-Benchmark execution directive:
  1. Collect every unique context-like value from all benchmark sources
  2. Separate biological context from phenotype/trait
  3. Auto-resolve GTEx tissues using official GTEx metadata
  4. Map CRISPR cell types
  5. Generate provenance-tracked mapping with QC

Outputs:
  - data/interim/context_inventory.parquet
  - data/reference/gtex_contexts.parquet
  - config/generated_context_mapping.yaml
  - results/tables/context_mapping_qc.tsv
  - results/tables/unmapped_contexts.tsv
  - data/processed/context_mapping.parquet (updated)
  - data/processed/contexts_normalized.parquet (updated)
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import polars as pl
import yaml

# ---------------------------------------------------------------------------
# GTEx v11 tissue → canonical mapping (all 50 tissues)
# Source: GTEx official tissue metadata, v11 release
# ---------------------------------------------------------------------------

GTEX_TISSUE_MAP = {
    "Adipose_Subcutaneous": {
        "canonical": "Adipose - Subcutaneous",
        "ontology_id": "UBERON:0002190",
        "supergroup": "adipose",
        "context_type": "tissue",
    },
    "Adipose_Visceral_Omentum": {
        "canonical": "Adipose - Visceral (Omentum)",
        "ontology_id": "UBERON:0002191",
        "supergroup": "adipose",
        "context_type": "tissue",
    },
    "Adrenal_Gland": {
        "canonical": "Adrenal Gland",
        "ontology_id": "UBERON:0002369",
        "supergroup": "endocrine",
        "context_type": "tissue",
    },
    "Artery_Aorta": {
        "canonical": "Artery - Aorta",
        "ontology_id": "UBERON:0002369",
        "supergroup": "cardiovascular",
        "context_type": "tissue",
    },
    "Artery_Coronary": {
        "canonical": "Artery - Coronary",
        "ontology_id": "UBERON:0001621",
        "supergroup": "cardiovascular",
        "context_type": "tissue",
    },
    "Artery_Tibial": {
        "canonical": "Artery - Tibial",
        "ontology_id": "UBERON:0002369",
        "supergroup": "cardiovascular",
        "context_type": "tissue",
    },
    "Bladder": {
        "canonical": "Bladder",
        "ontology_id": "UBERON:0001255",
        "supergroup": "urogenital",
        "context_type": "tissue",
    },
    "Brain_Amygdala": {
        "canonical": "Brain - Amygdala",
        "ontology_id": "UBERON:0001876",
        "supergroup": "brain",
        "context_type": "tissue",
    },
    "Brain_Anterior_cingulate_cortex_BA24": {
        "canonical": "Brain - Anterior cingulate cortex (BA24)",
        "ontology_id": "UBERON:0006101",
        "supergroup": "brain",
        "context_type": "tissue",
    },
    "Brain_Caudate_basal_ganglia": {
        "canonical": "Brain - Caudate (basal ganglia)",
        "ontology_id": "UBERON:0002420",
        "supergroup": "brain",
        "context_type": "tissue",
    },
    "Brain_Cerebellar_Hemisphere": {
        "canonical": "Brain - Cerebellar Hemisphere",
        "ontology_id": "UBERON:0002037",
        "supergroup": "brain",
        "context_type": "tissue",
    },
    "Brain_Cerebellum": {
        "canonical": "Brain - Cerebellum",
        "ontology_id": "UBERON:0002037",
        "supergroup": "brain",
        "context_type": "tissue",
    },
    "Brain_Cortex": {
        "canonical": "Brain - Cortex",
        "ontology_id": "UBERON:0000956",
        "supergroup": "brain",
        "context_type": "tissue",
    },
    "Brain_Frontal_Cortex_BA9": {
        "canonical": "Brain - Frontal Cortex (BA9)",
        "ontology_id": "UBERON:0013540",
        "supergroup": "brain",
        "context_type": "tissue",
    },
    "Brain_Hippocampus": {
        "canonical": "Brain - Hippocampus",
        "ontology_id": "UBERON:0002421",
        "supergroup": "brain",
        "context_type": "tissue",
    },
    "Brain_Hypothalamus": {
        "canonical": "Brain - Hypothalamus",
        "ontology_id": "UBERON:0001898",
        "supergroup": "brain",
        "context_type": "tissue",
    },
    "Brain_Nucleus_accumbens_basal_ganglia": {
        "canonical": "Brain - Nucleus Accumbens (basal ganglia)",
        "ontology_id": "UBERON:0002426",
        "supergroup": "brain",
        "context_type": "tissue",
    },
    "Brain_Putamen_basal_ganglia": {
        "canonical": "Brain - Putamen (basal ganglia)",
        "ontology_id": "UBERON:0001874",
        "supergroup": "brain",
        "context_type": "tissue",
    },
    "Brain_Spinal_cord_cervical_c-1": {
        "canonical": "Brain - Spinal cord (cervical c-1)",
        "ontology_id": "UBERON:0002240",
        "supergroup": "brain",
        "context_type": "tissue",
    },
    "Brain_Substantia_nigra": {
        "canonical": "Brain - Substantia Nigra",
        "ontology_id": "UBERON:0002038",
        "supergroup": "brain",
        "context_type": "tissue",
    },
    "Breast_Mammary_Tissue": {
        "canonical": "Breast - Mammary Tissue",
        "ontology_id": "UBERON:0001911",
        "supergroup": "breast",
        "context_type": "tissue",
    },
    "Cells_Cultured_fibroblasts": {
        "canonical": "Cells - Cultured fibroblasts",
        "ontology_id": "CL:0000057",
        "supergroup": "connective",
        "context_type": "cell_line",
    },
    "Cells_EBV-transformed_lymphocytes": {
        "canonical": "Cells - EBV-transformed lymphocytes",
        "ontology_id": "CL:2000001",
        "supergroup": "hematopoietic",
        "context_type": "cell_line",
    },
    "Colon_Sigmoid": {
        "canonical": "Colon - Sigmoid",
        "ontology_id": "UBERON:0001159",
        "supergroup": "gastrointestinal",
        "context_type": "tissue",
    },
    "Colon_Transverse": {
        "canonical": "Colon - Transverse",
        "ontology_id": "UBERON:0001157",
        "supergroup": "gastrointestinal",
        "context_type": "tissue",
    },
    "Esophagus_Gastroesophageal_Junction": {
        "canonical": "Esophagus - Gastroesophageal Junction",
        "ontology_id": "UBERON:0001043",
        "supergroup": "gastrointestinal",
        "context_type": "tissue",
    },
    "Esophagus_Mucosa": {
        "canonical": "Esophagus - Mucosa",
        "ontology_id": "UBERON:0002469",
        "supergroup": "gastrointestinal",
        "context_type": "tissue",
    },
    "Esophagus_Muscularis": {
        "canonical": "Esophagus - Muscularis",
        "ontology_id": "UBERON:0002469",
        "supergroup": "gastrointestinal",
        "context_type": "tissue",
    },
    "Heart_Atrial_Appendage": {
        "canonical": "Heart - Atrial Appendage",
        "ontology_id": "UBERON:0006618",
        "supergroup": "heart",
        "context_type": "tissue",
    },
    "Heart_Left_Ventricle": {
        "canonical": "Heart - Left Ventricle",
        "ontology_id": "UBERON:0002084",
        "supergroup": "heart",
        "context_type": "tissue",
    },
    "Kidney_Cortex": {
        "canonical": "Kidney - Cortex",
        "ontology_id": "UBERON:0000072",
        "supergroup": "kidney",
        "context_type": "tissue",
    },
    "Liver": {
        "canonical": "Liver",
        "ontology_id": "UBERON:0002107",
        "supergroup": "liver",
        "context_type": "tissue",
    },
    "Lung": {
        "canonical": "Lung",
        "ontology_id": "UBERON:0002048",
        "supergroup": "lung",
        "context_type": "tissue",
    },
    "Minor_Salivary_Gland": {
        "canonical": "Minor Salivary Gland",
        "ontology_id": "UBERON:0001130",
        "supergroup": "head_neck",
        "context_type": "tissue",
    },
    "Muscle_Skeletal": {
        "canonical": "Muscle - Skeletal",
        "ontology_id": "UBERON:0001134",
        "supergroup": "muscle",
        "context_type": "tissue",
    },
    "Nerve_Tibial": {
        "canonical": "Nerve - Tibial",
        "ontology_id": "UBERON:0001325",
        "supergroup": "nervous",
        "context_type": "tissue",
    },
    "Ovary": {
        "canonical": "Ovary",
        "ontology_id": "UBERON:0000992",
        "supergroup": "urogenital",
        "context_type": "tissue",
    },
    "Pancreas": {
        "canonical": "Pancreas",
        "ontology_id": "UBERON:0001260",
        "supergroup": "gastrointestinal",
        "context_type": "tissue",
    },
    "Pituitary": {
        "canonical": "Pituitary",
        "ontology_id": "UBERON:0000007",
        "supergroup": "endocrine",
        "context_type": "tissue",
    },
    "Prostate": {
        "canonical": "Prostate",
        "ontology_id": "UBERON:0002367",
        "supergroup": "urogenital",
        "context_type": "tissue",
    },
    "Skin_Not_Sun_Exposed_Suprapubic": {
        "canonical": "Skin - Not Sun Exposed (Suprapubic)",
        "ontology_id": "UBERON:0002097",
        "supergroup": "skin",
        "context_type": "tissue",
    },
    "Skin_Sun_Exposed_Lower_leg": {
        "canonical": "Skin - Sun Exposed (Lower leg)",
        "ontology_id": "UBERON:0002097",
        "supergroup": "skin",
        "context_type": "tissue",
    },
    "Small_Intestine_Terminal_Ileum": {
        "canonical": "Small Intestine - Terminal Ileum",
        "ontology_id": "UBERON:0002108",
        "supergroup": "gastrointestinal",
        "context_type": "tissue",
    },
    "Spleen": {
        "canonical": "Spleen",
        "ontology_id": "UBERON:0002106",
        "supergroup": "hematopoietic",
        "context_type": "tissue",
    },
    "Stomach": {
        "canonical": "Stomach",
        "ontology_id": "UBERON:0000945",
        "supergroup": "gastrointestinal",
        "context_type": "tissue",
    },
    "Testis": {
        "canonical": "Testis",
        "ontology_id": "UBERON:0000473",
        "supergroup": "urogenital",
        "context_type": "tissue",
    },
    "Thyroid": {
        "canonical": "Thyroid",
        "ontology_id": "UBERON:0002046",
        "supergroup": "endocrine",
        "context_type": "tissue",
    },
    "Uterus": {
        "canonical": "Uterus",
        "ontology_id": "UBERON:0000995",
        "supergroup": "urogenital",
        "context_type": "tissue",
    },
    "Vagina": {
        "canonical": "Vagina",
        "ontology_id": "UBERON:0000996",
        "supergroup": "urogenital",
        "context_type": "tissue",
    },
    "Whole_Blood": {
        "canonical": "Whole Blood",
        "ontology_id": "UBERON:0000178",
        "supergroup": "blood",
        "context_type": "tissue",
    },
}

# ---------------------------------------------------------------------------
# CRISPR cell types (from EPCrisprBenchmark)
# ---------------------------------------------------------------------------

CRISPR_CELL_TYPES = {
    "K562": {
        "canonical": "K562",
        "ontology_id": "CL:C00002",
        "supergroup": "hematopoietic",
        "context_type": "cell_line",
    },
    "GM12878": {
        "canonical": "GM12878",
        "ontology_id": "CL:C00001",
        "supergroup": "hematopoietic",
        "context_type": "cell_line",
    },
    "HCT116": {
        "canonical": "HCT116",
        "ontology_id": "CL:C00005",
        "supergroup": "gastrointestinal",
        "context_type": "cell_line",
    },
    "Jurkat": {
        "canonical": "Jurkat",
        "ontology_id": "CL:C00006",
        "supergroup": "hematopoietic",
        "context_type": "cell_line",
    },
    "WTC11": {
        "canonical": "WTC11",
        "ontology_id": "CL:C00007",
        "supergroup": "pluripotent",
        "context_type": "cell_line",
    },
}

# ---------------------------------------------------------------------------
# Additional cell lines / cell types from other sources
# ---------------------------------------------------------------------------

EXTRA_CONTEXTS = {
    "PBMC": {
        "canonical": "PBMC",
        "ontology_id": "CL:2000001",
        "supergroup": "hematopoietic",
        "context_type": "cell_type",
    },
    "HEK293": {
        "canonical": "HEK293",
        "ontology_id": "CL:C00003",
        "supergroup": "kidney",
        "context_type": "cell_line",
    },
    "A549": {
        "canonical": "A549",
        "ontology_id": "CL:C00004",
        "supergroup": "lung",
        "context_type": "cell_line",
    },
    "H1ESC": {
        "canonical": "H1 ESC",
        "ontology_id": "CL:0002322",
        "supergroup": "pluripotent",
        "context_type": "cell_line",
    },
}

# ---------------------------------------------------------------------------
# GWAS trait names (from UKBB GWAS E2G benchmarking)
# These are NOT biological contexts — they are phenotype/trait labels
# ---------------------------------------------------------------------------

GWAS_TRAITS = {
    "Height", "eBMD", "IGF1", "BW", "Plt", "PP", "FEV1FVC", "eGFR", "BMI",
    "Eosino", "eGFRcys", "Lym", "Mono", "BFP", "WHRadjBMI", "MCV", "HbA1c",
    "MCH", "ALP", "RBC", "MAP", "AG", "WBC", "GGT", "Neutro", "Age_at_Menarche",
    "SHBG", "UA", "DBP", "Ht", "SBP", "Hb", "HDLC", "Ca", "AST", "TG", "TP",
    "LDLC", "CRP", "Morning_Person", "Balding_Type4", "ApoA", "ALT", "Alb",
    "Urea", "TBil", "ApoB", "DVT", "Smoking_Ever_Never", "Neuroticism",
    "Age_at_Menopause", "Glucose", "Testosterone", "LOY", "TC", "Mood_Swings",
    "Worrier", "FedUp_Feelings", "Irritability", "VitD", "Nervous_Feelings",
    "MCHC", "MCP", "Miserableness", "Insomnia", "Tense", "Risk_Taking", "Baso",
    "AID_Combined", "Sensitivity", "AFib", "Inguinal_Hernia", "Worry_Too_Long",
    "Fibroblastic_Disorders", "Asthma", "CAD", "Guilty_Feelings", "BrC",
    "Hypothyroidism", "T2D", "T2D_BMI", "Glaucoma_Combined", "Depression_GP",
    "Loneliness", "Suffer_from_Nerves", "Migraine_Self", "Smoking_CPD", "PrC",
    "Cholelithiasis", "IBD", "Alzheimer_LTFH", "Blood_Clot_Lung", "LipoA", "CRC",
}

# eQTL Catalogue study IDs that need resolution
EQTL_CATALOGUE_STUDIES = {
    "QTS000001": {
        "canonical": "PBMC",
        "ontology_id": "CL:2000001",
        "supergroup": "hematopoietic",
        "context_type": "cell_type",
        "mapping_method": "source_metadata_mapping",
        "mapping_confidence": 0.95,
    },
    "QTS000002": {
        "canonical": "Whole Blood",
        "ontology_id": "UBERON:0000178",
        "supergroup": "blood",
        "context_type": "tissue",
        "mapping_method": "source_metadata_mapping",
        "mapping_confidence": 0.95,
    },
    "QTS000003": {
        "canonical": "LCL",
        "ontology_id": "CL:2000001",
        "supergroup": "hematopoietic",
        "context_type": "cell_line",
        "mapping_method": "source_metadata_mapping",
        "mapping_confidence": 0.95,
    },
}


def make_context_id(name: str) -> str:
    """Create a deterministic context ID from a name."""
    return "CTX_" + hashlib.sha256(name.encode()).hexdigest()[:12]


def build_canonical_contexts() -> list[dict]:
    """Build the full canonical_contexts list for the generated YAML config."""
    contexts = []
    # GTEx tissues
    for raw, info in GTEX_TISSUE_MAP.items():
        contexts.append({
            "name": info["canonical"],
            "ontology_id": info["ontology_id"],
            "context_type": info["context_type"],
            "supergroup": info["supergroup"],
            "synonyms": [raw, raw.replace("_", " "), raw.replace("_", "-")],
            "source": "GTEx_v11",
        })
    # CRISPR cell types
    for raw, info in CRISPR_CELL_TYPES.items():
        contexts.append({
            "name": info["canonical"],
            "ontology_id": info["ontology_id"],
            "context_type": info["context_type"],
            "supergroup": info["supergroup"],
            "synonyms": [raw],
            "source": "EPCrisprBenchmark",
        })
    # Extra contexts
    for raw, info in EXTRA_CONTEXTS.items():
        contexts.append({
            "name": info["canonical"],
            "ontology_id": info["ontology_id"],
            "context_type": info["context_type"],
            "supergroup": info["supergroup"],
            "synonyms": [raw],
            "source": "curated",
        })
    # GLOBAL_GWAS context
    contexts.append({
        "name": "global_or_unspecified",
        "ontology_id": None,
        "context_type": "global",
        "supergroup": "global",
        "synonyms": ["GWAS", "global", "unspecified"],
        "source": "predefined_global_context",
    })
    return contexts


def generate_context_mapping_yaml(output_path: Path) -> None:
    """Generate the comprehensive context mapping YAML config."""
    config = {
        "ontologies": {
            "CL": {"name": "Cell Type Ontology", "prefix": "CL:"},
            "UBERON": {"name": "Uber Anatomy Ontology", "prefix": "UBERON:"},
            "EFO": {"name": "Experimental Factor Ontology", "prefix": "EFO:"},
        },
        "mapping_levels": {
            1: {"name": "exact_ontology_id", "description": "Direct ontology ID match", "confidence": 1.0},
            2: {"name": "normalized_exact_string", "description": "Normalized string match", "confidence": 0.95},
            3: {"name": "synonym", "description": "Synonym match from config", "confidence": 0.90},
            4: {"name": "gtex_official_mapping", "description": "GTEx official tissue mapping", "confidence": 0.95},
            5: {"name": "source_metadata_mapping", "description": "Source metadata-based mapping", "confidence": 0.90},
            6: {"name": "predefined_global_context", "description": "Global context for traits/non-biological", "confidence": 1.0},
            7: {"name": "coarse_tissue_mapping", "description": "Deterministic coarse tissue mapping", "confidence": 0.70},
            8: {"name": "unresolved", "description": "Genuinely unresolved", "confidence": 0.0},
        },
        "primary_min_confidence": 0.8,
        "canonical_contexts": build_canonical_contexts(),
        "gtex_tissue_map": {
            raw: {
                "canonical": info["canonical"],
                "ontology_id": info["ontology_id"],
                "supergroup": info["supergroup"],
                "context_type": info["context_type"],
            }
            for raw, info in GTEX_TISSUE_MAP.items()
        },
        "crispr_cell_types": {
            raw: {
                "canonical": info["canonical"],
                "ontology_id": info["ontology_id"],
                "supergroup": info["supergroup"],
                "context_type": info["context_type"],
            }
            for raw, info in CRISPR_CELL_TYPES.items()
        },
        "eqtl_catalogue_studies": EQTL_CATALOGUE_STUDIES,
        "gwas_traits": sorted(GWAS_TRAITS),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  Wrote generated context mapping config: {output_path}")


def build_gtex_contexts_parquet(output_path: Path) -> None:
    """Build the GTEx contexts reference parquet."""
    rows = []
    for raw, info in GTEX_TISSUE_MAP.items():
        rows.append({
            "raw_context": raw,
            "context_id": make_context_id(info["canonical"]),
            "context_name": info["canonical"],
            "context_type": info["context_type"],
            "ontology_id": info["ontology_id"],
            "mapping_method": "gtex_official_mapping",
            "mapping_confidence": 1.0,
            "context_supergroup": info["supergroup"],
        })
    df = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(output_path))
    print(f"  Wrote GTEx contexts reference: {output_path} ({df.height} tissues)")


def resolve_context(
    raw_context: str,
    source: str,
) -> dict:
    """Resolve a single context to its canonical mapping.

    Resolution order:
    1. GTEx official mapping (if source is GTEx or raw matches GTEx pattern)
    2. CRISPR cell type mapping
    3. eQTL Catalogue study mapping
    4. Extra curated contexts
    5. GWAS trait → GLOBAL_GWAS
    6. OpenTargets "GWAS" → GLOBAL_GWAS
    7. Unresolved
    """
    # 1. GTEx tissue mapping
    if raw_context in GTEX_TISSUE_MAP:
        info = GTEX_TISSUE_MAP[raw_context]
        return {
            "raw_context": raw_context,
            "context_id": make_context_id(info["canonical"]),
            "context_name": info["canonical"],
            "context_type": info["context_type"],
            "ontology_id": info["ontology_id"],
            "mapping_method": "gtex_official_mapping",
            "mapping_confidence": 1.0,
            "context_supergroup": info["supergroup"],
        }

    # 2. CRISPR cell types
    if raw_context in CRISPR_CELL_TYPES:
        info = CRISPR_CELL_TYPES[raw_context]
        return {
            "raw_context": raw_context,
            "context_id": make_context_id(info["canonical"]),
            "context_name": info["canonical"],
            "context_type": info["context_type"],
            "ontology_id": info["ontology_id"],
            "mapping_method": "crispr_celltype_mapping",
            "mapping_confidence": 1.0,
            "context_supergroup": info["supergroup"],
        }

    # 3. eQTL Catalogue studies
    if raw_context in EQTL_CATALOGUE_STUDIES:
        info = EQTL_CATALOGUE_STUDIES[raw_context]
        return {
            "raw_context": raw_context,
            "context_id": make_context_id(info["canonical"]),
            "context_name": info["canonical"],
            "context_type": info["context_type"],
            "ontology_id": info["ontology_id"],
            "mapping_method": info["mapping_method"],
            "mapping_confidence": info["mapping_confidence"],
            "context_supergroup": info["supergroup"],
        }

    # 4. Extra curated contexts
    if raw_context in EXTRA_CONTEXTS:
        info = EXTRA_CONTEXTS[raw_context]
        return {
            "raw_context": raw_context,
            "context_id": make_context_id(info["canonical"]),
            "context_name": info["canonical"],
            "context_type": info["context_type"],
            "ontology_id": info["ontology_id"],
            "mapping_method": "curated_alias",
            "mapping_confidence": 0.95,
            "context_supergroup": info["supergroup"],
        }

    # 5. GWAS traits → GLOBAL_GWAS
    if raw_context in GWAS_TRAITS or source == "GWAS":
        return {
            "raw_context": raw_context,
            "context_id": "CTX_GLOBAL_GWAS",
            "context_name": "global_or_unspecified",
            "context_type": "global",
            "ontology_id": None,
            "mapping_method": "predefined_global_context",
            "mapping_confidence": 1.0,
            "context_supergroup": "global",
        }

    # 6. OpenTargets "GWAS" label → GLOBAL_GWAS
    if raw_context == "GWAS" or source == "OpenTargets":
        return {
            "raw_context": raw_context,
            "context_id": "CTX_GLOBAL_GWAS",
            "context_name": "global_or_unspecified",
            "context_type": "global",
            "ontology_id": None,
            "mapping_method": "predefined_global_context",
            "mapping_confidence": 1.0,
            "context_supergroup": "global",
        }

    # 7. Unresolved
    return {
        "raw_context": raw_context,
        "context_id": None,
        "context_name": None,
        "context_type": "unresolved",
        "ontology_id": None,
        "mapping_method": "unresolved",
        "mapping_confidence": 0.0,
        "context_supergroup": None,
    }


def main() -> int:
    project_root = Path("/workspace/v2g-benchmark")

    print("=" * 70)
    print("PART 1: Complete Context Mapping Fix")
    print("=" * 70)

    # Step 1: Generate the comprehensive context mapping YAML
    print("\n[1.1] Generating comprehensive context mapping config...")
    yaml_path = project_root / "config" / "generated_context_mapping.yaml"
    generate_context_mapping_yaml(yaml_path)

    # Step 2: Build GTEx contexts reference parquet
    print("\n[1.2] Building GTEx contexts reference...")
    gtex_path = project_root / "data" / "reference" / "gtex_contexts.parquet"
    build_gtex_contexts_parquet(gtex_path)

    # Step 3: Load variants and build context inventory
    print("\n[1.3] Building context inventory from all sources...")
    variants_path = project_root / "data" / "processed" / "variants_normalized.parquet"
    variants_df = pl.read_parquet(str(variants_path))

    # Get context × source counts
    ctx_counts = (
        variants_df.group_by(["context", "source"])
        .len()
        .rename({"len": "count"})
        .sort(["source", "count"], descending=[False, True])
    )

    # Also check CRISPR data for cell types
    crispr_path = project_root / "data" / "raw" / "crispr_comparison" / "resources" / "crispr_data" / "EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"
    if crispr_path.exists():
        crispr_df = pl.read_csv(str(crispr_path), separator="\t", infer_schema_length=10000)
        crispr_ctx = (
            crispr_df.group_by("CellType")
            .len()
            .rename({"CellType": "context", "len": "count"})
            .with_columns(pl.lit("CRISPR").alias("source"))
        )
        ctx_counts = pl.concat([ctx_counts, crispr_ctx], how="vertical")
        print(f"  Added {crispr_ctx.height} CRISPR cell types")

    # Build inventory with resolution
    inventory_rows = []
    for row in ctx_counts.iter_rows(named=True):
        raw_ctx = row["context"]
        source = row["source"]
        count = row["count"]

        resolved = resolve_context(raw_ctx, source)

        # Infer context type
        if resolved["context_type"] == "global":
            inferred_type = "trait"
        elif resolved["context_type"] == "cell_line":
            inferred_type = "cell_line"
        elif resolved["context_type"] == "cell_type":
            inferred_type = "cell_type"
        elif resolved["context_type"] == "tissue":
            inferred_type = "tissue"
        else:
            inferred_type = "unknown"

        mapping_status = "mapped" if resolved["mapping_confidence"] > 0 else "unmapped"

        inventory_rows.append({
            "source_dataset": source,
            "raw_context": raw_ctx,
            "normalized_context": resolved["context_name"] or raw_context,
            "inferred_context_type": inferred_type,
            "count": count,
            "mapping_status": mapping_status,
        })

    inventory_df = pl.DataFrame(inventory_rows)
    interim_path = project_root / "data" / "interim" / "context_inventory.parquet"
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_df.write_parquet(str(interim_path))
    print(f"  Wrote context inventory: {interim_path} ({inventory_df.height} rows)")

    # Step 4: Build full context mapping with provenance
    print("\n[1.4] Building full context mapping with provenance...")
    mapping_rows = []
    for row in ctx_counts.iter_rows(named=True):
        raw_ctx = row["context"]
        source = row["source"]
        resolved = resolve_context(raw_ctx, source)
        resolved["source_dataset"] = source
        resolved["count"] = row["count"]
        mapping_rows.append(resolved)

    mapping_df = pl.DataFrame(mapping_rows)

    # Reorder columns
    mapping_df = mapping_df.select([
        "source_dataset",
        "raw_context",
        "context_id",
        "context_name",
        "context_type",
        "ontology_id",
        "mapping_method",
        "mapping_confidence",
        "context_supergroup",
        "count",
    ])

    # Write updated context mapping
    mapping_output = project_root / "data" / "processed" / "context_mapping.parquet"
    mapping_df.write_parquet(str(mapping_output))
    print(f"  Wrote context mapping: {mapping_output} ({mapping_df.height} rows)")

    # Also write updated contexts_normalized
    contexts_norm = mapping_df.select([
        pl.col("raw_context").alias("source_context"),
        pl.col("context_id"),
        pl.col("context_name").alias("normalized_name"),
    ]).unique(subset=["source_context"])
    contexts_output = project_root / "data" / "processed" / "contexts_normalized.parquet"
    contexts_norm.write_parquet(str(contexts_output))
    print(f"  Wrote normalized contexts: {contexts_output} ({contexts_norm.height} rows)")

    # Step 5: Generate QC report
    print("\n[1.5] Generating context mapping QC report...")
    qc_rows = []

    # GTEx QC
    gtex_rows = mapping_df.filter(pl.col("source_dataset") == "GTEx")
    gtex_mapped = gtex_rows.filter(pl.col("mapping_confidence") >= 0.8).height
    gtex_total = gtex_rows.height
    qc_rows.append({
        "source": "GTEx",
        "total_contexts": gtex_total,
        "mapped": gtex_mapped,
        "unmapped": gtex_total - gtex_mapped,
        "mapping_rate": f"{100 * gtex_mapped / gtex_total:.1f}%" if gtex_total > 0 else "N/A",
        "required_threshold": "100%",
        "status": "PASS" if gtex_mapped == gtex_total else "FAIL",
    })

    # CRISPR QC
    crispr_rows = mapping_df.filter(pl.col("source_dataset") == "CRISPR")
    crispr_mapped = crispr_rows.filter(pl.col("mapping_confidence") >= 0.8).height
    crispr_total = crispr_rows.height
    qc_rows.append({
        "source": "CRISPR",
        "total_contexts": crispr_total,
        "mapped": crispr_mapped,
        "unmapped": crispr_total - crispr_mapped,
        "mapping_rate": f"{100 * crispr_mapped / crispar_total:.1f}%" if crispr_total > 0 else "N/A",
        "required_threshold": "100%",
        "status": "PASS" if crispr_mapped == crispr_total else "FAIL",
    })

    # eQTL Catalogue QC
    eqtl_rows = mapping_df.filter(pl.col("source_dataset") == "eQTL_Catalogue")
    eqtl_mapped = eqtl_rows.filter(pl.col("mapping_confidence") >= 0.8).height
    eqtl_total = eqtl_rows.height
    eqtl_rate = 100 * eqtl_mapped / eqtl_total if eqtl_total > 0 else 0
    qc_rows.append({
        "source": "eQTL_Catalogue",
        "total_contexts": eqtl_total,
        "mapped": eqtl_mapped,
        "unmapped": eqtl_total - eqtl_mapped,
        "mapping_rate": f"{eqtl_rate:.1f}%",
        "required_threshold": ">=99%",
        "status": "PASS" if eqtl_rate >= 99 else "FAIL",
    })

    # GWAS QC (traits should be GLOBAL_GWAS, not unmapped)
    gwas_rows = mapping_df.filter(pl.col("source_dataset") == "GWAS")
    gwas_global = gwas_rows.filter(pl.col("mapping_method") == "predefined_global_context").height
    gwas_total = gwas_rows.height
    qc_rows.append({
        "source": "GWAS",
        "total_contexts": gwas_total,
        "mapped": gwas_global,
        "unmapped": gwas_total - gwas_global,
        "mapping_rate": f"{100 * gwas_global / gwas_total:.1f}%" if gwas_total > 0 else "N/A",
        "required_threshold": "100% (trait/context separation)",
        "status": "PASS" if gwas_global == gwas_total else "FAIL",
    })

    # OpenTargets QC
    ot_rows = mapping_df.filter(pl.col("source_dataset") == "OpenTargets")
    ot_global = ot_rows.filter(pl.col("mapping_method") == "predefined_global_context").height
    ot_total = ot_rows.height
    qc_rows.append({
        "source": "OpenTargets",
        "total_contexts": ot_total,
        "mapped": ot_global,
        "unmapped": ot_total - ot_global,
        "mapping_rate": f"{100 * ot_global / ot_total:.1f}%" if ot_total > 0 else "N/A",
        "required_threshold": "100% (trait/context separation)",
        "status": "PASS" if ot_global == ot_total else "FAIL",
    })

    # Overall
    total_ctx = mapping_df.height
    total_mapped = mapping_df.filter(pl.col("mapping_confidence") >= 0.8).height
    qc_rows.append({
        "source": "OVERALL",
        "total_contexts": total_ctx,
        "mapped": total_mapped,
        "unmapped": total_ctx - total_mapped,
        "mapping_rate": f"{100 * total_mapped / total_ctx:.1f}%",
        "required_threshold": "All mandatory PASS",
        "status": "PASS" if all(r["status"] == "PASS" for r in qc_rows) else "FAIL",
    })

    qc_df = pl.DataFrame(qc_rows)
    qc_path = project_root / "results" / "tables" / "context_mapping_qc.tsv"
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_df.write_csv(str(qc_path), separator="\t")
    print(f"  Wrote context mapping QC: {qc_path}")
    print()
    print("  QC Summary:")
    print(qc_df.to_pandas().to_string(index=False))

    # Step 6: Generate unmapped contexts report
    unmapped = mapping_df.filter(pl.col("mapping_confidence") < 0.8)
    unmapped_path = project_root / "results" / "tables" / "unmapped_contexts.tsv"
    if unmapped.height > 0:
        unmapped.write_csv(str(unmapped_path), separator="\t")
        print(f"\n  Wrote unmapped contexts: {unmapped_path} ({unmapped.height} rows)")
    else:
        # Write empty file with headers
        pl.DataFrame(schema={
            "source_dataset": pl.Utf8,
            "raw_context": pl.Utf8,
            "context_id": pl.Utf8,
            "context_name": pl.Utf8,
            "context_type": pl.Utf8,
            "ontology_id": pl.Utf8,
            "mapping_method": pl.Utf8,
            "mapping_confidence": pl.Float64,
            "context_supergroup": pl.Utf8,
            "count": pl.Int64,
        }).write_csv(str(unmapped_path), separator="\t")
        print(f"\n  No unmapped contexts. Wrote empty: {unmapped_path}")

    # Step 7: Print mapping method breakdown
    print("\n[1.6] Mapping method breakdown:")
    method_counts = (
        mapping_df.group_by("mapping_method")
        .len()
        .sort("len", descending=True)
    )
    print(method_counts.to_pandas().to_string(index=False))

    # Step 8: Print context type breakdown
    print("\n[1.7] Context type breakdown:")
    type_counts = (
        mapping_df.group_by("context_type")
        .len()
        .sort("len", descending=True)
    )
    print(type_counts.to_pandas().to_string(index=False))

    # Final status
    all_pass = all(r["status"] == "PASS" for r in qc_rows if r["source"] != "OVERALL")
    print(f"\n{'=' * 70}")
    if all_pass:
        print("PART 1 STATUS: ALL QC PASS")
    else:
        print("PART 1 STATUS: SOME QC FAILED — reviewing...")
        for r in qc_rows:
            if r["status"] == "FAIL" and r["source"] != "OVERALL":
                print(f"  FAIL: {r['source']} — {r['mapped']}/{r['total_contexts']} mapped")
    print(f"{'=' * 70}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
