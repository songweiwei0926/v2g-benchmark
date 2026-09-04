# Harmonize rules — normalize variants, genes, contexts

rule normalize_variants:
    input:
        gtex = "data/raw/gtex_v11_susie.tar",
        eqtl = "data/raw/eqtl_catalogue_stable/",
        crispr = "data/raw/crispr_comparison/",
        gwas = "data/raw/gwas_e2g_benchmarking/",
        opentargets = "data/raw/opentargets_gold_standards/",
        pgboost = "data/raw/pgboost_zenodo/",
        fasta = "data/reference/GRCh38.fa"
    output:
        "data/processed/variants_normalized.parquet",
        "data/processed/variant_qc_report.tsv"
    shell:
        """
        python scripts/harmonize/normalize_variants.py \
            --gtex {input.gtex} \
            --eqtl {input.eqtl} \
            --crispr {input.crispr} \
            --gwas {input.gwas} \
            --opentargets {input.opentargets} \
            --pgboost {input.pgboost} \
            --fasta {input.fasta} \
            --output {output[0]} \
            --qc-output {output[1]}
        """

rule normalize_contexts:
    input:
        config = "config/context_mapping.yaml",
        variants = "data/processed/variants_normalized.parquet"
    output:
        "data/processed/contexts_normalized.parquet",
        "data/processed/context_mapping.parquet"
    shell:
        """
        python scripts/harmonize/normalize_contexts.py \
            --config {input.config} \
            --variants {input.variants} \
            --output {output[0]} \
            --mapping-output {output[1]}
        """
