# Benchmark rules — build gold registry, candidate sets, leakage, applicability

rule build_gold_registry:
    input:
        variants = "data/processed/variants_normalized.parquet",
        contexts = "data/processed/contexts_normalized.parquet",
        gene_master = "data/reference/gene_master.parquet",
        gtex = "data/raw/gtex_v11_susie.tar",
        eqtl = "data/raw/eqtl_catalogue_stable/",
        crispr = "data/raw/crispr_comparison/",
        gwas = "data/raw/gwas_e2g_benchmarking/",
        opentargets = "data/raw/opentargets_gold_standards/",
        pgboost = "data/raw/pgboost_zenodo/"
    output:
        "data/processed/evidence_long.parquet",
        "data/processed/canonical_pairs.parquet"
    shell:
        """
        python scripts/benchmark/build_gold_registry.py \
            --variants {input.variants} \
            --contexts {input.contexts} \
            --gene-master {input.gene_master} \
            --gtex {input.gtex} \
            --eqtl {input.eqtl} \
            --crispr {input.crispr} \
            --gwas {input.gwas} \
            --opentargets {input.opentargets} \
            --pgboost {input.pgboost} \
            --output {output[0]} \
            --canonical-output {output[1]}
        """

rule build_candidate_sets:
    input:
        variants = "data/processed/variants_normalized.parquet",
        gene_master = "data/reference/gene_master.parquet",
        gold = "data/processed/canonical_pairs.parquet",
        contexts = "data/processed/contexts_normalized.parquet"
    output:
        "data/processed/candidate_250k.parquet",
        "data/processed/candidate_500k.parquet",
        "data/processed/candidate_1m.parquet",
        "data/processed/candidate_v2g.parquet",
        "data/processed/gold_coverage_report.tsv"
    shell:
        """
        python scripts/benchmark/build_candidate_sets.py \
            --variants {input.variants} \
            --gene-master {input.gene_master} \
            --gold {input.gold} \
            --contexts {input.contexts} \
            --output-dir data/processed \
            --main-output {output[3]} \
            --coverage-report {output[4]}
        """

rule build_leakage_registry:
    input:
        models = "config/models.yaml"
    output:
        "data/processed/leakage_registry.parquet",
        "config/model_training_registry.tsv"
    shell:
        """
        python scripts/benchmark/build_leakage_registry.py \
            --models {input.models} \
            --output {output[0]} \
            --registry-output {output[1]}
        """

rule build_applicability_matrix:
    input:
        models = "config/models.yaml",
        contexts = "data/processed/contexts_normalized.parquet"
    output:
        "results/tables/model_context_matrix.tsv"
    shell:
        """
        python scripts/benchmark/build_applicability_matrix.py \
            --models {input.models} \
            --contexts {input.contexts} \
            --output {output}
        """

rule build_seq_core:
    input:
        evidence = "data/processed/evidence_long.parquet",
        candidates = "data/processed/candidate_1m.parquet"
    output:
        "data/processed/seq_core.parquet"
    shell:
        """
        python scripts/benchmark/build_seq_core.py \
            --evidence {input.evidence} \
            --candidates {input.candidates} \
            --output {output}
        """
