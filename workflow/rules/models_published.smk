# Published model rules — baselines and published predictions

rule score_baselines:
    input:
        candidates = "data/processed/candidate_1m.parquet",
        gene_master = "data/reference/gene_master.parquet"
    output:
        "predictions/baseline/baseline_predictions.parquet"
    shell:
        """
        python scripts/models/score_baselines.py \
            --candidates {input.candidates} \
            --gene-master {input.gene_master} \
            --output {output}
        """

rule import_published_models:
    input:
        candidates = "data/processed/candidate_1m.parquet",
        gene_master = "data/reference/gene_master.parquet",
        synapse = "data/raw/encode_predictions_bundle/",
        zenodo = "data/raw/pgboost_zenodo/",
        variants = "data/processed/variants_normalized.parquet"
    output:
        "predictions/published/published_predictions.parquet"
    shell:
        """
        python scripts/models/import_published.py \
            --candidates {input.candidates} \
            --gene-master {input.gene_master} \
            --synapse-dir {input.synapse} \
            --zenodo-dir {input.zenodo} \
            --variants {input.variants} \
            --output {output}
        """

rule discover_encode_models:
    input:
        synapse = "data/raw/encode_predictions_bundle/",
        candidates = "data/processed/candidate_1m.parquet"
    output:
        "predictions/published/encode_supplementary_predictions.parquet",
        "results/tables/all_encode_configurations.tsv"
    shell:
        """
        python scripts/models/discover_encode_models.py \
            --synapse-dir {input.synapse} \
            --candidates {input.candidates} \
            --output {output[0]} \
            --configurations-output {output[1]}
        """
