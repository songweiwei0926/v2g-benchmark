# Evaluation rules — ranking, classification, direction, bootstrap, stratification, failure analysis

rule evaluate_ranking:
    input:
        predictions = "data/processed/all_model_predictions.parquet",
        candidates = "data/processed/candidate_1m.parquet"
    output:
        "results/metrics/ranking_metrics.tsv"
    shell:
        """
        python scripts/evaluate/evaluate_ranking.py \
            --predictions {input.predictions} \
            --candidates {input.candidates} \
            --output {output}
        """

rule evaluate_classification:
    input:
        predictions = "data/processed/all_model_predictions.parquet",
        evidence = "data/processed/evidence_long.parquet"
    output:
        "results/metrics/classification_metrics.tsv"
    shell:
        """
        python scripts/evaluate/evaluate_classification.py \
            --predictions {input.predictions} \
            --evidence {input.evidence} \
            --output {output}
        """

rule evaluate_direction:
    input:
        predictions = "data/processed/all_model_predictions.parquet",
        evidence = "data/processed/evidence_long.parquet"
    output:
        "results/metrics/direction_metrics.tsv"
    shell:
        """
        python scripts/evaluate/evaluate_direction.py \
            --predictions {input.predictions} \
            --evidence {input.evidence} \
            --output {output}
        """

rule run_bootstrap:
    input:
        predictions = "data/processed/all_model_predictions.parquet",
        candidates = "data/processed/candidate_1m.parquet"
    output:
        "results/bootstrap/bootstrap_results.parquet",
        "results/bootstrap/paired_comparisons.tsv"
    shell:
        """
        python scripts/evaluate/run_bootstrap.py \
            --predictions {input.predictions} \
            --candidates {input.candidates} \
            --output {output[0]} \
            --paired-output {output[1]}
        """

rule run_stratified:
    input:
        predictions = "data/processed/all_model_predictions.parquet",
        candidates = "data/processed/candidate_1m.parquet",
        evidence = "data/processed/evidence_long.parquet"
    output:
        "results/stratified/stratified_metrics.parquet"
    shell:
        """
        python scripts/evaluate/run_stratified.py \
            --predictions {input.predictions} \
            --candidates {input.candidates} \
            --evidence {input.evidence} \
            --output {output}
        """

rule run_failure_analysis:
    input:
        predictions = "data/processed/all_model_predictions.parquet",
        candidates = "data/processed/candidate_1m.parquet",
        evidence = "data/processed/evidence_long.parquet"
    output:
        "results/failures/error_cases.parquet",
        "results/failures/interesting_loci.parquet"
    shell:
        """
        python scripts/evaluate/run_failure_analysis.py \
            --predictions {input.predictions} \
            --candidates {input.candidates} \
            --evidence {input.evidence} \
            --output {output[0]} \
            --interesting-output {output[1]}
        """
