# Figure and table rules

rule make_figures:
    input:
        ranking = "results/metrics/ranking_metrics.tsv",
        classification = "results/metrics/classification_metrics.tsv",
        stratified = "results/stratified/stratified_metrics.parquet",
        bootstrap = "results/bootstrap/bootstrap_results.parquet",
        predictions = "data/processed/all_model_predictions.parquet",
        evidence = "data/processed/evidence_long.parquet",
        candidates = "data/processed/candidate_1m.parquet",
        integrated = "results/metrics/integrated_metrics.tsv",
        failures = "results/failures/interesting_loci.parquet"
    output:
        "results/figures/fig1_overview.svg",
        "results/figures/fig2_heatmap.svg",
        "results/figures/fig3_distance.svg",
        "results/figures/fig4_context.svg",
        "results/figures/fig5_complementarity.svg",
        "results/figures/fig6_integrated.svg"
    shell:
        """
        python scripts/figures/make_figures.py \
            --ranking {input.ranking} \
            --classification {input.classification} \
            --stratified {input.stratified} \
            --bootstrap {input.bootstrap} \
            --predictions {input.predictions} \
            --evidence {input.evidence} \
            --candidates {input.candidates} \
            --integrated {input.integrated} \
            --failures {input.failures} \
            --output-dir results/figures
        """

rule make_supplementary_figures:
    input:
        stratified = "results/stratified/stratified_metrics.parquet",
        predictions = "data/processed/all_model_predictions.parquet",
        evidence = "data/processed/evidence_long.parquet",
        failures = "results/failures/error_cases.parquet"
    output:
        touch("results/figures/supplementary_done.txt")
    shell:
        """
        python scripts/figures/make_figures.py \
            --supplementary-only \
            --stratified {input.stratified} \
            --predictions {input.predictions} \
            --evidence {input.evidence} \
            --failures {input.failures} \
            --output-dir results/figures
        touch {output}
        """

rule make_tables:
    input:
        ranking = "results/metrics/ranking_metrics.tsv",
        classification = "results/metrics/classification_metrics.tsv",
        bootstrap = "results/bootstrap/paired_comparisons.tsv",
        stratified = "results/stratified/stratified_metrics.parquet",
        failures = "results/failures/error_cases.parquet",
        candidates = "data/processed/candidate_1m.parquet",
        evidence = "data/processed/evidence_long.parquet",
        predictions = "data/processed/all_model_predictions.parquet",
        integrated = "results/metrics/integrated_metrics.tsv",
        feature_importance = "results/tables/integrated_feature_importance.tsv",
        applicability = "results/tables/model_context_matrix.tsv",
        leakage = "data/processed/leakage_registry.parquet",
        encode_configs = "results/tables/all_encode_configurations.tsv"
    output:
        "results/tables/main_results.tsv",
        "results/tables/model_exclusions.tsv",
        "results/tables/mandatory_completion_matrix.tsv",
        "results/tables/model_score_qc.tsv",
        "results/tables/SupplementaryTables.xlsx"
    shell:
        """
        python scripts/figures/make_tables.py \
            --ranking {input.ranking} \
            --classification {input.classification} \
            --bootstrap {input.bootstrap} \
            --stratified {input.stratified} \
            --failures {input.failures} \
            --candidates {input.candidates} \
            --evidence {input.evidence} \
            --predictions {input.predictions} \
            --integrated {input.integrated} \
            --feature-importance {input.feature_importance} \
            --applicability {input.applicability} \
            --leakage {input.leakage} \
            --encode-configs {input.encode_configs} \
            --output-dir results/tables
        """

rule final_qc:
    input:
        rules.make_figures.output,
        rules.make_tables.output,
        "data/processed/all_model_predictions.parquet",
        "data/processed/candidate_v2g.parquet",
        "data/processed/evidence_long.parquet",
        "data/locked/run_lock.json"
    output:
        "results/release/SUCCESS",
        "results/release/run_manifest.json",
        "results/release/provenance.lock.yaml",
        "results/release/software_versions.tsv",
        "results/release/data_checksums.tsv",
        "results/release/model_registry.tsv",
        "results/release/dataset_registry.tsv",
        "results/release/benchmark_registry.parquet",
        "results/release/model_predictions.parquet",
        "results/release/metrics.tsv",
        "results/release/SupplementaryTables.xlsx"
    shell:
        """
        python scripts/preflight/final_qc.py \
            --config config/site.yaml \
            --run-lock data/locked/run_lock.json \
            --output results/release/
        """
