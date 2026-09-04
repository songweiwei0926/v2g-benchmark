# Ensemble rules — integrated models (Track E)

rule train_integrated_models:
    input:
        predictions = "data/processed/all_model_predictions.parquet",
        candidates = "data/processed/candidate_1m.parquet",
        evidence = "data/processed/evidence_long.parquet"
    output:
        "predictions/integrated/integrated_rank_predictions.parquet",
        "predictions/integrated/integrated_logistic_predictions.parquet",
        "predictions/integrated/integrated_xgboost_predictions.parquet",
        "results/tables/integrated_feature_importance.tsv"
    shell:
        """
        python scripts/models/train_integrated.py \
            --predictions {input.predictions} \
            --candidates {input.candidates} \
            --evidence {input.evidence} \
            --output-dir predictions/integrated \
            --feature-importance-output {output[3]}
        """

rule evaluate_integrated_models:
    input:
        integrated_rank = "predictions/integrated/integrated_rank_predictions.parquet",
        integrated_logistic = "predictions/integrated/integrated_logistic_predictions.parquet",
        integrated_xgboost = "predictions/integrated/integrated_xgboost_predictions.parquet",
        candidates = "data/processed/candidate_1m.parquet"
    output:
        "results/metrics/integrated_metrics.tsv"
    shell:
        """
        python scripts/evaluate/evaluate_ranking.py \
            --predictions {input.integrated_rank} \
            --candidates {input.candidates} \
            --output results/metrics/integrated_rank_metrics.tsv

        python scripts/evaluate/evaluate_ranking.py \
            --predictions {input.integrated_logistic} \
            --candidates {input.candidates} \
            --output results/metrics/integrated_logistic_metrics.tsv

        python scripts/evaluate/evaluate_ranking.py \
            --predictions {input.integrated_xgboost} \
            --candidates {input.candidates} \
            --output results/metrics/integrated_xgboost_metrics.tsv

        # Merge integrated metrics
        python -c "
        import polars as pl
        dfs = []
        for name, path in [('integrated_rank', 'results/metrics/integrated_rank_metrics.tsv'),
                           ('integrated_logistic', 'results/metrics/integrated_logistic_metrics.tsv'),
                           ('integrated_xgboost', 'results/metrics/integrated_xgboost_metrics.tsv')]:
            df = pl.read_csv(path, separator='\\t')
            df = df.with_columns(pl.lit(name).alias('model_id'))
            dfs.append(df)
        pl.concat(dfs).write_csv('{output}', separator='\\t')
        "
        """
