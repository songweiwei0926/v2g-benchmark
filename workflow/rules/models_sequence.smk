# Sequence model rules — Borzoi, Enformer, AlphaGenome

rule score_borzoi:
    input:
        seq_core = "data/processed/seq_core.parquet",
        gene_master = "data/reference/gene_master.parquet",
        fasta = "data/reference/GRCh38.fa"
    output:
        "predictions/sequence/borzoi_predictions.parquet"
    shell:
        """
        python scripts/models/score_borzoi.py \
            --seq-core {input.seq_core} \
            --gene-master {input.gene_master} \
            --fasta {input.fasta} \
            --output {output}
        """

rule score_enformer:
    input:
        seq_core = "data/processed/seq_core.parquet",
        gene_master = "data/reference/gene_master.parquet",
        fasta = "data/reference/GRCh38.fa"
    output:
        "predictions/sequence/enformer_predictions.parquet"
    shell:
        """
        python scripts/models/score_enformer.py \
            --seq-core {input.seq_core} \
            --gene-master {input.gene_master} \
            --fasta {input.fasta} \
            --output {output}
        """

rule score_alphagenome:
    input:
        seq_core = "data/processed/seq_core.parquet",
        gene_master = "data/reference/gene_master.parquet"
    output:
        "predictions/sequence/alphagenome_predictions.parquet"
    shell:
        """
        python scripts/models/score_alphagenome.py \
            --seq-core {input.seq_core} \
            --gene-master {input.gene_master} \
            --output {output}
        """

rule merge_all_predictions:
    input:
        baselines = "predictions/baseline/baseline_predictions.parquet",
        published = "predictions/published/published_predictions.parquet",
        encode_supp = "predictions/published/encode_supplementary_predictions.parquet",
        borzoi = "predictions/sequence/borzoi_predictions.parquet",
        enformer = "predictions/sequence/enformer_predictions.parquet",
        alphagenome = "predictions/sequence/alphagenome_predictions.parquet"
    output:
        "data/processed/all_model_predictions.parquet"
    shell:
        """
        python scripts/models/merge_predictions.py \
            --baselines {input.baselines} \
            --published {input.published} \
            --encode-supplementary {input.encode_supp} \
            --borzoi {input.borzoi} \
            --enformer {input.enformer} \
            --alphagenome {input.alphagenome} \
            --output {output}
        """
