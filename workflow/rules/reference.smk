# Reference rules — build gene master table, prepare reference data

rule build_gene_master:
    input:
        gtf = "data/reference/gencode.v47.genes.gtf"
    output:
        "data/reference/gene_master.parquet"
    shell:
        """
        python scripts/harmonize/normalize_genes.py \
            --gtf {input.gtf} \
            --output {output}
        """

rule index_reference_fasta:
    input:
        "data/reference/GRCh38.fa"
    output:
        "data/reference/GRCh38.fa.fai"
    shell:
        """
        samtools faidx {input}
        """
