# Download rules — reference genome, gold standards, published predictions

rule download_references:
    input:
        "data/locked/provenance.lock.yaml"
    output:
        "data/reference/GRCh38.fa",
        "data/reference/GRCh38.fa.fai",
        "data/reference/gencode.v47.genes.gtf"
    shell:
        """
        python scripts/download/download_references.py \
            --output-dir data/reference \
            --config config/datasets.yaml
        """

rule download_gtex:
    input:
        "data/locked/provenance.lock.yaml"
    output:
        "data/raw/gtex_v11_susie.tar"
    shell:
        """
        python scripts/download/download_gtex.py \
            --output-dir data/raw \
            --config config/datasets.yaml
        """

rule download_eqtl_catalogue:
    input:
        "data/locked/provenance.lock.yaml"
    output:
        "data/raw/eqtl_catalogue_stable/",
        "data/raw/eqtl_catalogue_r8_beta/"
    shell:
        """
        python scripts/download/download_eqtl_catalogue.py \
            --output-dir data/raw \
            --config config/datasets.yaml
        """

rule download_crispr:
    input:
        "data/locked/provenance.lock.yaml"
    output:
        "data/raw/crispr_comparison/"
    shell:
        """
        python scripts/download/download_crispr.py \
            --output-dir data/raw \
            --config config/datasets.yaml
        """

rule download_gwas:
    input:
        "data/locked/provenance.lock.yaml"
    output:
        "data/raw/gwas_e2g_benchmarking/"
    shell:
        """
        python scripts/download/download_gwas.py \
            --output-dir data/raw \
            --config config/datasets.yaml
        """

rule download_opentargets:
    input:
        "data/locked/provenance.lock.yaml"
    output:
        "data/raw/opentargets_gold_standards/"
    shell:
        """
        python scripts/download/download_opentargets.py \
            --output-dir data/raw \
            --config config/datasets.yaml
        """

rule download_zenodo:
    input:
        "data/locked/provenance.lock.yaml"
    output:
        "data/raw/pgboost_zenodo/"
    shell:
        """
        python scripts/download/download_zenodo.py \
            --output-dir data/raw \
            --config config/datasets.yaml
        """

rule download_synapse:
    input:
        "data/locked/provenance.lock.yaml"
    output:
        "data/raw/encode_predictions_bundle/"
    shell:
        """
        python scripts/download/download_synapse.py \
            --output-dir data/raw \
            --config config/datasets.yaml
        """

rule download_traitgym:
    input:
        "data/locked/provenance.lock.yaml"
    output:
        "data/raw/traitgym/"
    shell:
        """
        python scripts/download/download_traitgym.py \
            --output-dir data/raw \
            --config config/datasets.yaml
        """

rule download_published_predictions:
    input:
        rules.download_synapse.output,
        rules.download_zenodo.output
    output:
        "data/raw/published_predictions_ready.txt"
    shell:
        """
        python scripts/download/download_published_predictions.py \
            --output-dir data/raw \
            --config config/datasets.yaml
        touch {output}
        """
