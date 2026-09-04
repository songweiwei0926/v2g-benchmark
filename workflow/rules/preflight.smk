# Preflight rules — environment checks, resource discovery, locking

rule preflight:
    output:
        "data/locked/preflight_pass.txt"
    input:
        "config/site.yaml",
        "config/project.yaml",
        "config/models.yaml",
        "config/datasets.yaml"
    shell:
        """
        python scripts/preflight/preflight.py --config {input[0]}
        touch {output}
        """

rule discover_resources:
    input:
        "data/locked/preflight_pass.txt",
        "config/datasets.yaml"
    output:
        "data/locked/resource_manifest.lock.tsv"
    shell:
        """
        python scripts/preflight/discover_resources.py \
            --config config/site.yaml \
            --output {output}
        """

rule lock_resources:
    input:
        rules.discover_resources.output
    output:
        "data/locked/provenance.lock.yaml"
    run:
        import yaml
        from pathlib import Path
        import sys
        sys.path.insert(0, "src")
        from v2gbench.utils.provenance import generate_provenance_lock, get_git_info, compute_file_sha256
        from v2gbench.utils.config import compute_config_hash
        
        with open("config/datasets.yaml") as f:
            datasets = yaml.safe_load(f)
        with open("config/models.yaml") as f:
            models = yaml.safe_load(f)
        
        config_hash = compute_config_hash("config")
        git_info = get_git_info()
        
        generate_provenance_lock(
            str(output[0]),
            datasets.get("datasets", {}),
            models.get("models", {}),
            config_hash,
            git_info
        )
