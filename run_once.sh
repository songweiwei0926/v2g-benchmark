#!/usr/bin/env bash
set -euo pipefail

# V2G-Benchmark-OneShot v1.0 — Single entry point
# This script runs the entire pipeline from preflight to release.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load credentials from .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "FATAL: .env file not found. Copy .env.example to .env and fill in credentials."
    exit 1
fi

# Record run start
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
RUN_START=$(date -Iseconds)
HOSTNAME=$(hostname)

echo "============================================"
echo "V2G-Benchmark-OneShot v1.0"
echo "Run ID: ${RUN_ID}"
echo "Start: ${RUN_START}"
echo "Host: ${HOSTNAME}"
echo "============================================"

# Compute config hash (SHA256 of all config files)
CONFIG_HASH=$(find config/ -name '*.yaml' -o -name '*.yml' | sort | xargs cat | sha256sum | cut -d' ' -f1)
echo "Config hash: ${CONFIG_HASH}"

# Record git tree hash
GIT_TREE_HASH=""
GIT_DIRTY=""
if [ -d .git ]; then
    GIT_TREE_HASH=$(git rev-parse HEAD^{tree} 2>/dev/null || echo "no_commits")
    GIT_DIRTY=$(git status --porcelain 2>/dev/null | head -1 || echo "")
    if [ -n "$GIT_DIRTY" ]; then
        echo "WARNING: Git repo is dirty. Uncommitted changes detected."
    fi
fi

# Write run lock
mkdir -p data/locked
cat > data/locked/run_lock.json << EOF
{
    "run_id": "${RUN_ID}",
    "start": "${RUN_START}",
    "hostname": "${HOSTNAME}",
    "config_hash": "${CONFIG_HASH}",
    "git_tree_hash": "${GIT_TREE_HASH}",
    "git_dirty": "${GIT_DIRTY}"
}
EOF

# Phase 1: Preflight
echo ""
echo "=== Phase 1: Preflight ==="
python scripts/preflight/preflight.py --config config/site.yaml
PREFLIGHT_EXIT=$?
if [ $PREFLIGHT_EXIT -ne 0 ]; then
    echo "FATAL: Preflight failed. Formal computation will not start."
    exit 1
fi

# Phase 2: Resource discovery and lock
echo ""
echo "=== Phase 2: Resource Discovery ==="
python scripts/preflight/discover_resources.py --config config/site.yaml --output data/locked/resource_manifest.lock.tsv

# Phase 3: Snakemake pipeline
echo ""
echo "=== Phase 3: Snakemake Pipeline ==="
snakemake \
    --cores "${V2G_CORES:-8}" \
    --rerun-incomplete \
    --keep-going \
    --printshellcmds \
    --show-failed-logs \
    all

# Phase 4: Final QC
echo ""
echo "=== Phase 4: Final QC ==="
python scripts/preflight/final_qc.py \
    --config config/site.yaml \
    --run-lock data/locked/run_lock.json \
    --output results/release/

# Check for SUCCESS
if [ -f results/release/SUCCESS ]; then
    RUN_END=$(date -Iseconds)
    echo ""
    echo "============================================"
    echo "V2G-Benchmark-OneShot v1.0 COMPLETED"
    echo "End: ${RUN_END}"
    echo "SUCCESS: results/release/SUCCESS"
    echo "============================================"
    exit 0
else
    echo ""
    echo "FATAL: Final QC did not produce SUCCESS file."
    echo "PROJECT STATUS = INCOMPLETE"
    exit 1
fi
