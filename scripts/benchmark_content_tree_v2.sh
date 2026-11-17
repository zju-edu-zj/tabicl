#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR=${OUTPUT_DIR:-/ossfs/workspace/tabicl-content-tree-v2/benchmarks}
PYTHON=${PYTHON:-python}
REPEAT=${REPEAT:-7}
WARMUP=${WARMUP:-3}
mkdir -p "$OUTPUT_DIR"

for rows in 2048 10240 32768; do
    for attention in dense global_sparse dual_path adaptive_frontier; do
        partition=content_tree
        "$PYTHON" "$REPO_ROOT/scripts/benchmark_icl_attention.py" \
            --attention "$attention" \
            --rows "$rows" \
            --scope encoder \
            --phase train \
            --partition "$partition" \
            --block-size 128 \
            --selected-blocks 4 \
            --summary-slots 1 \
            --sparse-every 1 \
            --dtype bfloat16 \
            --repeat "$REPEAT" \
            --warmup "$WARMUP" \
            --output "$OUTPUT_DIR/${attention}_${rows}_train.json"
    done
done
