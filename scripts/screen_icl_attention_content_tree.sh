#!/usr/bin/env bash
set -euo pipefail

# Second screen: alternating global/sparse attention with content-tree
# partitioning and centroid top-k routing. It starts as an independent run from
# the common random initialization. The generated default directory is new on
# every launch. To resume, explicitly pass the exact previous OUTPUT_DIR.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python}
NUM_GPUS=${NUM_GPUS:-16}
RUN_ID=$(date -u +%Y%m%d_%H%M%S)_$$
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/checkpoints/icl_screen_content_tree_alternating_50k_$RUN_ID}
STEPS=${STEPS:-50000}

args=(
  run
  --output "$OUTPUT_DIR"
  --preset routing
  --steps "$STEPS"
  --variants global_sparse_alternating_content_tree
  --num-gpus "$NUM_GPUS"
  --no-recompute
  --eval-support 1024 2048 4096 8192 16384
  --eval-cases 16
  --eval-queries 128
  --diagnostics-every 200
  --eval-every 2000
)

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
printf 'Screen output: %s\n' "$OUTPUT_DIR"
exec "$PYTHON" "$SCRIPT_DIR/screen_icl_attention.py" "${args[@]}"
