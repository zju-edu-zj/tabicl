#!/usr/bin/env bash
set -euo pipefail

# First architecture screen: all branches start from the same initialization.
# The generated default directory is new on every launch. To resume, explicitly
# pass the exact previous OUTPUT_DIR; each variant then loads its own latest
# checkpoint with model, optimizer, scheduler, and global step.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python}
NUM_GPUS=${NUM_GPUS:-16}
RUN_ID=$(date -u +%Y%m%d_%H%M%S)_$$
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/checkpoints/icl_screen_round1_20k_$RUN_ID}
STEPS=${STEPS:-20000}

args=(
  run
  --output "$OUTPUT_DIR"
  --preset routing
  --steps "$STEPS"
  --variants dense global_elu global_full_elu global_induced
             global_sparse_alternating global_sparse_elu
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
