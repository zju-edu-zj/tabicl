#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "usage: $0 {plan|prepare|run|summarize} OUTPUT_DIR [extra options...]" >&2
    exit 2
fi

ACTION=$1
OUTPUT=$2
shift 2
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec "${PYTHON:-python}" "$REPO_ROOT/scripts/screen_icl_attention.py" \
    "$ACTION" \
    --output "$OUTPUT" \
    --preset stage1_2048 \
    --steps 20000 \
    --num-gpus 16 \
    --batch-size 64 \
    --micro-batch-size 4 \
    --dtype float32 \
    --optimizer hybrid \
    --variants dual_path_static dual_path_linear_v1 dual_path_evidence_v3 \
    --eval-support 512 1024 2048 \
    "$@"
