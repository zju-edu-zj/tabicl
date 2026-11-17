#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARCH=${ARCH:-dual_path}
if [[ "$ARCH" != dual_path && "$ARCH" != adaptive_frontier ]]; then
    echo "ARCH must be dual_path or adaptive_frontier" >&2
    exit 2
fi

export NUM_GPUS=${NUM_GPUS:-8}
export MAX_STEPS=${MAX_STEPS:-2000}
export TRAIN_DTYPE=${TRAIN_DTYPE:-float32}
export PRIOR_BACKEND=dataloader
export PRIOR_SCHEDULE=none
export PRIOR_IN_ORDER=True
export PRIOR_SEED_BY_STEP=True
export PRIOR_WORKERS=${PRIOR_WORKERS:-4}
export PRIOR_PREFETCH_FACTOR=${PRIOR_PREFETCH_FACTOR:-2}
export SAVE_TEMP_EVERY=${SAVE_TEMP_EVERY:-100}
export SAVE_PERM_EVERY=${SAVE_PERM_EVERY:-1000}
export WANDB_NAME=${WANDB_NAME:-tabicl_${ARCH}_pilot_${MAX_STEPS}}
export CKPT_DIR=${CKPT_DIR:-/ossfs/workspace/tabicl-content-tree-v2/pilots/${ARCH}_${MAX_STEPS}/checkpoints}
export LOG_DIR=${LOG_DIR:-/ossfs/workspace/tabicl-content-tree-v2/pilots/${ARCH}_${MAX_STEPS}/logs}

exec bash "$REPO_ROOT/scripts/train_v2_clf_stage1_${ARCH}.sh"
