#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export NUM_GPUS=${NUM_GPUS:-16}
export TRAIN_DTYPE=${TRAIN_DTYPE:-float32}
export ICL_ATTENTION=dual_path
export ICL_SPARSE_EVERY=1
export ICL_BLOCK_SIZE=${ICL_BLOCK_SIZE:-128}
export ICL_NUM_SELECTED_BLOCKS=${ICL_NUM_SELECTED_BLOCKS:-4}
export ICL_ROUTER=${ICL_ROUTER:-centroid}
export ICL_PARTITION=content_tree
export ICL_GLOBAL_TYPE=elu
export ICL_QK_NORM=True
export MUON_ROUTE_MODE=${MUON_ROUTE_MODE:-all_matrix}
export PRIOR_SEED_BY_STEP=${PRIOR_SEED_BY_STEP:-False}

export STAGE1_CKPT=${STAGE1_CKPT:-/ossfs/workspace/tabicl-content-tree-v2/dual-path/stage1/checkpoints/step-500000.ckpt}
export CKPT_DIR=${CKPT_DIR:-/ossfs/workspace/tabicl-content-tree-v2/dual-path/stage2/checkpoints}
export LOG_DIR=${LOG_DIR:-/ossfs/workspace/tabicl-content-tree-v2/dual-path/stage2/logs}
export WANDB_NAME=${WANDB_NAME:-tabicl_dual_path_stage2}

exec bash "$REPO_ROOT/scripts/train_v2_clf_stage2.sh"
