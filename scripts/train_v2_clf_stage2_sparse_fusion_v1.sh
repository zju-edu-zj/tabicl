#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export NUM_GPUS=${NUM_GPUS:-16}
export TRAIN_DTYPE=${TRAIN_DTYPE:-float32}
export ICL_ATTENTION=global_sparse
export ICL_SPARSE_EVERY=1
export ICL_BLOCK_SIZE=128
export ICL_NUM_SELECTED_BLOCKS=4
export ICL_ROUTER=centroid
export ICL_PARTITION=content_tree
export ICL_GLOBAL_TYPE=elu
export ICL_QK_NORM=True
export ICL_FUSION=independent_v1
export MUON_ROUTE_MODE=${MUON_ROUTE_MODE:-all_matrix}

export STAGE1_CKPT=${STAGE1_CKPT:-/input/long/tabicl-training-perf/checkpoints/tabiclv2-clf/stage1-fp32-linear-sparse-all12-ct/step-500000.ckpt}
export CKPT_DIR=${CKPT_DIR:-/ossfs/workspace/tabicl-stage2-fusion-v1-20261117/checkpoints}
export LOG_DIR=${LOG_DIR:-/ossfs/workspace/tabicl-stage2-fusion-v1-20261117/logs}
export WANDB_NAME=${WANDB_NAME:-tabicl_sparse_fusion_v1_stage2}

exec bash "$REPO_ROOT/scripts/train_v2_clf_stage2.sh"
