#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export NUM_GPUS=${NUM_GPUS:-16}
export TRAIN_DTYPE=${TRAIN_DTYPE:-float32}
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-2048}
export COMPILE_FRONTEND=${COMPILE_FRONTEND:-all}
export COMPILE_WARMUP=${COMPILE_WARMUP:-True}
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-auto}
export ICL_ATTENTION=dual_path
export ICL_DUAL_CONTROLLER=${ICL_DUAL_CONTROLLER:-none}
export ICL_SPARSE_EVERY=1
export ICL_BLOCK_SIZE=32
export ICL_NUM_SELECTED_BLOCKS=32
export ICL_STORAGE_PAGE_ROWS=128
export ICL_EXHAUSTIVE_SUPPORT_LIMIT=${ICL_EXHAUSTIVE_SUPPORT_LIMIT:-8192}
export ICL_ROUTER=${ICL_ROUTER:-centroid}
export ICL_PARTITION=content_tree
export ICL_GLOBAL_TYPE=elu
export ICL_QK_NORM=True
# Gate/control matrices stay on AdamW; representation and FFN matrices use Muon.
export MUON_ROUTE_MODE=${MUON_ROUTE_MODE:-attention}
export PRIOR_SEED_BY_STEP=${PRIOR_SEED_BY_STEP:-False}

export CKPT_DIR=${CKPT_DIR:-/mnt/aifordata2026-p/common/aifordata/447952/dual-path-v3-stage1-2048x512/checkpoints}
export LOG_DIR=${LOG_DIR:-/mnt/aifordata2026-p/common/aifordata/447952/dual-path-v3-stage1-2048x512/logs}
export WANDB_NAME=${WANDB_NAME:-tabicl_dual_path_v3_stage1_2048x512}

exec bash "$REPO_ROOT/scripts/train_v2_clf_stage1.sh"
