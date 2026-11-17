#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export ICL_ATTENTION=token_sparse
export ICL_TOKEN_TOPK=${ICL_TOKEN_TOPK:-128}
export ICL_INDEX_DIM=${ICL_INDEX_DIM:-16}
export ICL_INDEX_QUERY_SAMPLES=${ICL_INDEX_QUERY_SAMPLES:-32}
export ICL_INDEX_LOSS_WEIGHT=${ICL_INDEX_LOSS_WEIGHT:-0.05}
export ICL_INDEX_LOSS_WARMUP_STEPS=${ICL_INDEX_LOSS_WARMUP_STEPS:-5000}
export ICL_INDEX_LR=${ICL_INDEX_LR:-3e-4}
export TRAIN_DTYPE=${TRAIN_DTYPE:-float32}
export WANDB_NAME=${WANDB_NAME:-tabiclv2_clf_stage1_pure_token_sparse}
export CKPT_DIR=${CKPT_DIR:-/ossfs/workspace/tabicl/checkpoints/tabiclv2-clf/stage1-token-sparse}

exec bash "$REPO_ROOT/scripts/train_v2_clf_stage1_fast.sh" "$@"
