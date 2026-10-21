#!/usr/bin/env bash
set -euo pipefail

# Run on the same training machine/configuration as train_v2_clf_stage1.sh.
# First launch: initialize weights from 35k and reset optimizer/scheduler/step.
# Subsequent launches: fully resume the latest checkpoint in the new directory.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export INIT_CKPT="${INIT_CKPT:-/ossfs/workspace/tabicl/checkpoints/tabiclv2-clf/stage1/step-20000.ckpt}"
export CKPT_DIR="${CKPT_DIR:-/ossfs/workspace/tabicl/checkpoints/tabiclv2-clf/stage1_hybrid_from20k}"
export LOG_DIR="${LOG_DIR:-$CKPT_DIR/logs}"

export MUON=True
export LR="${LR:-4e-4}"
export ADAMW_LR="${ADAMW_LR:-1e-4}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export WARMUP_PROPORTION=-1
export WARMUP_STEPS="${WARMUP_STEPS:-500}"
export SAVE_TEMP_EVERY="${SAVE_TEMP_EVERY:-500}"
export SAVE_PERM_EVERY="${SAVE_PERM_EVERY:-5000}"
export WANDB_NAME="${WANDB_NAME:-tabiclv2_clf_stage1_hybrid_from20k}"

# The source checkpoint need not remain mounted once a new run can resume.
# Dry runs can also be inspected away from the training machine.
if [[ ${DRY_RUN:-0} != 1 ]] && ! compgen -G "$CKPT_DIR/step-*.ckpt" >/dev/null; then
    if [[ ! -f "$INIT_CKPT" ]]; then
        printf 'Initial checkpoint does not exist: %s\n' "$INIT_CKPT" >&2
        exit 1
    fi
fi

# GPU count, batch sizes, dtype, prior, ICL recipe and logging are inherited.
exec bash "$SCRIPT_DIR/train_v2_clf_stage1.sh"
