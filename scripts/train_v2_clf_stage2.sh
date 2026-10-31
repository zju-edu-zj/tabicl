#!/bin/bash
set -euo pipefail
# TabICLv2 pre-training -- CLASSIFIER, Stage 2 (of 3). See train_v2_clf_stage1.sh for
# the full recipe.
#
# Stage 2: 40K steps on datasets with 400-10,240 samples (log-uniform), 79-81% of
# samples for training, max LR 1e-4, gradient clipping 10. Continues from the Stage 1
# checkpoint (weights only). Compiled frontends default to native SDPA;
# compatible external FlashAttention-3 builds can opt in via USE_FLASH_ATTN3=True.
#
# Adjust the placeholder paths and --nproc_per_node / --n_jobs for your hardware.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

NUM_GPUS=${NUM_GPUS:-8}                                   # override for 16-GPU runs

# Fast default: overlap graph generation with training and balance accumulated
# micro-batches across ranks by predicted model cost. Every value remains overridable
# from the environment for ablations and smaller machines.
TRAIN_DTYPE=${TRAIN_DTYPE:-bfloat16}
COMPILE_FRONTEND=${COMPILE_FRONTEND:-all}
COMPILE_WARMUP=${COMPILE_WARMUP:-True}
ICL_PARTITION=${ICL_PARTITION:-content_tree}
PRIOR_BACKEND=${PRIOR_BACKEND:-node_pool}
if [[ -z "${PRIOR_SCHEDULE:-}" ]]; then
    if [[ "$PRIOR_BACKEND" == node_pool ]]; then
        PRIOR_SCHEDULE=balanced
    else
        PRIOR_SCHEDULE=none
    fi
fi
PRIOR_POOL_TRANSPORT=${PRIOR_POOL_TRANSPORT:-numpy}
PRIOR_POOL_KERNEL_THREADS=${PRIOR_POOL_KERNEL_THREADS:-8}
PRIOR_POOL_WORKERS=${PRIOR_POOL_WORKERS:-0}
PRIOR_POOL_WINDOW=${PRIOR_POOL_WINDOW:-128}
PRIOR_POOL_MB=${PRIOR_POOL_MB:-2048}

source "$REPO_ROOT/scripts/icl_recipe.sh"
source "$REPO_ROOT/scripts/prior_recipe.sh"

CKPT_DIR=${CKPT_DIR:-/ossfs/workspace/tabicl/checkpoints/tabiclv2-clf/stage2}
LOG_DIR=${LOG_DIR:-$CKPT_DIR/logs}
STAGE1_CKPT=${STAGE1_CKPT:-/ossfs/workspace/tabicl/checkpoints/tabiclv2-clf/stage1/step-500000.ckpt}
WANDB_LOG=${WANDB_LOG:-False}
TENSORBOARD_LOG=${TENSORBOARD_LOG:-False}

# Load the Stage 1 weights only on the very first launch. On later launches (e.g.
# restarts after a cluster time limit), CKPT_DIR already contains Stage 2
# checkpoints, and the trainer instead resumes from the latest one with full
# optimizer/scheduler state (--checkpoint_path would override that, so it must
# only be passed the first time).
RESUME_ARGS=(--checkpoint_path "$STAGE1_CKPT" --only_load_model True)
if compgen -G "$CKPT_DIR/step-*.ckpt" >/dev/null; then
    RESUME_ARGS=()
fi

run_pretrain torchrun --standalone --nproc_per_node=$NUM_GPUS -m tabicl.train \
            --wandb_log "$WANDB_LOG" \
            --wandb_project TabICLv2 \
            --wandb_name tabiclv2_clf_stage2 \
            --log_dir "$LOG_DIR" \
            --tensorboard_log "$TENSORBOARD_LOG" \
            --device cuda \
            --dtype "$TRAIN_DTYPE" \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 40000 \
            --batch_size 64 \
            --micro_batch_size 1 \
            --lr 1e-4 \
            --muon True \
            --beta1 0.9 \
            --weight_decay 0.01 \
            --use_cautious_wd False \
            --scheduler cosine_with_restarts \
            --warmup_proportion 0.01 \
            --cosine_num_cycles 1 \
            --cosine_amplitude_decay 1 \
            --cosine_lr_end 1e-7 \
            --gradient_clipping 10.0 \
            --prior_type graph_scm \
            --prior_device cpu \
            "${PRIOR_ARGS[@]}" \
            --batch_size_per_gp 1 \
            --min_features 1 \
            --max_features 100 \
            --max_classes 10 \
            --min_seq_len 400 \
            --max_seq_len 10240 \
            --log_seq_len True \
            --min_train_size 0.79 \
            --max_train_size 0.81 \
            --seq_len_per_gp True \
            --graph_noise False \
            --filter_unpredictable_graphs True \
            --filter_unpredictable_datasets True \
            --allow_act_warping False \
            --min_n_nodes 2 \
            --max_n_nodes 32 \
            --cauchy_dag_offset 0.0 \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 8 \
            --col_num_inds 128 \
            --col_affine False \
            --col_feature_group same \
            --col_feature_group_size 3 \
            --col_target_aware True \
            --col_ssmax True \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --row_rope_interleaved False \
            --icl_num_blocks 12 \
            --icl_nhead 8 \
            "${ICL_ARGS[@]}" \
            --ff_factor 2 \
            --norm_first True \
            --zero_init False \
            --use_flash_attn3 "${USE_FLASH_ATTN3:-True}" \
            --recompute "$RECOMPUTE" \
            --checkpoint_dir "$CKPT_DIR" \
            ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
            --save_temp_every 50 \
            --save_perm_every 1000
