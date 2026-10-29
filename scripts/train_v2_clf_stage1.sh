#!/bin/bash
set -euo pipefail
# TabICLv2 pre-training -- CLASSIFIER, Stage 1 (of 3).
# (The regressor is a separate checkpoint; see train_v2_reg_stage{1,2,3}.sh.)
#
# Recipe from "TabICLv2: A better, faster, scalable, and open tabular foundation
# model" (https://arxiv.org/abs/2602.11139), §4.1 / Appendix B.1. Three-stage
# curriculum that progressively increases dataset size (batch size 64 throughout):
#
#   Stage 1: 500K steps, 1,024 samples/dataset,       30-90% train, max LR 8e-4, grad clip 10
#   Stage 2:  40K steps, 400-10,240 samples (log-u), 79-81% train, max LR 1e-4, grad clip 10
#   Stage 3:  10K steps, 400-60,000 samples (log-u), 79-81% train, max LR 2e-5, grad clip  1
#
# Hybrid optimizer: transformer projection matrices use Muon; other parameters
# use AdamW. Norms and biases have no weight decay. Legacy all-Muon checkpoints
# must initialize a NEW run via INIT_CKPT (weights only).
# Common: weight decay parameter 0.01 (see note below),
# cosine schedule, per-micro-batch train/test sizes (--seq_len_per_gp),
# up to 100 features, 8 attention heads everywhere, graph_scm prior.
# TRAIN_DTYPE defaults to bfloat16 in icl_recipe.sh; set it to float32 for the
# FP32 path. Float16 can lead to training instabilities.
#
# The classifier uses LayerNorm WITH biases (the --norm_type default; paper Table A.1),
# unlike the regressor, which uses bias-free LayerNorm (--norm_type layernorm_nobias).
# v2 uses standard residual init (--zero_init False; v1 zero-initializes residual
# branches) and enables FlashAttention-3 only in stages 2 & 3 (--use_flash_attn3).
# Experimental recipe: ELU+1 global layers; every third layer fuses global
# and per-query/per-head selected-row softmax. See docs/icl_attention.md.
#
# Adjust the placeholder paths and --nproc_per_node / --n_jobs for your hardware.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

NUM_GPUS=${NUM_GPUS:-16}                      # the paper used 4 GPUs for pre-training
CKPT_DIR=${CKPT_DIR:-/ossfs/workspace/tabicl/checkpoints/tabiclv2-clf/stage1}
LOG_DIR=${LOG_DIR:-$CKPT_DIR/logs}
MAX_STEPS=${MAX_STEPS:-500000}
LR=${LR:-8e-4}                              # Muon matrix peak LR
ADAMW_LR=${ADAMW_LR:-1e-4}                  # hybrid AdamW peak LR
MUON=${MUON:-True}                          # False: all-AdamW control, using LR
BATCH_SIZE=${BATCH_SIZE:-64}                # global batch, before DDP splitting
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
WARMUP_PROPORTION=${WARMUP_PROPORTION:--1}
WARMUP_STEPS=${WARMUP_STEPS:-5000}
INIT_CKPT=${INIT_CKPT:-}                     # first launch only; reset optimizer/step
TENSORBOARD_LOG=${TENSORBOARD_LOG:-False}
SAVE_TEMP_EVERY=${SAVE_TEMP_EVERY:-500}
SAVE_PERM_EVERY=${SAVE_PERM_EVERY:-5000}
WANDB_LOG=${WANDB_LOG:-False}
WANDB_MODE=${WANDB_MODE:-offline}
PRIOR_WORKERS=${PRIOR_WORKERS:-8}             # 8 x 16 ranks = 128 workers on the training node
PRIOR_PREFETCH_FACTOR=${PRIOR_PREFETCH_FACTOR:-8}
PRIOR_IN_ORDER=${PRIOR_IN_ORDER:-False}       # avoid graph_scm head-of-line blocking
source "$REPO_ROOT/scripts/icl_recipe.sh"
mkdir -p "$LOG_DIR"

# Reusing the same command after interruption resumes the new hybrid run,
# rather than repeatedly reinitializing it from the old checkpoint.
RESUME_ARGS=()
if compgen -G "$CKPT_DIR/step-*.ckpt" >/dev/null; then
    : # Trainer automatically loads the latest checkpoint with optimizer/scheduler.
elif [[ -n "$INIT_CKPT" ]]; then
    RESUME_ARGS=(--checkpoint_path "$INIT_CKPT" --only_load_model True)
fi

run_pretrain torchrun --standalone --nproc_per_node=$NUM_GPUS -m tabicl.train \
            --wandb_log "$WANDB_LOG" \
            --wandb_project TabICLv2 \
            --wandb_name "${WANDB_NAME:-tabiclv2_clf_stage1_hybrid}" \
            --wandb_dir "$LOG_DIR/wandb" \
            --wandb_mode "$WANDB_MODE" \
            --log_dir "$LOG_DIR" \
            --tensorboard_log "$TENSORBOARD_LOG" \
            --device cuda \
            --dtype "$TRAIN_DTYPE" \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps "$MAX_STEPS" \
            --batch_size "$BATCH_SIZE" \
            --micro_batch_size "$MICRO_BATCH_SIZE" \
            --lr "$LR" \
            --adamw_lr "$ADAMW_LR" \
            --muon "$MUON" \
            --beta1 0.9 \
            --weight_decay 0.01 \
            --use_cautious_wd False \
            --scheduler cosine_with_restarts \
            --warmup_proportion "$WARMUP_PROPORTION" \
            --warmup_steps "$WARMUP_STEPS" \
            --cosine_num_cycles 1 \
            --cosine_amplitude_decay 1 \
            --cosine_lr_end 1e-7 \
            --gradient_clipping 10.0 \
            --prior_type graph_scm \
            --prior_device cpu \
            --n_jobs "$PRIOR_WORKERS" \
            --prior_prefetch_factor "$PRIOR_PREFETCH_FACTOR" \
            --prior_in_order "$PRIOR_IN_ORDER" \
            --batch_size_per_gp 4 \
            --min_features 1 \
            --max_features 100 \
            --max_classes 10 \
            --max_seq_len "$MAX_SEQ_LEN" \
            --min_train_size 0.3 \
            --max_train_size 0.9 \
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
            --use_flash_attn3 "${USE_FLASH_ATTN3:-False}" \
            --recompute "$RECOMPUTE" \
            --checkpoint_dir "$CKPT_DIR" \
            ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
            --save_temp_every "$SAVE_TEMP_EVERY" \
            --save_perm_every "$SAVE_PERM_EVERY"

# JSONL is always written; TensorBoard and W&B are opt-in.

# Note on cautious weight decay: the paper reports using cautious weight decay (parameter
# 0.01), but the reference pretraining runs that produced the released checkpoints
# did NOT actually pass it to Muon (it was left unwired), so --use_cautious_wd is set
# to False here to reproduce that behavior. Set it to True to enable it as described
# in the paper. --weight_decay 0.01 is the decay strength either way.
