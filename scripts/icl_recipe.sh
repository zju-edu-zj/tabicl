#!/usr/bin/env bash
# Shared experimental ICL recipe; source from all six v2 curriculum scripts.
ICL_ATTENTION=${ICL_ATTENTION:-elu_linear_flex}
ICL_SPARSE_EVERY=${ICL_SPARSE_EVERY:-3}
ICL_INDUCED_NUM_SLOTS=${ICL_INDUCED_NUM_SLOTS:-64}
ICL_FLEX_BLOCK_SIZE=${ICL_FLEX_BLOCK_SIZE:-128}
ICL_FLEX_NUM_KV_BLOCKS=${ICL_FLEX_NUM_KV_BLOCKS:-4}
ICL_FLEX_QK_NORM=${ICL_FLEX_QK_NORM:-True}
ICL_FLEX_SCALE_MODE=${ICL_FLEX_SCALE_MODE:-bounded}
ICL_LINEAR_NUM_FEATURES=${ICL_LINEAR_NUM_FEATURES:-64}
ICL_SSMAX=${ICL_SSMAX:-False}
SSMAX_TYPE=${SSMAX_TYPE:-qassmax-mlp-elementwise}
ICL_DIAGNOSTICS_EVERY=${ICL_DIAGNOSTICS_EVERY:-200}
ICL_DIAGNOSTIC_QUERIES=${ICL_DIAGNOSTIC_QUERIES:-16}
ICL_DIAGNOSTIC_TOPK=${ICL_DIAGNOSTIC_TOPK:-32}
DRY_RUN=${DRY_RUN:-0}
FAIL_FAST=${FAIL_FAST:-True}
TRAIN_DTYPE=${TRAIN_DTYPE:-float32}
RECOMPUTE=${RECOMPUTE:-False}
ICL_ARGS=(
    --icl_attention "$ICL_ATTENTION"
    --icl_sparse_every "$ICL_SPARSE_EVERY"
    --icl_induced_num_slots "$ICL_INDUCED_NUM_SLOTS"
    --icl_flex_block_size "$ICL_FLEX_BLOCK_SIZE"
    --icl_flex_num_kv_blocks "$ICL_FLEX_NUM_KV_BLOCKS"
    --icl_flex_qk_norm "$ICL_FLEX_QK_NORM"
    --icl_flex_scale_mode "$ICL_FLEX_SCALE_MODE"
    --icl_linear_num_features "$ICL_LINEAR_NUM_FEATURES"
    --icl_ssmax "$ICL_SSMAX"
    --ssmax_type "$SSMAX_TYPE"
    --icl_diagnostics_every "$ICL_DIAGNOSTICS_EVERY"
    --icl_diagnostic_queries "$ICL_DIAGNOSTIC_QUERIES"
    --icl_diagnostic_topk "$ICL_DIAGNOSTIC_TOPK"
    --fail_fast "$FAIL_FAST"
)
run_pretrain() {
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}
