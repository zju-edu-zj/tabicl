#!/usr/bin/env bash
# Shared experimental ICL recipe; source from all six v2 curriculum scripts.
ICL_ATTENTION=${ICL_ATTENTION:-global_sparse}
ICL_SPARSE_EVERY=${ICL_SPARSE_EVERY:-3}
ICL_BLOCK_SIZE=${ICL_BLOCK_SIZE:-128}
ICL_NUM_SELECTED_BLOCKS=${ICL_NUM_SELECTED_BLOCKS:-4}
ICL_QUERY_CHUNK_SIZE=${ICL_QUERY_CHUNK_SIZE:-64}
ICL_TOKEN_TOPK=${ICL_TOKEN_TOPK:-128}
ICL_INDEX_DIM=${ICL_INDEX_DIM:-16}
ICL_INDEX_QUERY_SAMPLES=${ICL_INDEX_QUERY_SAMPLES:-32}
ICL_INDEX_LOSS_WEIGHT=${ICL_INDEX_LOSS_WEIGHT:-0.05}
ICL_INDEX_LOSS_WARMUP_STEPS=${ICL_INDEX_LOSS_WARMUP_STEPS:-5000}
ICL_INDEX_LR=${ICL_INDEX_LR:-3e-4}
ICL_INDEX_REGRESSION_TEMPERATURE=${ICL_INDEX_REGRESSION_TEMPERATURE:-0.2}
ICL_ROUTER=${ICL_ROUTER:-centroid}
ICL_ROUTER_BEAM=${ICL_ROUTER_BEAM:-16}
ICL_PARTITION=${ICL_PARTITION:-projection}
ICL_GLOBAL_TYPE=${ICL_GLOBAL_TYPE:-elu}
ICL_NUM_INDUCED_SLOTS=${ICL_NUM_INDUCED_SLOTS:-128}
ICL_SSMAX=${ICL_SSMAX:-False}
ICL_QK_NORM=${ICL_QK_NORM:-True}
COL_SSMAX_TYPE=${COL_SSMAX_TYPE:-${SSMAX_TYPE:-qassmax-mlp-elementwise}}
ICL_DIAGNOSTICS_EVERY=${ICL_DIAGNOSTICS_EVERY:-200}
ICL_DIAGNOSTIC_QUERIES=${ICL_DIAGNOSTIC_QUERIES:-16}
ICL_DIAGNOSTIC_TOPK=${ICL_DIAGNOSTIC_TOPK:-32}
DRY_RUN=${DRY_RUN:-0}
FAIL_FAST=${FAIL_FAST:-True}
TRAIN_DTYPE=${TRAIN_DTYPE:-bfloat16}
RECOMPUTE=${RECOMPUTE:-False}
COMPILE_FRONTEND=${COMPILE_FRONTEND:-none}
COMPILE_OPTIMIZER=${COMPILE_OPTIMIZER:-False}
ICL_ARGS=(
    --compile_frontend "$COMPILE_FRONTEND"
    --compile_optimizer "$COMPILE_OPTIMIZER"
    --icl_attention "$ICL_ATTENTION"
    --icl_router "$ICL_ROUTER"
    --icl_router_beam "$ICL_ROUTER_BEAM"
    --icl_partition "$ICL_PARTITION"
    --icl_global_type "$ICL_GLOBAL_TYPE"
    --icl_num_induced_slots "$ICL_NUM_INDUCED_SLOTS"
    --icl_sparse_every "$ICL_SPARSE_EVERY"
    --icl_block_size "$ICL_BLOCK_SIZE"
    --icl_num_selected_blocks "$ICL_NUM_SELECTED_BLOCKS"
    --icl_query_chunk_size "$ICL_QUERY_CHUNK_SIZE"
    --icl_token_topk "$ICL_TOKEN_TOPK"
    --icl_index_dim "$ICL_INDEX_DIM"
    --icl_index_query_samples "$ICL_INDEX_QUERY_SAMPLES"
    --icl_index_loss_weight "$ICL_INDEX_LOSS_WEIGHT"
    --icl_index_loss_warmup_steps "$ICL_INDEX_LOSS_WARMUP_STEPS"
    --icl_index_lr "$ICL_INDEX_LR"
    --icl_index_regression_temperature "$ICL_INDEX_REGRESSION_TEMPERATURE"
    --icl_ssmax "$ICL_SSMAX"
    --icl_qk_norm "$ICL_QK_NORM"
    --col_ssmax_type "$COL_SSMAX_TYPE"
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
