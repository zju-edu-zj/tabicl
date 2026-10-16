#!/usr/bin/env bash
# Shared experimental ICL recipe; source from all six v2 curriculum scripts.
ICL_ATTENTION=${ICL_ATTENTION:-global_sparse}
ICL_SPARSE_EVERY=${ICL_SPARSE_EVERY:-3}
ICL_BLOCK_SIZE=${ICL_BLOCK_SIZE:-128}
ICL_NUM_SELECTED_BLOCKS=${ICL_NUM_SELECTED_BLOCKS:-4}
ICL_QUERY_CHUNK_SIZE=${ICL_QUERY_CHUNK_SIZE:-64}
ICL_ROUTER=${ICL_ROUTER:-centroid}
ICL_ROUTER_BEAM=${ICL_ROUTER_BEAM:-16}
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
    --icl_router "$ICL_ROUTER"
    --icl_router_beam "$ICL_ROUTER_BEAM"
    --icl_sparse_every "$ICL_SPARSE_EVERY"
    --icl_block_size "$ICL_BLOCK_SIZE"
    --icl_num_selected_blocks "$ICL_NUM_SELECTED_BLOCKS"
    --icl_query_chunk_size "$ICL_QUERY_CHUNK_SIZE"
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
