#!/usr/bin/env bash
# Shared node-level prior controls for all curriculum stages. Worker count is
# independent of GPU count; 112 workers leave 16 cores on the 128-core node.
PRIOR_BACKEND=${PRIOR_BACKEND:-node_pool}   # dataloader is default
PRIOR_POOL_WORKERS=${PRIOR_POOL_WORKERS:-112}
PRIOR_POOL_WINDOW=${PRIOR_POOL_WINDOW:-128}
PRIOR_POOL_MB=${PRIOR_POOL_MB:-2048}
PRIOR_WORKERS=${PRIOR_WORKERS:-8}
PRIOR_PREFETCH_FACTOR=${PRIOR_PREFETCH_FACTOR:-8}
PRIOR_IN_ORDER=${PRIOR_IN_ORDER:-False}
PRIOR_ARGS=(
    --prior_backend "$PRIOR_BACKEND"
    --prior_pool_workers "$PRIOR_POOL_WORKERS"
    --prior_pool_window "$PRIOR_POOL_WINDOW"
    --prior_pool_mb "$PRIOR_POOL_MB"
    --n_jobs "$PRIOR_WORKERS"
    --prior_prefetch_factor "$PRIOR_PREFETCH_FACTOR"
    --prior_in_order "$PRIOR_IN_ORDER"
)
