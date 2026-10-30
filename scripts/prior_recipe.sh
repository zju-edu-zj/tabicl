#!/usr/bin/env bash
# Worker count is per NODE; zero respects affinity and cgroup CPU quotas.
PRIOR_BACKEND=${PRIOR_BACKEND:-node_pool}
if [[ "$PRIOR_BACKEND" == node_pool ]]; then
    PRIOR_SCHEDULE=${PRIOR_SCHEDULE:-balanced}
else
    PRIOR_SCHEDULE=${PRIOR_SCHEDULE:-none}
fi
PRIOR_POOL_WORKERS=${PRIOR_POOL_WORKERS:-0}
RANK_TIMING_EVERY=${RANK_TIMING_EVERY:-100}
PRIOR_POOL_WINDOW=${PRIOR_POOL_WINDOW:-128}
PRIOR_POOL_MB=${PRIOR_POOL_MB:-2048}
PRIOR_WORKERS=${PRIOR_WORKERS:-8}
PRIOR_PREFETCH_FACTOR=${PRIOR_PREFETCH_FACTOR:-8}
PRIOR_IN_ORDER=${PRIOR_IN_ORDER:-False}
PRIOR_ARGS=(
    --prior_backend "$PRIOR_BACKEND"
    --prior_schedule "$PRIOR_SCHEDULE"
    --rank_timing_every "$RANK_TIMING_EVERY"
    --prior_pool_workers "$PRIOR_POOL_WORKERS"
    --prior_pool_window "$PRIOR_POOL_WINDOW"
    --prior_pool_mb "$PRIOR_POOL_MB"
    --n_jobs "$PRIOR_WORKERS"
    --prior_prefetch_factor "$PRIOR_PREFETCH_FACTOR"
    --prior_in_order "$PRIOR_IN_ORDER"
)
