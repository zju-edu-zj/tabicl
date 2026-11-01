#!/usr/bin/env bash
# Workers are per NODE. 56 leaves headroom on the measured 69-core/8-rank
# node; override with 0 for quota-aware auto sizing on other machines.
PRIOR_BACKEND=${PRIOR_BACKEND:-node_pool}
if [[ "$PRIOR_BACKEND" == node_pool ]]; then
    PRIOR_SCHEDULE=${PRIOR_SCHEDULE:-balanced}
else
    PRIOR_SCHEDULE=${PRIOR_SCHEDULE:-none}
fi
PRIOR_POOL_WORKERS=${PRIOR_POOL_WORKERS:-56}
RANK_TIMING_EVERY=${RANK_TIMING_EVERY:-100}
PRIOR_POOL_WINDOW=${PRIOR_POOL_WINDOW:-128}
PRIOR_POOL_MB=${PRIOR_POOL_MB:-2048}
PRIOR_POOL_TRANSPORT=${PRIOR_POOL_TRANSPORT:-numpy}
PRIOR_BUCKET_STEPS=${PRIOR_BUCKET_STEPS:-4}
PRIOR_POOL_KERNEL_THREADS=${PRIOR_POOL_KERNEL_THREADS:-1}
ADAPTIVE_FEATURE_LIMIT=${ADAPTIVE_FEATURE_LIMIT:-True}
PRIOR_WORKERS=${PRIOR_WORKERS:-8}
PRIOR_PREFETCH_FACTOR=${PRIOR_PREFETCH_FACTOR:-8}
PRIOR_IN_ORDER=${PRIOR_IN_ORDER:-False}
PRIOR_FILTER_MODE=${PRIOR_FILTER_MODE:-full}
PRIOR_READY_MAX_DELAY=${PRIOR_READY_MAX_DELAY:-2}
PRIOR_CATEGORICAL_SAMPLING=${PRIOR_CATEGORICAL_SAMPLING:-legacy}
PRIOR_ARGS=(
    --prior_backend "$PRIOR_BACKEND"
    --prior_schedule "$PRIOR_SCHEDULE"
    --rank_timing_every "$RANK_TIMING_EVERY"
    --prior_pool_workers "$PRIOR_POOL_WORKERS"
    --prior_pool_window "$PRIOR_POOL_WINDOW"
    --prior_pool_mb "$PRIOR_POOL_MB"
    --prior_pool_transport "$PRIOR_POOL_TRANSPORT"
    --prior_bucket_steps "$PRIOR_BUCKET_STEPS"
    --prior_pool_kernel_threads "$PRIOR_POOL_KERNEL_THREADS"
    --adaptive_feature_limit "$ADAPTIVE_FEATURE_LIMIT"
    --n_jobs "$PRIOR_WORKERS"
    --prior_prefetch_factor "$PRIOR_PREFETCH_FACTOR"
    --prior_in_order "$PRIOR_IN_ORDER"
    --dataset_filter_mode "$PRIOR_FILTER_MODE"
    --prior_ready_max_delay "$PRIOR_READY_MAX_DELAY"
    --categorical_sampling "$PRIOR_CATEGORICAL_SAMPLING"
)
