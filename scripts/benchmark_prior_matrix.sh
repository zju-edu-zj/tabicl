#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON=${PYTHON:-python}
NUM_RANKS=${NUM_RANKS:-16}
WORKERS_PER_RANK=${WORKERS_PER_RANK:-7}
POOL_WORKERS=${POOL_WORKERS:-112}
POOL_WINDOW=${POOL_WINDOW:-128}
POOL_MB=${POOL_MB:-2048}
STEPS=${STEPS:-200}
WARMUP=${WARMUP:-16}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-4}
REGRESSION=${REGRESSION:-False}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/prior_benchmarks}
mkdir -p "$OUTPUT_DIR"
# Equal total workers for the two backends. CPU/Gloo ranks emulate lockstep
# DDP consumers; no GPUs are required. Actual training is measured separately.
for stage in 1 2 3; do
    for backend in dataloader node_pool; do
        "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NUM_RANKS" \
            "$REPO_ROOT/scripts/benchmark_prior_pipeline.py" \
            --backend "$backend" --stage "$stage" --batch-size "$LOCAL_BATCH_SIZE" \
            --regression "$REGRESSION" --workers "$WORKERS_PER_RANK" \
            --pool-workers "$POOL_WORKERS" --pool-window "$POOL_WINDOW" --pool-mb "$POOL_MB" \
            --prefetch 8 --in-order False --consumer-ms 0 --warmup "$WARMUP" --steps "$STEPS" \
            --output "$OUTPUT_DIR/stage${stage}_${backend}_throughput.json"
    done
done
for backend in dataloader node_pool; do
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NUM_RANKS" \
        "$REPO_ROOT/scripts/benchmark_prior_pipeline.py" \
        --backend "$backend" --stage 1 --batch-size "$LOCAL_BATCH_SIZE" \
        --regression "$REGRESSION" --workers "$WORKERS_PER_RANK" \
        --pool-workers "$POOL_WORKERS" --pool-window "$POOL_WINDOW" --pool-mb "$POOL_MB" \
        --prefetch 8 --in-order False --consumer-ms 550 --warmup "$WARMUP" --steps "$STEPS" \
        --output "$OUTPUT_DIR/stage1_${backend}_consumer550.json"
done
# Exercise the largest table explicitly; log-uniform draws rarely hit 60k.
for backend in dataloader node_pool; do
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NUM_RANKS" \
        "$REPO_ROOT/scripts/benchmark_prior_pipeline.py" \
        --backend "$backend" --stage 3 --batch-size 1 --min-seq-len 60000 --max-seq-len 60000 \
        --regression "$REGRESSION" --workers "$WORKERS_PER_RANK" \
        --pool-workers "$POOL_WORKERS" --pool-window "$POOL_WINDOW" --pool-mb "$POOL_MB" \
        --prefetch 2 --in-order False --warmup 2 --steps "${LONG_STEPS:-10}" \
        --output "$OUTPUT_DIR/rows60000_${backend}.json"
done
