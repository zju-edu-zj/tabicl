# Training performance controls

See [the shape-cohort/profile audit](shape_cohort_profile_20261031.md) for
generation-before-conditioning scheduling, crop/no-crop CPU diagnostics, and
the reproducible per-rank GPU profile matrix. Two-seed target-device results
show mixed gains, so the new schedule remains opt-in; training quality is not
yet validated.

`train_v2_clf_stage1.sh` defaults to 2048 rows, eight GPUs, `COMPILE_FRONTEND=all`
and `COMPILE_WARMUP=True`. All remain environment overrides; for 16 GPUs set
`NUM_GPUS=16`. FP32 uses `TRAIN_DTYPE=float32`, BF16 uses `bfloat16` (default).
Optimizer compilation remains opt-in.

Stage1 FP32 defaults to a shared `node_pool` with 56 workers per node, using `balanced`
when each rank has multiple micro-batches, otherwise completion-order `none`.
Stage1 BF16 defaults to DataLoader: the 640-step measurement favored it over
the shared pool. Set `PRIOR_BACKEND=node_pool` to experiment with BF16; its
default schedule is `none`. Other curriculum scripts retain shared-pool defaults. Override with
`PRIOR_SCHEDULE=balanced/round_robin/none`, or `PRIOR_BACKEND=dataloader`.
See [the scheduling audit](training_schedule_20261030.md) for measured results,
stage2/3 behavior and the limits of fixed-cohort balancing.

Stage2/3 also default to 56 pool workers, reserving spare permits for large
distance kernels (`PRIOR_POOL_KERNEL_THREADS=8`). Large NaN/Inf fallbacks use
the same node CPU budget. Set `PRIOR_POOL_WORKERS=0` for quota-aware auto sizing;
56 is a measured-machine default, not a guarantee that every host has spare
cores. The [CPU-budget and centroid top-k audit](cpu_budget_topk_20261031.md)
defines per-step CPU generation, GPU activity and exposed generation stall,
and documents the selector's bounded reusable workspace and exact-route tests.

The shared `performance_recipe.sh` selects `ATTENTION_BACKEND=auto`. On the
measured ZW810 vendor PyTorch 2.8 stack, FP32 with at most 2048 maximum rows uses
MATH SDPA; longer sequences use MATH only in ROW attention to limit materialized
attention-weight memory. For this recipe, induced COL attention scales with
rows × inducing slots × columns; it is not dense quadratic attention over rows.
BF16 and other hardware retain native dispatch.
Compiled classifier scripts default to `USE_FLASH_ATTN3=False`: the external
vendor FA3 extension lacks FakeTensor support and fails during compilation.
Native SDPA still selects its fused backend for BF16. An explicit external-FA3
override remains available for compatible environments.
Overrides: `native`, `fp32_math`, `fp32_row_math`. Explicit FP32 math policies
require `USE_FLASH_ATTN3=False`; external FA3 would otherwise cast FP32 to FP16.
MATH changes accumulation order and therefore rounding, without downcasting.
The earlier 4096-row all-MATH experiment exhausted 96 GiB; do not extrapolate
2048-row speedups to longer sequences or different micro-batch/model sizes.

Warmup runs three forward/backward shapes concurrently on every rank, including
one-column specialization and the support-size interval. It restores RNG,
buffers and module modes, clears temporary gradients, never steps the optimizer
or scheduler, and consumes no prior batches. Rank count does not turn these into
24 sequential training steps. Startup and the final barrier are timed per rank.
This moves compilation to startup, but cannot promise no later compilation
stalls: different strides, shapes, dtypes or branches can create new guards.
For short experiments compare total wall time and consider
`COMPILE_WARMUP=False COMPILE_FRONTEND=none`.

Reproduce an isolated run with the vendor interpreter:

```bash
/tmp/tabicl-perf-venv/bin/python scripts/benchmark_training_schedule.py \
  --gpus 8 --dtype bfloat16 --steps 100 --output /tmp/tabicl-bench-bf16
```

The benchmark launches the real curriculum script and records total wall time,
including startup and the final checkpoint. Use a fresh output directory to
avoid automatic checkpoint resume. Per-step JSONL lives under `logs/`.
