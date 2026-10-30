# Training performance controls

`train_v2_clf_stage1.sh` defaults to 2048 rows, eight GPUs, `COMPILE_FRONTEND=all`
and `COMPILE_WARMUP=True`. All remain environment overrides; for 16 GPUs set
`NUM_GPUS=16`. FP32 uses `TRAIN_DTYPE=float32`, BF16 uses `bfloat16` (default).
Optimizer compilation remains opt-in.

The shared `performance_recipe.sh` selects `ATTENTION_BACKEND=auto`. On the
measured ZW810 vendor PyTorch 2.8 stack, FP32 with at most 2048 maximum rows uses
MATH SDPA; longer sequences use MATH only in ROW attention to avoid quadratic
COL/ICL temporary memory. BF16 and other hardware retain native dispatch.
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
