# Accelerated CUDA training

The tested runtime is Python 3.12, PyTorch 2.8.0+cu128 and Triton 3.4.0:

```bash
python -m pip install -r requirements-cu128.txt
python -m pip install -e '.[pretrain]'
python -m pip check
```

Use an NVIDIA driver compatible with the CUDA 12.8 wheel and a working C/C++
compiler (tested with GCC 11.2). PyTorch supplies its CUDA runtime; a matching
system `nvcc` is not required for the existing Triton/Inductor paths. Building
third-party CUDA extensions separately can require a matching toolkit.

Training supports two optional compiler settings:

```bash
COMPILE_FRONTEND=all COMPILE_OPTIMIZER=True TRAIN_DTYPE=bfloat16 \
    bash scripts/train_v2_clf_stage1.sh
```

Set the launcher's usual `NUM_GPUS`, `CKPT_DIR`, `LOG_DIR` and batch variables
for your installation. Use `TRAIN_DTYPE=float32` for FP32 training. Equivalent
CLI arguments are `--compile_frontend all --compile_optimizer True`.

`compile_frontend` accepts `none` (default), `row` or `all`. It compiles column
embedding and/or row aggregation with dynamic shapes before DDP wrapping;
routing and the existing Triton ICL kernels keep their own execution paths.
For these component calls, the Torch 2.8 DDP graph-bucket splitting compiler
pass is locally disabled to avoid its RoPE scalar-output compilation failure.
This does not disable DDP gradient reduction or persistently change the global
compiler configuration. Single-rank NCCL training and resume are tested;
multi-rank communication/scaling are not covered by that check.
SSMax length scaling stays inside the tensor graph so changing support lengths
does not specialize a Python logarithm and exhaust the default graph cache.
`compile_optimizer` defaults to false and compiles only Muon's Newton–Schulz
tensor algebra. It has no effect for all-AdamW training (`--muon False`). Both
compiler options require PyTorch >=2.8; this repository pins the tested 2.8.0 /
3.4.0 pair rather than claiming compatibility with every newer release.

Compilation adds cold-start latency and can compile additional graphs for new
shapes. Keep the compiler cache on persistent writable local storage. Large
tables may still need `RECOMPUTE=True` and a smaller micro-batch. Compilation
does not remove the need to provision CPU workers and prefetching for the
`graph_scm` prior.

The measured gains are steady-state, not launch-to-exit speedups. With the
tested 3-column/3-row/12-ICL-block model, batch 8 / micro-batch 4, a complete
1024-row, 100-column step improved from about 628 to 373 ms in BF16 and 1365
to 1093 ms in FP32, excluding the prior and DDP. Actual single-rank NCCL/prior
training improved from about 585 to 366 ms and 1351 to 1074 ms respectively
(training time only). Short tables also benefit. These are same-Torch-2.8
comparisons against the already optimized ICL baseline, not portable guarantees.
FP32 measurements allow TF32 matmul, as in the training configuration.
Frontend compilation supplies most of the gain; compiling Muon adds little
in these measurements and can be left off to reduce startup complexity.

Module registration, parameter identities, model checkpoint keys, optimizer
state layout and parameter routing stay unchanged. These options can be
enabled when loading existing weights. Floating point fusion can change
rounding, including FP32; sparse hard routing can amplify small differences.
On a 48-table saved-checkpoint check, the compiled frontend changed about
2.23% / 3.35% of FP32 / BF16 predictions. Average accuracy changed by -0.23 /
-0.24 percentage points; the per-table paired bootstrap intervals crossed zero,
but individual tables changed substantially. Strict loading is not a prediction
equivalence guarantee.
Validate saved models on your tasks when switching numerical implementations.
Defaults keep compilation optional. Long training quality and multi-node
scaling require separate evaluation.
