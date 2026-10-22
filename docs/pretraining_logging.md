# Stage 1 training records

`scripts/train_v2_clf_stage1.sh` always records JSONL metrics. TensorBoard and
W&B are optional and disabled by default:

```text
$CKPT_DIR/
  step-*.ckpt              model, optimizer, scheduler and current step
  logs/
    metrics.jsonl          one JSON object per optimizer step
    tensorboard/           optional TensorBoard event files
    wandb/                 optional W&B run files
```

The default checkpoint directory is
`/home/zj/table_foundation/checkpoints/tabiclv2-clf/stage1`. Override it, and
bound a trial run, without editing the shell script:

```bash
MAX_STEPS=10 \
CKPT_DIR=/home/zj/table_foundation/checkpoints/m3-stage1-smoke \
bash scripts/train_v2_clf_stage1.sh
```

## What is logged?

The main fields in `metrics.jsonl` are:

| Field | Meaning |
| --- | --- |
| `ce` | Cross-entropy loss on the query rows of the generated meta-training task. Lower is better. |
| `accuracy` | Query-row classification accuracy for that generated batch. |
| `grad_norm` | Gradient norm before clipping. |
| `grad_clipped` | 1 if the configured gradient limit was active. |
| `lr_used` | Learning rate used for the current optimizer update. |
| `oom_micro_batches` | Number of skipped CUDA OOM micro-batches; should remain zero. |
| `prior_time` | Time waiting for the synthetic Graph-SCM batch. |
| `train_time` | GPU forward/backward/update time. |

`ce` is the negative log likelihood in nats:

```text
CE = - mean(log p_model(correct class | support rows, query features))
```

The support rows are shown with labels. Query labels are hidden from the model
and used only to calculate this meta-training loss.

Important: the standard Stage 1 script has **no fixed validation set**. Every
step gets newly generated synthetic tasks, and the query rows in that task are
the training objective, not an independent validation curve. The pilot script
has a fixed synthetic validation set for smoke tests, but that is a separate
diagnostic.

## How to view it

To enable TensorBoard and monitor it live:

```bash
TENSORBOARD_LOG=True bash scripts/train_v2_clf_stage1.sh
tensorboard --logdir /home/zj/table_foundation/checkpoints/m3-stage1-smoke/logs/tensorboard \
  --host 127.0.0.1 --port 6006
```

Then open `http://127.0.0.1:6006`. On a remote machine, forward the port with
SSH. The most useful TensorBoard panels are `train/ce`, `train/accuracy`,
`train/grad_norm`, `train/lr_used`, `system/prior_time`, and
`system/train_time`.

W&B is also opt-in and defaults to offline mode when enabled:

```bash
WANDB_LOG=True bash scripts/train_v2_clf_stage1.sh
```

After the job finishes, inspect it locally or upload it later:

```bash
/home/zj/miniconda3/envs/stdplm/bin/wandb sync \
  /home/zj/table_foundation/checkpoints/m3-stage1-smoke/logs/wandb/offline-run-*
```

For a connected W&B server, launch with `WANDB_LOG=True WANDB_MODE=online`.
JSONL logging is independent of both optional integrations.

`metrics.jsonl` remains the source of truth and can be plotted with
`scripts/plot_pretrain_metrics.py` after a normal or interrupted run.

## What to inspect first

1. `train/ce`: it should decrease over a moving average. A single batch can be
   noisy because the prior is regenerated every step.
2. `train/lr_used`: confirm warmup reaches the intended peak and cosine decay
   starts at the expected step.
3. `train/grad_norm` and `train/grad_clipped`: persistent clipping or a sudden
   spike usually indicates an optimization or numerical problem.
4. `system/prior_time` versus `system/train_time`: if prior time is large,
   increase workers/prefetch; if it is near zero, the GPU/model is the bottleneck.
5. `system/peak_*` and `oom_micro_batches`: memory must remain below the device
   limit and OOM count should be zero.

Training accuracy alone is not sufficient evidence of useful pretraining. For
model selection, periodically evaluate checkpoints on a **fixed** synthetic
validation suite and on the real downstream benchmark.

## About the supplied AUC figure

The supplied figure is a downstream TabArena-style curve: mean AUC over many
real datasets for each checkpoint. It is not the same as CE. Generating it
requires an evaluation job that, at selected checkpoints, runs the model on the
same fixed dataset folds and appends records such as:

```json
{"step": 5000, "method": "m3", "mean_auc": 0.8512}
```

Those records can then be plotted automatically, but the benchmark evaluation
itself must be implemented/configured separately from Stage 1 training. Never
compare a training CE curve to that AUC curve as if they were the same metric.

Once an evaluation runner writes those records, render the figure with:

```bash
python scripts/plot_checkpoint_eval.py /path/to/checkpoint_eval.jsonl \
  --output /path/to/checkpoint_eval.png
```
