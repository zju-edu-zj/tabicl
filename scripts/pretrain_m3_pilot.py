"""Bounded, locally logged Stage-1 classification pilot using the official Trainer."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Allow ``python scripts/pretrain_m3_pilot.py`` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabicl.train._run import Trainer
from tabicl.train._train_config import build_parser


def evaluate(trainer, batch):
    """Use train forward under no_grad to bypass inference preprocessing/autobatching."""
    results = []
    with torch.no_grad(), trainer.amp_ctx, trainer._sdpa_ctx():
        for i in range(batch[0].shape[0]):
            x, y, d, lengths, sizes = [t[i:i+1] for t in batch]
            n, s = int(lengths[0]), int(sizes[0])
            x, y = trainer.align_micro_batch(x, y, d, n)
            x, y = x.to(trainer.config.device), y.to(trainer.config.device)
            logits = trainer.raw_model(x, y[:, :s], None).float()
            truth = y[:, s:].long()
            ce = F.cross_entropy(logits.flatten(0, 1), truth.flatten()).item()
            accuracy = (logits.argmax(-1) == truth).float().mean().item()
            counts = torch.bincount(y[:, :s].long().flatten(), minlength=10).float()
            probs = (counts + 1) / (s + 10)
            baseline = -probs[truth].log().mean().item()
            majority = (truth == counts.argmax()).float().mean().item()
            results.append([ce, accuracy, baseline, majority])
    means = np.mean(results, axis=0)
    return dict(zip(("val/ce", "val/accuracy", "val/prior_ce", "val/majority_accuracy"), means.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--schedule-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=20)
    parser.add_argument("--val-tables", type=int, default=16)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    if args.steps > args.schedule_steps:
        parser.error("steps must not exceed schedule-steps")
    if not 0 < args.val_tables <= args.batch_size:
        parser.error("val-tables must be between 1 and batch-size")
    if args.batch_size % args.micro_batch_size or 4 % args.micro_batch_size:
        parser.error("micro-batch-size must divide batch-size and the generation group size (4)")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "metrics.jsonl").exists():
        parser.error("run-dir already contains metrics; use a fresh directory")
    repo = Path(__file__).resolve().parents[1]
    tokens = [t for t in shlex.split((repo / "scripts/train_v2_clf_stage1.sh").read_text(), comments=True) if t.strip()]
    start = tokens.index("--wandb_log")
    config = build_parser().parse_args(tokens[start:])
    config.checkpoint_dir = str(run / "checkpoints")
    config.max_steps = args.schedule_steps
    config.warmup_proportion = -1
    config.warmup_steps = args.warmup_steps
    config.batch_size = args.batch_size
    config.micro_batch_size = args.micro_batch_size
    config.n_jobs = args.workers
    config.dtype = args.dtype
    config.amp = args.dtype != "float32"
    config.recompute = True
    config.icl_sparse_chunk_size = 128
    config.wandb_log = False
    config.fail_fast = True
    config.model_compile = False
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.set_num_threads(2)
    (run / "config.json").write_text(json.dumps(vars(config), indent=2))
    (run / "pilot.json").write_text(json.dumps(vars(args), default=str, indent=2))
    (run / "source.diff").write_text(subprocess.check_output(["git", "diff", "HEAD"], cwd=repo, text=True))
    (run / "launcher.py").write_text(Path(__file__).read_text())
    (run / "environment.txt").write_text(subprocess.check_output(["/home/zj/miniconda3/envs/stdplm/bin/python", "-m", "pip", "freeze"], text=True))
    trainer = Trainer(config)
    writer = SummaryWriter(str(run / "tensorboard"), flush_secs=10)
    metrics = (run / "metrics.jsonl").open("a", buffering=1)
    started = time.monotonic()
    (run / "status.json").write_text(json.dumps({"status": "running", "pid": os.getpid(), "step": 0}))

    def log(step, values):
        row = {"step": step, "elapsed_s": time.monotonic() - started, **values}
        metrics.write(json.dumps(row, allow_nan=False) + "\n")
        for key, value in values.items():
            writer.add_scalar(key, value, step)
        writer.flush()
        print(json.dumps(row), flush=True)

    try:
        iterator = iter(trainer.dataloader)
        print("Generating independent fixed validation tables...", flush=True)
        # First generated batch is reserved entirely for validation, never used for training.
        heldout = next(iterator)
        heldout = [t.to_padded_tensor(0.0) if t.is_nested else t for t in heldout]
        # Spread the selection across generation groups, rather than taking only
        # the first group (which shares sequence length and support fraction).
        indices = torch.linspace(0, args.batch_size - 1, args.val_tables).round().long()
        heldout = [t[indices].clone() for t in heldout]
        torch.save(heldout, run / "validation.pt")
        trainer.model.train()
        log(0, evaluate(trainer, heldout))
        for step in range(1, args.steps + 1):
            wait_start = time.monotonic()
            batch = next(iterator)
            wait_s = time.monotonic() - wait_start
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            train_start = time.monotonic()
            values = trainer.run_batch(batch)
            torch.cuda.synchronize()
            train_s = time.monotonic() - train_start
            trainer.curr_step = step
            values = {"train/" + k: v for k, v in values.items()}
            values.update({"system/data_wait_s": wait_s, "system/train_s": train_s,
                           "system/peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                           "system/peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                           "data/mean_features": batch[2].float().mean().item(),
                           "data/mean_support": batch[4].float().mean().item()})
            if step % args.val_every == 0 or step == args.steps:
                values.update(evaluate(trainer, heldout))
                trainer.save_checkpoint(f"step-{step}.ckpt")
            log(step, values)
            (run / "status.json").write_text(json.dumps({"status": "running", "pid": os.getpid(), "step": step}))
        (run / "status.json").write_text(json.dumps({"status": "completed", "step": trainer.curr_step}))
    except BaseException as exc:
        (run / "status.json").write_text(json.dumps({"status": "failed", "step": trainer.curr_step, "error": repr(exc)}))
        raise
    finally:
        writer.close()
        metrics.close()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
