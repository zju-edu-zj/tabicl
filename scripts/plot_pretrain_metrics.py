"""Render local pilot JSONL metrics; no TensorBoard server is required."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    def records(key):
        aliases = {
            "train/ce": ("train/ce", "ce"),
            "train/accuracy": ("train/accuracy", "accuracy"),
            "train/grad_norm": ("train/grad_norm", "grad_norm"),
            "train/lr_used": ("train/lr_used", "lr_used"),
            "system/train_s": ("system/train_s", "train_time"),
            "system/data_wait_s": ("system/data_wait_s", "prior_time"),
            "system/peak_allocated_gib": ("system/peak_allocated_gib",),
            "system/peak_reserved_gib": ("system/peak_reserved_gib",),
            "val/ce": ("val/ce",),
            "val/accuracy": ("val/accuracy",),
            "val/prior_ce": ("val/prior_ce",),
            "val/majority_accuracy": ("val/majority_accuracy",),
        }
        keys = aliases.get(key, (key,))
        return [(r["step"], next((r[k] for k in keys if k in r), None)) for r in rows]

    train = [r for r in rows if "train/ce" in r or "ce" in r]
    val = [r for r in rows if "val/ce" in r]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("Cross entropy (nats)", ["train/ce", "val/ce", "val/prior_ce"]),
        ("Accuracy", ["train/accuracy", "val/accuracy", "val/majority_accuracy"]),
        ("Gradient norm before clipping", ["train/grad_norm"]),
        ("Learning rate actually used", ["train/lr_used"]),
        ("Time per step (s)", ["system/train_s", "system/data_wait_s"]),
        ("Process CUDA memory (GiB)", ["system/peak_allocated_gib", "system/peak_reserved_gib"]),
    ]
    for ax, (title, keys) in zip(axes.flat, panels):
        for key in keys:
            selected = [(step, value) for step, value in records(key) if value is not None]
            if selected:
                ax.plot([step for step, _ in selected], [value for _, value in selected], marker=".", label=key)
        ax.set(title=title, xlabel="Optimizer step")
        ax.grid(alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)
    axes.flat[2].axhline(10, linestyle="--", color="red", alpha=0.5)
    fig.suptitle(args.run_dir.name + " (raw metrics; fixed synthetic validation)")
    fig.tight_layout()
    fig.savefig(args.run_dir / "curves.png", dpi=150)
    plt.close(fig)
    if train:
        summary = {
            "steps": len(train),
            "last_step": train[-1]["step"],
            "elapsed_s": rows[-1]["elapsed_s"],
            "mean_train_s": statistics.mean(next(r[k] for k in ("system/train_s", "train_time") if k in r) for r in train),
            "min_train_s": min(next(r[k] for k in ("system/train_s", "train_time") if k in r) for r in train),
            "max_train_s": max(next(r[k] for k in ("system/train_s", "train_time") if k in r) for r in train),
            "max_data_wait_s": max(next(r[k] for k in ("system/data_wait_s", "prior_time") if k in r) for r in train),
            "max_allocated_gib": max((r.get("system/peak_allocated_gib", 0) for r in train), default=0),
            "max_reserved_gib": max((r.get("system/peak_reserved_gib", 0) for r in train), default=0),
            "min_grad_norm": min(next(r[k] for k in ("train/grad_norm", "grad_norm") if k in r) for r in train),
            "max_grad_norm": max(next(r[k] for k in ("train/grad_norm", "grad_norm") if k in r) for r in train),
            "clipped_steps": sum(r.get("train/grad_clipped", r.get("grad_clipped", 0)) for r in train),
            "oom_micro_batches": sum(r.get("train/oom_micro_batches", r.get("oom_micro_batches", 0)) for r in train),
            "first_train": train[0], "last_train": train[-1],
            "first_val": val[0] if val else None, "last_val": val[-1] if val else None,
        }
        (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
