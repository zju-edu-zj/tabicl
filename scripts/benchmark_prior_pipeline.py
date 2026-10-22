"""Benchmark graph_scm supply latency, including the slowest DDP rank.

Run the same command shape as training, for example:

  torchrun --standalone --nproc_per_node=16 scripts/benchmark_prior_pipeline.py \
    --workers 8 --prefetch 8 --in-order False --consumer-ms 550 \
    --steps 300 --output prior_pipeline.json

``consumer-ms`` emulates GPU work between fetches so DataLoader prefetch has the
same opportunity to run as it does during training. The output reports local
waits and the per-step maximum across ranks; the latter is what DDP exposes.
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from tabicl.prior._dataset import PriorDataset
from tabicl.prior._genload import seed_worker
from tabicl.prior.graph_lib._config import PriorConfig


def str2bool(value):
    return value if isinstance(value, bool) else value.lower() == "true"


def quantiles(values):
    return {
        "mean_ms": float(values.mean() * 1000),
        "p50_ms": float(np.quantile(values, 0.50) * 1000),
        "p90_ms": float(np.quantile(values, 0.90) * 1000),
        "p95_ms": float(np.quantile(values, 0.95) * 1000),
        "p99_ms": float(np.quantile(values, 0.99) * 1000),
        "max_ms": float(values.max() * 1000),
        "blocked_100ms": float((values > 0.1).mean()),
        "blocked_1s": float((values > 1.0).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--in-order", type=str2bool, default=False)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--consumer-ms", type=float, default=0.0)
    parser.add_argument("--corrected-cat-meta-sampling", type=str2bool, default=False)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.workers, args.prefetch, args.warmup, args.steps) < 1 or args.consumer_ms < 0:
        parser.error("workers, prefetch, warmup and steps must be positive; consumer-ms must be nonnegative")

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)

    prior = PriorDataset(
        regression=False, batch_size=4, batch_size_per_gp=4,
        min_features=1, max_features=100, max_classes=10,
        max_seq_len=1024, min_train_size=0.3, max_train_size=0.9,
        seq_len_per_gp=True, prior_type="graph_scm", n_jobs=1, device="cpu",
        config=PriorConfig(
            add_gaussian_noise=False, filter_unpredictable_graphs=True,
            filter_unpredictable_datasets=True, allow_act_warping=False,
            min_n_nodes=2, max_n_nodes=32, cauchy_dag_offset=0.0,
            use_corrected_cat_meta_sampling=args.corrected_cat_meta_sampling,
        ),
    )
    loader = DataLoader(
        prior, batch_size=None, num_workers=args.workers,
        prefetch_factor=args.prefetch, persistent_workers=True,
        worker_init_fn=seed_worker, in_order=args.in_order,
    )
    iterator = iter(loader)
    waits = []
    started = time.perf_counter()
    for step in range(args.warmup + args.steps):
        before = time.perf_counter()
        batch = next(iterator)
        waited = time.perf_counter() - before
        if step >= args.warmup:
            waits.append(waited)
        # Metadata tensors have a regular batch dimension even when X/y are nested.
        if len(batch) != 5 or any(t.numel() != 4 for t in batch[2:]):
            raise RuntimeError("prior returned an incomplete local batch")
        if args.consumer_ms:
            time.sleep(args.consumer_ms / 1000)
    wall = time.perf_counter() - started
    local = torch.tensor(waits, dtype=torch.float64)
    if distributed:
        gathered = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)
        all_waits = torch.stack(gathered).numpy()
    else:
        all_waits = local.numpy()[None]
    result = {
        **vars(args), "output": str(args.output) if args.output else None,
        "world_size": world, "workers_per_node": world * args.workers,
        "wall_s": wall, "all_rank_wait": quantiles(all_waits.reshape(-1)),
        "max_rank_wait": quantiles(all_waits.max(axis=0)),
        "per_rank": [quantiles(row) for row in all_waits],
    }
    if rank == 0:
        text = json.dumps(result, indent=2)
        print(text, flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
