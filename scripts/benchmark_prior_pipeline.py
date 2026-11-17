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
import hashlib
import json
import multiprocessing as mp
import os
import random
from pathlib import Path
import time
import threading

import numpy as np
import psutil
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from tabicl.prior._dataset import PriorDataset, GraphPrior
from tabicl.prior._genload import seed_worker
from tabicl.prior._node_pool import NodePriorPool, iter_micro_batches
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior._task_plan import TaskPlanner


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


class PayloadPrior(GraphPrior):
    """Cheap fixed-shaped tables isolate transport from graph generation."""
    @staticmethod
    def generate_dataset(params, compact=False):
        rows, width = params["seq_len"], params["num_features"]
        return torch.ones(rows, width), torch.zeros(rows, dtype=torch.long), torch.tensor(width)


class ReplayDataset(IterableDataset):
    """DataLoader comparator with the SAME task IDs/seeds as round_robin pool.

    Compact Tensor IPC avoids conflating transport with whole-local-batch
    padding. This is a diagnostic comparator, not the legacy training loader.
    """
    def __init__(self, prior, rank, world, steps):
        self.prior, self.rank = prior, rank
        self.stop = mp.get_context("spawn").Event()
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        planner = TaskPlanner(prior, world, 1, {}, 42, "round_robin")
        self.plans = [planner.next() for _ in range(steps)]

    def __iter__(self):
        from tabicl.prior._node_pool import _init_worker
        _init_worker(self.prior)
        worker = get_worker_info()
        start, stride = (worker.id, worker.num_workers) if worker else (0, 1)
        n = self.prior.batch_size
        for step in range(start, len(self.plans), stride):
            plan = self.plans[step]
            indices = range(self.rank*n, (self.rank+1)*n)
            results = []
            for i in indices:
                if self.stop.is_set():
                    return
                seed = int(np.random.SeedSequence([42, plan.task_ids[i]]).generate_state(1)[0])
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                results.append(self.prior.generate_dataset(plan.params[i], compact=True))
            x, y, d = zip(*results)
            yield (list(x), list(y), torch.stack(d),
                   torch.tensor([plan.params[i]["seq_len"] for i in indices]),
                   torch.tensor([plan.params[i]["train_size"] for i in indices]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("dataloader", "node_pool"), default="dataloader")
    parser.add_argument("--schedule", choices=("none", "round_robin", "balanced", "window_fifo", "window_bucketed", "shape_cohort", "ready_micro"), default="none")
    parser.add_argument('--ready-max-delay', type=int, default=2)
    parser.add_argument('--categorical-sampling', choices=['legacy', 'marginal'], default='legacy')
    parser.add_argument("--adaptive-feature-limit", type=str2bool, default=True)
    parser.add_argument("--transport", choices=("numpy", "mmap"), default="numpy")
    parser.add_argument("--bucket-steps", type=int, default=4)
    parser.add_argument("--kernel-threads", type=int, default=1)
    parser.add_argument("--payload-only", action="store_true", help="Isolate node-pool data plane, NOT a graph benchmark")
    parser.add_argument("--fixed-task-stream", action="store_true",
                        help="DataLoader compact Tensor replay matching round_robin pool IDs/seeds; requires in-order")
    parser.add_argument("--hash-tasks", action="store_true", help="Audit consumed X/y multiset; adds CPU work")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--regression", type=str2bool, default=False)
    parser.add_argument("--batch-size", type=int, default=4, help="Local batch size")
    parser.add_argument("--pool-workers", type=int, default=8, help="Workers per NODE")
    parser.add_argument("--pool-window", type=int, default=128)
    parser.add_argument("--pool-mb", type=int, default=2048)
    parser.add_argument("--min-seq-len", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start-method", choices=("spawn", "fork"), default="spawn")
    parser.add_argument("--prefetch", type=int, default=8)
    parser.add_argument("--in-order", type=str2bool, default=False)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--consumer-ms", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=0)
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
    random.seed(42 + rank)
    torch.set_num_threads(1)

    prior = PriorDataset(
        regression=args.regression, batch_size=args.batch_size, batch_size_per_gp=4 if args.stage == 1 else 1,
        min_features=1, max_features=100, max_classes=10,
        min_seq_len=(None if args.min_seq_len is not None and args.min_seq_len == args.max_seq_len else
                     args.min_seq_len or (None if args.stage == 1 else 400)),
        max_seq_len=args.max_seq_len or {1: 2048, 2: 10240, 3: 60000}[args.stage],
        log_seq_len=args.stage > 1, min_train_size=0.3 if args.stage == 1 else 0.79,
        max_train_size=0.9 if args.stage == 1 else 0.81,
        seq_len_per_gp=True, prior_type="graph_scm", n_jobs=1, device="cpu",
        config=PriorConfig(
            add_gaussian_noise=False, filter_unpredictable_graphs=True,
            filter_unpredictable_datasets=True, allow_act_warping=False,
            min_n_nodes=2, max_n_nodes=32, cauchy_dag_offset=0.0,
            use_corrected_cat_meta_sampling=args.corrected_cat_meta_sampling,
            categorical_sampling=args.categorical_sampling,
        ),
    )
    prior.prior.adaptive_feature_limit = args.adaptive_feature_limit
    if args.payload_only:
        prior.prior = PayloadPrior(batch_size=args.batch_size,
                                   batch_size_per_gp=4 if args.stage == 1 else 1,
                                   min_features=99, max_features=100,
                                   max_seq_len=args.max_seq_len or 2048,
                                   seq_len_per_gp=True, n_jobs=1, device="cpu")
    if args.backend == "node_pool":
        loader = NodePriorPool(prior.prior, workers=args.pool_workers, window=args.pool_window,
                               byte_budget=args.pool_mb * 1024**2,
                               seed=42 + 100003 * int(os.environ.get("GROUP_RANK", 0)),
                               schedule=args.schedule, micro_size=4 if args.stage == 1 else 1,
                               transport=args.transport, bucket_steps=args.bucket_steps,
                               kernel_threads=args.kernel_threads, shape_seed=42,
                               ready_max_delay=args.ready_max_delay)
    else:
        if args.fixed_task_stream:
            if not args.in_order:
                parser.error("fixed-task-stream requires --in-order True")
            os.environ["PYTHONHASHSEED"] = "42"
            prior = ReplayDataset(prior.prior, rank, world, args.warmup+args.steps+32)
        loader = DataLoader(
            prior, batch_size=None, num_workers=args.workers,
            prefetch_factor=args.prefetch, persistent_workers=True,
            worker_init_fn=seed_worker, in_order=args.in_order,
            multiprocessing_context=args.start_method,
        )
    startup = time.perf_counter()
    peak_rss = [0]
    stop_monitor = threading.Event()

    def monitor_memory():
        root = psutil.Process()
        while not stop_monitor.wait(0.05):
            total = 0
            for process in [root, *root.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak_rss[0] = max(peak_rss[0], total)

    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()
    iterator = iter(loader)
    waits = []
    lengths, features, payloads = [], [], []
    padding_times = []
    hashes = []
    started = None
    startup_s = None
    for step in range(args.warmup + args.steps):
        # Lockstep fetch/consume, like DDP. Independent rank loops followed by
        # end-of-run gathering do NOT measure the real slowest-rank stalls.
        if distributed:
            dist.barrier()
        if step == args.warmup:
            started = time.perf_counter()
            startup_s = started - startup
        before = time.perf_counter()
        batch = next(iterator)
        waited = time.perf_counter() - before
        if step >= args.warmup:
            waits.append(waited)
        # Metadata tensors have a regular batch dimension even when X/y are nested.
        if len(batch) != 5 or any(t.numel() != args.batch_size for t in batch[2:]):
            raise RuntimeError("prior returned an incomplete local batch")
        padding_start = time.perf_counter()
        for micro in iter_micro_batches(batch, 4 if args.stage == 1 else 1):
            if not torch.isfinite(micro[0]).all():
                raise RuntimeError("nonfinite prior output")
        if step >= args.warmup:
            padding_times.append(time.perf_counter() - padding_start)
            if args.hash_tasks:
                hashes.extend(hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()
                              for x, y in zip(*batch[:2]))
            lengths.extend(batch[3].tolist())
            features.extend(batch[2].tolist())
            payloads.append(sum(x.numel() * x.element_size() for x in batch[0]) +
                            sum(y.numel() * y.element_size() for y in batch[1]))
        if distributed:
            dist.barrier()
        if args.consumer_ms:
            time.sleep(args.consumer_ms / 1000)
        if rank == 0 and args.progress_every and (step+1) % args.progress_every == 0:
            print(json.dumps(dict(step=step+1, elapsed_s=time.perf_counter()-startup,
                                  pool=loader.stats() if args.backend == "node_pool" else None)), flush=True)
    if distributed:
        dist.barrier()
    wall = time.perf_counter() - started
    stop_monitor.set()
    monitor.join()
    pool_stats = loader.stats() if args.backend == "node_pool" else None
    if distributed and args.hash_tasks:
        all_hashes = [None] * world
        dist.all_gather_object(all_hashes, hashes)
        hashes = [h for group in all_hashes for h in group]
    local = torch.tensor(waits, dtype=torch.float64)
    if distributed:
        gathered = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)
        all_waits = torch.stack(gathered).numpy()
    else:
        all_waits = local.numpy()[None]
    rss = torch.tensor([peak_rss[0]], dtype=torch.float64)
    if distributed:
        dist.all_reduce(rss, op=dist.ReduceOp.SUM)
    result = {
        **vars(args), "output": str(args.output) if args.output else None,
        "world_size": world, "workers_per_node": args.pool_workers if pool_stats else
        int(os.environ.get("LOCAL_WORLD_SIZE", 1)) * args.workers,
        "wall_s": wall, "startup_and_warmup_s": startup_s,
        "startup_warmup_and_measurement_s": startup_s + wall,
        "sum_rank_peak_rss_mb": rss.item() / 1024**2,
        "datasets_per_s": world * args.batch_size * args.steps / wall,
        "task_multiset_sha256": hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest() if args.hash_tasks else None,
        "pool_stats": pool_stats, "mean_payload_mb": float(np.mean(payloads) / 1024**2),
        "length_mean": float(np.mean(lengths)), "features_mean": float(np.mean(features)),
        "padding_and_validation": quantiles(np.asarray(padding_times)),
        "all_rank_wait": quantiles(all_waits.reshape(-1)),
        "max_rank_wait": quantiles(all_waits.max(axis=0)),
        "per_rank": [quantiles(row) for row in all_waits],
    }
    if rank == 0:
        text = json.dumps(result, indent=2)
        print(text, flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
    if args.backend == "node_pool":
        loader.close()
    else:
        if args.fixed_task_stream:
            # Let in-flight graph kernels finish and drain Tensor IPC before
            # stopping processes. An immediate shutdown can abort vendor Torch
            # native threads when a worker is still generating its next batch.
            prior.stop.set()
            for _ in iterator:
                pass
        iterator._shutdown_workers()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
