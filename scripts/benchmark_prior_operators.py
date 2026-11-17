"""Isolated CPU latency/RSS and exact-equivalence checks at all curriculum sizes.

  python scripts/benchmark_prior_operators.py --output prior_operators.json

Each path runs in a fresh spawned process, so max RSS is comparable. RSS includes
the Python/PyTorch runtime. Reference functions reproduce the previous operators.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
import psutil

from tabicl.prior._dataset import Prior, _bootstrap_pvalue
from tabicl.prior.graph_lib._function import RandomTreeFunction
from tabicl.prior.graph_lib._distance import nearest_center


def reference_unique(x, ds):
    result, counts = [], []
    for xi, d in zip(x, ds):
        mask = [len(torch.unique(xi[:, j])) > 1 for j in range(d)]
        count = sum(mask)
        result.append(F.pad(xi[:, :d][:, mask], (0, xi.shape[-1] - count)))
        counts.append(count)
    return torch.stack(result), torch.tensor(counts)


def reference_bootstrap(imp, n_boot=200):
    idx = np.random.default_rng(0).integers(0, len(imp), size=(n_boot, len(imp)))
    return float(np.mean(imp[idx].mean(axis=1) <= 0.0))


def reference_tree(tree, x):
    sides = (x[:, tree.split_dims_] > tree.split_points_).reshape(len(x), tree.n_trees_, tree.depth_)
    idx = torch.einsum("btd,d->bt", sides.float(), tree.idx_multipliers_.float()).long()
    trees = torch.arange(tree.n_trees_).expand(len(x), tree.n_trees_)
    return tree.leaf_values_[trees, idx].mean(dim=1)


def inputs(op, rows):
    torch.manual_seed(42)
    if op == "unique":
        x = torch.randn(1, rows, 100)
        x[..., ::5] = 0
        return lambda: reference_unique(x, torch.tensor([100])), lambda: Prior.delete_unique_features(x, torch.tensor([100]))
    if op == "bootstrap":
        imp = np.random.default_rng(42).normal(0.003, 1, rows)
        return lambda: reference_bootstrap(imp), lambda: _bootstrap_pvalue(imp)
    if op.startswith("distance_"):
        width = int(op.removeprefix("distance_"))
        x, centers, p = torch.randn(rows, width), torch.randn(128, width), 1.37
        return lambda: torch.cdist(x, centers, p=p).argmin(dim=-1), lambda: nearest_center(x, centers, p)
    x = torch.randn(rows, 32)
    tree = RandomTreeFunction.__new__(RandomTreeFunction)
    tree.n_trees_, tree.depth_, tree.out_features = 128, 7, 32
    tree.split_dims_ = torch.randint(32, (128 * 7,))
    tree.split_points_ = torch.randn(128 * 7)
    tree.idx_multipliers_ = 2 ** torch.arange(7)
    tree.leaf_values_ = torch.randn(128, 128, 32)
    return lambda: reference_tree(tree, x), lambda: tree._transform(x)


def worker(queue, op, rows, optimized, repeats):
    try:
        torch.set_num_threads(1)
        reference, actual = inputs(op, rows)
        fn = actual if optimized else reference
        process = psutil.Process()
        baseline_rss = process.memory_info().rss
        samples = [baseline_rss]
        stopped = threading.Event()

        def sample_rss():
            while not stopped.wait(0.001):
                samples.append(process.memory_info().rss)

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        fn()
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            timings.append(time.perf_counter() - start)
        samples.append(process.memory_info().rss)
        stopped.set()
        sampler.join()
        queue.put(dict(median_ms=float(np.median(timings) * 1000), times_ms=[v * 1000 for v in timings],
                       baseline_rss_mb=baseline_rss / 1024**2, peak_rss_mb=max(samples) / 1024**2,
                       incremental_peak_mb=(max(samples) - baseline_rss) / 1024**2))
    except BaseException:
        import traceback
        queue.put(dict(error=traceback.format_exc()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 10240, 60000])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    ctx = mp.get_context("spawn")
    for op in ("unique", "bootstrap", "tree", "distance_1", "distance_8", "distance_32"):
        for rows in args.rows:
            record = dict(operator=op, rows=rows)
            # Equality is checked outside timed children to avoid polluting RSS.
            ref, optimized = inputs(op, rows)
            expected, actual = ref(), optimized()
            if op.startswith("distance_"):
                width = int(op.removeprefix("distance_"))
                torch.manual_seed(42)
                x, centers = torch.randn(rows, width), torch.randn(128, width)
                distances = torch.cdist(x, centers, p=1.37)
                expected_distance = distances.gather(1, expected[:, None]).squeeze(1)
                actual_distance = distances.gather(1, actual[:, None]).squeeze(1)
                excess = (actual_distance - expected_distance).clamp_min(0) / expected_distance.clamp_min(1e-12)
                record["argmin_match_rate"] = float((actual == expected).float().mean())
                record["max_relative_distance_excess"] = float(excess.max())
                record["within_tolerance"] = bool((excess <= 1e-5).all())
                if not record["within_tolerance"]:
                    raise AssertionError(f"nearest-center distance error exceeds tolerance: {record}")
            else:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                record["exact_match"] = True
            del ref, optimized
            for name, is_opt in (("reference", False), ("optimized", True)):
                queue = ctx.Queue()
                process = ctx.Process(target=worker, args=(queue, op, rows, is_opt, args.repeats))
                process.start()
                record[name] = queue.get(timeout=300)
                process.join(timeout=30)
                if process.exitcode != 0 or "error" in record[name]:
                    raise RuntimeError(record[name])
            record["speedup"] = record["reference"]["median_ms"] / record["optimized"]["median_ms"]
            results.append(record)
            print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
