"""Separate graph/retry/filter costs; never time array IPC as graph generation.

Run in an idle node with PYTHONHASHSEED=42 OMP_NUM_THREADS=1. The wrappers
delegate unchanged arguments/results and live only inside one worker call.
"""
import argparse
import cProfile
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
import platform
from pathlib import Path
import random
import time
import resource
import sys

import numpy as np
import scipy
import torch

from tabicl.prior import _dataset as source
from tabicl.prior._node_pool import _init_worker
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib import _cpu_budget
from tabicl.prior._scheduling import available_cpu_cores


def profile(item):
    params, task_id, distance_threads, cprofile_path = item
    from tabicl.prior import _node_pool
    from tabicl.prior.graph_lib import _function, _converter
    seed = int(np.random.SeedSequence([42, task_id]).generate_state(1)[0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    original_graph, original_filter = source.GraphSCM, source.should_filter
    original_distance, original_converter_distance = _function.nearest_center, _converter.nearest_center
    original_cdist = torch.cdist
    original_cleanup = source.Prior.delete_unique_features
    original_sanity = source.Prior.cls_sanity_check
    record = dict(task_id=task_id, rows=params["seq_len"], features=params["num_features"],
                  graph_s=0., filter_s=0., attempts=0, filter_calls=0, filter_rejects=0,
                  train_size=params["train_size"], num_classes=params["num_classes"],
                  feature_cap=params["max_features"], seed=seed,
                  feature_cleanup_s=0., class_check_s=0., distances=[], cdist_kernels=[])

    def timed_cdist(x, centers, p=2., compute_mode="use_mm_for_euclid_dist_if_necessary"):
        # Also capture direct users outside nearest_center (e.g. EM assignment).
        # These kernel times are NESTED in graph/nearest-center times.
        frame = sys._getframe(1)
        caller = f"{Path(frame.f_code.co_filename).name}:{frame.f_code.co_name}:{frame.f_lineno}"
        started = time.perf_counter()
        try:
            return original_cdist(x, centers, p=p, compute_mode=compute_mode)
        finally:
            record["cdist_kernels"].append(dict(caller=caller, rows=x.shape[-2], width=x.shape[-1],
                centers=centers.shape[-2], p=float(p), threads=torch.get_num_threads(),
                seconds=time.perf_counter()-started))

    def timed_check(function, key):
        def call(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                record[key] += time.perf_counter()-started
        return staticmethod(call)

    def timed_distance(x, centers, p):
        started = time.perf_counter()
        # Diagnostic only: parallelize the independent non-Euclidean point
        # distances. Keep GEMM (p=2) and the optimized 1-D path unchanged.
        parallel = (distance_threads > 1 and x.device.type == "cpu" and
                    x.shape[1] > 1 and p != 2 and
                    x.numel() * len(centers) >= 10_000_000)
        before = torch.get_num_threads()
        try:
            if parallel:
                torch.set_num_threads(distance_threads)
            return original_distance(x, centers, p)
        finally:
            if parallel:
                torch.set_num_threads(before)
            record["distances"].append(dict(rows=len(x), width=x.shape[1], centers=len(centers),
                                            p=float(p), threads=distance_threads if parallel else before,
                                            seconds=time.perf_counter()-started))

    class TimedGraph:
        def __init__(self, *a, **kw):
            started = time.perf_counter()
            self.graph = original_graph(*a, **kw)
            record["graph_s"] += time.perf_counter()-started

        def __call__(self):
            record["attempts"] += 1
            started = time.perf_counter()
            try:
                return self.graph()
            finally:
                record["graph_s"] += time.perf_counter()-started

    def timed_filter(*a, **kw):
        started = time.perf_counter()
        try:
            rejected = original_filter(*a, **kw)
            record["filter_calls"] += 1
            record["filter_rejects"] += int(rejected)
            return rejected
        finally:
            record["filter_s"] += time.perf_counter()-started

    source.GraphSCM, source.should_filter = TimedGraph, timed_filter
    source.Prior.delete_unique_features = timed_check(original_cleanup, "feature_cleanup_s")
    source.Prior.cls_sanity_check = timed_check(original_sanity, "class_check_s")
    _function.nearest_center = _converter.nearest_center = timed_distance
    torch.cdist = timed_cdist
    started = time.perf_counter()
    cpu_started = time.process_time()
    profiler = cProfile.Profile() if cprofile_path else None
    try:
        with _cpu_budget.generation_slot():
            record["cpu_permit_wait_s"] = time.perf_counter() - started
            if profiler:
                profiler.enable()
            x, y, width = _node_pool._worker_prior.generate_dataset(params, compact=True)
    finally:
        if profiler:
            profiler.disable()
            profiler.dump_stats(cprofile_path)
        source.GraphSCM, source.should_filter = original_graph, original_filter
        source.Prior.delete_unique_features = staticmethod(original_cleanup)
        source.Prior.cls_sanity_check = staticmethod(original_sanity)
        _function.nearest_center, _converter.nearest_center = original_distance, original_converter_distance
        torch.cdist = original_cdist
    record.update(total_s=time.perf_counter()-started, process_cpu_s=time.process_time()-cpu_started,
                  accepted_features=int(width),
                  sha256=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest())
    record["worker_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    if _cpu_budget.worker_budget:
        record.update(_cpu_budget.worker_budget.stats)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--tasks", type=int, default=256)
    parser.add_argument("--start-task", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--distance-threads", type=int, default=1,
                        help="Diagnostic only: threads for large non-Euclidean distance calls")
    parser.add_argument("--pool-kernel-threads", type=int, default=1,
                        help="Exercise production shared-permit distance kernels instead of diagnostic forcing")
    parser.add_argument("--cpu-reserve", type=int, default=1,
                        help="CPU cores reserved from kernel permits; set to local GPU ranks to match training")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-limit", choices=("adaptive", "none"), default="adaptive")
    parser.add_argument("--rows", type=int, help="Optional fixed rows for a controlled shape sweep")
    parser.add_argument("--cprofile-dir", type=Path, help="Optional per-task Python CPU call profiles; adds overhead")
    args = parser.parse_args()
    if (min(args.tasks, args.workers, args.distance_threads, args.pool_kernel_threads) < 1
            or min(args.start_task, args.cpu_reserve) < 0):
        parser.error("positive tasks/workers and nonnegative start-task required")
    if args.distance_threads > 1 and args.pool_kernel_threads > 1:
        parser.error("select diagnostic forcing OR production shared permits")
    os.environ["PYTHONHASHSEED"] = "42"
    torch.set_num_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    prior = source.GraphPrior(batch_size=8, batch_size_per_gp=4 if args.stage == 1 else 1,
              min_features=1, max_features=100, max_classes=10,
              min_seq_len=None if args.stage == 1 else 400,
              max_seq_len={1:2048, 2:10240, 3:60000}[args.stage], log_seq_len=args.stage > 1,
              seq_len_per_gp=True, min_train_size=.3 if args.stage == 1 else .79,
              max_train_size=.9 if args.stage == 1 else .81, n_jobs=1,
              config=PriorConfig(add_gaussian_noise=False, filter_unpredictable_graphs=True,
                  filter_unpredictable_datasets=True, allow_act_warping=False,
                  min_n_nodes=2, max_n_nodes=32, cauchy_dag_offset=0.))
    prior.adaptive_feature_limit = args.feature_limit == "adaptive"
    if args.rows is not None:
        if args.rows < 2:
            parser.error("rows must be >=2")
        prior.min_seq_len, prior.max_seq_len = None, args.rows
    if args.cprofile_dir:
        args.cprofile_dir.mkdir(parents=True, exist_ok=True)
    params = []
    while len(params) < args.start_task + args.tasks:
        params.extend(prior.sample_parameters())
    params = params[args.start_task:args.start_task + args.tasks]
    started = time.perf_counter()
    context = mp.get_context("spawn")
    permit_count = max(1, available_cpu_cores()-args.cpu_reserve)
    permits = context.BoundedSemaphore(permit_count) if args.pool_kernel_threads > 1 else None
    with ProcessPoolExecutor(args.workers, mp_context=context,
                             initializer=_init_worker, initargs=(prior, None, permits, args.pool_kernel_threads)) as executor:
        records = list(executor.map(profile, ((p, i, args.distance_threads,
                       str(args.cprofile_dir / f"task-{i}.prof") if args.cprofile_dir else None) for p, i in
                       zip(params, range(args.start_task, args.start_task+args.tasks)))))
    from scipy.stats import spearmanr
    totals = np.asarray([r["total_s"] for r in records])
    sizes = [r["rows"]*r["features"] for r in records]
    correlation = (float(spearmanr(sizes, totals).statistic)
                   if len(set(sizes)) > 1 and len(set(totals)) > 1 else None)
    summary = dict(wall_s=time.perf_counter()-started, tasks=args.tasks, workers=args.workers,
                   stage=args.stage, start_task=args.start_task, distance_threads=args.distance_threads,
                   pool_kernel_threads=args.pool_kernel_threads,
                   feature_limit=args.feature_limit, fixed_rows=args.rows,
                   graph_share=sum(r["graph_s"] for r in records)/totals.sum(),
                   filter_share=sum(r["filter_s"] for r in records)/totals.sum(),
                   distance_share=sum(c["seconds"] for r in records for c in r["distances"])/totals.sum(),
                   cdist_kernel_share=sum(c["seconds"] for r in records for c in r["cdist_kernels"])/totals.sum(),
                   p50_s=float(np.quantile(totals, .5)), p95_s=float(np.quantile(totals, .95)),
                   p99_s=float(np.quantile(totals, .99)), max_s=float(totals.max()),
                   attempts_mean=float(np.mean([r["attempts"] for r in records])),
                   rows_features_spearman=correlation)
    summary["task_total_s"] = float(totals.sum())
    summary["task_process_cpu_s"] = sum(r["process_cpu_s"] for r in records)
    for phase in ("feature_cleanup_s", "class_check_s", "cpu_permit_wait_s"):
        summary[f"{phase}_share"] = sum(r[phase] for r in records)/totals.sum()
    summary["supply_tasks_per_s_including_startup"] = args.tasks/summary["wall_s"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                       numpy=np.__version__, scipy=scipy.__version__, available_cpu_cores=available_cpu_cores(),
                       cpu_reserve=args.cpu_reserve, cpu_permits=permit_count if permits is not None else None)
    args.output.write_text(json.dumps(dict(summary=summary, environment=environment, records=records), indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
