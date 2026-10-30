"""Metadata-only scheduling headroom: predicted operations, NEVER device latency.

Compare the same complete windows at a fixed global batch for 8/16 ranks.
Requested widths overestimate some accepted widths; CPU generation tails and
communication/compile/backend costs are deliberately outside this audit.
"""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from tabicl.prior._dataset import GraphPrior
from tabicl.prior._task_plan import TaskPlanner, bucket_window
from tabicl.prior.graph_lib._config import PriorConfig


def audit(stage, ranks, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    micro = 4 if stage == 1 else 1
    local = args.global_batch // ranks
    if args.global_batch % ranks or local % micro:
        raise ValueError("global batch must divide ranks and complete micro-batches")
    prior = GraphPrior(batch_size=local, batch_size_per_gp=micro,
                       min_features=1, max_features=100, max_classes=10,
                       min_seq_len=None if stage == 1 else 400,
                       max_seq_len={1: 2048, 2: 10240, 3: 60000}[stage],
                       log_seq_len=stage > 1, seq_len_per_gp=True,
                       min_train_size=.3 if stage == 1 else .79,
                       max_train_size=.9 if stage == 1 else .81, n_jobs=1,
                       config=PriorConfig(add_gaussian_noise=False,
                           filter_unpredictable_graphs=True, filter_unpredictable_datasets=True,
                           allow_act_warping=False, min_n_nodes=2, max_n_nodes=32,
                           cauchy_dag_offset=0.))
    planner = TaskPlanner(prior, ranks, micro, {}, args.seed, "window_fifo", args.bucket_steps)
    records = []
    for _ in range(args.windows):
        params = [p for _ in range(args.bucket_steps) for p in planner.next().params]
        _, stats = bucket_window(params, micro, local, ranks, {})
        records.append(stats)
    before = sum(r["window_predicted_before"] for r in records)
    after = sum(r["window_predicted_after"] for r in records)
    return dict(stage=stage, ranks=ranks, global_batch=args.global_batch,
                micro=micro, micros_per_rank=local//micro,
                windows=args.windows, bucket_steps=args.bucket_steps,
                predicted_gain=before/max(after, 1.),
                window_gain_p50=float(np.median([r["window_predicted_gain"] for r in records])),
                window_gain_max=max(r["window_predicted_gain"] for r in records),
                seed=args.seed, requested_features=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--stages", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument("--global-batch", type=int, default=64)
    parser.add_argument("--bucket-steps", type=int, default=4)
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.ranks, args.global_batch, args.bucket_steps, args.windows) < 1:
        parser.error("sizes must be positive")
    torch.set_num_threads(1)
    results = [audit(stage, ranks, args) for stage in args.stages for ranks in args.ranks]
    args.output.write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
