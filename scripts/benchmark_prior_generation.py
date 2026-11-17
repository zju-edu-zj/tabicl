"""Paired, seeded complete GraphSCM/filter benchmark against original operators.

Unlike the pipeline benchmark, each pair uses exactly the same dataset random
stream, including rejected draws. Both filter decisions and outputs must match.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np
import torch

from benchmark_prior_operators import reference_bootstrap, reference_tree, reference_unique
import tabicl.prior._dataset as dataset
import tabicl.prior.graph_lib._function as function
import tabicl.prior.graph_lib._converter as converter
from tabicl.prior.graph_lib._config import PriorConfig


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def original_distance(x, centers, p):
    return torch.cdist(x, centers, p=p).argmin(dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 10240, 60000])
    parser.add_argument("--datasets", type=int, default=20)
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    records = []
    for rows in args.rows:
        prior = dataset.GraphPrior(
            regression=args.regression, batch_size=4, batch_size_per_gp=4,
            min_features=1, max_features=100, max_classes=10, max_seq_len=rows,
            seq_len_per_gp=True, min_train_size=.3 if rows == 1024 else .79,
            max_train_size=.9 if rows == 1024 else .81, n_jobs=1,
            config=PriorConfig(filter_unpredictable_datasets=True, filter_unpredictable_graphs=True),
        )
        seed(9)
        params = [p for _ in range((args.datasets + 3) // 4) for p in prior.sample_parameters()][:args.datasets]
        timings = {"reference": [], "optimized": []}
        for i, param in enumerate(params):
            results = {}
            # Alternate path order to reduce warm-cache and frequency bias.
            paths = ("reference", "optimized") if i % 2 else ("optimized", "reference")
            for path in paths:
                with ExitStack() as stack:
                    if path == "reference":
                        stack.enter_context(patch.object(dataset.Prior, "delete_unique_features", staticmethod(reference_unique)))
                        stack.enter_context(patch.object(dataset, "_bootstrap_pvalue", reference_bootstrap))
                        stack.enter_context(patch.object(function.RandomTreeFunction, "_transform", reference_tree))
                        stack.enter_context(patch.object(function, "nearest_center", original_distance))
                        stack.enter_context(patch.object(converter, "nearest_center", original_distance))
                    seed(i + 314)
                    start = time.perf_counter()
                    results[path] = prior.generate_dataset(param, compact=True)
                    timings[path].append(time.perf_counter() - start)
            for a, b in zip(results["reference"], results["optimized"]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        record = dict(rows=rows, datasets=args.datasets, regression=args.regression, exact_match=True,
                      reference_total_s=sum(timings["reference"]), optimized_total_s=sum(timings["optimized"]),
                      reference_times_s=timings["reference"], optimized_times_s=timings["optimized"])
        record["speedup"] = record["reference_total_s"] / record["optimized_total_s"]
        records.append(record)
        print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
