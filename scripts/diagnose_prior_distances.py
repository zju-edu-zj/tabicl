"""Replay expensive distance inputs captured from complete real prior tasks."""
import argparse
import heapq
import json
import random
import time
import numpy as np
import torch
from benchmark_prior_exact import initialize, generate
from tabicl.prior._dataset import GraphPrior
from tabicl.prior.graph_lib import _function, _converter, _distance
from tabicl.prior.graph_lib._config import PriorConfig

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=int, default=64)
    a = p.parse_args()
    torch.set_num_threads(1)
    random.seed(83); np.random.seed(83); torch.manual_seed(83)
    prior = GraphPrior(batch_size=8, batch_size_per_gp=4, n_jobs=1,
        min_features=1, max_features=100, max_classes=10, max_seq_len=2048,
        seq_len_per_gp=True, min_train_size=.3, max_train_size=.9,
        config=PriorConfig(categorical_sampling='marginal',
            filter_unpredictable_graphs=True, filter_unpredictable_datasets=True))
    params = []
    while len(params) < a.tasks:
        params.extend(prior.sample_parameters())
    initialize(prior, 83)
    captures = []
    serial = 0
    original = _distance.nearest_center
    def capture(x, c, p, **kwargs):
        nonlocal serial
        start = time.perf_counter()
        result = original(x, c, p)
        elapsed = time.perf_counter() - start
        if x.shape[1] > 1 and p != 2 and torch.isfinite(x).all() and torch.isfinite(c).all():
            compact, _ = _distance._compact_rows(x, len(c))
            serial += 1
            item = (elapsed, serial, compact.clone(), c.clone(), p)
            if len(captures) < 24: heapq.heappush(captures, item)
            elif elapsed > captures[0][0]: heapq.heapreplace(captures, item)
        return result
    _function.nearest_center = _converter.nearest_center = capture
    for i, param in enumerate(params[:a.tasks]): generate((i, param))
    for elapsed, _, x, c, p in sorted(captures, reverse=True):
        times = {}
        for name, fn in [('scalar', torch.ops.tabicl_prior_exact.nearest_cpu),
                         ('vector', torch.ops.tabicl_prior_exact.nearest_vector_cpu),
                         ('torch', lambda x,c,p: torch.cdist(x,c,p=p).argmin(-1))]:
            values = []
            for _ in range(3):
                start = time.perf_counter(); fn(x, c, p); values.append(time.perf_counter()-start)
            times[name] = min(values)
        print(json.dumps(dict(shape=list(x.shape), centers=len(c), p=p,
                              captured_s=elapsed, **times)), flush=True)

if __name__ == '__main__': main()
