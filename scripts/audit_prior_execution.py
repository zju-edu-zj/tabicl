"""Measure a label-first execution upper bound on ORIGINAL complete candidates.

This never rejects a candidate or reorders generation. Per-node instrumentation
adds overhead, so use benchmark_prior_exact.py for throughput. The saved time
is an optimistic bound: label-only-invalid candidates times nodes outside the
label's ancestor closure. It is not a demonstrated speedup.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import threadpoolctl

from tabicl.prior import _dataset
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._graph_function import RandomGraphFunction
from tabicl.prior.graph_lib import _function


def initialize():
    torch.set_num_threads(1)
    threadpoolctl.threadpool_limits(1)


def worker(item):
    index, seed_base, rows, reuse = item
    seed = int(np.random.SeedSequence([seed_base, index]).generate_state(1)[0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    prior = _dataset.GraphPrior(batch_size=1, n_jobs=1, min_features=1,
        max_features=100, max_classes=10, max_seq_len=rows, seq_len_per_gp=True,
        min_train_size=.3, max_train_size=.9,
        config=PriorConfig(filter_unpredictable_graphs=True, filter_unpredictable_datasets=True))
    params = prior.sample_parameters(1)[0]
    original = RandomGraphFunction._transform
    records = []
    function_records, function_originals = [], {}
    if reuse:
        # Counterfactual transforms run AFTER the ordinary full transform;
        # fitted parameters and candidate outputs remain unchanged. These
        # timings include instrumentation/cache effects and are not throughput.
        for name in ('RandomGPFunction', 'RandomTreeFunction', 'RandomDiscretizationFunction',
                     'RandomEMAssignmentFunction', 'RandomLinearFunction', 'RandomQuadraticFunction'):
            cls = getattr(_function, name)
            old = cls._transform
            function_originals[cls] = old
            def measure(self, x, _old=old, _name=name):
                started = time.perf_counter()
                result = _old(self, x)
                full_s = time.perf_counter()-started
                record = dict(kind=_name, rows=len(x), width=x.shape[1], full_s=full_s)
                if len(x) >= 256:
                    started = time.perf_counter()
                    unique, inverse = torch.unique(x, dim=0, return_inverse=True)
                    record.update(unique_rows=len(unique), discover_s=time.perf_counter()-started)
                    started = time.perf_counter()
                    compact = _old(self, unique)[inverse]
                    record['compact_s'] = time.perf_counter()-started
                    record['optimistic_saved_s'] = max(0., full_s-record['compact_s'])
                    del compact
                function_records.append(record)
                return result
            cls._transform = measure

    def observe(graph, n_samples):
        closure = {i for i, specs in enumerate(graph.node_feature_specs)
                   if any(s.group == 'y' for s in specs.values())}
        for i in reversed(range(len(graph.dag))):
            if i in closure:
                closure.update(graph.dag[i])
        timings, originals = {}, {}
        for i, node in enumerate(graph.nodes_):
            old = node._transform
            originals[i] = old
            def timed(*args, _i=i, _old=old, **kwargs):
                start = time.perf_counter()
                result = _old(*args, **kwargs)
                timings[_i] = time.perf_counter()-start
                return result
            node._transform = timed
        try:
            features = original(graph, n_samples)
        finally:
            for i, method in originals.items():
                graph.nodes_[i]._transform = method
        y_names = [name for specs in graph.node_feature_specs for name, s in specs.items() if s.group == 'y']
        y = features[y_names[0]].view(-1)
        _, counts = torch.unique(y, return_counts=True)
        # With fewer than two rows of any class, no permutation can place
        # every class in both train and test; no predictability proxy involved.
        invalid = not bool(torch.isfinite(y).all()) or len(counts) < 2 or int(counts.min()) < 2
        outside = sum(v for i, v in timings.items() if i not in closure)
        records.append(dict(nodes=len(graph.dag), live_nodes=sum(graph.should_compute_),
                            y_ancestor_nodes=len(closure), label_invalid=invalid,
                            node_s=sum(timings.values()), outside_y_s=outside,
                            avoidable_s=outside if invalid else 0.))
        return features

    RandomGraphFunction._transform = observe
    start = time.perf_counter()
    try:
        prior.generate_dataset(params, compact=True)
    finally:
        RandomGraphFunction._transform = original
        for cls, method in function_originals.items():
            cls._transform = method
    return dict(task=index, seconds=time.perf_counter()-start, graphs=records, functions=function_records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=int, default=512)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--rows', type=int, default=2048)
    p.add_argument('--seed', type=int, default=47)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--row-reuse', action='store_true', help='also measure an optimistic known-row reuse bound')
    a = p.parse_args()
    if min(a.tasks, a.workers, a.rows) < 1:
        p.error('all sizes must be positive')
    os.environ.setdefault('PYTHONHASHSEED', '42')
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn'), initializer=initialize) as ex:
        records = list(ex.map(worker, ((i, a.seed, a.rows, a.row_reuse) for i in range(a.tasks))))
    graphs = [g for r in records for g in r['graphs']]
    total = sum(r['seconds'] for r in records)
    summary = dict(tasks=a.tasks, rows=a.rows, seed=a.seed, candidates=len(graphs),
                   label_invalid=sum(g['label_invalid'] for g in graphs), measured_task_s=total,
                   measured_node_s=sum(g['node_s'] for g in graphs),
                   optimistic_avoidable_s=sum(g['avoidable_s'] for g in graphs))
    summary['optimistic_fraction'] = summary['optimistic_avoidable_s']/total
    functions = [f for r in records for f in r['functions']]
    summary['function_reuse'] = {
        name: dict(calls=len(group), full_s=sum(f['full_s'] for f in group),
                   discover_s=sum(f.get('discover_s', 0.) for f in group),
                   optimistic_saved_s=sum(f.get('optimistic_saved_s', 0.) for f in group),
                   compressible_calls=sum(f.get('unique_rows', f['rows']) < f['rows']//2 for f in group))
        for name in sorted({f['kind'] for f in functions})
        for group in [[f for f in functions if f['kind'] == name]]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(summary=summary, records=records), indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
