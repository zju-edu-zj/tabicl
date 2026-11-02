"""Distribution/complexity audit on complete accepted GraphSCM tasks.

Compare reference seed A vs reference seed B before interpreting optimized.
Holdout tree/ridge probes characterize task difficulty; they cannot certify
the eventual foundation model's generalization. Instrumentation is NOT used
for throughput measurement. Every graph/function/filter remains enabled.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import re

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
import torch
import threadpoolctl

from tabicl.prior._dataset import GraphPrior
from tabicl.prior.graph_lib._base import GlobalSampler
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._graph_function import RandomGraphFunction
from tabicl.prior.graph_lib._converter import CategoricalConverter, CategoricalSoftmaxDiscretizer
from tabicl.prior.graph_lib._points import RandomPoints


def initialize():
    torch.set_num_threads(1)
    global limits
    limits = threadpoolctl.threadpool_limits(1)


def audit(item):
    task, seed_base, rows, execution = item
    mode = 'marginal'
    seed = int(np.random.SeedSequence([seed_base, task]).generate_state(1)[0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    prior = GraphPrior(batch_size=1, n_jobs=1, min_features=1, max_features=100,
        max_classes=10, max_seq_len=rows, seq_len_per_gp=True, min_train_size=.3, max_train_size=.9,
        config=PriorConfig(filter_unpredictable_graphs=True, filter_unpredictable_datasets=True,
                           categorical_sampling=mode, graph_execution=execution))
    params = prior.sample_parameters(1)[0]
    choice_original, graph_original = GlobalSampler.choice, RandomGraphFunction._transform
    counts, graphs = Counter(), []
    converter_original = CategoricalConverter._transform
    softmax_original = CategoricalSoftmaxDiscretizer._fit
    points_original = RandomPoints.sample
    scope = dict(converter_mode=None, embedding=False, dead=False)

    def converter(self, x):
        previous = scope['converter_mode']
        scope['converter_mode'] = self.mode
        try:
            return converter_original(self, x)
        finally:
            scope['converter_mode'] = previous

    def softmax(self, x):
        previous = scope['embedding']
        scope['embedding'] = scope['converter_mode'] in ('softmax_id', 'softmax_int')
        try:
            return softmax_original(self, x)
        finally:
            scope['embedding'] = previous

    def points(self, *args, **kwargs):
        previous = scope['dead']
        scope['dead'] = previous or scope['embedding']
        try:
            return points_original(self, *args, **kwargs)
        finally:
            scope['dead'] = previous

    def choice(self, name, categories, mode='meta'):
        value = choice_original(self, name, categories, mode)
        # functools.partial repr includes a process-specific function address.
        label = re.sub(r' at 0x[0-9a-fA-F]+', '', getattr(value, '__name__', str(value)))
        counts[('dead_embedding/' if scope['dead'] else '') + name + '/' + label] += 1
        return value

    def graph(self, n):
        before = counts.copy()
        result = graph_original(self, n)
        depths = []
        for parents in self.dag:
            depths.append(1 + max((depths[p] for p in parents), default=0))
        graphs.append(dict(nodes=len(self.dag), edges=sum(map(len, self.dag)),
                           live_nodes=sum(self.should_compute_), depth=max(depths),
                           components=dict(counts-before)))
        return result

    GlobalSampler.choice, RandomGraphFunction._transform = choice, graph
    CategoricalConverter._transform, CategoricalSoftmaxDiscretizer._fit = converter, softmax
    RandomPoints.sample = points
    try:
        x, y, d = prior.generate_dataset(params, compact=True)
    finally:
        GlobalSampler.choice, RandomGraphFunction._transform = choice_original, graph_original
        CategoricalConverter._transform, CategoricalSoftmaxDiscretizer._fit = converter_original, softmax_original
        RandomPoints.sample = points_original
    x, y = x.numpy(), y.numpy()
    assert np.isfinite(x).all() and np.isfinite(y).all()
    classes, frequencies = np.unique(y, return_counts=True)
    support = params['train_size']
    assert set(np.unique(y[:support])) == set(classes) == set(np.unique(y[support:]))
    probes = {}
    for name, model in [('ridge', RidgeClassifier()),
                        ('tree', ExtraTreesClassifier(n_estimators=64, min_samples_leaf=2,
                                                      n_jobs=1, random_state=123))]:
        model.fit(x[:support], y[:support])
        probes[name+'_balanced_accuracy'] = balanced_accuracy_score(y[support:], model.predict(x[support:]))
    p = frequencies / len(y)
    return dict(task=task, seed=seed_base, mode=execution, rows=rows, width=int(d),
                classes=len(classes), minority=float(p.min()),
                normalized_entropy=float(-(p*np.log(p)).sum()/np.log(len(p))),
                constant_fraction=float(np.mean(np.ptp(x, axis=0) == 0)),
                support=support, attempts=len(graphs), accepted_graph=graphs[-1],
                all_components=dict(counts), **probes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=int, default=512)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--rows', type=int, default=2048)
    p.add_argument('--seeds', type=int, nargs='+', default=[83, 84])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min(a.tasks, a.workers, a.rows, *a.seeds) < 1:
        p.error('sizes/seeds must be positive')
    os.environ.setdefault('PYTHONHASHSEED', '42')
    records = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn'), initializer=initialize) as ex:
        for seed in a.seeds:
            for mode in ('reference', 'optimized'):
                group = list(ex.map(audit, [(i, seed, a.rows, mode) for i in range(a.tasks)]))
                records.extend(group)
                print(json.dumps(dict(seed=seed, mode=mode, tasks=len(group))), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(arguments=vars(a) | {'output': str(a.output)},
                                         records=records), indent=2))


if __name__ == '__main__':
    main()
