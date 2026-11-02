"""Paired accepted-prior quality probes, outside throughput measurements.

Measures table complexity and held-out independent linear/boosted predictors.
Does not substitute for downstream real-data validation of a trained model.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import threadpoolctl
import torch

from benchmark_prior_exact import initialize
from tabicl.prior._dataset import GraphPrior
from tabicl.prior.graph_lib._config import PriorConfig


def probe(item):
    identity, params, seed = item
    from benchmark_prior_exact import prior
    records = []
    for mode in (('sklearn', 'native') if identity % 2 else ('native', 'sklearn')):
        os.environ['TABICL_PRIOR_FOREST'] = mode
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        x, y, d = prior.generate_dataset(params, compact=True)
        x, y = x.numpy(), y.numpy().astype(np.int64)
        train = params['train_size']
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts/len(y)
        record = dict(task_id=identity, mode=mode, width=int(d), classes=len(counts),
            entropy=float(-(probabilities*np.log(probabilities)).sum()), minority=float(probabilities.min()),
            unique_rows=len(np.unique(x, axis=0)),
            digest=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),
            baseline_accuracy=float(np.mean(y[train:] == np.bincount(y[:train]).argmax())))
        for name, model in (
            ('linear', RidgeClassifier(alpha=1.)),
            ('boosted', HistGradientBoostingClassifier(max_iter=40, max_depth=4, max_leaf_nodes=15,
                min_samples_leaf=10, l2_regularization=1., random_state=17))):
            with threadpoolctl.threadpool_limits(1):
                model.fit(x[:train], y[:train])
                prediction = model.predict(x[train:])
            record[name+'_accuracy'] = accuracy_score(y[train:], prediction)
            record[name+'_balanced_accuracy'] = balanced_accuracy_score(y[train:], prediction)
        records.append(record)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--rows', type=int, default=2048)
    p.add_argument('--seed', type=int, default=99)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if min(args.tasks, args.workers, args.rows) < 1:
        p.error('tasks, workers and rows must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    prior = GraphPrior(batch_size=8, batch_size_per_gp=4, n_jobs=1,
        min_features=1, max_features=100, max_classes=10, max_seq_len=args.rows,
        seq_len_per_gp=True, min_train_size=.3, max_train_size=.9,
        config=PriorConfig(categorical_sampling='marginal', graph_execution='optimized',
            filter_execution='sufficient', filter_unpredictable_graphs=True,
            filter_unpredictable_datasets=True))
    params = []
    while len(params) < args.tasks:
        params.extend(prior.sample_parameters())
    jobs = [(i, param, int(np.random.SeedSequence([args.seed, i]).generate_state(1)[0]))
            for i, param in enumerate(params[:args.tasks])]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(prior, args.seed)) as pool:
        records = [r for pair in pool.map(probe, jobs) for r in pair]
    metrics = ['width', 'classes', 'entropy', 'minority', 'unique_rows', 'baseline_accuracy',
               'linear_accuracy', 'linear_balanced_accuracy', 'boosted_accuracy', 'boosted_balanced_accuracy']
    summary = dict(rows=args.rows, seed=args.seed, tasks=args.tasks,
        modes={mode:{metric:float(np.mean([r[metric] for r in records if r['mode']==mode]))
                      for metric in metrics} for mode in ('sklearn', 'native')})
    # Resample intact groups of four: feature/shape parameters are shared
    # within the original prior group. A/B remain paired by task identity.
    rng = np.random.default_rng(1701)
    group_starts = np.arange(0, args.tasks, 4)
    group_lengths = np.minimum(4, args.tasks-group_starts)
    indices = rng.integers(len(group_starts), size=(2000, len(group_starts)))
    paired = {mode:sorted([r for r in records if r['mode']==mode], key=lambda r:r['task_id'])
              for mode in ('sklearn', 'native')}
    summary['paired_difference_ci95'] = {}
    for metric in metrics:
        differences = np.array([b[metric]-a[metric]
            for a, b in zip(paired['sklearn'], paired['native'])])
        group_sums = np.add.reduceat(differences, group_starts)
        means = group_sums[indices].sum(1)/group_lengths[indices].sum(1)
        summary['paired_difference_ci95'][metric] = np.quantile(means, [.025, .975]).tolist()
    summary['bootstrap_unit'] = 'intact four-task prior parameter group; paired by task_id'
    summary['same_table'] = sum(a['digest']==b['digest'] for a,b in zip(paired['sklearn'], paired['native']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, records=records), indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
