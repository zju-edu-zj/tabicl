"""Audit full-table progressive filtering without changing accepted tasks.

The first four trees are reused in the final 25-tree forest. All generation
and fitting sees N rows; no short-table pilot and no weaker SCM are involved.
This script always returns the final decision, even when an early gate fires.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import time
import warnings

import numpy as np
import torch
import threadpoolctl
from scipy.stats import beta
from sklearn.ensemble import ExtraTreesRegressor

from tabicl.prior import _dataset
from tabicl.prior.graph_lib._config import PriorConfig


def improvement(model, target):
    pred = model.oob_prediction_.reshape(target.shape)
    baseline = target.mean(0, keepdims=True)
    imp = ((target-baseline)**2).sum(1)-((target-pred)**2).sum(1)
    return imp


def audit_filter(x, y):
    classes = int(y.max())+1
    target = torch.nn.functional.one_hot(y.long(), classes).float().numpy()
    if classes == 2:
        target = target[:, :1]
    fit_target = target[:, 0] if target.shape[1] == 1 else target
    start = time.perf_counter()
    forest = ExtraTreesRegressor(n_estimators=4, bootstrap=True, oob_score=True,
                                n_jobs=1, random_state=1, max_depth=6, warm_start=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        forest.fit(x.numpy(), fit_target)
    imp = improvement(forest, target)
    z = float(imp.mean()/(imp.std(ddof=1)/np.sqrt(len(imp))+1e-30))
    first_s = time.perf_counter()-start
    start = time.perf_counter()
    forest.set_params(n_estimators=25).fit(x.numpy(), fit_target)
    full_p = float(_dataset._bootstrap_pvalue(improvement(forest, target)))
    tail_s = time.perf_counter()-start
    return dict(z=z, first_s=first_s, tail_s=tail_s, full_p=full_p,
                reject=full_p >= .05, classes=classes, width=int((x.std(0)>0).sum()), rows=len(x))


def initialize():
    torch.set_num_threads(1)
    threadpoolctl.threadpool_limits(1)


def worker(item):
    task, params, seed_base = item
    seed = int(np.random.SeedSequence([seed_base, task]).generate_state(1)[0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    p = _dataset.GraphPrior(batch_size=1, n_jobs=1,
        config=PriorConfig(filter_unpredictable_graphs=True, filter_unpredictable_datasets=True))
    records, old = [], _dataset.should_filter
    def observe(x, y, config, *, is_classif, **kwargs):
        assert is_classif and not config.remove_trivial_datasets
        result = audit_filter(x, y)
        result['task'] = task
        records.append(result)
        return result['reject']
    _dataset.should_filter = observe
    try:
        p.generate_dataset(params, compact=True)
    finally:
        _dataset.should_filter = old
    return records


def upper_error(errors, count):
    return float(beta.ppf(.95, errors+1, count-errors)) if errors < count else 1.


def summarize(records, threshold):
    gated = [r for r in records if np.isfinite(r['z']) and r['z'] >= threshold]
    errors = sum(r['reject'] for r in gated)
    return dict(candidates=len(records), gated=len(gated), false_accepts=errors,
                error_upper_95=upper_error(errors, len(gated)),
                optimistic_filter_saved_s=sum(r['tail_s'] for r in gated),
                measured_filter_s=sum(r['first_s']+r['tail_s'] for r in records))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--tasks', type=int, default=2048)
    a.add_argument('--workers', type=int, default=8)
    a.add_argument('--rows', type=int, default=2048)
    a.add_argument('--seed', type=int, default=48)
    a.add_argument('--fixed-z', type=float, default=None,
                   help='freeze a previously selected threshold; use all tasks only for validation')
    a.add_argument('--output', type=Path, required=True)
    args = a.parse_args()
    if min(args.tasks, args.workers, args.rows) < 1:
        a.error('all sizes must be positive')
    if args.fixed_z is not None and (not np.isfinite(args.fixed_z) or args.fixed_z <= 0):
        a.error('fixed-z must be finite and positive')
    os.environ.setdefault('PYTHONHASHSEED', '42')
    initialize()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    p = _dataset.GraphPrior(batch_size=8, batch_size_per_gp=4, n_jobs=1,
        min_features=1, max_features=100, max_classes=10, max_seq_len=args.rows,
        seq_len_per_gp=True, min_train_size=.3, max_train_size=.9,
        config=PriorConfig(filter_unpredictable_graphs=True, filter_unpredictable_datasets=True))
    params = []
    while len(params) < args.tasks:
        params.extend(p.sample_parameters())
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), initializer=initialize) as ex:
        records = [r for group in ex.map(worker, ((i, param, args.seed)
                     for i, param in enumerate(params[:args.tasks]))) for r in group]
    # Keep parameter-sharing groups together when separating development and
    # validation; adjacent tasks may share a task shape.
    train = [r for r in records if (r['task']//4) % 2 == 0]
    validation = [r for r in records if (r['task']//4) % 2 == 1]
    # Choose on development tasks only. One held-out evaluation follows.
    grid = {str(t): summarize(train, t) for t in (2, 3, 4, 5, 6, 8, 10, 15, 20)}
    eligible = [(float(t), s) for t, s in grid.items() if s['error_upper_95'] <= .005]
    chosen = max(eligible, key=lambda pair: pair[1]['optimistic_filter_saved_s'])[0] if eligible else None
    if args.fixed_z is not None:
        chosen, validation, grid = args.fixed_z, records, {}
    summary = dict(tasks=args.tasks, rows=args.rows, seed=args.seed, wall_s=time.perf_counter()-start,
                   development=grid, chosen_z=chosen,
                   validation=summarize(validation, chosen) if chosen is not None else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, records=records), indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
