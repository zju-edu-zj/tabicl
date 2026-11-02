"""Fixed-task complete prior benchmark, including rejected attempts and filtering.

Run ablations in fresh processes, sequentially (e.g. ABBA), with identical
--seed and --rows. Environment switches are recorded, not changed by workers.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import resource
import sys
import time

import numpy as np
import torch
import threadpoolctl

from tabicl.prior import _dataset
from tabicl.prior.graph_lib._config import PriorConfig


def initialize(prototype, seed, disable=(), ready=None):
    global prior, limits, task_seed
    prior, task_seed = prototype, seed
    torch.set_num_threads(1)
    limits = threadpoolctl.threadpool_limits(1)
    # Benchmark-only ablations; production has a single execution switch.
    from dataclasses import replace
    from tabicl.prior.graph_lib import _function, _converter
    # Distance ablation must also bypass EM's fused distance/posterior kernel.
    if 'em' in disable or 'distance' in disable:
        def unfused_em(self, x):
            distances = _function.compact_cdist(x, self.x_ind_, p=self.p_, vectorized=True)
            probs = torch.softmax(self.consts - (distances / self.stds_[None, :]).clamp_min(0.) ** self.q_, -1)
            return self.out_func_(probs)
        _function.RandomEMAssignmentFunction._transform = unfused_em
    if 'distance' in disable:
        for module in (_function, _converter):
            for name in ('nearest_center', 'compact_cdist'):
                if hasattr(module, name):
                    original = getattr(module, name)
                    def reference(*args, _original=original, **kwargs):
                        kwargs['vectorized'] = False
                        return _original(*args, **kwargs)
                    setattr(module, name, reference)
    if 'quadratic' in disable:
        original_init = _function.RandomQuadraticFunction.__init__
        def init_quadratic(self, *args, _original=original_init, **kwargs):
            _original(self, *args, **kwargs)
            self.config = replace(self.config, graph_execution='reference')
        _function.RandomQuadraticFunction.__init__ = init_quadratic
    if 'embedding' in disable:
        for cls in (_converter.CategoricalSoftmaxDiscretizer, _converter.CategoricalNeighborDiscretizer):
            original_init = cls.__init__
            def init_disc(self, *args, _original=original_init, **kwargs):
                kwargs['return_values'] = True
                _original(self, *args, **kwargs)
            cls.__init__ = init_disc
    if ready is not None:
        from tabicl.prior.graph_lib._distance import _native_available
        _native_available()  # load the configured library before steady timing
        ready.wait(timeout=180)


def worker_ready():
    return os.getpid()


def generate(item):
    task_id, params = item
    if prior.config != params['config']:
        raise ValueError('worker prior configuration must match the planned task')
    seed = int(np.random.SeedSequence([task_seed, task_id]).generate_state(1)[0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    attempts, times = [], dict(graph_s=0., filter_s=0., graph_calls=0)
    original_filter = _dataset.should_filter
    original_scm = _dataset.GraphSCM.__call__
    original_fit, forest_sizes = _dataset.ExtraTreesRegressor.fit, []
    original_planned = _dataset.planned_oob_prediction
    original_classification = _dataset.planned_classification_oob_prediction

    def audit_classification(x, labels, classes, n_estimators=25):
        result = original_classification(x, labels, classes, n_estimators)
        if result is not None:
            forest_sizes.append(n_estimators)
        return result

    def audit_planned(x, y, n_estimators=25):
        result = original_planned(x, y, n_estimators)
        if result is not None:
            forest_sizes.append(n_estimators)
        return result

    def audit_fit(model, *args, **kwargs):
        forest_sizes.append(model.n_estimators)
        return original_fit(model, *args, **kwargs)

    def audit_filter(*args, **kwargs):
        started = time.perf_counter()
        result = original_filter(*args, **kwargs)
        times['filter_s'] += time.perf_counter()-started
        attempts.append(bool(result))
        return result

    def audit_scm(*args, **kwargs):
        started = time.perf_counter()
        result = original_scm(*args, **kwargs)
        times['graph_s'] += time.perf_counter()-started
        times['graph_calls'] += 1
        return result

    _dataset.should_filter = audit_filter
    _dataset.GraphSCM.__call__ = audit_scm
    _dataset.ExtraTreesRegressor.fit = audit_fit
    _dataset.planned_oob_prediction = audit_planned
    _dataset.planned_classification_oob_prediction = audit_classification
    started, cpu = time.perf_counter(), time.process_time()
    try:
        x, y, d = prior.generate_dataset(params, compact=True)
    finally:
        _dataset.should_filter = original_filter
        _dataset.GraphSCM.__call__ = original_scm
        _dataset.ExtraTreesRegressor.fit = original_fit
        _dataset.planned_oob_prediction = original_planned
        _dataset.planned_classification_oob_prediction = original_classification
    elapsed, cpu = time.perf_counter()-started, time.process_time()-cpu
    _, counts = torch.unique(y, return_counts=True)
    probs = counts.double()/len(y)
    return dict(task=task_id, seconds=elapsed, cpu_s=cpu, **times,
                digest=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest(),
                filter_decisions=attempts, width=int(d),
                forest_fit_sizes=dict(Counter(forest_sizes)),
                entropy=float(-(probs*probs.log()).sum()), minority=float(probs.min()),
                classes=len(counts),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*
                               (1 if sys.platform == 'darwin' else 1024))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', type=int, default=512)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--rows', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--filter-mode', choices=['full', 'progressive'], default='full')
    parser.add_argument('--categorical-sampling', choices=['legacy', 'marginal'], default='legacy')
    parser.add_argument('--graph-execution', choices=['reference', 'optimized'], default='reference')
    parser.add_argument('--filter-execution', choices=['reference', 'planned', 'sufficient'], default='reference')
    parser.add_argument('--disable', nargs='*', choices=['distance', 'quadratic', 'embedding', 'em'], default=[])
    parser.add_argument('--warm-workers', action='store_true',
                        help='Also report generation time after every worker has initialized')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.tasks, args.workers, args.rows) < 1:
        parser.error('tasks, workers and rows must be positive')
    os.environ.setdefault('PYTHONHASHSEED', '42')
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    p = _dataset.GraphPrior(batch_size=8, batch_size_per_gp=4, n_jobs=1,
         min_features=1, max_features=100, max_classes=10, max_seq_len=args.rows,
         seq_len_per_gp=True, min_train_size=.3, max_train_size=.9,
         config=PriorConfig(filter_unpredictable_graphs=True, filter_unpredictable_datasets=True,
                            dataset_filter_mode=args.filter_mode, categorical_sampling=args.categorical_sampling,
                            graph_execution=args.graph_execution, filter_execution=args.filter_execution))
    params = []
    while len(params) < args.tasks:
        params.extend(p.sample_parameters())
    context = mp.get_context('spawn')
    ready = context.Barrier(args.workers + 1) if args.warm_workers else None
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers, mp_context=context,
                   initializer=initialize, initargs=(p, args.seed, args.disable, ready)) as executor:
        if ready is not None:
            futures = [executor.submit(worker_ready) for _ in range(args.workers)]
            ready.wait(timeout=180)
            for future in futures:
                future.result()
        generation_started = time.perf_counter()
        records = list(executor.map(generate, enumerate(params[:args.tasks])))
        generation_s = time.perf_counter() - generation_started
    total = time.perf_counter()-started
    summary = dict(wall_s=total, cpu_s=sum(r['cpu_s'] for r in records), tasks=args.tasks,
                   workers=args.workers, rows=args.rows, seed=args.seed,
                   filter_mode=args.filter_mode,
                   categorical_sampling=args.categorical_sampling,
                   graph_execution=args.graph_execution,
                   filter_execution=args.filter_execution,
                   disabled=args.disable,
                   warm_workers=args.warm_workers, generation_s=generation_s,
                   generation_tables_per_s=args.tasks/generation_s,
                   tree=os.environ.get('TABICL_PRIOR_TREE', 'native'),
                   one_dimensional_mode=os.environ.get('TABICL_PRIOR_1D_MODE', 'legacy'),
                   native_library=os.environ.get('TABICL_PRIOR_DISTANCE_LIBRARY', ''),
                   compaction=os.environ.get('TABICL_PRIOR_COMPACT', 'hash'),
                   statistics=os.environ.get('TABICL_PRIOR_STATISTICS', 'native'),
                   forest=os.environ.get('TABICL_PRIOR_FOREST', 'sklearn'),
                   python_hash_seed=os.environ['PYTHONHASHSEED'],
                   rejected_by_filter=sum(sum(r['filter_decisions']) for r in records),
                   filter_calls=sum(len(r['filter_decisions']) for r in records),
                   forest_fit_sizes=dict(sum((Counter(r['forest_fit_sizes']) for r in records), Counter())),
                   graph_calls=sum(r['graph_calls'] for r in records),
                   graph_s=sum(r['graph_s'] for r in records), filter_s=sum(r['filter_s'] for r in records),
                   p50_s=float(np.median([r['seconds'] for r in records])),
                   p95_s=float(np.quantile([r['seconds'] for r in records], .95)),
                   p99_s=float(np.quantile([r['seconds'] for r in records], .99)),
                   tables_per_s=args.tasks/total)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, records=records), indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
