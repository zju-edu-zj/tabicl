"""Paired replay of real filter candidates.

Captures on the existing sklearn/rows path, then alternates replay order.
This is a single-worker screening experiment, not a training benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from benchmark_prior_exact import initialize, generate
from tabicl.prior import _filter_forest as forest
from tabicl.prior._dataset import GraphPrior, _bootstrap_pvalue, _classification_filter_statistics
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._distance import _native_available


def statistics(y, prediction, classes):
    if classes:
        improvement, baseline, model = _classification_filter_statistics(y.astype(np.int64), prediction)
    else:
        target = y.reshape(len(y), -1)
        baseline_errors = ((target-target.mean(0, keepdims=True))**2).sum(1)
        model_errors = ((target-prediction)**2).sum(1)
        improvement, baseline, model = baseline_errors-model_errors, baseline_errors.mean(), model_errors.mean()
    pvalue = _bootstrap_pvalue(improvement)
    return dict(pvalue=pvalue, reject=bool(pvalue >= .05),
                baseline=float(baseline), model_error=float(model))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, required=True)
    parser.add_argument('--candidates', type=int, default=128)
    parser.add_argument('--seed', type=int, default=99)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not _native_available():
        raise RuntimeError('Set TABICL_PRIOR_DISTANCE_LIBRARY to the current runtime')
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['TABICL_PRIOR_FOREST'] = 'sklearn'
    prior = GraphPrior(batch_size=8, batch_size_per_gp=4, n_jobs=1,
        min_features=1, max_features=100, max_classes=10, max_seq_len=args.rows,
        seq_len_per_gp=True, min_train_size=.3, max_train_size=.9,
        config=PriorConfig(categorical_sampling='marginal', graph_execution='optimized',
            filter_execution='sufficient', filter_unpredictable_graphs=True,
            filter_unpredictable_datasets=True))
    params = []
    while len(params) < args.candidates*3:
        params.extend(prior.sample_parameters())
    initialize(prior, args.seed)
    original = forest._planned_prediction
    candidates, tasks = [], []

    def capture(x, y, n_estimators, classes=None):
        prediction = original(x, y, n_estimators, classes)
        if prediction is not None and len(candidates) < args.candidates:
            candidates.append((x.copy(), y.copy(), n_estimators, classes, prediction.copy()))
        return prediction

    forest._planned_prediction = capture
    try:
        for task_id, param in enumerate(params):
            tasks.append(generate((task_id, param)))
            if len(candidates) >= args.candidates:
                break
    finally:
        forest._planned_prediction = original
    if len(candidates) != args.candidates:
        raise RuntimeError('Insufficient candidates')
    forest_records = []
    for repetition in range(args.repetitions):
        for index, (x, y, trees, classes, reference) in enumerate(candidates):
            modes = ('sklearn', 'native') if (index+repetition)%2 == 0 else ('native', 'sklearn')
            for mode in modes:
                os.environ['TABICL_PRIOR_FOREST'] = mode
                started, cpu = time.perf_counter(), time.process_time()
                prediction = original(x, y, trees, classes)
                elapsed, cpu_s = time.perf_counter()-started, time.process_time()-cpu
                record = dict(repetition=repetition, candidate=index, mode=mode,
                    seconds=elapsed, cpu_s=cpu_s, classes=classes, trees=trees,
                    width=x.shape[1], active_width=int(np.count_nonzero(np.ptp(x, axis=0))))
                if repetition == 0:
                    record.update(reference=statistics(y, reference, classes),
                        replay=statistics(y, prediction, classes),
                        prediction_rmse=float(np.sqrt(np.mean((reference-prediction)**2))))
                forest_records.append(record)
    summary = dict(rows=args.rows, seed=args.seed, candidates=len(candidates),
        repetitions=args.repetitions,
        forest_cpu_s={mode:sum(r['cpu_s'] for r in forest_records if r['mode']==mode)
                      for mode in ('sklearn', 'native')},
        forest_decision_flips=sum(r['reference']['reject'] != r['replay']['reject']
            for r in forest_records if r['repetition']==0 and r['mode']=='native'),
        note='Paired real-candidate single-worker replay; not full-prior or training throughput.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, forest=forest_records,
        capture_tasks=tasks), indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
