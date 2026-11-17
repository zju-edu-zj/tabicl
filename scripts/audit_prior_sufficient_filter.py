"""Paired complete-filter audit on actual prior candidates, before retry selection.

Both forests see the SAME candidate. Use MSE decisions to advance the generator;
separate complete-task ABBA runs measure each mode's own retry/output distribution.
This instrumented audit must not be interpreted as a throughput measurement.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
import random

import numpy as np
import torch

import benchmark_prior_exact as bench
from tabicl.prior import _dataset as ds
from tabicl.prior._filter_forest import planned_oob_prediction, planned_classification_oob_prediction
from tabicl.prior.graph_lib._config import PriorConfig


def audit(item):
    candidates = []
    original = ds.should_filter
    def paired(x, y, config, *, is_classif, n_estimators=25, n_boot=200):
        if not is_classif or int(y.max()) < 2:
            return original(x, y, config, is_classif=is_classif, n_estimators=n_estimators, n_boot=n_boot)
        labels = y.long().numpy()
        c = int(labels.max())+1
        targets = np.eye(c, dtype=np.float32)[labels]
        xp = x.numpy()
        ref = planned_oob_prediction(xp, targets, n_estimators)
        new = planned_classification_oob_prediction(xp, labels, c, n_estimators)
        assert ref is not None and new is not None
        base_error = ((targets-targets.mean(0))**2).sum(1)
        old_error = ((targets-ref)**2).sum(1)
        old = (base_error-old_error, base_error.mean(), old_error.mean())
        new_stats = ds._classification_filter_statistics(labels, new)
        pvalues = [ds._bootstrap_pvalue(s[0], n_boot) for s in (old, new_stats)]
        decisions = [ds._filter_from_statistics(*s, config, n_boot) for s in (old, new_stats)]
        trivial = replace(config, remove_trivial_datasets=True)
        trivial_decisions = [ds._filter_from_statistics(*s, trivial, n_boot) for s in (old, new_stats)]
        effective = np.ptp(xp, axis=0) > 0
        cols = xp[:, effective]
        unique_fraction = np.mean([len(np.unique(col))/len(col) for col in cols.T]) if cols.shape[1] else 0.
        candidates.append(dict(classes=c, width=int(effective.sum()), padded_width=x.shape[1],
            unique_fraction=float(unique_fraction), baseline_mse=float(old[1]),
            mse_old=float(old[2]), mse_new=float(new_stats[2]), p_old=pvalues[0], p_new=pvalues[1],
            reject_old=decisions[0], reject_new=decisions[1],
            trivial_old=trivial_decisions[0], trivial_new=trivial_decisions[1],
            prediction_rmse=float(np.sqrt(np.mean((ref-new)**2)))))
        return decisions[0]
    ds.should_filter = paired
    try:
        result = bench.generate(item)
    finally:
        ds.should_filter = original
    return dict(task=result['task'], final_width=result['width'], final_classes=result['classes'], candidates=candidates)


def summarize(records):
    all_candidates = [c for r in records for c in r['candidates']]
    def group(cs):
        if not cs:
            return dict(n=0)
        return dict(n=len(cs), old_rejected=sum(c['reject_old'] for c in cs),
            new_rejected=sum(c['reject_new'] for c in cs),
            flips=sum(c['reject_old'] != c['reject_new'] for c in cs),
            old_reject_new_accept=sum(c['reject_old'] and not c['reject_new'] for c in cs),
            old_accept_new_reject=sum(not c['reject_old'] and c['reject_new'] for c in cs),
            trivial_flips=sum(c['trivial_old'] != c['trivial_new'] for c in cs),
            mean_p_delta=float(np.mean([c['p_new']-c['p_old'] for c in cs])),
            mean_error_delta=float(np.mean([c['mse_new']-c['mse_old'] for c in cs])),
            mean_prediction_rmse=float(np.mean([c['prediction_rmse'] for c in cs])))
    groups = {'all': group(all_candidates)}
    for c in range(3, 11):
        groups[f'C{c}'] = group([v for v in all_candidates if v['classes'] == c])
    for name, predicate in [
        ('width_1_8', lambda c: c['width'] <= 8), ('width_9_32', lambda c: 8 < c['width'] <= 32),
        ('width_33_100', lambda c: c['width'] > 32),
        ('discrete', lambda c: c['unique_fraction'] < .1),
        ('continuous', lambda c: c['unique_fraction'] >= .1),
        ('boundary', lambda c: .02 <= c['p_old'] <= .08),
        ('strong', lambda c: c['p_old'] == 0.), ('weak', lambda c: c['p_old'] > .08)]:
        groups[name] = group([v for v in all_candidates if predicate(v)])
    # Resample requests, preserving within-request retry dependence.
    counts = np.array([len(r['candidates']) for r in records])
    deltas = np.array([sum(int(c['reject_new'])-int(c['reject_old']) for c in r['candidates']) for r in records])
    rng = np.random.default_rng(123)
    indices = rng.integers(0, len(records), size=(2000, len(records)))
    totals = counts[indices].sum(1)
    valid = totals > 0
    ci = np.quantile(deltas[indices].sum(1)[valid]/totals[valid], [.025, .975]) if valid.any() else [0., 0.]
    return dict(tasks=len(records), groups=groups, rejection_delta_task_bootstrap_ci95=list(ci))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=int, default=512)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--rows', type=int, default=2048)
    p.add_argument('--seed', type=int, default=93)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    prior = ds.GraphPrior(batch_size=8, batch_size_per_gp=4, n_jobs=1,
        min_features=1, max_features=100, max_classes=10, max_seq_len=a.rows,
        seq_len_per_gp=True, min_train_size=.3, max_train_size=.9,
        config=PriorConfig(filter_unpredictable_graphs=True, filter_unpredictable_datasets=True,
            categorical_sampling='marginal', graph_execution='optimized', filter_execution='planned'))
    params = []
    while len(params) < a.tasks:
        params.extend(prior.sample_parameters())
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn'),
            initializer=bench.initialize, initargs=(prior, a.seed)) as executor:
        records = list(executor.map(audit, enumerate(params[:a.tasks])))
    summary = summarize(records)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(rows=a.rows, seed=a.seed, summary=summary, records=records), indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
