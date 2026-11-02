"""Paired full-task audit and ABBA throughput for CPU runtime experiments."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def summarize(path):
    groups = {}
    for file in sorted(path.glob('s*-n*-w*-*.json')):
        group, tag = file.stem.rsplit('-', 1)
        groups.setdefault(group, {})[tag] = json.loads(file.read_text())
    report = {}
    for name, runs in groups.items():
        if set(runs) != {'a1', 'a2', 'b1', 'b2'}:
            continue
        for tag, run in runs.items():
            s = run['summary']
            assert s['tasks'] == len(run['records'])
            assert set(s['forest_fit_sizes']) == {'25'}, (name, tag)
        reproducibility = {}
        for mode in ('a', 'b'):
            first, second = (runs[mode+str(i)]['records'] for i in (1, 2))
            assert len(first) == len(second)
            reproducibility[mode] = all(x['digest'] == y['digest'] and
                x['filter_decisions'] == y['filter_decisions'] for x, y in zip(first, second))
            assert reproducibility[mode], (name, mode)
        old, new = [runs['a1']['records'], runs['b1']['records']]
        paired = {}
        for field in ('width', 'classes', 'entropy', 'minority'):
            x = np.array([b[field]-a[field] for a, b in zip(old, new)], dtype=float)
            rng = np.random.default_rng(617)
            means = x[rng.integers(len(x), size=(2000, len(x)))].mean(1)
            paired[field] = dict(mean_difference=float(x.mean()),
                bootstrap_mean_95ci=list(np.quantile(means, [.025, .975])))
        a = [runs[tag]['summary'] for tag in ('a1', 'a2')]
        b = [runs[tag]['summary'] for tag in ('b1', 'b2')]
        def total(rows, field): return sum(r[field] for r in rows)
        report[name] = dict(runs={tag: run['summary'] for tag, run in runs.items()},
            generation_throughput_gain=total(a, 'generation_s')/total(b, 'generation_s')-1,
            wall_throughput_gain=total(a, 'wall_s')/total(b, 'wall_s')-1,
            cpu_work_reduction=1-total(b, 'cpu_s')/total(a, 'cpu_s'),
            graph_interval_reduction=1-total(b, 'graph_s')/total(a, 'graph_s'),
            filter_interval_reduction=1-total(b, 'filter_s')/total(a, 'filter_s'),
            repeat_reproducibility=reproducibility, tasks=len(old),
            exact_outputs=sum(x['digest']==y['digest'] for x, y in zip(old, new)),
            same_filter_sequences=sum(x['filter_decisions']==y['filter_decisions'] for x, y in zip(old, new)),
            same_width_classes=sum((x['width'],x['classes'])==(y['width'],y['classes']) for x,y in zip(old,new)),
            paired_statistics=paired)
    return report


def summarize_quality(path):
    records = json.loads(path.read_text())['records']
    fields = ('width', 'classes', 'minority', 'normalized_entropy', 'constant_fraction',
              'attempts', 'ridge_balanced_accuracy', 'tree_balanced_accuracy',
              'nodes', 'edges', 'live_nodes', 'depth')
    groups = {mode: {(r['seed'], r['task']): r | r['accepted_graph']
                     for r in records if r['mode'] == mode}
              for mode in ('runtime_old', 'runtime_new')}
    assert groups['runtime_old'].keys() == groups['runtime_new'].keys()
    keys = sorted(groups['runtime_old'])
    assert keys
    result = dict(paired_tasks=len(keys), statistics={}, mixtures={})
    for field in fields:
        old, new = [np.array([groups[mode][key][field] for key in keys], dtype=float)
                    for mode in ('runtime_old', 'runtime_new')]
        delta = new-old
        rng = np.random.default_rng(617)
        means = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(1)
        result['statistics'][field] = dict(old_mean=float(old.mean()), new_mean=float(new.mean()),
            mean_difference=float(delta.mean()), bootstrap_mean_95ci=list(np.quantile(means, [.025, .975])))
    for seed in sorted({key[0] for key in keys}):
        result['mixtures'][str(seed)] = {}
        for mode in ('runtime_old', 'runtime_new'):
            counts = Counter()
            for key, row in groups[mode].items():
                if key[0] == seed:
                    counts.update({k: v for k, v in row['components'].items()
                                   if k.startswith('random_function_fct_type/')})
            result['mixtures'][str(seed)][mode] = dict(counts)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--quality', action='store_true', help='root is an audit_prior_graph_cpu.py runtime JSON')
    a = p.parse_args()
    result = summarize_quality(a.root) if a.quality else summarize(a.root)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result if a.quality else {
        k:{f:v for f,v in r.items() if f not in ('runs','paired_statistics')}
        for k,r in result.items()}), flush=True)


if __name__ == '__main__':
    main()
