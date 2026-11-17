"""Collect this round's measured results without mixing prior experiments."""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = dict(base_commit='05c11f80e4e4e6ae1cd146c4a0b9e3642eaead60',
        experiment_directory=str(a.root),
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         sklearn=sklearn.__version__, torch=torch.__version__), cpu={})
    for name in ('abba-final', 'abba-seed84'):
        path = a.root/name/'summary.json'
        if path.exists():
            result['cpu'][name] = json.loads(path.read_text())
    for name in ('summary', 'comparisons'):
        path = a.root/'training'/f'{name}.json'
        if path.exists():
            result['training_'+name] = json.loads(path.read_text())
    result['validation'] = {name: (a.root/name).read_text().splitlines()[-18:]
        for name in ('regression.log', 'training-recipe-test.log') if (a.root/name).exists()}
    path = a.root/'standardize-probe.json'
    if path.exists():
        result['std_mean_probe_not_retained'] = json.loads(path.read_text())
    path = a.root/'gini-probe.json'
    if path.exists():
        result['class_count_forest_investigation_only'] = json.loads(path.read_text())
    result['profiles'] = {}
    for rows in (2048, 4096):
        path = a.root/f'profile{rows}-final.json'
        if path.exists():
            profile = json.loads(path.read_text())
            functions = profile['functions']
            result['profiles'][str(rows)] = dict(
                tasks=len(profile['records']),
                self_total_s=sum(f['self_s'] for f in functions),
                top_self=sorted(functions, key=lambda f: f['self_s'], reverse=True)[:20],
                top_cumulative=sorted(functions, key=lambda f: f['cumulative_s'], reverse=True)[:25])
    result['caveats'] = [
        'CPU ABBA includes accepted tasks, retries and process startup; compare both arms within one seed/length.',
        'Worker graph_s/filter_s are interval wall sums; cpu_s is separately measured process CPU time.',
        'cProfile cumulative entries overlap and profiling timings are not throughput timings.',
        'Training uses ready ordering, so finite-window shapes and timing remain variable.',
        'Shared compiler caches preclude controlled launch-wall claims.',
        'Equality on recorded tasks supports this execution-only change, not overall prior optimality or downstream convergence.'
    ]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(cpu={name: {n: r['throughput_gain'] for n, r in runs.items()}
        for name, runs in result['cpu'].items()}, training=result.get('training_comparisons', {}))))


if __name__ == '__main__':
    main()
