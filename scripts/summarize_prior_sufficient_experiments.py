"""Summarize independent ABBA runs, paired candidates, and installed-wheel checks."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def abba(directory):
    groups = defaultdict(dict)
    for path in sorted(directory.glob('*.json')):
        prefix, tag = path.stem.rsplit('-', 1)
        groups[prefix][tag] = json.loads(path.read_text())
    output = []
    for name, runs in groups.items():
        assert set(runs) == {'a1', 'a2', 'b1', 'b2'}, (name, list(runs))
        for run in runs.values():
            s, records = run['summary'], run['records']
            assert len(records) == s['tasks']
            assert {r['task'] for r in records} == set(range(s['tasks']))
            assert s['forest_fit_sizes'] == {'25': s['filter_calls']}, s
        def records(tag):
            return sorted(runs[tag]['records'], key=lambda r: r['task'])
        for first, second in [('a1', 'a2'), ('b1', 'b2')]:
            assert [(r['digest'], r['filter_decisions']) for r in records(first)] == [
                (r['digest'], r['filter_decisions']) for r in records(second)], (name, first, second)
        old, new = records('a1'), records('b1')
        item = dict(name=name, tasks=len(old), rows=runs['a1']['summary']['rows'],
                    workers=runs['a1']['summary']['workers'], measurements={}, gains={})
        for key in ['wall_s', 'generation_s', 'cpu_s', 'filter_s', 'graph_s', 'p95_s', 'p99_s']:
            if key not in runs['a1']['summary']:
                continue
            a = [runs[t]['summary'][key] for t in ('a1', 'a2')]
            b = [runs[t]['summary'][key] for t in ('b1', 'b2')]
            item['measurements'][key] = dict(a=a, b=b)
            item['gains'][key] = dict(speedup=float(np.mean(a)/np.mean(b)-1),
                                     reduction=float(1-np.mean(b)/np.mean(a)))
        item['paired_tasks'] = dict(
            equal_digest=sum(a['digest'] == b['digest'] for a, b in zip(old, new)),
            equal_filter_decisions=sum(a['filter_decisions'] == b['filter_decisions'] for a, b in zip(old, new)),
            equal_width_classes=sum((a['width'], a['classes']) == (b['width'], b['classes']) for a, b in zip(old, new)))
        item['paired_distribution'] = {}
        for key in ('entropy', 'minority', 'width', 'classes'):
            a = np.asarray([r[key] for r in old])
            b = np.asarray([r[key] for r in new])
            delta = b-a
            rng = np.random.default_rng(42)
            samples = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(1)
            item['paired_distribution'][key] = dict(old_mean=float(a.mean()), new_mean=float(b.mean()),
                mean_delta=float(delta.mean()), max_abs_delta=float(np.abs(delta).max()),
                paired_bootstrap_mean_delta_ci95=np.quantile(samples, [.025, .975]).tolist())
        item['filter_calls'] = {tag: run['summary']['filter_calls'] for tag, run in runs.items()}
        item['rejected'] = {tag: run['summary']['rejected_by_filter'] for tag, run in runs.items()}
        output.append(item)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--filter-dir', type=Path)
    p.add_argument('--em-dir', type=Path)
    p.add_argument('--scale-dir', type=Path)
    p.add_argument('--quality-dir', type=Path)
    p.add_argument('--api-dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = {}
    for name in ['filter', 'em', 'scale']:
        directory = getattr(args, name + '_dir')
        if directory is not None:
            result[name] = abba(directory)
    if args.quality_dir is not None:
        result['quality'] = [dict(file=f.name, **json.loads(f.read_text())['summary'])
                             for f in sorted(args.quality_dir.glob('*.json'))]
    if args.api_dir is not None:
        result['sklearn'] = [{k: v for k, v in json.loads(f.read_text()).items() if k != 'records'}
                             for f in sorted(args.api_dir.glob('api-*.json'))]
        assert all(v['passed'] for v in result['sklearn'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for name in ['filter', 'em', 'scale']:
        for row in result.get(name, []):
            print(name, row['name'], json.dumps(row['gains']), row['paired_tasks'])


if __name__ == '__main__':
    main()
