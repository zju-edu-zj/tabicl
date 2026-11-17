"""ABBA full-task comparison; both arms retain the optimized graph baseline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', type=int, default=512)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--rows', type=int, nargs='+', default=[2048, 4096])
    p.add_argument('--seed', type=int, default=83)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root/'src'), PYTHONHASHSEED='42',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    results = {}
    for rows in a.rows:
        records = {}
        for tag, mode in [('a1', 'reference'), ('b1', 'planned'), ('b2', 'planned'), ('a2', 'reference')]:
            name = f'n{rows}_w{a.workers}_{tag}'
            result = a.output/f'{name}.json'
            if result.exists():
                raise FileExistsError(result)
            command = [sys.executable, 'scripts/benchmark_prior_exact.py', '--tasks', str(a.tasks),
                       '--workers', str(a.workers), '--rows', str(rows), '--seed', str(a.seed),
                       '--categorical-sampling', 'marginal', '--graph-execution', 'optimized',
                       '--filter-execution', mode, '--output', str(result)]
            print(json.dumps(dict(start=name, command=command)), flush=True)
            with (a.output/f'{name}.log').open('w') as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            records[tag] = json.loads(result.read_text())
            if [r['task'] for r in records[tag]['records']] != list(range(a.tasks)):
                raise RuntimeError(f'{name}: incomplete, reordered or duplicate task records')
            print(json.dumps(dict(finished=name, summary=records[tag]['summary'])), flush=True)
        reference = records['a1']['records']
        equivalence = {}
        for tag, run in records.items():
            equivalence[tag] = {key: sum(x[key] != y[key] for x, y in zip(reference, run['records']))
                for key in ['digest', 'filter_decisions', 'width', 'classes', 'forest_fit_sizes']}
        means = {mode: {key: sum(records[tag]['summary'][key] for tag in tags)/2
                 for key in ['wall_s', 'cpu_s', 'filter_s', 'graph_s', 'tables_per_s']}
                 for mode, tags in [('reference', ['a1', 'a2']), ('planned', ['b1', 'b2'])]}
        results[str(rows)] = dict(means=means, mismatches=equivalence,
            throughput_gain=means['reference']['wall_s']/means['planned']['wall_s']-1,
            cpu_reduction=1-means['planned']['cpu_s']/means['reference']['cpu_s'],
            summaries={tag: run['summary'] for tag, run in records.items()})
        (a.output/'summary.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(results[str(rows)]), flush=True)
        if any(count for fields in equivalence.values() for count in fields.values()):
            raise RuntimeError(f'{rows}: complete-task equivalence audit failed; see summary.json')


if __name__ == '__main__':
    main()
