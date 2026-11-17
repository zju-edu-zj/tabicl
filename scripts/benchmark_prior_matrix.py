"""Serial ABBA full-prior benchmark; keep accelerator training stopped.

Each cell starts a fresh spawned worker pool and records cold/startup wall
time as well as warm generation throughput and total task process CPU time.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=56)
    p.add_argument('--tasks', type=int, default=2048)
    p.add_argument('--rows', type=int, nargs='+', default=[2048, 4096])
    p.add_argument('--seeds', type=int, nargs='+', default=[98, 99])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if not os.environ.get('TABICL_PRIOR_DISTANCE_LIBRARY'):
        p.error('build and export TABICL_PRIOR_DISTANCE_LIBRARY first')
    root = Path(__file__).resolve().parents[1]
    results = []
    for rows in a.rows:
        for seed in a.seeds:
            for position, mode in enumerate(('A', 'B', 'B', 'A')):
                env = dict(os.environ, TABICL_PRIOR_FOREST='sklearn',
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONHASHSEED='42')
                if mode == 'B':
                    env['TABICL_PRIOR_FOREST'] = 'native'
                path = a.output/f'n{rows}-seed{seed}-{position}-{mode}.json'
                command = [sys.executable, str(root/'scripts/benchmark_prior_exact.py'),
                    '--workers', str(a.workers), '--tasks', str(a.tasks), '--rows', str(rows),
                    '--seed', str(seed), '--warm-workers', '--categorical-sampling', 'marginal',
                    '--graph-execution', 'optimized', '--filter-execution', 'sufficient', '--output', str(path)]
                with path.with_suffix('.log').open('x') as log:
                    subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                summary = json.loads(path.read_text())['summary']
                summary.update(mode=mode, position=position, proposal='forest')
                results.append(summary)
                (a.output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
                print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
