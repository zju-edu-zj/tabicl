"""Isolated ABBA complete-task comparisons against the SAME marginal baseline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', type=int, default=1024)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--rows', type=int, nargs='+', default=[2048, 4096])
    p.add_argument('--seed', type=int, default=83)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root/'src'), PYTHONHASHSEED='42',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for rows in a.rows:
        for tag, mode in [('a1', 'reference'), ('b1', 'optimized'), ('b2', 'optimized'), ('a2', 'reference')]:
            name = f'n{rows}_w{a.workers}_{tag}'
            result = a.output/f'{name}.json'
            if result.exists():
                raise FileExistsError(result)
            command = [sys.executable, 'scripts/benchmark_prior_exact.py', '--tasks', str(a.tasks),
                       '--workers', str(a.workers), '--rows', str(rows), '--seed', str(a.seed),
                       '--categorical-sampling', 'marginal', '--graph-execution', mode,
                       '--output', str(result)]
            print(json.dumps(dict(start=name, command=command)), flush=True)
            with (a.output/f'{name}.log').open('w') as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(json.dumps(dict(finished=name, summary=json.loads(result.read_text())['summary'])), flush=True)


if __name__ == '__main__':
    main()
