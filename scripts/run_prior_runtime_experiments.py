"""Sequential full-prior ABBA comparisons of statistics and row compaction."""
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
    p.add_argument('--seeds', type=int, nargs='+', default=[96, 97])
    p.add_argument('--rows', type=int, nargs='+', default=[2048, 4096])
    p.add_argument('--component', choices=['statistics', 'compact', 'both'], default='both')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root/'src'), PYTHONHASHSEED='42',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for seed in a.seeds:
        for rows in a.rows:
            for tag in ('a1', 'b1', 'b2', 'a2'):
                enabled = tag.startswith('b')
                extra = dict(TABICL_PRIOR_STATISTICS='native' if enabled and a.component in ('statistics', 'both') else 'torch',
                    TABICL_PRIOR_COMPACT='hash' if enabled and a.component in ('compact', 'both') else 'sort')
                name = f's{seed}-n{rows}-w{a.workers}-{tag}'
                output = a.output/f'{name}.json'
                log = a.output/f'{name}.log'
                if log.exists() or output.exists():
                    raise FileExistsError(name)
                command = [sys.executable, 'scripts/benchmark_prior_exact.py', '--warm-workers',
                    '--tasks', str(a.tasks), '--workers', str(a.workers), '--rows', str(rows), '--seed', str(seed),
                    '--categorical-sampling', 'marginal', '--graph-execution', 'optimized',
                    '--filter-execution', 'sufficient', '--output', str(output)]
                print(json.dumps(dict(start=name, component=a.component, environment=extra)), flush=True)
                with log.open('w') as stream:
                    subprocess.run(command, cwd=root, env=env | extra, stdout=stream,
                        stderr=subprocess.STDOUT, check=True)
                print(json.dumps(dict(finished=name, summary=json.loads(output.read_text())['summary'])), flush=True)


if __name__ == '__main__':
    main()
