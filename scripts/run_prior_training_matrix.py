"""Sequential real-prior eight-rank comparisons, never synthetic shape tasks.

Select screening, ABBA finalist verification or longer steady-state tests.
Existing runs are refused to prevent accidental checkpoint resume.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from summarize_prior_ready_experiments import training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=240)
    p.add_argument('--skip', type=int, default=40)
    p.add_argument('--rows', type=int, nargs='+', default=[2048])
    p.add_argument('--dtypes', nargs='+', default=['float32', 'bfloat16'],
        choices=['float32', 'bfloat16'])
    p.add_argument('--cases', nargs='+', default=['legacy', 'previous', 'ready', 'loader', 'balanced'],
        choices=['legacy', 'previous', 'ready', 'loader', 'balanced'])
    p.add_argument('--abba', action='store_true')
    p.add_argument('--pool-workers', type=int, default=56)
    p.add_argument('--loader-workers', type=int, default=8)
    p.add_argument('--tag', default='')
    a = p.parse_args()
    if a.abba and len(a.cases) != 2:
        p.error('--abba requires exactly two cases')
    root = Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(root/'src'), PYTHONHASHSEED='42',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
        PRIOR_READY_MAX_DELAY='2', PRIOR_FILTER_MODE='full', PRIOR_POOL_WORKERS=str(a.pool_workers),
        PRIOR_WORKERS=str(a.loader_workers), RANK_TIMING_EVERY='1',
        COMPILE_FRONTEND='all', COMPILE_WARMUP='True',
        PRIOR_GRAPH_EXECUTION='optimized', PRIOR_CATEGORICAL_SAMPLING='marginal',
        PRIOR_FILTER_EXECUTION='sufficient', TABICL_PRIOR_COMPACT='hash',
        TABICL_PRIOR_STATISTICS='native')
    settings = {
        'legacy': dict(PRIOR_GRAPH_EXECUTION='reference', PRIOR_CATEGORICAL_SAMPLING='legacy',
            TABICL_PRIOR_COMPACT='sort', TABICL_PRIOR_STATISTICS='torch'),
        'previous': dict(PRIOR_BACKEND='node_pool', PRIOR_SCHEDULE='ready_micro',
            TABICL_PRIOR_COMPACT='sort', TABICL_PRIOR_STATISTICS='torch'),
        'ready': dict(PRIOR_BACKEND='node_pool', PRIOR_SCHEDULE='ready_micro'),
        'loader': dict(PRIOR_BACKEND='dataloader', PRIOR_SCHEDULE='none'),
        'balanced': dict(PRIOR_BACKEND='node_pool', PRIOR_SCHEDULE='balanced')}
    specs = [(case, 'single') for case in a.cases]
    if a.abba:
        specs = list(zip([a.cases[0], a.cases[1], a.cases[1], a.cases[0]], ['a1','b1','b2','a2']))
    reports = {}
    for rows in a.rows:
        for dtype in a.dtypes:
            for case, tag in specs:
                name = f'{a.tag}{dtype}-n{rows}-{case}-{tag}'
                path = a.output/name
                if path.exists() or (a.output/f'{name}.log').exists():
                    raise FileExistsError(path)
                extra = settings[case]
                if case == 'legacy':
                    extra = extra | dict(PRIOR_BACKEND='node_pool' if dtype == 'float32' else 'dataloader',
                        PRIOR_SCHEDULE='balanced' if dtype == 'float32' else 'none')
                command = [sys.executable, 'scripts/benchmark_training_schedule.py',
                    '--gpus', '8', '--steps', str(a.steps), '--dtype', dtype, '--output', str(path)]
                print(json.dumps(dict(start=name, settings=extra)), flush=True)
                with (a.output/f'{name}.log').open('w') as log:
                    subprocess.run(command, cwd=root, env=env | {'MAX_SEQ_LEN': str(rows)} | extra,
                        stdout=log, stderr=subprocess.STDOUT, check=True)
                reports[name] = training(path, a.skip)
                reports[name]['settings'] = extra
                (a.output/'summary.json').write_text(json.dumps(reports, indent=2)+'\n')
                print(json.dumps(dict(finished=name, steady_step_ms=reports[name]['steady_step_ms'],
                    wall_s=reports[name]['wall_s'], oom=reports[name]['oom'], nonfinite=reports[name]['nonfinite'])), flush=True)


if __name__ == '__main__':
    main()
