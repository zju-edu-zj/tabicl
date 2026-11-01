"""Sequential isolated ABBA experiments; raw logs/results stay in --output.

Use the vendor SDK environment for training. CPU benchmarking and DDP training
must not run concurrently: both compete for the same prior generation CPUs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['cpu', 'training', 'delay'], required=True)
    p.add_argument('--dtypes', choices=['bfloat16', 'float32'], nargs='+',
                   default=['bfloat16', 'float32'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', type=int, default=1024)
    p.add_argument('--steps', type=int, default=120)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root/'src'), PYTHONHASHSEED='42',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if a.phase == 'cpu':
        cases = [(f'n{n}_{name}', ['scripts/benchmark_prior_exact.py', '--tasks', str(a.tasks),
                   '--workers', '8', '--rows', str(n), '--seed', str(53 + n//2048),
                   '--categorical-sampling', mode, '--output', str(a.output/f'n{n}_{name}.json')], {})
                 for n in (2048, 4096)
                 for name, mode in [('a1', 'legacy'), ('b1', 'marginal'),
                                    ('b2', 'marginal'), ('a2', 'legacy')]]
    else:
        cases = []
        # Isolate delivery from categorical generation, then isolate generation
        # with ready delivery held fixed. Compare BF16's existing DataLoader too.
        for dtype in a.dtypes:
            baseline = 'none' if dtype == 'bfloat16' else 'balanced'
            if a.phase == 'delay':
                specs = [(name, 'ready_micro', 'marginal', delay)
                         for name, delay in [('delay_a1', 2), ('delay_b1', 8),
                                             ('delay_b2', 8), ('delay_a2', 2)]]
            else:
                specs = [('a1', baseline, 'legacy', 2), ('b1', 'ready_micro', 'legacy', 2),
                         ('b2', 'ready_micro', 'legacy', 2), ('a2', baseline, 'legacy', 2),
                         ('c1', 'ready_micro', 'marginal', 2), ('c2', 'ready_micro', 'marginal', 2)]
                if dtype == 'bfloat16':
                    specs += [('d1', 'dataloader', 'legacy', 2), ('d2', 'dataloader', 'marginal', 2)]
            for name, schedule, mode, delay in specs:
                name = f'{dtype}_{name}'
                settings = dict(PRIOR_BACKEND='dataloader' if schedule == 'dataloader' else 'node_pool',
                    PRIOR_SCHEDULE='none' if schedule == 'dataloader' else schedule,
                    PRIOR_CATEGORICAL_SAMPLING=mode, PRIOR_FILTER_MODE='full',
                    RANK_TIMING_EVERY='1', PRIOR_READY_MAX_DELAY=str(delay),
                    COMPILE_FRONTEND='all', COMPILE_WARMUP='True')
                cases.append((name, ['scripts/benchmark_training_schedule.py', '--gpus', '8',
                    '--steps', str(a.steps), '--dtype', dtype, '--output', str(a.output/name)], settings))
    for name, command, settings in cases:
        if a.phase != 'cpu' and (a.output/name).exists():
            raise FileExistsError(f'{a.output/name}: use a fresh output directory to prevent checkpoint resume')
        print(json.dumps(dict(start=name, settings=settings)), flush=True)
        with (a.output/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable, *command], cwd=root, env=env | settings,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps(dict(finished=name)), flush=True)


if __name__ == '__main__':
    main()
