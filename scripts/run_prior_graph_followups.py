"""Isolated ablations or real 8-rank training; never overlap these workloads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['ablation', 'training'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', type=int, default=512)
    p.add_argument('--steps', type=int, default=160)
    p.add_argument('--resume', action='store_true', help='Skip completed successful cases')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root/'src'), PYTHONHASHSEED='42',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    cases = []
    if a.phase == 'ablation':
        for seed in (83, 84):
            for label, mode, disabled in [
                ('reference', 'reference', []),
                ('embedding', 'optimized', ['distance', 'quadratic']),
                ('distance', 'optimized', ['embedding', 'quadratic']),
                ('quadratic', 'optimized', ['distance', 'embedding']),
                ('all', 'optimized', [])]:
                name = f's{seed}_{label}'
                command = [sys.executable, 'scripts/benchmark_prior_exact.py', '--tasks', str(a.tasks),
                    '--workers', '8', '--rows', '2048', '--seed', str(seed),
                    '--categorical-sampling', 'marginal', '--graph-execution', mode,
                    '--output', str(a.output/f'{name}.json'), '--disable', *disabled]
                cases.append((name, command, {}))
    else:
        for dtype in ('float32', 'bfloat16'):
            for label, mode in [('a1', 'reference'), ('b1', 'optimized'),
                                ('b2', 'optimized'), ('a2', 'reference')]:
                name = f'{dtype}_{label}'
                command = [sys.executable, 'scripts/benchmark_training_schedule.py',
                    '--gpus', '8', '--steps', str(a.steps), '--dtype', dtype,
                    '--output', str(a.output/name)]
                settings = dict(PRIOR_BACKEND='node_pool', PRIOR_SCHEDULE='ready_micro',
                    PRIOR_READY_MAX_DELAY='2', PRIOR_CATEGORICAL_SAMPLING='marginal',
                    PRIOR_FILTER_MODE='full', PRIOR_GRAPH_EXECUTION=mode,
                    RANK_TIMING_EVERY='1', COMPILE_FRONTEND='all', COMPILE_WARMUP='True')
                cases.append((name, command, settings))
    for name, command, settings in cases:
        completed = a.output/f'{name}.json' if a.phase == 'ablation' else a.output/name/'run.json'
        if a.resume and completed.exists():
            saved = json.loads(completed.read_text())
            if a.phase == 'ablation' or saved.get('exit_code') == 0:
                print(json.dumps(dict(skipped_completed=name)), flush=True)
                continue
        if completed.exists() or (a.output/name).exists():
            raise FileExistsError(name)
        print(json.dumps(dict(start=name, settings=settings)), flush=True)
        with (a.output/f'{name}.log').open('w') as log:
            subprocess.run(command, cwd=root, env=env | settings,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        result = a.output/f'{name}.json'
        print(json.dumps(dict(finished=name,
            summary=json.loads(result.read_text())['summary'] if result.exists() else {})), flush=True)


if __name__ == '__main__':
    main()
