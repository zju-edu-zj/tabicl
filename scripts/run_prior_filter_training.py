"""Real eight-rank ABBA comparison against the current optimized graph path."""
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
    p.add_argument('--dtypes', nargs='+', choices=['float32', 'bfloat16'], default=['bfloat16'])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root/'src'), PYTHONHASHSEED='42',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PRIOR_BACKEND='node_pool', PRIOR_SCHEDULE='ready_micro',
               PRIOR_READY_MAX_DELAY='2', PRIOR_CATEGORICAL_SAMPLING='marginal',
               PRIOR_FILTER_MODE='full', PRIOR_GRAPH_EXECUTION='optimized',
               RANK_TIMING_EVERY='1', COMPILE_FRONTEND='all', COMPILE_WARMUP='True')
    reports = {}
    for dtype in a.dtypes:
        for tag, mode in [('a1', 'reference'), ('b1', 'planned'), ('b2', 'planned'), ('a2', 'reference')]:
            name = f'{dtype}_{tag}'
            if (a.output/name).exists():
                raise FileExistsError(name)
            command = [sys.executable, 'scripts/benchmark_training_schedule.py',
                '--gpus', '8', '--steps', str(a.steps), '--dtype', dtype, '--output', str(a.output/name)]
            print(json.dumps(dict(start=name, filter_execution=mode)), flush=True)
            with (a.output/f'{name}.log').open('w') as log:
                subprocess.run(command, cwd=root, env=env | {'PRIOR_FILTER_EXECUTION': mode},
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            reports[name] = training(a.output/name, a.skip)
            reports[name]['configuration'].update(graph_execution='optimized', filter_execution=mode)
            (a.output/'summary.json').write_text(json.dumps(reports, indent=2))
            print(json.dumps(dict(finished=name, report=reports[name])), flush=True)
    comparisons = {}
    for dtype in a.dtypes:
        old = sum(reports[f'{dtype}_{s}']['steady_step_ms'] for s in ('a1', 'a2'))/2
        new = sum(reports[f'{dtype}_{s}']['steady_step_ms'] for s in ('b1', 'b2'))/2
        comparisons[dtype] = dict(reference_ms=old, planned_ms=new, throughput_gain=old/new-1,
            caveat='Short steady windows include scheduling/shape variation; shared compile caches '
                   'make launch wall unsuitable for performance comparisons.')
    (a.output/'comparisons.json').write_text(json.dumps(comparisons, indent=2))
    print(json.dumps(comparisons), flush=True)


if __name__ == '__main__':
    main()
