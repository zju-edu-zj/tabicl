"""Serial real-prior eight-rank ABBA validation; no shape-only substitute."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from summarize_training_schedule import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--skip', type=int, default=80)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dtype', nargs='+', choices=['float32', 'bfloat16'], default=['float32', 'bfloat16'])
    a = p.parse_args()
    if not 0 <= a.skip < a.steps:
        p.error('require 0 <= skip < steps')
    a.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    results = []
    for dtype in a.dtype:
        for index, mode in enumerate(('sklearn', 'native', 'native', 'sklearn')):
            path = a.output/f'{dtype}-{index}-{mode}'
            env = dict(os.environ, TABICL_PRIOR_FOREST=mode, PYTHONHASHSEED='42')
            env.pop('TABICL_PRIOR_TRACE_DIR', None)  # instrumentation belongs in a separate run
            command = [sys.executable, str(root/'scripts/benchmark_training_schedule.py'),
                '--gpus', '8', '--steps', str(a.steps), '--dtype', dtype, '--output', str(path),
                '--train-args', '--rank_timing_every', '1']
            with path.with_suffix('.log').open('x') as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            record = summarize(path, a.skip)
            record.update(forest=mode, position=index)
            results.append(record)
            (a.output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
            print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
