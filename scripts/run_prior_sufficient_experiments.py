"""Sequential correctness, quality and ABBA experiments; never start training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['api', 'audit', 'throughput', 'regression'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', type=int, default=512)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seeds', type=int, nargs='+', default=[93, 94])
    p.add_argument('--comparison', choices=['filter', 'em', 'combined'], default='filter')
    p.add_argument('--warm-workers', action='store_true')
    p.add_argument('--versions', nargs='+', default=['1.3.2', '1.4.0', '1.4.1.post1', '1.4.2',
        '1.5.0', '1.5.1', '1.5.2', '1.6.0', '1.6.1', '1.7.0', '1.7.1', '1.7.2', '1.8.0', '1.9.0', '1.9.1'])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root/'src'), PYTHONHASHSEED='42',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    def run(name, args, extra=None):
        print(json.dumps(dict(start=name)), flush=True)
        log = a.output / f'{name}.log'
        if log.exists():
            raise FileExistsError(log)
        with log.open('w') as output:
            subprocess.run([sys.executable, *args], cwd=root, env=env | (extra or {}),
                           stdout=output, stderr=subprocess.STDOUT, check=True)
        print(json.dumps(dict(finished=name)), flush=True)
    if a.phase == 'api':
        # Wheels contain many small files. Install on node-local storage, not
        # the persistent network mount; keep only logs/results on that mount.
        packages = Path(tempfile.mkdtemp(prefix='tabicl-sklearn-matrix-'))
        common = packages / 'common'
        if any(tuple(map(int, v.split('.')[:2])) >= (1, 9) for v in a.versions):
            # sklearn 1.9 added this runtime dependency; don't change the
            # training environment merely to inspect another sklearn wheel.
            run('install-narwhals', ['-m', 'pip', 'install', '--disable-pip-version-check',
                '--no-deps', '--no-compile', '--target', str(common), 'narwhals>=2.0.1'])
        for version in a.versions:
            target = packages / version
            run(f'install-{version}', ['-m', 'pip', 'install', '--disable-pip-version-check',
                '--no-deps', '--no-compile', '--only-binary=:all:', '--target', str(target), f'scikit-learn=={version}'])
            run(f'api-{version}', ['scripts/check_prior_forest_api.py', '--output', str(a.output/f'api-{version}.json')],
                dict(PYTHONPATH=os.pathsep.join(map(str, [target, common, root/'src']))))
    elif a.phase in ('audit', 'throughput'):
        for seed in a.seeds:
            for rows in (2048, 4096):
                common = ['--tasks', str(a.tasks), '--workers', str(a.workers), '--rows', str(rows), '--seed', str(seed)]
                if a.phase == 'audit':
                    name = f'audit-s{seed}-n{rows}'
                    run(name, ['scripts/audit_prior_sufficient_filter.py', *common, '--output', str(a.output/f'{name}.json')])
                else:
                    for tag, mode in [('a1', 'planned'), ('b1', 'sufficient'), ('b2', 'sufficient'), ('a2', 'planned')]:
                        name = f's{seed}-n{rows}-w{a.workers}-{tag}'
                        extra = ['--warm-workers'] if a.warm_workers else []
                        if a.comparison in ('em', 'combined'):
                            if a.comparison == 'em':
                                mode = 'sufficient'
                            if tag.startswith('a'):
                                extra.extend(['--disable', 'em'])
                        run(name, ['scripts/benchmark_prior_exact.py', *common, '--categorical-sampling', 'marginal',
                            '--graph-execution', 'optimized', '--filter-execution', mode, '--output', str(a.output/f'{name}.json'), *extra])
    else:
        tests = sorted(str(path.relative_to(root)) for path in (root/'tests').glob('test_prior*.py'))
        run('regression', ['-m', 'pytest', '-q', '--color=no', *tests, 'tests/test_node_prior_pool.py', 'tests/test_training_recipe.py'])


if __name__ == '__main__':
    main()
