"""Summarize ABBA throughput, aligned max-rank steps, and task-level audits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np


def summary(values):
    x = np.asarray(values, dtype=float)
    return dict(mean=float(x.mean()), p95=float(np.quantile(x, .95)), maximum=float(x.max()))


def training(path, skip):
    run = json.loads((path / 'run.json').read_text())
    records = [json.loads(line) for line in (path/'logs/metrics.jsonl').read_text().splitlines()]
    if not 1 <= skip < len(records):
        raise ValueError(f'{path}: skip must be between 1 and {len(records)-1}')
    rows = records[skip:]
    per_step = [max(r[f'rank_timing/rank{i}_prior_s'] + r[f'rank_timing/rank{i}_train_s']
                    for i in range(run['gpus'])) * 1000 for r in rows]
    elapsed = np.diff([r['elapsed_s'] for r in records[skip-1:]]) * 1000
    flags = ['dtype', 'batch_size', 'micro_batch_size', 'prior_backend', 'prior_schedule',
             'prior_ready_max_delay', 'prior_pool_workers', 'prior_pool_window', 'prior_pool_mb',
             'n_jobs', 'categorical_sampling', 'dataset_filter_mode', 'compile_frontend',
             'compile_warmup', 'rank_timing_every']
    configuration = {name: run['command'][run['command'].index('--'+name)+1]
                     for name in flags if '--'+name in run['command']}
    pool_rows = [dict(logged_step=i+1, **{k: v for k, v in r.items() if k.startswith('prior_pool/')})
                 for i, r in enumerate(records) if any(k.startswith('prior_pool/') for k in r)]
    return dict(directory=str(path), exit_code=run['exit_code'], wall_s=run['wall_seconds'],
                configuration=configuration,
                latest_pool_diagnostics=pool_rows[-1] if pool_rows else {},
                steps=len(records), skip=skip,
                steady_step_ms=(records[-1]['elapsed_s']-records[skip-1]['elapsed_s']) / len(rows)*1000,
                steady_elapsed_step_ms=summary(elapsed),
                aligned_max_rank_step_ms=summary(per_step),
                max_rank_prior_ms=summary([r['rank_timing/prior_max_s']*1000 for r in rows]),
                rank_mean_prior_ms=summary([r['rank_timing/prior_mean_s']*1000 for r in rows]),
                max_rank_train_ms=summary([r['rank_timing/train_max_s']*1000 for r in rows]),
                per_rank={str(i): {phase: summary([r[f'rank_timing/rank{i}_{phase}_s']*1000 for r in rows])
                                   for phase in ('prior', 'train')} for i in range(run['gpus'])},
                oom=sum(r.get('oom_micro_batches', 0) for r in records),
                nonfinite=sum(not np.isfinite(r['ce']) for r in records))


def audit(path):
    records = json.loads(path.read_text())['records']
    result = {}
    fields = ['width', 'classes', 'minority', 'normalized_entropy', 'attempts',
              'ridge_balanced_accuracy', 'tree_balanced_accuracy', 'constant_fraction',
              'nodes', 'edges', 'live_nodes', 'depth']
    for seed, mode in sorted({(r['seed'], r['mode']) for r in records}):
        rows = [r | {k: v for k, v in r['accepted_graph'].items() if k != 'components'}
                for r in records if r['seed'] == seed and r['mode'] == mode]
        stats = {}
        for field in fields:
            values = np.asarray([r[field] for r in rows], dtype=float)
            rng = np.random.default_rng(617)
            means = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
            stats[field] = dict(mean=float(values.mean()),
                               bootstrap_mean_95ci=list(np.quantile(means, [.025, .975])))
        components = Counter()
        for r in rows:
            for name, count in r['accepted_graph']['components'].items():
                # Also normalize audit files generated before stable labels.
                components[re.sub(r' at 0x[0-9a-fA-F]+', '', name)] += count
        result[f'{seed}_{mode}'] = dict(tasks=len(rows), statistics=stats,
                                       accepted_components=dict(components))
    # Candidate random streams diverge, but task IDs share the same requested
    # metadata. Bootstrap matched IDs, without treating graph/node choices as
    # independent observations or claiming equivalence from a nonsignificant p.
    paired = {mode: {(r['seed'], r['task']): r | r['accepted_graph']
                     for r in records if r['mode'] == mode}
              for mode in ('legacy', 'marginal')}
    keys = sorted(paired['legacy'].keys() & paired['marginal'].keys())
    differences = {}
    for field in fields:
        x = np.array([paired['marginal'][k][field] - paired['legacy'][k][field] for k in keys])
        rng = np.random.default_rng(617)
        means = x[rng.integers(len(x), size=(2000, len(x)))].mean(1)
        differences[field] = dict(mean=float(x.mean()),
                                  bootstrap_mean_95ci=list(np.quantile(means, [.025, .975])))
    result['paired_marginal_minus_legacy'] = dict(tasks=len(keys), statistics=differences)
    # Counts are descriptive: choices within a graph are correlated, so do not
    # turn node-level sample sizes into task-level significance claims.
    mixtures = {}
    for label, group in result.items():
        if 'accepted_components' not in group:
            continue
        counts = {name.split('/', 1)[1]: count
                  for name, count in group['accepted_components'].items()
                  if name.startswith('random_function_fct_type/')}
        mixtures[label] = {name: count / sum(counts.values()) for name, count in counts.items()}
    def variation(left, right):
        return sum(abs(left.get(k, 0)-right.get(k, 0)) for k in left.keys() | right.keys()) / 2
    seeds = sorted({r['seed'] for r in records})
    result['function_mixture'] = dict(proportions=mixtures,
        same_seed_mode_total_variation={str(s): variation(mixtures[f'{s}_legacy'], mixtures[f'{s}_marginal'])
                                        for s in seeds},
        legacy_between_seed_total_variation={f'{s}_{t}': variation(mixtures[f'{s}_legacy'], mixtures[f'{t}_legacy'])
                                             for i, s in enumerate(seeds) for t in seeds[i+1:]})
    return result


def comparisons(runs):
    def aggregate(names):
        selected = [runs[name] for name in names]
        steps = sum(r['steps'] - r['skip'] for r in selected)
        return dict(runs=names, steady_steps=steps,
                    steady_step_ms=sum(r['steady_step_ms'] * (r['steps'] - r['skip'])
                                       for r in selected) / steps,
                    total_wall_s=sum(r['wall_s'] for r in selected),
                    mean_run_step_p95_ms=float(np.mean([
                        r['steady_elapsed_step_ms']['p95'] for r in selected])),
                    worst_step_ms=max(r['steady_elapsed_step_ms']['maximum'] for r in selected))
    specs = {}
    for dtype in ('bfloat16', 'float32'):
        baseline = [f'{dtype}_{s}' for s in ('a1', 'a2')]
        specs[f'{dtype}_ready_delivery'] = (baseline, [f'{dtype}_{s}' for s in ('b1', 'b2')])
        specs[f'{dtype}_ready_and_marginal'] = (baseline, [f'{dtype}_{s}' for s in ('c1', 'c2')])
        specs[f'{dtype}_delay_2_to_8'] = ([f'{dtype}_delay_{s}' for s in ('a1', 'a2')],
                                         [f'{dtype}_delay_{s}' for s in ('b1', 'b2')])
    specs['bfloat16_dataloader_marginal_single_run'] = (['bfloat16_d1'], ['bfloat16_d2'])
    result = {}
    for name, (before, after) in specs.items():
        if not all(n in runs for n in before + after):
            continue
        old, new = aggregate(before), aggregate(after)
        result[name] = dict(before=old, after=new,
            steady_throughput_gain=old['steady_step_ms'] / new['steady_step_ms'] - 1,
            total_wall_reduction=1 - new['total_wall_s'] / old['total_wall_s'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--skip', type=int, default=20)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = dict(cpu={}, training={}, audit={})
    for rows in (2048, 4096):
        runs = {p.stem: json.loads(p.read_text())['summary']
                for p in (a.root/'cpu').glob(f'n{rows}_*.json')}
        if runs:
            entry = dict(runs=runs)
            if len(runs) == 4:
                legacy = [runs[f'n{rows}_{s}'] for s in ('a1', 'a2')]
                marginal = [runs[f'n{rows}_{s}'] for s in ('b1', 'b2')]
                entry.update(throughput_gain=sum(r['wall_s'] for r in legacy)/sum(r['wall_s'] for r in marginal)-1,
                    cpu_reduction=1-sum(r['cpu_s'] for r in marginal)/sum(r['cpu_s'] for r in legacy))
            result['cpu'][rows] = entry
    for path in sorted((a.root/'training').glob('*/run.json')):
        result['training'][path.parent.name] = training(path.parent, a.skip)
    result['training_comparisons'] = comparisons(result['training'])
    for path in sorted((a.root/'audit').glob('*.json')):
        result['audit'][path.stem] = audit(path)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(cpu={k: {f: v for f, v in r.items() if f != 'runs'} for k, r in result['cpu'].items()},
        training={k: {f: r[f] for f in ('steady_step_ms', 'max_rank_prior_ms', 'oom', 'nonfinite')}
                  for k, r in result['training'].items()})), flush=True)


if __name__ == '__main__':
    main()
