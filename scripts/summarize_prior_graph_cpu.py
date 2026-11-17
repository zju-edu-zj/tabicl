"""Summarize complete-task throughput, real DDP timing and task-level audits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from summarize_prior_ready_experiments import training


def audit(path):
    records = json.loads(path.read_text())['records']
    fields = ['width', 'classes', 'minority', 'normalized_entropy', 'attempts',
              'ridge_balanced_accuracy', 'tree_balanced_accuracy', 'constant_fraction',
              'nodes', 'edges', 'live_nodes', 'depth']
    rows = [r | {k:v for k,v in r['accepted_graph'].items() if k != 'components'} for r in records]
    groups = {}
    mixtures = {}
    for seed, mode in sorted({(r['seed'], r['mode']) for r in rows}):
        selected = [r for r in rows if (r['seed'], r['mode']) == (seed, mode)]
        components = Counter()
        for r in selected: components.update(r['accepted_graph']['components'])
        live = {k.split('/', 1)[1]: v for k,v in components.items()
                if k.startswith('random_function_fct_type/')}
        proportions = {k:v/sum(live.values()) for k,v in live.items()}
        mixtures[f'{seed}_{mode}'] = proportions
        groups[f'{seed}_{mode}'] = dict(tasks=len(selected),
            means={f:float(np.mean([r[f] for r in selected])) for f in fields},
            live_function_proportions=proportions,
            accepted_components=dict(components))
    paired = {mode: {(r['seed'], r['task']):r for r in rows if r['mode'] == mode}
              for mode in ('reference', 'optimized')}
    keys = sorted(paired['reference'].keys() & paired['optimized'].keys())
    deltas = {}
    for field in fields:
        values = np.asarray([paired['optimized'][k][field]-paired['reference'][k][field] for k in keys])
        rng = np.random.default_rng(617)
        means = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
        deltas[field] = dict(mean=float(values.mean()), bootstrap_95ci=list(np.quantile(means, [.025, .975])))
    def tv(a,b): return sum(abs(a.get(k,0)-b.get(k,0)) for k in a.keys() | b.keys())/2
    seeds = sorted({r['seed'] for r in rows})
    return dict(groups=groups, paired_optimized_minus_reference=deltas,
        live_function_total_variation=dict(
            same_seed={str(s):tv(mixtures[f'{s}_reference'], mixtures[f'{s}_optimized']) for s in seeds},
            reference_between_seeds={f'{s}_{t}':tv(mixtures[f'{s}_reference'], mixtures[f'{t}_reference'])
                for i,s in enumerate(seeds) for t in seeds[i+1:]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--skip', type=int, default=20)
    a = p.parse_args()
    result = dict(cpu={}, ablation={}, audit={}, training={})
    for rows in (2048, 4096):
        runs = {f.stem:json.loads(f.read_text())['summary']
                for f in (a.root/'ordered-cpu').glob(f'n{rows}_*.json')}
        if len(runs) == 4:
            old = [r for r in runs.values() if r['graph_execution'] == 'reference']
            new = [r for r in runs.values() if r['graph_execution'] == 'optimized']
            result['cpu'][rows] = dict(runs=runs,
                throughput_gain=sum(r['wall_s'] for r in old)/sum(r['wall_s'] for r in new)-1,
                cpu_reduction=1-sum(r['cpu_s'] for r in new)/sum(r['cpu_s'] for r in old))
    for path in sorted((a.root/'ablation').glob('*.json')):
        result['ablation'][path.stem] = json.loads(path.read_text())['summary']
    for path in sorted(a.root.glob('audit*.json')):
        result['audit'][path.stem] = audit(path)
    for path in sorted((a.root/'training').glob('*/run.json')):
        report = training(path.parent, a.skip)
        command = json.loads(path.read_text())['command']
        report['configuration']['graph_execution'] = command[command.index('--graph_execution')+1]
        result['training'][path.parent.name] = report
    result['training_comparisons'] = {}
    for dtype in ('float32', 'bfloat16'):
        runs = result['training']
        names = [f'{dtype}_{label}' for label in ('a1', 'b1', 'b2', 'a2')]
        if all(n in runs for n in names):
            old, new = [runs[names[i]] for i in (0,3)], [runs[names[i]] for i in (1,2)]
            result['training_comparisons'][dtype] = dict(
                throughput_gain=sum(r['steady_step_ms'] for r in old)/sum(r['steady_step_ms'] for r in new)-1,
                observed_launch_wall_reduction=1-sum(r['wall_s'] for r in new)/sum(r['wall_s'] for r in old),
                launch_wall_comparison_controlled=False,
                launch_wall_caveat='Runs share compiler caches; the first run can pay cold compilation. '
                    'Launch wall includes startup, compilation, warmup and shutdown. '
                    'Use steady-step throughput for the ABBA performance comparison.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(cpu={k:{f:v for f,v in r.items() if f != 'runs'} for k,r in result['cpu'].items()},
                         training=result['training_comparisons'])), flush=True)


if __name__ == '__main__':
    main()
