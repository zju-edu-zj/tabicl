"""Join bounded worker/service JSONL traces and attribute actual wait intervals.

Wait events from concurrent rank RPCs overlap: report their UNION, never the
sum of rank waits. Missing tasks may be running or queued during one interval.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


def union_seconds(intervals):
    end, total = -float('inf'), 0.
    for start, stop in sorted(intervals):
        total += max(0., stop-max(end, start))
        end = max(end, stop)
    return total


def quantiles(values):
    return {name:float(np.quantile(values, q)) for name, q in
            (('p50', .5), ('p95', .95), ('p99', .99), ('max', 1.))} if values else {}


def analyze(directory, skip_steps=0):
    tasks, completions, waits, dispatches = {}, {}, [], []
    paths = sorted(directory.glob('*.jsonl'))
    if not paths:
        raise FileNotFoundError(f'No JSONL traces in {directory}')
    for path in paths:
        with path.open() as file:
            for line in file:
                record = json.loads(line)
                if record['event'] == 'task':
                    if record['task_id'] in tasks:
                        raise ValueError('Duplicate task_id: analyze only one isolated node/run directory')
                    tasks[record['task_id']] = record
                elif record['event'] == 'complete':
                    if record['task_id'] in completions:
                        raise ValueError('Duplicate completion: analyze only one isolated node/run directory')
                    completions[record['task_id']] = record
                elif record['event'] == 'wait':
                    waits.append(record)
                elif record['event'] == 'dispatch':
                    dispatches.append(record)
    dispatches = [r for r in dispatches if r['step'] >= skip_steps]
    selected = {identity for r in dispatches for identity in r['task_ids']}
    if skip_steps:
        waits = [r for r in waits if r['step'] >= skip_steps]
        tasks = {k:v for k,v in tasks.items() if k in selected}
        completions = {k:v for k,v in completions.items() if k in selected}
    component, stages, rejected_stages = Counter(), Counter(), Counter()
    for task in tasks.values():
        for name, value in task['components'].items():
            component[name] += value['exclusive_s']
        for attempt in task['attempts']:
            for name in ('graph_s', 'sanity_s', 'filter_s'):
                stages[name] += attempt.get(name, 0.)
                if attempt['result'] != 'accepted':
                    rejected_stages[name] += attempt.get(name, 0.)
    spans, queued, running, transit = defaultdict(list), [], [], []
    blockers = Counter()
    for event in waits:
        start, end = event['started_s'], event['finished_s']
        spans[event['reason']].append((start, end))
        if event['reason'] != 'due_micro':
            continue
        for identity in event.get('missing_task_ids', []):
            blockers[identity] += end-start  # ranking only, NOT a wall-time total
            task = completions.get(identity)
            if task is not None:
                began, ended = task['started_monotonic_s'], task['encoded_monotonic_s']
                if start < min(end, began):
                    queued.append((start, min(end, began)))
                if max(start, began) < min(end, ended):
                    running.append((max(start, began), min(end, ended)))
                received = task['received_s']
                if max(start, ended) < min(end, received):
                    transit.append((max(start, ended), min(end, received)))
    slow = sorted(tasks.values(), key=lambda x:x['wall_s'], reverse=True)[:24]
    return dict(tasks=len(tasks), completions=len(completions), dispatched_steps=len(dispatches), skip_steps=skip_steps,
        task_wall_s=quantiles([r['wall_s'] for r in tasks.values()]),
        task_cpu_s=quantiles([r['cpu_s'] for r in tasks.values()]),
        queue_s=quantiles([r['started_monotonic_s']-r['submitted_s'] for r in completions.values()]),
        transit_s=quantiles([r['received_s']-r['encoded_monotonic_s'] for r in completions.values()]),
        wait_union_s={reason:union_seconds(intervals) for reason, intervals in spans.items()},
        due_wait_with_queued_blocker_union_s=union_seconds(queued),
        due_wait_with_running_blocker_union_s=union_seconds(running),
        due_wait_with_transit_blocker_union_s=union_seconds(transit),
        due_wait_with_known_blocker_union_s=union_seconds(queued+running+transit),
        retry_results=dict(Counter(a['result'] for r in tasks.values() for a in r['attempts'])),
        attempt_stage_wall_s=dict(stages),
        rejected_attempt_stage_wall_s=dict(rejected_stages),
        attempts_per_task=quantiles([len(r['attempts']) for r in tasks.values()]),
        component_exclusive_s=dict(component.most_common()),
        slow_tasks=slow,
        due_blockers=[dict(task_id=identity, rank_wait_s=duration, task=tasks.get(identity),
                          timing=completions.get(identity)) for identity, duration in blockers.most_common(24)],
        note='Queued/running/transit unions can overlap across different tasks; do not add them. '
             'Task/queue statistics select tasks by dispatched step when skip_steps>0. '
             'Component times are traced wall time, not process CPU.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--skip-steps', type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.skip_steps < 0:
        p.error('--skip-steps must be nonnegative')
    result = analyze(args.directory, args.skip_steps)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k, v in result.items() if k not in ('slow_tasks', 'due_blockers')}))


if __name__ == '__main__':
    main()
