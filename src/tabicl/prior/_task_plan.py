"""Metadata-only finite-window planner. Never selects by completion time.

Identity and random seed precede reordering. Each disjoint window is drained
exactly once; every task's optimizer-step displacement is at most window-1.
Only parameters are buffered here: generation/output memory admission is a
separate concern. A micro-batch stays intact; cross-micro group correlations
are retained statistically, but may now span optimizer steps.
"""
from collections import deque
from dataclasses import dataclass
import time

import numpy as np

from ._scheduling import micro_cost, balanced_partition


def bucket_window(params, micro_size, local_batch, ranks, model):
    """Return complete step index lists, plus prediction-only diagnostics.

    Similar-cost micro-batches form a step; ranks within it are balanced later.
    Retain baseline if total predicted makespan increases. This is a heuristic,
    not an assertion about observed latency or model quality.
    """
    step_size = local_batch * ranks
    if (min(micro_size, local_batch, ranks) < 1 or not params or
            local_batch % micro_size or len(params) % step_size):
        raise ValueError("window requires complete equal-cardinality steps")
    units = [list(range(i, i + micro_size)) for i in range(0, len(params), micro_size)]
    requested = [(None, None, p["num_features"]) for p in params]
    costs = [micro_cost(params, requested, u, model) for u in units]
    per_step = step_size // micro_size
    baseline = [list(range(i, i + per_step)) for i in range(0, len(units), per_step)]
    ordered = sorted(range(len(units)), key=lambda i: (costs[i], i))
    candidate = [ordered[i:i + per_step] for i in range(0, len(ordered), per_step)]

    def total(groups):
        return sum(max(sum(costs[g[i]] for i in indices)
                       for indices in balanced_partition([costs[u] for u in g], ranks)) for g in groups)

    before, after = total(baseline), total(candidate)
    if after > before:
        candidate, after = baseline, before
    return [[i for u in g for i in units[u]] for g in candidate], {
        "window_predicted_before": before, "window_predicted_after": after,
        "window_predicted_gain": before / max(after, 1.),
    }


@dataclass
class PlannedBatch:
    params: list
    task_ids: list


class TaskPlanner:
    def __init__(self, prior, ranks, micro_size, model, seed, schedule, bucket_steps=4):
        if min(ranks, micro_size, bucket_steps) < 1:
            raise ValueError("planner sizes must be positive")
        self.prior, self.ranks, self.micro_size = prior, ranks, micro_size
        self.model, self.seed, self.schedule = model, seed, schedule
        self.bucket_steps = bucket_steps
        self.next_task = self.next_window = 0
        self.ready = deque()
        self.stats = {}

    def next(self):
        if not self.ready:
            started = time.perf_counter()
            count = self.bucket_steps if self.schedule.startswith("window_") else 1
            params = []
            for _ in range(count):
                for _ in range(self.ranks if self.schedule != "none" else 1):
                    params.extend(self.prior.sample_parameters())
            ids = list(range(self.next_task, self.next_task + len(params)))
            self.next_task += len(params)
            step_size = self.prior.batch_size * (self.ranks if self.schedule != "none" else 1)
            groups = [list(range(i, i + step_size)) for i in range(0, len(params), step_size)]
            if self.schedule == "window_bucketed":
                groups, self.stats = bucket_window(params, self.micro_size, self.prior.batch_size,
                                                   self.ranks, self.model)
                # Avoid an always-short-to-long periodic curriculum. This RNG
                # is separate from parameter sampling and is timing-independent.
                # Shared ordinal permutation across nodes: node-specific task
                # seeds must not put their cheap/expensive quantiles in opposite
                # steps. This aligns quantiles, not exact cross-node loads.
                rng = np.random.default_rng(np.random.SeedSequence([0x71C1, self.next_window]))
                groups = [groups[i] for i in rng.permutation(len(groups))]
            self.next_window += 1
            self.stats.update(planned_windows=self.next_window, sampled_tasks=self.next_task,
                              metadata_tasks=len(params), window_steps=count,
                              window_planning_ms=(time.perf_counter()-started)*1000)
            self.ready.extend(PlannedBatch([params[i] for i in g], [ids[i] for i in g]) for g in groups)
        return self.ready.popleft()
