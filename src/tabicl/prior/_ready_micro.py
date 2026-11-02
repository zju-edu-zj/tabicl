"""Bounded completion-order delivery of intact, independently seeded micros.

Source steps contain equal numbers of units. A unit may be consumed early, but
must be delivered by source_step + max_delay_steps. An unfinished due unit
blocks dispatch: fairness never means dropping, timing out, or resampling it.
"""
from dataclasses import dataclass


@dataclass
class MicroUnit:
    batch: int
    indices: tuple
    task_ids: tuple
    left: int
    completed: int = 0


class ReadyMicros:
    def __init__(self, local_batch, ranks, micro_size, max_delay_steps):
        if (min(local_batch, ranks, micro_size) < 1 or local_batch % micro_size
                or max_delay_steps < 0):
            raise ValueError('ready micros require complete batches and nonnegative delay')
        self.micro_size = micro_size
        self.per_step = local_batch * ranks // micro_size
        self.max_delay = max_delay_steps
        self.units = {}
        self.next_source = self.next_step = 0
        self.max_late_steps = self.max_early_steps = self.forced_waits = 0

    def register(self, batch, task_ids):
        if batch != self.next_source or len(task_ids) != self.per_step * self.micro_size:
            raise ValueError('source steps must be contiguous and complete')
        for start in range(0, len(task_ids), self.micro_size):
            indices = tuple(range(start, start + self.micro_size))
            self.units[batch, start // self.micro_size] = MicroUnit(
                batch, indices, tuple(task_ids[i] for i in indices), self.micro_size)
        self.next_source += 1

    def complete(self, batch, index):
        unit = self.units[batch, index // self.micro_size]
        bit = 1 << (index % self.micro_size)
        if unit.completed & bit:
            raise RuntimeError('duplicate micro completion')
        unit.completed |= bit
        unit.left -= 1

    def pop_step(self):
        # Dictionary insertion order is source order, independent of the order
        # in which futures happen to be returned by wait().
        due = [u for u in self.units.values() if u.batch <= self.next_step - self.max_delay]
        if any(u.left for u in due):
            self.forced_waits += 1
            return None
        if len(due) > self.per_step:
            raise RuntimeError('source-step fairness invariant violated')
        selected = {(u.batch, u.indices[0] // self.micro_size): u for u in due}
        for key, unit in self.units.items():
            if len(selected) == self.per_step:
                break
            if unit.left == 0 and key not in selected:
                selected[key] = unit
        if len(selected) < self.per_step:
            return None
        for key, unit in selected.items():
            del self.units[key]
            self.max_late_steps = max(self.max_late_steps, self.next_step - unit.batch)
            self.max_early_steps = max(self.max_early_steps, unit.batch - self.next_step)
        self.next_step += 1
        return list(selected.values())

    def stats(self):
        return dict(ready_micro_units=sum(u.left == 0 for u in self.units.values()),
                    pending_micro_units=len(self.units), ready_micro_steps=self.next_step,
                    ready_micro_max_late_steps=self.max_late_steps,
                    ready_micro_max_early_steps=self.max_early_steps,
                    ready_micro_fairness_wait_checks=self.forced_waits)

    def wait_state(self):
        """Diagnostic only; called after pop_step failed, under service lock."""
        due = [u for u in self.units.values()
               if u.batch <= self.next_step-self.max_delay and u.left]
        missing = [task for u in due for position, task in enumerate(u.task_ids)
                   if not u.completed & (1 << position)]
        return dict(reason='due_micro' if due else 'insufficient_ready',
                    missing_task_ids=missing,
                    ready_units=sum(u.left == 0 for u in self.units.values()),
                    due_unfinished_units=len(due))
