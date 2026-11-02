"""Opt-in, bounded task diagnostics. Disabled runs install no component wrappers.

One JSONL file per process; monotonic clocks share the node's time base.
Component exclusive times avoid adding nested inclusive times more than once.
Only shapes/counters/times are written, never generated feature or label values.
"""
import atexit
from collections import defaultdict
from contextvars import ContextVar
from functools import wraps
import json
import os
from pathlib import Path
import time


_current = ContextVar('tabicl_prior_task_trace', default=None)
_installed = False


class Writer:
    def __init__(self, directory, kind, limit):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.file = (directory/f'{kind}-{os.getpid()}.jsonl').open('x', buffering=1)
        self.limit, self.count = limit, 0
        atexit.register(self.file.close)

    def write(self, record):
        if self.count < self.limit:
            self.file.write(json.dumps(record, allow_nan=False)+'\n')
            self.count += 1


class TaskTrace:
    def __init__(self, writer, identity):
        self.writer, self.identity = writer, identity
        self.attempts, self.stack = [], []
        self.components = defaultdict(lambda: [0, 0., 0.])
        self.active, self.stage = None, None

    def __enter__(self):
        self.started, self.cpu = time.perf_counter(), time.process_time()
        self.token = _current.set(self)
        return self

    def __exit__(self, kind, value, traceback):
        finished, cpu = time.perf_counter(), time.process_time()
        _current.reset(self.token)
        if self.active is not None:
            self.end_attempt('error' if kind else 'accepted')
        self.writer.write(dict(event='task', pid=os.getpid(), **self.identity,
            started_s=self.started, finished_s=finished, wall_s=finished-self.started,
            cpu_s=cpu-self.cpu, error=kind.__name__ if kind else None,
            attempts=self.attempts,
            components={name:dict(calls=v[0], inclusive_s=v[1], exclusive_s=v[2])
                        for name, v in self.components.items()}))

    def begin_attempt(self):
        self.active = {}
        self.stage, self.last = 'graph', time.perf_counter()

    def mark(self, stage):
        now = time.perf_counter()
        if self.stage:
            self.active[self.stage+'_s'] = self.active.get(self.stage+'_s', 0.)+now-self.last
        self.stage, self.last = stage, now

    def end_attempt(self, reason):
        self.mark(None)
        self.active['result'] = reason
        self.attempts.append(self.active)
        self.active = None


def _timed(function, label):
    @wraps(function)
    def call(*args, **kwargs):
        task = _current.get()
        if task is None:
            return function(*args, **kwargs)
        name = label or type(args[0]).__name__
        started, frame = time.perf_counter(), [0.]
        task.stack.append(frame)
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = time.perf_counter()-started
            task.stack.pop()
            if task.stack:
                task.stack[-1][0] += elapsed
            stat = task.components[name]
            stat[0] += 1
            stat[1] += elapsed
            stat[2] += elapsed-frame[0]
    return call


def install_component_trace():
    global _installed
    if _installed:
        return
    from .graph_lib import _base, _function, _converter
    _base.RandomTransformer.__call__ = _timed(_base.RandomTransformer.__call__, None)
    # These kernels are nested in transformer calls. Record them separately
    # and subtract their elapsed time from the parent's exclusive total.
    for module in (_function, _converter):
        for name in ('nearest_center', 'compact_em_probabilities'):
            if hasattr(module, name):
                setattr(module, name, _timed(getattr(module, name), name))
    _installed = True


def worker_writer():
    directory = os.environ.get('TABICL_PRIOR_TRACE_DIR')
    if not directory:
        return None
    limit = int(os.environ.get('TABICL_PRIOR_TRACE_TASKS_PER_WORKER', '2048'))
    if limit < 1:
        raise ValueError('TABICL_PRIOR_TRACE_TASKS_PER_WORKER must be positive')
    install_component_trace()
    return Writer(directory, 'worker', limit)
