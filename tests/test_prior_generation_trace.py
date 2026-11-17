import json
import random

import numpy as np
import pytest
import torch

from tabicl.prior import _generation_trace as diagnostics
from tabicl.prior._dataset import GraphPrior
from tabicl.prior._ready_micro import ReadyMicros
from tabicl.prior.graph_lib._config import PriorConfig


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def test_generation_trace_preserves_output_and_records_attempt_phases(tmp_path):
    torch.set_num_threads(1)
    prior = GraphPrior(batch_size=4, batch_size_per_gp=4, n_jobs=1,
        min_features=3, max_features=6, max_classes=3, max_seq_len=64,
        min_train_size=.4, max_train_size=.6, config=PriorConfig(min_n_nodes=2, max_n_nodes=4))
    seed(33)
    param = prior.sample_parameters()[0]
    seed(34)
    expected = prior.generate_dataset(param, compact=True)
    writer = diagnostics.Writer(tmp_path, 'test', 1)
    seed(34)
    with diagnostics.TaskTrace(writer, dict(task_id=17, seed=34)) as trace:
        actual = prior.generate_dataset(param, compact=True, trace=trace)
    for reference, result in zip(expected, actual):
        torch.testing.assert_close(reference, result, rtol=0, atol=0)
    record = json.loads(next(tmp_path.glob('*.jsonl')).read_text())
    assert record['task_id'] == 17 and record['wall_s'] >= record['cpu_s']*.5 > 0
    assert record['attempts'][-1]['result'] == 'accepted'
    for attempt in record['attempts']:
        assert attempt['graph_s'] > 0 and attempt['sanity_s'] >= 0
    assert sum(sum(v for k, v in a.items() if k.endswith('_s')) for a in record['attempts']) <= record['wall_s']
    writer.write(dict(event='should_not_fit'))
    assert len(next(tmp_path.glob('*.jsonl')).read_text().splitlines()) == 1
    writer.file.close()


def test_nested_component_exclusive_accounting_and_exception_cleanup(tmp_path):
    writer = diagnostics.Writer(tmp_path, 'nested', 2)
    inner = diagnostics._timed(lambda: sum(range(1000)), 'inner')
    outer = diagnostics._timed(lambda: [inner() for _ in range(3)], 'outer')
    with diagnostics.TaskTrace(writer, {}) as trace:
        outer()
    assert trace.components['inner'][0] == 3
    assert trace.components['outer'][0] == 1
    assert trace.components['outer'][1] >= trace.components['inner'][1]
    assert trace.components['outer'][2] == pytest.approx(
        trace.components['outer'][1]-trace.components['inner'][1], abs=1e-12)
    with pytest.raises(ValueError):
        with diagnostics.TaskTrace(writer, {}) as trace:
            trace.begin_attempt()
            raise ValueError('test')
    assert diagnostics._current.get() is None
    records = [json.loads(s) for s in next(tmp_path.glob('*.jsonl')).read_text().splitlines()]
    assert records[-1]['attempts'][0]['result'] == 'error'
    writer.file.close()


def test_trace_wait_reason_identifies_missing_tasks_without_changing_fairness():
    ready = ReadyMicros(4, 2, 2, 0)
    ready.register(0, list(range(8)))
    for index in range(7):
        ready.complete(0, index)
    assert ready.pop_step() is None
    reason = ready.wait_state()
    assert reason == dict(reason='due_micro', missing_task_ids=[7], ready_units=3, due_unfinished_units=1)
    ready.complete(0, 7)
    assert len(ready.pop_step()) == 4
    assert not ready.units
    assert ready.wait_state()['reason'] == 'insufficient_ready'


def test_trace_is_opt_in_and_does_not_wrap_components_by_default(monkeypatch):
    monkeypatch.delenv('TABICL_PRIOR_TRACE_DIR', raising=False)
    assert diagnostics.worker_writer() is None


def test_spawned_pool_trace_task_identity_and_dispatch(tmp_path, monkeypatch):
    from tabicl.prior._node_pool import NodePriorPool
    directory = tmp_path/'trace'
    monkeypatch.setenv('TABICL_PRIOR_TRACE_DIR', str(directory))
    monkeypatch.setenv('TABICL_PRIOR_TRACE_TASKS_PER_WORKER', '4')
    prior = GraphPrior(batch_size=4, batch_size_per_gp=4, n_jobs=1,
        min_features=2, max_features=4, max_classes=3, max_seq_len=32,
        min_train_size=.4, max_train_size=.6, config=PriorConfig(min_n_nodes=2, max_n_nodes=4))
    with NodePriorPool(prior, workers=2, window=3, schedule='ready_micro', micro_size=2,
                       ready_max_delay=1, key=str(tmp_path), local_rank=0, local_world_size=1) as pool:
        next(pool)
        next(pool)
    tasks = [json.loads(line) for path in directory.glob('worker-*.jsonl')
             for line in path.read_text().splitlines()]
    events = [json.loads(line) for path in directory.glob('service-*.jsonl')
              for line in path.read_text().splitlines()]
    assert tasks and len(tasks) <= 8
    assert len({r['task_id'] for r in tasks}) == len(tasks)
    assert any(r['components'] for r in tasks)
    assert all(r['attempts'][-1]['result'] == 'accepted' for r in tasks)
    assert len([r for r in events if r['event']=='dispatch']) >= 2
    assert all(r['received_s'] >= r['generated_monotonic_s'] >= r['submitted_s']
               for r in events if r['event']=='complete')
