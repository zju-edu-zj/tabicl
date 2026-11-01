"""Completion-order delivery: intact units, finite lateness, ownership and errors."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import random
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from tabicl.prior._ready_micro import ReadyMicros
from tabicl.prior._node_pool import NodePriorPool, _Service, _output_bytes
from test_node_prior_pool import small_prior
from test_prior_scheduling import digest_batch


def register(policy, batches):
    size = policy.per_step * policy.micro_size
    for b in range(batches):
        policy.register(b, list(range(b * size, (b + 1) * size)))


def test_bypass_straggler_then_force_its_deadline_without_discard():
    p = ReadyMicros(4, 2, 2, 2)
    register(p, 4)
    for b in range(4):
        for i in range(8):
            if (b, i) != (0, 0):
                p.complete(b, i)
    steps = [p.pop_step(), p.pop_step()]
    assert all(u.left == 0 for step in steps for u in step)
    assert all(0 not in u.task_ids for step in steps for u in step)
    before = list(p.units)
    assert p.pop_step() is None  # source 0 is due, even though later units are ready
    assert list(p.units) == before
    p.complete(0, 0)
    steps += [p.pop_step(), p.pop_step()]
    assert sorted(i for step in steps for u in step for i in u.task_ids) == list(range(32))
    assert p.stats()['ready_micro_max_late_steps'] == 2
    assert not p.units


@pytest.mark.parametrize('seed', range(30))
@pytest.mark.parametrize('delay', [0, 1, 2, 5])
def test_random_completion_preserves_every_task_and_micro(seed, delay):
    p = ReadyMicros(8, 8, 4, delay)
    register(p, 15)
    tasks = [(b, i) for b in range(15) for i in range(64)]
    random.Random(seed).shuffle(tasks)
    seen = []
    for b, i in tasks:
        p.complete(b, i)
        while (units := p.pop_step()) is not None:
            step = p.next_step - 1
            assert len(units) == 16
            for u in units:
                assert step - u.batch <= delay
                assert u.task_ids == tuple(range(u.task_ids[0], u.task_ids[0] + 4))
                seen.extend(u.task_ids)
    assert sorted(seen) == list(range(15 * 64))
    assert not p.units


def test_duplicate_completion_and_invalid_registration():
    p = ReadyMicros(4, 1, 2, 1)
    with pytest.raises(ValueError):
        p.register(1, [0, 1, 2, 3])
    register(p, 1)
    p.complete(0, 0)
    with pytest.raises(RuntimeError, match='duplicate'):
        p.complete(0, 0)
    with pytest.raises(ValueError):
        ReadyMicros(3, 2, 2, 1)


def fake_service():
    """No processes/timers: test manager ownership with rank 0 ACK before rank 1 fetch."""
    s = _Service.__new__(_Service)
    s.condition = threading.Condition()
    s.ready_micros = ReadyMicros(2, 2, 2, 1)
    register(s.ready_micros, 2)
    s.prior, s.clients, s.micro_size, s.cost_model = small_prior(batch_size=2), 2, 2, {}
    s.micro_steps, s.leases, s.batches = {}, {}, {}
    s.rank_sequences, s.released = [0, 0], set()
    s.error, s.closed, s.reserved = None, False, 0
    s.delivered_batches, s.fetch_wait_seconds = 0, 0.
    s.output = Mock()
    for b in range(2):
        params = [dict(seq_len=32, train_size=16, num_features=2, max_features=2)] * 4
        results = [(np.ones((32, 2)), np.ones(32), 2) for _ in params]
        times = [dict(generation_started_monotonic_s=1., generated_monotonic_s=2.,
                      encoded_monotonic_s=3., process_cpu_s=.1)] * 4
        s.batches[b] = dict(params=params, results=results, task_timings=times,
                            task_ids=list(range(b * 4, b * 4 + 4)), bytes=_output_bytes(params),
                            submitted_monotonic_s=0.)
        s.reserved += _output_bytes(params)
        for i in range(4):
            s.ready_micros.complete(b, i)
    return s


def test_partial_ack_reclaims_only_owned_tables_and_allows_lagging_rank():
    s = fake_service()
    initial = s.reserved
    _, t0 = s.get_batch(0, 0, True)
    s.acknowledge(0, 0)
    assert s.reserved == initial - _output_bytes([s.batches[0]['params'][0]] * 2)
    _, t1 = s.get_batch(1, 0, True)
    assert set(t0['task_ids']).isdisjoint(t1['task_ids'])
    s.acknowledge(1, 0)
    assert 0 not in s.batches and 0 not in s.micro_steps
    for rank in (1, 0):
        s.get_batch(rank, 1)
        s.acknowledge(rank, 1)
    assert s.reserved == 0 and not s.batches and not s.leases
    assert s.output.release.call_count == 4
    with pytest.raises(ValueError, match='next sequence'):
        s.get_batch(0, 1)


def test_source_deleted_before_lagging_rank_fetch_uses_cached_dependency_time():
    s = fake_service()
    # First step bypasses source 0's second unit, taking source 1's first unit.
    u = s.ready_micros.units[0, 1]
    u.left, u.completed = 1, 2
    for rank in (0, 1):
        s.get_batch(rank, 0)
        s.acknowledge(rank, 0)
    s.ready_micros.complete(0, 2)
    s.get_batch(0, 1)
    s.acknowledge(0, 1)
    assert 0 not in s.batches
    _, timing = s.get_batch(1, 1, True)
    assert timing['dispatch_generation_ready_monotonic_s'] == 2.
    s.acknowledge(1, 1)
    assert s.reserved == 0


@pytest.mark.parametrize('failure', ['error', 'closed', 'released'])
def test_failed_or_released_rank_never_delivers_cached_step(failure):
    s = fake_service()
    s.get_batch(0, 0)
    if failure == 'error':
        s.error = 'injected failure'
    elif failure == 'closed':
        s.closed = True
    else:
        s.release(1)
    with pytest.raises(RuntimeError):
        s.get_batch(1, 0)


@pytest.mark.parametrize('transport', ['numpy', 'mmap'])
@pytest.mark.parametrize('byte_budget', [1, 1024 ** 2])
def test_real_shared_pool_ready_micros(tmp_path, transport, byte_budget):
    prior = small_prior()
    with ExitStack() as stack:
        pools = [stack.enter_context(NodePriorPool(prior, workers=2, window=8,
            byte_budget=byte_budget, key=str(tmp_path), local_rank=r, local_world_size=2,
            schedule='ready_micro', micro_size=2, ready_max_delay=2, transport=transport)) for r in range(2)]
        seen = set()
        with ThreadPoolExecutor(2) as clients:
            for step in range(6):
                batches = list(clients.map(next, pools))
                for pool, batch in zip(pools, batches):
                    t = pool.last_batch_timing
                    assert t['dispatch_step'] == step
                    assert t['dispatch_generation_ready_monotonic_s'] >= t['generation_ready_monotonic_s']
                    assert all(step - b <= 2 for b in t['source_steps'])
                    for i in range(0, 4, 2):
                        ids = t['task_ids'][i:i+2]
                        assert ids == list(range(ids[0], ids[0]+2)) and ids[0] % 2 == 0
                    assert not seen.intersection(t['task_ids'])
                    seen.update(t['task_ids'])
                    assert len(batch[0]) == 4
                    assert all(x.shape == (int(n), int(d)) for x, d, n in zip(batch[0], batch[2], batch[3]))
        # Every source that has reached its deadline is present, including slow tasks.
        assert set(range((6-2)*8)) <= seen
        stats = pools[0].stats()
        assert stats['ready_micro_max_late_steps'] <= 2
        assert stats['buffered_batches'] <= 4
        assert stats['active_leases'] == 0


def test_ready_settings_and_resume():
    # The service permits one oversized source cohort for a sub-cohort window;
    # delay zero must remain usable there, just as with legacy scheduling.
    NodePriorPool(small_prior(), 1, window=1, local_world_size=2,
                  schedule='ready_micro', micro_size=2, ready_max_delay=0)
    with pytest.raises(ValueError, match='cohort window'):
        NodePriorPool(small_prior(), 1, window=4, local_world_size=2,
                      schedule='ready_micro', micro_size=2, ready_max_delay=2)
    pool = NodePriorPool(small_prior(), 1, window=8, local_world_size=2,
                         schedule='ready_micro', micro_size=2, ready_max_delay=2)
    pool.set_start_step(100)
    assert pool.args[4] == 142 and pool.args[-2:] == (100, 2)


def test_zero_delay_preserves_baseline_generated_task_multiset(tmp_path):
    streams = []
    for schedule in ('balanced', 'ready_micro'):
        with NodePriorPool(small_prior(), 2, window=3, key=str(tmp_path / schedule),
                           schedule=schedule, micro_size=2, ready_max_delay=0) as pool:
            streams.append([sorted(digest_batch(next(pool))) for _ in range(4)])
    assert streams[0] == streams[1]


def test_ready_propagates_real_worker_failure(tmp_path):
    from tabicl.prior.graph_lib._config import PriorConfig
    with NodePriorPool(small_prior(config=PriorConfig(fct_types='invalid')), 1, window=3,
                       key=str(tmp_path), schedule='ready_micro', micro_size=2) as pool:
        with pytest.raises(RuntimeError, match='node prior worker failed'):
            next(pool)
