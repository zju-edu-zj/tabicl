"""CPU permits bound all generating threads; failures return every permit."""
import multiprocessing as mp

import pytest
import torch

from tabicl.prior.graph_lib import _cpu_budget
from tabicl.prior.graph_lib._distance import nearest_center


@pytest.fixture(autouse=True)
def single_worker_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("slots,cap", [(1, 8), (3, 8), (8, 2)])
def test_borrow_only_free_permits_and_restore_after_failure(slots, cap):
    permits = mp.get_context("spawn").BoundedSemaphore(slots)
    worker = _cpu_budget.WorkerBudget(permits, cap)
    before = torch.get_num_threads()
    with pytest.raises(ValueError, match="kernel failure"):
        with worker.generation(), worker.distance():
            assert torch.get_num_threads() == min(slots, cap)
            raise ValueError("kernel failure")
    assert torch.get_num_threads() == before
    assert not worker.active
    assert sum(permits.acquire(False) for _ in range(slots+1)) == slots
    for _ in range(slots):
        permits.release()


def test_two_workers_share_base_and_borrowed_permits():
    permits = mp.get_context("spawn").BoundedSemaphore(4)
    a, b = [_cpu_budget.WorkerBudget(permits, 8) for _ in range(2)]
    with a.generation(), b.generation(), a.distance():
        assert torch.get_num_threads() == 3  # one base + two spare, b owns one
        assert not permits.acquire(False)
    assert permits.acquire(False)
    permits.release()


@pytest.mark.parametrize("p", [.5, .712428, 1., 1.7, 3.5, 4.])
def test_parallel_nearest_center_matches_single_thread_exactly(p):
    generator = torch.Generator().manual_seed(41)
    x, centers = [torch.randn(shape, generator=generator) for shape in [(2000, 32), (160, 32)]]
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        expected = nearest_center(x, centers, p)
        permits = mp.get_context("spawn").BoundedSemaphore(4)
        _cpu_budget.configure_worker_budget(permits, 4)
        with _cpu_budget.generation_slot():
            actual = nearest_center(x, centers, p)
            assert _cpu_budget.worker_budget.stats["parallel_distance_max_threads"] == 4
        assert torch.equal(actual, expected)
        assert torch.get_num_threads() == 1
    finally:
        _cpu_budget.configure_worker_budget()
        torch.set_num_threads(before)


def test_default_small_and_euclidean_paths_do_not_borrow():
    permits = mp.get_context("spawn").BoundedSemaphore(4)
    _cpu_budget.configure_worker_budget(permits, 4)
    try:
        with _cpu_budget.generation_slot():
            nearest_center(torch.ones(20, 2), torch.ones(4, 2), .7)
            nearest_center(torch.ones(2000, 32), torch.ones(160, 32), 2.)
            assert _cpu_budget.worker_budget.stats["parallel_distance_calls"] == 0
        assert sum(permits.acquire(False) for _ in range(5)) == 4
    finally:
        _cpu_budget.configure_worker_budget()
