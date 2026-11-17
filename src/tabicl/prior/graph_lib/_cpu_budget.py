"""Optional node-wide CPU permits for independent nearest-center distances.

Each generating worker holds one permit. A large distance kernel can borrow
currently free permits without waiting; all other graph/filter work stays
single-threaded. Permits include base worker threads, preventing a second,
unaccounted thread pool. The manager passes a spawn-compatible semaphore.
"""
from contextlib import contextmanager
import time

import torch


class WorkerBudget:
    def __init__(self, permits, max_threads):
        self.permits, self.max_threads = permits, max_threads
        self.active = False
        self.reset_stats()

    def reset_stats(self):
        self.stats = dict(parallel_distance_calls=0, parallel_distance_max_threads=1,
                          parallel_distance_extra_thread_seconds=0.)

    @contextmanager
    def generation(self):
        if self.active:
            raise RuntimeError("nested worker generation is unsupported")
        self.permits.acquire()
        self.active = True
        self.reset_stats()
        try:
            yield
        finally:
            self.active = False
            self.permits.release()

    @contextmanager
    def distance(self):
        # Outside a pool generation there is no owned base permit.
        if not self.active:
            yield
            return
        borrowed = 0
        before = torch.get_num_threads()
        started = time.perf_counter()
        try:
            while borrowed < self.max_threads-1 and self.permits.acquire(False):
                borrowed += 1
            if borrowed:
                torch.set_num_threads(1+borrowed)
                self.stats["parallel_distance_calls"] += 1
                self.stats["parallel_distance_max_threads"] = max(
                    self.stats["parallel_distance_max_threads"], 1+borrowed)
            yield
        finally:
            try:
                if borrowed:
                    torch.set_num_threads(before)
            finally:
                self.stats["parallel_distance_extra_thread_seconds"] += borrowed * (time.perf_counter()-started)
                for _ in range(borrowed):
                    self.permits.release()


worker_budget = None


def configure_worker_budget(permits=None, max_threads=1):
    global worker_budget
    worker_budget = WorkerBudget(permits, max_threads) if permits is not None and max_threads > 1 else None


@contextmanager
def generation_slot():
    if worker_budget is None:
        yield
    else:
        with worker_budget.generation():
            yield


@contextmanager
def parallel_distance(x, centers, p):
    # Point distances are independent; retain p=2 GEMM. The sorted finite
    # 1-D path never enters this context; its nonfinite fallback can borrow.
    # A workload threshold avoids threading overhead for small kernels.
    if (worker_budget is None or x.device.type != "cpu" or p == 2 or
            x.numel()*len(centers) < 10_000_000):
        yield
    else:
        with worker_budget.distance():
            yield
