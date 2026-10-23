"""Node-local, bounded, single-dataset generation shared by all training ranks.

The manager and workers use spawn: training may already have initialized CUDA.
Only NumPy arrays cross the manager connection, avoiding CUDA/fd ownership and
NestedTensor IPC. A batch keeps its original parameter-sharing boundaries.
Completion order is intentionally nondeterministic; no slow tasks are dropped.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from multiprocessing.util import Finalize
import os
import pickle
import random
import tempfile
import threading
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import threadpoolctl

from ._dataset import GraphPrior


_worker_prior = None
_worker_limits = None


def _init_worker(prior):
    global _worker_prior, _worker_limits
    torch.set_num_threads(1)
    _worker_limits = threadpoolctl.threadpool_limits(limits=1)
    _worker_prior = prior


def _generate(params, seed):
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, y, d = _worker_prior.generate_dataset(params, compact=True)
    return x.numpy(), y.numpy(), int(d), time.perf_counter() - started


def _select_admission(admission, reserved, byte_budget, allow_oversized=False):
    estimates = [sum(4 * p["seq_len"] * (p["max_features"] + 1) for p in params)
                 for params in admission]
    for index, size in enumerate(estimates):
        if reserved + size <= byte_budget:
            return index
    return 0 if admission and allow_oversized else None


def iter_micro_batches(batch, size):
    """Pad one micro-batch at a time, for dense, nested, or compact list input."""
    if size < 1:
        raise ValueError("micro batch size must be positive")
    xs, ys, ds, lengths, trains = batch
    for start in range(0, len(ds), size):
        stop = min(start + size, len(ds))
        d, length, train = ds[start:stop], lengths[start:stop], trains[start:stop]
        # Slice before padding; long tables never pad unrelated micro-batches.
        width, rows = int(d.max()), int(length.max())
        if isinstance(xs, torch.Tensor) and not xs.is_nested and not ys.is_nested:
            # Keep pinned storage and avoid another copy on the legacy dense path.
            yield xs[start:stop, :rows, :width], ys[start:stop, :rows], d, length, train
            continue
        xx = [xs[i][:int(lengths[i]), :int(ds[i])] for i in range(start, stop)]
        yy = [ys[i][:int(lengths[i])] for i in range(start, stop)]
        x = torch.stack([F.pad(t, (0, width - t.shape[1], 0, rows - t.shape[0])) for t in xx])
        y = torch.stack([F.pad(t, (0, rows - t.shape[0])) for t in yy])
        yield x, y, d, length, train


class _Service:
    def __init__(self, prior, workers, window, byte_budget, seed, clients):
        self.signature = pickle.dumps((prior, workers, window, byte_budget, seed, clients))
        self.prior, self.window, self.byte_budget = prior, window, byte_budget
        self.seed, self.clients = seed, clients
        self.condition = threading.Condition()
        self.ready = deque()
        self.admission = deque()
        self.pending = {}
        self.batches = {}
        self.reserved = 0
        self.peak_reserved = 0
        self.completed_datasets = 0
        self.delivered_batches = 0
        self.admission_bypasses = 0
        self.admission_blocked = 0
        self.generation_seconds = deque(maxlen=4096)
        self.admission_lookahead = max(4, min(window, workers * 2))
        self.released = set()
        self.closed = False
        self.error = None
        # These are inherited before spawned workers import NumPy/SciPy. Setting
        # thread limits only in the initializer is too late for import-time BLAS.
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
        self.executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"),
            initializer=_init_worker, initargs=(prior,),
        )
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()

    def _produce(self):
        try:
            torch.set_num_threads(1)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            batch_id = task_id = 0
            while True:
                with self.condition:
                    if self.closed:
                        return
                    # Count ready AND in-flight batches against both limits.
                    while len(self.batches) < self.window:
                        if not self.admission:
                            self.admission.append(self.prior.sample_parameters())
                        available = _select_admission(self.admission, self.reserved, self.byte_budget)
                        if available is None and len(self.admission) < self.admission_lookahead:
                            self.admission.append(self.prior.sample_parameters())
                            continue
                        if available is None and not self.batches:
                            # Only start an oversized batch after searching the
                            # bounded lookahead for a batch that fits.
                            available = _select_admission(
                                self.admission, self.reserved, self.byte_budget, allow_oversized=True
                            )
                        if available is None:
                            self.admission_blocked += 1
                            break
                        if available:
                            self.admission_bypasses += 1
                        params = self.admission[available]
                        del self.admission[available]
                        estimate = sum(4 * p["seq_len"] * (p["max_features"] + 1) for p in params)
                        state = dict(params=params, results=[None] * len(params), left=len(params), bytes=estimate)
                        self.batches[batch_id] = state
                        self.reserved += estimate
                        self.peak_reserved = max(self.peak_reserved, self.reserved)
                        for i, p in enumerate(params):
                            seed = int(np.random.SeedSequence([self.seed, task_id]).generate_state(1)[0])
                            future = self.executor.submit(_generate, p, seed)
                            self.pending[future] = (batch_id, i)
                            task_id += 1
                        batch_id += 1
                    if not self.pending:
                        self.condition.wait(timeout=0.1)
                        continue
                done, _ = wait(self.pending, timeout=0.1, return_when=FIRST_COMPLETED)
                with self.condition:
                    for future in done:
                        bid, idx = self.pending.pop(future)
                        state = self.batches[bid]
                        x, y, d, elapsed = future.result()
                        state["results"][idx] = (x, y, d)
                        self.generation_seconds.append(elapsed)
                        state["left"] -= 1
                        self.completed_datasets += 1
                        if state["left"] == 0:
                            self.ready.append(bid)
                    self.condition.notify_all()
        except BaseException:
            with self.condition:
                self.error = traceback.format_exc()
                self.condition.notify_all()

    def get_batch(self):
        with self.condition:
            while not self.ready and not self.error and not self.closed:
                self.condition.wait(timeout=1)
            if self.error:
                raise RuntimeError("node prior worker failed:\n" + self.error)
            if self.closed:
                raise RuntimeError("node prior pool is closed")
            state = self.batches.pop(self.ready.popleft())
            self.reserved -= state["bytes"]
            self.delivered_batches += 1
            self.condition.notify_all()
        xs, ys, ds = zip(*state["results"])
        return xs, ys, np.asarray(ds, dtype=np.int64), np.asarray(
            [p["seq_len"] for p in state["params"]], dtype=np.int64
        ), np.asarray([p["train_size"] for p in state["params"]], dtype=np.int64)

    def stats(self):
        with self.condition:
            durations = np.asarray(self.generation_seconds, dtype=np.float64)
            task_latency = ({f"generation_{key}_ms": float(np.quantile(durations, q) * 1000)
                             for key, q in (("p50", .50), ("p95", .95), ("p99", .99))}
                            if len(durations) else {})
            return dict(completed_datasets=self.completed_datasets, delivered_batches=self.delivered_batches,
                        buffered_batches=len(self.batches), ready_batches=len(self.ready),
                        reserved_bytes=self.reserved, peak_reserved_bytes=self.peak_reserved,
                        admission_batches=len(self.admission), admission_bypasses=self.admission_bypasses,
                        admission_blocked=self.admission_blocked, **task_latency,
                        worker_pids=list(self.executor._processes or {}))

    def release(self, rank):
        with self.condition:
            self.released.add(rank)
            self.condition.notify_all()

    def await_release(self, timeout):
        with self.condition:
            return self.condition.wait_for(lambda: len(self.released) >= self.clients, timeout=timeout)

    def stop(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=2)
        # Do not wait for a pathological graph during shutdown/error cleanup.
        processes = list((self.executor._processes or {}).values())
        self.executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()


_service = None
_service_lock = threading.Lock()


def _get_service(*args):
    global _service
    with _service_lock:
        signature = pickle.dumps(args)
        if _service is None:
            _service = _Service(*args)
            # Manager.shutdown and parent finalizers must also reap workers.
            Finalize(None, _service.stop, exitpriority=10)
        elif _service.signature != signature:
            raise ValueError("ranks on this node have inconsistent prior pool settings")
        return _service


class _Manager(BaseManager):
    pass


_Manager.register("service", callable=_get_service)


class NodePriorPool:
    """Iterable training loader; ``workers`` is node-wide, never per rank.

    Queue bytes bound reserved final outputs, not GraphSCM/ExtraTrees transient
    working memory or active RPC copies. For long tables also limit worker count.
    Use torchrun's node-local rank/world environment, or an explicit unique key.
    """
    def __init__(self, prior, workers, window=128, byte_budget=2 * 1024**3,
                 seed=42, key=None, local_rank=None, local_world_size=None, timeout=180):
        if not isinstance(prior, GraphPrior) or str(prior.device) != "cpu":
            raise ValueError("node_pool requires online graph_scm on CPU")
        if min(workers, window, byte_budget) < 1:
            raise ValueError("pool workers, window and byte budget must be positive")
        self.dataset = prior
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if local_rank is None else local_rank
        self.local_world = int(os.environ.get("LOCAL_WORLD_SIZE", 1)) if local_world_size is None else local_world_size
        run_key = key or (os.environ.get("TORCHELASTIC_RUN_ID", "") + ":" + os.environ.get("MASTER_PORT", ""))
        if not key and self.local_world == 1 and not os.environ.get("TORCHELASTIC_RUN_ID"):
            run_key += ":" + str(os.getpid())
        digest = hashlib.sha256(run_key.encode()).hexdigest()[:24]
        self.address = os.path.join(tempfile.gettempdir(), f"tabicl-prior-{digest}.sock")
        self.authkey = digest.encode()
        self.args = (prior, workers, window, byte_budget, seed, self.local_world)
        self.timeout = timeout
        self.manager = self.service = None

    def __iter__(self):
        if self.service is not None:
            return self
        self.manager = _Manager(address=self.address, authkey=self.authkey, ctx=mp.get_context("spawn"))
        if self.local_rank == 0:
            # Refuse collisions instead of unlinking another running job's socket.
            self.manager.start()
        else:
            deadline = time.monotonic() + self.timeout
            while True:
                try:
                    self.manager.connect()
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("node prior leader did not start")
                    time.sleep(0.1)
        try:
            self.service = self.manager.service(*self.args)
        except BaseException:
            if self.local_rank == 0:
                self.manager.shutdown()
            raise
        return self

    def __next__(self):
        if self.service is None:
            iter(self)
        xs, ys, ds, lengths, trains = self.service.get_batch()
        return ([torch.from_numpy(x) for x in xs], [torch.from_numpy(y) for y in ys],
                torch.from_numpy(ds), torch.from_numpy(lengths), torch.from_numpy(trains))

    def stats(self):
        return self.service.stats() if self.service is not None else {}

    def set_start_step(self, step):
        """Use a fresh stream on resume; exact task/optimizer-step replay is unsupported."""
        if self.service is not None:
            raise RuntimeError("cannot reseed an active node pool")
        self.args = (*self.args[:4], self.args[4] + int(step), self.args[5])

    def close(self):
        if self.service is None:
            return
        try:
            self.service.release(self.local_rank)
            if self.local_rank == 0:
                self.service.await_release(15)
                self.service.stop()
        except (EOFError, BrokenPipeError, ConnectionError):
            pass
        finally:
            if self.local_rank == 0:
                self.manager.shutdown()
            self.service = self.manager = None

    def __enter__(self):
        return iter(self)

    def __exit__(self, *exc):
        self.close()
