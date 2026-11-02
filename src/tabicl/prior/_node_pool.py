"""Node-local, bounded, single-dataset generation shared by all training ranks.

The manager and workers use spawn: training may already have initialized CUDA.
Outputs cross the manager connection as NumPy arrays or tmpfs path descriptors,
avoiding CUDA/fd ownership and NestedTensor IPC.
Unscheduled batches use completion order. Balanced cohorts and bounded ready
micros are addressed by rank and step, so clients cannot steal assignments.
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
from ._scheduling import schedule_cohort, available_cpu_cores
from ._task_plan import TaskPlanner
from ._ready_micro import ReadyMicros
from ._pool_transport import OutputStore, encode_table, decode_batch
from .graph_lib import _cpu_budget


_worker_prior = None
_worker_limits = None
_worker_output_dir = None
_worker_trace = None


def _init_worker(prior, output_dir=None, cpu_permits=None, kernel_threads=1):
    global _worker_prior, _worker_limits, _worker_output_dir, _worker_trace
    torch.set_num_threads(1)
    _worker_limits = threadpoolctl.threadpool_limits(limits=1)
    _worker_prior = prior
    _worker_output_dir = output_dir
    _cpu_budget.configure_worker_budget(cpu_permits, kernel_threads)
    from ._generation_trace import worker_writer
    _worker_trace = worker_writer()


def _generate(params, seed, task_id=None):
    started = time.perf_counter()
    cpu_started = time.process_time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with _cpu_budget.generation_slot():
        generation_started = time.perf_counter()
        if _worker_trace is not None and _worker_trace.count < _worker_trace.limit:
            from ._generation_trace import TaskTrace
            classes = params.get('num_classes')
            identity = dict(task_id=task_id, seed=int(seed), rows=int(params['seq_len']),
                            requested_width=int(params['num_features']),
                            classes=None if classes is None else int(classes))
            with TaskTrace(_worker_trace, identity) as trace:
                x, y, d = _worker_prior.generate_dataset(params, compact=True, trace=trace)
        else:
            x, y, d = _worker_prior.generate_dataset(params, compact=True)
        generated = time.perf_counter()
        cpu_seconds = time.process_time() - cpu_started
        x, y = encode_table(x.numpy(), y.numpy(), _worker_output_dir)
        finished = time.perf_counter()
    budget_stats = _cpu_budget.worker_budget.stats if _cpu_budget.worker_budget else {}
    timing = dict(started_monotonic_s=started, generation_started_monotonic_s=generation_started,
                  generated_monotonic_s=generated, encoded_monotonic_s=finished,
                  process_cpu_s=cpu_seconds)
    return x, y, int(d), generated - started, finished - generated, finished, budget_stats, timing


def _output_bytes(params):
    # Compact outputs can only DELETE requested features. max_features is the
    # padded prior/filter width and can greatly overreserve long narrow tables.
    return sum(p["seq_len"] * (4 * p.get("num_features", p["max_features"]) +
                              (4 if p.get("regression", True) else 8)) for p in params)


def _select_admission(admission, reserved, byte_budget, allow_oversized=False):
    estimates = [_output_bytes(params)
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
        if stop - start == 1:
            # No allocation/copy for the Stage2/3 micro=1 path, including mmap.
            yield xs[start][:rows, :width].unsqueeze(0), ys[start][:rows].unsqueeze(0), d, length, train
            continue
        xx = [xs[i][:int(lengths[i]), :int(ds[i])] for i in range(start, stop)]
        yy = [ys[i][:int(lengths[i])] for i in range(start, stop)]
        x = torch.stack([F.pad(t, (0, width - t.shape[1], 0, rows - t.shape[0])) for t in xx])
        y = torch.stack([F.pad(t, (0, rows - t.shape[0])) for t in yy])
        yield x, y, d, length, train


class _Service:
    def __init__(self, prior, workers, window, byte_budget, seed, clients,
                 schedule="none", micro_size=1, cost_model=None, transport="numpy", bucket_steps=4, kernel_threads=1,
                 shape_seed=None, start_step=0, ready_max_delay=2):
        self.signature = pickle.dumps((prior, workers, window, byte_budget, seed, clients,
                                       schedule, micro_size, cost_model, transport, bucket_steps, kernel_threads,
                                       shape_seed, start_step, ready_max_delay))
        self.prior, self.window, self.byte_budget = prior, window, byte_budget
        self.schedule, self.micro_size, self.cost_model = schedule, micro_size, cost_model or {}
        # Window policies assign from metadata before generation. A rank can
        # begin its GPU work once its OWN tables are ready, rather than waiting
        # for the node's slowest graph. Legacy policies retain actual-width
        # assignment after the complete cohort for controlled comparisons.
        self.early_dispatch = schedule.startswith("window_") or schedule == "shape_cohort"
        self.cohort_window = max(1, window // clients) if schedule != "none" else window
        self.ready_micros = (ReadyMicros(prior.batch_size, clients, micro_size, ready_max_delay)
                             if schedule == 'ready_micro' else None)
        self.micro_steps = {}
        self.rank_sequences = [0] * clients
        self.last_schedule = {}
        self.planner = TaskPlanner(prior, clients, micro_size, self.cost_model, seed, schedule, bucket_steps,
                                   shape_seed=shape_seed, start_step=start_step)
        self.output = OutputStore(transport)
        self.leases = {}
        self.encoding_seconds = deque(maxlen=4096)
        self.result_transit_seconds = deque(maxlen=4096)
        self.fetch_wait_seconds = 0.
        self.worker_seconds = 0.
        self.kernel_stats = dict(parallel_distance_calls=0, parallel_distance_max_threads=1,
                                 parallel_distance_extra_thread_seconds=0.)
        self.started = time.perf_counter()
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
        self.trace = None
        if os.environ.get('TABICL_PRIOR_TRACE_DIR'):
            from ._generation_trace import Writer
            self.trace = Writer(os.environ['TABICL_PRIOR_TRACE_DIR'], 'service', 500000)
        # These are inherited before spawned workers import NumPy/SciPy. Setting
        # thread limits only in the initializer is too late for import-time BLAS.
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
        # Graph traversal uses hashed containers. Seeding NumPy/Torch alone does
        # not stabilize their iteration order across independently spawned workers.
        os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
        context = mp.get_context("spawn")
        self.cpu_slots = max(1, available_cpu_cores()-clients)
        self.cpu_permits = context.BoundedSemaphore(self.cpu_slots) if kernel_threads > 1 else None
        self.executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=context,
            initializer=_init_worker, initargs=(prior, self.output.path, self.cpu_permits, kernel_threads),
        )
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()

    def _produce(self):
        try:
            torch.set_num_threads(1)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            batch_id = 0
            while True:
                with self.condition:
                    if self.closed:
                        return
                    # Count ready AND in-flight batches against both limits.
                    while len(self.batches) < self.cohort_window:
                        if not self.admission:
                            self.admission.append(self.planner.next())
                        available = _select_admission([p.params for p in self.admission], self.reserved, self.byte_budget)
                        if available is None and self.schedule == "none" and len(self.admission) < self.admission_lookahead:
                            self.admission.append(self.planner.next())
                            continue
                        if available is None and not self.batches:
                            # Only start an oversized batch after searching the
                            # bounded lookahead for a batch that fits.
                            available = _select_admission(
                                [p.params for p in self.admission], self.reserved, self.byte_budget, allow_oversized=True
                            )
                        if available is None:
                            self.admission_blocked += 1
                            break
                        if available:
                            self.admission_bypasses += 1
                        planned = self.admission[available]
                        params = planned.params
                        del self.admission[available]
                        estimate = _output_bytes(params)
                        state = dict(params=params, results=[None] * len(params), left=len(params),
                                     bytes=estimate, delivered=set(), task_timings=[None] * len(params),
                                     task_ids=planned.task_ids,
                                     submitted_monotonic_s=time.perf_counter())
                        if self.early_dispatch:
                            state["assignments"], state["schedule_stats"] = schedule_cohort(
                                params, [(None, None, p["num_features"]) for p in params], self.prior.batch_size,
                                self.clients, self.micro_size, self.cost_model)
                        self.batches[batch_id] = state
                        if self.ready_micros is not None:
                            self.ready_micros.register(batch_id, planned.task_ids)
                            state['task_ids'] = planned.task_ids
                        self.reserved += estimate
                        self.peak_reserved = max(self.peak_reserved, self.reserved)
                        # Identity/seeds precede execution order. Long generation
                        # tasks start first; no sampled task is dropped or retried.
                        order = sorted(range(len(params)), key=lambda i: -(params[i]["seq_len"] * params[i]["num_features"]))
                        for i in order:
                            p = params[i]
                            seed = int(np.random.SeedSequence([self.seed, planned.task_ids[i]]).generate_state(1)[0])
                            future = self.executor.submit(_generate, p, seed, planned.task_ids[i])
                            self.pending[future] = (batch_id, i)
                        batch_id += 1
                    if not self.pending:
                        self.condition.wait(timeout=0.1)
                        continue
                done, _ = wait(self.pending, timeout=0.1, return_when=FIRST_COMPLETED)
                with self.condition:
                    for future in done:
                        bid, idx = self.pending.pop(future)
                        state = self.batches[bid]
                        x, y, d, elapsed, encoding, finished, kernel_stats, timing = future.result()
                        for name, value in kernel_stats.items():
                            self.kernel_stats[name] = (max(self.kernel_stats[name], value) if name.endswith("max_threads")
                                                       else self.kernel_stats[name]+value)
                        self.result_transit_seconds.append(time.perf_counter() - finished)
                        state["results"][idx] = (x, y, d)
                        state["task_timings"][idx] = timing
                        if self.trace is not None:
                            self.trace.write(dict(event='complete', source=bid, index=idx,
                                task_id=state['task_ids'][idx], submitted_s=state['submitted_monotonic_s'],
                                received_s=time.perf_counter(), **timing))
                        self.generation_seconds.append(elapsed)
                        self.encoding_seconds.append(encoding)
                        self.worker_seconds += elapsed + encoding
                        state["left"] -= 1
                        self.completed_datasets += 1
                        if self.ready_micros is not None:
                            self.ready_micros.complete(bid, idx)
                        if state["left"] == 0:
                            if self.schedule != "none" and not self.early_dispatch and self.ready_micros is None:
                                state["assignments"], state["schedule_stats"] = schedule_cohort(
                                    state["params"], state["results"], self.prior.batch_size,
                                    self.clients, self.micro_size, self.cost_model,
                                    balanced=self.schedule in ("balanced", "window_fifo", "window_bucketed"))
                            if self.ready_micros is None:
                                self.ready.append(bid)
                    self.condition.notify_all()
        except BaseException:
            with self.condition:
                self.error = traceback.format_exc()
                self.condition.notify_all()

    def get_batch(self, rank=None, sequence=None, with_timing=False):
        if self.ready_micros is not None:
            return self._get_micro_batch(rank, sequence, with_timing)
        started = time.perf_counter()
        with self.condition:
            def available():
                if self.early_dispatch:
                    state = self.batches.get(sequence)
                    return state is not None and all(state["results"][i] is not None for i in state["assignments"][rank])
                return (sequence in self.ready) if self.schedule != "none" else bool(self.ready)
            if self.schedule != "none" and (rank is None or not 0 <= rank < self.clients):
                raise ValueError("balanced fetch requires a valid local rank and sequence")
            while not available() and not self.error and not self.closed:
                self.condition.wait(timeout=1)
            if self.error:
                raise RuntimeError("node prior worker failed:\n" + self.error)
            if self.closed:
                raise RuntimeError("node prior pool is closed")
            self.fetch_wait_seconds += time.perf_counter() - started
            if self.schedule != "none":
                state = self.batches[sequence]
                if rank in state["delivered"]:
                    raise RuntimeError("duplicate rank/cohort delivery")
                indices = state["assignments"][rank]
                results = [state["results"][i] for i in indices]
                params = [state["params"][i] for i in indices]
                state["delivered"].add(rank)
                self.last_schedule = state["schedule_stats"]
                if len(state["delivered"]) == self.clients:
                    self.ready.remove(sequence)
                lease = (rank, sequence)
                self.leases[lease] = (sequence, results)
            else:
                bid = self.ready.popleft()
                state = self.batches[bid]
                results, params = state["results"], state["params"]
                lease = (rank, sequence)
                self.leases[lease] = (bid, results)
                indices = range(len(results))
            task_times = [state["task_timings"][i] for i in indices]
            # Balanced assignment depends on every table in the cohort. A rank
            # whose own tables finished early still waits for the last table.
            dependency_times = task_times if self.early_dispatch else state["task_timings"]
            timing = dict(
                tasks=len(task_times), submitted_monotonic_s=state["submitted_monotonic_s"],
                first_generation_monotonic_s=min(t["generation_started_monotonic_s"] for t in task_times),
                generation_ready_monotonic_s=max(t["generated_monotonic_s"] for t in task_times),
                dispatch_generation_ready_monotonic_s=max(t["generated_monotonic_s"]
                                                          for t in dependency_times),
                payload_ready_monotonic_s=max(t["encoded_monotonic_s"] for t in task_times),
                process_cpu_s=sum(t["process_cpu_s"] for t in task_times),
                task_wall_s=sum(t["generated_monotonic_s"]-t["generation_started_monotonic_s"]
                                for t in task_times),
                service_wait_s=time.perf_counter()-started,
            )
            self.delivered_batches += 1
            self.condition.notify_all()
        xs, ys, ds = zip(*results)
        payload = xs, ys, np.asarray(ds, dtype=np.int64), np.asarray(
            [p["seq_len"] for p in params], dtype=np.int64
        ), np.asarray([p["train_size"] for p in params], dtype=np.int64)
        return (payload, timing) if with_timing else payload

    def _get_micro_batch(self, rank, sequence, with_timing):
        started = time.perf_counter()
        with self.condition:
            if rank is None or not 0 <= rank < self.clients or sequence != self.rank_sequences[rank]:
                raise ValueError('ready_micro requires the next sequence for a valid rank')
            while sequence not in self.micro_steps:
                if self.error:
                    raise RuntimeError('node prior worker failed:\n' + self.error)
                if self.closed or rank in self.released:
                    raise RuntimeError('node prior pool is closed or rank released')
                units = self.ready_micros.pop_step()
                if units is None:
                    waiting = time.perf_counter() if self.trace is not None else None
                    reason = self.ready_micros.wait_state() if self.trace is not None else None
                    self.condition.wait(timeout=.1)
                    if self.trace is not None:
                        self.trace.write(dict(event='wait', step=sequence, rank=rank,
                            started_s=waiting, finished_s=time.perf_counter(), **reason))
                    continue
                # Balance actual GPU cost only among complete, intact micros.
                # Materialization, seeds, parameter sampling and generation stay
                # in the existing executor. All ranks receive equal slot counts.
                refs = [(u.batch, i) for u in units for i in u.indices]
                params = [self.batches[b]['params'][i] for b, i in refs]
                results = [self.batches[b]['results'][i] for b, i in refs]
                groups, stats = schedule_cohort(params, results, self.prior.batch_size,
                                               self.clients, self.micro_size, self.cost_model)
                step = self.ready_micros.next_step - 1
                self.micro_steps[step] = dict(refs=refs, groups=groups, delivered=set(),
                    acknowledged=set(), stats=stats,
                    dispatch_ready=max(self.batches[b]['task_timings'][i]['generated_monotonic_s']
                                       for b, i in refs))
                if self.trace is not None:
                    self.trace.write(dict(event='dispatch', step=step, at_s=time.perf_counter(),
                        task_ids=[self.batches[b]['task_ids'][i] for b, i in refs],
                        source_steps=[b for b, _ in refs], groups=groups))
                self.condition.notify_all()
            state = self.micro_steps[sequence]
            if self.error:
                raise RuntimeError('node prior worker failed:\n' + self.error)
            if self.closed or rank in self.released:
                raise RuntimeError('node prior pool is closed or rank released')
            if rank in state['delivered']:
                raise RuntimeError('duplicate ready_micro delivery')
            refs = [state['refs'][i] for i in state['groups'][rank]]
            results = [self.batches[b]['results'][i] for b, i in refs]
            params = [self.batches[b]['params'][i] for b, i in refs]
            times = [self.batches[b]['task_timings'][i] for b, i in refs]
            timing = dict(tasks=len(refs),
                task_ids=[self.batches[b]['task_ids'][i] for b, i in refs],
                source_steps=[b for b, _ in refs], dispatch_step=sequence,
                submitted_monotonic_s=min(self.batches[b]['submitted_monotonic_s'] for b, _ in refs),
                first_generation_monotonic_s=min(t['generation_started_monotonic_s'] for t in times),
                generation_ready_monotonic_s=max(t['generated_monotonic_s'] for t in times),
                dispatch_generation_ready_monotonic_s=state['dispatch_ready'],
                payload_ready_monotonic_s=max(t['encoded_monotonic_s'] for t in times),
                process_cpu_s=sum(t['process_cpu_s'] for t in times),
                task_wall_s=sum(t['generated_monotonic_s']-t['generation_started_monotonic_s'] for t in times),
                service_wait_s=time.perf_counter()-started)
            state['delivered'].add(rank)
            self.rank_sequences[rank] += 1
            self.last_schedule = state['stats']
            self.leases[rank, sequence] = (refs, results)
            self.delivered_batches += 1
            self.fetch_wait_seconds += time.perf_counter()-started
            self.condition.notify_all()
        xs, ys, ds = zip(*results)
        payload = (xs, ys, np.asarray(ds, dtype=np.int64),
                   np.asarray([p['seq_len'] for p in params], dtype=np.int64),
                   np.asarray([p['train_size'] for p in params], dtype=np.int64))
        return (payload, timing) if with_timing else payload

    def acknowledge(self, rank, sequence):
        """Client mapped the payload: reclaim names, not still-live mappings."""
        with self.condition:
            bid, results = self.leases.pop((rank, sequence))
            self.output.release(results)
            if self.ready_micros is not None:
                for source, index in bid:
                    source_state = self.batches[source]
                    released_bytes = _output_bytes([source_state['params'][index]])
                    self.reserved -= released_bytes
                    source_state['bytes'] -= released_bytes
                    source_state['results'][index] = None
                    source_state.setdefault('acked_indices', set()).add(index)
                    if len(source_state['acked_indices']) == len(source_state['params']):
                        del self.batches[source]
                step = self.micro_steps[sequence]
                step['acknowledged'].add(rank)
                if len(step['acknowledged']) == self.clients:
                    del self.micro_steps[sequence]
                self.condition.notify_all()
                return
            state = self.batches[bid]
            state.setdefault("acknowledged", set()).add(rank)
            if self.schedule == "none" or len(state["acknowledged"]) == self.clients:
                del self.batches[bid]
                self.reserved -= state["bytes"]
            self.condition.notify_all()

    def stats(self):
        with self.condition:
            durations = np.asarray(self.generation_seconds, dtype=np.float64)
            task_latency = ({f"generation_{key}_ms": float(np.quantile(durations, q) * 1000)
                             for key, q in (("p50", .50), ("p95", .95), ("p99", .99))}
                            if len(durations) else {})
            return dict(completed_datasets=self.completed_datasets, delivered_batches=self.delivered_batches,
                        buffered_batches=len(self.batches), ready_batches=(
                            sum(len(s['delivered']) < self.clients for s in self.micro_steps.values())
                            if self.ready_micros is not None else len(self.ready)),
                        buffered_local_batches=len(self.batches) * (self.clients if self.schedule != "none" else 1),
                        reserved_bytes=self.reserved, peak_reserved_bytes=self.peak_reserved,
                        admission_batches=len(self.admission), admission_bypasses=self.admission_bypasses,
                        admission_blocked=self.admission_blocked, **task_latency,
                        encoding_mean_ms=float(np.mean(self.encoding_seconds) * 1000) if self.encoding_seconds else 0.,
                        result_transit_mean_ms=float(np.mean(self.result_transit_seconds) * 1000) if self.result_transit_seconds else 0.,
                        worker_busy_seconds=self.worker_seconds, service_fetch_wait_seconds=self.fetch_wait_seconds,
                        service_uptime_seconds=time.perf_counter() - self.started,
                        active_leases=len(self.leases), cpu_slots=self.cpu_slots, **self.kernel_stats, **self.planner.stats,
                        assignment_uses_requested_features=int(self.early_dispatch),
                        **self.last_schedule,
                        **(self.ready_micros.stats() if self.ready_micros is not None else {}),
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
                process.join(timeout=2)
        self.output.close()


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
                 seed=42, key=None, local_rank=None, local_world_size=None, timeout=180,
                 schedule="none", micro_size=1, cost_model=None, transport="numpy", bucket_steps=4, kernel_threads=1,
                 shape_seed=None, start_step=0, ready_max_delay=2):
        if not isinstance(prior, GraphPrior) or str(prior.device) != "cpu":
            raise ValueError("node_pool requires online graph_scm on CPU")
        if min(workers, window, byte_budget) < 1:
            raise ValueError("pool workers, window and byte budget must be positive")
        self.dataset = prior
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if local_rank is None else local_rank
        self.local_world = int(os.environ.get("LOCAL_WORLD_SIZE", 1)) if local_world_size is None else local_world_size
        if self.local_world < 1 or not 0 <= self.local_rank < self.local_world:
            raise ValueError("local rank must be within the positive local world size")
        if schedule not in ("none", "round_robin", "balanced", "window_fifo", "window_bucketed", "shape_cohort", "ready_micro") or micro_size < 1:
            raise ValueError("invalid prior scheduling settings")
        if transport not in ("numpy", "mmap") or min(bucket_steps, kernel_threads) < 1:
            raise ValueError("invalid transport, bucket_steps or kernel_threads")
        if ready_max_delay < 0 or (schedule == 'ready_micro' and
                                   ready_max_delay >= max(1, window // self.local_world)):
            raise ValueError('ready_micro delay must be nonnegative and smaller than the cohort window')
        if schedule != "none" and (prior.batch_size % micro_size or
                min(prior.batch_size, prior.batch_size_per_gp) % micro_size):
            raise ValueError("balanced scheduling requires micro size dividing local batch and prior group size")
        self.schedule, self.sequence = schedule, 0
        self.transport = transport
        self.client_times = dict(rpc_seconds=0., decode_seconds=0., acknowledge_seconds=0.)
        run_key = key or (os.environ.get("TORCHELASTIC_RUN_ID", "") + ":" + os.environ.get("MASTER_PORT", ""))
        if not key and self.local_world == 1 and not os.environ.get("TORCHELASTIC_RUN_ID"):
            run_key += ":" + str(os.getpid())
        digest = hashlib.sha256(run_key.encode()).hexdigest()[:24]
        self.address = os.path.join(tempfile.gettempdir(), f"tabicl-prior-{digest}.sock")
        self.authkey = digest.encode()
        self.args = (prior, workers, window, byte_budget, seed, self.local_world, schedule, micro_size, cost_model,
                     transport, bucket_steps, kernel_threads, seed if shape_seed is None else shape_seed, start_step,
                     ready_max_delay)
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
        sequence = self.sequence
        started = time.perf_counter()
        payload, timing = self.service.get_batch(self.local_rank, sequence, with_timing=True)
        received = time.perf_counter()
        try:
            batch = decode_batch(payload, self.transport)
        finally:
            decoded = time.perf_counter()
            self.service.acknowledge(self.local_rank, sequence)
        self.client_times["rpc_seconds"] += received - started
        self.client_times["decode_seconds"] += decoded - received
        self.client_times["acknowledge_seconds"] += time.perf_counter() - decoded
        self.last_batch_timing = dict(timing, fetch_started_monotonic_s=started,
                                     rpc_s=received-started, decode_s=decoded-received,
                                     acknowledge_s=time.perf_counter()-decoded)
        self.sequence += 1
        return batch

    def stats(self):
        return {**self.service.stats(), **self.client_times} if self.service is not None else {}

    def set_start_step(self, step):
        """Use a fresh stream on resume; exact task/optimizer-step replay is unsupported."""
        if self.service is not None:
            raise RuntimeError("cannot reseed an active node pool")
        self.args = (*self.args[:4], self.args[4] + int(step), *self.args[5:-2], int(step), self.args[-1])

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
