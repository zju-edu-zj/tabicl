"""Window membership/gradient contract and transport ownership, independently."""
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from tabicl.prior._node_pool import NodePriorPool, iter_micro_batches, _output_bytes
from tabicl.prior._pool_transport import OutputStore, encode_table, decode_batch
from tabicl.prior._task_plan import TaskPlanner, bucket_window
from tabicl.prior._dataset import GraphPrior
from test_node_prior_pool import small_prior, seed
from test_prior_scheduling import digest_batch


@pytest.mark.parametrize("ranks", [1, 2, 8, 16])
@pytest.mark.parametrize("micro,local", [(1, 4), (1, 1), (4, 4)])
@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_window_preserves_tasks_micros_weights_and_bounds(ranks, micro, local, steps):
    prior = small_prior(batch_size=local, batch_size_per_gp=micro,
                        batch_size_per_subgp=micro, max_seq_len=60000)
    streams = []
    for mode in ("window_fifo", "window_bucketed"):
        seed(93)
        planner = TaskPlanner(prior, ranks, micro, {}, 93, mode, steps)
        stream = [planner.next() for _ in range(2 * steps)]
        streams.append(stream)
        assert len(planner.ready) == 0
        assert planner.stats["metadata_tasks"] == steps * local * ranks
    fifo, bucketed = streams
    by_id = {i: pickle.dumps(p) for b in fifo for i, p in zip(b.task_ids, b.params)}
    assert sorted(i for b in bucketed for i in b.task_ids) == list(range(2*steps*local*ranks))
    for step, b in enumerate(bucketed):
        assert all(by_id[i] == pickle.dumps(p) for i, p in zip(b.task_ids, b.params))
        assert all(abs(i // (local*ranks) - step) <= steps-1 for i in b.task_ids)
        assert all(i // (local*ranks*steps) == step // steps for i in b.task_ids)
        for start in range(0, len(b.params), micro):
            assert b.task_ids[start:start+micro] == list(range(b.task_ids[start], b.task_ids[start]+micro))
        # Equal micro counts and sizes keep each dataset's global-mean weight.
        values = torch.tensor(b.task_ids, dtype=torch.float64).reshape(ranks, local)
        torch.testing.assert_close(values.mean(1).mean(), values.flatten().mean())


def test_window_reduces_constructed_single_slot_stragglers():
    params = [dict(seq_len=n, train_size=n//2, num_features=4) for n in [64, 1024, 64, 1024]*4]
    groups, stats = bucket_window(params, 1, 1, 4, {})
    assert stats["window_predicted_gain"] > 1.5
    assert sorted(i for group in groups for i in group) == list(range(16))


def test_output_admission_reserves_compact_width_not_filter_padding():
    p = dict(seq_len=60000, num_features=2, max_features=100, regression=False)
    assert _output_bytes([p]) == 60000*(4*2+8)


@pytest.mark.parametrize("ranks,local,micro", [(0, 1, 1), (1, 0, 1), (1, 1, 0), (2, 3, 2)])
def test_invalid_window_sizes(ranks, local, micro):
    with pytest.raises(ValueError):
        bucket_window([{}], micro, local, ranks, {})


@pytest.mark.parametrize("regression", [False, True])
def test_transport_maps_survive_ack_and_shutdown_without_payload_copy(regression):
    x = np.arange(120, dtype=np.float32).reshape(30, 4)
    y = np.arange(30, dtype=np.float32 if regression else np.int64)
    store = OutputStore("mmap")
    directory = Path(store.path)
    descriptor, empty = encode_table(x, y, store.path)
    batch = decode_batch(([descriptor], [empty], np.array([4]), np.array([30]), np.array([20])), "mmap")
    store.release([(descriptor, empty, 4)])
    assert not list(directory.iterdir())
    store.close()
    assert not directory.exists()
    micro = next(iter_micro_batches(batch, 1))
    assert micro[0].data_ptr() == batch[0][0].data_ptr()
    np.testing.assert_array_equal(micro[0][0].numpy(), x)
    np.testing.assert_array_equal(micro[1][0].numpy(), y)
    micro[0][0, 0, 0] = -42
    assert x[0, 0] == 0


def test_mmap_allocation_failure_cleans_partial_file(monkeypatch):
    import os
    def full(*args):
        raise OSError("test: no space")
    monkeypatch.setattr(os, "posix_fallocate", full, raising=False)
    store = OutputStore("mmap")
    try:
        with pytest.raises(OSError, match="no space"):
            encode_table(np.ones((2, 2)), np.ones(2), store.path)
        assert not list(Path(store.path).iterdir())
    finally:
        store.close()


@pytest.mark.parametrize("byte_budget", [1, 1024**2])
@pytest.mark.parametrize("kernel_threads", [1, 4])
def test_generated_window_tasks_identical_across_order_and_transport(tmp_path, byte_budget, kernel_threads):
    prior = small_prior(batch_size=2, batch_size_per_gp=1, batch_size_per_subgp=1)
    reference = []
    for mode, transport in (("window_fifo", "numpy"), ("window_fifo", "mmap"), ("window_bucketed", "mmap")):
        windows = []
        with ExitStack() as stack:
            pools = [stack.enter_context(NodePriorPool(prior, workers=2, window=4, seed=93,
                        byte_budget=byte_budget, schedule=mode, micro_size=1, transport=transport,
                        kernel_threads=kernel_threads,
                        bucket_steps=2, local_rank=r, local_world_size=2,
                        key=str(tmp_path / f"{mode}-{transport}"))) for r in range(2)]
            held = None
            for _ in range(2):
                hashes = []
                for _ in range(2):
                    for pool in pools:
                        held = next(pool)
                        hashes.extend(digest_batch(held))
                windows.append(sorted(hashes))
            assert pools[0].stats()["active_leases"] == 0
            assert all(isinstance(value, (int, float)) for key, value in pools[0].stats().items()
                       if key != "worker_pids")
            assert pools[0].stats()["buffered_batches"] <= (1 if byte_budget == 1 else 2)
        assert digest_batch(held)  # mappings remain valid after service shutdown
        reference.append(windows)
    assert reference[0] == reference[1] == reference[2]


class GatedPrior(GraphPrior):
    """Second rank's graph is held until the first rank has received its data."""
    def sample_parameters(self):
        params = super().sample_parameters()
        for p in params:
            p.update(seq_len=32, train_size=16, num_features=2,
                     gate=self.gate if getattr(self, "sample_count", 0) % 2 else None)
        self.sample_count = getattr(self, "sample_count", 0) + 1
        return params

    @staticmethod
    def generate_dataset(params, compact=False):
        import time
        deadline = time.monotonic() + 60
        while params["gate"] and not Path(params["gate"]).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("gate was never released")
            time.sleep(.02)
        return torch.ones(32, 2), torch.zeros(32, dtype=torch.long), torch.tensor(2)


def test_window_dispatch_does_not_wait_for_other_ranks_graph(tmp_path):
    prior = GatedPrior(batch_size=1, batch_size_per_gp=1, min_features=2, max_features=6,
                       max_seq_len=32, n_jobs=1)
    gate = tmp_path / "release-worker"
    prior.gate = str(gate)
    with ExitStack() as stack:
        pools = [stack.enter_context(NodePriorPool(prior, workers=2, window=2, schedule="window_fifo",
                    transport="mmap", bucket_steps=2, local_rank=r, local_world_size=2,
                    key=str(tmp_path / "early"))) for r in range(2)]
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(next, pools[0])
            try:
                batch = future.result(timeout=30)
                assert len(batch[0]) == 1 and not gate.exists()
            finally:
                gate.write_text("ready")
        assert len(next(pools[1])[0]) == 1
