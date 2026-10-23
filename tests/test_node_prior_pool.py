import multiprocessing as mp
import os
import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tabicl.prior._dataset import GraphPrior, Prior
from tabicl.prior._dataset import _bootstrap_pvalue
from tabicl.prior._node_pool import NodePriorPool, iter_micro_batches
from tabicl.prior.graph_lib._config import PriorConfig


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def small_prior(**kwargs):
    settings = dict(batch_size=4, batch_size_per_gp=2, batch_size_per_subgp=2,
                    min_features=2, max_features=6, max_classes=3, min_seq_len=32,
                    max_seq_len=64, seq_len_per_gp=True, min_train_size=0.4,
                    max_train_size=0.6, n_jobs=1,
                    config=PriorConfig(min_n_nodes=2, max_n_nodes=4))
    settings.update(kwargs)
    return GraphPrior(**settings)


@pytest.mark.parametrize("rows", [1, 2, 127])
def test_linear_unique_filter_matches_sort(rows):
    x = torch.randn(2, rows, 8)
    x[:, :, 0] = 3
    x[:, :, 1] = float("inf")
    x[:, :, 2] = float("nan")
    x[:, :, 3] = -0.0
    ds = torch.tensor([5, 8])
    actual, actual_d = Prior.delete_unique_features(x, ds)
    for i, d in enumerate(ds):
        mask = [len(torch.unique(x[i, :, j])) > 1 for j in range(d)]
        expected = F.pad(x[i, :, :d][:, mask], (0, 8 - sum(mask)))
        torch.testing.assert_close(actual[i], expected, equal_nan=True)
        assert actual_d[i] == sum(mask)


def test_parameter_sampling_preserves_group_correlations():
    prior = small_prior(batch_size=7, batch_size_per_gp=4, batch_size_per_subgp=2)
    params = prior.sample_parameters()
    assert len(params) == 7
    assert len({(p["seq_len"], p["train_size"]) for p in params[:4]}) == 1
    assert len({(p["seq_len"], p["train_size"]) for p in params[4:]}) == 1
    for start in (0, 2, 4):
        assert params[start]["num_features"] == params[start + 1]["num_features"]


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
def test_compact_generation_is_seed_equivalent(regression, filtered):
    torch.set_num_threads(1)
    prior = small_prior(regression=regression, config=PriorConfig(
        min_n_nodes=2, max_n_nodes=4, filter_unpredictable_datasets=filtered))
    seed(172)
    params = prior.sample_parameters()[0]
    seed(173)
    x, y, d = prior.generate_dataset(params)
    seed(173)
    compact_x, compact_y, compact_d = prior.generate_dataset(params, compact=True)
    assert compact_x.shape == (params["seq_len"], int(d))
    torch.testing.assert_close(F.pad(compact_x, (0, x.shape[1] - int(d))), x, rtol=0, atol=0)
    torch.testing.assert_close(compact_y, y, rtol=0, atol=0)
    assert compact_d == d


@pytest.mark.parametrize("kind", ["dense", "nested", "list"])
def test_padding_matches_old_training_inputs(kind):
    ds, lengths, trains = torch.tensor([2, 4, 3]), torch.tensor([32, 32, 60000]), torch.tensor([16, 16, 48000])
    xs = [torch.randn(int(n), int(d)) for n, d in zip(lengths, ds)]
    ys = [torch.randn(int(n)) for n in lengths]
    padded_x = torch.stack([F.pad(x, (0, 4 - x.shape[1], 0, 60000 - x.shape[0])) for x in xs])
    padded_y = torch.stack([F.pad(y, (0, 60000 - y.shape[0])) for y in ys])
    if kind == "dense":
        batch = padded_x, padded_y, ds, lengths, trains
    elif kind == "nested":
        batch = torch.nested.nested_tensor(xs), torch.nested.nested_tensor(ys), ds, lengths, trains
    else:
        batch = xs, ys, ds, lengths, trains
    micros = iter_micro_batches(batch, 2)
    first = next(micros)
    assert first[0].shape == (2, 32, 4)
    assert first[0].numel() == 256  # no 60k padding for these two datasets
    for start, micro in zip((0, 2), [first, next(micros)]):
        x, y, d, length, train = micro
        stop = start + len(d)
        torch.testing.assert_close(x, padded_x[start:stop, :int(length.max()), :int(d.max())])
        torch.testing.assert_close(y, padded_y[start:stop, :int(length.max())])


def _peer(prior, key, queue):
    pool = NodePriorPool(prior, workers=2, window=3, byte_budget=100000,
                         key=key, local_rank=1, local_world_size=2)
    try:
        with pool:
            batch = next(pool)
            queue.put((len(batch[0]), pool.stats()["worker_pids"]))
    except Exception as exc:
        queue.put(str(exc))


def test_node_pool_shared_workers_bounded_queue_and_cleanup(tmp_path):
    ctx = mp.get_context("spawn")
    prior, key = small_prior(), str(tmp_path)
    queue = ctx.Queue()
    peer = ctx.Process(target=_peer, args=(prior, key, queue))
    pool = NodePriorPool(prior, workers=2, window=3, byte_budget=100000,
                         key=key, local_rank=0, local_world_size=2)
    pids = []
    try:
        with pool:
            peer.start()
            for _ in range(3):
                batch = next(pool)
                assert len(batch[0]) == 4
                assert torch.equal(batch[3][::2], batch[3][1::2])
                for x, y, d, length in zip(*batch[:4]):
                    assert x.shape == (int(length), int(d))
                    assert len(y) == length
                    assert torch.isfinite(x).all()
            stats = pool.stats()
            pids = stats["worker_pids"]
            assert len(pids) == 2
            assert stats["buffered_batches"] <= 3
            assert stats["peak_reserved_bytes"] <= 100000
            assert stats["generation_p50_ms"] > 0
            assert stats["generation_p95_ms"] >= stats["generation_p50_ms"]
            assert stats["admission_batches"] >= 0
            count, peer_pids = queue.get(timeout=60)
            assert count == 4 and sorted(peer_pids) == sorted(pids)
            peer.join(timeout=15)
            assert peer.exitcode == 0
    finally:
        pool.close()
        if peer.is_alive():
            peer.terminate()
            peer.join()
    assert not os.path.exists(pool.address)
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_node_pool_propagates_worker_errors(tmp_path):
    prior = small_prior(config=PriorConfig(fct_types="invalid"))
    with NodePriorPool(prior, workers=1, window=1, key=str(tmp_path)) as pool:
        with pytest.raises(RuntimeError, match="node prior worker failed"):
            next(pool)


def test_node_pool_rejects_incompatible_prior():
    with pytest.raises(ValueError, match="graph_scm"):
        NodePriorPool(object(), workers=1)


def test_resume_changes_pool_rng_stream_before_start():
    pool = NodePriorPool(small_prior(), workers=1, seed=42)
    pool.set_start_step(20000)
    assert pool.args[4] == 20042


@pytest.mark.parametrize("rows", [32, 1024, 10240, 60000])
def test_bootstrap_chunking_preserves_exact_rng_and_pvalue(rows):
    rng = np.random.default_rng(19)
    imp = rng.normal(0.003, 1, rows)
    idx = np.random.default_rng(0).integers(0, rows, size=(200, rows))
    expected = float(np.mean(imp[idx].mean(axis=1) <= 0))
    assert _bootstrap_pvalue(imp) == expected


def test_tree_chunking_preserves_exact_output():
    from scripts.benchmark_prior_operators import inputs
    reference, actual = inputs("tree", 4096)
    torch.testing.assert_close(actual(), reference(), rtol=0, atol=0)


def test_tree_chunking_accepts_numpy_hyperparameters():
    from scripts.benchmark_prior_operators import inputs
    reference, actual = inputs("tree", 4096)
    # Graph-lib integer samplers return NumPy scalars in real generation.
    tree = actual.__closure__[0].cell_contents
    tree.n_trees_ = np.int64(tree.n_trees_)
    tree.depth_ = np.int64(tree.depth_)
    torch.testing.assert_close(actual(), reference(), rtol=0, atol=0)


@pytest.mark.parametrize("p", [0.5, 0.71, 1., 1.37, 2., 2.57, 4.])
@pytest.mark.parametrize("width", [1, 7])
def test_nearest_center_indices_match_cdist(p, width):
    from tabicl.prior.graph_lib._distance import nearest_center
    generator = torch.Generator().manual_seed(72)
    x = torch.randn(17000, width, generator=generator)
    centers = torch.randn(71, width, generator=generator)
    centers[3] = centers[0]
    x[:3] = centers[:3]
    expected = torch.cdist(x, centers, p=p).argmin(dim=-1)
    actual = nearest_center(x, centers, p)
    reference_distances = torch.cdist(x, centers, p=p)
    assert (reference_distances.gather(1, actual[:, None])[:, 0] <=
            reference_distances.gather(1, expected[:, None])[:, 0] * (1 + 1e-5) + 1e-7).all()


@pytest.mark.parametrize("p", [0.5, 1.37, 4.])
def test_nearest_center_rounding_ties_and_extremes(p):
    from tabicl.prior.graph_lib._distance import nearest_center
    x = torch.tensor([[0.], [1.], [1e-12], [1e10], [float("nan")], [float("inf")]])
    centers = torch.tensor([[1.], [1.0000001], [-1.], [0.], [1e-12], [float("inf")]])
    assert torch.equal(nearest_center(x, centers, p), torch.cdist(x, centers, p=p).argmin(dim=-1))


def test_nearest_center_preserves_underflow_first_index_tie():
    from tabicl.prior.graph_lib._distance import nearest_center
    x = torch.tensor([[-1.3123e-18]])
    centers = torch.tensor([[-3.755e-19], [-1.3123e-18]])
    actual = nearest_center(x, centers, 3.2815875545948123)
    reference_distances = torch.cdist(x, centers, p=3.2815875545948123)
    assert reference_distances[0, actual[0]] == reference_distances.min()


def test_node_pool_admission_can_bypass_oversized_head_batch():
    from collections import deque
    from tabicl.prior._node_pool import _select_admission

    admission = deque([[{"seq_len": 20, "max_features": 1}],
                       [{"seq_len": 5, "max_features": 1}]])
    fitting = _select_admission(admission, reserved=50, byte_budget=100)
    assert fitting == 1
    assert admission[fitting][0]["seq_len"] == 5
    assert _select_admission(admission, 100, 100, allow_oversized=False) is None
    assert _select_admission(admission, 100, 100, allow_oversized=True) == 0


def test_complete_graph_generation_matches_original_distance(monkeypatch):
    import tabicl.prior.graph_lib._function as function
    import tabicl.prior.graph_lib._converter as converter
    torch.set_num_threads(1)
    prior = small_prior()
    for value in range(10):
        seed(value)
        params = prior.sample_parameters()[0]
        seed(value + 100)
        actual = prior.generate_dataset(params, compact=True)
        with monkeypatch.context() as m:
            reference = lambda x, centers, p: torch.cdist(x, centers, p=p).argmin(dim=-1)
            m.setattr(function, "nearest_center", reference)
            m.setattr(converter, "nearest_center", reference)
            seed(value + 100)
            expected = prior.generate_dataset(params, compact=True)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
