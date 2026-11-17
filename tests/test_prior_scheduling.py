import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import numpy as np
import pytest
import torch

from tabicl.prior._node_pool import NodePriorPool
from tabicl.prior._scheduling import balanced_partition, schedule_cohort, available_cpu_cores, micro_cost
from test_node_prior_pool import small_prior


@pytest.mark.parametrize("ranks", [1, 2, 3, 8, 16])
@pytest.mark.parametrize("slots", [1, 2, 4, 8])
def test_balancer_preserves_cardinality_membership_and_never_worsens_proxy(ranks, slots):
    for seed in range(10):
        costs = np.random.default_rng(seed).lognormal(2, 2, ranks * slots).tolist()
        result = balanced_partition(costs, ranks)
        assert sorted(i for g in result for i in g) == list(range(len(costs)))
        assert all(len(g) == slots for g in result)
        baseline = max(sum(costs[r*slots:(r+1)*slots]) for r in range(ranks))
        assert max(sum(costs[i] for i in g) for g in result) <= baseline + 1e-8
        assert balanced_partition(costs, ranks) == result
        if slots == 1:
            assert max(costs[g[0]] for g in result) == baseline


def test_general_multi_rank_balancer_reduces_constructed_imbalance():
    costs = [100, 90, 80, 70, 4, 3, 2, 1]
    result = balanced_partition(costs, 4)
    assert max(sum(costs[i] for i in g) for g in result) <= 105


def test_cost_proxy_tracks_actual_feature_grouping_tokens():
    from tabicl._model.embedding import ColEmbedding
    table = torch.randn(1, 32, 10)
    params, results = [dict(seq_len=32, train_size=16)], [(None, None, 10)]
    costs = {}
    for mode in (False, "same", "valid"):
        embedding = ColEmbedding(embed_dim=16, num_blocks=1, nhead=2, dim_feedforward=32,
                                 num_inds=4, feature_group=mode, feature_group_size=3)
        tokens = embedding.feature_grouping(table).shape[2]
        model = dict(col_feature_group=mode, col_feature_group_size=3, icl_num_blocks=0,
                     row_num_blocks=0, col_num_blocks=1, embed_dim=16, col_num_inds=4)
        costs[mode] = micro_cost(params, results, [0], model)
        assert costs[mode] > 0
        assert costs[mode] / tokens == costs[False] / 10
    assert costs[False] == costs["same"] > costs["valid"]


def test_cost_proxy_respects_hybrid_layers_and_centroid_routing():
    params, results = [dict(seq_len=60000, train_size=48000)], [(None, None, 20)]
    base = dict(col_num_blocks=0, row_num_blocks=0, icl_num_blocks=12)
    def cost(**kw):
        return micro_cost(params, results, [0], {**base, **kw})
    assert cost(icl_attention="dense") > cost(icl_attention="global_sparse") > cost(icl_attention="global")
    assert cost(icl_attention="global_sparse", icl_sparse_every=1) > cost(icl_attention="global_sparse", icl_sparse_every=3)
    assert cost(icl_attention="dual_path") > cost(icl_attention="global_sparse")
    assert cost(icl_attention="adaptive_frontier") < cost(icl_attention="dense")
    assert cost(icl_attention="sparse", icl_block_size=64) > cost(icl_attention="sparse", icl_block_size=128,
                                                              icl_num_selected_blocks=2)


@pytest.mark.parametrize("micro", [1, 4])
def test_cohort_preserves_micro_boundaries_and_gradient_weights(micro):
    ranks, local = 16, 4
    params = [dict(seq_len=2048, train_size=1200 + (i // micro)) for i in range(ranks * local)]
    results = [(None, None, 1 + i % 100) for i in range(len(params))]
    assigned, stats = schedule_cohort(params, results, local, ranks, micro, {})
    assert sorted(i for group in assigned for i in group) == list(range(ranks * local))
    # DDP rank means equal the original global dataset mean after reassignment.
    values = torch.randn(ranks * local, 5, dtype=torch.float64)
    original = values.mean(0)
    actual = torch.stack([values[group].mean(0) for group in assigned]).mean(0)
    torch.testing.assert_close(actual, original, atol=1e-15, rtol=1e-15)
    for group in assigned:
        for start in range(0, local, micro):
            unit = group[start:start+micro]
            assert unit == list(range(unit[0], unit[0]+micro))
            assert unit[0] % micro == 0
    assert stats["predicted_gain"] >= 1 - 1e-12


@pytest.mark.parametrize("costs,ranks", [([], 2), ([1, 2, 3], 2), ([float("nan")], 1), ([1], 0)])
def test_invalid_balancing_rejected(costs, ranks):
    with pytest.raises(ValueError):
        balanced_partition(costs, ranks)


def digest_batch(batch):
    return [hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()
            for x, y in zip(*batch[:2])]


@pytest.mark.parametrize("schedule", ["round_robin", "balanced"])
@pytest.mark.parametrize("byte_budget", [1, 2 * 1024**3])
def test_shared_pool_fixed_cohorts_match_unscheduled_task_stream_and_slow_client(tmp_path, schedule, byte_budget):
    prior = small_prior(batch_size=2, batch_size_per_gp=2)
    reference = []
    with NodePriorPool(prior, workers=2, window=1, seed=93, key=str(tmp_path / "reference")) as pool:
        for _ in range(4):
            reference.append(digest_batch(next(pool)))
    with ExitStack() as stack:
        leader = stack.enter_context(NodePriorPool(prior, workers=2, window=4, seed=93, byte_budget=byte_budget,
            schedule=schedule, micro_size=1, local_rank=0, local_world_size=2, key=str(tmp_path / "balanced")))
        peer = stack.enter_context(NodePriorPool(prior, workers=2, window=4, seed=93, byte_budget=byte_budget,
            schedule=schedule, micro_size=1, local_rank=1, local_world_size=2, key=str(tmp_path / "balanced")))
        first = next(leader)
        # Fast rank may request the next cohort before its peer consumes this
        # one. Delivery is addressed by (rank, sequence), never first-ready.
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(next, leader)
            peer_first = next(peer)
            second = future.result(timeout=60)
        peer_second = next(peer)
        assert sorted(digest_batch(first) + digest_batch(peer_first)) == sorted(reference[0] + reference[1])
        assert sorted(digest_batch(second) + digest_batch(peer_second)) == sorted(reference[2] + reference[3])
        assert leader.stats()["predicted_gain"] >= 1 - 1e-12
        assert leader.stats()["buffered_batches"] <= 2
        if byte_budget == 1:
            assert leader.stats()["buffered_batches"] <= 1


def test_cpu_auto_count_respects_quota_and_affinity():
    visible = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    assert 1 <= available_cpu_cores() <= visible


def test_cpu_quota_overrides_host_affinity(monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(192)), raising=False)
    def read(path, *args, **kwargs):
        if str(path) == "/sys/fs/cgroup/cpu.max":
            return "6976000 100000"
        raise FileNotFoundError(str(path))
    monkeypatch.setattr(Path, "read_text", read)
    assert available_cpu_cores() == 69


def test_rejects_incomplete_micro_batches():
    with pytest.raises(ValueError, match="dividing"):
        NodePriorPool(small_prior(batch_size=3), workers=1, schedule="balanced", micro_size=2)


@pytest.mark.parametrize("rank,world", [(0, 0), (-1, 2), (2, 2)])
def test_rejects_invalid_local_rank_world(rank, world):
    with pytest.raises(ValueError, match="local rank"):
        NodePriorPool(small_prior(), workers=1, local_rank=rank, local_world_size=world)
