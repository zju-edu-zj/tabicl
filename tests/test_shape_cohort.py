"""Conditioning law, cross-node agreement, resume and real transport contracts."""
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from scipy.stats import ks_2samp

from tabicl.prior._task_plan import TaskPlanner
from tabicl.prior._node_pool import NodePriorPool
from test_node_prior_pool import small_prior, seed


def planner(prior, ranks=8, seed_value=42, shape_seed=17, start_step=0):
    return TaskPlanner(prior, ranks, 1, {}, seed_value, "shape_cohort",
                       shape_seed=shape_seed, start_step=start_step)


@pytest.mark.parametrize("ranks", [1, 2, 8, 16])
@pytest.mark.parametrize("regression", [False, True])
def test_entire_optimizer_step_shares_shape_but_not_classes(ranks, regression):
    prior = small_prior(batch_size=64//ranks, batch_size_per_gp=1, regression=regression,
                        max_classes=10, min_seq_len=400, max_seq_len=60000, log_seq_len=True)
    p = planner(prior, ranks)
    shapes = []
    for step in range(8):
        batch = p.next()
        assert batch.task_ids == list(range(step*64, (step+1)*64))
        assert len(batch.params) == 64
        keys = ("seq_len", "train_size", "num_features", "max_features")
        shape = {tuple(t[k] for k in keys) for t in batch.params}
        assert len(shape) == 1
        shapes.extend(shape)
        classes = {t["num_classes"] for t in batch.params}
        assert classes == {None} if regression else len(classes) > 1
        # Each dataset still carries 1/global_batch gradient weight.
        values = torch.tensor(batch.task_ids, dtype=torch.float64).reshape(ranks, -1)
        torch.testing.assert_close(values.mean(1).mean(), values.flatten().mean())
    assert len(set(shapes)) > 1
    assert not p.ready  # no unbounded metadata window


def test_shape_rng_is_global_not_node_seed_world_size_or_class_rng():
    priors = [small_prior(batch_size=local, batch_size_per_gp=1, min_seq_len=400,
                         max_seq_len=60000, log_seq_len=True) for local in (8, 4, 2)]
    planners = [planner(p, ranks, seed_value=42+100003*i) for i, (p, ranks) in
                enumerate(zip(priors, (8, 8, 16)))]
    for _ in range(32):
        batches = []
        for i, p in enumerate(planners):
            seed(80+i)  # emulate independent node-local class streams
            batches.append(p.next())
        assert len({tuple(b.params[0][k] for k in
                         ("seq_len", "train_size", "num_features", "max_features")) for b in batches}) == 1
        assert len({tuple(t["num_classes"] for t in b.params[:16]) for b in batches}) > 1


def test_shape_resume_uses_absolute_optimizer_step():
    prior = small_prior(batch_size=1)
    original, resumed = planner(prior), planner(prior, start_step=12)
    for _ in range(12):
        original.next()
    for _ in range(5):
        a, b = original.next().params[0], resumed.next().params[0]
        assert all(a[k] == b[k] for k in ("seq_len", "train_size", "num_features", "max_features"))


@pytest.mark.parametrize("log_features", [False, True])
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("crop", [False, True])
def test_shape_marginals_match_original_table_law(log_features, replay, crop):
    prior = small_prior(batch_size=1, batch_size_per_gp=1, min_features=1, max_features=100,
                        min_seq_len=400, max_seq_len=60000, log_seq_len=True,
                        log_n_features=log_features, replay_small=replay)
    prior.adaptive_feature_limit = crop
    seed(102)
    baseline = [prior.sample_parameters()[0] for _ in range(5000)]
    rng = np.random.RandomState(203)
    conditioned = [prior.sample_shape(rng) for _ in range(5000)]
    for key in ("seq_len", "train_size", "num_features", "max_features"):
        assert ks_2samp([p[key] for p in baseline], [p[key] for p in conditioned]).statistic < .045
    for p in conditioned:
        assert p["num_features"] <= p["max_features"]
        assert p["max_features"] == (prior.feature_limit(p["seq_len"]) if crop else 100)


def test_shape_sampling_does_not_consume_class_rng():
    prior = small_prior()
    seed(127)
    expected = np.random.randint(10000, size=10)
    seed(127)
    prior.sample_shape(np.random.RandomState(19))
    np.testing.assert_array_equal(np.random.randint(10000, size=10), expected)


@pytest.mark.parametrize("per_group", [False, True])
@pytest.mark.parametrize("log_features", [False, True])
@pytest.mark.parametrize("crop", [False, True])
def test_joint_shape_law_matches_original_sampler_with_identical_random_draws(per_group, log_features, crop):
    # Marginal histograms alone miss a changed dependency between length,
    # support and the adaptive feature cap. Compare an entire single-table
    # draw against the existing sampler, including its replay-small mixture.
    prior = small_prior(batch_size=1, batch_size_per_gp=1, min_features=1, max_features=100,
                        min_seq_len=400, max_seq_len=60000, log_seq_len=True,
                        log_n_features=log_features, replay_small=True, seq_len_per_gp=per_group)
    prior.adaptive_feature_limit = crop
    for value in range(64):
        seed(value)
        expected = prior.sample_parameters()[0]
        actual = prior.sample_shape(np.random.RandomState(value))
        assert actual == {key: expected[key] for key in actual}


@pytest.mark.parametrize("bad", [dict(seq_len=10),
    dict(seq_len=10, train_size=10, num_features=2, max_features=6),
    dict(seq_len=10, train_size=5, num_features=7, max_features=6),
    dict(seq_len=10.5, train_size=5, num_features=2, max_features=6)])
def test_invalid_shape_rejected(bad):
    with pytest.raises(ValueError, match="shape"):
        small_prior().sample_parameters(shape=bad)


@pytest.mark.parametrize("transport", ["numpy", "mmap"])
@pytest.mark.parametrize("budget", [1, 1024**2])
def test_real_graph_cohort_bounded_transport_and_no_duplicate_tables(tmp_path, transport, budget):
    prior = small_prior(batch_size=2, batch_size_per_gp=1, batch_size_per_subgp=1)
    hashes = []
    import hashlib
    with ExitStack() as stack:
        pools = [stack.enter_context(NodePriorPool(prior, workers=2, window=4, seed=93,
                 byte_budget=budget, schedule="shape_cohort", transport=transport,
                 local_rank=r, local_world_size=2, shape_seed=17,
                 key=str(tmp_path / "cohort"))) for r in range(2)]
        for _ in range(3):
            with ThreadPoolExecutor(2) as executor:
                batches = list(executor.map(next, pools))
            assert len({int(n) for b in batches for n in b[3]}) == 1
            assert len({int(n) for b in batches for n in b[4]}) == 1
            for xs, ys, ds, ns, _ in batches:
                for x, y, d, n in zip(xs, ys, ds, ns):
                    assert x.shape == (n, d)  # accepted widths may shrink independently
                    hashes.append(hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest())
            assert pools[0].stats()["active_leases"] == 0
            assert pools[0].stats()["buffered_batches"] <= (1 if budget == 1 else 2)
    assert len(set(hashes)) == len(hashes)


@pytest.mark.parametrize("shape_seed", [None, 42])
def test_pool_resume_passes_common_seed_and_absolute_step(shape_seed):
    pool = NodePriorPool(small_prior(), workers=1, schedule="shape_cohort", seed=100045, shape_seed=shape_seed)
    pool.set_start_step(123)
    assert pool.args[4] == 100168
    assert pool.args[-2:] == (100045 if shape_seed is None else 42, 123)


def test_cohort_graph_identity_survives_worker_count_and_transport(tmp_path):
    from test_prior_scheduling import digest_batch
    prior = small_prior(batch_size=2, batch_size_per_gp=1)
    streams = []
    for workers, transport in ((1, "numpy"), (2, "numpy"), (2, "mmap")):
        stream = []
        with NodePriorPool(prior, workers=workers, schedule="shape_cohort", seed=93,
                           shape_seed=17, transport=transport, window=2,
                           key=str(tmp_path / f"{workers}-{transport}")) as pool:
            for _ in range(3):
                stream.append(digest_batch(next(pool)))
        streams.append(stream)
    assert streams[0] == streams[1] == streams[2]


@pytest.mark.parametrize("ranks", [8, 16])
def test_global_batch_64_real_pool_delivery_for_stage23(tmp_path, ranks):
    # Exercise the real manager/worker/lease path for every logical rank,
    # without requiring 16 large Torch interpreter processes on a laptop.
    prior = small_prior(batch_size=64//ranks, batch_size_per_gp=1)
    with ExitStack() as stack:
        pools = [stack.enter_context(NodePriorPool(prior, workers=2, window=ranks,
                 schedule="shape_cohort", micro_size=1, local_rank=r,
                 local_world_size=ranks, shape_seed=42, key=str(tmp_path))) for r in range(ranks)]
        with ThreadPoolExecutor(ranks) as executor:
            batches = list(executor.map(next, pools))
        assert [len(b[2]) for b in batches] == [64//ranks]*ranks
        assert len({int(n) for b in batches for n in b[3]}) == 1
        assert len({int(n) for b in batches for n in b[4]}) == 1
        assert pools[0].stats()["active_leases"] == 0


@pytest.mark.parametrize("field,value", [("shape_seed", -1), ("start_step", -1)])
def test_negative_shape_stream_identity_rejected(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError, match="nonnegative"):
        TaskPlanner(small_prior(), 8, 1, {}, 42, "shape_cohort", **kwargs)
