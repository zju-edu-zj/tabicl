"""Full-complexity execution must preserve indices, data and RNG states."""
import random
import weakref

import numpy as np
import pytest
import torch

from tabicl.prior.graph_lib import _distance
from tabicl.prior.graph_lib._base import Context, FeatureSpec
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._graph_function import RandomGraphFunction


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(autouse=True)
def strict_one_dimensional_reference(monkeypatch):
    monkeypatch.setenv("TABICL_PRIOR_1D_MODE", "reference")


def test_legacy_is_default_and_reference_is_explicit(monkeypatch):
    x = torch.tensor([[0.], [.1], [.2], [1.8], [2.]])
    centers = torch.tensor([[0.], [0.], [2.]])
    monkeypatch.delenv("TABICL_PRIOR_1D_MODE", raising=False)
    assert _distance.nearest_center(x, centers, 1.3).tolist() == [0, 1, 1, 2, 2]
    monkeypatch.setenv("TABICL_PRIOR_1D_MODE", "reference")
    assert _distance.nearest_center(x, centers, 1.3).tolist() == [0, 0, 0, 2, 2]
    monkeypatch.setenv("TABICL_PRIOR_1D_MODE", "typo")
    with pytest.raises(ValueError, match="TABICL_PRIOR_1D_MODE"):
        _distance.nearest_center(x, centers, 1.3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("p", [.5, .550479, 1., 1.9999, 2., 3.2815875545948123, 4.])
@pytest.mark.parametrize("width", [1, 2, 31, 108])
def test_repeated_rows_equal_cdist(dtype, p, width):
    g = torch.Generator().manual_seed(72)
    base = torch.randn(17, width, generator=g, dtype=dtype)
    x = base[torch.arange(2048) % 17]
    centers = torch.cat([base[:9], base[:9], -base[:9]], 0)
    before = torch.get_rng_state().clone()
    expected = torch.cdist(x, centers, p=p)
    assert torch.equal(_distance.nearest_center(x, centers, p), expected.argmin(-1))
    torch.testing.assert_close(_distance.compact_cdist(x, centers, p), expected, rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("p", [.5, 1.3, 2., 3.2815875545948123, 4.])
@pytest.mark.parametrize("scale", [0., 1e-30, 1e-12, 1., 1e20, 1e38])
def test_one_dimensional_ties_extremes(p, scale):
    centers = torch.tensor([[0.], [2.], [2.], [1.], [-1.]]) * scale
    x = torch.tensor([[1.8], [2.1], [1.5], [0.], [-0.], [-1.]]) * scale
    assert torch.equal(_distance.nearest_center(x, centers, p), torch.cdist(x, centers, p=p).argmin(-1))


@pytest.mark.parametrize("p", [.7, 2., 3.2])
def test_noncontiguous_nonfinite_empty_and_zero_width(p):
    for x, centers in [
        (torch.randn(160, 8)[:, ::2], torch.randn(20, 8)[:, ::2]),
        (torch.tensor([[float('nan'), 0.], [float('inf'), 1.]]), torch.zeros(3, 2)),
        (torch.empty(0, 2), torch.zeros(3, 2)),
        (torch.empty(160, 0), torch.empty(20, 0)),
    ]:
        expected = torch.cdist(x, centers, p=p)
        assert torch.equal(_distance.nearest_center(x, centers, p), expected.argmin(-1))
        torch.testing.assert_close(_distance.compact_cdist(x, centers, p), expected,
                                   rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_randomized_adversarial_distance_differential(dtype):
    g = torch.Generator().manual_seed(293)
    for i in range(120):
        width = [1, 2, 7, 31, 108][i % 5]
        p = float(torch.empty(()).uniform_(.5, 4., generator=g))
        scale = 10. ** ((i % 21 - 10) * (2 if dtype == torch.float32 else 15))
        x = torch.randn(39, width, dtype=dtype, generator=g) * scale
        centers = torch.randn(19, width, dtype=dtype, generator=g) * scale
        centers[5] = centers[2]
        x[0] = centers[5]
        x[1] = (centers[0] + centers[1]) / 2
        expected = torch.cdist(x, centers, p=p).argmin(-1)
        assert torch.equal(_distance.nearest_center(x, centers, p), expected), (i, width, p, scale)


def test_dead_latent_storage_released_but_output_views_survive():
    graph = RandomGraphFunction(Context(), [[], [0], [1], []],
                                [{}, {}, {"y": FeatureSpec(group="y", cat_size=0)}, {}])
    graph._fit(8)
    references = []

    def root(parents, n):
        x = torch.ones(n, 4)
        references.append(weakref.ref(x))
        return x, {}

    def child(parents, n):
        assert references[0]() is parents[0]
        return parents[0] + 1, {}

    def leaf(parents, n):
        assert references[0]() is None
        x = parents[0] + 1
        return x, {"y": x[:, :1]}

    graph.nodes_ = [root, child, leaf, lambda *_: pytest.fail("pruned node ran")]
    torch.testing.assert_close(graph._transform(8)["y"], torch.full((8, 1), 3.))


@pytest.mark.parametrize("seed", range(8))
def test_full_filtered_prior_and_rng_equal_reference(seed, monkeypatch):
    from tabicl.prior import _dataset
    from tabicl.prior.graph_lib import _function, _converter

    def generate():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        config = PriorConfig(filter_unpredictable_graphs=True, filter_unpredictable_datasets=True)
        prior = _dataset.GraphPrior(config=config, n_jobs=1)
        params = dict(regression=False, seq_len=256, train_size=180, num_features=12,
                      max_features=100, num_classes=3, config=config, device="cpu")
        output = prior.generate_dataset(params, compact=True)
        return output, torch.get_rng_state(), np.random.get_state(), random.getstate()

    actual = generate()
    with monkeypatch.context() as m:
        m.setattr(_function, "nearest_center", lambda x, c, p: torch.cdist(x, c, p=p).argmin(-1))
        m.setattr(_converter, "nearest_center", lambda x, c, p: torch.cdist(x, c, p=p).argmin(-1))
        m.setattr(_function, "compact_cdist", lambda x, c, p: torch.cdist(x, c, p=p))
        expected = generate()
    for a, b in zip(actual[0], expected[0]):
        assert torch.equal(a, b)
    assert torch.equal(actual[1], expected[1])
    assert np.array_equal(actual[2][1], expected[2][1])
    assert actual[2][2:] == expected[2][2:]
    assert actual[3] == expected[3]


def load_benchmark():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_prior_exact.py"
    spec = importlib.util.spec_from_file_location("prior_exact_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_executes_filter_and_retries_same_task(monkeypatch):
    """A worker must really filter, not just plan tasks with filtering enabled."""
    from tabicl.prior import _dataset
    benchmark = load_benchmark()
    benchmark.initialize(_dataset.GraphPrior(n_jobs=1, config=PriorConfig(
        filter_unpredictable_graphs=True, filter_unpredictable_datasets=True)), 42)
    seen = []

    def reject_first(x, y, config, **kwargs):
        assert config.filter_unpredictable_graphs
        assert config.filter_unpredictable_datasets
        seen.append((x.shape, y.shape))
        return len(seen) == 1

    monkeypatch.setattr(_dataset, "should_filter", reject_first)
    params = dict(regression=False, seq_len=256, train_size=180, num_features=12,
                  max_features=100, num_classes=3, config=benchmark.prior.config, device="cpu")
    try:
        result = benchmark.generate((0, params))
        assert result["filter_decisions"] == [True, False]
        assert [shape for shape, _ in seen] == [torch.Size([256, 100])] * 2
        assert _dataset.should_filter is reject_first
    finally:
        benchmark.limits.restore_original_limits()


def test_benchmark_rejects_worker_task_configuration_mismatch():
    from tabicl.prior import _dataset
    benchmark = load_benchmark()
    benchmark.initialize(_dataset.GraphPrior(n_jobs=1, config=PriorConfig(
        filter_unpredictable_graphs=True, filter_unpredictable_datasets=True)), 42)
    try:
        with pytest.raises(ValueError, match="worker prior configuration"):
            benchmark.generate((0, {"config": PriorConfig()}))
    finally:
        benchmark.limits.restore_original_limits()


@pytest.mark.parametrize('disabled,vectorized', [('em', True), ('distance', False)])
def test_benchmark_ablation_bypasses_fused_em(monkeypatch, disabled, vectorized):
    from types import SimpleNamespace
    from tabicl.prior import _dataset
    from tabicl.prior.graph_lib import _function, _converter
    benchmark = load_benchmark()
    # initialize() mutates worker-local functions; register their restoration
    # because this test runs in the pytest process rather than a spawned worker.
    for module in (_function, _converter):
        for name in ('nearest_center', 'compact_cdist'):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(_function.RandomEMAssignmentFunction, '_transform',
                        _function.RandomEMAssignmentFunction._transform)
    calls = []
    def distances(x, centers, p, *, vectorized):
        calls.append(vectorized)
        return torch.cdist(x, centers, p=p)
    monkeypatch.setattr(_function, 'compact_cdist', distances)
    def forbidden(*args, **kwargs):
        raise AssertionError('ablation must not execute the fused EM path')
    monkeypatch.setattr(_function, 'compact_em_probabilities', forbidden)
    benchmark.initialize(_dataset.GraphPrior(n_jobs=1), 42, disable=[disabled])
    try:
        x = torch.arange(8.).reshape(4, 2)
        centers = torch.arange(6.).reshape(3, 2)
        obj = SimpleNamespace(x_ind_=centers, p_=1.3, q_=1.7, stds_=torch.ones(3),
                              consts=torch.zeros(1, 3), out_func_=lambda t: t)
        got = _function.RandomEMAssignmentFunction._transform(obj, x)
        assert calls == [vectorized]
        torch.testing.assert_close(got, torch.softmax(-torch.cdist(x, centers, p=1.3)**1.7, -1))
    finally:
        benchmark.limits.restore_original_limits()
