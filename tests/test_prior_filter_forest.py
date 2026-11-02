"""Guard the fixed full-forest executor against sklearn's fitted estimator."""
from dataclasses import replace
import warnings

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor
import torch

from tabicl.prior import _dataset, _filter_forest
from tabicl.prior.graph_lib._config import PriorConfig


requires_abi = pytest.mark.skipif(
    _filter_forest.sklearn_tree_api() is None,
    reason='low-level full-forest executor is gated to its audited sklearn ABI')


def reference(x, y, n_estimators=25):
    return ExtraTreesRegressor(n_estimators=n_estimators, bootstrap=True,
        oob_score=True, n_jobs=1, random_state=1, max_depth=6).fit(x, y).oob_prediction_


@requires_abi
@pytest.mark.parametrize('n,d', [(32, 1), (133, 7), (257, 100), (2048, 16), (4096, 100)])
@pytest.mark.parametrize('outputs', [1, 3, 10])
@pytest.mark.parametrize('kind', ['continuous', 'discrete', 'constant_padded'])
def test_predictions_equal(n, d, outputs, kind):
    rng = np.random.RandomState(13)
    x = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.normal(size=(n, outputs))
    if kind == 'discrete':
        x = np.round(x)
        y = np.eye(outputs)[rng.randint(outputs, size=n)] if outputs > 1 else (y > 0).astype(float)
    if kind == 'constant_padded':
        x[:, d//2:] = 0
        y[:, -1] = 1
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        got = _filter_forest.planned_oob_prediction(x, y)
        expected = reference(x, y[:, 0] if outputs == 1 else y)
    np.testing.assert_array_equal(got, expected.reshape(n, outputs))


@requires_abi
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_strides_and_repeated_calls_never_mutate_data_or_plans(dtype):
    rng = np.random.RandomState(2)
    x = rng.normal(size=(513, 18)).astype(dtype)[::2, ::2]
    y = rng.normal(size=(513, 4)).astype(dtype)[::2, ::2]
    x0, y0 = x.copy(), y.copy()
    expected = reference(x, y)
    plans, votes, _ = _filter_forest._bootstrap_plan(len(x), 25)
    copies = [(w.copy(), o.copy()) for _, w, o in plans]
    for _ in range(3):
        np.testing.assert_array_equal(_filter_forest.planned_oob_prediction(x, y), expected)
    np.testing.assert_array_equal(x, x0)
    np.testing.assert_array_equal(y, y0)
    for (_, w, o), (w0, o0) in zip(plans, copies):
        np.testing.assert_array_equal(w, w0)
        np.testing.assert_array_equal(o, o0)
        assert not w.flags.writeable and not o.flags.writeable
    assert not votes.flags.writeable


@requires_abi
def test_all_trees_rebuilt_for_new_data():
    rng = np.random.RandomState(2)
    x = rng.normal(size=(512, 8)).astype(np.float32)
    for k in range(4):
        y = rng.normal(size=(512, k+1))
        expected = reference(x, y[:, 0] if k == 0 else y).reshape(y.shape)
        np.testing.assert_array_equal(_filter_forest.planned_oob_prediction(x, y), expected)


@requires_abi
@pytest.mark.parametrize('classes', [0, 2, 5, 10])
@pytest.mark.parametrize('remove_trivial', [False, True])
def test_filter_decisions_equal(classes, remove_trivial):
    rng = np.random.RandomState(12)
    x = torch.from_numpy(rng.normal(size=(2048, 20)).astype(np.float32))
    cfg = PriorConfig(filter_unpredictable_datasets=True, remove_trivial_datasets=remove_trivial,
                      filter_execution='reference')
    for signal in [False, True]:
        for _ in range(3):
            y = x[:, 0] if signal else torch.from_numpy(rng.normal(size=2048).astype(np.float32))
            if classes:
                y = torch.bucketize(y, torch.linspace(-1.5, 1.5, classes-1))
            expected = _dataset.should_filter(x, y, cfg, is_classif=bool(classes))
            got = _dataset.should_filter(x, y, replace(cfg, filter_execution='planned'), is_classif=bool(classes))
            assert got == expected


@requires_abi
def test_small_forest_zero_oob_votes_match():
    x = np.random.RandomState(7).normal(size=(16, 4)).astype(np.float32)
    y = np.arange(16.)
    with pytest.warns(UserWarning, match='Some inputs do not have OOB scores'):
        got = _filter_forest.planned_oob_prediction(x, y, 1)
    with pytest.warns(UserWarning, match='Some inputs do not have OOB scores'):
        expected = reference(x, y, 1)
    np.testing.assert_array_equal(got[:, 0], expected)


def test_version_and_nonfinite_fallback(monkeypatch):
    x, y = np.ones((32, 4), np.float32), np.ones((32, 1))
    monkeypatch.setattr(_filter_forest.sklearn, '__version__', 'unsupported')
    assert _filter_forest.planned_oob_prediction(x, y) is None
    monkeypatch.setattr(_filter_forest.sklearn, '__version__', '1.7.2')
    for val in [np.nan, np.inf, -np.inf]:
        bad = x.copy()
        bad[0, 0] = val
        assert _filter_forest.planned_oob_prediction(bad, y) is None
        bad_y = y.copy()
        bad_y[0, 0] = val
        assert _filter_forest.planned_oob_prediction(x, bad_y) is None
    assert _filter_forest.planned_oob_prediction(np.ones((8192, 1)), np.ones(8192), 25) is None


@requires_abi
def test_nan_dispatch_falls_back_to_sklearn(monkeypatch):
    x = torch.randn(128, 4)
    x[0, 0] = float('nan')
    y = torch.randint(2, size=(128,))
    calls = []
    fit = _dataset.ExtraTreesRegressor.fit
    def audit(model, *args, **kwargs):
        calls.append(model.n_estimators)
        return fit(model, *args, **kwargs)
    monkeypatch.setattr(_dataset.ExtraTreesRegressor, 'fit', audit)
    cfg = PriorConfig(filter_unpredictable_datasets=True, filter_execution='planned')
    _dataset.should_filter(x, y, cfg, is_classif=True)
    assert calls == [25]


def test_config_cli():
    import argparse
    p = argparse.ArgumentParser()
    PriorConfig.add_args_to_parser(p)
    a = p.parse_args(['--filter_execution', 'planned'])
    assert PriorConfig.from_args(a).filter_execution == 'planned'


@requires_abi
def test_seed_adapter_is_immutable_and_matches_first_numpy_draw():
    plans, _, _ = _filter_forest._bootstrap_plan(2048, 25)
    seeds = np.random.RandomState(1).randint(np.iinfo(np.int32).max, size=25)
    for (adapter, _, _), seed in zip(plans, seeds):
        expected = np.random.RandomState(int(seed)).randint(0, np.iinfo(np.int32).max)
        assert adapter.randint(0, np.iinfo(np.int32).max) == expected
        assert adapter.randint(0, np.iinfo(np.int32).max) == expected
        with pytest.raises(AttributeError):
            adapter.value = 2
        with pytest.raises(RuntimeError, match='RNG contract'):
            adapter.randint(0, 3)


def test_fallback_decision_matches_reference(monkeypatch):
    monkeypatch.setattr(_filter_forest.sklearn, '__version__', 'unsupported')
    rng = np.random.RandomState(19)
    x = torch.from_numpy(rng.normal(size=(257, 13)).astype(np.float32))
    y = torch.from_numpy(rng.randint(4, size=257))
    cfg = PriorConfig(filter_unpredictable_datasets=True)
    assert (_dataset.should_filter(x, y, cfg, is_classif=True) ==
            _dataset.should_filter(x, y, replace(cfg, filter_execution='planned'), is_classif=True))


@requires_abi
def test_cache_is_bounded_and_contains_no_fitted_models():
    for n in range(130, 136):
        _filter_forest._bootstrap_plan(n, 25)
    assert _filter_forest._bootstrap_plan.cache_info().currsize <= 4
    plans, votes, missing = _filter_forest._bootstrap_plan(2048, 25)
    assert len(plans) == 25 and votes.shape == (2048,) and isinstance(missing, bool)
    assert all(isinstance(p[0], _filter_forest._SplitterSeed) and
               p[1].shape == (2048,) and p[2].ndim == 1 for p in plans)


@requires_abi
def test_full_forest_builds_twenty_five_fresh_trees_per_table(monkeypatch):
    from sklearn.tree import _tree
    original, trees = _tree.DepthFirstTreeBuilder, []
    class Builder:
        def __init__(self, *args):
            self.builder = original(*args)
        def build(self, tree, *args):
            trees.append(tree)
            return self.builder.build(tree, *args)
    monkeypatch.setattr(_tree, 'DepthFirstTreeBuilder', Builder)
    rng = np.random.RandomState(22)
    x = rng.normal(size=(257, 8)).astype(np.float32)
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    for _ in range(2):
        y = rng.normal(size=257)
        assert _filter_forest.planned_oob_prediction(x, y).shape == (257, 1)
    assert len(trees) == 50 and len({id(t) for t in trees}) == 50
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
