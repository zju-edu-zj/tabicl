"""Independent split oracle, tree invariants and full bootstrap OOB accounting."""
import os
import warnings

import numpy as np
import pytest
import torch

from tabicl.prior import _filter_forest as forest
from tabicl.prior.graph_lib._distance import _native_available


@pytest.fixture(scope='module', autouse=True)
def runtime():
    if not _native_available() or not hasattr(torch.ops.tabicl_prior_exact, 'forest_tree_cpu'):
        pytest.skip('build the current prior runtime and set TABICL_PRIOR_DISTANCE_LIBRARY')
    torch.set_num_threads(1)


class Random:
    def __init__(self, seed):
        self.state = seed
    def next(self):
        value = self.state or 1
        value ^= (value << 13) & 0xffffffff
        value ^= value >> 17
        value ^= (value << 5) & 0xffffffff
        self.state = value
        return value % 2**31


def data(n, d, classes, kind='random'):
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n, d)).astype(np.float32)
    if kind == 'discrete':
        x = x.round()
    if kind == 'padding':
        x[:, d//2:] = 0
    if kind == 'constant':
        x[:] = 0
    if kind == 'near_constant':
        x[:] = 1 + x*1e-8
    y = rng.integers(classes, size=(n, 1)).astype(np.float64) if classes else rng.normal(size=(n, 2))
    w = rng.multinomial(n, np.ones(n)/n).astype(np.float64)
    return x, y, w


def tree(x, y, w, seed, classes, depth):
    return [v.numpy() for v in torch.ops.tabicl_prior_exact.forest_tree_cpu(
        torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w), seed, classes, depth)]


def independent_split(x, y, w, seed, classes):
    rng = Random(seed)
    active = w > 0
    target = np.eye(classes)[y[:, 0].astype(int)] if classes else y
    best = None
    candidates = list(range(x.shape[1]))
    for remaining in range(x.shape[1], 0, -1):
        j = rng.next() % remaining
        feature = candidates[j]
        candidates[j], candidates[remaining-1] = candidates[remaining-1], candidates[j]
        low, high = x[active, feature].min(), x[active, feature].max()
        if high <= np.float32(low + np.float32(1e-7)):
            continue
        threshold = float(low)+(float(high)-float(low))*(rng.next()/(2**31-1))
        if threshold == high:
            threshold = float(low)
        left = active & (x[:, feature].astype(float) <= threshold)
        right = active & ~left
        if not left.any() or not right.any():
            continue
        # Independent objective: directly sum weighted squared residuals or
        # Gini over both complete child subsets, no right=total-left shortcut.
        error = 0.
        for mask in (left, right):
            values, weights = target[mask], w[mask]
            mean = np.average(values, axis=0, weights=weights)
            if classes:
                error += weights.sum() * (1-mean@mean)
            else:
                error += ((values-mean)**2*weights[:, None]).sum()
        if best is None or error < best[0]:
            best = (error, feature, threshold)
    return best


@pytest.mark.parametrize('classes', [0, 2, 3, 10])
@pytest.mark.parametrize('kind', ['random', 'discrete', 'padding', 'constant', 'near_constant'])
@pytest.mark.parametrize('seed', [0, 1, 2147483646])
def test_single_node_matches_bruteforce(classes, kind, seed):
    x, y, w = data(133, 12, classes, kind)
    nodes, _ = tree(x, y, w, seed, classes, 1)
    expected = independent_split(x, y, w, seed, classes)
    if expected is None:
        assert len(nodes) == 1 and nodes[0, 0] == -1
    else:
        assert nodes[0, 0] == expected[1]
        assert nodes[0, 1] == pytest.approx(expected[2], abs=1e-14)


@pytest.mark.parametrize('classes', [0, 2, 5, 10])
@pytest.mark.parametrize('kind', ['random', 'discrete', 'padding', 'constant'])
@pytest.mark.parametrize('n', [16, 257, 2048, 4096])
def test_complete_tree_leaf_values_and_constraints(classes, kind, n):
    x, y, w = data(n, 8, classes, kind)
    nodes, values = tree(x, y, w, 71, classes, 6)
    target = np.eye(classes)[y[:, 0].astype(int)] if classes else y
    def visit(index, mask, depth):
        feature, threshold, left, right, count, weight = nodes[index]
        assert count == mask.sum()
        assert weight == pytest.approx(w[mask].sum())
        np.testing.assert_allclose(values[index], np.average(target[mask], axis=0, weights=w[mask]), atol=1e-14)
        assert depth <= 6
        if feature < 0:
            assert left == right == -1
            return
        assert depth < 6
        split = x[:, int(feature)].astype(float) <= threshold
        assert (mask & split).any() and (mask & ~split).any()
        visit(int(left), mask & split, depth+1)
        visit(int(right), mask & ~split, depth+1)
    visit(0, w > 0, 0)
    assert len(nodes) <= 127


@pytest.mark.parametrize('classes', [0, 2, 5])
def test_full_forest_oob_uses_exact_bootstrap_and_all_trees(classes, monkeypatch):
    monkeypatch.setenv('TABICL_PRIOR_FOREST', 'native')
    x, y, _ = data(133, 8, classes)
    plans, votes, _ = forest._bootstrap_plan(len(x), 25)
    expected = np.zeros((len(x), classes or y.shape[1]))
    for seed, w, oob in plans:
        nodes, values = tree(x, y, w.copy(), seed.value, classes, 6)
        for row in oob:
            index = 0
            while nodes[index, 0] >= 0:
                feature, threshold, left, right = nodes[index, :4]
                index = int(left if float(x[row, int(feature)]) <= threshold else right)
            expected[row] += values[index]
    expected /= votes[:, None]
    got = forest._planned_prediction(x, y, 25, classes or None)
    np.testing.assert_allclose(got, expected, atol=1e-14)


@pytest.mark.parametrize('classes', [0, 3])
def test_pure_targets_depth_zero_and_zero_oob(classes):
    x, y, w = data(16, 3, classes)
    y[:] = 0
    nodes, values = tree(x, y, w, 1, classes, 6)
    assert len(nodes) == 1
    np.testing.assert_allclose(values[0], np.eye(classes)[0] if classes else [0, 0])
    y = data(16, 3, classes)[1]
    nodes, _ = tree(x, y, w, 1, classes, 0)
    assert len(nodes) == 1
    result = torch.ops.tabicl_prior_exact.forest_oob_cpu(torch.from_numpy(x), torch.from_numpy(y),
        torch.ones((25, 16), dtype=torch.float64), torch.arange(25), classes, 6)
    assert torch.count_nonzero(result) == 0


@pytest.mark.parametrize('bad', ['nan_x', 'inf_y', 'negative_weight', 'zero_weight', 'invalid_class', 'fractional_class'])
def test_native_boundary_rejects_unsafe_inputs(bad):
    x, y, w = data(16, 3, 3)
    if bad == 'nan_x': x[0, 0] = np.nan
    if bad == 'inf_y': y[0, 0] = np.inf
    if bad == 'negative_weight': w[0] = -1
    if bad == 'zero_weight': w[:] = 0
    if bad == 'invalid_class': y[0, 0] = 3
    if bad == 'fractional_class': y[0, 0] = .5
    with pytest.raises(RuntimeError):
        tree(x, y, w, 1, 3, 6)


def test_missing_runtime_never_silently_uses_another_algorithm(monkeypatch):
    monkeypatch.setenv('TABICL_PRIOR_FOREST', 'native')
    monkeypatch.setattr('tabicl.prior.graph_lib._distance._native_available', lambda: False)
    x, y, _ = data(16, 3, 0)
    with pytest.raises(RuntimeError, match='current prior runtime'):
        forest._planned_prediction(x, y, 25)


def test_native_forest_does_not_depend_on_sklearn_private_tree_api(monkeypatch):
    monkeypatch.setenv('TABICL_PRIOR_FOREST', 'native')
    monkeypatch.setattr(forest.sklearn, '__version__', 'unsupported')
    x, y, _ = data(133, 8, 3)
    result = forest.planned_classification_oob_prediction(x, y[:, 0].astype(np.int64), 3)
    assert result.shape == (133, 3)
    np.testing.assert_allclose(result.sum(1), 1.)


def test_native_wrapper_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv('TABICL_PRIOR_FOREST', 'typo')
    x, y, _ = data(16, 3, 0)
    with pytest.raises(ValueError, match='TABICL_PRIOR_FOREST'):
        forest._planned_prediction(x, y, 25)


@pytest.mark.parametrize('unsupported', ['features', 'outputs', 'classes', 'trees'])
def test_native_wrapper_allows_public_fallback_outside_kernel_limits(unsupported, monkeypatch):
    monkeypatch.setenv('TABICL_PRIOR_FOREST', 'native')
    x, y, _ = data(16, 3, 0)
    classes, trees = None, 25
    if unsupported == 'features': x = np.zeros((16, 4097), dtype=np.float32)
    if unsupported == 'outputs': y = np.zeros((16, 257), dtype=np.float64)
    if unsupported == 'classes':
        y = np.zeros((16, 1), dtype=np.float64)
        classes = 257
    if unsupported == 'trees': trees = 257
    # Unsupported sizes must not allocate plans or enter the native kernel.
    def unexpected_plan(*args):
        pytest.fail('unsupported input reached bootstrap allocation')
    monkeypatch.setattr(forest, '_native_bootstrap_plan', unexpected_plan)
    assert forest._planned_prediction(x, y, trees, classes) is None
