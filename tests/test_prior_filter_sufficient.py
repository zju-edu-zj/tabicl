"""Full-count forests and the original complete Brier/bootstrap criterion."""
from dataclasses import replace
import warnings

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesClassifier
import torch

from tabicl.prior import _dataset, _filter_forest as forest
from tabicl.prior.graph_lib._config import PriorConfig


requires_api = pytest.mark.skipif(forest.sklearn_tree_api() is None, reason='unsupported sklearn tree API')


@requires_api
@pytest.mark.parametrize('n,d,c', [(16, 1, 3), (133, 8, 5), (257, 100, 10), (2048, 8, 5), (4096, 16, 10)])
@pytest.mark.parametrize('kind', ['continuous', 'discrete', 'constant', 'padding', 'missing_class'])
def test_counts_match_complete_classifier(n, d, c, kind):
    rng = np.random.RandomState(42)
    x = rng.normal(size=(n, d)).astype(np.float32)
    labels = rng.randint(c, size=n)
    if kind == 'discrete':
        x = np.round(x)
    if kind == 'constant':
        x[:] = 0
    if kind == 'padding':
        x[:, d//2:] = 0
    if kind == 'missing_class':
        labels[labels == 1] = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        model = ExtraTreesClassifier(n_estimators=25, bootstrap=True, oob_score=True,
                    n_jobs=1, random_state=1, max_depth=6, max_features=1.).fit(x, labels)
        expected = np.zeros((n, c))
        expected[:, model.classes_] = model.oob_decision_function_
        got = forest.planned_classification_oob_prediction(x, labels, c)
    np.testing.assert_allclose(got, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('c', [3, 5, 10])
@pytest.mark.parametrize('kind', ['random', 'perfect', 'zero_votes', 'invalid_rows', 'missing_class'])
def test_brier_without_one_hot(c, kind):
    rng = np.random.RandomState(7)
    labels = rng.randint(c, size=257)
    if kind == 'missing_class':
        labels[labels == 1] = 0
    targets = np.eye(c, dtype=np.float32)[labels]
    pred = rng.dirichlet(np.ones(c), size=len(labels))
    if kind == 'perfect':
        pred = targets.astype(float)
    if kind == 'zero_votes':
        pred[:5] = 0
    if kind == 'invalid_rows':
        pred[:5] = np.nan
    mask = ~np.isnan(pred).any(axis=1)
    base_err = ((targets[mask] - targets.mean(0))**2).sum(1)
    model_err = ((targets[mask] - pred[mask])**2).sum(1)
    imp, base, model = _dataset._classification_filter_statistics(labels, pred)
    # The old baseline is float32, while counts compute its value in float64.
    np.testing.assert_allclose(imp, base_err-model_err, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose([base, model], [base_err.mean(), model_err.mean()], atol=2e-7)
    assert _dataset._bootstrap_pvalue(imp) == _dataset._bootstrap_pvalue(base_err-model_err)


@requires_api
@pytest.mark.parametrize('classes', [0, 2, 5])
@pytest.mark.parametrize('remove_trivial', [False, True])
def test_dispatch_retains_full_forest_and_no_one_hot_for_multiclass(monkeypatch, classes, remove_trivial):
    x = torch.randn(257, 8)
    y = torch.arange(257) % classes if classes else torch.randn(257)
    cfg = PriorConfig(filter_unpredictable_datasets=True, remove_trivial_datasets=remove_trivial,
                      filter_execution='sufficient')
    builds = []
    from sklearn.tree import _tree
    original = _tree.DepthFirstTreeBuilder
    class Builder:
        def __init__(self, *args):
            self.inner = original(*args)
        def build(self, tree, *args):
            builds.append(tree)
            return self.inner.build(tree, *args)
    monkeypatch.setattr(_tree, 'DepthFirstTreeBuilder', Builder)
    if classes > 2:
        def no_one_hot(*args, **kwargs):
            raise AssertionError('sufficient path must not allocate one-hot')
        monkeypatch.setattr(_dataset.F, 'one_hot', no_one_hot)
    _dataset.should_filter(x, y, cfg, is_classif=bool(classes))
    assert len(builds) == 25 and len({id(t) for t in builds}) == 25


@requires_api
def test_sufficient_fallback_and_progressive(monkeypatch):
    x, y = torch.randn(257, 8), torch.arange(257) % 5
    cfg = PriorConfig(filter_execution='sufficient', filter_unpredictable_datasets=True)
    monkeypatch.setattr(forest.sklearn, '__version__', 'unsupported')
    expected = _dataset.should_filter(x, y, replace(cfg, filter_execution='reference'), is_classif=True)
    assert _dataset.should_filter(x, y, cfg, is_classif=True) == expected
    assert (_dataset.should_filter(x, y, replace(cfg, dataset_filter_mode='progressive'), is_classif=True)
            == _dataset.should_filter(x, y, replace(cfg, filter_execution='reference', dataset_filter_mode='progressive'), is_classif=True))


@requires_api
@pytest.mark.parametrize('bad', [np.array([-1, 1]), np.array([0, 3]), np.array([0., 1.]), np.array([[0], [1]])])
def test_unsafe_class_codes_fall_back(bad):
    assert forest.planned_classification_oob_prediction(np.ones((2, 1)), bad, 3) is None


@pytest.mark.parametrize('version,api', [('1.3.0', 'legacy'), ('1.3.2', 'legacy'),
    ('1.4.0', 'monotonic'), ('1.4.1.post1', 'monotonic'), ('1.5.2', 'monotonic'),
    ('1.6.1', 'monotonic'), ('1.7.2', 'monotonic'), ('1.8.0', 'monotonic'), ('1.9.1', 'monotonic'),
    ('1.9.1+vendor', 'monotonic'), ('1.9.0rc1', None), ('1.10.0', None), ('2.0.0', None), ('1.2.2', None)])
def test_version_contract(version, api):
    assert forest.sklearn_tree_api(version) == api


def test_config_default_and_overrides():
    import argparse
    p = argparse.ArgumentParser()
    PriorConfig.add_args_to_parser(p)
    assert PriorConfig().filter_execution == 'sufficient'
    assert p.parse_args([]).filter_execution == 'sufficient'
    for mode in ['reference', 'planned', 'sufficient']:
        assert PriorConfig.from_args(p.parse_args(['--filter_execution', mode])).filter_execution == mode


def test_shell_default_and_explicit_override():
    import os
    from pathlib import Path
    import subprocess
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop('PRIOR_FILTER_EXECUTION', None)
    command = 'source scripts/prior_recipe.sh; printf "%s" "$PRIOR_FILTER_EXECUTION"'
    assert subprocess.check_output(['bash', '-c', command], cwd=root, env=env, text=True) == 'sufficient'
    env['PRIOR_FILTER_EXECUTION'] = 'reference'
    assert subprocess.check_output(['bash', '-c', command], cwd=root, env=env, text=True) == 'reference'
