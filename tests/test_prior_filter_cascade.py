"""Progressive filtering uses the complete table and retains hard-task fallback."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from sklearn.ensemble import ExtraTreesRegressor

from tabicl.prior import _dataset
from tabicl.prior.graph_lib._config import PriorConfig


def fit_calls(monkeypatch):
    original, calls = ExtraTreesRegressor.fit, []
    def fit(model, x, y, *args, **kwargs):
        old = list(getattr(model, 'estimators_', []))
        result = original(model, x, y, *args, **kwargs)
        calls.append((model.n_estimators, len(x), old, list(model.estimators_)))
        return result
    monkeypatch.setattr(ExtraTreesRegressor, 'fit', fit)
    return calls


@pytest.mark.parametrize('rows', [2048, 4096])
def test_strong_full_table_accept_skips_remaining_trees(monkeypatch, rows):
    g = torch.Generator().manual_seed(3)
    x = torch.randn(rows, 12, generator=g)
    y = (x[:, 0] > 0).long()
    calls = fit_calls(monkeypatch)
    cfg = PriorConfig(filter_unpredictable_datasets=True, dataset_filter_mode='progressive')
    assert not _dataset.should_filter(x, y, cfg, is_classif=True)
    assert [(n, size) for n, size, _, _ in calls] == [(4, rows)]
    assert not _dataset.should_filter(x, y, replace(cfg, dataset_filter_mode='full'), is_classif=True)


def test_uncertain_task_reuses_first_four_trees(monkeypatch):
    g = torch.Generator().manual_seed(5)
    x = torch.randn(2048, 12, generator=g)
    y = torch.randint(4, (2048,), generator=g)
    calls = fit_calls(monkeypatch)
    cfg = PriorConfig(filter_unpredictable_datasets=True, dataset_filter_mode='progressive')
    actual = _dataset.should_filter(x, y, cfg, is_classif=True)
    assert [(n, size) for n, size, _, _ in calls] == [(4, 2048), (25, 2048)]
    assert all(before is after for before, after in zip(calls[0][3], calls[1][3]))
    expected = _dataset.should_filter(x, y, replace(cfg, dataset_filter_mode='full'), is_classif=True)
    assert actual == expected


@pytest.mark.parametrize('option', ['regression', 'trivial', 'small', 'large', 'trees', 'classes',
                                   'bootstrap', 'width'])
def test_unvalidated_modes_use_complete_filter(monkeypatch, option):
    g = torch.Generator().manual_seed(8)
    rows = 256 if option == 'small' else 5000 if option == 'large' else 2048
    x = torch.randn(rows, 101 if option == 'width' else 8, generator=g)
    y = torch.randint(12 if option == 'classes' else 2, (rows,), generator=g)
    cfg = PriorConfig(filter_unpredictable_datasets=True, dataset_filter_mode='progressive',
                      remove_trivial_datasets=option == 'trivial', filter_execution='reference')
    calls = fit_calls(monkeypatch)
    trees = 8 if option == 'trees' else 25
    _dataset.should_filter(x, y, cfg, is_classif=option != 'regression', n_estimators=trees,
                           n_boot=8 if option == 'bootstrap' else 200)
    assert [(n, size) for n, size, _, _ in calls] == [(trees, rows)]


def test_warm_start_has_identical_complete_forest():
    rng = np.random.default_rng(8)
    x, y = rng.normal(size=(2048, 8)), rng.integers(4, size=(2048, 3))
    kwargs = dict(bootstrap=True, oob_score=True, n_jobs=1, random_state=1, max_depth=6)
    with pytest.warns(UserWarning, match='OOB scores'):
        staged = ExtraTreesRegressor(n_estimators=4, warm_start=True, **kwargs).fit(x, y)
    staged.set_params(n_estimators=25).fit(x, y)
    complete = ExtraTreesRegressor(n_estimators=25, **kwargs).fit(x, y)
    np.testing.assert_array_equal(staged.oob_prediction_, complete.oob_prediction_)
    for a, b in zip(staged.estimators_, complete.estimators_):
        assert a.random_state == b.random_state
        np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)


@pytest.mark.parametrize('classes', [2, 3, 10])
def test_xor_preserves_hard_task_fallback_and_rng(classes):
    g = torch.Generator().manual_seed(19)
    x = torch.randn(2048, 20, generator=g)
    if classes == 2:
        y = ((x[:, 0] > 0) != (x[:, 1] > 0)).long()
    else:
        y = torch.floor(torch.sigmoid(x[:, 0]*x[:, 1])*classes).long().clamp_max(classes-1)
    cfg = PriorConfig(filter_unpredictable_datasets=True, dataset_filter_mode='progressive')
    state = torch.get_rng_state().clone()
    full = _dataset.should_filter(x, y, replace(cfg, dataset_filter_mode='full'), is_classif=True)
    progressive = _dataset.should_filter(x, y, cfg, is_classif=True)
    assert full == progressive
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize('rows', [2048, 4096])
@pytest.mark.parametrize('signal', [False, True])
@pytest.mark.parametrize('flip', [False, True])
def test_rare_class_and_label_flip_match_full_filter(rows, signal, flip):
    # A four-tree forest has rows with no OOB vote. Check both label encodings:
    # sklearn's zero-filled predictions must not turn rare pure noise into signal.
    g = torch.Generator().manual_seed(1000)
    x = torch.randn(rows, 12, generator=g)
    y = (x[:, 0] > 2).long() if signal else (torch.rand(rows, generator=g) < .02).long()
    if flip:
        y = 1 - y
    cfg = PriorConfig(filter_unpredictable_datasets=True, dataset_filter_mode='progressive')
    full = _dataset.should_filter(x, y, replace(cfg, dataset_filter_mode='full'), is_classif=True)
    progressive = _dataset.should_filter(x, y, cfg, is_classif=True)
    assert full == progressive
    assert full == (not signal)
