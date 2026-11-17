"""Full-row equality, stable full-table statistics and fitted-state checks."""
import os

import pytest
import torch

from tabicl.prior.graph_lib import _distance as dist
from tabicl.prior.graph_lib._activation import Standardize
from tabicl.prior.graph_lib._base import Context
from tabicl.prior.graph_lib._config import PriorConfig


def native(name):
    if not dist._native_available() or not hasattr(torch.ops.tabicl_prior_exact, name):
        pytest.skip('new native CPU library not configured')
    return getattr(torch.ops.tabicl_prior_exact, name)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('d', [1, 3, 32, 100])
@pytest.mark.parametrize('kind', ['continuous', 'repeated', 'signed_zero', 'probe_collision'])
def test_hash_compaction_full_rows(dtype, d, kind):
    op = native('compact_rows_cpu')
    generator = torch.Generator().manual_seed(57)
    x = torch.randn(2050, d*2, dtype=dtype, generator=generator)[::2, ::2]
    if kind == 'repeated':
        x = x[:7].repeat(147, 1)[:1025]
    elif kind == 'signed_zero':
        x.zero_()
        x[::2] = -0.
    elif kind == 'probe_collision' and d > 4:
        x[:, :4] = 0.  # Equal probe columns must not merge unequal full rows.
    before, rng = x.clone(), torch.get_rng_state().clone()
    compact, inverse = op(x)
    recovered = compact[inverse] if len(inverse) else compact
    torch.testing.assert_close(recovered, x, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    if kind in ('repeated', 'signed_zero'):
        assert len(compact) == len(torch.unique(x, dim=0))
        assert len(inverse) == len(x)
    else:
        assert len(inverse) == 0


@pytest.mark.parametrize('n,d', [(1, 1), (17, 2), (17, 3), (2048, 8), (4096, 100)])
@pytest.mark.parametrize('kind', ['normal', 'constant', 'offset', 'tiny', 'extreme', 'nonfinite'])
def test_statistics_against_float64_oracle(n, d, kind):
    op = native('column_std_mean_cpu')
    x = torch.randn(n*2, d*2, generator=torch.Generator().manual_seed(17))[::2, ::2]
    if kind == 'constant':
        x.fill_(3.2)
    elif kind == 'offset':
        x += 1e6
    elif kind == 'tiny':
        x *= 1e-20
    elif kind == 'extreme':
        x *= 1e30
    elif kind == 'nonfinite':
        x[0, 0] = float('inf')
        if d > 1:
            x[0, 1] = float('nan')
    rng = torch.get_rng_state().clone()
    expected = torch.stack((x.double().mean(0), x.double().std(0, correction=0))).float()
    torch.testing.assert_close(op(x), expected, rtol=3e-6, atol=1e-30, equal_nan=True)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('enabled', [False, True])
def test_standardize_fitted_statistics_reused(dtype, enabled, monkeypatch):
    if enabled:
        native('column_std_mean_cpu')
    monkeypatch.setenv('TABICL_PRIOR_STATISTICS', 'native' if enabled else 'torch')
    x = torch.randn(2048, 3, dtype=dtype, generator=torch.Generator().manual_seed(31))
    x[:, 0] = 2.
    new = x[:17] + 3.
    obj = Standardize(Context(config=PriorConfig(graph_execution='optimized')))
    got = obj(x)
    expected = (x-x.mean(0))/(x.std(0, correction=0)+1e-4)
    torch.testing.assert_close(got, expected, atol=3e-6, rtol=3e-6)
    mean, inv_std = obj.mean_.clone(), obj.inv_std_.clone()
    torch.testing.assert_close(obj(new), (new-mean)*inv_std, rtol=0, atol=0)
    assert torch.equal(obj.mean_, mean) and torch.equal(obj.inv_std_, inv_std)


def test_old_library_and_no_library_fallback(monkeypatch):
    monkeypatch.setattr(dist, '_native_available', lambda: False)
    x = torch.arange(4.).repeat(256)[:, None]
    compact, inverse = dist._compact_rows(x, 8)
    assert len(compact) == 4
    assert torch.equal(compact[inverse], x)
    obj = Standardize(Context(config=PriorConfig(graph_execution='optimized')))
    torch.testing.assert_close(obj(x), (x-x.mean(0))/(x.std(0, correction=0)+1e-4))


def test_scalar_compaction_uses_torch(monkeypatch):
    def unexpected_native():
        raise AssertionError('scalar compaction should use the cheaper torch.unique')
    monkeypatch.setenv('TABICL_PRIOR_COMPACT', 'hash')
    monkeypatch.setattr(dist, '_native_available', unexpected_native)
    x = torch.tensor([3., 1., 2., 1.]).repeat(256)[:, None]
    compact, inverse = dist._compact_rows(x, 8)
    torch.testing.assert_close(compact[:, 0], torch.tensor([1., 2., 3.]), rtol=0, atol=0)
    torch.testing.assert_close(compact[inverse], x, rtol=0, atol=0)


@pytest.mark.parametrize('columns,gradient', [(1, False), (3, True)])
def test_scalar_and_autograd_use_torch(columns, gradient, monkeypatch):
    def unexpected_native():
        raise AssertionError('native prior statistics must not handle this input')
    monkeypatch.setattr(dist, '_native_available', unexpected_native)
    x = torch.randn(256, columns, requires_grad=gradient)
    obj = Standardize(Context(config=PriorConfig(graph_execution='optimized')))
    actual = obj(x)
    expected = (x-x.mean(0))/(x.std(0, correction=0)+1e-4)
    torch.testing.assert_close(actual, expected)
    if gradient:
        torch.testing.assert_close(torch.autograd.grad(actual.square().sum(), x)[0],
                                   torch.autograd.grad(expected.square().sum(), x)[0])
