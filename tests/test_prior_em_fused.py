"""Retain the full EM component, its fitted state and replay behavior."""
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from tabicl.prior.graph_lib import _distance as dist
from tabicl.prior.graph_lib._base import Context
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._function import RandomEMAssignmentFunction


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def native():
    if not dist._native_available() or not hasattr(torch.ops.tabicl_prior_exact, 'em_probabilities_cpu'):
        pytest.skip('new native EM library not configured')
    return torch.ops.tabicl_prior_exact.em_probabilities_cpu


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('p,q', [(.5, 1.1), (1., 1.), (1.3, 1.8), (2., 1.5), (3.9, 2.)])
@pytest.mark.parametrize('n,d,k', [(1, 1, 2), (17, 3, 5), (133, 20, 16), (2048, 8, 33)])
def test_fused_full_posterior(dtype, p, q, n, d, k):
    op = native()
    gen = torch.Generator().manual_seed(n+d+k)
    # Strided inputs, duplicates, exact zero distances and SIMD tail rows.
    x = torch.randn(n*2, d*2, generator=gen, dtype=dtype)[::2, ::2]
    centers = torch.randn(k, d, generator=gen, dtype=dtype)
    centers[0] = x[0]
    stds = torch.linspace(.2, 3., k, dtype=dtype)
    const = -torch.log(2*torch.pi*stds[None, :]**2)/2
    expected = torch.softmax(const-(torch.cdist(x, centers, p=p,
        compute_mode='donot_use_mm_for_euclid_dist')/stds)**q, -1)
    got = op(x, centers, stds.pow(-q), const, p, q)
    # Sharp softmaxes amplify float32 distance rounding (especially p < 1).
    # This compares complete probabilities, not an intermediate distance.
    tol = 6e-5 if dtype == torch.float32 else 2e-13
    torch.testing.assert_close(got, expected, atol=tol, rtol=tol)
    torch.testing.assert_close(got.sum(-1), torch.ones(n, dtype=dtype))
    assert (got >= 0).all()


@pytest.mark.parametrize('fallback', [False, True])
@pytest.mark.parametrize('p', [1.3, 2.])
def test_compact_chain_matches_and_retains_p2_dispatch(monkeypatch, fallback, p):
    if fallback:
        monkeypatch.setattr(dist, '_native_available', lambda: False)
    else:
        native()
    x = torch.arange(5.).repeat(410)[:2048, None]
    c = torch.linspace(-1, 6, 16)[:, None]
    stds = torch.linspace(.3, 2., len(c))
    const = -torch.log(2*torch.pi*stds[None, :]**2)/2
    probs, inverse = dist.compact_em_probabilities(x, c, stds, const, p, 1.7, stds.pow(-1.7))
    if p == 2.:
        assert inverse is None
    else:
        assert probs.shape[0] == 5 and inverse.shape == (2048,)
    got = probs if inverse is None else probs[inverse]
    expected = torch.softmax(const-(torch.cdist(x, c, p=p)/stds)**1.7, -1)
    torch.testing.assert_close(got, expected, atol=8e-6, rtol=8e-6)


@pytest.mark.parametrize('ensure_iid', [False, True])
@pytest.mark.parametrize('repeated', [False, True])
def test_fitted_component_and_new_rows_replay(ensure_iid, repeated):
    native()
    cfg = PriorConfig(graph_execution='optimized', categorical_sampling='marginal', ensure_iid=ensure_iid)
    x = torch.randn(2048, 3, generator=torch.Generator().manual_seed(38))
    if repeated:
        x = x[:4].repeat(512, 1)
    outputs = []
    states = []
    for mode in ['reference', 'optimized']:
        random.seed(12)
        np.random.seed(12)
        torch.manual_seed(12)
        obj = RandomEMAssignmentFunction(Context(config=replace(cfg, graph_execution=mode), seed=12), 3, 8)
        train = obj(x)
        test = obj(x[:17] + .05)
        outputs.append((train, test))
        states.append((obj.x_ind_.clone(), obj.out_func_.matrix_.clone(), torch.get_rng_state().clone()))
    for a, b in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    for a, b in zip(states[0], states[1]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('d,k,repeated,q,expected', [
    (32, 8, False, 1.7, 'distance'),
    (32, 5, False, 1.7, 'fused'),
    (8, 8, False, 1.7, 'fused'),
    (32, 8, True, 1.7, 'fused'),
    (32, 8, False, 1.3, 'fused'),
])
def test_shape_dispatch_keeps_complete_metric(monkeypatch, d, k, repeated, q, expected):
    native()
    ops = torch.ops.tabicl_prior_exact
    calls = []
    for name, key in [('cdist_vector_cpu', 'distance'), ('em_probabilities_cpu', 'fused')]:
        original = getattr(ops, name)
        def record(*args, original=original, key=key):
            calls.append(key)
            return original(*args)
        monkeypatch.setattr(ops, name, record)
    gen = torch.Generator().manual_seed(123)
    x = torch.randn(256, d, generator=gen)
    if repeated:
        x = x[:4].repeat(64, 1)
    centers = torch.randn(k, d, generator=gen)
    stds = torch.linspace(1., 10., k)
    const = -torch.log(2*torch.pi*stds[None, :]**2)/2
    result, inverse = dist.compact_em_probabilities(x, centers, stds, const, 1.3, q, stds.pow(-q))
    assert calls == [expected]
    if inverse is not None:
        result = result[inverse]
    oracle = torch.softmax(const.double()-(torch.cdist(x.double(), centers.double(), p=1.3)
        /stds.double())**q, -1).float()
    torch.testing.assert_close(result, oracle, atol=6e-5, rtol=6e-5)


def test_mixed_scale_dtype_retains_torch_promotion():
    native()
    gen = torch.Generator().manual_seed(28)
    x = torch.randn(133, 3, dtype=torch.float64, generator=gen)
    centers = torch.randn(5, 3, dtype=torch.float64, generator=gen)
    stds = torch.linspace(.3, 2., 5, dtype=torch.float32)
    const = -torch.log(2*torch.pi*stds[None, :]**2)/2
    got, inverse = dist.compact_em_probabilities(x, centers, stds, const, 1.3, 1.7, stds.pow(-1.7))
    assert inverse is None and got.dtype == torch.float64
    expected = torch.softmax(const-(torch.cdist(x, centers, p=1.3)/stds)**1.7, -1)
    torch.testing.assert_close(got, expected, atol=2e-13, rtol=2e-13)
