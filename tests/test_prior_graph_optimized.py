"""Full-complexity execution: polynomial identity, distances and live latents."""
from dataclasses import replace
import random

import numpy as np
import pytest
import torch

from tabicl.prior.graph_lib._base import Context
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._converter import CategoricalConverter, CategoricalSoftmaxDiscretizer
from tabicl.prior.graph_lib._distance import _native_available, compact_cdist, nearest_center
from tabicl.prior.graph_lib._function import RandomQuadraticFunction
from tabicl.prior.graph_lib._points import RandomPoints


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def native(name):
    if not _native_available() or not hasattr(torch.ops.tabicl_prior_exact, name):
        pytest.skip('build the updated native prior library')
    return getattr(torch.ops.tabicl_prior_exact, name)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('p', [.5, .713, 1., 1.999, 2., 3.2837, 4.])
@pytest.mark.parametrize('width,count', [(1, 1), (2, 3), (3, 8), (11, 17), (108, 35)])
def test_vector_distances_keep_metric_and_all_dimensions(dtype, p, width, count):
    near, dist = native('nearest_vector_cpu'), native('cdist_vector_cpu')
    g = torch.Generator().manual_seed(728)
    x = torch.randn(67, width*2, dtype=dtype, generator=g)[:, ::2]
    centers = torch.randn(count, width*2, dtype=dtype, generator=g)[:, ::2]
    ref = torch.cdist(x, centers, p=p, compute_mode='donot_use_mm_for_euclid_dist')
    state = torch.get_rng_state().clone()
    torch.testing.assert_close(dist(x, centers, p), ref, rtol=8e-6, atol=1e-6)
    idx = near(x, centers, p)
    torch.testing.assert_close(ref.gather(1, idx[:, None]).squeeze(1), ref.min(1).values,
                               rtol=8e-6, atol=1e-6)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize('rows,count,width', [(0, 7, 2), (3, 0, 2), (3, 5, 0)])
def test_native_distance_empty_shapes(rows, count, width):
    op = native('cdist_vector_cpu')
    x, c = torch.zeros(rows, width), torch.zeros(count, width)
    torch.testing.assert_close(op(x, c, 1.3), torch.cdist(x, c, p=1.3))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('p', [.5, .873, 1., 1.999, 2., 3.7, 4.])
def test_nearest_structured_repeated_rows_and_heterogeneous_dimensions(dtype, p):
    near, dist = native('nearest_vector_cpu'), native('cdist_vector_cpu')
    g = torch.Generator().manual_seed(972)
    # Include zeros, exact duplicates, partial SIMD blocks and coordinates
    # spanning twelve orders of magnitude. Keep the important coordinate last.
    x = torch.randn(43, 15, generator=g, dtype=dtype)
    x[1:7] = x[0]
    scales = torch.logspace(-6, 6, 15, dtype=dtype)
    x *= scales
    c = torch.cat([x[:11], x[:3], torch.zeros(1, 15, dtype=dtype)])
    ref = torch.cdist(x, c, p=p, compute_mode='donot_use_mm_for_euclid_dist')
    idx = near(x, c, p)
    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(ref.gather(1, idx[:, None]).flatten(), ref.min(1).values,
                               rtol=tolerance, atol=1e-7)
    torch.testing.assert_close(dist(x, c, p), ref, rtol=tolerance, atol=1e-7)
    # A coordinate permutation must retain the metric and all informative dims.
    perm = torch.randperm(15, generator=g)
    idx = near(x[:, perm], c[:, perm], p)
    torch.testing.assert_close(ref.gather(1, idx[:, None]).flatten(), ref.min(1).values,
                               rtol=tolerance, atol=1e-7)


def test_distance_validation_and_python_fallback(monkeypatch):
    op = native('nearest_vector_cpu')
    with pytest.raises(RuntimeError):
        op(torch.zeros(3, 2), torch.zeros(4, 1), 1.3)
    with pytest.raises(RuntimeError):
        op(torch.zeros(3, 2), torch.zeros(0, 2), 1.3)
    monkeypatch.delenv('TABICL_PRIOR_DISTANCE_LIBRARY', raising=False)
    for x in (torch.randn(139, 3), torch.tensor([[float('nan'), 1., 2.]])):
        c = torch.randn(17, 3)
        assert torch.equal(nearest_center(x, c, 1.3, vectorized=True),
                           nearest_center(x, c, 1.3))
        torch.testing.assert_close(compact_cdist(x, c, 1.3, vectorized=True),
                                   compact_cdist(x, c, 1.3), equal_nan=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('width,outputs', [(1, 1), (5, 3), (20, 1), (20, 57)])
def test_quadratic_basis_retains_every_coefficient(dtype, width, outputs):
    phi = native('quadratic_features_cpu')
    g = torch.Generator().manual_seed(28)
    x = torch.randn(71, width*2, dtype=dtype, generator=g)[:, ::2]
    q = torch.randn(outputs, width, width, dtype=dtype, generator=g)
    i, j = torch.triu_indices(width, width)
    coeff = (q[:, i, j] + q[:, j, i] * (i != j)).t().contiguous()
    torch.testing.assert_close(phi(x), x[:, i]*x[:, j], rtol=0, atol=0)
    torch.testing.assert_close(phi(x) @ coeff, torch.einsum('oij,bi,bj->bo', q, x, x),
                               rtol=5e-5, atol=2e-4 if dtype == torch.float32 else 1e-10)


@pytest.mark.parametrize('width,outputs', [(1, 1), (8, 3), (23, 41)])
@pytest.mark.parametrize('library', [True, False])
def test_fitted_quadratic_replay_and_fallback(width, outputs, library, monkeypatch):
    if not library:
        monkeypatch.delenv('TABICL_PRIOR_DISTANCE_LIBRARY', raising=False)
    torch.manual_seed(52)
    x = torch.randn(97, width)
    f = RandomQuadraticFunction(Context(config=PriorConfig(graph_execution='optimized'), seed=39), width, outputs)
    actual = f(x)
    state = torch.get_rng_state().clone()
    f.config = replace(f.config, graph_execution='reference')
    expected = f(x)
    torch.testing.assert_close(actual, expected, rtol=8e-5, atol=3e-5)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize('mode', ['softmax_id', 'softmax_int'])
@pytest.mark.parametrize('ensure_iid', [False, True])
def test_unused_embedding_is_never_generated_but_index_law_is_retained(mode, ensure_iid, monkeypatch):
    def forbidden(*args):
        raise AssertionError('unobserved embedding was generated')
    monkeypatch.setattr(RandomPoints, 'sample', forbidden)
    cfg = PriorConfig(graph_execution='optimized', categorical_sampling='marginal',
                      cat_modes=mode, ensure_iid=ensure_iid)
    conv = CategoricalConverter(Context(config=cfg, seed=17), 5)
    x = torch.randn(5000, 5)
    if ensure_iid:
        conv(x)
    values, idx = conv(x)
    assert conv.disc.embeddings_ is None
    assert idx.shape == (5000, 1) and 0 <= idx.min() <= idx.max() < 5
    if mode == 'softmax_id':
        assert values is x
    else:
        assert torch.equal(values, idx.float())


@pytest.mark.parametrize('global_mode,corrected', [(False, False), (True, False), (False, True)])
def test_shared_cache_modes_and_live_embedding_are_retained(global_mode, corrected, monkeypatch):
    calls = []
    def points(self, rows, width):
        calls.append((rows, width))
        return torch.arange(rows*width).reshape(rows, width).float()
    monkeypatch.setattr(RandomPoints, 'sample', points)
    cfg = PriorConfig(graph_execution='optimized', meta_sampling_mode='global' if global_mode else 'meta',
                      use_corrected_cat_meta_sampling=corrected)
    disc = CategoricalSoftmaxDiscretizer(Context(config=cfg, seed=11), 4,
                                        return_values=not (global_mode or corrected))
    out, idx = disc(torch.randn(32, 4))
    assert calls == [(4, 4)]
    if not (global_mode or corrected):
        assert torch.equal(out, disc.embeddings_[idx[:, 0]])


@pytest.mark.parametrize('mode', ['neighbor_id', 'neighbor_int', 'neighbor_disc', 'neighbor_func',
                                   'softmax_id', 'softmax_int', 'softmax_disc'])
@pytest.mark.parametrize('corrected_num', [False, True])
def test_complete_prior_all_converter_modes_and_split_coverage(mode, corrected_num):
    from tabicl.prior._dataset import GraphPrior
    random.seed(137)
    np.random.seed(137)
    torch.manual_seed(137)
    cfg = PriorConfig(graph_execution='optimized', categorical_sampling='marginal', cat_modes=mode,
                      use_corrected_num_converters=corrected_num, min_n_nodes=2, max_n_nodes=6,
                      filter_unpredictable_datasets=True, filter_unpredictable_graphs=True)
    prior = GraphPrior(config=cfg, n_jobs=1)
    params = dict(regression=False, seq_len=256, train_size=160, num_features=9,
                  max_features=100, num_classes=3, config=cfg, device='cpu')
    x, y, d = prior.generate_dataset(params, compact=True)
    assert x.shape == (256, int(d))
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    assert set(y[:160].tolist()) == set(y[160:].tolist()) and len(set(y.tolist())) > 1


def test_config_parser_and_recipe_propagate_execution_mode(tmp_path):
    import os
    from pathlib import Path
    import shlex
    import subprocess
    from tabicl.train._train_config import build_parser
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, DRY_RUN='1', PRIOR_GRAPH_EXECUTION='optimized',
               CKPT_DIR=str(tmp_path/'ckpt'), LOG_DIR=str(tmp_path/'logs'))
    text = subprocess.check_output(['bash', str(root/'scripts/train_v2_clf_stage1.sh')], env=env, text=True)
    tokens = shlex.split(text)
    args = build_parser().parse_args(tokens[tokens.index('--wandb_log'):])
    assert PriorConfig.from_args(args).graph_execution == 'optimized'


@pytest.mark.parametrize('disabled', [[], ['distance'], ['quadratic'], ['embedding'],
                                     ['distance', 'quadratic'], ['embedding', 'quadratic'],
                                     ['embedding', 'distance']])
def test_benchmark_ablation_combinations_do_not_change_component_constructors(disabled, monkeypatch):
    from pathlib import Path
    from tabicl.prior._dataset import GraphPrior
    from tabicl.prior.graph_lib import _converter, _function
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from benchmark_prior_exact import initialize
    # Record each target so monkeypatch restores the benchmark-only mutations.
    for module in (_function, _converter):
        for name in ('nearest_center', 'compact_cdist'):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, getattr(module, name))
    for cls in (RandomQuadraticFunction, CategoricalSoftmaxDiscretizer,
                _converter.CategoricalNeighborDiscretizer):
        monkeypatch.setattr(cls, '__init__', cls.__init__)
    cfg = PriorConfig(graph_execution='optimized', categorical_sampling='marginal')
    initialize(GraphPrior(config=cfg, n_jobs=1), 83, disabled)
    f = RandomQuadraticFunction(Context(config=cfg, seed=41), 6, 3)
    assert f(torch.randn(19, 6)).shape == (19, 3)
    assert (f.config.graph_execution == 'reference') == ('quadratic' in disabled)
    disc = CategoricalSoftmaxDiscretizer(Context(config=cfg, seed=42), 4, return_values=False)
    values, idx = disc(torch.randn(19, 4))
    assert idx.shape == (19, 1)
    assert (values is not None) == ('embedding' in disabled)
