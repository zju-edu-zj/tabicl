"""Marginalization applies only to the fresh, exchangeable legacy latent."""
import numpy as np
import pytest
import torch

from tabicl.prior.graph_lib._base import GlobalSampler
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._weights import SimpleRandomWeights
from test_node_prior_pool import seed


def sampler(mode='marginal', **kwargs):
    return GlobalSampler(seed=19, config=PriorConfig(categorical_sampling=mode, **kwargs))


def test_marginal_never_constructs_throwaway_weights(monkeypatch):
    def forbidden(*args):
        raise AssertionError('latent weights should be integrated out')
    monkeypatch.setattr(SimpleRandomWeights, 'sample', forbidden)
    s = sampler()
    for k in (1, 2, 3, 8, 100):
        assert 0 <= s.categorical('same_name', k) < k


@pytest.mark.parametrize('corrected', [False, True])
def test_corrected_and_existing_legacy_caches_keep_original_behavior(corrected):
    streams = []
    for mode in ('legacy', 'marginal'):
        seed(827)
        s = sampler(mode, use_corrected_cat_meta_sampling=corrected)
        if not corrected:
            # Existing legacy name cache is unusual, but do not reinterpret it.
            s.globals['choice'] = True
            s.globals['choice__meta_weights'] = torch.tensor([.01, .98, .01])
        streams.append([s.categorical('choice', 3) for _ in range(1000)])
    assert streams[0] == streams[1]


@pytest.mark.parametrize('meta_mode', ['local', 'global'])
def test_nonmeta_configuration_and_global_identity_unchanged(meta_mode):
    streams = []
    for mode in ('legacy', 'marginal'):
        seed(17)
        s = sampler(mode, meta_sampling_mode=meta_mode)
        streams.append([s.categorical('x', 9) for _ in range(100)])
        assert len({s.categorical('cached', 9, mode='global') for _ in range(20)}) == 1
    assert streams[0] == streams[1]


@pytest.mark.parametrize('k', [2, 7, 25])
def test_legacy_and_marginal_uniform_law_and_no_repeated_name_correlation(k):
    # Fixed seeds, generous 7-sigma bound; test frequencies AND temporal pairs.
    for mode in ('legacy', 'marginal'):
        seed(786)
        s = sampler(mode)
        values = np.array([s.categorical('reused', k) for _ in range(12000)])
        counts = np.bincount(values, minlength=k)
        assert np.max(abs(counts-12000/k)) < 7*np.sqrt(12000/k)
        matches = (values[::2] == values[1::2]).sum()
        assert abs(matches-6000/k) < 7*np.sqrt(6000/k)
