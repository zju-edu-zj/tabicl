"""Full-tree execution keeps all sampled mechanisms."""
import pytest
import torch

from tabicl.prior.graph_lib._base import Context
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.prior.graph_lib._distance import _native_available
from tabicl.prior.graph_lib._function import RandomTreeFunction


@pytest.fixture(scope='module', autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def native_tree():
    if not _native_available() or not hasattr(torch.ops.tabicl_prior_exact, 'tree_cpu'):
        pytest.skip('build scripts/build_prior_exact.py and set TABICL_PRIOR_DISTANCE_LIBRARY')
    return torch.ops.tabicl_prior_exact.tree_cpu


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('trees,depth,outputs', [(1, 1, 1), (5, 4, 17), (128, 7, 120)])
@pytest.mark.parametrize('rows', [0, 1, 2048])
def test_fused_all_trees(dtype, trees, depth, outputs, rows):
    op = native_tree()
    g = torch.Generator().manual_seed(4)
    x = torch.randn(rows, 18, dtype=dtype, generator=g)[:, ::2]
    dims = torch.randint(9, (trees*depth,), generator=g)
    splits = torch.randn(trees*depth, dtype=dtype, generator=g)
    values = torch.randn(trees, 2**depth, outputs, dtype=dtype, generator=g)
    leaf = ((x[:, dims] > splits).reshape(rows, trees, depth).long() *
            (2**torch.arange(depth))).sum(-1)
    expected = values[torch.arange(trees)[None], leaf].mean(1)
    state = torch.get_rng_state().clone()
    actual = op(x, dims, splits, values)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert torch.equal(state, torch.get_rng_state())


def test_fused_tree_special_values_and_rejects_bad_dimensions():
    op = native_tree()
    x = torch.tensor([[float('nan')], [float('inf')], [float('-inf')], [0.]])
    dims, splits, values = torch.tensor([0]), torch.tensor([0.]), torch.tensor([[[2.], [3.]]])
    assert op(x, dims, splits, values).flatten().tolist() == [2., 3., 2., 2.]
    with pytest.raises(RuntimeError):
        op(x, torch.tensor([1]), splits, values)


def test_tree_fits_full_population_once(monkeypatch):
    native_tree()
    torch.manual_seed(3)
    x = torch.randn(2048, 11)
    tree = RandomTreeFunction(Context(config=PriorConfig()), 11, 15)
    monkeypatch.setenv('TABICL_PRIOR_TREE', 'torch')
    expected = tree(x)
    state, thresholds = torch.get_rng_state().clone(), tree.split_points_.clone()
    monkeypatch.setenv('TABICL_PRIOR_TREE', 'native')
    actual = tree(x)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert torch.equal(tree.split_points_, thresholds)
    assert torch.equal(torch.get_rng_state(), state)


def test_no_native_library_falls_back(monkeypatch):
    monkeypatch.delenv('TABICL_PRIOR_DISTANCE_LIBRARY', raising=False)
    monkeypatch.setenv('TABICL_PRIOR_TREE', 'native')
    x = torch.randn(13, 3)
    tree = RandomTreeFunction(Context(), 3, 5)
    assert tree(x).shape == (13, 5)
