import pytest
import torch

from tabicl._model.content_index_v3 import ContentIndexV3, DualPathLayout
from tabicl._model.dual_path_store import DualPathPageStore
from tabicl._model.encoders import Encoder
from tabicl._model.layerwise_inference import layerwise_dual_path_forward


def _index(key, support):
    layout = DualPathLayout()
    initial = torch.arange(support).expand(key.shape[0], key.shape[1], -1)
    return ContentIndexV3.build(
        key, layout, use_triton=False, initial_order=initial
    )


@pytest.mark.parametrize("support", [31, 32, 33, 127, 128, 129])
def test_logical_leaf_and_physical_page_tails(support):
    key = torch.randn(1, 2, support, 16)
    value = torch.randn_like(key)
    index = _index(key, support)
    store = DualPathPageStore.pack(key, value, index)
    assert store.key.shape[-2:] == (128, 16)
    assert index.leaf_lengths.sum() == support
    assert index.leaf_lengths.max() <= 32
    route = index.search(torch.randn(1, 2, 3, 16))
    assert (route.candidate_rows == support).all() if support <= 1024 else True


def test_page_fetch_excludes_unselected_rows_from_same_page():
    support = 128
    key = torch.arange(1 * 1 * support * 4, dtype=torch.float32).reshape(1, 1, support, 4)
    value = -key
    index = _index(key, support)
    store = DualPathPageStore.pack(key, value, index)
    selected = torch.tensor([[[[2]]]], dtype=torch.int32)
    candidates = store.candidate_tensors(selected, torch.device("cpu"))
    expected_key = store.key[:, :, 0, 64:96].reshape_as(candidates.key)
    expected_value = store.value[:, :, 0, 64:96].reshape_as(candidates.value)
    torch.testing.assert_close(candidates.key, expected_key)
    torch.testing.assert_close(candidates.value, expected_value)
    assert candidates.valid.all()


def test_page_fetch_deduplicates_shared_pages_without_copying_unselected_rows():
    support = 128
    key = torch.arange(support * 4, dtype=torch.float32).reshape(1, 1, support, 4)
    index = _index(key, support)
    store = DualPathPageStore.pack(key, -key, index)
    selected = torch.tensor([[[[0, 1, 3]]]], dtype=torch.int32)
    candidates = store.candidate_tensors(selected, torch.device("cpu"))
    for slot, leaf_id in enumerate(selected[0, 0, 0].tolist()):
        start = leaf_id * 32
        torch.testing.assert_close(
            candidates.key[0, 0, 0, slot], store.key[0, 0, 0, start : start + 32]
        )
        torch.testing.assert_close(
            candidates.value[0, 0, 0, slot], store.value[0, 0, 0, start : start + 32]
        )


def test_vectorized_tree_counts_partial_pages_and_padding():
    support = 4097
    key = torch.randn(1, 1, support, 8)
    index = _index(key, support)
    assert index.order.dtype == torch.int32
    assert index.num_pages == 33
    assert index.page_capacity == 64
    assert index.tree_rep_counts[0].sum().item() == support
    assert index.tree_rep_counts[0].tolist() == [4096, 1]
    assert index.tree_rep_counts[64 + 31].tolist() == [1, 0]
    weighted_root = (
        index.tree_representatives[0, 0, 0]
        * index.tree_rep_counts[0, :, None]
    ).sum(0) / support
    expected_root = (
        index.leaf_representatives[0, 0]
        * index.leaf_rep_counts[:, :, None]
    ).sum((0, 1)) / support
    torch.testing.assert_close(weighted_root, expected_root)
    assert (index.tree_rep_counts[64 + 33 :] == 0).all()


def test_layerwise_executor_matches_regular_encoder():
    torch.manual_seed(20261125)
    encoder = Encoder(
        2,
        32,
        2,
        64,
        attention_type="dual_path",
        sparse_every=1,
        block_size=32,
        num_selected_blocks=32,
        query_chunk_size=7,
        router="hierarchical",
        router_beam=64,
        partition="content_tree",
        dual_controller="none",
        qk_norm=True,
        zero_init=False,
    ).eval()
    for block in encoder.blocks:
        block.attn.use_triton = False
    support = torch.randn(1, 65, 32)
    query = torch.randn(1, 11, 32)
    with torch.inference_mode():
        expected = encoder(
            torch.cat((support, query), 1), train_size=support.shape[1]
        )[:, support.shape[1] :]
        actual = layerwise_dual_path_forward(
            encoder,
            support,
            query,
            support_tile_rows=17,
            query_tile_rows=3,
        )
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)
