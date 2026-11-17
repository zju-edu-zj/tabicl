import copy

import pytest
import torch
from torch.nn import functional as F

from tabicl._model.encoders import Encoder
from tabicl._model.kv_cache import KVCache
from tabicl._model.token_sparse import TokenSparseAttention


def _module(device="cpu", regression=False):
    module = TokenSparseAttention(
        32,
        2,
        topk=8,
        index_dim=4,
        query_chunk_size=5,
        index_query_samples=3,
        index_regression=regression,
        index_num_classes=4,
        qk_norm=True,
    ).to(device)
    module.use_triton = device == "cuda"
    return module


def test_token_route_is_shared_per_query_and_raw_token_attention_matches_masked_dense():
    torch.manual_seed(600)
    module = _module()
    query = torch.randn(2, 13, 32, requires_grad=True)
    support = torch.randn(2, 19, 32, requires_grad=True)
    actual = module(query, support, support)
    q, k, v = module._project(query, support, support, None)
    q_index, k_index = module._index_embeddings(q, k)
    indices = module.route(q_index, k_index)
    assert indices.shape == (2, 13, 8)
    assert (indices[:, 1:] != indices[:, :-1]).any()
    mask = torch.zeros((2, 13, 19), dtype=torch.bool)
    mask.scatter_(2, indices.long(), True)
    attention_q = module.qk_norm.scale_query(q, 8)
    heads = F.scaled_dot_product_attention(attention_q, k, v, attn_mask=mask[:, None])
    reference = module.out_proj(heads.transpose(1, 2).reshape_as(query))
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)

    upstream = torch.randn_like(actual)
    expected = torch.autograd.grad(reference, (query, support), upstream)
    gradients = torch.autograd.grad(actual, (query, support), upstream)
    for gradient, target in zip(gradients, expected):
        torch.testing.assert_close(gradient, target, atol=3e-6, rtol=3e-5)


def test_index_loss_updates_only_indexer_and_is_label_permutation_invariant():
    torch.manual_seed(601)
    module = _module()
    query = torch.randn(2, 13, 32, requires_grad=True)
    support = torch.randn(2, 9, 32, requires_grad=True)
    targets = torch.tensor([
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        [2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2],
    ])
    positions = torch.tensor([9, 11, 12])
    output, index_loss = module(
        query,
        support,
        support,
        index_targets=targets,
        index_query_indices=positions,
        return_index_loss=True,
    )
    index_parameters = (
        module.index_q_proj.weight,
        module.index_k_proj.weight,
        module.index_logit_scale,
    )
    index_gradients = torch.autograd.grad(index_loss, index_parameters, retain_graph=True)
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in index_gradients)
    backbone = torch.autograd.grad(index_loss, (query, support), allow_unused=True, retain_graph=True)
    assert all(gradient is None or not gradient.any() for gradient in backbone)
    task_index_gradients = torch.autograd.grad(
        output.square().mean(), index_parameters, allow_unused=True
    )
    assert all(gradient is None for gradient in task_index_gradients)

    permutation = torch.tensor([2, 0, 1, 3])
    _, permuted_loss = module(
        query,
        support,
        support,
        index_targets=permutation[targets],
        index_query_indices=positions,
        return_index_loss=True,
    )
    torch.testing.assert_close(index_loss, permuted_loss)


def test_regression_index_loss_handles_constant_support_targets():
    torch.manual_seed(602)
    module = _module(regression=True)
    query = torch.randn(1, 13, 32, requires_grad=True)
    targets = torch.ones(1, 13)
    _, loss = module(
        query,
        query[:, :9],
        query[:, :9],
        index_targets=targets,
        index_query_indices=torch.tensor([9, 11, 12]),
        return_index_loss=True,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(module.index_q_proj.weight.grad).all()


@pytest.mark.parametrize("recompute", [False, True])
def test_every_layer_is_token_sparse_and_checkpoint_has_full_gradients(recompute):
    torch.manual_seed(603)
    model = Encoder(
        3,
        32,
        2,
        64,
        attention_type="token_sparse",
        sparse_every=3,
        token_topk=8,
        token_index_dim=4,
        token_index_query_samples=3,
        qk_norm=True,
        recompute=recompute,
        zero_init=False,
    )
    assert all(isinstance(block.attn, TokenSparseAttention) for block in model.blocks)
    x = torch.randn(2, 17, 32, requires_grad=True)
    targets = torch.randint(0, 4, (2, 17))
    output, loss = model(
        x,
        train_size=11,
        index_targets=targets,
        index_query_indices=torch.tensor([11, 14, 16]),
    )
    (output.square().mean() + 0.05 * loss).backward()
    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    assert missing == []


def test_checkpoint_and_kv_cache_match_uncheckpointed_full_forward():
    torch.manual_seed(604)
    model = Encoder(
        2, 32, 2, 64, attention_type="token_sparse", token_topk=8,
        token_index_dim=4, qk_norm=True, zero_init=False,
    ).eval()
    checkpointed = copy.deepcopy(model)
    checkpointed.recompute = True
    x = torch.randn(1, 19, 32)
    support = 11
    with torch.no_grad():
        full = model(x, train_size=support)
        replayed = checkpointed(x, train_size=support)
    torch.testing.assert_close(full, replayed)

    cache = KVCache()
    with torch.no_grad():
        model.forward_with_cache(x[:, :support], cache, train_size=support, store_cache=True)
        cached = model.forward_with_cache(
            x[:, support:], cache, use_cache=True, store_cache=False
        )
    torch.testing.assert_close(cached, full[:, support:], atol=3e-6, rtol=3e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_kernel_matches_reference_output_and_qkv_gradients(dtype):
    from tabicl._model._token_sparse_triton import token_sparse_attention

    torch.manual_seed(605)
    q = torch.randn(2, 3, 67, 64, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(2, 3, 53, 64, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    indices = torch.rand(2, 67, 53, device="cuda").topk(16, -1).indices.to(torch.int32)
    actual = token_sparse_attention(q, k, v, indices)
    mask = torch.zeros((2, 67, 53), device="cuda", dtype=torch.bool)
    mask.scatter_(2, indices.long(), True)
    reference = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
    tolerance = 6e-3 if dtype == torch.bfloat16 else 4e-5
    torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
    upstream = torch.randn_like(actual) / 10
    gradients = torch.autograd.grad(actual, (q, k, v), upstream)
    expected = torch.autograd.grad(reference, (q, k, v), upstream)
    for gradient, target in zip(gradients, expected):
        error = (gradient.float() - target.float()).norm() / target.float().norm().clamp_min(1e-8)
        assert error < (0.03 if dtype == torch.bfloat16 else 4e-4), error
