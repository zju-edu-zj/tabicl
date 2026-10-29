"""Projection views must preserve gradients as well as attention results."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from tabicl._model.attention import _project_qkv
from tabicl._model.layers import InducedSelfAttentionBlock
from tabicl._model.layers import MultiheadAttentionBlock
from tabicl._model import attention
from tabicl._model.kv_cache import KVCache


@pytest.mark.parametrize("sharing", ["qkv", "kv", "none"])
@pytest.mark.parametrize("bias", [False, True])
def test_projection_views_match_packed_outputs_and_gradients(sharing, bias):
    torch.manual_seed(239)
    q = torch.randn(2, 3, 7, 16, dtype=torch.float64, requires_grad=True)
    k = q if sharing == "qkv" else torch.randn(2, 3, 11, 16, dtype=torch.float64, requires_grad=True)
    v = k if sharing != "none" else torch.randn_like(k, requires_grad=True)
    weight = torch.randn(48, 16, dtype=torch.float64, requires_grad=True)
    offset = torch.randn(48, dtype=torch.float64, requires_grad=True) if bias else None
    inputs = list(dict.fromkeys([q, k, v, weight] + ([offset] if bias else [])))
    actual = _project_qkv(q, k, v, weight, offset)
    expected = F._in_projection_packed(q, k, v, weight, offset)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    upstream = [torch.randn_like(a) for a in actual]
    a_grad = torch.autograd.grad(actual, inputs, upstream)
    b_grad = torch.autograd.grad(expected, inputs, upstream)
    for a, b in zip(a_grad, b_grad):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("norm_first", [False, True])
def test_shared_support_matches_separate_slice_reference(norm_first):
    torch.manual_seed(581)
    block = InducedSelfAttentionBlock(16, 2, 32, 4, norm_first=norm_first, zero_init=False).double()
    reference = deepcopy(block)
    x = torch.randn(2, 3, 17, 16, dtype=torch.float64, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    actual = block.induced_attention(x, 11)
    inds = reference.ind_vectors.expand(2, 3, 4, 16)
    hidden = reference.multihead_attn1(inds, other[..., :11, :], other[..., :11, :])
    expected = reference.multihead_attn2(other, hidden, hidden)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, other.grad, rtol=1e-10, atol=1e-12)
    for a, b in zip(block.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("self_attention", [False, True])
def test_cuda_sdpa_accepts_projection_views_and_matches_packed_backward(monkeypatch, dtype, self_attention):
    torch.manual_seed(581)
    block = MultiheadAttentionBlock(128, 8, 256, zero_init=False).cuda()
    reference = deepcopy(block)
    x = torch.randn(2, 3, 35, 128, device="cuda", requires_grad=True)
    other = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        if self_attention:
            actual = block(x)
        else:
            support = x[..., :23, :]
            actual = block(x[..., :4, :], support, support)
        with monkeypatch.context() as context:
            context.setattr(attention, "_project_qkv", F._in_projection_packed)
            if self_attention:
                expected = reference(other)
            else:
                support = other[..., :23, :]
                expected = reference(other[..., :4, :], support, support)
    torch.testing.assert_close(actual, expected, rtol=.02 if dtype == torch.bfloat16 else 2e-5,
                               atol=.01 if dtype == torch.bfloat16 else 2e-6)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    left = torch.cat([x.grad.flatten()] + [p.grad.flatten() for p in block.parameters()])
    right = torch.cat([other.grad.flatten()] + [p.grad.flatten() for p in reference.parameters()])
    assert (left - right).norm() / right.norm() < (.02 if dtype == torch.bfloat16 else 2e-5)


@pytest.mark.parametrize("norm_first", [False, True])
def test_projection_views_preserve_isab_cached_query_chunks(monkeypatch, norm_first):
    torch.manual_seed(907)
    block = InducedSelfAttentionBlock(16, 2, 32, 4, norm_first=norm_first, zero_init=False).double().eval()
    x = torch.randn(2, 3, 23, 16, dtype=torch.float64)
    support = 11
    with torch.no_grad():
        # Use the original projection layout and distinct support slices as
        # reference, so both changes are exercised by the cached path.
        with monkeypatch.context() as context:
            context.setattr(attention, "_project_qkv", F._in_projection_packed)
            inds = block.ind_vectors.expand(2, 3, 4, 16)
            hidden = block.multihead_attn1(inds, x[..., :support, :], x[..., :support, :])
            expected = block.multihead_attn2(x, hidden, hidden)
        cache = KVCache()
        stored = block.induced_attention_with_cache(x[..., :support, :], cache, 0,
                                                    train_size=support, store_cache=True)
        torch.testing.assert_close(stored, expected[..., :support, :], rtol=1e-12, atol=1e-12)
        chunks = [block.induced_attention_with_cache(part, cache, 0, use_cache=True, store_cache=False)
                  for part in x[..., support:, :].split(5, dim=-2)]
        torch.testing.assert_close(torch.cat(chunks, dim=-2), expected[..., support:, :],
                                   rtol=1e-12, atol=1e-12)
