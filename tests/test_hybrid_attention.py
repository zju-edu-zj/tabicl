import torch

from tabicl._model.layers import HybridLinearTopKAttention
from tabicl._model.encoders import Encoder
from tabicl._model.kv_cache import KVCache


def _attention():
    return HybridLinearTopKAttention(
        embed_dim=32,
        num_heads=4,
        ssmax="qassmax-mlp-elementwise",
        exact_topk=8,
        sparse_chunk_size=3,
    )


def test_hybrid_attention_shape_and_gradients():
    torch.manual_seed(0)
    attention = _attention().train()
    query = torch.randn(2, 11, 32, requires_grad=True)
    support = torch.randn(2, 7, 32)
    output = attention(query, support, support)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert attention.in_proj_weight.grad is not None
    assert torch.isfinite(attention.in_proj_weight.grad).all()


def test_hybrid_attention_is_support_query_separated():
    torch.manual_seed(1)
    attention = _attention().eval()
    query = torch.randn(1, 4, 32)
    support = torch.randn(1, 9, 32)
    first = attention(query, support, support)
    changed_query = attention(query + 100.0, support, support)
    changed_support = attention(query, support + 100.0, support + 100.0)
    assert not torch.allclose(first, changed_query)
    assert not torch.allclose(first, changed_support)


def test_hybrid_attention_support_permutation_invariance():
    torch.manual_seed(2)
    attention = _attention().eval()
    query = torch.randn(1, 5, 32)
    support = torch.randn(1, 13, 32)
    permutation = torch.randperm(support.shape[1])
    first = attention(query, support, support)
    permuted = attention(query, support[:, permutation], support[:, permutation])
    assert torch.allclose(first, permuted, atol=1e-5, rtol=1e-5)


def test_hybrid_attention_cache_matches_full_prefix_path():
    torch.manual_seed(3)
    encoder = Encoder(
        num_blocks=2,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        ssmax="qassmax-mlp-elementwise",
        zero_init=False,
        attention_type="hybrid_linear_topk",
        exact_topk=8,
        sparse_chunk_size=3,
    ).eval()
    source = torch.randn(1, 11, 32)
    train_size = 7
    full = encoder(source, train_size=train_size)[:, train_size:]
    cache = KVCache()
    encoder.forward_with_cache(
        source[:, :train_size], cache, train_size=train_size, use_cache=False, store_cache=True
    )
    cached = encoder.forward_with_cache(source[:, train_size:], cache, use_cache=True, store_cache=False)
    assert torch.allclose(full, cached, atol=1e-5, rtol=1e-5)
