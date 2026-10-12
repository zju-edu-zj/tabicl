import torch
import pytest
from torch.nn import functional as F

from tabicl._model.encoders import Encoder
from tabicl._model.icl_alternatives import (
    FlexBlockSparseAttention,
    InducedSetAttention,
    PerformerLinearAttention,
)
from tabicl._model.kv_cache import KVCache


def _run_grad(module, query, support):
    module.train()
    query = query.clone().requires_grad_()
    out = module(query, support, support)
    assert out.shape == query.shape
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_induced_set_attention_forward_backward_and_cache():
    torch.manual_seed(0)
    attention = InducedSetAttention(32, 4, num_slots=5).eval()
    query = torch.randn(1, 11, 32)
    support = torch.randn(1, 7, 32)
    full = attention(query, support, support)
    cached = attention(query[:, 7:], support, support, need_kv=True)
    entry = type("Entry", (), {"key": cached[1], "value": cached[2]})()
    test_only = attention(query[:, 7:], cached_kv=entry)
    assert torch.allclose(full[:, 7:], test_only, atol=1e-5, rtol=1e-5)
    _run_grad(attention, query, support)


def test_flex_block_sparse_forward_backward_and_support_prefix():
    torch.manual_seed(1)
    attention = FlexBlockSparseAttention(
        32, 4, block_size=128, num_kv_blocks=2
    )
    query = torch.randn(1, 19, 32)
    support = torch.randn(1, 11, 32)
    _run_grad(attention, query, support)


def test_linear_induced_encoder_schedule_and_cache():
    torch.manual_seed(2)
    encoder = Encoder(
        num_blocks=5,
        d_model=32,
        nhead=4,
        dim_feedforward=64,
        zero_init=False,
        attention_type="linear_induced",
        sparse_every=2,
    ).eval()
    assert encoder.sparse_layers == [False, True, False, True, False]
    assert [type(block.attn).__name__ for block in encoder.blocks] == [
        "PerformerLinearAttention", "InducedSetAttention", "PerformerLinearAttention",
        "InducedSetAttention", "PerformerLinearAttention",
    ]
    source = torch.randn(1, 13, 32)
    train_size = 8
    full = encoder(source, train_size=train_size)[:, train_size:]
    cache = KVCache()
    encoder.forward_with_cache(source[:, :train_size], cache, train_size=train_size,
                               use_cache=False, store_cache=True)
    cached = encoder.forward_with_cache(source[:, train_size:], cache,
                                        use_cache=True, store_cache=False)
    assert torch.allclose(full, cached, atol=1e-5, rtol=1e-5)


def test_induced_flex_encoder_schedule():
    encoder = Encoder(
        num_blocks=4, d_model=32, nhead=4, dim_feedforward=64,
        zero_init=False, attention_type="induced_flex", sparse_every=2,
    )
    assert [type(block.attn).__name__ for block in encoder.blocks] == [
        "InducedSetAttention", "FlexBlockSparseAttention",
        "InducedSetAttention", "FlexBlockSparseAttention",
    ]


def test_linear_flex_encoder_schedule():
    encoder = Encoder(
        num_blocks=4, d_model=32, nhead=4, dim_feedforward=64,
        zero_init=False, attention_type="linear_flex", sparse_every=2,
    )
    assert [type(block.attn).__name__ for block in encoder.blocks] == [
        "PerformerLinearAttention", "FlexBlockSparseAttention",
        "PerformerLinearAttention", "FlexBlockSparseAttention",
    ]


def test_elu_linear_flex_encoder_schedule():
    encoder = Encoder(
        num_blocks=4, d_model=32, nhead=4, dim_feedforward=64,
        zero_init=False, attention_type="elu_linear_flex", sparse_every=2,
    )
    assert [type(block.attn).__name__ for block in encoder.blocks] == [
        "GlobalLinearAttention", "FlexBlockSparseAttention",
        "GlobalLinearAttention", "FlexBlockSparseAttention",
    ]


def _dense_flex_reference(module, query, support):
    q, k, v = module._project(query, support, support, None)
    q, k = module._prepare_qk(q, k)
    order, starts, budget = module._route(q, k)
    k = k.gather(2, order[:, None, :, None].expand_as(k))
    v = v.gather(2, order[:, None, :, None].expand_as(v))
    size = module.block_size
    visible = ((starts + budget) * size).clamp_max(k.shape[-2]) - starts * size
    q = module._scale_queries(q, visible)
    pos = torch.arange(k.shape[-2], device=q.device)[None, None, :]
    mask = (pos >= starts[..., None] * size) & (pos < (starts[..., None] + budget) * size)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None])
    return module.out_proj(out.transpose(1, 2).reshape_as(query))


@pytest.mark.parametrize("mode", ["none", "ssmax", "bounded"])
def test_flex_matches_dense_masked_forward_and_gradients(mode):
    torch.manual_seed(11)
    module = FlexBlockSparseAttention(32, 2, num_kv_blocks=2, qk_norm=True,
                                     scale_mode=mode, ssmax="qassmax-mlp-elementwise")
    query = torch.randn(2, 137, 32, requires_grad=True)
    support = torch.randn(2, 533, 32, requires_grad=True)
    output = module(query, support, support)
    reference = _dense_flex_reference(module, query, support)
    assert torch.allclose(output, reference, atol=2e-5, rtol=2e-4)
    tensors = (query, support, *tuple(module.parameters()))
    grads = torch.autograd.grad(output.square().mean(), tensors, retain_graph=True)
    reference_grads = torch.autograd.grad(reference.square().mean(), tensors)
    for actual, expected in zip(grads, reference_grads):
        assert torch.isfinite(actual).all()
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-3)


def test_flex_query_batch_independence_support_permutation_and_cache():
    torch.manual_seed(12)
    module = FlexBlockSparseAttention(32, 2, num_kv_blocks=2, qk_norm=True,
                                     scale_mode="bounded").eval()
    query, support = torch.randn(2, 151, 32), torch.randn(2, 639, 32)
    full, ck, cv = module(query, support, support, need_kv=True)
    entry = type("Entry", (), {"key": ck, "value": cv})()
    split = torch.cat([module(query[:, :39], cached_kv=entry),
                       module(query[:, 39:], cached_kv=entry)], dim=1)
    assert torch.allclose(full, split, atol=2e-5, rtol=2e-4)
    permutation = torch.randperm(support.shape[1])
    permuted = module(query, support[:, permutation], support[:, permutation])
    assert torch.allclose(full, permuted, atol=2e-5, rtol=2e-4)


def test_flex_recall_diagnostics_full_and_sparse_budget():
    torch.manual_seed(13)
    module = FlexBlockSparseAttention(32, 2, num_kv_blocks=2, scale_mode="bounded")
    q, k = torch.randn(2, 2, 37, 16), torch.randn(2, 2, 511, 16)
    stats = module.routing_recall(q, k, topk=17)
    assert 0 <= stats["recall_at_k"] <= 1
    assert 0 < stats["candidate_fraction"] < 1
    assert 0 <= stats["retained_mass"] <= 1
    assert stats["nonfinite_fraction"] == 0
    module.num_kv_blocks = 4
    stats = module.routing_recall(q, k)
    assert stats["recall_at_k"] == 1
    assert stats["candidate_fraction"] == 1
    assert abs(stats["retained_mass"] - 1) < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Flex kernel validation")
def test_cuda_flex_bfloat16_matches_dense_reference_and_gradients():
    torch.manual_seed(14)
    module = FlexBlockSparseAttention(64, 4, num_kv_blocks=2, qk_norm=True,
                                     scale_mode="bounded").cuda()
    query = torch.randn(2, 137, 64, device="cuda", requires_grad=True)
    support = torch.randn(2, 533, 64, device="cuda", requires_grad=True)
    module.diagnostics_enabled = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = module(query, support, support)
        reference = _dense_flex_reference(module, query, support)
    assert torch.allclose(output, reference, atol=8e-3, rtol=3e-2)
    grads = torch.autograd.grad(output.float().square().mean(), (query, support), retain_graph=True)
    reference_grads = torch.autograd.grad(reference.float().square().mean(), (query, support))
    for actual, expected in zip(grads, reference_grads):
        assert torch.isfinite(actual).all()
        relative = (actual - expected).norm() / expected.norm().clamp_min(1e-8)
        assert relative < 0.04
    assert module.last_diagnostics["nonfinite_fraction"] == 0
    assert not module.diagnostics_enabled


def test_old_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unknown attention_type"):
        Encoder(num_blocks=3, d_model=32, nhead=2, dim_feedforward=64,
                attention_type="linear_sparse")


def test_elu_flex_encoder_prefix_cache_and_checkpoint_recompute():
    torch.manual_seed(15)
    encoder = Encoder(num_blocks=3, d_model=32, nhead=2, dim_feedforward=64,
                      zero_init=False, attention_type="elu_linear_flex", sparse_every=3,
                      flex_num_kv_blocks=2, flex_qk_norm=True, flex_scale_mode="bounded").eval()
    source = torch.randn(1, 557, 32)
    train_size = 521
    full = encoder(source, train_size=train_size)[:, train_size:]
    cache = KVCache()
    encoder.forward_with_cache(source[:, :train_size], cache, train_size=train_size,
                               use_cache=False, store_cache=True)
    cached = encoder.forward_with_cache(source[:, train_size:], cache,
                                        use_cache=True, store_cache=False)
    assert torch.allclose(full, cached, atol=2e-5, rtol=2e-4)
    encoder.train()
    encoder.recompute = True
    source.requires_grad_()
    for block in encoder.blocks:
        block.attn.diagnostics_enabled = True
    loss = encoder(source, train_size=train_size)[:, train_size:].square().mean()
    loss.backward()
    assert torch.isfinite(source.grad).all()
    for block in encoder.blocks:
        assert not block.attn.diagnostics_enabled
        assert block.attn.last_diagnostics
        for p in block.attn.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
