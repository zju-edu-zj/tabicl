import copy

import pytest
import torch
from torch.nn import functional as F

from tabicl._model.icl_attention import (
    AdaptiveResolutionAttention,
    DualPathRetrievalAttention,
)
from tabicl._model.kv_cache import KVCacheEntry
from tabicl._model.layers import MultiheadAttentionBlock


def _manual_frontier(module, q, k, v):
    order, selected = module.route(q, k)
    leaves = (k.shape[-2] + module.block_size - 1) // module.block_size
    lengths = (
        k.shape[-2] - torch.arange(leaves, device=k.device) * module.block_size
    ).clamp(1, module.block_size)
    summary_k, summary_v = module._leaf_summaries(k, v, order, lengths)
    return module._frontier_reference(
        q, k, v, order, selected, summary_k, summary_v, lengths
    )


@pytest.mark.parametrize("attention_class", [DualPathRetrievalAttention, AdaptiveResolutionAttention])
def test_new_attention_requires_content_tree(attention_class):
    with pytest.raises(ValueError, match="partition=content_tree"):
        attention_class(32, 2, block_size=16, num_selected_blocks=1)


def test_dual_path_v3_requires_qk_norm():
    with pytest.raises(ValueError, match="qk_norm=True"):
        DualPathRetrievalAttention(
            32, 2, block_size=16, num_selected_blocks=1,
            partition="content_tree", qk_norm=False,
        )


def test_dual_path_cache_gates_and_all_branch_gradients():
    torch.manual_seed(701)
    module = DualPathRetrievalAttention(
        32, 2, block_size=16, num_selected_blocks=2,
        partition="content_tree", qk_norm=True,
    ).eval()
    module.use_triton = False
    query = torch.randn(2, 11, 32, requires_grad=True)
    support = torch.randn(2, 83, 32, requires_grad=True)
    output, packed_k, packed_v = module(query, support, support, need_kv=True)
    assert packed_k.shape == (2, 4, 83, 16)
    assert packed_v.shape == packed_k.shape
    cached = module(query, cached_kv=KVCacheEntry(packed_k, packed_v))
    torch.testing.assert_close(cached, output, atol=3e-6, rtol=3e-5)

    loss = output.float().square().mean()
    loss.backward()
    required = (
        module.in_proj_weight,
        module.global_in_proj_weight,
        module.out_proj.weight,
        module.global_out_proj.weight,
        module.gate_controller.global_query_weight,
        module.gate_controller.sparse_query_weight,
        module.gate_controller.global_route_weight,
        module.gate_controller.sparse_route_weight,
        module.gate_controller.output_weight,
    )
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert support.grad is not None and torch.isfinite(support.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in required)

    q = torch.randn(1, 2, 3, 16)
    quality = torch.tensor([[[[1.0, 0.25, 0.7, 8.0, 4.0, 0.5]]]]).expand(1, 2, 3, 6)
    features = module._gate_route_features(quality)
    assert features.shape == quality.shape
    assert ((features >= 0) & (features <= 1)).all()


def test_dual_path_v3_starts_at_exact_v1_half_gates():
    torch.manual_seed(708)
    module = DualPathRetrievalAttention(
        32, 2, block_size=16, num_selected_blocks=1,
        partition="content_tree", qk_norm=True,
    )
    global_q = torch.randn(2, 2, 7, 16)
    sparse_q = torch.randn(2, 2, 7, 16)
    quality = torch.rand(2, 2, 7, 6)
    global_gate, sparse_gate, diagnostics = module.gate_controller(
        global_q, sparse_q, quality, support_size=97
    )
    torch.testing.assert_close(global_gate, torch.full_like(global_gate, 0.5))
    torch.testing.assert_close(sparse_gate, torch.full_like(sparse_gate, 0.5))
    assert diagnostics["fusion_common_delta_rms"] == 0
    assert diagnostics["fusion_differential_delta_rms"] == 0


def test_dual_path_v3_packed_projection_matches_independent_projection():
    torch.manual_seed(709)
    module = DualPathRetrievalAttention(
        32, 2, block_size=16, num_selected_blocks=1,
        partition="content_tree", qk_norm=True,
    )
    query, support = torch.randn(2, 7, 32), torch.randn(2, 41, 32)
    actual = module._project_dual(query, support, support, None)
    sparse = module._project(query, support, support, None)
    global_branch = module._project_global(query, support, support, None)
    for got, expected in zip(actual, (*sparse, *global_branch)):
        torch.testing.assert_close(got, expected)


def test_dual_path_v3_evidence_is_bounded_at_ten_million_rows():
    module = DualPathRetrievalAttention(
        32, 2, block_size=128, num_selected_blocks=4,
        partition="content_tree", qk_norm=True,
    )
    quality = torch.tensor([[[[2.0, 512 / 10_000_000, 0.7, 8.0, 4.0, 0.5]]]])
    evidence = module.gate_controller._evidence_features(quality, 10_000_000)
    assert evidence.shape[-1] == module.gate_controller._EVIDENCE_FEATURES
    assert torch.isfinite(evidence).all()
    assert ((evidence >= -1) & (evidence <= 1)).all()


def test_dual_path_branch_policy_is_native_gate_knockout():
    torch.manual_seed(710)
    module = DualPathRetrievalAttention(
        32, 2, block_size=16, num_selected_blocks=1,
        partition="content_tree", qk_norm=True,
    )
    module.use_triton = False
    query, support = torch.randn(1, 5, 32), torch.randn(1, 41, 32)
    outputs = {}
    for policy in ("native", "global_only", "sparse_only"):
        module._branch_policy = policy
        outputs[policy] = module(query, support, support)
    module._branch_policy = "native"
    bias = module.out_proj.bias
    torch.testing.assert_close(
        outputs["native"],
        outputs["global_only"] + outputs["sparse_only"] - bias,
        atol=3e-6,
        rtol=3e-5,
    )


@pytest.mark.parametrize(
    "attention_class", [DualPathRetrievalAttention, AdaptiveResolutionAttention]
)
def test_inference_tensor_cache_reuses_prepared_support(attention_class):
    torch.manual_seed(707)
    module = attention_class(
        32, 2, block_size=16, num_selected_blocks=1,
        partition="content_tree", qk_norm=True,
    ).eval()
    module.use_triton = False
    support, query = torch.randn(1, 65, 32), torch.randn(1, 5, 32)
    with torch.inference_mode():
        _, k, v = module(query, support, support, need_kv=True)
        cache = KVCacheEntry(k, v)
        first = module(query, cached_kv=cache)
        assert "state" in cache._derived
        prepared_id = id(cache._derived["state"])
        second = module(query, cached_kv=cache)
        assert id(cache._derived["state"]) == prepared_id
    torch.testing.assert_close(first, second)


def test_adaptive_frontier_matches_explicit_union_softmax_and_gradients():
    torch.manual_seed(702)
    module = AdaptiveResolutionAttention(
        32, 2, block_size=16, num_selected_blocks=1,
        partition="content_tree", qk_norm=True, summary_slots=1,
    )
    module.use_triton = False
    query = torch.randn(2, 9, 32, requires_grad=True)
    support = torch.randn(2, 67, 32, requires_grad=True)
    actual = module(query, support, support)
    q, k, v = module._project(query, support, support, None)
    attention_q = module.qk_norm.scale_query(q, support.shape[1])
    heads = _manual_frontier(module, attention_q, k, v)
    expected = module.out_proj(heads.transpose(1, 2).reshape_as(query))
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)

    upstream = torch.randn_like(actual) / actual.numel() ** 0.5
    variables = (query, support, *module.parameters())
    actual_grad = torch.autograd.grad(actual, variables, upstream, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, variables, upstream)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, atol=5e-6, rtol=5e-4)


def test_adaptive_frontier_rejects_nonconservative_multiple_summaries():
    with pytest.raises(ValueError, match="summary_slots=1"):
        AdaptiveResolutionAttention(
            32, 2, block_size=16, num_selected_blocks=1,
            partition="content_tree", summary_slots=2,
        )


def test_adaptive_frontier_is_exact_when_rows_within_each_leaf_are_identical():
    torch.manual_seed(705)
    module = AdaptiveResolutionAttention(
        32, 2, block_size=16, num_selected_blocks=1,
        partition="content_tree", summary_slots=1,
    )
    module.use_triton = False
    q = torch.randn(1, 2, 7, 16)
    leaf_k = torch.randn(1, 2, 4, 16)
    leaf_v = torch.randn(1, 2, 4, 16)
    k = leaf_k.repeat_interleave(16, dim=2)
    v = leaf_v.repeat_interleave(16, dim=2)
    order = torch.arange(64, dtype=torch.int32).expand(1, 2, -1).contiguous()
    selected = torch.randint(0, 4, (1, 2, 7, 1), dtype=torch.int32)
    lengths = torch.full((4,), 16)
    summary_k, summary_v = module._leaf_summaries(k, v, order, lengths)
    frontier = module._frontier_reference(
        q, k, v, order, selected, summary_k, summary_v, lengths
    )
    dense = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(frontier, dense, atol=2e-6, rtol=2e-6)


def test_adaptive_frontier_cache_and_full_budget_fast_path():
    torch.manual_seed(703)
    module = AdaptiveResolutionAttention(
        32, 2, block_size=16, num_selected_blocks=2,
        partition="content_tree", summary_slots=1, qk_norm=True,
    ).eval()
    module.use_triton = False
    query = torch.randn(1, 7, 32)
    support = torch.randn(1, 65, 32)
    output, k, v = module(query, support, support, need_kv=True)
    with torch.no_grad():
        cached = module(query, cached_kv=KVCacheEntry(k, v))
    torch.testing.assert_close(cached, output, atol=3e-6, rtol=3e-5)

    small = torch.randn(1, 31, 32)
    module.route = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("full-budget path must not route")
    )
    actual = module(query, small, small)
    q, k, v = module._project(query, small, small, None)
    q = module.qk_norm.scale_query(q, small.shape[1])
    expected = module.out_proj(
        F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape_as(query)
    )
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("attention", ["dual_path", "adaptive_frontier"])
def test_new_attention_zero_init_has_finite_two_step_bootstrap(attention):
    torch.manual_seed(706)
    block = MultiheadAttentionBlock(
        32, 2, 64, zero_init=True, attention_type=attention,
        block_size=16, num_selected_blocks=1, partition="content_tree",
        qk_norm=True,
    )
    block.attn.use_triton = False
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)
    query, support = torch.randn(2, 9, 32), torch.randn(2, 49, 32)
    initial = block.attn(query, support, support)
    torch.testing.assert_close(initial, torch.zeros_like(initial))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = block.attn(query, support, support)
        output.add(1).square().mean().backward()
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in block.parameters() if parameter.grad is not None
        )
        optimizer.step()
    assert block.attn(query, support, support).abs().max() > 0


@pytest.mark.parametrize("selected_blocks", [1, 3, 4, 5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_frontier_matches_portable_forward_backward(dtype, selected_blocks):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(704)
    fast = AdaptiveResolutionAttention(
        64, 2, block_size=32, num_selected_blocks=selected_blocks,
        partition="content_tree", qk_norm=True, summary_slots=1,
    ).cuda()
    portable = copy.deepcopy(fast)
    portable.use_triton = False
    query = torch.randn(1, 73, 64, device="cuda", requires_grad=True)
    support = torch.randn(
        1, 32 * (selected_blocks + 3) + 1, 64, device="cuda", requires_grad=True
    )
    query_ref = query.detach().clone().requires_grad_()
    support_ref = support.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=dtype):
        actual = fast(query, support, support)
        expected = portable(query_ref, support_ref, support_ref)
    tolerance = 8e-3 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    upstream = torch.randn_like(actual) / actual.numel() ** 0.5
    actual_grad = torch.autograd.grad(actual, (query, support, *fast.parameters()), upstream)
    expected_grad = torch.autograd.grad(
        expected, (query_ref, support_ref, *portable.parameters()), upstream
    )
    for got, want in zip(actual_grad, expected_grad):
        error = (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-6)
        assert error < (0.06 if dtype == torch.bfloat16 else 0.025), error
