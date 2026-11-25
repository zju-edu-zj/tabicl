import copy

import pytest
import torch
from torch.nn import functional as F

from tabicl._model.icl_attention import (
    AdaptiveResolutionAttention,
)
from tabicl._model.dual_path_v3 import DualPathRetrievalAttentionV3
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


@pytest.mark.parametrize("attention_class", [DualPathRetrievalAttentionV3, AdaptiveResolutionAttention])
def test_new_attention_requires_content_tree(attention_class):
    block_size = 32 if attention_class is DualPathRetrievalAttentionV3 else 16
    selected = 32 if attention_class is DualPathRetrievalAttentionV3 else 1
    with pytest.raises(ValueError, match="partition=content_tree"):
        attention_class(32, 2, block_size=block_size, num_selected_blocks=selected)


def test_dual_path_v3_requires_qk_norm():
    with pytest.raises(ValueError, match="qk_norm=True"):
        DualPathRetrievalAttentionV3(
            32, 2, block_size=32, num_selected_blocks=32,
            partition="content_tree", qk_norm=False,
        )


def test_dual_path_v3_all_branch_gradients():
    torch.manual_seed(701)
    module = DualPathRetrievalAttentionV3(
        32, 2, block_size=32, num_selected_blocks=32,
        partition="content_tree", qk_norm=True, dual_controller="semantic",
    )
    module.use_triton = False
    query = torch.randn(2, 11, 32, requires_grad=True)
    support = torch.randn(2, 83, 32, requires_grad=True)
    output = module(query, support, support)

    loss = output.float().square().mean()
    loss.backward()
    required = (
        module.in_proj_weight,
        module.global_in_proj_weight,
        module.out_proj.weight,
        module.global_out_proj.weight,
        module.gate_controller.query_weight,
        module.gate_controller.evidence_weight,
        module.gate_controller.output_weight,
    )
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert support.grad is not None and torch.isfinite(support.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in required)

def test_dual_path_v3_controller_starts_at_unit_modulation():
    torch.manual_seed(708)
    module = DualPathRetrievalAttentionV3(
        32, 2, block_size=32, num_selected_blocks=32,
        partition="content_tree", qk_norm=True, dual_controller="semantic",
    )
    query = torch.randn(2, 7, 32)
    evidence = torch.rand(2, 2, 7, 3)
    candidates = torch.full((2, 2, 7), 97.0)
    global_scale, sparse_scale, diagnostics = module.gate_controller(
        query, evidence, support_size=97, candidate_rows=candidates
    )
    torch.testing.assert_close(global_scale, torch.ones_like(global_scale))
    torch.testing.assert_close(sparse_scale, torch.ones_like(sparse_scale))
    assert diagnostics["fusion_controller_logit_rms"] == 0


def test_dual_path_v3_default_has_no_controller_parameters():
    module = DualPathRetrievalAttentionV3(
        32, 2, block_size=32, num_selected_blocks=32,
        partition="content_tree", qk_norm=True, dual_controller="none",
    )
    query = torch.randn(2, 7, 32)
    evidence = torch.rand(2, 2, 7, 3)
    candidates = torch.full((2, 2, 7), 97.0)
    global_scale, sparse_scale, _ = module.gate_controller(
        query, evidence, support_size=97, candidate_rows=candidates
    )
    assert not tuple(module.gate_controller.parameters())
    torch.testing.assert_close(global_scale, torch.ones_like(global_scale))
    torch.testing.assert_close(sparse_scale, torch.ones_like(sparse_scale))


def test_dual_path_v3_packed_projection_matches_independent_projection():
    torch.manual_seed(709)
    module = DualPathRetrievalAttentionV3(
        32, 2, block_size=32, num_selected_blocks=32,
        partition="content_tree", qk_norm=True,
    )
    query, support = torch.randn(2, 7, 32), torch.randn(2, 41, 32)
    actual = module._project_dual(query, support, None)
    sparse = module._project(query, support, support, None)
    for got, expected in zip(actual[:3], sparse):
        torch.testing.assert_close(got, expected)
    assert all(tensor.shape == sparse[index].shape for index, tensor in enumerate(actual[3:]))
    assert all(torch.isfinite(tensor).all() for tensor in actual)


def test_dual_path_branch_policy_is_native_gate_knockout():
    torch.manual_seed(710)
    module = DualPathRetrievalAttentionV3(
        32, 2, block_size=32, num_selected_blocks=32,
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


def test_dual_path_prepared_support_matches_direct_forward():
    torch.manual_seed(707)
    module = DualPathRetrievalAttentionV3(
        32, 2, block_size=32, num_selected_blocks=32,
        partition="content_tree", qk_norm=True,
    ).eval()
    module.use_triton = False
    support, query = torch.randn(1, 65, 32), torch.randn(1, 5, 32)
    with torch.inference_mode():
        direct = module(query, support, support)
        prepared = module.prepare_support(support)
        paged = module.apply_prepared(query, prepared)
    torch.testing.assert_close(direct, paged, atol=3e-6, rtol=3e-5)


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


def test_adaptive_frontier_zero_init_has_finite_two_step_bootstrap():
    torch.manual_seed(706)
    block = MultiheadAttentionBlock(
        32, 2, 64, zero_init=True, attention_type="adaptive_frontier",
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
