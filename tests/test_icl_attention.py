"""Mathematical, set-semantics and CUDA/reference checks for the five recipes."""

import copy

import pytest
import torch
from torch.nn import functional as F

from tabicl._model.encoders import Encoder
from tabicl._model.icl_attention import ICL_ATTENTION_TYPES, SetRetrievalAttention
from tabicl._model.kv_cache import KVCache


@pytest.mark.parametrize("router", ["centroid", "hierarchical"])
@pytest.mark.parametrize("partition", ["projection", "content_tree"])
@pytest.mark.parametrize("mode", ["sparse", "fused_sparse", "fused_full"])
@pytest.mark.parametrize("device,dtype", [("cpu", torch.float32), ("cuda", torch.float32),
                                         ("cuda", torch.bfloat16), ("cuda", torch.float16)])
def test_selected_dense_mask_output_and_all_gradients(router, partition, mode, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(454)
    module = SetRetrievalAttention(
        64, 2, block_size=32, num_selected_blocks=2, router=router, router_beam=4, mode=mode, partition=partition
    ).to(device)
    query = torch.randn(2, 37, 64, device=device, requires_grad=True)
    support = torch.randn(2, 193, 64, device=device, requires_grad=True)
    with torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32):
        actual = module(query, support, support)
        q, k, v = module._project(query, support, support, None)
        order, chosen = module.route(q, k)
        ks = k.gather(2, order[..., None].expand_as(k))
        vs = v.gather(2, order[..., None].expand_as(v))
        ids = torch.arange(k.shape[2], device=device) // module.block_size
        mask = None if mode == "fused_full" else (chosen[..., None] == ids).any(-2)
        sparse = F.scaled_dot_product_attention(q, ks, vs, attn_mask=mask)
        if mode != "sparse":
            # Independent explicit NxS kernel reference, not the state helper.
            pq, pk = (F.elu(q.float()) + 1).clamp_min(1e-6), (F.elu(k.float()) + 1).clamp_min(1e-6)
            with torch.autocast(device, enabled=False):
                weights = pq @ pk.transpose(-1, -2)
                global_heads = ((weights / weights.sum(-1, keepdim=True)) @ v.float()).to(q.dtype)
                gate = torch.sigmoid(
                    (q.float() * module.fusion_gate_weight[None, :, None]).sum(-1)
                    + module.fusion_gate_bias[None, :, None]
                ).to(q.dtype)[..., None]
            sparse = gate * sparse + (1 - gate) * global_heads
        reference = module.out_proj(sparse.transpose(1, 2).reshape_as(query))
    tolerance = 4e-3 if dtype == torch.bfloat16 else 1e-3 if dtype == torch.float16 else 4e-5
    torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
    variables = (query, support, *module.parameters())
    upstream = torch.randn_like(actual) / actual.numel() ** 0.5
    expected = torch.autograd.grad(reference, variables, upstream)
    gradients = torch.autograd.grad(actual, variables, upstream)
    for grad, target in zip(gradients, expected):
        torch.testing.assert_close(
            grad,
            target,
            atol=7e-4 if dtype == torch.bfloat16 else 2e-4 if dtype == torch.float16 else 3e-6,
            rtol=0.04 if dtype == torch.bfloat16 else 0.01 if dtype == torch.float16 else 3e-4,
        )


@pytest.mark.parametrize("router", ["centroid", "hierarchical"])
@pytest.mark.parametrize("partition", ["projection", "content_tree"])
@pytest.mark.parametrize("support", [1, 65, 193, 1025])
@pytest.mark.parametrize("beam", [4, 5, 16])
def test_cuda_routing_matches_portable_reference(router, partition, support, beam):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(88)
    module = SetRetrievalAttention(64, 2, block_size=32, num_selected_blocks=2, router=router,
                                   router_beam=beam, partition=partition).cuda()
    q = torch.randn(2, 2, 67, 32, device="cuda")
    k = torch.randn(2, 2, support, 32, device="cuda")
    order, selected = module.route(q, k)
    module.use_triton = False
    order_ref, expected = module.route(q, k)
    torch.testing.assert_close(order, order_ref)
    torch.testing.assert_close(selected, expected)
    assert (selected >= 0).all() and (selected < (support + 31) // 32).all()
    assert (selected.sort(-1).values.diff(dim=-1) > 0).all()


@pytest.mark.parametrize("router", ["centroid", "hierarchical"])
@pytest.mark.parametrize("partition", ["projection", "content_tree"])
def test_routing_permutations_and_query_batching(router, partition):
    torch.manual_seed(89)
    module = SetRetrievalAttention(32, 2, block_size=32, num_selected_blocks=2, router=router,
                                   router_beam=4, partition=partition)
    q, k = torch.randn(1, 2, 37, 16), torch.randn(1, 2, 193, 16)
    order, chosen = module.route(q, k)
    perm = torch.randperm(k.shape[2])
    order_perm, chosen_perm = module.route(q, k[:, :, perm])
    torch.testing.assert_close(order, perm[order_perm.long()].to(order.dtype))
    torch.testing.assert_close(chosen, chosen_perm)
    pieces = [module.route(part, k)[1] for part in q.split([13, 24], dim=2)]
    torch.testing.assert_close(chosen, torch.cat(pieces, 2))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nonfinite_content_scores_never_turn_padding_into_key_indices(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    module = SetRetrievalAttention(32, 2, block_size=32, num_selected_blocks=2,
                                   partition="content_tree").to(device)
    k = torch.full((1, 2, 65, 16), float("nan"), device=device)
    q = torch.randn(1, 2, 7, 16, device=device)
    order, _ = module.route(q, k)
    torch.testing.assert_close(
        order.sort(-1).values,
        torch.arange(65, device=device, dtype=torch.int32).expand_as(order),
    )


@pytest.mark.parametrize("mode", ICL_ATTENTION_TYPES)
@pytest.mark.parametrize("partition", ["projection", "content_tree"])
@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("router", ["centroid", "hierarchical"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_multilayer_visibility_cache_and_permutations(mode, partition, norm_first, router, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(90)
    model = (
        Encoder(
            3,
            32,
            2,
            64,
            attention_type=mode,
            sparse_every=2,
            block_size=32,
            num_selected_blocks=2,
            router_beam=4,
            router=router,
            partition=partition,
            norm_first=norm_first,
            zero_init=False,
        )
        .to(device)
        .eval()
    )
    x, s = torch.randn(1, 143, 32, device=device), 97
    full = model(x, train_size=s)
    changed = x.clone()
    changed[:, s:] *= 5
    torch.testing.assert_close(model(changed, train_size=s)[:, :s], full[:, :s])
    cache = KVCache()
    model.forward_with_cache(x[:, :s], cache, train_size=s, store_cache=True)
    outputs = [
        model.forward_with_cache(part, cache, use_cache=True, store_cache=False)
        for part in x[:, s:].split([13, 33], dim=1)
    ]
    torch.testing.assert_close(torch.cat(outputs, 1), full[:, s:], atol=3e-6, rtol=3e-5)
    perm = torch.randperm(s, device=device)
    reordered = torch.cat((x[:, perm], x[:, s:]), 1)
    torch.testing.assert_close(model(reordered, train_size=s)[:, s:], full[:, s:], atol=3e-6, rtol=3e-5)


def test_hierarchy_exhaustive_beam_is_exact_centroid_topk():
    torch.manual_seed(91)
    m = SetRetrievalAttention(32, 2, block_size=32, num_selected_blocks=2, router_beam=32)
    q, k = torch.randn(1, 2, 31, 16), torch.randn(1, 2, 641, 16)
    exact = m.route(q, k)[1]
    m.router = "hierarchical"
    torch.testing.assert_close(m.route(q, k)[1], exact)


@pytest.mark.parametrize("mode", ICL_ATTENTION_TYPES)
@pytest.mark.parametrize("partition", ["projection", "content_tree"])
def test_checkpoint_replay_matches_uncheckpointed(mode, partition):
    torch.manual_seed(92)
    m = Encoder(3, 32, 2, 64, attention_type=mode, block_size=32, num_selected_blocks=2,
                zero_init=False, partition=partition)
    other = copy.deepcopy(m)
    other.recompute = True
    x = torch.randn(1, 137, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = m(x, train_size=97), other(y, train_size=97)
    torch.testing.assert_close(a, b)
    a[:, 97:].square().mean().backward()
    b[:, 97:].square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad)
    for p, q in zip(m.parameters(), other.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [16, 32, 64, 128])
@pytest.mark.parametrize("hotspot", [True, False])
def test_grouped_sparse_backward_matches_masked_dense(dtype, head_dim, hotspot):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from tabicl._model._sparse_triton import routed_attention

    torch.manual_seed(93)
    q = torch.randn(2, 2, 273, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(2, 2, 193, head_dim, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    # Three distinct leaves, including the one-row tail and heavily reused leaves.
    ids = torch.rand(2, 2, 273, 7, device="cuda").topk(3, -1).indices
    if hotspot:
        ids = torch.stack(
            (torch.full_like(ids[..., 0], 6), torch.zeros_like(ids[..., 0]), torch.ones_like(ids[..., 0])), -1
        )
    mask = (ids[..., None] == torch.arange(193, device="cuda") // 32).any(-2)
    order = torch.rand(2, 2, k.shape[-2], device="cuda").argsort(-1).to(torch.int32)
    actual = routed_attention(q, k, v, order, ids.to(torch.int32), 32)
    gather = order.long()[..., None].expand_as(k)
    reference = F.scaled_dot_product_attention(q, k.gather(2, gather), v.gather(2, gather), attn_mask=mask)
    tolerance = 0.008 if dtype == torch.bfloat16 else 0.001
    torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
    upstream = torch.randn_like(actual) / 10
    gradients = torch.autograd.grad(actual, (q, k, v), upstream)
    expected = torch.autograd.grad(reference, (q, k, v), upstream)
    for grad, target in zip(gradients, expected):
        error = (grad.float() - target.float()).norm() / target.float().norm()
        assert error < tolerance, error


@pytest.mark.parametrize("router", ["centroid", "hierarchical"])
def test_long_support_checkpoint_preserves_sampled_diagnostics(router):
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(94)
    m = SetRetrievalAttention(32, 2, block_size=32, num_selected_blocks=2, router=router, router_beam=4)
    q = torch.randn(1, 17, 32, requires_grad=True)
    k = torch.randn(1, 4101, 32, requires_grad=True)
    m.diagnostics_enabled = True
    out = checkpoint(m, q, k, k, use_reentrant=False, context_fn=m.checkpoint_contexts)
    sampled = dict(m.last_diagnostics)
    out.square().mean().backward()
    assert sampled and m.last_diagnostics.keys() == sampled.keys()
    assert m.last_diagnostics["denominator_min"] > 0
    assert m.last_diagnostics["nonfinite_fraction"] == 0
    assert not m.diagnostics_enabled
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()


@pytest.mark.parametrize("mode", ["sparse", "fused_sparse", "fused_full"])
def test_full_budget_fast_path_skips_routing(mode):
    m = SetRetrievalAttention(32, 2, mode=mode, block_size=32, num_selected_blocks=3)

    def forbidden(*args):
        raise AssertionError("full-support fast path must not route")

    m.route = forbidden
    q, k = torch.randn(1, 7, 32), torch.randn(1, 65, 32)
    result = m(q, k, k)
    qp, kp, vp = m._project(q, k, k, None)
    sparse = F.scaled_dot_product_attention(qp, kp, vp)
    heads = sparse if mode == "sparse" else 0.5 * (sparse + m._global_attention(qp, kp, vp))
    torch.testing.assert_close(result, m.out_proj(heads.transpose(1, 2).reshape_as(q)))


def test_tail_centroid_and_weighted_tree():
    # The last, one-token leaf must not be divided by the padded block size.
    m = SetRetrievalAttention(32, 2, block_size=32, num_selected_blocks=1, router_beam=4)
    q = torch.ones(1, 2, 1, 16)
    k = torch.zeros(1, 2, 65, 16)
    k[:, :, :32] = 1
    k[:, :, 32:64] = 2
    k[:, :, 64:] = 3
    m.route_direction.fill_(1)
    assert (m.route(q, k)[1] == 2).all()
    m.router = "hierarchical"
    assert (m.route(q, k)[1] == 2).all()


@pytest.mark.parametrize("mode", ICL_ATTENTION_TYPES[1:])
def test_experiments_reject_unsupported_scaling_and_dropout(mode):
    with pytest.raises(ValueError, match="ssmax=False"):
        Encoder(3, 32, 2, 64, attention_type=mode, ssmax=True)
    with pytest.raises(ValueError, match="dropout=0"):
        Encoder(3, 32, 2, 64, attention_type=mode, dropout=0.1)
