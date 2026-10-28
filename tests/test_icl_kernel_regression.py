"""Numerical regressions for grouped edges and split support reductions."""

import pytest
import torch
from torch.nn import functional as F

from tabicl._model._elu_kernel import elu_kernel_attention
from tabicl._model._sparse_triton import routed_attention


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dim", [16, 32, 64, 128])
@pytest.mark.parametrize("block,support,queries,chosen", [(32, 193, 259, 3), (128, 397, 273, 4),
                                                         (64, 1089, 257, 1),
                                                         (16, 97, 17, 2), (96, 289, 257, 2),
                                                         (256, 513, 17, 2)])
@pytest.mark.parametrize("hotspot", [False, True])
def test_grouped_output_and_all_gradients(dtype, dim, block, support, queries, chosen, hotspot):
    torch.manual_seed(810)
    # Transpose deliberately exercises noncontiguous head-major input views.
    q = torch.randn(1, queries, 2, dim, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
    k = torch.randn(1, support, 2, dim, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
    v = torch.randn_like(k).requires_grad_()
    leaves = (support + block - 1) // block
    order = torch.rand(1, 2, support, device="cuda").argsort(-1).int()
    ids = torch.rand(1, 2, queries, leaves, device="cuda").topk(chosen, -1).indices.int()
    if hotspot:
        ids[:] = torch.arange(leaves - chosen, leaves, device="cuda", dtype=torch.int32)
    actual = routed_attention(q, k, v, order, ids, block)
    index = order.long()[..., None].expand_as(k)
    mask = (ids[..., None] == torch.arange(support, device="cuda") // block).any(-2)
    # Explicit FP64 reference avoids an approximate TF32 dense baseline.
    with torch.autocast("cuda", enabled=False):
        scores = (q.double() @ k.gather(2, index).double().transpose(-1, -2)) * dim**-0.5
        # Torch 2.5's CUDA FP64 softmax can mis-normalize odd widths >512.
        # CPU FP64 softmax also provides an independent reduction backend.
        expected = (scores.masked_fill(~mask, -torch.inf).cpu().softmax(-1).to(q.device)
                    @ v.gather(2, index).double()).to(dtype)
    tolerance = 8e-3 if dtype == torch.bfloat16 else 1e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    upstream = torch.randn_like(actual) / 10
    actual_grads = torch.autograd.grad(actual, (q, k, v), upstream)
    expected_grads = torch.autograd.grad(expected, (q, k, v), upstream)
    for grad, reference in zip(actual_grads, expected_grads):
        assert torch.isfinite(grad).all()
        relative_error = (grad.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-10)
        # A one-token leaf has exactly zero Q/K gradient. An output rounded
        # before delta=dot(O,dO) can leave a small absolute residual.
        absolute = 8e-4 if dtype == torch.bfloat16 else 2e-4 if dtype == torch.float16 else 3e-6
        assert relative_error < tolerance or (grad.float() - reference.float()).abs().max() < absolute, relative_error


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("support,queries,dim,value_dim", [(1, 7, 16, 16), (257, 259, 64, 64),
                                                         (2051, 529, 32, 32), (53, 67, 32, 17)])
def test_elu_split_tail_and_unequal_values(dtype, support, queries, dim, value_dim):
    torch.manual_seed(811)
    q = torch.randn(2, 2, queries, dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(2, 2, support, dim, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(2, 2, support, value_dim, device="cuda", dtype=dtype, requires_grad=True)
    pq, pk = (F.elu(q.double()) + 1).clamp_min(1e-6), (F.elu(k.double()) + 1).clamp_min(1e-6)
    expected = (pq @ (pk.transpose(-1, -2) @ v.double()) /
                (pq * pk.sum(-2).unsqueeze(-2)).sum(-1, keepdim=True).clamp_min(1e-6)).to(dtype)
    actual = elu_kernel_attention(q, k, v)
    tol = 5e-3 if dtype == torch.bfloat16 else 1e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    upstream = torch.randn_like(actual) / 10
    grads = torch.autograd.grad(actual, (q, k, v), upstream)
    references = torch.autograd.grad(expected, (q, k, v), upstream)
    for grad, reference in zip(grads, references):
        assert torch.isfinite(grad).all()
        relative_error = (grad.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-10)
        assert relative_error < tol or (grad.float() - reference.float()).abs().max() < 2e-6, relative_error


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [17, 64, 128])
def test_elu_saturated_features_and_denominator_floor(dtype, dim):
    # Both the feature floor and denominator floor must disable their gradient.
    q = torch.full((1, 2, 259, dim), -30., device="cuda", dtype=dtype, requires_grad=True)
    k = torch.full((1, 2, 257, dim), -30., device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn_like(k).requires_grad_()
    actual = elu_kernel_attention(q, k, v)
    pq, pk = (F.elu(q.double()) + 1).clamp_min(1e-6), (F.elu(k.double()) + 1).clamp_min(1e-6)
    expected = (pq @ (pk.transpose(-1, -2) @ v.double()) /
                (pq * pk.sum(-2).unsqueeze(-2)).sum(-1, keepdim=True).clamp_min(1e-6)).to(dtype)
    tol = 5e-3 if dtype == torch.bfloat16 else 3e-5
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=tol)
    grad = torch.randn_like(actual)
    for a, b in zip(torch.autograd.grad(actual, (q, k, v), grad),
                    torch.autograd.grad(expected, (q, k, v), grad)):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=tol)


@pytest.mark.parametrize("queries", [17, 259])
def test_mixed_fp32_qk_bfloat16_values(queries):
    torch.manual_seed(817)
    q = torch.randn(1, 2, queries, 32, device="cuda", requires_grad=True)
    k = torch.randn(1, 2, 193, 32, device="cuda", requires_grad=True)
    v = torch.randn_like(k, dtype=torch.bfloat16).requires_grad_()
    order = torch.arange(193, device="cuda", dtype=torch.int32).expand(1, 2, -1).contiguous()
    ids = torch.tensor([0, 2], device="cuda", dtype=torch.int32).expand(1, 2, queries, -1).contiguous()
    mask = ((torch.arange(193, device="cuda") // 64)[None, :] == ids[..., None]).any(-2)
    expected = ((q.double() @ k.double().transpose(-1, -2) / 32**0.5)
                .masked_fill(~mask, -torch.inf).cpu().softmax(-1).to(q.device) @ v.double()).float()
    actual = routed_attention(q, k, v, order, ids, 64)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    grad = torch.randn_like(actual) / 10
    for a, b in zip(torch.autograd.grad(actual, (q, k, v), grad),
                    torch.autograd.grad(expected, (q, k, v), grad)):
        if a.dtype == torch.bfloat16:
            # Final V-gradient cast may straddle one BF16 rounding boundary.
            assert (a.float() - b.float()).norm() / b.float().norm() < 5e-3
        else:
            torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dim", [16, 64, 128, 256])
def test_qk_norm_projection_layout_dtype_and_gradients(dtype, dim):
    from tabicl._model.qk_norm import ICLQKNorm

    torch.manual_seed(812)
    norm = ICLQKNorm(dim, 3).cuda()
    norm.q_weight.data.uniform_(0.5, 1.5)
    x = torch.randn(2, 137, 3, dim, device="cuda", dtype=dtype).transpose(1, 2).requires_grad_()
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        actual = norm.normalize_q(x)
    reference = F.rms_norm(x.double(), (dim,), norm.q_weight.to(dtype).double(), norm.eps).to(dtype)
    assert actual.dtype == dtype
    assert actual.is_contiguous()
    tol = 5e-3 if dtype == torch.bfloat16 else 8e-4 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(actual, reference, atol=tol, rtol=tol)
    grad = torch.randn_like(actual) / 100
    observed = torch.autograd.grad(actual, (x, norm.q_weight), grad)
    expected = torch.autograd.grad(reference, (x, norm.q_weight), grad)
    for a, b in zip(observed, expected):
        torch.testing.assert_close(a, b, atol=tol / 10, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["global", "sparse", "fused_sparse"])
def test_derived_cache_reuse_invalidation_and_transforms(dtype, mode):
    from tabicl._model.icl_attention import SetRetrievalAttention
    from tabicl._model.layers import GlobalLinearAttention
    from tabicl._model.kv_cache import KVCacheEntry

    torch.manual_seed(813)
    model = (GlobalLinearAttention(64, 2, qk_norm=True) if mode == "global" else
             SetRetrievalAttention(64, 2, mode=mode, qk_norm=True,
                                   block_size=64, num_selected_blocks=2)).cuda().eval()
    q = torch.randn(2, 257, 64, device="cuda")
    k = torch.randn(2, 513, 64, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        expected, key, value = model(q, k, k, need_kv=True)
        entry = KVCacheEntry(key=key, value=value)
        actual = model(q, cached_kv=entry)
        state = entry._derived["state"]
        again = model(q, cached_kv=entry)
        assert entry._derived["state"] is state
        tol = 5e-3 if dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
        torch.testing.assert_close(again, actual)
        entry.value.add_(0.125)
        changed = model(q, cached_kv=entry)
        assert entry._derived["state"] is not state
        fresh = KVCacheEntry(key=entry.key, value=entry.value)
        torch.testing.assert_close(changed, model(q, cached_kv=fresh))
        sliced = entry[:1]
        moved = sliced.to("cuda", dtype=dtype)
        concatenated = KVCacheEntry.concat([sliced, entry[1:]])
        assert not sliced._derived and not moved._derived and not concatenated._derived
        torch.testing.assert_close(model(q, cached_kv=concatenated), changed)
        model(q, cached_kv=moved) if q.shape[0] == 1 else model(q[:1], cached_kv=moved)
        state = entry._derived["state"]
        model.qk_norm.q_weight.add_(0.01)
        model(q, cached_kv=entry)
        assert entry._derived["state"] is not state
        entry[:1] = sliced
        assert not entry._derived


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_inference_tensor_cache_uses_safe_fallback(dtype):
    from tabicl._model.kv_cache import KVCacheEntry
    from tabicl._model.icl_attention import SetRetrievalAttention

    model = SetRetrievalAttention(64, 2, mode="fused_sparse", qk_norm=True,
                                  block_size=64, num_selected_blocks=2).cuda().eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        q, k = torch.randn(1, 259, 64, device="cuda"), torch.randn(1, 257, 64, device="cuda")
        expected, key, value = model(q, k, k, need_kv=True)
        entry = KVCacheEntry(key=key, value=value)
        actual = model(q, cached_kv=entry)
        assert not entry._derived
        torch.testing.assert_close(actual, expected, atol=5e-3 if dtype == torch.bfloat16 else 2e-5, rtol=5e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["dense", "global", "global_full", "sparse", "global_sparse", "global_sparse_alternating"])
def test_amp_checkpoint_optimizer_and_full_parameter_gradients(dtype, mode):
    import copy
    from tabicl._model.encoders import Encoder

    torch.manual_seed(814)
    model = Encoder(3, 64, 2, 128, attention_type=mode, block_size=64,
                    num_selected_blocks=2, qk_norm=True, zero_init=False).cuda().train()
    replay = copy.deepcopy(model)
    replay.recompute = True
    # Same parameter names/shapes: existing checkpoints require no migration.
    replay.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(2, 769, 64, device="cuda", requires_grad=True)
    y = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        a, b = model(x, train_size=513), replay(y, train_size=513)
        la, lb = a[:, 513:].float().square().mean(), b[:, 513:].float().square().mean()
    la.backward()
    lb.backward()
    tol = 5e-3 if dtype == torch.bfloat16 else 3e-5
    torch.testing.assert_close(a, b, atol=tol, rtol=tol)
    torch.testing.assert_close(x.grad, y.grad, atol=tol / 1000, rtol=tol)
    for p, r in zip(model.parameters(), replay.parameters()):
        if not p.requires_grad:
            continue
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, r.grad, atol=tol / 100, rtol=tol)
    torch.optim.AdamW(model.parameters(), lr=1e-4).step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_dynamic_support_matches_eager_output_and_gradients(dtype, monkeypatch):
    import copy
    from tabicl._model.encoders import Encoder

    torch.manual_seed(818)
    eager = Encoder(2, 64, 2, 128, attention_type="global_sparse", sparse_every=1,
                    block_size=64, num_selected_blocks=2, qk_norm=True,
                    zero_init=False).cuda().train()
    candidate = copy.deepcopy(eager)
    # Compilation can fuse BF16 projection/normalization arithmetic. A rounded
    # top-k tie can then change a whole leaf: test numerical kernels/gradients
    # on identical routes, rather than asserting continuity of hard retrieval.
    # The integration benchmark separately runs the real dynamic router.
    from tabicl._model.icl_attention import SetRetrievalAttention

    def fixed_route(self, q, k, prepared=None):
        b, h, n, _ = q.shape
        order = torch.arange(k.shape[-2], device=k.device, dtype=torch.int32).expand(b, h, -1).contiguous()
        ids = torch.arange(self.num_selected_blocks, device=k.device, dtype=torch.int32)
        return order, ids.expand(b, h, n, -1).contiguous()

    monkeypatch.setattr(SetRetrievalAttention, "route", fixed_route)
    compiled = torch.compile(candidate, dynamic=True)
    for support in (257, 321, 385):
        eager.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        x = torch.randn(1, 513, 64, device="cuda", requires_grad=True)
        y = x.detach().clone().requires_grad_()
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            a, b = eager(x, train_size=support), compiled(y, train_size=support)
            la, lb = a[:, support:].float().square().mean(), b[:, support:].float().square().mean()
        la.backward()
        lb.backward()
        tol = 1e-2 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)
        torch.testing.assert_close(x.grad, y.grad, atol=tol / 1000, rtol=tol)
        for p, r in zip(eager.parameters(), candidate.parameters()):
            if p.requires_grad:
                assert p.grad is not None and r.grad is not None
                assert torch.isfinite(r.grad).all()
                torch.testing.assert_close(p.grad, r.grad, atol=tol / 100, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_random_lengths_reuse_triton_variants(dtype):
    from tabicl._model import _sparse_grouped, _norm_triton, _elu_kernel
    from tabicl._model.qk_norm import ICLQKNorm

    kernels = [_sparse_grouped._count, _sparse_grouped._scatter, _sparse_grouped._leaf_tiles,
               _norm_triton._norm, _norm_triton._norm_backward,
               _elu_kernel._elu_state, _elu_kernel._elu_backward_query]

    def counts():
        result = []
        for kernel in kernels:
            cache = getattr(kernel, "device_caches", None)
            result.append(sum(len(v[0]) for v in cache.values()) if cache is not None else
                          sum(len(v) for v in kernel.cache.values()))
        return result

    norm = ICLQKNorm(32, 2).cuda()
    for i, (queries, support) in enumerate([(259, 257), (323, 513), (387, 769)]):
        q = torch.randn(1, 2, queries, 32, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn(1, 2, support, 32, device="cuda", dtype=dtype, requires_grad=True)
        v = torch.randn_like(k).requires_grad_()
        order = torch.arange(support, device="cuda", dtype=torch.int32).expand(1, 2, -1).contiguous()
        ids = torch.tensor([0, 2], device="cuda", dtype=torch.int32).expand(1, 2, queries, -1).contiguous()
        out = routed_attention(q, k, v, order, ids, 64) + elu_kernel_attention(q, k, v) + norm.normalize_q(q)
        out.float().square().mean().backward()
        assert torch.isfinite(out).all() and torch.isfinite(q.grad).all()
        if i == 1:
            warmed = counts()
        elif i == 2:
            assert counts() == warmed
