import pytest
import torch
import types
from torch.nn import functional as F

from tabicl._model._elu_kernel import elu_kernel_attention
from tabicl._model.qk_norm import ICLQKNorm


def test_qk_norm_has_bounded_positive_length_scaling():
    norm = ICLQKNorm(16, 4)
    q = torch.randn(2, 4, 7, 16)
    k = torch.randn(2, 4, 11, 16)
    qn, kn = norm.normalize_q(q), norm.normalize_k(k)
    torch.testing.assert_close(qn.float().square().mean(-1), torch.ones_like(qn[..., 0]), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(kn.float().square().mean(-1), torch.ones_like(kn[..., 0]), atol=2e-5, rtol=2e-5)
    short, long = norm.scale_query(qn, 1), norm.scale_query(qn, 1_000_000)
    short_ratio = short.norm(dim=-1) / qn.norm(dim=-1)
    long_ratio = long.norm(dim=-1) / qn.norm(dim=-1)
    assert (short_ratio >= 0.5).all() and (short_ratio <= 2).all()
    assert (long_ratio >= 0.5).all() and (long_ratio <= 2).all()


def test_checkpoint_reuses_discrete_route():
    from tabicl._model.encoders import Encoder

    model = Encoder(3, 32, 2, 64, attention_type="global_sparse", sparse_every=3,
                    block_size=32, num_selected_blocks=2, recompute=True, zero_init=False)
    retrieval = model.blocks[2].attn
    original = retrieval.route
    calls = []

    def counted(self, q, k):
        if self._route_replay is None:
            calls.append(1)
        return original(q, k)

    retrieval.route = types.MethodType(counted, retrieval)
    x = torch.randn(1, 137, 32, requires_grad=True)
    model(x, train_size=97).square().mean().backward()
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["dense", "global", "global_full", "global_sparse_alternating", "global_sparse"])
def test_all_screening_branches_train_with_qk_norm(mode):
    from tabicl._model.encoders import Encoder

    torch.manual_seed(458)
    model = Encoder(3, 32, 2, 64, attention_type=mode, sparse_every=3,
                    block_size=32, num_selected_blocks=2, qk_norm=True,
                    zero_init=False)
    x = torch.randn(1, 137, 32, requires_grad=True)
    output = model(x, train_size=97)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    for block in model.blocks:
        assert block.attn.qk_norm.q_weight.grad is not None
        assert block.attn.qk_norm.k_weight.grad is not None
    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    assert missing == []


def test_elu_global_only_disables_unused_softmax_length_scale():
    from tabicl._model.encoders import Encoder

    elu = Encoder(3, 32, 2, 64, attention_type="global_sparse", sparse_every=3,
                  qk_norm=True, global_type="elu")
    induced = Encoder(3, 32, 2, 64, attention_type="global_sparse", sparse_every=3,
                      qk_norm=True, global_type="induced")
    for block in elu.blocks[:2]:
        assert not block.attn.qk_norm.base_log_scale.requires_grad
        assert not block.attn.qk_norm.log_slope_logit.requires_grad
    assert elu.blocks[2].attn.qk_norm.base_log_scale.requires_grad
    assert elu.blocks[2].attn.qk_norm.log_slope_logit.requires_grad
    assert all(block.attn.qk_norm.base_log_scale.requires_grad for block in induced.blocks)
    assert all(block.attn.qk_norm.log_slope_logit.requires_grad for block in induced.blocks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_gate_matches_layout_and_gradients(dtype):
    from tabicl._model._fusion_triton import fused_gate

    torch.manual_seed(459)
    inputs = [torch.randn(2, 3, 67, 32, device="cuda", dtype=dtype, requires_grad=True)
              for _ in range(3)]
    weight = torch.randn(3, 32, device="cuda", requires_grad=True) / 32**0.5
    bias = torch.randn(3, device="cuda", requires_grad=True)
    q, sparse, global_ = inputs
    gate = torch.sigmoid(torch.einsum("bhnd,hd->bhn", q.float(), weight.float())
                         + bias.float()[None, :, None])[..., None]
    mixed = global_.float() + gate * (sparse.float() - global_.float())
    reference = mixed.to(dtype).transpose(1, 2).reshape(2, 67, 96)
    actual = fused_gate(q, sparse, global_, weight, bias)
    tolerance = 4e-3 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
    upstream = torch.randn_like(actual) / 10
    expected = torch.autograd.grad(reference, (*inputs, weight, bias), upstream)
    gradients = torch.autograd.grad(actual, (*inputs, weight, bias), upstream)
    for grad, target in zip(gradients, expected):
        torch.testing.assert_close(
            grad, target,
            atol=8e-4 if dtype == torch.bfloat16 else 3e-5,
            rtol=0.04 if dtype == torch.bfloat16 else 3e-4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fused_elu_bfloat16_matches_eager_output_and_gradients():
    torch.manual_seed(457)
    q = torch.randn(1, 2, 257, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 2, 193, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    pq, pk = F.elu(q.float()) + 1, F.elu(k.float()) + 1
    reference = (pq @ (pk.transpose(-1, -2) @ v.float()) /
                 (pq * pk.sum(-2).unsqueeze(-2)).sum(-1, keepdim=True)).to(q.dtype)
    actual = elu_kernel_attention(q, k, v)
    torch.testing.assert_close(actual, reference, atol=2e-3, rtol=2e-3)
    upstream = torch.randn_like(actual) / 10
    expected = torch.autograd.grad(reference, (q, k, v), upstream)
    gradients = torch.autograd.grad(actual, (q, k, v), upstream)
    for grad, target in zip(gradients, expected):
        torch.testing.assert_close(grad, target, atol=8e-4, rtol=0.04)


@pytest.mark.parametrize("saturated", [False, True])
def test_recomputed_elu_output_and_gradients(saturated):
    torch.manual_seed(451)
    q = torch.randn(2, 3, 17, 16, requires_grad=True)
    k = torch.randn(2, 3, 23, 16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    if saturated:
        q = (q.detach() - 13).requires_grad_()
        k = (k.detach() - 13).requires_grad_()
    pq, pk = (F.elu(q) + 1).clamp_min(1e-6), (F.elu(k) + 1).clamp_min(1e-6)
    state = pk.transpose(-1, -2) @ v
    den = (pq * pk.sum(-2).unsqueeze(-2)).sum(-1, keepdim=True).clamp_min(1e-6)
    reference = pq @ state / den
    actual = elu_kernel_attention(q, k, v)
    torch.testing.assert_close(actual, reference)
    upstream = torch.randn_like(actual)
    expected = torch.autograd.grad(reference, (q, k, v), upstream)
    gradients = torch.autograd.grad(actual, (q, k, v), upstream)
    for grad, target in zip(gradients, expected):
        torch.testing.assert_close(grad, target, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_chunked_elu_support_and_queries_match_eager_gradients(device):
    torch.manual_seed(455)
    q = torch.randn(1, 1, 16387, 4, device=device, requires_grad=True)
    k = torch.randn(1, 1, 17009, 4, device=device, requires_grad=True)
    v = torch.randn(1, 1, 17009, 7, device=device, requires_grad=True)
    pq, pk = (F.elu(q) + 1).clamp_min(1e-6), (F.elu(k) + 1).clamp_min(1e-6)
    reference = pq @ (pk.transpose(-1, -2) @ v) / (pq * pk.sum(-2).unsqueeze(-2)).sum(-1, keepdim=True).clamp_min(1e-6)
    actual = elu_kernel_attention(q, k, v)
    torch.testing.assert_close(actual, reference, atol=2e-7, rtol=3e-5)
    upstream = torch.randn_like(actual)
    expected = torch.autograd.grad(reference, (q, k, v), upstream)
    gradients = torch.autograd.grad(actual, (q, k, v), upstream)
    for grad, target in zip(gradients, expected):
        torch.testing.assert_close(grad, target, atol=3e-7, rtol=3e-5)
