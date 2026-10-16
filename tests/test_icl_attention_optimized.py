import pytest
import torch
from torch.nn import functional as F

from tabicl._model._elu_kernel import elu_kernel_attention


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
