import pytest
import torch
from tabicl._model.layers import GlobalLinearAttention


@pytest.mark.parametrize("cls", [GlobalLinearAttention])
def test_kernel_attention_matches_explicit_kernel_and_gradients(cls):
    torch.manual_seed(42)
    module = cls(32, 4, ssmax=True)
    assert module.ssmax_layer is None
    q = torch.randn(2, 9, 32, requires_grad=True)
    k = torch.randn(2, 17, 32, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    output = module(q, k, v)
    qp, kp, vp = module._project(q, k, v, None)
    weights = module._feature_map(qp) @ module._feature_map(kp).transpose(-1, -2)
    weights = weights / weights.sum(-1, keepdim=True)
    ref = module.out_proj((weights @ vp).transpose(1, 2).reshape_as(q))
    assert torch.allclose(output, ref, atol=1e-6, rtol=1e-5)
    grads = torch.autograd.grad(output.square().mean(), (q, k, v), retain_graph=True)
    reference = torch.autograd.grad(ref.square().mean(), (q, k, v))
    for actual, expected in zip(grads, reference):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("cls", [GlobalLinearAttention])
def test_kernel_diagnostics_and_support_permutation(cls):
    torch.manual_seed(43)
    module = cls(32, 4).eval()
    module.diagnostics_enabled = True
    q, k = torch.randn(2, 13, 32), torch.randn(2, 521, 32)
    output = module(q, k, k)
    assert module.last_diagnostics["denominator_min"] > 0
    assert module.last_diagnostics["nonfinite_fraction"] == 0
    assert not module.diagnostics_enabled
    permutation = torch.randperm(k.shape[1])
    assert torch.allclose(output, module(q, k[:, permutation], k[:, permutation]), atol=1e-5, rtol=1e-4)
