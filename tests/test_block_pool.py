import pytest
import torch

from tabicl._model._block_pool import block_mean


def _permutation(batch, heads, support, device):
    return torch.rand(batch, heads, support, device=device).argsort(-1).to(torch.int32)


def _block_mean_reference(x, order, block_size):
    gathered = x.gather(
        2, order.long().unsqueeze(-1).expand(-1, -1, -1, x.shape[-1])
    )
    return torch.stack(
        [gathered[:, :, start:start + block_size].float().mean(2).to(x.dtype)
         for start in range(0, x.shape[2], block_size)],
        dim=2,
    )


@pytest.mark.parametrize("support,block_size", [(1, 4), (13, 4), (17, 5), (33, 64)])
def test_block_mean_cpu_output_and_gradient(support, block_size):
    torch.manual_seed(1201 + support)
    x = torch.randn(2, support, 3, 7).transpose(1, 2).requires_grad_()
    order = _permutation(2, 3, support, "cpu")
    actual = block_mean(x, order, block_size)
    expected = _block_mean_reference(x, order, block_size)
    torch.testing.assert_close(actual, expected)

    upstream = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, x, upstream, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, x, upstream)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_block_mean_validates_metadata():
    x = torch.randn(1, 2, 7, 3)
    order = torch.arange(7, dtype=torch.int64).expand(1, 2, -1)
    with pytest.raises(ValueError, match="int32"):
        block_mean(x, order, 4)
    with pytest.raises(ValueError, match="positive"):
        block_mean(x, order.int(), 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("support,block_size,dim", [(193, 64, 16), (137, 48, 19), (31, 64, 7)])
def test_block_mean_cuda_output_and_gradient(dtype, support, block_size, dim):
    torch.manual_seed(1251 + support)
    x = torch.randn(2, support, 3, dim, device="cuda", dtype=dtype).transpose(1, 2)
    x.requires_grad_()
    order = _permutation(2, 3, support, "cuda")
    actual = block_mean(x, order, block_size)
    expected = _block_mean_reference(x, order, block_size)
    tolerance = 3e-3 if dtype == torch.bfloat16 else 8e-4 if dtype == torch.float16 else 2e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    upstream = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, x, upstream, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, x, upstream)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)
