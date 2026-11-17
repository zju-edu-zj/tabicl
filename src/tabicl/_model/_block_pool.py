"""Differentiable block pooling over a routed support permutation."""

import torch


def _cpu_block_mean(x, order, block_size):
    b, h, support, d = x.shape
    leaves = (support + block_size - 1) // block_size
    sorted_x = x.gather(2, order.long().unsqueeze(-1).expand(-1, -1, -1, d))
    if leaves * block_size != support:
        sorted_x = torch.nn.functional.pad(
            sorted_x, (0, 0, 0, leaves * block_size - support)
        )
    sums = sorted_x.reshape(b, h, leaves, block_size, d).float().sum(dim=3)
    lengths = torch.full((leaves,), block_size, device=x.device, dtype=torch.float32)
    lengths[-1] = support - (leaves - 1) * block_size
    return (sums / lengths.view(1, 1, leaves, 1)).to(x.dtype)


def block_mean(x, order, block_size):
    """Mean-pool rows in each ordered leaf with FP32 accumulation."""
    if x.ndim != 4:
        raise ValueError("Expected x shaped [B,H,S,D]")
    if order.shape != x.shape[:3]:
        raise ValueError("Expected order shaped [B,H,S]")
    if order.dtype != torch.int32:
        raise ValueError("order must use int32")
    if order.device != x.device:
        raise ValueError("x and order must be on the same device")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if x.shape[2] == 0:
        raise ValueError("block_mean requires at least one support row")
    if not x.is_cuda:
        return _cpu_block_mean(x, order, block_size)
    from ._block_pool_triton import block_mean_cuda

    return block_mean_cuda(x, order, block_size)
