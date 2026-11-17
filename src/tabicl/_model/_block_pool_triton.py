"""CUDA kernels for differentiable content-tree leaf means."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["S"])
def _block_mean_forward(X, ORDER, OUT, S, D: tl.constexpr, BS: tl.constexpr,
                        RB: tl.constexpr, BD: tl.constexpr):
    leaf = tl.program_id(0)
    feature_block = tl.program_id(1)
    bh = tl.program_id(2)
    offset = tl.arange(0, RB)
    feature = feature_block * BD + tl.arange(0, BD)
    row = leaf * BS + offset
    valid_row = (offset < BS) & (row < S)
    valid = valid_row[:, None] & (feature[None, :] < D)
    original = tl.load(ORDER + bh * S + row, valid_row, other=0)
    values = tl.load(X + (bh * S + original[:, None]) * D + feature[None, :],
                     valid, other=0.0)
    length = tl.minimum(BS, S - leaf * BS)
    mean = tl.sum(values.to(tl.float32), axis=0) / length
    leaves = tl.cdiv(S, BS)
    tl.store(OUT + (bh * leaves + leaf) * D + feature, mean, feature < D)


@triton.jit(do_not_specialize=["S"])
def _block_mean_backward(DOUT, ORDER, DX, S, D: tl.constexpr, BS: tl.constexpr,
                         BR: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    feature = tl.program_id(1) * BD + tl.arange(0, BD)
    bh = tl.program_id(2)
    valid = (row[:, None] < S) & (feature[None, :] < D)
    original = tl.load(ORDER + bh * S + row, row < S, other=0)
    leaf = row // BS
    length = tl.minimum(BS, S - leaf * BS)
    leaves = tl.cdiv(S, BS)
    grad = tl.load(
        DOUT + (bh * leaves + leaf[:, None]) * D + feature[None, :],
        valid,
        other=0.0,
    )
    tl.store(DX + (bh * S + original[:, None]) * D + feature[None, :],
             grad / length[:, None], valid)


class _BlockMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, order, block_size):
        x = x.contiguous()
        order = order.contiguous()
        b, h, support, d = x.shape
        leaves = (support + block_size - 1) // block_size
        output = torch.empty((b, h, leaves, d), device=x.device, dtype=x.dtype)
        bd = min(32, triton.next_power_of_2(d))
        _block_mean_forward[(leaves, triton.cdiv(d, bd), b * h)](
            x, order, output, support, d, block_size,
            triton.next_power_of_2(block_size), bd, num_warps=4,
        )
        ctx.save_for_backward(order)
        ctx.input_shape = x.shape
        ctx.block_size = block_size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (order,) = ctx.saved_tensors
        b, h, support, d = ctx.input_shape
        grad_output = grad_output.contiguous()
        grad_input = torch.empty(ctx.input_shape, device=grad_output.device,
                                 dtype=grad_output.dtype)
        br, bd = 32, min(32, triton.next_power_of_2(d))
        _block_mean_backward[(triton.cdiv(support, br), triton.cdiv(d, bd), b * h)](
            grad_output, order, grad_input, support, d, ctx.block_size, br, bd,
            num_warps=4,
        )
        return grad_input, None, None


def block_mean_cuda(x, order, block_size):
    return _BlockMean.apply(x, order, block_size)
