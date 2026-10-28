"""RMSNorm with FP32 reductions and a head-contiguous, input-dtype output.

The projection views need not be contiguous. Weight gradients use private
partials rather than atomics so checkpoint replay has a stable reduction.
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit(do_not_specialize=["ROWS", "N", "SB", "SH"])
def _norm(X, W, Y, R, ROWS, N, H: tl.constexpr, D: tl.constexpr,
          SB, SH, SN: tl.constexpr, SD: tl.constexpr,
          EPS: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    d = tl.arange(0, BD)
    valid = row < ROWS
    x = tl.load(X + row[:, None] // (H * N) * SB +
                (row[:, None] // N % H) * SH + (row[:, None] % N) * SN + d[None, :] * SD,
                valid[:, None] & (d[None, :] < D), other=0).to(tl.float32)
    w = tl.load(W + d, d < D, other=0).to(X.dtype.element_ty).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    tl.store(Y + row[:, None] * D + d[None, :], x * inv[:, None] * w[None, :],
             valid[:, None] & (d[None, :] < D))
    tl.store(R + row, inv, valid)


@triton.jit(do_not_specialize=["ROWS", "N", "SB", "SH"])
def _norm_backward(X, W, R, DY, DX, DW, ROWS, N, H: tl.constexpr, D: tl.constexpr,
                   SB, SH, SN: tl.constexpr, SD: tl.constexpr,
                   BD: tl.constexpr, BR: tl.constexpr):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    d = tl.arange(0, BD)
    valid = (row[:, None] < ROWS) & (d[None, :] < D)
    x = tl.load(X + row[:, None] // (H * N) * SB +
                (row[:, None] // N % H) * SH + (row[:, None] % N) * SN + d[None, :] * SD,
                valid, other=0).to(tl.float32)
    w = tl.load(W + d, d < D, other=0).to(X.dtype.element_ty).to(tl.float32)
    inv = tl.load(R + row, row < ROWS, other=0)
    xn = x * inv[:, None]
    dy = tl.load(DY + row[:, None] * D + d[None, :], valid, other=0).to(tl.float32)
    weighted = dy * w[None, :]
    dx = (weighted - xn * (tl.sum(weighted * xn, 1) / D)[:, None]) * inv[:, None]
    tl.store(DX + row[:, None] * D + d[None, :], dx, valid)
    tl.store(DW + tl.program_id(0) * D + d, tl.sum(dy * xn, 0), d < D)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        b, h, n, d = x.shape
        rows = b * h * n
        output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        inv = torch.empty(rows, device=x.device, dtype=torch.float32)
        _norm[(triton.cdiv(rows, 32),)](
            x, weight, output, inv, rows, n, h, d,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3), eps,
            triton.next_power_of_2(d), 32, num_warps=4,
        )
        ctx.save_for_backward(x, weight, inv)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        x, weight, inv = ctx.saved_tensors
        b, h, n, d = x.shape
        rows = b * h * n
        tiles = triton.cdiv(rows, 32)
        dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        partial = torch.empty((tiles, d), device=x.device, dtype=torch.float32)
        _norm_backward[(tiles,)](
            x, weight, inv, grad.contiguous(), dx, partial, rows, n, h, d,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            triton.next_power_of_2(d), 32, num_warps=4,
        )
        return dx, partial.sum(0).to(x.dtype).to(weight.dtype), None


@torch.compiler.disable
def rms_norm(x, weight, eps):
    # Torch 2.5 cannot trace symbolic strides through this autograd Function.
    # Keep this fused kernel opaque; surrounding projections/FFN can compile.
    return _RMSNorm.apply(x, weight, eps)
