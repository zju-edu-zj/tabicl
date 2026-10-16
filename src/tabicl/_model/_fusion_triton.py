"""Fused retrieval/global gate and head-to-token layout conversion."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _forward(Q, SPARSE, GLOBAL, WEIGHT, BIAS, OUT, GATE, N,
             H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    tile, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = tile * BT + tl.arange(0, BT)
    d = tl.arange(0, BD)
    head_offset = ((b * H + h) * N + row[:, None]) * D + d[None, :]
    mask = (row[:, None] < N) & (d[None, :] < D)
    q = tl.load(Q + head_offset, mask, other=0.0).to(tl.float32)
    weight = tl.load(WEIGHT + h * D + d, d < D, other=0.0).to(tl.float32)
    bias = tl.load(BIAS + h).to(tl.float32)
    gate = tl.sigmoid(tl.sum(q * weight[None, :], axis=1) + bias)
    sparse = tl.load(SPARSE + head_offset, mask, other=0.0).to(tl.float32)
    global_ = tl.load(GLOBAL + head_offset, mask, other=0.0).to(tl.float32)
    token_offset = ((b * N + row[:, None]) * H + h) * D + d[None, :]
    tl.store(OUT + token_offset, global_ + gate[:, None] * (sparse - global_), mask)
    tl.store(GATE + (b * H + h) * N + row, gate, row < N)


@triton.jit(do_not_specialize=["N"])
def _backward(Q, SPARSE, GLOBAL, WEIGHT, GATE, DOUT, DQ, DSPARSE, DGLOBAL,
              DWEIGHT, DBIAS, N, H: tl.constexpr, D: tl.constexpr,
              BT: tl.constexpr, BD: tl.constexpr):
    tile, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = tile * BT + tl.arange(0, BT)
    d = tl.arange(0, BD)
    mask = (row[:, None] < N) & (d[None, :] < D)
    head_offset = ((b * H + h) * N + row[:, None]) * D + d[None, :]
    token_offset = ((b * N + row[:, None]) * H + h) * D + d[None, :]
    q = tl.load(Q + head_offset, mask, other=0.0).to(tl.float32)
    sparse = tl.load(SPARSE + head_offset, mask, other=0.0).to(tl.float32)
    global_ = tl.load(GLOBAL + head_offset, mask, other=0.0).to(tl.float32)
    dout = tl.load(DOUT + token_offset, mask, other=0.0).to(tl.float32)
    gate = tl.load(GATE + (b * H + h) * N + row, row < N, other=0.0)
    weight = tl.load(WEIGHT + h * D + d, d < D, other=0.0).to(tl.float32)
    dlogit = tl.sum(dout * (sparse - global_), axis=1) * gate * (1.0 - gate)
    tl.store(DQ + head_offset, dlogit[:, None] * weight[None, :], mask)
    tl.store(DSPARSE + head_offset, dout * gate[:, None], mask)
    tl.store(DGLOBAL + head_offset, dout * (1.0 - gate[:, None]), mask)
    dw = tl.sum(dlogit[:, None] * q, axis=0)
    db = tl.sum(dlogit, axis=0)
    tl.atomic_add(DWEIGHT + h * D + d, dw, d < D, sem="relaxed")
    tl.atomic_add(DBIAS + h, db, sem="relaxed")


class _FusedGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, sparse, global_, weight, bias):
        b, h, n, d = q.shape
        q, sparse, global_ = q.contiguous(), sparse.contiguous(), global_.contiguous()
        out = torch.empty((b, n, h * d), device=q.device, dtype=q.dtype)
        gate = torch.empty((b, h, n), device=q.device, dtype=torch.float32)
        _forward[(triton.cdiv(n, 32), h, b)](
            q, sparse, global_, weight, bias, out, gate, n, h, d, 32,
            triton.next_power_of_2(d), num_warps=4
        )
        ctx.save_for_backward(q, sparse, global_, weight, gate)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, sparse, global_, weight, gate = ctx.saved_tensors
        b, h, n, d = q.shape
        dq, dsparse, dglobal = torch.empty_like(q), torch.empty_like(sparse), torch.empty_like(global_)
        dweight = torch.zeros_like(weight, dtype=torch.float32)
        dbias = torch.zeros((h,), device=q.device, dtype=torch.float32)
        _backward[(triton.cdiv(n, 32), h, b)](
            q, sparse, global_, weight, gate, dout.contiguous(), dq, dsparse, dglobal,
            dweight, dbias, n, h, d, 32, triton.next_power_of_2(d), num_warps=4
        )
        return dq, dsparse, dglobal, dweight.to(weight.dtype), dbias.to(weight.dtype)


def fused_gate(q, sparse, global_, weight, bias):
    return _FusedGate.apply(q, sparse, global_, weight, bias)
