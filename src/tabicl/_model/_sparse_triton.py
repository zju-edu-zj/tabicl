"""Direct indexed attention over routed support blocks (CUDA only).

Forward and dQ group queries by support leaf to reuse K/V in matrix tiles.
Backward computes dK/dV per support leaf without token-wise gradient atomics.
Small queries and unsupported block sizes retain the direct reference path.
Routing and ELU fusion live outside. FP32 GEMMs use three-term TF32.
"""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "S"])
def _forward(
    Q,
    K,
    V,
    I,
    ORDER,
    O,
    LSE,
    N,
    S,
    H: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    NC: tl.constexpr,
    C: tl.constexpr,
    SCALE: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    r = tl.arange(0, C)
    d = tl.arange(0, D)
    blocks = tl.load(I + ((b * H + h) * N + t) * NC + r // BS, r < NC * BS, other=0)
    sorted_pos = blocks * BS + r % BS
    valid_row = (r < NC * BS) & (sorted_pos < S)
    pos = tl.load(ORDER + (b * H + h) * S + sorted_pos, valid_row, other=0)
    off = ((b * H + h) * S + pos[:, None]) * D + d[None, :]
    key = tl.load(K + off, valid_row[:, None], other=0).to(tl.float32)
    value = tl.load(V + off, valid_row[:, None], other=0).to(tl.float32)
    q = tl.load(Q + (((b * H + h) * N + t) * D + d)).to(tl.float32)
    logits = tl.sum(key * q[None, :], 1) * SCALE
    logits = tl.where(valid_row, logits, -float("inf"))
    m = tl.max(logits, 0)
    ex = tl.exp(logits - m)
    z = tl.sum(ex, 0)
    prob = ex / z
    out = tl.sum(prob[:, None] * value, 0)
    tl.store(O + (((b * H + h) * N + t) * D + d), out)
    tl.store(LSE + (b * H + h) * N + t, m + tl.log(z))


@triton.jit(do_not_specialize=["N", "S"])
def _backward(
    Q,
    K,
    V,
    I,
    ORDER,
    O,
    LSE,
    DO,
    DQ,
    DK,
    DV,
    N,
    S,
    H: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    NC: tl.constexpr,
    C: tl.constexpr,
    SCALE: tl.constexpr,
    QUERY_ONLY: tl.constexpr = False,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    r = tl.arange(0, C)
    d = tl.arange(0, D)
    blocks = tl.load(I + ((b * H + h) * N + t) * NC + r // BS, r < NC * BS, other=0)
    sorted_pos = blocks * BS + r % BS
    valid_row = (r < NC * BS) & (sorted_pos < S)
    pos = tl.load(ORDER + (b * H + h) * S + sorted_pos, valid_row, other=0)
    off = ((b * H + h) * S + pos[:, None]) * D + d[None, :]
    valid = valid_row[:, None]
    key = tl.load(K + off, valid, other=0).to(tl.float32)
    value = tl.load(V + off, valid, other=0).to(tl.float32)
    qo = ((b * H + h) * N + t) * D + d
    q = tl.load(Q + qo).to(tl.float32)
    dout = tl.load(DO + qo).to(tl.float32)
    out = tl.load(O + qo).to(tl.float32)
    lse = tl.load(LSE + (b * H + h) * N + t)
    score = tl.sum(key * q[None, :], 1) * SCALE
    p = tl.where(valid_row, tl.exp(score - lse), 0.0)
    dp = tl.sum(value * dout[None, :], 1)
    delta = tl.sum(out * dout, 0)
    ds = p * (dp - delta) * SCALE
    dq = tl.sum(ds[:, None] * key, 0)
    tl.store(DQ + qo, dq)
    if not QUERY_ONLY:
        tl.atomic_add(DK + off, ds[:, None] * q[None, :], valid, sem="relaxed")
        tl.atomic_add(DV + off, p[:, None] * dout[None, :], valid, sem="relaxed")


@triton.jit(do_not_specialize=["EDGES", "N", "NK"])
def _edge_count(
    I, COUNTS, EDGES, N, NC: tl.constexpr, NK, TILE: tl.constexpr
):
    edge = tl.program_id(0) * TILE + tl.arange(0, TILE)
    block = tl.load(I + edge, edge < EDGES, other=0)
    key = edge // (N * NC) * NK + block
    tl.atomic_add(COUNTS + key, 1, edge < EDGES, sem="relaxed")


@triton.jit(do_not_specialize=["EDGES", "N", "NK"])
def _edge_scatter(
    I,
    STARTS,
    CURSOR,
    REVERSE,
    EDGES,
    N,
    NC: tl.constexpr,
    NK,
    TILE: tl.constexpr,
):
    edge = tl.program_id(0) * TILE + tl.arange(0, TILE)
    block = tl.load(I + edge, edge < EDGES, other=0)
    key = edge // (N * NC) * NK + block
    slot = tl.atomic_add(CURSOR + key, 1, edge < EDGES, sem="relaxed")
    start = tl.load(STARTS + key, edge < EDGES, other=0)
    tl.store(REVERSE + start + slot, edge // NC % N, edge < EDGES)


@triton.jit(do_not_specialize=["N", "S", "NK"])
def _backward_kv(
    Q,
    K,
    V,
    O,
    LSE,
    DO,
    STARTS,
    REVERSE,
    ORDER,
    DK,
    DV,
    N,
    S,
    D: tl.constexpr,
    NK,
    BS: tl.constexpr,
    SCALE: tl.constexpr,
    BT: tl.constexpr = 32,
    BQ: tl.constexpr = 32,
    NC: tl.constexpr = 1,
    EDGE_IDS: tl.constexpr = False,
):
    leaf, bh, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r = leaf * BS + tile * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    valid_row = r < S
    original = tl.load(ORDER + bh * S + r, valid_row, other=0)
    ko = (bh * S + original[:, None]) * D + d[None, :]
    key = tl.load(K + ko, valid_row[:, None], other=0).to(Q.dtype.element_ty)
    value = tl.load(V + ko, valid_row[:, None], other=0).to(Q.dtype.element_ty)
    start = tl.load(STARTS + bh * NK + leaf)
    end = tl.load(STARTS + bh * NK + leaf + 1)
    dk = tl.full((BT, D), 0, tl.float32)
    dv = tl.full((BT, D), 0, tl.float32)
    for begin in range(start, end, BQ):
        offset = begin + tl.arange(0, BQ)
        edge = tl.load(REVERSE + offset, offset < end, other=-1)
        valid_query = (offset < end) & (edge >= 0)
        qi = edge // NC % N if EDGE_IDS else edge
        qo = (bh * N + qi[:, None]) * D + d[None, :]
        q = tl.load(Q + qo, valid_query[:, None], other=0)
        dout = tl.load(DO + qo, valid_query[:, None], other=0)
        out = tl.load(O + qo, valid_query[:, None], other=0).to(tl.float32)
        lse = tl.load(LSE + bh * N + qi, valid_query, other=0)
        score = tl.dot(key, tl.trans(q), input_precision="tf32x3") * SCALE
        p = tl.exp(score - lse[None, :])
        p = tl.where(valid_row[:, None] & valid_query[None, :], p, 0.0)
        dp = tl.dot(value, tl.trans(dout), input_precision="tf32x3")
        delta = tl.sum(out * dout.to(tl.float32), axis=1)
        ds = p * (dp - delta[None, :]) * SCALE
        dk += tl.dot(ds.to(q.dtype), q, input_precision="tf32x3")
        dv += tl.dot(p.to(dout.dtype), dout, input_precision="tf32x3")
    tl.store(DK + ko, dk, valid_row[:, None])
    tl.store(DV + ko, dv, valid_row[:, None])


class _RoutedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, order, indices, block_size):
        b, h, n, d = q.shape
        s = k.shape[2]
        nc = indices.shape[-1]
        c = triton.next_power_of_2(nc * block_size)
        # Compact head-major layout shared by all kernels.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        indices = indices.contiguous()
        order = order.contiguous()
        scale = d**-0.5
        # Wide scalar reductions are expensive and unreliable on older Triton;
        # use matrix tiles for wide leaves even when there are few queries.
        # The support tile must divide a leaf exactly.
        ctx.grouped = block_size % 32 == 0 and (n >= 256 or block_size > 128)
        metadata = ()
        if ctx.grouped:
            from ._sparse_grouped import grouped_attention

            output, lse, metadata = grouped_attention(q, k, v, order, indices, block_size)
        else:
            output = torch.empty_like(q)
            lse = torch.empty((b, h, n), device=q.device, dtype=torch.float32)
            _forward[(n, h, b)](q, k, v, indices, order, output, lse, n, s, h, d, block_size, nc, c, scale, num_warps=8)
        ctx.save_for_backward(q, k, v, order, indices, output, lse, *metadata)
        ctx.args = (block_size, nc, c, scale)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        q, k, v, order, indices, out, lse, *metadata = ctx.saved_tensors
        bs, nc, c, scale = ctx.args
        b, h, n, d = q.shape
        s = k.shape[2]
        dq = torch.empty_like(q)
        # Many queries share a tiny support prefix. FP32 atomic reduction can
        # land on opposite sides of a BF16 rounding boundary on two backward
        # executions (including checkpoint replay). Wider accumulation keeps
        # this small-prefix case stable without changing long-table kernels.
        dout = dout.contiguous()
        if ctx.grouped:
            from ._sparse_grouped import grouped_attention

            dq, _, _ = grouped_attention(q, k, v, order, indices, bs,
                                         metadata, (out, lse, dout))
            nk = triton.cdiv(s, bs)
            dk, dv = torch.empty_like(k), torch.empty_like(v)
            starts, reverse = metadata
            _backward_kv[(nk, b * h, triton.cdiv(bs, 32))](
                q, k, v, out, lse, dout, starts, reverse, order, dk, dv,
                n, s, d, nk, bs, scale, NC=nc, EDGE_IDS=True, num_warps=4, num_stages=1,
            )
        else:
            accumulation_dtype = torch.float64 if k.dtype == torch.bfloat16 and s <= 512 else torch.float32
            dk = torch.zeros(k.shape, device=k.device, dtype=accumulation_dtype)
            dv = torch.zeros(v.shape, device=v.device, dtype=accumulation_dtype)
            _backward[(n, h, b)](
                q, k, v, indices, order, out, lse, dout, dq, dk, dv, n, s, h, d, bs, nc, c, scale, num_warps=8
            )
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None


def routed_attention(q, k, v, order, indices, block_size):
    if q.shape[-1] not in (16, 32, 64, 128):
        raise ValueError("Triton routed attention supports head dimensions 16/32/64/128")
    if indices.ndim != 4 or indices.shape[:3] != q.shape[:3]:
        raise ValueError("Expected per-query/per-head block indices [B,H,N,K]")
    if indices.shape[-1] * block_size > 2048:
        raise ValueError("Triton routed attention supports at most 2048 candidate tokens")
    if order.dtype != torch.int32 or indices.dtype != torch.int32:
        raise ValueError("Routing metadata must use int32")
    return _RoutedAttention.apply(q, k, v, order, indices, block_size)
