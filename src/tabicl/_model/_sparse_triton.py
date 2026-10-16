"""Direct indexed attention over routed support blocks (CUDA only).

Forward and dQ use one program per query/head. For half-precision training,
backward inverts compact routing edges and computes dK/dV per support leaf
using tensor-core reductions, without token-wise gradient atomics. Small or
FP32 inputs use a direct atomic reference. Routing and ELU fusion live outside.
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
):
    leaf, bh, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r = leaf * BS + tile * BT + tl.arange(0, BT)
    d = tl.arange(0, D)
    valid_row = r < S
    original = tl.load(ORDER + bh * S + r, valid_row, other=0)
    ko = (bh * S + original[:, None]) * D + d[None, :]
    key = tl.load(K + ko, valid_row[:, None], other=0)
    value = tl.load(V + ko, valid_row[:, None], other=0)
    start = tl.load(STARTS + bh * NK + leaf)
    end = tl.load(STARTS + bh * NK + leaf + 1)
    dk = tl.full((BT, D), 0, tl.float32)
    dv = tl.full((BT, D), 0, tl.float32)
    for begin in range(start, end, BQ):
        offset = begin + tl.arange(0, BQ)
        qi = tl.load(REVERSE + offset, offset < end, other=0)
        qo = (bh * N + qi[:, None]) * D + d[None, :]
        q = tl.load(Q + qo, offset[:, None] < end, other=0)
        dout = tl.load(DO + qo, offset[:, None] < end, other=0)
        out = tl.load(O + qo, offset[:, None] < end, other=0).to(tl.float32)
        lse = tl.load(LSE + bh * N + qi, offset < end, other=0)
        score = tl.dot(key, tl.trans(q)) * SCALE
        p = tl.exp(score - lse[None, :])
        p = tl.where(valid_row[:, None] & (offset[None, :] < end), p, 0.0)
        dp = tl.dot(value, tl.trans(dout))
        delta = tl.sum(out * dout.to(tl.float32), axis=1)
        ds = p * (dp - delta[None, :]) * SCALE
        dk += tl.dot(ds.to(q.dtype), q)
        dv += tl.dot(p.to(dout.dtype), dout)
    tl.store(DK + ko, dk, valid_row[:, None])
    tl.store(DV + ko, dv, valid_row[:, None])


class _RoutedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, order, indices, block_size):
        b, h, n, d = q.shape
        s = k.shape[2]
        nc = indices.shape[-1]
        c = triton.next_power_of_2(nc * block_size)
        # One program per query/head; contiguous head-major layout.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        indices = indices.contiguous()
        order = order.contiguous()
        output = torch.empty_like(q)
        lse = torch.empty((b, h, n), device=q.device, dtype=torch.float32)
        scale = d**-0.5
        _forward[(n, h, b)](q, k, v, indices, order, output, lse, n, s, h, d, block_size, nc, c, scale, num_warps=8)
        ctx.save_for_backward(q, k, v, order, indices, output, lse)
        ctx.args = (block_size, nc, c, scale)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        q, k, v, order, indices, out, lse = ctx.saved_tensors
        bs, nc, c, scale = ctx.args
        b, h, n, d = q.shape
        s = k.shape[2]
        dq = torch.empty_like(q)
        # Many queries share a tiny support prefix. FP32 atomic reduction can
        # land on opposite sides of a BF16 rounding boundary on two backward
        # executions (including checkpoint replay). Wider accumulation keeps
        # this small-prefix case stable without changing long-table kernels.
        dout = dout.contiguous()
        grouped = q.dtype in (torch.bfloat16, torch.float16) and n >= 256 and bs >= 32
        if grouped:
            # Invert only the compact block edges. Each leaf owns its K/V
            # gradients and reduces them with tensor-core GEMMs, without
            # per-token/per-dimension atomic scatter or Q*C*d materialization.
            nk = triton.cdiv(s, bs)
            edges = indices.numel()
            counts = torch.zeros(b * h * nk, device=q.device, dtype=torch.int32)
            _edge_count[(triton.cdiv(edges, 256),)](indices, counts, edges, n, nc, nk, 256)
            starts = torch.empty(counts.numel() + 1, device=q.device, dtype=torch.int32)
            starts[0] = 0
            torch.cumsum(counts, 0, dtype=torch.int32, out=starts[1:])
            cursor = torch.zeros_like(counts)
            reverse = torch.empty(edges, device=q.device, dtype=torch.int32)
            _edge_scatter[(triton.cdiv(edges, 256),)](indices, starts, cursor, reverse, edges, n, nc, nk, 256)
            dk, dv = torch.empty_like(k), torch.empty_like(v)
            _backward[(n, h, b)](
                q, k, v, indices, order, out, lse, dout, dq, dk, dv, n, s, h, d, bs, nc, c, scale, True, num_warps=8
            )
            _backward_kv[(nk, b * h, triton.cdiv(bs, 32))](
                q, k, v, out, lse, dout, starts, reverse, order, dk, dv, n, s, d, nk, bs, scale, num_warps=4
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
