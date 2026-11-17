"""Unified exact-row and summary-token attention for adaptive frontiers.

The exact branch reuses the grouped routed-attention kernels.  The summary
branch covers every unselected leaf, and both branches are merged with their
log-sum-exp statistics.  This is mathematically one softmax over the union of
exact rows and summary tokens, not a mixture of two normalized attentions.
"""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from ._sparse_grouped import grouped_attention
from ._sparse_triton import _backward_kv


@triton.jit(do_not_specialize=["N", "LEAVES"])
def _summary_forward(
    Q, SK, SV, INDICES, LENGTHS, BIAS, COUNT_SCALE, OUT, LSE,
    N, LEAVES,
    H: tl.constexpr, D: tl.constexpr, TOPK: tl.constexpr, BTOPK: tl.constexpr,
    SLOTS: tl.constexpr, WIDTH: tl.constexpr, BD: tl.constexpr,
    SCALE: tl.constexpr,
):
    n, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p = tl.arange(0, WIDTH)
    d = tl.arange(0, BD)
    leaf = p // SLOTS
    live = p < LEAVES * SLOTS
    selected_offset = tl.arange(0, BTOPK)
    selected = tl.load(
        INDICES + ((b * H + h) * N + n) * TOPK + selected_offset,
        selected_offset < TOPK,
        other=-1,
    )
    hidden = tl.sum((leaf[:, None] == selected[None, :]).to(tl.int32), axis=1) > 0
    visible = live & ~hidden
    q = tl.load(Q + ((b * H + h) * N + n) * D + d, d < D, other=0.0)
    sk = tl.load(
        SK + (((b * H + h) * LEAVES * SLOTS + p[:, None]) * D + d[None, :]),
        visible[:, None] & (d[None, :] < D), other=0.0,
    ).to(q.dtype)
    sv = tl.load(
        SV + (((b * H + h) * LEAVES * SLOTS + p[:, None]) * D + d[None, :]),
        visible[:, None] & (d[None, :] < D), other=0.0,
    ).to(q.dtype)
    length = tl.load(LENGTHS + leaf, live, other=1).to(tl.float32)
    prior = tl.log(length / SLOTS)
    bias = tl.load(BIAS + h).to(tl.float32)
    count_scale = tl.load(COUNT_SCALE + h).to(tl.float32)
    score = tl.sum(sk.to(tl.float32) * q[None, :].to(tl.float32), axis=1) * SCALE
    score += bias + count_scale * prior
    score = tl.where(visible, score, -float("inf"))
    maximum = tl.max(score, axis=0)
    weight = tl.exp(score - maximum)
    normalizer = tl.sum(weight, axis=0)
    output = tl.sum(weight[:, None].to(sv.dtype) * sv, axis=0) / normalizer
    tl.store(OUT + ((b * H + h) * N + n) * D + d, output, d < D)
    tl.store(LSE + (b * H + h) * N + n, maximum + tl.log(normalizer))


@triton.jit(do_not_specialize=["N"])
def _merge_forward(EXACT, EXACT_LSE, SUMMARY, SUMMARY_LSE, OUT, LSE, N,
                   H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    n, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, BD)
    row = (b * H + h) * N + n
    exact_lse = tl.load(EXACT_LSE + row)
    summary_lse = tl.load(SUMMARY_LSE + row)
    maximum = tl.maximum(exact_lse, summary_lse)
    exact_weight = tl.exp(exact_lse - maximum)
    summary_weight = tl.exp(summary_lse - maximum)
    normalizer = exact_weight + summary_weight
    exact = tl.load(EXACT + row * D + d, d < D, other=0.0).to(tl.float32)
    summary = tl.load(SUMMARY + row * D + d, d < D, other=0.0).to(tl.float32)
    output = (exact_weight * exact + summary_weight * summary) / normalizer
    tl.store(OUT + row * D + d, output, d < D)
    tl.store(LSE + row, maximum + tl.log(normalizer))


@triton.jit(do_not_specialize=["N", "LEAVES"])
def _summary_backward_query(
    Q, SK, SV, INDICES, LENGTHS, BIAS, COUNT_SCALE,
    OUT, LSE, DOUT, DQ, N, LEAVES,
    H: tl.constexpr, D: tl.constexpr, TOPK: tl.constexpr, BTOPK: tl.constexpr,
    SLOTS: tl.constexpr, WIDTH: tl.constexpr, BD: tl.constexpr,
    SCALE: tl.constexpr,
):
    n, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p = tl.arange(0, WIDTH)
    d = tl.arange(0, BD)
    leaf = p // SLOTS
    live = p < LEAVES * SLOTS
    selected_offset = tl.arange(0, BTOPK)
    selected = tl.load(
        INDICES + ((b * H + h) * N + n) * TOPK + selected_offset,
        selected_offset < TOPK,
        other=-1,
    )
    hidden = tl.sum((leaf[:, None] == selected[None, :]).to(tl.int32), axis=1) > 0
    visible = live & ~hidden
    row = (b * H + h) * N + n
    q = tl.load(Q + row * D + d, d < D, other=0.0)
    dout = tl.load(DOUT + row * D + d, d < D, other=0.0)
    output = tl.load(OUT + row * D + d, d < D, other=0.0).to(tl.float32)
    lse = tl.load(LSE + row)
    sk = tl.load(
        SK + (((b * H + h) * LEAVES * SLOTS + p[:, None]) * D + d[None, :]),
        visible[:, None] & (d[None, :] < D), other=0.0,
    ).to(q.dtype)
    sv = tl.load(
        SV + (((b * H + h) * LEAVES * SLOTS + p[:, None]) * D + d[None, :]),
        visible[:, None] & (d[None, :] < D), other=0.0,
    ).to(dout.dtype)
    length = tl.load(LENGTHS + leaf, live, other=1).to(tl.float32)
    prior = tl.log(length / SLOTS)
    bias = tl.load(BIAS + h).to(tl.float32)
    count_scale = tl.load(COUNT_SCALE + h).to(tl.float32)
    score = tl.sum(sk.to(tl.float32) * q[None, :].to(tl.float32), axis=1) * SCALE
    score += bias + count_scale * prior
    probability = tl.where(visible, tl.exp(score - lse), 0.0)
    dp = tl.sum(sv.to(tl.float32) * dout[None, :].to(tl.float32), axis=1)
    delta = tl.sum(output * dout.to(tl.float32), axis=0)
    ds = probability * (dp - delta)
    dq = tl.sum(ds[:, None] * sk.to(tl.float32), axis=0) * SCALE
    tl.store(DQ + row * D + d, dq, d < D)


@triton.jit(do_not_specialize=["N", "LEAVES"])
def _summary_backward_support(
    Q, SK, SV, INDICES, LENGTHS, BIAS, COUNT_SCALE,
    OUT, LSE, DOUT, DSK, DSV, DBIAS, DCOUNT,
    N, LEAVES,
    H: tl.constexpr, D: tl.constexpr, TOPK: tl.constexpr, BTOPK: tl.constexpr,
    SLOTS: tl.constexpr, BD: tl.constexpr, BQ: tl.constexpr,
    SCALE: tl.constexpr,
):
    p, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    leaf = p // SLOTS
    d = tl.arange(0, BD)
    sk = tl.load(
        SK + (((b * H + h) * LEAVES * SLOTS + p) * D + d),
        d < D, other=0.0,
    )
    sv = tl.load(
        SV + (((b * H + h) * LEAVES * SLOTS + p) * D + d),
        d < D, other=0.0,
    )
    length = tl.load(LENGTHS + leaf).to(tl.float32)
    prior = tl.log(length / SLOTS)
    bias = tl.load(BIAS + h).to(tl.float32)
    count_scale = tl.load(COUNT_SCALE + h).to(tl.float32)
    dsk = tl.zeros((BD,), tl.float32)
    dsv = tl.zeros((BD,), tl.float32)
    dbias = 0.0
    dcount = 0.0
    for start in range(0, N, BQ):
        n = start + tl.arange(0, BQ)
        live = n < N
        selected_offset = tl.arange(0, BTOPK)
        selected = tl.load(
            INDICES + (((b * H + h) * N + n[:, None]) * TOPK + selected_offset[None, :]),
            live[:, None] & (selected_offset[None, :] < TOPK), other=-1,
        )
        visible = live & ~(tl.sum((selected == leaf).to(tl.int32), axis=1) > 0)
        row = (b * H + h) * N + n
        q = tl.load(Q + row[:, None] * D + d[None, :], live[:, None] & (d[None, :] < D), other=0.0)
        dout = tl.load(
            DOUT + row[:, None] * D + d[None, :], live[:, None] & (d[None, :] < D), other=0.0
        )
        output = tl.load(
            OUT + row[:, None] * D + d[None, :], live[:, None] & (d[None, :] < D), other=0.0
        ).to(tl.float32)
        lse = tl.load(LSE + row, live, other=0.0)
        score = tl.sum(q.to(tl.float32) * sk[None, :].to(tl.float32), axis=1) * SCALE
        score += bias + count_scale * prior
        probability = tl.where(visible, tl.exp(score - lse), 0.0)
        dp = tl.sum(dout.to(tl.float32) * sv[None, :].to(tl.float32), axis=1)
        delta = tl.sum(output * dout.to(tl.float32), axis=1)
        ds = probability * (dp - delta)
        dsk += tl.sum(ds[:, None] * q.to(tl.float32), axis=0) * SCALE
        dsv += tl.sum(probability[:, None] * dout.to(tl.float32), axis=0)
        dbias += tl.sum(ds, axis=0)
        dcount += tl.sum(ds * prior, axis=0)
    offset = ((b * H + h) * LEAVES * SLOTS + p) * D + d
    tl.store(DSK + offset, dsk, d < D)
    tl.store(DSV + offset, dsv, d < D)
    tl.atomic_add(DBIAS + h, dbias, sem="relaxed")
    tl.atomic_add(DCOUNT + h, dcount, sem="relaxed")


class _FrontierAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, order, indices, summary_k, summary_v,
                lengths, summary_bias, log_count_scale, block_size):
        b, h, n, d = q.shape
        leaves, slots = summary_k.shape[2:4]
        topk = indices.shape[-1]
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        order, indices = order.contiguous(), indices.contiguous()
        summary_k, summary_v = summary_k.contiguous(), summary_v.contiguous()
        exact, exact_lse, metadata = grouped_attention(
            q, k, v, order, indices, block_size
        )
        summary = torch.empty_like(q)
        summary_lse = torch.empty((b, h, n), device=q.device, dtype=torch.float32)
        width = triton.next_power_of_2(leaves * slots)
        btopk = triton.next_power_of_2(topk)
        _summary_forward[(n, h, b)](
            q, summary_k, summary_v, indices, lengths, summary_bias,
            log_count_scale, summary, summary_lse, n, leaves, h, d, topk,
            btopk, slots, width, triton.next_power_of_2(d), d**-0.5, num_warps=8,
        )
        output = torch.empty_like(q)
        lse = torch.empty_like(summary_lse)
        _merge_forward[(n, h, b)](
            exact, exact_lse, summary, summary_lse, output, lse,
            n, h, d, triton.next_power_of_2(d), num_warps=4,
        )
        starts, reverse = metadata
        ctx.save_for_backward(
            q, k, v, order, indices, summary_k, summary_v, lengths,
            summary_bias, log_count_scale, output, lse, starts, reverse,
        )
        ctx.args = (block_size, slots)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        (q, k, v, order, indices, summary_k, summary_v, lengths,
         summary_bias, log_count_scale, output, lse, starts, reverse) = ctx.saved_tensors
        block_size, slots = ctx.args
        b, h, n, d = q.shape
        s, leaves, topk = k.shape[2], summary_k.shape[2], indices.shape[-1]
        dout = dout.contiguous()
        dq_exact, _, _ = grouped_attention(
            q, k, v, order, indices, block_size,
            (starts, reverse), (output, lse, dout),
        )
        dq_summary = torch.empty_like(q)
        width = triton.next_power_of_2(leaves * slots)
        _summary_backward_query[(n, h, b)](
            q, summary_k, summary_v, indices, lengths, summary_bias,
            log_count_scale, output, lse, dout, dq_summary, n, leaves,
            h, d, topk, triton.next_power_of_2(topk), slots, width,
            triton.next_power_of_2(d), d**-0.5,
            num_warps=8,
        )
        nk = triton.cdiv(s, block_size)
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        _backward_kv[(nk, b * h, triton.cdiv(block_size, 32))](
            q, k, v, output, lse, dout, starts, reverse, order, dk, dv,
            n, s, d, nk, block_size, d**-0.5,
            NC=topk, EDGE_IDS=True, num_warps=4, num_stages=1,
        )
        dsk, dsv = torch.empty_like(summary_k), torch.empty_like(summary_v)
        dbias = torch.zeros_like(summary_bias, dtype=torch.float32)
        dcount = torch.zeros_like(log_count_scale, dtype=torch.float32)
        _summary_backward_support[(leaves * slots, h, b)](
            q, summary_k, summary_v, indices, lengths, summary_bias,
            log_count_scale, output, lse, dout, dsk, dsv, dbias, dcount,
            n, leaves, h, d, topk, triton.next_power_of_2(topk), slots,
            triton.next_power_of_2(d), 32,
            d**-0.5, num_warps=4,
        )
        return (
            dq_exact + dq_summary, dk, dv, None, None, dsk, dsv, None,
            dbias.to(summary_bias.dtype), dcount.to(log_count_scale.dtype), None,
        )


def frontier_attention(q, k, v, order, indices, summary_k, summary_v,
                       lengths, summary_bias, log_count_scale, block_size):
    if q.shape[-1] not in (16, 32, 64, 128):
        raise ValueError("Frontier attention supports head dimensions 16/32/64/128")
    if order.dtype != torch.int32 or indices.dtype != torch.int32:
        raise ValueError("Frontier routing metadata must use int32")
    if summary_k.shape != summary_v.shape or summary_k.ndim != 5:
        raise ValueError("Summary K/V must have shape [B,H,leaves,slots,D]")
    if summary_k.shape[:2] != q.shape[:2] or summary_k.shape[-1] != q.shape[-1]:
        raise ValueError("Summary K/V batch, head and feature dimensions must match Q")
    if summary_k.shape[2] <= indices.shape[-1]:
        raise ValueError("Frontier requires at least one unselected summary leaf")
    if summary_k.shape[2] * summary_k.shape[3] > 512:
        raise ValueError("Frontier v1 supports at most 512 summary tokens")
    return _FrontierAttention.apply(
        q, k, v, order, indices, summary_k, summary_v, lengths,
        summary_bias, log_count_scale, block_size,
    )
