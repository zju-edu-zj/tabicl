"""Leaf-grouped routed attention with FP32 softmax and edge reductions.

Queries sharing a support leaf form a matrix tile. The tile uses tensor-core
GEMMs, then writes one partial per original routing edge. A logsumexp merge
recovers softmax over all selected leaves, including a short final leaf.
FP32 GEMMs use three-term TF32 (not the lower-accuracy single TF32 product).
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["E", "N", "NK"])
def _count(I, COUNTS, E, N, NC: tl.constexpr, NK, T: tl.constexpr):
    e = tl.program_id(0) * T + tl.arange(0, T)
    leaf = tl.load(I + e, e < E, other=0)
    key = e // (N * NC) * NK + leaf
    tl.atomic_add(COUNTS + key, 1, e < E, sem="relaxed")


@triton.jit(do_not_specialize=["E", "N", "NK"])
def _scatter(I, STARTS, CURSOR, REVERSE, E, N, NC: tl.constexpr, NK, T: tl.constexpr):
    e = tl.program_id(0) * T + tl.arange(0, T)
    leaf = tl.load(I + e, e < E, other=0)
    key = e // (N * NC) * NK + leaf
    slot = tl.atomic_add(CURSOR + key, 1, e < E, sem="relaxed")
    start = tl.load(STARTS + key, e < E, other=0)
    tl.store(REVERSE + start + slot, e, e < E)


def invert_routes(indices, support, block_size):
    b, h, n, nc = indices.shape
    nk = triton.cdiv(support, block_size)
    keys, edges = b * h * nk, indices.numel()
    counts = torch.zeros(keys, device=indices.device, dtype=torch.int32)
    _count[(triton.cdiv(edges, 256),)](indices, counts, edges, n, nc, nk, 256)
    padded = (counts + 31) // 32 * 32
    starts = torch.empty(keys + 1, device=indices.device, dtype=torch.int32)
    starts[0] = 0
    torch.cumsum(padded, 0, dtype=torch.int32, out=starts[1:])
    # Upper bound avoids a GPU->CPU size read/synchronization. Padding slots
    # are sentinels, and unused tasks exit before loading query/key tensors.
    reverse = torch.full((edges + keys * 31,), -1, device=indices.device, dtype=torch.int32)
    cursor = torch.zeros_like(counts)
    _scatter[(triton.cdiv(edges, 256),)](indices, starts, cursor, reverse, edges, n, nc, nk, 256)
    return starts, reverse


@triton.jit(do_not_specialize=["N", "S", "KEYS", "NK"])
def _leaf_tiles(Q, K, V, ORDER, STARTS, REVERSE, PART, PLSE,
                O, LSE, DO, N, S, KEYS,
                D: tl.constexpr, NC: tl.constexpr, NK,
                BS: tl.constexpr, SCALE: tl.constexpr, BACKWARD: tl.constexpr,
                BT: tl.constexpr, BQ: tl.constexpr = 32):
    begin = tl.program_id(0) * BQ
    # Upper-bound binary search handles empty leaves and repeated starts.
    lo, hi = 0, KEYS
    while lo < hi:
        mid = (lo + hi) // 2
        start = tl.load(STARTS + mid)
        right = start <= begin
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    leafkey = lo - 1
    total = tl.load(STARTS + KEYS)
    if begin < total:
        bh = leafkey // NK
        leaf = leafkey % NK
        offset = begin + tl.arange(0, BQ)
        edge = tl.load(REVERSE + offset)
        valid = edge >= 0
        qi = edge // NC % N
        d = tl.arange(0, D)
        qo = (bh * N + qi[:, None]) * D + d[None, :]
        q = tl.load(Q + qo, valid[:, None], other=0)
        acc = tl.full((BQ, D), 0, tl.float32)
        m = tl.full((BQ,), -float("inf"), tl.float32)
        z = tl.full((BQ,), 0, tl.float32)
        if BACKWARD:
            lse = tl.load(LSE + bh * N + qi, valid, other=0)
            grad = tl.load(DO + qo, valid[:, None], other=0).to(q.dtype)
            out = tl.load(O + qo, valid[:, None], other=0).to(tl.float32)
            delta = tl.sum(out * grad.to(tl.float32), 1)
        for start in range(0, BS, BT):
            r = leaf * BS + start + tl.arange(0, BT)
            original = tl.load(ORDER + bh * S + r, r < S, other=0)
            ko = (bh * S + original[:, None]) * D + d[None, :]
            k = tl.load(K + ko, (r < S)[:, None], other=0).to(q.dtype)
            v = tl.load(V + ko, (r < S)[:, None], other=0).to(q.dtype)
            score = tl.dot(q, tl.trans(k), input_precision="tf32x3") * SCALE
            score = tl.where((r < S)[None, :], score, -float("inf"))
            if BACKWARD:
                p = tl.exp(score - lse[:, None])
                dp = tl.dot(grad, tl.trans(v), input_precision="tf32x3")
                ds = p * (dp - delta[:, None]) * SCALE
                acc += tl.dot(ds.to(q.dtype), k, input_precision="tf32x3")
            else:
                new_m = tl.maximum(m, tl.max(score, 1))
                # A tail leaf can have entire empty tiles after its first row.
                ex = tl.exp(score - new_m[:, None])
                alpha = tl.exp(m - new_m)
                acc = acc * alpha[:, None] + tl.dot(ex.to(v.dtype), v, input_precision="tf32x3")
                z = z * alpha + tl.sum(ex, 1)
                m = new_m
        if BACKWARD:
            part = acc
        else:
            part = acc / z[:, None]
            tl.store(PLSE + edge, m + tl.log(z), valid)
        tl.store(PART + edge[:, None] * D + d[None, :], part, valid[:, None])


@triton.jit
def _merge(PART, PLSE, OUT, LSE, ROWS, D: tl.constexpr,
           NC: tl.constexpr, BC: tl.constexpr, BACKWARD: tl.constexpr):
    row = tl.program_id(0)
    c, d = tl.arange(0, BC), tl.arange(0, D)
    part = tl.load(PART + (row * NC + c[:, None]) * D + d[None, :], c[:, None] < NC, other=0)
    if BACKWARD:
        out = tl.sum(part, 0)
    else:
        lse = tl.load(PLSE + row * NC + c, c < NC, other=-float("inf"))
        m = tl.max(lse, 0)
        w = tl.exp(lse - m)
        z = tl.sum(w, 0)
        out = tl.sum(part * (w / z)[:, None], 0)
        tl.store(LSE + row, m + tl.log(z))
    tl.store(OUT + row * D + d, out)


def grouped_attention(q, k, v, order, indices, block_size, metadata=None, backward=None):
    b, h, n, d = q.shape
    s, nc = k.shape[2], indices.shape[-1]
    nk = triton.cdiv(s, block_size)
    if metadata is None:
        metadata = invert_routes(indices, s, block_size)
    starts, reverse = metadata
    partial = torch.empty((*indices.shape, d), device=q.device, dtype=torch.float32)
    partial_lse = torch.empty(indices.shape, device=q.device, dtype=torch.float32)
    output = torch.empty_like(q)
    lse = torch.empty((b, h, n), device=q.device, dtype=torch.float32)
    out, full_lse, grad = (q, lse, q) if backward is None else backward
    _leaf_tiles[(triton.cdiv(reverse.numel(), 32),)](
        q, k, v, order, starts, reverse, partial, partial_lse,
        out, full_lse, grad, n, s, b * h * nk, d, nc, nk,
        block_size, d**-0.5, backward is not None,
        32 if d >= 128 or block_size % 64 else 64,
        num_warps=4, num_stages=1,
    )
    _merge[(b * h * n,)](partial, partial_lse, output, lse, b * h * n,
                        d, nc, triton.next_power_of_2(nc), backward is not None, num_warps=4)
    return output, lse, metadata
