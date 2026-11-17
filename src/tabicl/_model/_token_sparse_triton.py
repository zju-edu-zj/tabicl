"""Direct attention over shared per-query token indices (CUDA only)."""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "S"])
def _forward(Q, K, V, I, O, LSE, N, S, H: tl.constexpr, D: tl.constexpr,
             KSEL: tl.constexpr, KR: tl.constexpr, SCALE: tl.constexpr):
    t, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r = tl.arange(0, KR)
    d = tl.arange(0, D)
    valid = r < KSEL
    pos = tl.load(I + (b * N + t) * KSEL + r, valid, other=0)
    off = ((b * H + h) * S + pos[:, None]) * D + d[None, :]
    key = tl.load(K + off, valid[:, None], other=0).to(tl.float32)
    value = tl.load(V + off, valid[:, None], other=0).to(tl.float32)
    qoff = ((b * H + h) * N + t) * D + d
    q = tl.load(Q + qoff).to(tl.float32)
    logits = tl.sum(key * q[None, :], axis=1) * SCALE
    logits = tl.where(valid, logits, -float("inf"))
    maximum = tl.max(logits, axis=0)
    probability = tl.exp(logits - maximum)
    denominator = tl.sum(probability, axis=0)
    probability /= denominator
    output = tl.sum(probability[:, None] * value, axis=0)
    tl.store(O + qoff, output)
    tl.store(LSE + (b * H + h) * N + t, maximum + tl.log(denominator))


@triton.jit(do_not_specialize=["N", "S"])
def _backward_q(Q, K, V, I, O, LSE, DO, DQ, N, S, H: tl.constexpr,
                D: tl.constexpr, KSEL: tl.constexpr, KR: tl.constexpr,
                SCALE: tl.constexpr):
    t, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r = tl.arange(0, KR)
    d = tl.arange(0, D)
    valid = r < KSEL
    pos = tl.load(I + (b * N + t) * KSEL + r, valid, other=0)
    off = ((b * H + h) * S + pos[:, None]) * D + d[None, :]
    key = tl.load(K + off, valid[:, None], other=0).to(tl.float32)
    value = tl.load(V + off, valid[:, None], other=0).to(tl.float32)
    qoff = ((b * H + h) * N + t) * D + d
    q = tl.load(Q + qoff).to(tl.float32)
    dout = tl.load(DO + qoff).to(tl.float32)
    output = tl.load(O + qoff).to(tl.float32)
    lse = tl.load(LSE + (b * H + h) * N + t)
    score = tl.sum(key * q[None, :], axis=1) * SCALE
    probability = tl.where(valid, tl.exp(score - lse), 0.0)
    dp = tl.sum(value * dout[None, :], axis=1)
    delta = tl.sum(output * dout, axis=0)
    ds = probability * (dp - delta) * SCALE
    tl.store(DQ + qoff, tl.sum(ds[:, None] * key, axis=0))


@triton.jit(do_not_specialize=["EDGES", "N", "S"])
def _edge_count(I, COUNTS, EDGES, N, S, KSEL: tl.constexpr, TILE: tl.constexpr):
    edge = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = edge < EDGES
    token = tl.load(I + edge, valid, other=0)
    batch = edge // (N * KSEL)
    tl.atomic_add(COUNTS + batch * S + token, 1, valid, sem="relaxed")


@triton.jit(do_not_specialize=["EDGES", "N", "S"])
def _edge_scatter(I, STARTS, CURSOR, REVERSE, EDGES, N, S,
                  KSEL: tl.constexpr, TILE: tl.constexpr):
    edge = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = edge < EDGES
    token = tl.load(I + edge, valid, other=0)
    batch = edge // (N * KSEL)
    key = batch * S + token
    slot = tl.atomic_add(CURSOR + key, 1, valid, sem="relaxed")
    start = tl.load(STARTS + batch * (S + 1) + token, valid, other=0)
    query = edge // KSEL % N
    tl.store(REVERSE + batch * N * KSEL + start + slot, query, valid)


@triton.jit(do_not_specialize=["N", "S"])
def _backward_kv(Q, K, V, O, LSE, DO, STARTS, REVERSE, DK, DV, N, S,
                 H: tl.constexpr, D: tl.constexpr, KSEL: tl.constexpr,
                 SCALE: tl.constexpr, BQ: tl.constexpr):
    token, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, D)
    koff = ((b * H + h) * S + token) * D + d
    key = tl.load(K + koff).to(tl.float32)
    value = tl.load(V + koff).to(tl.float32)
    start = tl.load(STARTS + b * (S + 1) + token)
    end = tl.load(STARTS + b * (S + 1) + token + 1)
    dk = tl.zeros((D,), tl.float32)
    dv = tl.zeros((D,), tl.float32)
    for begin in range(start, end, BQ):
        offsets = begin + tl.arange(0, BQ)
        valid = offsets < end
        query = tl.load(REVERSE + b * N * KSEL + offsets, valid, other=0)
        qoff = ((b * H + h) * N + query[:, None]) * D + d[None, :]
        q = tl.load(Q + qoff, valid[:, None], other=0).to(tl.float32)
        dout = tl.load(DO + qoff, valid[:, None], other=0).to(tl.float32)
        output = tl.load(O + qoff, valid[:, None], other=0).to(tl.float32)
        lse = tl.load(LSE + (b * H + h) * N + query, valid, other=0)
        score = tl.sum(q * key[None, :], axis=1) * SCALE
        probability = tl.where(valid, tl.exp(score - lse), 0.0)
        dp = tl.sum(dout * value[None, :], axis=1)
        delta = tl.sum(dout * output, axis=1)
        ds = probability * (dp - delta) * SCALE
        dk += tl.sum(ds[:, None] * q, axis=0)
        dv += tl.sum(probability[:, None] * dout, axis=0)
    tl.store(DK + koff, dk)
    tl.store(DV + koff, dv)


class _TokenSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices):
        b, h, n, d = q.shape
        support, selected = k.shape[2], indices.shape[-1]
        q, k, v, indices = q.contiguous(), k.contiguous(), v.contiguous(), indices.contiguous()
        output = torch.empty_like(q)
        lse = torch.empty((b, h, n), device=q.device, dtype=torch.float32)
        rounded = triton.next_power_of_2(selected)
        scale = d**-0.5
        _forward[(n, h, b)](
            q, k, v, indices, output, lse, n, support, h, d, selected, rounded, scale,
            num_warps=8,
        )
        ctx.save_for_backward(q, k, v, indices, output, lse)
        ctx.args = (selected, rounded, scale)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        q, k, v, indices, output, lse = ctx.saved_tensors
        selected, rounded, scale = ctx.args
        b, h, n, d = q.shape
        support = k.shape[2]
        dout = dout.contiguous()
        dq = torch.empty_like(q)
        _backward_q[(n, h, b)](
            q, k, v, indices, output, lse, dout, dq, n, support, h, d,
            selected, rounded, scale, num_warps=8,
        )

        edges = b * n * selected
        counts = torch.zeros((b, support), device=q.device, dtype=torch.int32)
        _edge_count[(triton.cdiv(edges, 256),)](
            indices, counts, edges, n, support, selected, 256, num_warps=4,
        )
        cumulative = counts.cumsum(-1, dtype=torch.int32)
        starts = torch.cat((torch.zeros((b, 1), device=q.device, dtype=torch.int32), cumulative), -1)
        cursor = torch.zeros_like(counts)
        reverse = torch.empty((b, n * selected), device=q.device, dtype=torch.int32)
        _edge_scatter[(triton.cdiv(edges, 256),)](
            indices, starts, cursor, reverse, edges, n, support, selected, 256, num_warps=4,
        )
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        _backward_kv[(support, h, b)](
            q, k, v, output, lse, dout, starts, reverse, dk, dv, n, support,
            h, d, selected, scale, 32, num_warps=4,
        )
        return dq, dk, dv, None


def token_sparse_attention(q, k, v, indices):
    if q.shape[-1] not in (16, 32, 64, 128):
        raise ValueError("Token sparse Triton attention supports head dimensions 16/32/64/128")
    if indices.ndim != 3 or indices.shape[:2] != (q.shape[0], q.shape[2]):
        raise ValueError("Expected shared token indices [B,N,K]")
    if indices.dtype != torch.int32:
        raise ValueError("Token routing metadata must use int32")
    if indices.shape[-1] > 2048:
        raise ValueError("Token sparse Triton attention supports at most 2048 selected tokens")
    return _TokenSparseAttention.apply(q, k, v, indices)
