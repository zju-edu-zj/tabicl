"""CUDA routing primitives. Routing is discrete and has no autograd state."""

import torch
import triton
import triton.language as tl


@triton.jit
def _centroids(
    K,
    ORDER,
    OUT,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    NK: tl.constexpr,
    KB: tl.constexpr,
    KH: tl.constexpr,
    KS: tl.constexpr,
    KD: tl.constexpr,
    BD: tl.constexpr,
):
    block, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r = block * BS + tl.arange(0, BS)
    d = tl.arange(0, BD)
    pos = tl.load(ORDER + (b * H + h) * S + r, r < S, other=0)
    x = tl.load(
        K + b * KB + h * KH + pos[:, None] * KS + d[None, :] * KD, (r[:, None] < S) & (d[None, :] < D), other=0
    ).to(tl.float32)
    mean = tl.sum(x, axis=0) / tl.minimum(BS, S - block * BS)
    tl.store(OUT + ((b * H + h) * NK + block) * D + d, mean, d < D)


@triton.jit
def _beam_search(
    Q,
    TREE,
    COUNTS,
    OUT,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    DEPTH: tl.constexpr,
    BEAM: tl.constexpr,
    WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    QB: tl.constexpr,
    QH: tl.constexpr,
    QN: tl.constexpr,
    QD: tl.constexpr,
    BD: tl.constexpr,
):
    n, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.arange(0, 2 * WIDTH)
    d = tl.arange(0, BD)
    q = tl.load(Q + b * QB + h * QH + n * QN + d * QD, d < D, other=0).to(tl.float32)
    # Heap index zero is the root. Only its first beam slot is live initially.
    nodes = tl.full((2 * WIDTH,), 0, tl.int32)
    for level in tl.static_range(DEPTH):
        parent = tl.gather(nodes, slot // 2, axis=0)
        children = parent * 2 + 1 + slot % 2
        live = slot < 2 * min(BEAM, 2**level)
        count = tl.load(COUNTS + children, live, other=0)
        centers = tl.load(
            TREE + (((b * H + h) * (2 * L - 1) + children[:, None]) * D + d[None, :]),
            live[:, None] & (d[None, :] < D),
            other=0,
        )
        scores = tl.sum(centers * q[None, :], axis=1)
        scores = tl.where(live & (count > 0), scores, -float("inf"))
        # Preserve the complete FP32 score and its node ID through sorting.
        # Integer ordering maps negative IEEE floats by complementing bits.
        bits = scores.to(tl.int32, bitcast=True)
        rank = tl.where(bits < 0, ~bits, bits ^ -2147483648).to(tl.uint32)
        packed = (rank.to(tl.uint64) << 32) | children.to(tl.uint64)
        packed = tl.sort(packed, descending=True)
        nodes = (packed & 0xFFFFFFFF).to(tl.int32)
    tl.store(OUT + ((b * H + h) * N + n) * TOPK + slot, nodes - (L - 1), slot < TOPK)


def block_centroids(k, order, block_size):
    b, h, s, d = k.shape
    nk = triton.cdiv(s, block_size)
    out = torch.empty((b, h, nk, d), device=k.device, dtype=torch.float32)
    _centroids[(nk, h, b)](k, order, out, s, h, d, block_size, nk, *k.stride(), triton.next_power_of_2(d), num_warps=4)
    return out


def beam_search(q, tree, counts, leaves, beam, topk):
    b, h, n, d = q.shape
    out = torch.empty((b, h, n, topk), device=q.device, dtype=torch.long)
    _beam_search[(n, h, b)](
        q,
        tree,
        counts,
        out,
        n,
        h,
        d,
        leaves,
        (leaves - 1).bit_length(),
        beam,
        triton.next_power_of_2(beam),
        topk,
        *q.stride(),
        triton.next_power_of_2(d),
        num_warps=4,
    )
    return out
