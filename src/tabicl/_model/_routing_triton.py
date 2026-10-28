"""CUDA routing primitives. Routing is discrete and has no autograd state."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["S", "KB", "KH", "KS"])
def _projection_scores(K, DIRECTION, OUT, S, H: tl.constexpr, D: tl.constexpr,
                       KB, KH, KS, KD: tl.constexpr, BD: tl.constexpr,
                       TILE: tl.constexpr):
    tile, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = tile * TILE + tl.arange(0, TILE)
    d = tl.arange(0, BD)
    x = tl.load(K + b * KB + h * KH + row[:, None] * KS + d[None, :] * KD,
                (row[:, None] < S) & (d[None, :] < D), other=0).to(tl.float32)
    direction = tl.load(DIRECTION + d, d < D, other=0).to(tl.float32)
    tl.store(OUT + (b * H + h) * S + row, tl.sum(x * direction[None, :], 1), row < S)


def projection_scores(k, direction):
    b, h, s, d = k.shape
    out = torch.empty((b, h, s), device=k.device, dtype=torch.float32)
    _projection_scores[(triton.cdiv(s, 128), h, b)](
        k, direction, out, s, h, d, *k.stride(), triton.next_power_of_2(d), 128, num_warps=4
    )
    return out


@triton.jit(do_not_specialize=["N", "NK", "QB", "QH", "QN"])
def _centroid_scores(Q, CENTROIDS, OUT, N, NK, H: tl.constexpr, D: tl.constexpr,
                     QB, QH, QN, QD: tl.constexpr, BL: tl.constexpr,
                     LEAF_TILES: tl.constexpr,
                     BD: tl.constexpr, BT: tl.constexpr):
    program, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    query_tile, leaf_tile = program // LEAF_TILES, program % LEAF_TILES
    row = query_tile * BT + tl.arange(0, BT)
    leaf = leaf_tile * BL + tl.arange(0, BL)
    d = tl.arange(0, BD)
    q = tl.load(Q + b * QB + h * QH + row[:, None] * QN + d[None, :] * QD,
                (row[:, None] < N) & (d[None, :] < D), other=0).to(tl.float32)
    centers = tl.load(CENTROIDS + ((b * H + h) * NK + leaf[:, None]) * D + d[None, :],
                      (leaf[:, None] < NK) & (d[None, :] < D), other=0)
    score = tl.dot(q, tl.trans(centers), input_precision="ieee")
    offsets = ((b * H + h) * N + row[:, None]) * NK + leaf[None, :]
    tl.store(OUT + offsets, score, (row[:, None] < N) & (leaf[None, :] < NK))


def centroid_scores(q, centroids):
    b, h, n, d = q.shape
    nk = centroids.shape[-2]
    out = torch.empty((b, h, n, nk), device=q.device, dtype=torch.float32)
    leaf_tiles = triton.cdiv(nk, 128)
    _centroid_scores[(triton.cdiv(n, 16) * leaf_tiles, h, b)](
        q, centroids, out, n, nk, h, d, *q.stride(), 128, leaf_tiles,
        triton.next_power_of_2(d), 16, num_warps=4
    )
    return out


@triton.jit(do_not_specialize=["S", "KB", "KH", "KS"])
def _content_pivots(K, ORDER, AXES, S, H: tl.constexpr, D: tl.constexpr,
                    CAP: tl.constexpr, WIDTH: tl.constexpr, GROUPS: tl.constexpr,
                    KB, KH, KS, KD: tl.constexpr,
                    BD: tl.constexpr):
    group, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t, d = tl.arange(0, 16), tl.arange(0, BD)
    length = tl.maximum(1, tl.minimum(WIDTH, S - group * WIDTH))
    pos = group * WIDTH + t * (length - 1) // 15
    ids = tl.minimum(tl.load(ORDER + (b * H + h) * CAP + pos), S - 1)
    x = tl.load(K + b * KB + h * KH + ids[:, None] * KS + d[None, :] * KD,
                d[None, :] < D, other=0).to(tl.float32)
    mean = tl.sum(x, 0) / 16
    far = tl.argmax(tl.sum((x - mean[None, :]) * (x - mean[None, :]), 1), 0)
    a = tl.sum(tl.where(t[:, None] == far, x, 0), 0)
    other = tl.argmax(tl.sum((x - a[None, :]) * (x - a[None, :]), 1), 0)
    z = tl.sum(tl.where(t[:, None] == other, x, 0), 0)
    tl.store(AXES + ((b * H + h) * GROUPS + group) * D + d, z - a, d < D)


@triton.jit(do_not_specialize=["S", "KB", "KH", "KS"])
def _content_scores(K, ORDER, AXES, OUT, S, H: tl.constexpr, D: tl.constexpr,
                    CAP: tl.constexpr, WIDTH: tl.constexpr, GROUPS: tl.constexpr,
                    KB, KH, KS, KD: tl.constexpr,
                    BD: tl.constexpr, TILE: tl.constexpr):
    tile, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    pos = tile * TILE + tl.arange(0, TILE)
    d = tl.arange(0, BD)
    ids = tl.load(ORDER + (b * H + h) * CAP + pos, pos < CAP, other=S)
    axis = tl.load(AXES + (((b * H + h) * GROUPS + pos // WIDTH)[:, None] * D + d[None, :]),
                   (pos[:, None] < CAP) & (d[None, :] < D), other=0)
    x = tl.load(K + b * KB + h * KH + ids[:, None] * KS + d[None, :] * KD,
                (ids[:, None] < S) & (d[None, :] < D), other=0).to(tl.float32)
    score = tl.sum(x * axis, 1)
    # Only padding may be +inf: radix argsort can otherwise interleave sentinel
    # IDs with valid NaN/inf rows. Nonfinite model scores are diagnostic-only.
    finite_max: tl.constexpr = 3.4028234663852886e38
    score = tl.where(score == score, tl.maximum(-finite_max, tl.minimum(score, finite_max)), 0.0)
    tl.store(OUT + (b * H + h) * CAP + pos, tl.where(ids < S, score, float("inf")), pos < CAP)


def content_split_scores(k, order, width):
    b, h, s, d = k.shape
    cap, groups = order.shape[-1], order.shape[-1] // width
    axes = torch.empty((b, h, groups, d), device=k.device, dtype=torch.float32)
    scores = torch.empty((b, h, cap), device=k.device, dtype=torch.float32)
    args = (s, h, d, cap, width, groups, *k.stride(), triton.next_power_of_2(d))
    _content_pivots[(groups, h, b)](k, order, axes, *args, num_warps=4)
    _content_scores[(triton.cdiv(cap, 128), h, b)](k, order, axes, scores, *args, 128, num_warps=4)
    return scores


@triton.jit(do_not_specialize=["S", "NK", "KB", "KH", "KS"])
def _centroids(
    K,
    ORDER,
    OUT,
    S,
    H: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    NK,
    KB,
    KH,
    KS,
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


@triton.jit(do_not_specialize=["N", "QB", "QH", "QN"])
def _beam_search(
    Q,
    TREE,
    COUNTS,
    OUT,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    DEPTH: tl.constexpr,
    BEAM: tl.constexpr,
    WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    QB,
    QH,
    QN,
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
        # Portable register gather: tl.gather is absent in Triton 3.0/3.1.
        # Beam widths are small; this also avoids a framework version pin.
        parent = tl.sum(tl.where(slot[:, None] // 2 == slot[None, :], nodes[None, :], 0), axis=1)
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
    out = torch.empty((b, h, n, topk), device=q.device, dtype=torch.int32)
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
