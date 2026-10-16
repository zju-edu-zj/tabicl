"""Balanced, support-only content partitioning (no autograd or learned router).

Each node chooses an approximate farthest pair from 16 content-ordered samples,
then bisects by their difference direction. Leaves are contiguous, at most B
rows, so the existing gathered softmax and its grouped backward are reusable.
The tree is rebuilt for each layer/head from that layer's current support K.
Scalar keys are ordered within each node; full K/V tensors are never reordered
during tree construction.
"""

import torch
from torch.nn import functional as F


@torch.no_grad()
def content_tree_order(k, initial_order, block_size, use_triton=True):
    b, h, s, d = k.shape
    leaves = (s + block_size - 1) // block_size
    capacity = block_size * (1 << (leaves - 1).bit_length())
    order = F.pad(initial_order, (0, capacity - s), value=s)
    width = capacity
    while width > block_size:
        groups = capacity // width
        if k.is_cuda and use_triton:
            from ._routing_triton import content_split_scores

            scores = content_split_scores(k, order, width)
        else:
            starts = torch.arange(groups, device=k.device) * width
            lengths = (s - starts).clamp(1, width)
            positions = starts[:, None] + (torch.arange(16, device=k.device) * (lengths[:, None] - 1) // 15)
            positions = positions.clamp_max(capacity - 1)
            ids = order[:, :, positions].clamp_max(s - 1)
            points = k.float().gather(2, ids.flatten(2)[..., None].expand(b, h, groups * 16, d))
            points = points.reshape(b, h, groups, 16, d)
            mean = points.mean(-2, keepdim=True)
            a_id = (points - mean).square().sum(-1).argmax(-1)
            a = points.gather(-2, a_id[..., None, None].expand(b, h, groups, 1, d))
            b_id = (points - a).square().sum(-1).argmax(-1)
            z = points.gather(-2, b_id[..., None, None].expand(b, h, groups, 1, d))
            direction = z - a
            keys = k.float().gather(2, order.clamp_max(s - 1)[..., None].expand(b, h, capacity, d))
            scores = (keys.reshape(b, h, groups, width, d) * direction).sum(-1).flatten(2)
            finite = torch.finfo(scores.dtype)
            scores.nan_to_num_(nan=0.0, posinf=finite.max, neginf=finite.min)
            scores.masked_fill_(order == s, float("inf"))
        # Two top-k median-selection implementations were slower than CUDA's
        # segmented radix ordering because complement compaction dominated.
        # Split scores are continuous; equal-score set semantics are explicitly
        # unspecified, so stable input-position tie breaking adds no guarantee.
        local = scores.reshape(b, h, groups, width).argsort(dim=-1, stable=False)
        order = order.reshape(b, h, groups, width).gather(-1, local).flatten(2)
        width //= 2
    return order[..., :s].contiguous()
