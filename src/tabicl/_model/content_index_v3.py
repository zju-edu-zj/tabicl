"""Content index for 32-row logical leaves stored in 128-row pages."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from ._block_pool import block_mean
from ._content_tree import content_tree_order


@dataclass(frozen=True)
class DualPathLayout:
    logical_leaf_rows: int = 32
    storage_page_rows: int = 128
    selected_logical_leaves: int = 32
    search_beam_leaves: int = 64
    proposal_leaves: int = 64
    exhaustive_support_limit: int = 8192
    query_tile_rows: int = 256

    def __post_init__(self):
        if self.logical_leaf_rows != 32:
            raise ValueError("dual-path v3 currently requires 32-row logical leaves")
        if self.storage_page_rows % self.logical_leaf_rows:
            raise ValueError("storage pages must contain whole logical leaves")
        if self.selected_logical_leaves < 1:
            raise ValueError("selected logical leaves must be positive")
        if self.search_beam_leaves < 1 or self.proposal_leaves < 1:
            raise ValueError("search beam and proposal leaf counts must be positive")
        if self.query_tile_rows < 1:
            raise ValueError("query tile rows must be positive")

    @property
    def leaves_per_page(self) -> int:
        return self.storage_page_rows // self.logical_leaf_rows

    @property
    def selected_rows(self) -> int:
        return self.logical_leaf_rows * self.selected_logical_leaves


@dataclass
class RoutingResult:
    logical_leaf_ids: Tensor
    candidate_rows: Tensor
    boundary_margin: Tensor
    proposal_entropy: Tensor
    visited_nodes: Tensor


@dataclass
class ContentIndexV3:
    order: Tensor
    leaf_representatives: Tensor
    leaf_rep_counts: Tensor
    leaf_lengths: Tensor
    tree_representatives: Tensor
    tree_rep_counts: Tensor
    leaf_capacity: int
    num_leaves: int
    support_size: int
    layout: DualPathLayout

    @property
    def num_pages(self) -> int:
        return (self.num_leaves + self.layout.leaves_per_page - 1) // self.layout.leaves_per_page

    @staticmethod
    def _lengths(total: int, unit: int, device) -> Tensor:
        count = (total + unit - 1) // unit
        return (total - torch.arange(count, device=device) * unit).clamp(1, unit)

    @classmethod
    @torch.no_grad()
    def build(
        cls, k: Tensor, layout: DualPathLayout, use_triton: bool = True,
        initial_order: Tensor | None = None,
    ):
        if k.ndim != 4 or k.shape[-2] < 1:
            raise ValueError("content index expects K shaped [B,H,S,D] with S > 0")
        b, h, support, d = k.shape
        if initial_order is None:
            direction = F.normalize(
                torch.arange(1, d + 1, device=k.device, dtype=torch.float32), dim=0
            )
            initial_order = torch.einsum(
                "bhsd,d->bhs", k.float(), direction
            ).argsort(dim=-1, stable=True)
        order = content_tree_order(
            k, initial_order, layout.logical_leaf_rows, use_triton=use_triton
        ).to(torch.int32)

        micro_rows = layout.logical_leaf_rows // 2
        micro = block_mean(k, order.to(torch.int32), micro_rows).float()
        micro_lengths = cls._lengths(support, micro_rows, k.device)
        leaves = (support + layout.logical_leaf_rows - 1) // layout.logical_leaf_rows
        padded_micro = F.pad(micro, (0, 0, 0, leaves * 2 - micro.shape[-2]))
        leaf_representatives = padded_micro.reshape(b, h, leaves, 2, d)
        leaf_rep_counts = F.pad(
            micro_lengths.float(), (0, leaves * 2 - micro_lengths.numel())
        ).reshape(leaves, 2)
        leaf_lengths = cls._lengths(support, layout.logical_leaf_rows, k.device)

        capacity = 1 << (leaves - 1).bit_length()
        leaf_sums = (
            leaf_representatives * leaf_rep_counts[None, None, :, :, None]
        ).sum(3)
        padded_leaf_sums = F.pad(leaf_sums, (0, 0, 0, capacity - leaves))
        padded_leaf_lengths = F.pad(
            leaf_lengths, (0, capacity - leaves)
        )

        nodes = 2 * capacity - 1
        sums = torch.zeros((b, h, nodes, d), device=k.device, dtype=torch.float32)
        counts = torch.zeros(nodes, device=k.device, dtype=torch.float32)
        leaf_offset = capacity - 1
        sums[:, :, leaf_offset:] = padded_leaf_sums
        counts[leaf_offset:] = padded_leaf_lengths
        representatives = torch.zeros(
            (b, h, nodes, 2, d), device=k.device, dtype=torch.float32
        )
        rep_counts = torch.zeros((nodes, 2), device=k.device, dtype=torch.float32)
        # Build each tree level in parallel. Python work now scales with tree
        # depth, not with the number of physical pages or internal nodes.
        level_start = leaf_offset
        level_width = capacity
        while level_width > 1:
            parent_start = (level_start - 1) // 2
            parent_width = level_width // 2
            left = sums[:, :, level_start : level_start + level_width : 2]
            right = sums[:, :, level_start + 1 : level_start + level_width : 2]
            left_count = counts[level_start : level_start + level_width : 2]
            right_count = counts[level_start + 1 : level_start + level_width : 2]
            parent_slice = slice(parent_start, parent_start + parent_width)
            sums[:, :, parent_slice] = left + right
            counts[parent_slice] = left_count + right_count
            representatives[:, :, parent_slice, 0] = (
                left / left_count.clamp_min(1)[None, None, :, None]
            )
            representatives[:, :, parent_slice, 1] = (
                right / right_count.clamp_min(1)[None, None, :, None]
            )
            rep_counts[parent_slice, 0] = left_count
            rep_counts[parent_slice, 1] = right_count
            level_start, level_width = parent_start, parent_width

        representatives[:, :, leaf_offset : leaf_offset + leaves] = leaf_representatives
        rep_counts[leaf_offset : leaf_offset + leaves] = leaf_rep_counts

        return cls(
            order=order,
            leaf_representatives=leaf_representatives,
            leaf_rep_counts=leaf_rep_counts,
            leaf_lengths=leaf_lengths,
            tree_representatives=representatives,
            tree_rep_counts=rep_counts,
            leaf_capacity=capacity,
            num_leaves=leaves,
            support_size=support,
            layout=layout,
        )

    @staticmethod
    def _scores(q: Tensor, representatives: Tensor, counts: Tensor) -> Tensor:
        logits = torch.einsum("bhtd,bhurd->bhtur", q.float(), representatives)
        logits = logits / math.sqrt(q.shape[-1])
        logits = logits + counts.clamp_min(1).log()[None, None, None]
        logits = logits.masked_fill(counts[None, None, None] == 0, -float("inf"))
        return torch.logsumexp(logits, dim=-1)

    def _leaf_scores(self, q: Tensor, leaf_ids: Tensor | None = None) -> Tensor:
        if leaf_ids is None:
            return self._scores(q, self.leaf_representatives, self.leaf_rep_counts)
        b, h, t, count = leaf_ids.shape
        d = q.shape[-1]
        reps = self.leaf_representatives[:, :, None].expand(-1, -1, t, -1, -1, -1)
        reps = reps.gather(
            3, leaf_ids[..., None, None].expand(b, h, t, count, 2, d)
        )
        counts = self.leaf_rep_counts[leaf_ids]
        logits = (q[..., None, None, :].float() * reps).sum(-1) / math.sqrt(d)
        logits = logits + counts.clamp_min(1).log()
        logits = logits.masked_fill(counts == 0, -float("inf"))
        return torch.logsumexp(logits, -1)

    def _leaf_proposals(self, q: Tensor) -> tuple[Tensor, int]:
        b, h, t, d = q.shape
        nodes = torch.zeros((b, h, t, 1), device=q.device, dtype=torch.long)
        depth = self.leaf_capacity.bit_length() - 1
        visited = 0
        for _ in range(depth):
            children = (nodes[..., None] * 2 + 1 + torch.arange(2, device=q.device)).flatten(-2)
            count = children.shape[-1]
            reps = self.tree_representatives[:, :, None].expand(-1, -1, t, -1, -1, -1)
            reps = reps.gather(
                3, children[..., None, None].expand(b, h, t, count, 2, d)
            )
            rep_counts = self.tree_rep_counts[children]
            logits = (q[..., None, None, :].float() * reps).sum(-1) / math.sqrt(d)
            logits = logits + rep_counts.clamp_min(1).log()
            logits = logits.masked_fill(rep_counts == 0, -float("inf"))
            scores = torch.logsumexp(logits, -1)
            keep = min(self.layout.search_beam_leaves, count)
            nodes = children.gather(-1, scores.topk(keep, -1).indices)
            visited += count
        leaves = nodes - (self.leaf_capacity - 1)
        valid = leaves < self.num_leaves
        leaves = leaves.masked_fill(~valid, 0)
        keep = min(self.layout.proposal_leaves, leaves.shape[-1])
        return leaves[..., :keep], valid[..., :keep], visited

    @torch.no_grad()
    def search(self, q: Tensor) -> RoutingResult:
        leaves = self.leaf_lengths.numel()
        budget = min(self.layout.selected_logical_leaves, leaves)
        if self.support_size <= self.layout.exhaustive_support_limit:
            proposal_ids = None
            scores = self._leaf_scores(q)
            visited = leaves
        else:
            proposal_ids, valid, visited = self._leaf_proposals(q)
            scores = self._leaf_scores(q, proposal_ids).masked_fill(~valid, -float("inf"))

        top = scores.topk(min(budget + 1, scores.shape[-1]), -1)
        selected_pos = top.indices[..., :budget]
        selected = selected_pos if proposal_ids is None else proposal_ids.gather(-1, selected_pos)
        selected_lengths = self.leaf_lengths[selected]
        candidate_rows = selected_lengths.sum(-1)

        std = scores.masked_fill(~torch.isfinite(scores), 0).std(-1, unbiased=False).clamp_min(1e-6)
        if scores.shape[-1] > budget:
            margin = ((top.values[..., budget - 1] - top.values[..., budget]) / std).clamp(0, 8) / 8
        else:
            margin = torch.zeros_like(candidate_rows, dtype=torch.float32)
        log_prob = scores.log_softmax(-1)
        prob = log_prob.exp()
        finite_count = torch.isfinite(scores).sum(-1).clamp_min(2)
        entropy = (-(prob * log_prob.nan_to_num()).sum(-1) / finite_count.float().log()).clamp(0, 1)
        return RoutingResult(
            logical_leaf_ids=selected.to(torch.int32),
            candidate_rows=candidate_rows.to(torch.float32),
            boundary_margin=margin,
            proposal_entropy=entropy,
            visited_nodes=torch.full_like(candidate_rows, visited, dtype=torch.int32),
        )
