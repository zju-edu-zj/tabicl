"""ICL set retrieval: shared leaves, exact centroid or approximate tree routing.

All queries route independently per head, and only support rows supply K/V.
The hard router has no trainable parameters. Q/K still learn through attention.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .layers import GlobalLinearAttention

ICL_ATTENTION_TYPES = (
    "dense",
    "global",
    "global_full",
    "global_sparse_alternating",
    "global_sparse",
)


class SetRetrievalAttention(GlobalLinearAttention):
    """Exact selected-row softmax, optionally fused with an ELU+1 summary."""

    def __init__(
        self,
        *args,
        mode="fused_sparse",
        block_size=128,
        num_selected_blocks=4,
        query_chunk_size=64,
        router="centroid",
        router_beam=16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if mode not in ("sparse", "fused_sparse", "fused_full"):
            raise ValueError(f"Unknown retrieval mode: {mode}")
        if block_size < 16 or block_size & (block_size - 1):
            raise ValueError("block_size must be a power of two >= 16")
        if num_selected_blocks < 1 or query_chunk_size < 1:
            raise ValueError("selected blocks and query chunk size must be positive")
        if router not in ("centroid", "hierarchical"):
            raise ValueError("router must be centroid or hierarchical")
        if router_beam < 1 or (router == "hierarchical" and router_beam < num_selected_blocks):
            raise ValueError("router_beam must be >= num_selected_blocks")
        # The reference and fused kernels use zero attention dropout alike.
        if self.dropout:
            raise ValueError("ICL retrieval currently requires dropout=0")
        self.mode = mode
        self.block_size = block_size
        self.num_selected_blocks = num_selected_blocks
        self.query_chunk_size = query_chunk_size
        self.router = router
        self.router_beam = router_beam
        # A 32 MiB score tile amortizes GEMM/top-k dispatch at long contexts.
        # This is a workspace bound, not an attention or candidate budget.
        self.route_score_chunk_elements = 8 * 1024**2
        self.diagnostic_query_start = 0
        self.use_triton = True
        self.register_buffer("route_direction", F.normalize(torch.randn(self.head_dim), dim=0))
        if mode != "sparse":
            self.fusion_gate_weight = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
            self.fusion_gate_bias = nn.Parameter(torch.zeros(self.num_heads))

    @torch.no_grad()
    def route(self, q, k):
        """Return head-specific support ordering and selected leaf IDs."""
        if k.shape[-2] == 0:
            raise ValueError("ICL retrieval needs at least one support row")
        size, s = self.block_size, k.shape[-2]
        nk = (s + size - 1) // size
        budget = min(self.num_selected_blocks, nk)
        with torch.autocast(device_type=q.device.type, enabled=False):
            score = torch.einsum("bhsd,d->bhs", k.float(), self.route_direction.float())
            order = score.argsort(dim=-1, stable=True)
            lengths = (s - torch.arange(nk, device=k.device) * size).clamp(1, size)
            if k.is_cuda and self.use_triton:
                from ._routing_triton import block_centroids

                centroids = block_centroids(k, order, size)
            else:
                sorted_k = k.float().gather(2, order[..., None].expand_as(k))
                padded = F.pad(sorted_k, (0, 0, 0, nk * size - s))
                centroids = padded.reshape(*k.shape[:2], nk, size, k.shape[-1]).sum(3)
                centroids = centroids / lengths[None, None, :, None]
            if self.router == "hierarchical":
                selected = self._route_tree(q, centroids, lengths, budget)
            else:
                # Routing scores are discrete; positive scaling by sqrt(d)
                # cannot change top-k, so no temperature operation is needed.
                chunk = max(1, self.route_score_chunk_elements // (q.shape[0] * q.shape[1] * nk))
                selected = torch.cat(
                    [
                        (q[:, :, start : start + chunk].float() @ centroids.transpose(-1, -2)).topk(budget, -1).indices
                        for start in range(0, q.shape[-2], chunk)
                    ],
                    dim=2,
                )
        return order, selected

    def _route_tree(self, q, centroids, lengths, budget):
        nk = centroids.shape[-2]
        leaves = 1 << (nk - 1).bit_length()
        beam = min(self.router_beam, nk)
        counts = F.pad(lengths.float(), (0, leaves - nk))
        sums = F.pad(centroids * lengths[None, None, :, None], (0, 0, 0, leaves - nk))
        levels, count_levels = [], []
        while True:
            levels.append(sums / counts.clamp_min(1)[None, None, :, None])
            count_levels.append(counts)
            if counts.numel() == 1:
                break
            sums = sums.reshape(*sums.shape[:2], -1, 2, sums.shape[-1]).sum(3)
            counts = counts.reshape(-1, 2).sum(1)
        tree = torch.cat(levels[::-1], dim=2).contiguous()
        counts = torch.cat(count_levels[::-1])
        if q.is_cuda and self.use_triton and leaves > 1:
            from ._routing_triton import beam_search

            return beam_search(q, tree, counts, leaves, beam, budget)
        outputs = []
        # Portable reference. The CUDA path never materializes these gathers.
        for start in range(0, q.shape[-2], 256):
            queries = q[:, :, start : start + 256].float()
            nodes = torch.zeros((*queries.shape[:3], 1), device=q.device, dtype=torch.long)
            for depth in range((leaves - 1).bit_length()):
                children = (nodes[..., None] * 2 + 1 + torch.arange(2, device=q.device)).flatten(-2)
                centers = (
                    tree[:, :, None]
                    .expand(-1, -1, queries.shape[2], -1, -1)
                    .gather(3, children[..., None].expand(-1, -1, -1, -1, q.shape[-1]))
                )
                scores = (queries[..., None, :] * centers).sum(-1)
                scores = scores.masked_fill(counts[children] == 0, -float("inf"))
                keep = min(beam, 2 ** (depth + 1))
                if depth == (leaves - 1).bit_length() - 1:
                    keep = budget
                nodes = children.gather(-1, scores.topk(keep, -1).indices)
            outputs.append(nodes[..., :budget] - (leaves - 1))
        return torch.cat(outputs, dim=2)

    @staticmethod
    def _selected_attention(q, k, v, selected, size):
        b, h, n, d = q.shape
        pos = (selected[..., None] * size + torch.arange(size, device=q.device)).flatten(-2)
        valid = pos < k.shape[-2]
        index = pos.clamp_max(k.shape[-2] - 1)[..., None].expand(-1, -1, -1, -1, d)
        keys = k[:, :, None].expand(-1, -1, n, -1, -1).gather(3, index)
        values = v[:, :, None].expand(-1, -1, n, -1, -1).gather(3, index)
        return F.scaled_dot_product_attention(
            q.reshape(b * h * n, 1, d),
            keys.reshape(b * h * n, -1, d),
            values.reshape(b * h * n, -1, d),
            attn_mask=valid.reshape(b * h * n, 1, -1),
        ).reshape(b, h, n, d)

    def _sparse_attention(self, q, k, v):
        order, selected = self.route(q, k)
        index = order[..., None].expand_as(k)
        ks, vs = k.gather(2, index), v.gather(2, index)
        if (
            q.is_cuda
            and self.use_triton
            and q.shape[-1] in (16, 32, 64, 128)
            and self.num_selected_blocks * self.block_size <= 2048
        ):
            from ._sparse_triton import routed_attention

            out = routed_attention(q, ks, vs, selected, self.block_size)
        else:
            outputs = []
            for start in range(0, q.shape[-2], self.query_chunk_size):
                args = (
                    q[:, :, start : start + self.query_chunk_size],
                    ks,
                    vs,
                    selected[:, :, start : start + self.query_chunk_size],
                    self.block_size,
                )
                if torch.is_grad_enabled() and any(t.requires_grad for t in args[:3]):
                    outputs.append(checkpoint(self._selected_attention, *args, use_reentrant=False))
                else:
                    outputs.append(self._selected_attention(*args))
            out = torch.cat(outputs, dim=2)
        if self.diagnostics_enabled:
            self._diagnose(q, ks, vs, selected)
        return out

    @torch.no_grad()
    def _diagnose(self, q, k, v, selected=None):
        first = min(max(self.diagnostic_query_start, 0), q.shape[-2] - 1)
        sample = torch.linspace(
            first, q.shape[-2] - 1, min(self.diagnostic_queries, q.shape[-2] - first), device=q.device
        ).long()
        with torch.autocast(device_type=q.device.type, enabled=False):
            logits = q[:1, :, sample].float() @ k[:1].float().transpose(-1, -2) / self.head_dim**0.5
            if selected is None:
                mask = torch.ones_like(logits, dtype=torch.bool)
            else:
                ids = torch.arange(k.shape[-2], device=q.device) // self.block_size
                mask = (selected[:1, :, sample, :, None] == ids).any(-2)
            prob = logits.softmax(-1)
            sparse_prob = logits.masked_fill(~mask, -float("inf")).softmax(-1)
            exact = logits.topk(min(self.diagnostic_topk, k.shape[-2]), -1).indices
            dense_out, sparse_out = prob @ v[:1].float(), sparse_prob @ v[:1].float()
            self.last_diagnostics.update(
                {
                    "recall_at_k": mask.gather(-1, exact).float().mean(),
                    "retained_mass": (prob * mask).sum(-1).mean(),
                    "candidate_fraction": mask.float().mean(),
                    "visible_keys_min": mask.sum(-1).min().float(),
                    "visible_keys_max": mask.sum(-1).max().float(),
                    "dense_output_relative_error": (sparse_out - dense_out).square().mean().sqrt()
                    / dense_out.square().mean().sqrt().clamp_min(1e-6),
                    "nonfinite_fraction": (~torch.isfinite(logits)).float().mean(),
                }
            )
        self.diagnostics_enabled = False

    def _hybrid_attention(self, q, k, v):
        if not k.shape[-2]:
            raise ValueError("ICL retrieval needs at least one support row")
        diagnose = self.diagnostics_enabled
        if diagnose:
            self.last_diagnostics = {}
        global_heads = None
        if self.mode != "sparse":
            global_heads = self._global_attention(q, k, v)
        self.diagnostics_enabled = diagnose
        if self.mode == "fused_full" or k.shape[-2] <= self.block_size * self.num_selected_blocks:
            sparse_heads = F.scaled_dot_product_attention(q, k, v)
            if diagnose:
                self._diagnose(q, k, v)
        else:
            sparse_heads = self._sparse_attention(q, k, v)
        heads = sparse_heads
        if global_heads is not None:
            with torch.autocast(device_type=q.device.type, enabled=False):
                gate = torch.sigmoid(
                    torch.einsum("bhtd,hd->bht", q.float(), self.fusion_gate_weight.float())
                    + self.fusion_gate_bias.float()[None, :, None]
                ).to(q.dtype)[..., None]
            heads = gate * sparse_heads + (1 - gate) * global_heads
            if diagnose:
                self.last_diagnostics["fusion_sparse_gate_mean"] = gate.detach().float().mean()
        return self.out_proj(heads.transpose(-3, -2).contiguous().view(*q.shape[:-3], q.shape[-2], self.embed_dim))
