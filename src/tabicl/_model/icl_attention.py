"""ICL set retrieval: shared leaves, exact centroid or approximate tree routing.

All queries route independently per head, and only support rows supply K/V.
The hard router has no trainable parameters. Q/K still learn through attention.
"""

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ._dual_gate_v3 import DualPathGateV3
from .layers import GlobalLinearAttention
from .qk_norm import ICLQKNorm

ICL_ATTENTION_TYPES = (
    "dense",
    "global",
    "sparse",
    "global_full",
    "global_sparse_alternating",
    "global_sparse",
    "dual_path",
    "adaptive_frontier",
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
        partition="projection",
        fusion="legacy",
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
        if partition not in ("projection", "content_tree"):
            raise ValueError("partition must be projection or content_tree")
        if fusion not in ("legacy", "independent_v1"):
            raise ValueError("fusion must be legacy or independent_v1")
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
        self.partition = partition
        self.fusion = fusion
        # A 32 MiB route tile amortizes GEMM/top-k dispatch at long contexts.
        # Fast selection includes candidates/sort buffers in this bound.
        # This is a workspace bound, not an attention or candidate budget.
        self.route_score_chunk_elements = 8 * 1024**2
        # Keep a reference switch for backend validation and matched ablations.
        self.use_fast_centroid_topk = True
        self.diagnostic_query_start = 0
        self.use_triton = True
        self._route_record = None
        self._route_replay = None
        self.register_buffer("route_direction", F.normalize(torch.randn(self.head_dim), dim=0))
        if mode != "sparse":
            self.fusion_gate_weight = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
            self.fusion_gate_bias = nn.Parameter(torch.zeros(self.num_heads))
            if fusion == "independent_v1":
                # Effective projections are O_global=O and O_local=O+Delta.
                # Zero Delta preserves the legacy fused projection at step zero
                # while retaining the full independent two-matrix function class.
                self.local_out_delta = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
                nn.init.zeros_(self.local_out_delta.weight)
                self.fusion_gate_query_weight = nn.Parameter(
                    torch.zeros(self.num_heads, 2, self.head_dim)
                )
                self.fusion_route_weight = nn.Parameter(torch.zeros(self.num_heads, 2, 6))
                self.fusion_route_bias = nn.Parameter(torch.zeros(self.num_heads, 2))

    @torch.no_grad()
    def _prepare_route(self, k):
        if k.shape[-2] == 0:
            raise ValueError("ICL retrieval needs at least one support row")
        size, s = self.block_size, k.shape[-2]
        nk = (s + size - 1) // size
        with torch.autocast(device_type=k.device.type, enabled=False):
            if k.is_cuda and self.use_triton:
                from ._routing_triton import projection_scores

                score = projection_scores(k, self.route_direction)
            else:
                score = torch.einsum("bhsd,d->bhs", k.float(), self.route_direction.float())
            order = score.argsort(dim=-1, stable=True)
            if self.partition == "content_tree":
                from ._content_tree import content_tree_order

                order = content_tree_order(k, order, size, self.use_triton)
            lengths = (s - torch.arange(nk, device=k.device) * size).clamp(1, size)
            if k.is_cuda and self.use_triton:
                from ._routing_triton import block_centroids

                centroids = block_centroids(k, order, size)
            else:
                sorted_k = k.float().gather(2, order[..., None].expand_as(k))
                padded = F.pad(sorted_k, (0, 0, 0, nk * size - s))
                centroids = padded.reshape(*k.shape[:2], nk, size, k.shape[-1]).sum(3)
                centroids = centroids / lengths[None, None, :, None]
        return order, centroids, lengths

    def _prepare_support(self, k, v):
        prepared = super()._prepare_support(k, v) if self.mode != "sparse" else {}
        if self.mode != "fused_full" and k.shape[-2] > self.block_size * self.num_selected_blocks:
            prepared["route"] = self._prepare_route(k)
        return prepared

    @torch.no_grad()
    def route(self, q, k, prepared=None, return_quality=False):
        """Return head-specific support ordering and selected leaf IDs."""
        if self._route_replay is not None:
            return next(self._route_replay)
        order, centroids, lengths = self._prepare_route(k) if prepared is None else prepared
        nk = centroids.shape[-2]
        budget = min(self.num_selected_blocks, nk)
        with torch.autocast(device_type=q.device.type, enabled=False):
            quality = None
            if self.router == "hierarchical":
                selected = self._route_tree(q, centroids, lengths, budget)
                if return_quality:
                    _, quality = self._centroid_route_quality(
                        q, centroids, lengths, budget, k.shape[-2], selected
                    )
            elif (
                return_quality
                and q.is_cuda
                and self.use_triton
                and self.use_fast_centroid_topk
                and budget <= 8
                and nk <= 1024
            ):
                from ._routing_triton import centroid_topk_quality

                selected, quality = centroid_topk_quality(
                    q, centroids, lengths, budget, k.shape[-2], self.route_score_chunk_elements
                )
            elif return_quality:
                selected, quality = self._centroid_route_quality(
                    q, centroids, lengths, budget, k.shape[-2]
                )
            elif (q.is_cuda and self.use_triton and self.use_fast_centroid_topk
                  and budget <= 8 and nk <= 1024):
                from ._routing_triton import centroid_topk

                selected = centroid_topk(q, centroids, budget, self.route_score_chunk_elements)
            else:
                # Routing scores are discrete; positive scaling by sqrt(d)
                # cannot change top-k, so no temperature operation is needed.
                chunk = max(1, self.route_score_chunk_elements // (q.shape[0] * q.shape[1] * nk))
                pieces = []
                for start in range(0, q.shape[-2], chunk):
                    queries = q[:, :, start : start + chunk]
                    if q.is_cuda and self.use_triton:
                        from ._routing_triton import centroid_scores

                        scores = centroid_scores(queries, centroids)
                    else:
                        scores = queries.float() @ centroids.transpose(-1, -2)
                    pieces.append(scores.topk(budget, -1).indices)
                selected = torch.cat(pieces, dim=2)
        result = (order.to(torch.int32), selected.to(torch.int32), quality) if return_quality else (
            order.to(torch.int32), selected.to(torch.int32)
        )
        if self._route_record is not None:
            self._route_record.append(result)
        return result

    @torch.no_grad()
    def _centroid_route_quality(self, q, centroids, lengths, budget, support_size, selected=None):
        """Select leaves and expose calibrated, bounded-cost router statistics."""
        nk = centroids.shape[-2]
        chunk = max(1, self.route_score_chunk_elements // (q.shape[0] * q.shape[1] * nk))
        selected_parts, feature_parts = [], []
        for start in range(0, q.shape[-2], chunk):
            queries = q[:, :, start : start + chunk]
            if q.is_cuda and self.use_triton:
                from ._routing_triton import centroid_scores

                raw_scores = centroid_scores(queries, centroids)
            else:
                raw_scores = queries.float() @ centroids.transpose(-1, -2)
            raw_scores = torch.nan_to_num(raw_scores.float(), nan=0.0, posinf=1e4, neginf=-1e4)
            chosen = (
                raw_scores.topk(budget, -1).indices
                if selected is None
                else selected[:, :, start : start + chunk].long()
            )
            route_logits = raw_scores / math.sqrt(q.shape[-1]) + lengths.float().log()
            log_prob = route_logits.log_softmax(-1)
            prob = log_prob.exp()
            selected_mass = prob.gather(-1, chosen).sum(-1)
            candidate_fraction = lengths[chosen].sum(-1).float() / support_size
            mean = raw_scores.mean(-1)
            std = raw_scores.std(-1, unbiased=False).clamp_min(1e-6)
            boundary = raw_scores.topk(min(budget + 1, nk), -1).values
            top_score = ((boundary[..., 0] - mean) / std).clamp(-8, 8)
            if nk > budget:
                margin = ((boundary[..., budget - 1] - boundary[..., budget]) / std).clamp(0, 8)
            else:
                margin = torch.zeros_like(top_score)
            if nk > 1:
                entropy = (-(prob * log_prob).sum(-1) / math.log(nk)).clamp(0, 1)
            else:
                entropy = torch.zeros_like(top_score)
            log_n = torch.full_like(selected_mass, math.log1p(support_size) / math.log(65537))
            feature_parts.append(torch.stack(
                (log_n, candidate_fraction, selected_mass, top_score, margin, entropy), -1
            ))
            selected_parts.append(chosen.to(torch.int32))
        return torch.cat(selected_parts, 2), torch.cat(feature_parts, 2)

    @staticmethod
    def _full_route_quality(q, support_size):
        shape = (*q.shape[:-1], 6)
        quality = torch.zeros(shape, device=q.device, dtype=torch.float32)
        quality[..., 0] = math.log1p(support_size) / math.log(65537)
        quality[..., 1:3] = 1
        return quality

    def checkpoint_contexts(self):
        """Record discrete routing once and replay it during block checkpointing."""
        records = []
        diagnose = self.diagnostics_enabled

        @contextmanager
        def record():
            self._route_record = records
            try:
                yield
            finally:
                self._route_record = None

        @contextmanager
        def replay():
            self._route_replay = iter(records)
            self.diagnostics_enabled = diagnose
            try:
                yield
            finally:
                self._route_replay = None

        return record(), replay()

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
    def _selected_attention(q, k, v, order, selected, size):
        b, h, n, d = q.shape
        pos = (selected.long()[..., None] * size + torch.arange(size, device=q.device)).flatten(-2)
        valid = pos < k.shape[-2]
        sorted_pos = pos.clamp_max(k.shape[-2] - 1)
        original = order.long()[:, :, None].expand(-1, -1, n, -1).gather(3, sorted_pos)
        index = original[..., None].expand(-1, -1, -1, -1, d)
        keys = k[:, :, None].expand(-1, -1, n, -1, -1).gather(3, index)
        values = v[:, :, None].expand(-1, -1, n, -1, -1).gather(3, index)
        return F.scaled_dot_product_attention(
            q.reshape(b * h * n, 1, d),
            keys.reshape(b * h * n, -1, d),
            values.reshape(b * h * n, -1, d),
            attn_mask=valid.reshape(b * h * n, 1, -1),
        ).reshape(b, h, n, d)

    def _sparse_attention(self, q, k, v, prepared=None, return_quality=False):
        routed = (
            self.route(q, k, return_quality=return_quality)
            if prepared is None or "route" not in prepared
            else self.route(q, k, prepared["route"], return_quality=return_quality)
        )
        if return_quality:
            order, selected, route_quality = routed
        else:
            order, selected = routed
        visible = min(k.shape[-2], self.num_selected_blocks * self.block_size)
        attention_q = self.qk_norm.scale_query(q, visible) if self.qk_norm is not None else q
        if (
            q.is_cuda
            and self.use_triton
            and q.shape[-1] in (16, 32, 64, 128)
            and self.num_selected_blocks * self.block_size <= 2048
        ):
            from ._sparse_triton import routed_attention

            out = routed_attention(attention_q, k, v, order, selected, self.block_size)
        else:
            outputs = []
            for start in range(0, q.shape[-2], self.query_chunk_size):
                args = (
                    attention_q[:, :, start : start + self.query_chunk_size],
                    k,
                    v,
                    order,
                    selected[:, :, start : start + self.query_chunk_size],
                    self.block_size,
                )
                if torch.is_grad_enabled() and any(t.requires_grad for t in args[:3]):
                    outputs.append(checkpoint(self._selected_attention, *args, use_reentrant=False))
                else:
                    outputs.append(self._selected_attention(*args))
            out = torch.cat(outputs, dim=2)
        if self.diagnostics_enabled:
            index = order.long()[..., None].expand_as(k)
            ks, vs = k.gather(2, index), v.gather(2, index)
            self._diagnose(attention_q, ks, vs, selected)
        return (out, route_quality) if return_quality else out

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

    @torch.compiler.disable
    def _hybrid_attention(self, q, k, v, prepared=None):
        # Compile projections/FFN while the dynamic routed kernels stay opaque.
        if not k.shape[-2]:
            raise ValueError("ICL retrieval needs at least one support row")
        diagnose = self.diagnostics_enabled
        if diagnose:
            self.last_diagnostics = {}
        global_heads = None
        if self.mode != "sparse":
            global_heads = self._global_attention(q, k, v, prepared)
        self.diagnostics_enabled = diagnose
        if self.mode == "fused_full" or k.shape[-2] <= self.block_size * self.num_selected_blocks:
            attention_q = self.qk_norm.scale_query(q, k.shape[-2]) if self.qk_norm is not None else q
            sparse_heads = F.scaled_dot_product_attention(attention_q, k, v)
            if diagnose:
                self._diagnose(attention_q, k, v)
            route_quality = self._full_route_quality(q, k.shape[-2])
        else:
            if self.fusion == "independent_v1" and global_heads is not None:
                sparse_heads, route_quality = self._sparse_attention(
                    q, k, v, prepared, return_quality=True
                )
            else:
                sparse_heads = self._sparse_attention(q, k, v, prepared)
        heads = sparse_heads
        if global_heads is not None:
            if self.fusion == "independent_v1":
                with torch.autocast(device_type=q.device.type, enabled=False):
                    base_logit = (
                        torch.einsum("bhtd,hd->bht", q.float(), self.fusion_gate_weight.float())
                        + self.fusion_gate_bias.float()[None, :, None]
                    )
                    query_delta = torch.einsum(
                        "bhtd,hcd->bhtc", q.float(), self.fusion_gate_query_weight.float()
                    )
                    route_delta = (
                        torch.einsum(
                            "bhtf,hcf->bhtc", route_quality, self.fusion_route_weight.float()
                        )
                        + self.fusion_route_bias.float()[None, :, None]
                    )
                    base_gate = torch.sigmoid(base_logit)[..., None]
                    local_gate = torch.sigmoid(
                        base_logit + query_delta[..., 1] + route_delta[..., 1]
                    )[..., None]
                    global_gate = 1 - torch.sigmoid(
                        base_logit - query_delta[..., 0] - route_delta[..., 0]
                    )[..., None]
                if q.is_cuda and q.ndim == 4 and q.shape[-1] <= 128:
                    from ._fusion_triton import fused_gate

                    shared_flat = fused_gate(
                        q, sparse_heads, global_heads,
                        self.fusion_gate_weight, self.fusion_gate_bias,
                    )
                else:
                    shared_heads = (
                        global_heads.float()
                        + base_gate * (sparse_heads.float() - global_heads.float())
                    ).to(q.dtype)
                    shared_flat = shared_heads.transpose(-3, -2).contiguous().view(
                        *q.shape[:-3], q.shape[-2], self.embed_dim
                    )
                with torch.autocast(device_type=q.device.type, enabled=False):
                    # The base is the exact legacy execution path. Independent
                    # gates only add corrections relative to that function.
                    correction = (
                        (global_gate - (1 - base_gate)) * global_heads.float()
                        + (local_gate - base_gate) * sparse_heads.float()
                    ).to(q.dtype)
                    local_scaled = (sparse_heads.float() * local_gate).to(q.dtype)
                correction_flat = correction.transpose(-3, -2).contiguous().view(
                    *q.shape[:-3], q.shape[-2], self.embed_dim
                )
                shared_flat = shared_flat + correction_flat
                local_flat = local_scaled.transpose(-3, -2).contiguous().view_as(shared_flat)
                output = F.linear(shared_flat, self.out_proj.weight, self.out_proj.bias)
                output = output + F.linear(local_flat, self.local_out_delta.weight)
                if diagnose:
                    self.last_diagnostics.update({
                        "fusion_global_gate_mean": global_gate.detach().mean(),
                        "fusion_sparse_gate_mean": local_gate.detach().mean(),
                        "route_candidate_fraction": route_quality[..., 1].mean(),
                        "route_selected_mass_proxy": route_quality[..., 2].mean(),
                        "route_top_score_z": route_quality[..., 3].mean(),
                        "route_boundary_margin": route_quality[..., 4].mean(),
                        "route_entropy": route_quality[..., 5].mean(),
                    })
                return output
            if q.is_cuda and q.ndim == 4 and q.shape[-1] <= 128:
                from ._fusion_triton import fused_gate

                combined = fused_gate(
                    q, sparse_heads, global_heads, self.fusion_gate_weight, self.fusion_gate_bias
                )
                if diagnose:
                    with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
                        gate = torch.sigmoid(
                            torch.einsum("bhtd,hd->bht", q.float(), self.fusion_gate_weight.float())
                            + self.fusion_gate_bias.float()[None, :, None]
                        )
                        self.last_diagnostics["fusion_sparse_gate_mean"] = gate.mean()
                return self.out_proj(combined)
            with torch.autocast(device_type=q.device.type, enabled=False):
                gate = torch.sigmoid(
                    torch.einsum("bhtd,hd->bht", q.float(), self.fusion_gate_weight.float())
                    + self.fusion_gate_bias.float()[None, :, None]
                )[..., None]
                heads = (
                    global_heads.float() + gate * (sparse_heads.float() - global_heads.float())
                ).to(q.dtype)
            if diagnose:
                self.last_diagnostics["fusion_sparse_gate_mean"] = gate.detach().float().mean()
        combined = heads.transpose(-3, -2).contiguous().view(*q.shape[:-3], q.shape[-2], self.embed_dim)
        return self.out_proj(combined)


class DualPathRetrievalAttention(SetRetrievalAttention):
    """Independent global and routed-softmax experts with additive V3 fusion."""

    def __init__(self, *args, dual_gate="evidence_v3", **kwargs):
        kwargs.pop("mode", None)
        kwargs.pop("fusion", None)
        super().__init__(*args, mode="sparse", fusion="legacy", **kwargs)
        if self.partition != "content_tree":
            raise ValueError("dual_path requires partition=content_tree")
        if self.global_type != "elu":
            raise ValueError("dual_path currently requires global_type=elu")
        if self.qk_norm is None:
            raise ValueError("dual_path v3 requires qk_norm=True")
        self.global_in_proj_weight = nn.Parameter(
            torch.empty(3 * self.embed_dim, self.embed_dim)
        )
        self.global_in_proj_bias = nn.Parameter(torch.zeros(3 * self.embed_dim))
        nn.init.xavier_uniform_(self.global_in_proj_weight)
        self.global_qk_norm = (
            ICLQKNorm(
                self.head_dim,
                self.num_heads,
                trainable_length_scale=self.global_type != "elu",
            )
            if self.qk_norm is not None
            else None
        )
        # ``out_proj`` belongs exclusively to the routed exact-softmax path.
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.global_out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.global_out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.dual_gate = dual_gate
        self.gate_controller = DualPathGateV3(
            self.num_heads, self.head_dim, mode=dual_gate
        )
        self._branch_policy = "native"
        self._force_dense_sparse = False

    def zero_init_output(self):
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.global_out_proj.weight)

    def _project_global(self, query, key, value, rope):
        bshape = query.shape[:-2]
        tgt_len, src_len = query.shape[-2], key.shape[-2]
        q, k, v = F._in_projection_packed(
            query, key, value, self.global_in_proj_weight, self.global_in_proj_bias
        )
        q = q.view(*bshape, tgt_len, self.num_heads, self.head_dim).transpose(-3, -2)
        k = k.view(*bshape, src_len, self.num_heads, self.head_dim).transpose(-3, -2)
        v = v.view(*bshape, src_len, self.num_heads, self.head_dim).transpose(-3, -2)
        if rope is not None:
            q = rope.rotate_queries_or_keys(q)
            k = rope.rotate_queries_or_keys(k)
        if self.global_qk_norm is not None:
            q = self.global_qk_norm.normalize_q(q)
            k = self.global_qk_norm.normalize_k(k)
        return q, k, v

    def _project_dual(self, query, key, value, rope):
        """Project both experts with two packed GEMMs while retaining distinct parameters."""
        bshape = query.shape[:-2]
        tgt_len, src_len = query.shape[-2], key.shape[-2]
        q_weight = torch.cat((
            self.in_proj_weight[: self.embed_dim],
            self.global_in_proj_weight[: self.embed_dim],
        ))
        q_bias = torch.cat((
            self.in_proj_bias[: self.embed_dim],
            self.global_in_proj_bias[: self.embed_dim],
        ))
        sparse_q, global_q = F.linear(query, q_weight, q_bias).split(
            self.embed_dim, dim=-1
        )
        if key is value:
            kv_weight = torch.cat((
                self.in_proj_weight[self.embed_dim :],
                self.global_in_proj_weight[self.embed_dim :],
            ))
            kv_bias = torch.cat((
                self.in_proj_bias[self.embed_dim :],
                self.global_in_proj_bias[self.embed_dim :],
            ))
            sparse_k, sparse_v, global_k, global_v = F.linear(
                key, kv_weight, kv_bias
            ).split(self.embed_dim, dim=-1)
        else:
            k_weight = torch.cat((
                self.in_proj_weight[self.embed_dim : 2 * self.embed_dim],
                self.global_in_proj_weight[self.embed_dim : 2 * self.embed_dim],
            ))
            v_weight = torch.cat((
                self.in_proj_weight[2 * self.embed_dim :],
                self.global_in_proj_weight[2 * self.embed_dim :],
            ))
            k_bias = torch.cat((
                self.in_proj_bias[self.embed_dim : 2 * self.embed_dim],
                self.global_in_proj_bias[self.embed_dim : 2 * self.embed_dim],
            ))
            v_bias = torch.cat((
                self.in_proj_bias[2 * self.embed_dim :],
                self.global_in_proj_bias[2 * self.embed_dim :],
            ))
            sparse_k, global_k = F.linear(key, k_weight, k_bias).split(
                self.embed_dim, dim=-1
            )
            sparse_v, global_v = F.linear(value, v_weight, v_bias).split(
                self.embed_dim, dim=-1
            )

        def heads(x, length):
            return x.view(
                *bshape, length, self.num_heads, self.head_dim
            ).transpose(-3, -2)

        sparse_q, global_q = heads(sparse_q, tgt_len), heads(global_q, tgt_len)
        sparse_k, global_k = heads(sparse_k, src_len), heads(global_k, src_len)
        sparse_v, global_v = heads(sparse_v, src_len), heads(global_v, src_len)
        if rope is not None:
            sparse_q = rope.rotate_queries_or_keys(sparse_q)
            global_q = rope.rotate_queries_or_keys(global_q)
            sparse_k = rope.rotate_queries_or_keys(sparse_k)
            global_k = rope.rotate_queries_or_keys(global_k)
        if self.qk_norm is not None:
            sparse_q = self.qk_norm.normalize_q(sparse_q)
            sparse_k = self.qk_norm.normalize_k(sparse_k)
        if self.global_qk_norm is not None:
            global_q = self.global_qk_norm.normalize_q(global_q)
            global_k = self.global_qk_norm.normalize_k(global_k)
        return sparse_q, sparse_k, sparse_v, global_q, global_k, global_v

    def _project_cached_query(self, query, weight, bias, norm, rope):
        bshape, length = query.shape[:-2], query.shape[-2]
        q = F.linear(query, weight[: self.embed_dim], bias[: self.embed_dim])
        q = q.view(*bshape, length, self.num_heads, self.head_dim).transpose(-3, -2)
        if rope is not None:
            q = rope.rotate_queries_or_keys(q)
        return norm.normalize_q(q) if norm is not None else q

    def _project_cached_queries(self, query, rope):
        bshape, length = query.shape[:-2], query.shape[-2]
        weight = torch.cat((
            self.in_proj_weight[: self.embed_dim],
            self.global_in_proj_weight[: self.embed_dim],
        ))
        bias = torch.cat((
            self.in_proj_bias[: self.embed_dim],
            self.global_in_proj_bias[: self.embed_dim],
        ))
        sparse_q, global_q = F.linear(query, weight, bias).split(
            self.embed_dim, dim=-1
        )
        sparse_q = sparse_q.view(
            *bshape, length, self.num_heads, self.head_dim
        ).transpose(-3, -2)
        global_q = global_q.view(
            *bshape, length, self.num_heads, self.head_dim
        ).transpose(-3, -2)
        if rope is not None:
            sparse_q = rope.rotate_queries_or_keys(sparse_q)
            global_q = rope.rotate_queries_or_keys(global_q)
        if self.qk_norm is not None:
            sparse_q = self.qk_norm.normalize_q(sparse_q)
        if self.global_qk_norm is not None:
            global_q = self.global_qk_norm.normalize_q(global_q)
        return sparse_q, global_q

    def _prepare_dual_support(self, sparse_k, sparse_v, global_k, global_v):
        prepared = {
            "global": GlobalLinearAttention._prepare_support(self, global_k, global_v)
        }
        if sparse_k.shape[-2] > self.block_size * self.num_selected_blocks:
            prepared["route"] = self._prepare_route(sparse_k)
        return prepared

    @staticmethod
    def _gate_route_features(route_quality):
        return DualPathGateV3._base_features(route_quality)

    def forward(
        self,
        query,
        key=None,
        value=None,
        cached_kv=None,
        key_padding_mask=None,
        attn_mask=None,
        rope=None,
        need_kv=False,
    ):
        if key_padding_mask is not None or attn_mask is not None:
            raise NotImplementedError("Dual-path attention requires an explicit support prefix")
        if cached_kv is not None:
            if cached_kv.key.shape[-3] != 2 * self.num_heads:
                raise ValueError("Dual-path cache must contain sparse and global K/V heads")
            sparse_k, global_k = cached_kv.key.split(self.num_heads, dim=-3)
            sparse_v, global_v = cached_kv.value.split(self.num_heads, dim=-3)
            sparse_q, global_q = self._project_cached_queries(query, rope)
            prepared = None
            if (
                not self.training
                and not torch.is_grad_enabled()
                and not self.diagnostics_enabled
            ):
                signature = (
                    id(self), id(cached_kv.key), self._tensor_version(cached_kv.key),
                    id(cached_kv.value), self._tensor_version(cached_kv.value),
                    tuple(p._version for p in self.parameters()), self.block_size,
                    tuple(self._tensor_version(b) for b in self.buffers()),
                    self.partition, self.use_triton, self.global_type,
                )
                if cached_kv._derived.get("signature") != signature:
                    cached_kv._derived.clear()
                    cached_kv._derived.update(
                        signature=signature,
                        state=self._prepare_dual_support(
                            sparse_k, sparse_v, global_k, global_v
                        ),
                    )
                prepared = cached_kv._derived["state"]
            return self._dual_attention(
                sparse_q, sparse_k, sparse_v, global_q, global_k, global_v, prepared
            )

        key = query if key is None else key
        value = key if value is None else value
        (sparse_q, sparse_k, sparse_v,
         global_q, global_k, global_v) = self._project_dual(query, key, value, rope)
        output = self._dual_attention(
            sparse_q, sparse_k, sparse_v, global_q, global_k, global_v
        )
        if need_kv:
            return (
                output,
                torch.cat((sparse_k, global_k), dim=-3),
                torch.cat((sparse_v, global_v), dim=-3),
            )
        return output

    @torch.compiler.disable
    def _dual_attention(
        self, sparse_q, sparse_k, sparse_v, global_q, global_k, global_v,
        prepared=None,
    ):
        if not sparse_k.shape[-2]:
            raise ValueError("ICL retrieval needs at least one support row")
        diagnose = self.diagnostics_enabled
        if diagnose:
            self.last_diagnostics = {}
        global_prepared = None if prepared is None else prepared.get("global")
        global_heads = self._global_attention(
            global_q, global_k, global_v, global_prepared
        )
        self.diagnostics_enabled = diagnose
        if self._force_dense_sparse or sparse_k.shape[-2] <= (
            self.block_size * self.num_selected_blocks
        ):
            attention_q = (
                self.qk_norm.scale_query(sparse_q, sparse_k.shape[-2])
                if self.qk_norm is not None else sparse_q
            )
            sparse_heads = F.scaled_dot_product_attention(
                attention_q, sparse_k, sparse_v
            )
            route_quality = self._full_route_quality(sparse_q, sparse_k.shape[-2])
            if diagnose:
                self._diagnose(attention_q, sparse_k, sparse_v)
        else:
            route_prepared = None if prepared is None else prepared.get("route")
            sparse_heads, route_quality = self._sparse_attention(
                sparse_q, sparse_k, sparse_v,
                {"route": route_prepared} if route_prepared is not None else None,
                return_quality=True,
            )
        global_gate, sparse_gate, gate_diagnostics = self.gate_controller(
            global_q, sparse_q, route_quality, sparse_k.shape[-2]
        )
        with torch.autocast(device_type=sparse_q.device.type, enabled=False):
            global_scaled = global_heads.float() * global_gate
            sparse_scaled = sparse_heads.float() * sparse_gate
            if self._branch_policy == "global_only":
                sparse_scaled = torch.zeros_like(sparse_scaled)
            elif self._branch_policy == "sparse_only":
                global_scaled = torch.zeros_like(global_scaled)
            elif self._branch_policy != "native":
                raise ValueError(f"Unknown dual branch policy: {self._branch_policy!r}")
            global_scaled = global_scaled.to(sparse_q.dtype)
            sparse_scaled = sparse_scaled.to(sparse_q.dtype)
        global_flat = global_scaled.transpose(-3, -2).contiguous().view(
            *sparse_q.shape[:-3], sparse_q.shape[-2], self.embed_dim
        )
        sparse_flat = sparse_scaled.transpose(-3, -2).contiguous().view_as(global_flat)
        packed = torch.cat((global_flat, sparse_flat), dim=-1)
        packed_weight = torch.cat(
            (self.global_out_proj.weight, self.out_proj.weight), dim=1
        )
        output = F.linear(packed, packed_weight, self.out_proj.bias)
        if diagnose:
            global_contribution = F.linear(global_flat, self.global_out_proj.weight)
            sparse_contribution = F.linear(sparse_flat, self.out_proj.weight)
            diagnostics = {
                "fusion_global_gate_mean": global_gate.detach().mean(),
                "fusion_sparse_gate_mean": sparse_gate.detach().mean(),
                "fusion_gate_sum_mean": (global_gate + sparse_gate).detach().mean(),
                "global_branch_rms": global_heads.detach().float().square().mean().sqrt(),
                "sparse_branch_rms": sparse_heads.detach().float().square().mean().sqrt(),
                "global_contribution_rms": global_contribution.detach().float().square().mean().sqrt(),
                "sparse_contribution_rms": sparse_contribution.detach().float().square().mean().sqrt(),
                "fusion_global_gate_p10": torch.quantile(global_gate.detach().float(), 0.1),
                "fusion_global_gate_p90": torch.quantile(global_gate.detach().float(), 0.9),
                "fusion_sparse_gate_p10": torch.quantile(sparse_gate.detach().float(), 0.1),
                "fusion_sparse_gate_p90": torch.quantile(sparse_gate.detach().float(), 0.9),
                "fusion_gate_saturation_fraction": (
                    ((global_gate.detach() < 0.05) | (global_gate.detach() > 0.95)).float().mean()
                    + ((sparse_gate.detach() < 0.05) | (sparse_gate.detach() > 0.95)).float().mean()
                ) / 2,
                "route_candidate_fraction": route_quality[..., 1].mean(),
                "route_selected_mass_proxy": route_quality[..., 2].mean(),
                "route_top_score_z": route_quality[..., 3].mean(),
                "route_boundary_margin": route_quality[..., 4].mean(),
                "route_entropy": route_quality[..., 5].mean(),
                **gate_diagnostics,
            }
            first = min(
                max(self.diagnostic_query_start, 0), global_gate.shape[-2] - 1
            )
            for prefix, gates in (
                ("fusion_global_gate", global_gate),
                ("fusion_sparse_gate", sparse_gate),
            ):
                values_by_head = gates.detach().float()[:, :, first:, 0].transpose(0, 1)
                for head, values in enumerate(values_by_head):
                    values = values.reshape(-1)
                    diagnostics[f"head_{head}/{prefix}_mean"] = values.mean()
                    diagnostics[f"head_{head}/{prefix}_std"] = values.std(unbiased=False)
                    diagnostics[f"head_{head}/{prefix}_min"] = values.min()
                    diagnostics[f"head_{head}/{prefix}_max"] = values.max()
            self.last_diagnostics.update(diagnostics)
        return output


class AdaptiveResolutionAttention(SetRetrievalAttention):
    """Query-adaptive exact leaves plus summaries under one softmax."""

    def __init__(self, *args, summary_slots=1, **kwargs):
        if summary_slots != 1:
            raise ValueError("adaptive_frontier v1 requires summary_slots=1")
        kwargs.pop("mode", None)
        kwargs.pop("fusion", None)
        super().__init__(*args, mode="sparse", fusion="legacy", **kwargs)
        if self.partition != "content_tree":
            raise ValueError("adaptive_frontier requires partition=content_tree")
        self.summary_slots = summary_slots
        self.summary_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.register_buffer("summary_log_count_scale", torch.ones(self.num_heads))

    def _leaf_summaries(self, k, v, order, lengths):
        from ._block_pool import block_mean

        return (
            block_mean(k, order, self.block_size).unsqueeze(3),
            block_mean(v, order, self.block_size).unsqueeze(3),
        )

    def _prepare_support(self, k, v):
        if k.shape[-2] <= self.block_size * self.num_selected_blocks:
            return {}
        route = self._prepare_route(k)
        order, _, lengths = route
        summaries = self._leaf_summaries(k, v, order, lengths)
        return {"route": route, "summaries": summaries}

    def _frontier_reference(
        self, q, k, v, order, selected, summary_k, summary_v, lengths
    ):
        outputs = []
        b, h, _, d = q.shape
        leaves = summary_k.shape[2]
        for start in range(0, q.shape[-2], self.query_chunk_size):
            query = q[:, :, start : start + self.query_chunk_size]
            chosen = selected[:, :, start : start + self.query_chunk_size]
            n = query.shape[2]
            pos = (
                chosen.long()[..., None] * self.block_size
                + torch.arange(self.block_size, device=q.device)
            ).flatten(-2)
            valid_exact = pos < k.shape[-2]
            sorted_pos = pos.clamp_max(k.shape[-2] - 1)
            original = order.long()[:, :, None].expand(-1, -1, n, -1).gather(
                3, sorted_pos
            )
            exact_index = original[..., None].expand(-1, -1, -1, -1, d)
            exact_k = k[:, :, None].expand(-1, -1, n, -1, -1).gather(
                3, exact_index
            )
            exact_v = v[:, :, None].expand(-1, -1, n, -1, -1).gather(
                3, exact_index
            )
            slots = summary_k.shape[3]
            flat_k = summary_k.flatten(2, 3)
            flat_v = summary_v.flatten(2, 3)
            summary_leaf = torch.arange(leaves, device=q.device).repeat_interleave(slots)
            visible_summary = ~(chosen[..., None] == summary_leaf).any(-2)
            exact_logits = torch.einsum("bhnd,bhncd->bhnc", query.float(), exact_k.float())
            exact_logits = exact_logits / math.sqrt(d)
            exact_logits = exact_logits.masked_fill(~valid_exact, -float("inf"))
            summary_logits = torch.einsum("bhnd,bhpd->bhnp", query.float(), flat_k.float())
            summary_logits = summary_logits / math.sqrt(d)
            prior = (lengths.float().log() - math.log(slots)).repeat_interleave(slots)
            summary_logits = (
                summary_logits
                + self.summary_bias.float()[None, :, None, None]
                + self.summary_log_count_scale.float()[None, :, None, None]
                * prior[None, None, None]
            ).masked_fill(~visible_summary, -float("inf"))
            logits = torch.cat((exact_logits, summary_logits), dim=-1)
            values = torch.cat(
                (exact_v, flat_v[:, :, None].expand(-1, -1, n, -1, -1)), dim=-2
            )
            outputs.append(
                torch.einsum("bhnc,bhncd->bhnd", logits.softmax(-1).to(values.dtype), values)
            )
        return torch.cat(outputs, dim=2)

    @torch.no_grad()
    def _diagnose_frontier(
        self, q, k, v, order, selected, summary_k, summary_v, lengths
    ):
        first = min(max(self.diagnostic_query_start, 0), q.shape[-2] - 1)
        sample = torch.linspace(
            first, q.shape[-2] - 1,
            min(self.diagnostic_queries, q.shape[-2] - first),
            device=q.device,
        ).long()
        query = q[:1, :, sample]
        chosen = selected[:1, :, sample]
        frontier = self._frontier_reference(
            query, k[:1], v[:1], order[:1], chosen,
            summary_k[:1], summary_v[:1], lengths,
        )
        dense = F.scaled_dot_product_attention(query, k[:1], v[:1])
        d = q.shape[-1]
        positions = (
            chosen.long()[..., None] * self.block_size
            + torch.arange(self.block_size, device=q.device)
        ).flatten(-2)
        valid = positions < k.shape[-2]
        sorted_pos = positions.clamp_max(k.shape[-2] - 1)
        original = order[:1, :, None].long().expand(-1, -1, query.shape[2], -1).gather(
            3, sorted_pos
        )
        exact_k = k[:1, :, None].expand(-1, -1, query.shape[2], -1, -1).gather(
            3, original[..., None].expand(-1, -1, -1, -1, d)
        )
        exact_logits = torch.einsum(
            "bhnd,bhncd->bhnc", query.float(), exact_k.float()
        ) / math.sqrt(d)
        exact_lse = torch.logsumexp(
            exact_logits.masked_fill(~valid, -float("inf")), dim=-1
        )
        flat_summary = summary_k[:1].flatten(2, 3)
        summary_logits = torch.einsum(
            "bhnd,bhpd->bhnp", query.float(), flat_summary.float()
        ) / math.sqrt(d)
        summary_leaf = torch.arange(summary_k.shape[2], device=q.device)
        visible = ~(chosen[..., None] == summary_leaf).any(-2)
        summary_logits = (
            summary_logits
            + self.summary_bias.float()[None, :, None, None]
            + lengths.float().log()[None, None, None]
        ).masked_fill(~visible, -float("inf"))
        summary_lse = torch.logsumexp(summary_logits, dim=-1)
        total_lse = torch.logaddexp(exact_lse, summary_lse)
        exact_mass = torch.exp(exact_lse - total_lse)
        exact_only_error = self.last_diagnostics.pop(
            "dense_output_relative_error", torch.tensor(float("nan"), device=q.device)
        )
        self.last_diagnostics.update({
            "frontier_exact_only_relative_error": exact_only_error,
            "dense_output_relative_error": (
                (frontier.float() - dense.float()).square().mean().sqrt()
                / dense.float().square().mean().sqrt().clamp_min(1e-6)
            ),
            "frontier_exact_softmax_mass": exact_mass.mean(),
            "frontier_summary_softmax_mass": (1 - exact_mass).mean(),
        })

    @torch.compiler.disable
    def _hybrid_attention(self, q, k, v, prepared=None):
        if not k.shape[-2]:
            raise ValueError("ICL retrieval needs at least one support row")
        diagnose = self.diagnostics_enabled
        if diagnose:
            self.last_diagnostics = {}
        leaves = (k.shape[-2] + self.block_size - 1) // self.block_size
        attention_q = self.qk_norm.scale_query(q, k.shape[-2]) if self.qk_norm is not None else q
        if leaves <= self.num_selected_blocks:
            heads = F.scaled_dot_product_attention(attention_q, k, v)
            if diagnose:
                self._diagnose(attention_q, k, v)
        else:
            route_prepared = None if prepared is None else prepared["route"]
            order, selected = self.route(attention_q, k, route_prepared)
            if prepared is None:
                lengths = (
                    k.shape[-2]
                    - torch.arange(leaves, device=k.device) * self.block_size
                ).clamp(1, self.block_size)
                summary_k, summary_v = self._leaf_summaries(k, v, order, lengths)
            else:
                lengths = route_prepared[2]
                summary_k, summary_v = prepared["summaries"]
            if (
                q.is_cuda and self.use_triton and q.shape[-1] in (16, 32, 64, 128)
                and self.block_size % 32 == 0
                and leaves * self.summary_slots <= 512
            ):
                from ._frontier_triton import frontier_attention

                heads = frontier_attention(
                    attention_q, k, v, order, selected, summary_k, summary_v,
                    lengths, self.summary_bias, self.summary_log_count_scale,
                    self.block_size,
                )
            else:
                heads = self._frontier_reference(
                    attention_q, k, v, order, selected,
                    summary_k, summary_v, lengths,
                )
            if diagnose:
                self._diagnose(attention_q, k.gather(2, order.long()[..., None].expand_as(k)),
                               v.gather(2, order.long()[..., None].expand_as(v)), selected)
                self._diagnose_frontier(
                    attention_q, k, v, order, selected,
                    summary_k, summary_v, lengths,
                )
                self.last_diagnostics.update({
                    "frontier_exact_fraction": lengths[selected.long()].sum(-1).float().mean()
                    / k.shape[-2],
                    "frontier_summary_tokens": torch.tensor(
                        (leaves - self.num_selected_blocks) * self.summary_slots,
                        device=q.device,
                        dtype=torch.float32,
                    ),
                    "frontier_candidate_tokens": torch.tensor(
                        self.num_selected_blocks * self.block_size
                        + (leaves - self.num_selected_blocks) * self.summary_slots,
                        device=q.device,
                        dtype=torch.float32,
                    ),
                    "frontier_log_count_scale": self.summary_log_count_scale.detach().mean(),
                })
        combined = heads.transpose(-3, -2).contiguous().view(
            *q.shape[:-3], q.shape[-2], self.embed_dim
        )
        return self.out_proj(combined)
