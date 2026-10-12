"""Experimental, set-equivariant ICL attention alternatives.

These operators receive K/V from the support prefix via MultiheadAttentionBlock.
Dense, positive-kernel, induced, and Flex paths are available for comparison.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import GlobalLinearAttention, SupportAttention
from .kv_cache import KVCacheEntry

_compiled_flex = None


class PerformerLinearAttention(GlobalLinearAttention):
    """Bidirectional FAVOR+ style linear attention for support sets.

    The positive random feature map approximates the softmax kernel instead of
    using ELU+1 as a generic feature map.  Features are fixed per head so the
    support summary is cacheable and the operator remains permutation equivariant.
    """

    def __init__(self, *args, num_features: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        if num_features < 1:
            raise ValueError("num_features must be positive")
        self.num_features = num_features
        # Independent orthogonal blocks with Gaussian radial lengths. Repeating
        # the same basis would add no approximation capacity when m > head_dim.
        blocks = []
        for _ in range((num_features + self.head_dim - 1) // self.head_dim):
            basis, triangular = torch.linalg.qr(
                torch.randn(self.num_heads, self.head_dim, self.head_dim))
            signs = triangular.diagonal(dim1=-2, dim2=-1).sign()
            blocks.append((basis * signs[:, None, :]).transpose(-1, -2))
        directions = torch.cat(blocks, dim=1)[:, :num_features]
        radii = torch.randn(self.num_heads, num_features, self.head_dim).norm(dim=-1)
        self.register_buffer("feature_matrix", directions * radii[..., None])

    def _performer_features(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = x.float() / (self.head_dim**0.25)
            projected = torch.einsum("...htd,hmd->...htm", z, self.feature_matrix.float())
            # Clamping is a deliberate finite-precision guard for long tables;
            # the denominator keeps the normalized attention well behaved.
            projected = projected.clamp(-6.0, 6.0)
            norm = (-0.5 * z.square().sum(dim=-1, keepdim=True)).clamp(-10.0, 0.0)
            return torch.exp(projected + norm) / (self.num_features**0.5)

    def _feature_map(self, x):
        return self._performer_features(x)


class InducedSetAttention(SupportAttention):
    """Two softmax cross-attentions: learned slots read support; rows read slots.

Complexity is O((support_len + query_len) * num_slots) per head.  Unlike
ELU+1 linear attention, the slots can learn distinct content-dependent summaries.
The KV cache contains the compressed slots, not all support rows.
"""

    def __init__(self, *args, num_slots: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        if num_slots < 1:
            raise ValueError("num_slots must be positive")
        self.num_slots = num_slots
        self.slots = nn.Parameter(torch.empty(self.num_heads, num_slots, self.head_dim))
        nn.init.normal_(self.slots, std=self.head_dim ** -0.5)

    def forward(self, query, key=None, value=None, cached_kv=None,
                key_padding_mask=None, attn_mask=None, rope=None, need_kv=False):
        if key_padding_mask is not None or attn_mask is not None:
            raise NotImplementedError("Induced ICL attention requires the explicit support prefix")
        if cached_kv is None:
            key = query if key is None else key
            value = key if value is None else value
            q, k, v = self._project(query, key, value, rope)
            slots = self.slots.to(q.dtype).unsqueeze(0).expand(q.shape[0], -1, -1, -1)
            slot_q = self.ssmax_layer(slots, k.shape[-2]) if self.ssmax_layer is not None else slots
            memory = F.scaled_dot_product_attention(
                slot_q, k, v, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            memory = cached_kv.key
            q_weight = self.in_proj_weight[:self.embed_dim]
            q_bias = self.in_proj_bias[:self.embed_dim]
            q = F.linear(query, q_weight, q_bias)
            q = q.view(*query.shape[:-2], query.shape[-2], self.num_heads, self.head_dim).transpose(-3, -2)
            if rope is not None:
                q = rope.rotate_queries_or_keys(q)
        q = self.ssmax_layer(q, memory.shape[-2]) if self.ssmax_layer is not None else q
        out = F.scaled_dot_product_attention(
            q, memory, memory, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(-3, -2).contiguous().view(*query.shape[:-2], query.shape[-2], self.embed_dim)
        out = self.out_proj(out)
        if need_kv and cached_kv is None:
            return out, memory, memory
        return out


class FlexBlockSparseAttention(SupportAttention):
    """Scalar content routing with exact softmax over a per-query KV window.

    Every query chooses its window using only its own content and the support.
    Sorted query tiles visit the union of their windows; mask_mod enforces each
    row's own window. This preserves batching/cache semantics even at tile edges.
    Scalar routing is an experimental heuristic, not a nearest-neighbor guarantee.
    """

    def __init__(self, *args, block_size=128, num_kv_blocks=4,
                 qk_norm=False, scale_mode="auto", **kwargs):
        super().__init__(*args, **kwargs)
        if block_size < 128 or block_size % 128 or num_kv_blocks < 1:
            raise ValueError("block_size must be divisible by 128; num_kv_blocks must be positive")
        if scale_mode == "auto":
            scale_mode = "ssmax" if self.ssmax_layer is not None else "none"
        if scale_mode not in ("none", "ssmax", "bounded"):
            raise ValueError("Flex scale_mode must be none, ssmax or bounded")
        if scale_mode == "ssmax" and self.ssmax_layer is None:
            raise ValueError("Flex ssmax mode requires icl_ssmax=True and ssmax_type != none")
        if scale_mode != "ssmax":
            self.ssmax_layer = None
        self.block_size = block_size
        self.num_kv_blocks = num_kv_blocks
        self.qk_norm = qk_norm
        self.scale_mode = scale_mode
        self.register_buffer("route_direction", F.normalize(torch.randn(self.head_dim), dim=0))
        if scale_mode == "bounded":
            # Limix-style positive bounded coefficients: w in (0,.5), beta in (0,5).
            # Initialize beta=1 and w~.009; unlike an unconstrained MLP this
            # cannot reverse similarity rankings or grow coefficients unbounded.
            self.length_weight = nn.Parameter(torch.full((self.num_heads,), -2.0))
            self.base_scale = nn.Parameter(torch.full((self.num_heads,), -0.69314718056))
        self.diagnostics_enabled = False
        self.diagnostic_queries = 16
        self.diagnostic_topk = 32
        self.diagnostic_query_start = 0
        self.last_diagnostics = {}

    def _prepare_qk(self, q, k):
        if self.qk_norm:
            # FP32 statistics with no learned affine gain; temperature carries
            # the learnable magnitude. Leave V unchanged.
            q = (q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + 1e-5)).to(q.dtype)
            k = (k.float() * torch.rsqrt(k.float().square().mean(-1, keepdim=True) + 1e-5)).to(k.dtype)
        return q, k

    @torch.no_grad()
    def _route(self, q, k):
        size = self.block_size
        nk = (k.shape[-2] + size - 1) // size
        budget = min(self.num_kv_blocks, nk)
        with torch.autocast(device_type=q.device.type, enabled=False):
            q_score = torch.einsum("btd,d->bt", q.float().mean(1), self.route_direction.float())
            k_score = torch.einsum("bsd,d->bs", k.float().mean(1), self.route_direction.float())
        k_order = k_score.argsort(dim=-1, stable=True)
        k_sorted = k_score.gather(1, k_order)
        centers = torch.searchsorted(k_sorted[:, ::size].contiguous(), q_score.contiguous())
        starts = (centers - budget // 2).clamp(0, nk - budget)
        return k_order, starts, budget

    def _scale_queries(self, q, visible):
        if self.scale_mode == "ssmax":
            return self.ssmax_layer(q, visible)
        if self.scale_mode == "bounded":
            logn = visible.float().clamp_min(1).log()[:, None, :, None]
            w = (0.25 * (self.length_weight.float().tanh() + 1))[None, :, None, None]
            beta = (2.5 * (self.base_scale.float().tanh() + 1))[None, :, None, None]
            return (q.float() * beta * (1 + w * logn)).to(q.dtype)
        return q

    @torch.no_grad()
    def _diagnose(self, q, k, v, starts, budget):
        # Inspect one dataset, selected queries, ALL support keys and each head.
        # No training-size Q x K matrix and no Q x candidates x topk expansion.
        first = self.diagnostic_query_start
        if not 0 <= first < q.shape[-2]:
            first = 0
        count = min(self.diagnostic_queries, q.shape[-2] - first)
        sample = torch.linspace(first, q.shape[-2] - 1, count, device=q.device).long()
        with torch.autocast(device_type=q.device.type, enabled=False):
            qs, ks = q[:1, :, sample].float(), k[:1].float()
            logits = qs @ ks.transpose(-1, -2) / self.head_dim**0.5
            start = starts[:1, sample] * self.block_size
            pos = torch.arange(k.shape[-2], device=q.device)
            mask = (pos[None, None, :] >= start[..., None]) & (
                pos[None, None, :] < (start + budget * self.block_size)[..., None])
            mask = mask[:, None].expand_as(logits)
            topk = min(self.diagnostic_topk, k.shape[-2])
            exact = logits.topk(topk, -1).indices
            recall = mask.gather(-1, exact).float().mean(-1)
            dense_prob = logits.softmax(-1)
            mass = (dense_prob * mask).sum(-1)
            sparse_prob = logits.masked_fill(~mask, float("-inf")).softmax(-1)
            entropy = -(sparse_prob * sparse_prob.clamp_min(1e-30).log()).sum(-1)
            visible = mask.sum(-1)
            normalized_entropy = entropy / visible.clamp_min(2).float().log()
            stats = {
                "recall_at_k": recall.mean(), "recall_worst_head": recall.mean((0, 2)).min(),
                "retained_mass": mass.mean(), "retained_mass_min": mass.min(),
                "candidate_fraction": mask.float().mean(), "visible_keys_min": visible.min().float(),
                "visible_keys_max": visible.max().float(), "logit_std": logits.std(unbiased=False),
                "logit_abs_max": logits.abs().max(), "logit_abs_max_selected": logits[mask].abs().max(),
                "entropy_normalized": normalized_entropy.mean(),
                "sparse_top1_prob": sparse_prob.max(-1).values.mean(),
                "q_rms": qs.square().mean().sqrt(), "k_rms": ks.square().mean().sqrt(),
                "nonfinite_fraction": (~torch.isfinite(logits)).float().mean(),
                "recall_k": torch.tensor(float(topk), device=q.device),
            }
            if v is not None:
                dense_out = dense_prob @ v[:1].float()
                sparse_out = sparse_prob @ v[:1].float()
                stats["dense_output_relative_error"] = (
                    (sparse_out - dense_out).square().mean().sqrt() /
                    dense_out.square().mean().sqrt().clamp_min(1e-6))
            self.last_diagnostics = stats
            if self.scale_mode == "bounded":
                w = 0.25 * (self.length_weight.float().tanh() + 1)
                beta = 2.5 * (self.base_scale.float().tanh() + 1)
                scales = beta[None, :, None] * (1 + w[None, :, None] * visible.float().log())
                stats.update({"temperature_min": scales.min(), "temperature_max": scales.max(),
                              "length_weight_max": w.max(), "base_scale_max": beta.max()})
            return stats

    @torch.no_grad()
    def routing_recall(self, q, k, topk=None):
        """Bounded-memory per-head recall using the actual normalized/scaled logits."""
        q, k = self._prepare_qk(q, k)
        order, starts, budget = self._route(q, k)
        k = k.gather(2, order[:, None, :, None].expand_as(k))
        visible = ((starts + budget) * self.block_size).clamp_max(k.shape[-2]) - starts * self.block_size
        q = self._scale_queries(q, visible)
        previous = self.diagnostic_topk
        if topk is not None:
            if topk < 1:
                raise ValueError("diagnostic topk must be positive")
            self.diagnostic_topk = topk
        try:
            stats = self._diagnose(q, k, None, starts, budget)
            return {name: float(value) for name, value in stats.items()}
        finally:
            self.diagnostic_topk = previous

    def _hybrid_attention(self, q, k, v):
        try:
            from torch.nn.attention.flex_attention import BlockMask, flex_attention
        except ImportError as exc:
            raise RuntimeError("flex_sparse requires PyTorch >= 2.5 with FlexAttention") from exc
        if self.training and self.dropout:
            raise NotImplementedError("FlexAttention has no attention dropout; set dropout=0")
        bsz, nheads, tgt_len, head_dim = q.shape
        src_len = k.shape[-2]
        if min(tgt_len, src_len) < 1:
            raise ValueError("Flex attention requires nonempty query and support")
        q, k = self._prepare_qk(q, k)
        k_order, starts, budget = self._route(q, k)
        k = k.gather(2, k_order[:, None, :, None].expand_as(k))
        v = v.gather(2, k_order[:, None, :, None].expand_as(v))
        visible = ((starts + budget) * self.block_size).clamp_max(src_len) - starts * self.block_size
        q = self._scale_queries(q, visible)
        if self.diagnostics_enabled:
            self._diagnose(q.detach(), k.detach(), v.detach(), starts, budget)
            # Avoid duplicate diagnostics during gradient checkpoint recomputation.
            self.diagnostics_enabled = False

        # Packing changes execution order only: each row retains its own window.
        order = starts.argsort(dim=-1, stable=True)
        undo = order.argsort(-1)
        q = q.gather(2, order[:, None, :, None].expand_as(q))
        starts = starts.gather(1, order)
        size = self.block_size
        nq, nk = (tgt_len + size - 1) // size, (src_len + size - 1) // size
        q_pad, k_pad = nq * size - tgt_len, nk * size - src_len
        if q_pad:
            q = F.pad(q, (0, 0, 0, q_pad))
            starts = F.pad(starts, (0, q_pad), mode="replicate")
        if k_pad:
            k = F.pad(k, (0, 0, 0, k_pad))
            v = F.pad(v, (0, 0, 0, k_pad))
        tiles = starts.view(bsz, nq, size)
        lower = tiles.amin(-1)
        upper = tiles.amax(-1) + budget
        counts = (upper - lower).to(torch.int32)[:, None]
        # from_kv_blocks needs full nk capacity to form backward transposition.
        offsets = torch.arange(nk, device=q.device)
        ids = (lower[..., None] + offsets).clamp_max(nk - 1).to(torch.int32)[:, None]

        def row_window(b, h, qi, ki):
            lo = starts[b, qi] * size
            return (ki < src_len) & (ki >= lo) & (ki < lo + budget * size)

        block_mask = BlockMask.from_kv_blocks(
            counts, ids, BLOCK_SIZE=size, mask_mod=row_window,
            seq_lengths=(nq * size, nk * size),
        )
        if self.last_diagnostics and not self.diagnostics_enabled:
            # Updated only for the diagnostic forward, not on ordinary steps.
            if "kernel_block_fraction" not in self.last_diagnostics:
                self.last_diagnostics["kernel_block_fraction"] = counts.float().mean() / nk
        global _compiled_flex
        use_compiled = q.is_cuda and head_dim >= 16
        if use_compiled and _compiled_flex is None:
            _compiled_flex = torch.compile(flex_attention)
        out = (_compiled_flex if use_compiled else flex_attention)(
            q, k, v, block_mask=block_mask)
        out = out[:, :, :tgt_len].gather(2, undo[:, None, :, None].expand(bsz, nheads, tgt_len, head_dim))
        out = out.transpose(-3, -2).contiguous().view(bsz, tgt_len, self.embed_dim)
        return self.out_proj(out)
