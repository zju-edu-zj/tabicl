"""Train/inference unified Dual-Path V3 attention."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ._dual_gate_v3 import DualPathGateV3
from .content_index_v3 import ContentIndexV3, DualPathLayout
from .dual_path_store import DualPathPageStore, PreparedDualPathSupport
from .icl_attention import SetRetrievalAttention
from .layers import GlobalLinearAttention
from .qk_norm import ICLQKNorm


class DualPathRetrievalAttentionV3(SetRetrievalAttention):
    """Independent global and sparse experts sharing one candidate definition."""

    def __init__(
        self,
        *args,
        dual_controller="none",
        storage_page_rows=128,
        exhaustive_support_limit=8192,
        **kwargs,
    ):
        kwargs.pop("mode", None)
        kwargs.pop("fusion", None)
        super().__init__(*args, mode="sparse", fusion="legacy", **kwargs)
        if self.partition != "content_tree":
            raise ValueError("dual_path requires partition=content_tree")
        if self.global_type != "elu":
            raise ValueError("dual_path requires global_type=elu")
        if self.qk_norm is None:
            raise ValueError("dual_path requires qk_norm=True")
        if self.block_size != 32 or self.num_selected_blocks != 32:
            raise ValueError("dual_path v3 requires 32 logical rows x 32 selected leaves")
        self.layout = DualPathLayout(
            logical_leaf_rows=self.block_size,
            storage_page_rows=storage_page_rows,
            selected_logical_leaves=self.num_selected_blocks,
            search_beam_pages=max(self.router_beam, 64),
            proposal_pages=max(self.router_beam, 64),
            exhaustive_support_limit=exhaustive_support_limit,
            query_tile_rows=self.query_chunk_size,
        )
        self.global_in_proj_weight = nn.Parameter(
            torch.empty(3 * self.embed_dim, self.embed_dim)
        )
        self.global_in_proj_bias = nn.Parameter(torch.zeros(3 * self.embed_dim))
        nn.init.xavier_uniform_(self.global_in_proj_weight)
        self.global_qk_norm = ICLQKNorm(
            self.head_dim, self.num_heads, trainable_length_scale=False
        )
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.global_out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.global_out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.dual_controller = dual_controller
        self.gate_controller = DualPathGateV3(
            self.embed_dim, self.num_heads, mode=dual_controller
        )
        self._branch_policy = "native"

    def zero_init_output(self):
        # Nonzero O matrices let Q/K/V and the optional controller receive task
        # gradient on the first update.
        return None

    def _heads(self, x, length):
        return x.view(
            *x.shape[:-2], length, self.num_heads, self.head_dim
        ).transpose(-3, -2)

    def _project_dual(self, query, key, rope):
        q_weight = torch.cat((
            self.in_proj_weight[: self.embed_dim],
            self.global_in_proj_weight[: self.embed_dim],
        ))
        q_bias = torch.cat((
            self.in_proj_bias[: self.embed_dim],
            self.global_in_proj_bias[: self.embed_dim],
        ))
        sparse_q, global_q = F.linear(query, q_weight, q_bias).split(
            self.embed_dim, -1
        )
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
        ).split(self.embed_dim, -1)
        sparse_q = self._heads(sparse_q, query.shape[-2])
        global_q = self._heads(global_q, query.shape[-2])
        sparse_k = self._heads(sparse_k, key.shape[-2])
        sparse_v = self._heads(sparse_v, key.shape[-2])
        global_k = self._heads(global_k, key.shape[-2])
        global_v = self._heads(global_v, key.shape[-2])
        if rope is not None:
            sparse_q = rope.rotate_queries_or_keys(sparse_q)
            global_q = rope.rotate_queries_or_keys(global_q)
            sparse_k = rope.rotate_queries_or_keys(sparse_k)
            global_k = rope.rotate_queries_or_keys(global_k)
        return (
            self.qk_norm.normalize_q(sparse_q),
            self.qk_norm.normalize_k(sparse_k),
            sparse_v,
            self.global_qk_norm.normalize_q(global_q),
            self.global_qk_norm.normalize_k(global_k),
            global_v,
        )

    def _project_support(self, support, rope):
        weight = torch.cat((
            self.in_proj_weight[self.embed_dim :],
            self.global_in_proj_weight[self.embed_dim :],
        ))
        bias = torch.cat((
            self.in_proj_bias[self.embed_dim :],
            self.global_in_proj_bias[self.embed_dim :],
        ))
        sparse_k, sparse_v, global_k, global_v = F.linear(
            support, weight, bias
        ).split(self.embed_dim, -1)
        sparse_k = self._heads(sparse_k, support.shape[-2])
        sparse_v = self._heads(sparse_v, support.shape[-2])
        global_k = self._heads(global_k, support.shape[-2])
        global_v = self._heads(global_v, support.shape[-2])
        if rope is not None:
            sparse_k = rope.rotate_queries_or_keys(sparse_k)
            global_k = rope.rotate_queries_or_keys(global_k)
        return (
            self.qk_norm.normalize_k(sparse_k), sparse_v,
            self.global_qk_norm.normalize_k(global_k), global_v,
        )

    def _project_queries(self, query, rope):
        weight = torch.cat((
            self.in_proj_weight[: self.embed_dim],
            self.global_in_proj_weight[: self.embed_dim],
        ))
        bias = torch.cat((
            self.in_proj_bias[: self.embed_dim],
            self.global_in_proj_bias[: self.embed_dim],
        ))
        sparse_q, global_q = F.linear(query, weight, bias).split(self.embed_dim, -1)
        sparse_q = self._heads(sparse_q, query.shape[-2])
        global_q = self._heads(global_q, query.shape[-2])
        if rope is not None:
            sparse_q = rope.rotate_queries_or_keys(sparse_q)
            global_q = rope.rotate_queries_or_keys(global_q)
        return (
            self.qk_norm.normalize_q(sparse_q),
            self.global_qk_norm.normalize_q(global_q),
        )

    def _build_index(self, sparse_k):
        with torch.no_grad(), torch.autocast(
            device_type=sparse_k.device.type, enabled=False
        ):
            initial = torch.einsum(
                "bhsd,d->bhs", sparse_k.float(), self.route_direction.float()
            ).argsort(dim=-1, stable=True)
            return ContentIndexV3.build(
                sparse_k,
                self.layout,
                use_triton=self.use_triton,
                initial_order=initial,
            )

    @staticmethod
    def _selected_attention_evidence(q, k, v, order, selected, size):
        b, h, n, d = q.shape
        positions = (
            selected.long()[..., None] * size
            + torch.arange(size, device=q.device)
        ).flatten(-2)
        valid = positions < k.shape[-2]
        sorted_positions = positions.clamp_max(k.shape[-2] - 1)
        original = order.long()[:, :, None].expand(-1, -1, n, -1).gather(
            3, sorted_positions
        )
        index = original[..., None].expand(-1, -1, -1, -1, d)
        keys = k[:, :, None].expand(-1, -1, n, -1, -1).gather(3, index)
        values = v[:, :, None].expand(-1, -1, n, -1, -1).gather(3, index)
        logits = torch.einsum(
            "bhnd,bhnkd->bhnk", q.float(), keys.float()
        ) / math.sqrt(d)
        logits = logits.masked_fill(~valid, -float("inf"))
        prob = logits.softmax(-1)
        out = torch.einsum(
            "bhnk,bhnkd->bhnd", prob.to(values.dtype), values
        )
        entropy = -(prob * prob.clamp_min(1e-20).log()).sum(-1)
        p2 = prob.square().sum(-1).clamp_min(1e-20)
        evidence = torch.stack(
            (prob.max(-1).values, entropy, p2.reciprocal()), -1
        )
        return out, evidence

    def _sparse_attention_v3(self, q, k, v, index, order=None):
        route = index.search(q)
        visible = min(index.support_size, self.layout.selected_rows)
        attention_q = self.qk_norm.scale_query(q, visible)
        order = index.order if order is None else order
        need_evidence = self.dual_controller == "semantic" or self.diagnostics_enabled
        if (
            q.is_cuda
            and self.use_triton
            and q.shape[-1] in (16, 32, 64, 128)
            and self.layout.selected_rows <= 2048
        ):
            from ._sparse_triton import routed_attention

            result = routed_attention(
                attention_q,
                k,
                v,
                order.to(torch.int32),
                route.logical_leaf_ids,
                self.layout.logical_leaf_rows,
                return_evidence=need_evidence,
            )
            if need_evidence:
                out, evidence = result
            else:
                out = result
                evidence = torch.zeros(
                    (*q.shape[:-1], 3), device=q.device, dtype=torch.float32
                )
        else:
            outputs, evidence_parts = [], []
            for start in range(0, q.shape[-2], self.layout.query_tile_rows):
                out, evidence = self._selected_attention_evidence(
                    attention_q[:, :, start : start + self.layout.query_tile_rows],
                    k,
                    v,
                    order,
                    route.logical_leaf_ids[
                        :, :, start : start + self.layout.query_tile_rows
                    ],
                    self.layout.logical_leaf_rows,
                )
                outputs.append(out)
                evidence_parts.append(evidence)
            out = torch.cat(outputs, 2)
            evidence = torch.cat(evidence_parts, 2)
        if self.diagnostics_enabled:
            self.last_diagnostics = self._diagnose(
                attention_q,
                k,
                v,
                route.logical_leaf_ids,
                order,
            )
        return out, evidence, route

    @torch.no_grad()
    def _diagnose(self, q, k, v, selected=None, order=None):
        """Collect semantic retrieval diagnostics without changing execution."""
        diagnostics = {}
        first = min(max(self.diagnostic_query_start, 0), q.shape[-2] - 1)
        count = min(self.diagnostic_queries, q.shape[-2] - first)
        sample = torch.linspace(first, q.shape[-2] - 1, count, device=q.device).long()
        if order is None:
            order = torch.arange(k.shape[-2], device=k.device, dtype=torch.int64)
            order = order[None, None].expand(k.shape[0], k.shape[1], -1)
        sorted_k = k.gather(
            2, order.long()[..., None].expand_as(k)
        )
        sorted_v = v.gather(
            2, order.long()[..., None].expand_as(v)
        )
        logits = torch.einsum(
            "bhqd,bhsd->bhqs", q[:, :, sample].float(), sorted_k.float()
        ) / math.sqrt(self.head_dim)
        dense_prob = logits.softmax(-1)
        if selected is None:
            mask = torch.ones_like(logits, dtype=torch.bool)
        else:
            positions = torch.arange(k.shape[-2], device=q.device)
            leaf_ids = positions // self.layout.logical_leaf_rows
            mask = (selected[:, :, sample, :, None] == leaf_ids).any(-2)
        sparse_prob = logits.masked_fill(~mask, -float("inf")).softmax(-1)
        topk = logits.topk(min(self.diagnostic_topk, k.shape[-2]), -1).indices
        dense_out = torch.einsum("bhqs,bhsd->bhqd", dense_prob, sorted_v.float())
        sparse_out = torch.einsum("bhqs,bhsd->bhqd", sparse_prob, sorted_v.float())
        for head in range(self.num_heads):
            dense_rms = dense_out[:, head].square().mean().sqrt().clamp_min(1e-6)
            diagnostics[f"head_{head}/recall_at_k"] = (
                mask[:, head].gather(-1, topk[:, head]).float().mean()
            )
            diagnostics[f"head_{head}/retained_mass"] = (
                dense_prob[:, head] * mask[:, head]
            ).sum(-1).mean()
            diagnostics[f"head_{head}/dense_output_relative_error"] = (
                (sparse_out[:, head] - dense_out[:, head]).square().mean().sqrt()
                / dense_rms
            )
        return diagnostics

    def _fuse(
        self,
        normalized_query,
        sparse_q,
        global_heads,
        sparse_heads,
        attention_evidence,
        route,
        support_size,
    ):
        global_scale, sparse_scale, controller_diagnostics = self.gate_controller(
            normalized_query,
            attention_evidence,
            support_size,
            route.candidate_rows,
        )
        with torch.autocast(device_type=sparse_q.device.type, enabled=False):
            global_scaled = global_heads.float() * global_scale
            sparse_scaled = sparse_heads.float() * sparse_scale
            if self._branch_policy == "global_only":
                sparse_scaled = torch.zeros_like(sparse_scaled)
            elif self._branch_policy == "sparse_only":
                global_scaled = torch.zeros_like(global_scaled)
            elif self._branch_policy != "native":
                raise ValueError(f"Unknown dual branch policy: {self._branch_policy!r}")
        global_flat = global_scaled.to(sparse_q.dtype).transpose(-3, -2).contiguous().view(
            *normalized_query.shape[:-1], self.embed_dim
        )
        sparse_flat = sparse_scaled.to(sparse_q.dtype).transpose(-3, -2).contiguous().view_as(
            global_flat
        )
        output = F.linear(global_flat, self.global_out_proj.weight)
        output = output + F.linear(
            sparse_flat, self.out_proj.weight, self.out_proj.bias
        )
        if self.diagnostics_enabled:
            retrieval_diagnostics = dict(self.last_diagnostics)
            diagnostics = {}
            for head in range(self.num_heads):
                values = attention_evidence[:, head, :, 0].detach().float().reshape(-1)
                diagnostics[f"head_{head}/sparse_peak_probability_mean"] = values.mean()
                diagnostics[f"head_{head}/sparse_peak_probability_std"] = values.std(unbiased=False)
                diagnostics[f"head_{head}/sparse_peak_probability_min"] = values.min()
                diagnostics[f"head_{head}/sparse_peak_probability_max"] = values.max()
            self.last_diagnostics = {
                "fusion_global_scale_mean": global_scale.detach().mean(),
                "fusion_sparse_scale_mean": sparse_scale.detach().mean(),
                "sparse_attention_entropy": attention_evidence[..., 1].detach().mean(),
                "sparse_attention_ess": attention_evidence[..., 2].detach().mean(),
                "route_boundary_margin": route.boundary_margin.detach().mean(),
                "route_proposal_entropy": route.proposal_entropy.detach().mean(),
                "route_candidate_rows": route.candidate_rows.detach().mean(),
                "route_visited_nodes": route.visited_nodes.detach().float().mean(),
                **controller_diagnostics,
                **diagnostics,
                **retrieval_diagnostics,
            }
            self.diagnostics_enabled = False
        return output

    @torch.no_grad()
    def prepare_support(self, normalized_support, rope=None, location="device"):
        sparse_k, sparse_v, global_k, global_v = self._project_support(
            normalized_support, rope
        )
        index = self._build_index(sparse_k)
        global_state = GlobalLinearAttention._prepare_support(self, global_k, global_v)
        page_store = DualPathPageStore.pack(
            sparse_k, sparse_v, index, location=location
        )
        return PreparedDualPathSupport(
            index=index,
            page_store=page_store,
            global_state=global_state,
            support_size=sparse_k.shape[-2],
        )

    @torch.no_grad()
    def prepare_support_streaming(
        self,
        support,
        normalizer,
        rope=None,
        location="device",
        chunk_rows=65536,
    ):
        """Two-pass preparation without a second full-sized K/V allocation."""
        if rope is not None:
            raise NotImplementedError(
                "streaming Dual-Path preparation needs offset-aware RoPE"
            )
        if support.ndim != 3 or chunk_rows < 1:
            raise ValueError("support must be [B,N,D] and chunk_rows must be positive")
        b, rows, _ = support.shape
        sparse_key = torch.empty(
            (b, self.num_heads, rows, self.head_dim),
            device=support.device,
            dtype=support.dtype,
        )
        key_weight = self.in_proj_weight[
            self.embed_dim : 2 * self.embed_dim
        ]
        key_bias = self.in_proj_bias[self.embed_dim : 2 * self.embed_dim]
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            normalized = normalizer(support[:, start:end])
            projected = F.linear(normalized, key_weight, key_bias)
            projected = self._heads(projected, end - start)
            sparse_key[:, :, start:end] = self.qk_norm.normalize_k(projected)
        index = self._build_index(sparse_key)
        del sparse_key

        page_rows = self.layout.storage_page_rows
        pages = (rows + page_rows - 1) // page_rows
        store_device = support.device if location == "device" else torch.device("cpu")
        pin = location == "host" and support.device.type == "cuda"
        shape = (b, self.num_heads, pages, page_rows, self.head_dim)
        packed_key = torch.empty(
            shape, device=store_device, dtype=support.dtype, pin_memory=pin
        )
        packed_value = torch.empty_like(packed_key)
        flat_key = packed_key.flatten(2, 3)
        flat_value = packed_value.flatten(2, 3)
        if pages * page_rows > rows:
            flat_key[..., rows:, :].zero_()
            flat_value[..., rows:, :].zero_()

        inverse = torch.empty_like(index.order, dtype=torch.int64)
        sorted_position = torch.arange(rows, device=support.device)[None, None].expand(
            b, self.num_heads, -1
        )
        inverse.scatter_(2, index.order.long(), sorted_position)
        state = torch.zeros(
            (b, self.num_heads, self.head_dim, self.head_dim),
            device=support.device,
            dtype=torch.float32,
        )
        normalizer_state = torch.zeros(
            (b, self.num_heads, self.head_dim),
            device=support.device,
            dtype=torch.float32,
        )
        for start in range(0, rows, chunk_rows):
            end = min(start + chunk_rows, rows)
            normalized = normalizer(support[:, start:end])
            sparse_k, sparse_v, global_k, global_v = self._project_support(
                normalized, rope=None
            )
            positions = inverse[:, :, start:end]
            scatter = positions[..., None].expand_as(sparse_k)
            if location == "host":
                flat_key.scatter_(
                    2, scatter.cpu(), sparse_k.cpu()
                )
                flat_value.scatter_(
                    2, scatter.cpu(), sparse_v.cpu()
                )
            else:
                flat_key.scatter_(2, scatter, sparse_k)
                flat_value.scatter_(2, scatter, sparse_v)
            feature = self._feature_map(global_k)
            state += torch.einsum(
                "bhsm,bhsv->bhmv", feature, global_v.float()
            )
            normalizer_state += feature.sum(-2)
        del inverse
        return PreparedDualPathSupport(
            index=index,
            page_store=DualPathPageStore(
                packed_key, packed_value, rows, self.layout
            ),
            global_state={"elu": (state, normalizer_state)},
            support_size=rows,
        )

    def apply_prepared(self, normalized_query, prepared, rope=None):
        from ._elu_kernel import apply_elu_state

        sparse_q, global_q = self._project_queries(normalized_query, rope)
        route = prepared.index.search(sparse_q)
        visible = min(prepared.support_size, self.layout.selected_rows)
        attention_q = self.qk_norm.scale_query(sparse_q, visible)
        if prepared.page_store.device == sparse_q.device:
            key, value = prepared.page_store.flattened()
            key = key[..., : prepared.support_size, :]
            value = value[..., : prepared.support_size, :]
            identity = torch.arange(
                prepared.support_size, device=key.device, dtype=torch.int32
            )[None, None].expand(key.shape[0], key.shape[1], -1)
            sparse_heads, evidence, route = self._sparse_attention_v3(
                sparse_q, key, value, prepared.index, order=identity
            )
        else:
            candidates = prepared.page_store.candidate_tensors(
                route.logical_leaf_ids, sparse_q.device
            )
            logits = torch.einsum(
                "bhnd,bhnkd->bhnk", attention_q.float(), candidates.key.float()
            ) / math.sqrt(self.head_dim)
            logits = logits.masked_fill(~candidates.valid, -float("inf"))
            prob = logits.softmax(-1)
            sparse_heads = torch.einsum(
                "bhnk,bhnkd->bhnd",
                prob.to(candidates.value.dtype),
                candidates.value,
            )
            evidence = torch.stack(
                (
                    prob.max(-1).values,
                    -(prob * prob.clamp_min(1e-20).log()).sum(-1),
                    prob.square().sum(-1).clamp_min(1e-20).reciprocal(),
                ),
                -1,
            )
        global_heads = apply_elu_state(global_q, *prepared.global_state["elu"])
        return self._fuse(
            normalized_query,
            sparse_q,
            global_heads,
            sparse_heads,
            evidence,
            route,
            prepared.support_size,
        )

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
        if cached_kv is not None or need_kv:
            raise ValueError(
                "dual_path v3 uses prepare_support/apply_prepared or layerwise "
                "inference, not the legacy all-layer KV cache"
            )
        if key_padding_mask is not None or attn_mask is not None:
            raise NotImplementedError("dual_path requires an explicit support prefix")
        key = query if key is None else key
        sparse_q, sparse_k, sparse_v, global_q, global_k, global_v = self._project_dual(
            query, key, rope
        )
        index = self._build_index(sparse_k)
        diagnostics_requested = self.diagnostics_enabled
        global_heads = self._global_attention(global_q, global_k, global_v)
        global_diagnostics = (
            dict(self.last_diagnostics) if diagnostics_requested else {}
        )
        if diagnostics_requested:
            self.diagnostics_enabled = True
        sparse_heads, evidence, route = self._sparse_attention_v3(
            sparse_q, sparse_k, sparse_v, index
        )
        if diagnostics_requested:
            self.last_diagnostics = {
                **{f"global/{key}": value for key, value in global_diagnostics.items()},
                **self.last_diagnostics,
            }
        return self._fuse(
            query,
            sparse_q,
            global_heads,
            sparse_heads,
            evidence,
            route,
            sparse_k.shape[-2],
        )
