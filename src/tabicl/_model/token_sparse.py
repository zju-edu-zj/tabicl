"""Learned per-query token retrieval for pure sparse ICL attention."""

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .layers import SupportAttention


class TokenSparseAttention(SupportAttention):
    """Select raw support rows with a learned shared index, then apply exact MHA."""

    def __init__(
        self,
        *args,
        topk=128,
        index_dim=16,
        query_chunk_size=64,
        index_query_samples=32,
        index_regression=False,
        index_num_classes=10,
        index_regression_temperature=0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if topk < 1 or index_dim < 1 or query_chunk_size < 1 or index_query_samples < 1:
            raise ValueError("token sparse sizes must be positive")
        if index_regression_temperature <= 0:
            raise ValueError("index_regression_temperature must be positive")
        if self.dropout:
            raise ValueError("Token sparse ICL attention currently requires dropout=0")
        self.topk = topk
        self.index_dim = index_dim
        self.query_chunk_size = query_chunk_size
        self.index_query_samples = index_query_samples
        self.index_regression = index_regression
        self.index_num_classes = index_num_classes
        self.index_regression_temperature = index_regression_temperature
        # A 32M-element score workspace keeps GEMM/top-k launches efficient at
        # long context while remaining small relative to model activations.
        self.route_score_chunk_elements = 32 * 1024**2
        self.index_q_proj = nn.Linear(self.embed_dim, index_dim, bias=False)
        self.index_k_proj = nn.Linear(self.embed_dim, index_dim, bias=False)
        nn.init.xavier_uniform_(self.index_q_proj.weight)
        nn.init.xavier_uniform_(self.index_k_proj.weight)
        # Cosine scores need a useful initial dynamic range for the auxiliary loss.
        self.index_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.use_triton = True
        self.diagnostics_enabled = False
        self.diagnostic_queries = 16
        self.last_diagnostics = {}
        self._route_record = None
        self._route_replay = None

    @staticmethod
    def _merge_heads(x):
        return x.transpose(1, 2).contiguous().flatten(-2)

    def _index_embeddings(self, q, k):
        # The indexer learns retrieval without pushing its auxiliary objective into
        # the main attention projections or the preceding transformer state.
        q_index = F.normalize(self.index_q_proj(self._merge_heads(q).detach()).float(), dim=-1)
        k_index = F.normalize(self.index_k_proj(self._merge_heads(k).detach()).float(), dim=-1)
        return q_index, k_index

    def _prepare_support(self, k, v):
        del v
        merged = self._merge_heads(k).detach()
        return {"index_k": F.normalize(self.index_k_proj(merged).float(), dim=-1)}

    @torch.no_grad()
    def route(self, q_index, k_index):
        """Return shared raw support indices with shape ``[B, N, K]``."""
        if self._route_replay is not None:
            return next(self._route_replay)
        support = k_index.shape[1]
        budget = min(self.topk, support)
        if support == 0:
            raise ValueError("Token sparse ICL attention needs at least one support row")
        chunk = max(1, self.route_score_chunk_elements // (q_index.shape[0] * support))
        route_dtype = torch.bfloat16 if q_index.is_cuda else torch.float32
        keys = k_index.to(route_dtype).transpose(-1, -2)
        pieces = []
        for start in range(0, q_index.shape[1], chunk):
            scores = q_index[:, start : start + chunk].to(route_dtype) @ keys
            pieces.append(scores.topk(budget, dim=-1, sorted=False).indices.to(torch.int32))
        indices = torch.cat(pieces, dim=1)
        if self._route_record is not None:
            self._route_record.append(indices)
        return indices

    def checkpoint_contexts(self):
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

    def _classification_index_loss(self, scores, support_targets, query_targets):
        class_logits = []
        class_counts = []
        for class_id in range(self.index_num_classes):
            mask = support_targets == class_id
            count = mask.sum(-1).clamp_min(1)
            class_counts.append(mask.sum(-1))
            selected = scores.masked_fill(~mask[:, None], -float("inf"))
            value = selected.logsumexp(-1) - count[:, None].float().log()
            value = value.masked_fill(~mask.any(-1)[:, None], -float("inf"))
            class_logits.append(value)
        logits = torch.stack(class_logits, dim=-1)
        counts = torch.stack(class_counts, dim=-1)
        targets = query_targets.long()
        valid = counts.gather(1, targets).gt(0)
        if not valid.any():
            return scores.sum() * 0.0
        return F.cross_entropy(logits[valid], targets[valid])

    def _regression_index_loss(self, scores, support_targets, query_targets):
        center = support_targets.median(-1).values
        scale = (support_targets - center[:, None]).abs().median(-1).values.clamp_min(1e-4)
        distance = (query_targets[:, :, None] - support_targets[:, None]).abs()
        target_logits = -distance / (scale[:, None, None] * self.index_regression_temperature)
        target = target_logits.softmax(-1)
        return -(target * scores.log_softmax(-1)).sum(-1).mean()

    def _index_loss(self, q_index, k_index, support, targets, positions):
        if targets is None or positions is None:
            raise ValueError("index targets and sampled query positions are required")
        if positions.ndim != 1 or positions.numel() > self.index_query_samples:
            raise ValueError("invalid token index query positions")
        if positions.numel() == 0:
            raise ValueError("token index supervision needs at least one test query")
        sampled_q = q_index.index_select(1, positions)
        scale = self.index_logit_scale.exp().clamp(max=100.0)
        scores = scale * (sampled_q @ k_index.transpose(-1, -2))
        support_targets = targets[:, :support]
        query_targets = targets.index_select(1, positions)
        if self.index_regression:
            return self._regression_index_loss(
                scores, support_targets.float(), query_targets.float()
            )
        return self._classification_index_loss(scores, support_targets.long(), query_targets.long())

    @staticmethod
    def _selected_attention(q, k, v, indices):
        b, h, n, d = q.shape
        index = indices[:, None, :, :, None].long().expand(-1, h, -1, -1, d)
        keys = k[:, :, None].expand(-1, -1, n, -1, -1).gather(3, index)
        values = v[:, :, None].expand(-1, -1, n, -1, -1).gather(3, index)
        return F.scaled_dot_product_attention(
            q.reshape(b * h * n, 1, d),
            keys.reshape(b * h * n, -1, d),
            values.reshape(b * h * n, -1, d),
        ).reshape(b, h, n, d)

    def _sparse_attention(self, q, k, v, indices):
        attention_q = self.qk_norm.scale_query(q, indices.shape[-1]) if self.qk_norm is not None else q
        if q.is_cuda and self.use_triton and q.shape[-1] in (16, 32, 64, 128):
            from ._token_sparse_triton import token_sparse_attention

            return token_sparse_attention(attention_q, k, v, indices)
        outputs = []
        for start in range(0, q.shape[-2], self.query_chunk_size):
            args = (
                attention_q[:, :, start : start + self.query_chunk_size],
                k,
                v,
                indices[:, start : start + self.query_chunk_size],
            )
            if torch.is_grad_enabled() and any(t.requires_grad for t in args[:3]):
                outputs.append(checkpoint(self._selected_attention, *args, use_reentrant=False))
            else:
                outputs.append(self._selected_attention(*args))
        return torch.cat(outputs, dim=2)

    @torch.compiler.disable
    def _hybrid_attention(
        self,
        q,
        k,
        v,
        prepared=None,
        index_targets=None,
        index_query_indices=None,
        return_index_loss=False,
    ):
        if not k.shape[-2]:
            raise ValueError("Token sparse ICL attention needs at least one support row")
        q_index = F.normalize(
            self.index_q_proj(self._merge_heads(q).detach()).float(), dim=-1
        )
        computed_k_index = None
        if prepared is not None and "index_k" in prepared:
            k_index = prepared["index_k"]
        else:
            computed_k_index = F.normalize(
                self.index_k_proj(self._merge_heads(k).detach()).float(), dim=-1
            )
            k_index = computed_k_index
        index_loss = None
        if return_index_loss:
            if computed_k_index is None:
                computed_k_index = F.normalize(
                    self.index_k_proj(self._merge_heads(k).detach()).float(), dim=-1
                )
            index_loss = self._index_loss(
                q_index, computed_k_index, k.shape[-2], index_targets, index_query_indices
            )
        if k.shape[-2] <= self.topk:
            attention_q = self.qk_norm.scale_query(q, k.shape[-2]) if self.qk_norm is not None else q
            heads = F.scaled_dot_product_attention(attention_q, k, v)
            indices = None
        else:
            indices = self.route(q_index.detach(), k_index.detach())
            heads = self._sparse_attention(q, k, v, indices)
        if self.diagnostics_enabled:
            with torch.no_grad():
                metrics = {
                    "candidate_fraction": torch.tensor(
                        min(self.topk, k.shape[-2]) / k.shape[-2], device=q.device
                    ),
                    "output_rms": heads.float().square().mean().sqrt(),
                    "nonfinite_fraction": (~torch.isfinite(heads)).float().mean(),
                }
                if index_loss is not None:
                    metrics["index_loss"] = index_loss.detach()
                if indices is not None and index_targets is not None:
                    support_targets = index_targets[:, : k.shape[-2]]
                    selected_targets = support_targets[:, None].expand(-1, indices.shape[1], -1).gather(
                        2, indices.long()
                    )
                    query_targets = index_targets[:, : indices.shape[1]]
                    metrics["selected_same_target_fraction"] = (
                        selected_targets == query_targets[:, :, None]
                    ).float().mean()
                self.last_diagnostics = metrics
            self.diagnostics_enabled = False
        combined = heads.transpose(1, 2).contiguous().view(q.shape[0], q.shape[2], self.embed_dim)
        output = self.out_proj(combined)
        return (output, index_loss) if return_index_loss else output

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
        index_targets=None,
        index_query_indices=None,
        return_index_loss=False,
    ):
        if not return_index_loss:
            return super().forward(
                query, key, value, cached_kv=cached_kv,
                key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                rope=rope, need_kv=need_kv,
            )
        if cached_kv is not None or key_padding_mask is not None or attn_mask is not None or need_kv:
            raise ValueError("index supervision requires an uncached support-prefix forward")
        key = query if key is None else key
        value = key if value is None else value
        q, k, v = self._project(query, key, value, rope)
        return self._hybrid_attention(
            q, k, v,
            index_targets=index_targets,
            index_query_indices=index_query_indices,
            return_index_loss=True,
        )
