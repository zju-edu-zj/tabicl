"""Per-head controller for additive global and sparse attention experts."""

import math

import torch
from torch import nn
from torch.nn import functional as F


DUAL_GATE_MODES = ("static", "linear_v1", "evidence_v3")


class DualPathGateV3(nn.Module):
    """V1-compatible gates with a zero-initialized nonlinear evidence adapter.

    The two sigmoid outputs are independent modulation coefficients, not a
    probability distribution.  ``evidence_v3`` strictly contains the V1 linear
    controller: zeroing the adapter output recovers ``linear_v1`` exactly.
    """

    _BASE_ROUTE_FEATURES = 6
    _EVIDENCE_FEATURES = 9

    def __init__(self, num_heads, head_dim, mode="evidence_v3", hidden_dim=8, delta_max=2.0):
        super().__init__()
        if mode not in DUAL_GATE_MODES:
            raise ValueError(f"dual gate mode must be one of {DUAL_GATE_MODES}; got {mode!r}")
        if hidden_dim < 1 or delta_max <= 0:
            raise ValueError("dual gate hidden_dim and delta_max must be positive")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.delta_max = float(delta_max)

        self.global_query_weight = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.sparse_query_weight = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.global_route_weight = nn.Parameter(
            torch.zeros(num_heads, self._BASE_ROUTE_FEATURES)
        )
        self.sparse_route_weight = nn.Parameter(
            torch.zeros(num_heads, self._BASE_ROUTE_FEATURES)
        )
        self.base_bias = nn.Parameter(torch.zeros(num_heads, 2))

        self.global_query_adapter = nn.Parameter(
            torch.empty(num_heads, head_dim, hidden_dim)
        )
        self.sparse_query_adapter = nn.Parameter(
            torch.empty(num_heads, head_dim, hidden_dim)
        )
        self.evidence_adapter = nn.Parameter(
            torch.empty(num_heads, self._EVIDENCE_FEATURES, hidden_dim)
        )
        self.hidden_bias = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        self.output_weight = nn.Parameter(torch.zeros(num_heads, hidden_dim, 2))
        self.output_bias = nn.Parameter(torch.zeros(num_heads, 2))
        nn.init.xavier_uniform_(self.global_query_adapter)
        nn.init.xavier_uniform_(self.sparse_query_adapter)
        nn.init.xavier_uniform_(self.evidence_adapter)

        if mode == "static":
            for parameter in self.parameters():
                parameter.requires_grad_(False)
        elif mode == "linear_v1":
            for parameter in (
                self.global_query_adapter,
                self.sparse_query_adapter,
                self.evidence_adapter,
                self.hidden_bias,
                self.output_weight,
                self.output_bias,
            ):
                parameter.requires_grad_(False)

    @staticmethod
    def _base_features(route_quality):
        quality = route_quality.detach().float()
        return torch.stack(
            (
                quality[..., 0].clamp(0, 1),
                quality[..., 1].clamp(0, 1),
                quality[..., 2].clamp(0, 1),
                (quality[..., 3] / 8).clamp(-1, 1),
                (quality[..., 4] / 8).clamp(0, 1),
                quality[..., 5].clamp(0, 1),
            ),
            dim=-1,
        )

    @classmethod
    def _evidence_features(cls, route_quality, support_size):
        quality = route_quality.detach().float()
        candidate_fraction = quality[..., 1].clamp(0, 1)
        candidate_count = candidate_fraction * float(support_size)
        log_n = math.log1p(support_size) / math.log1p(2**24)
        log_n = quality.new_full(candidate_fraction.shape, min(log_n, 1.0))
        log_candidate = torch.log1p(candidate_count) / math.log1p(4096)
        compression = torch.log1p(
            float(support_size) / candidate_count.clamp_min(1)
        ) / math.log1p(2**24)
        full_budget = (candidate_count >= support_size - 0.5).float()
        return torch.stack(
            (
                log_n,
                candidate_fraction,
                log_candidate.clamp(0, 1),
                compression.clamp(0, 1),
                full_budget,
                quality[..., 2].clamp(0, 1),
                (quality[..., 3] / 8).clamp(-1, 1),
                (quality[..., 4] / 8).clamp(0, 1),
                quality[..., 5].clamp(0, 1),
            ),
            dim=-1,
        )

    def forward(self, global_q, sparse_q, route_quality, support_size):
        with torch.autocast(device_type=global_q.device.type, enabled=False):
            global_q32 = global_q.float()
            sparse_q32 = sparse_q.float()
            base_features = self._base_features(route_quality.detach().float())
            query_features = torch.stack((global_q32, sparse_q32), dim=-2)
            query_weight = torch.stack(
                (
                    self.global_query_weight.float(),
                    self.sparse_query_weight.float(),
                ),
                dim=1,
            )
            route_weight = torch.stack(
                (
                    self.global_route_weight.float(),
                    self.sparse_route_weight.float(),
                ),
                dim=1,
            )
            logits = (
                torch.einsum("bhtcd,hcd->bhtc", query_features, query_weight)
                + torch.einsum("bhtf,hcf->bhtc", base_features, route_weight)
                + self.base_bias.float()[None, :, None]
            )

            common_delta = differential_delta = torch.zeros_like(logits[..., 0])
            if self.mode == "evidence_v3":
                evidence = self._evidence_features(route_quality, support_size)
                adapter_features = torch.cat(
                    (global_q32, sparse_q32, evidence), dim=-1
                )
                adapter_weight = torch.cat(
                    (
                        self.global_query_adapter.float(),
                        self.sparse_query_adapter.float(),
                        self.evidence_adapter.float(),
                    ),
                    dim=1,
                )
                hidden = F.silu(
                    torch.einsum(
                        "bhtf,hfk->bhtk", adapter_features, adapter_weight
                    )
                    + self.hidden_bias.float()[None, :, None]
                )
                raw_delta = (
                    torch.einsum(
                        "bhtk,hkc->bhtc", hidden, self.output_weight.float()
                    )
                    + self.output_bias.float()[None, :, None]
                )
                bounded = self.delta_max * torch.tanh(raw_delta / self.delta_max)
                common_delta, differential_delta = bounded.unbind(-1)
                logits = logits + torch.stack(
                    (
                        common_delta - differential_delta,
                        common_delta + differential_delta,
                    ),
                    dim=-1,
                )

            global_gate, sparse_gate = torch.sigmoid(logits).unbind(-1)
            global_gate = global_gate[..., None]
            sparse_gate = sparse_gate[..., None]
        diagnostics = {
            "fusion_common_delta_rms": common_delta.detach().square().mean().sqrt(),
            "fusion_differential_delta_rms": differential_delta.detach().square().mean().sqrt(),
        }
        return global_gate, sparse_gate, diagnostics
