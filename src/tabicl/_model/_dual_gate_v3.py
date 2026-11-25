"""Optional semantic modulation for two peer attention experts."""

import math

import torch
from torch import nn
from torch.nn import functional as F


DUAL_CONTROLLER_MODES = ("none", "semantic")


class DualPathGateV3(nn.Module):
    """Bounded expert modulation driven only by model-semantic inputs.

    The default ``none`` mode is the canonical architecture and returns unit
    coefficients without parameters. ``semantic`` is an experimental extension
    driven by the normalized layer state, support cardinality, and the entropy
    of the exact selected-row softmax. Search and storage telemetry never enters
    this module.
    """

    def __init__(self, embed_dim, num_heads, mode="none", hidden_dim=32):
        super().__init__()
        if mode not in DUAL_CONTROLLER_MODES:
            raise ValueError(f"dual controller must be one of {DUAL_CONTROLLER_MODES}")
        self.mode = mode
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        if mode == "semantic":
            self.query_weight = nn.Parameter(torch.empty(embed_dim, hidden_dim))
            self.evidence_weight = nn.Parameter(torch.empty(num_heads, 2, hidden_dim))
            self.hidden_bias = nn.Parameter(torch.zeros(num_heads, hidden_dim))
            self.output_weight = nn.Parameter(torch.zeros(num_heads, hidden_dim, 2))
            self.output_bias = nn.Parameter(torch.zeros(num_heads, 2))
            nn.init.xavier_uniform_(self.query_weight)
            nn.init.xavier_uniform_(self.evidence_weight)

    def forward(self, normalized_query, attention_evidence, support_size, candidate_rows):
        # Attention tensors are head-major: [batch, heads, query, 1].
        shape = (
            *normalized_query.shape[:-2],
            self.num_heads,
            normalized_query.shape[-2],
            1,
        )
        if self.mode == "none":
            one = torch.ones(shape, device=normalized_query.device, dtype=torch.float32)
            zero = one.new_zeros(())
            return one, one, {
                "fusion_controller_hidden_rms": zero,
                "fusion_controller_logit_rms": zero,
            }

        with torch.autocast(device_type=normalized_query.device.type, enabled=False):
            query_hidden = normalized_query.float() @ self.query_weight.float()
            entropy = attention_evidence.detach().float()[..., 1]
            normalized_entropy = entropy / torch.log(
                candidate_rows.detach().float().clamp_min(2)
            )
            log_n = normalized_entropy.new_full(
                normalized_entropy.shape,
                min(math.log1p(support_size) / math.log1p(100_000_000), 1.0),
            )
            evidence = torch.stack((log_n, normalized_entropy.clamp(0, 1)), -1)
            hidden = F.silu(
                query_hidden[:, None]
                + torch.einsum(
                    "bhtf,hfk->bhtk", evidence, self.evidence_weight.float()
                )
                + self.hidden_bias.float()[None, :, None]
            )
            logits = (
                torch.einsum("bhtk,hkc->bhtc", hidden, self.output_weight.float())
                + self.output_bias.float()[None, :, None]
            )
            # Unit initialization and a symmetric, bounded multiplicative range.
            coefficients = torch.exp(math.log(2.0) * torch.tanh(logits))
        return coefficients[..., 0, None], coefficients[..., 1, None], {
            "fusion_controller_hidden_rms": hidden.detach().square().mean().sqrt(),
            "fusion_controller_logit_rms": logits.detach().square().mean().sqrt(),
        }
