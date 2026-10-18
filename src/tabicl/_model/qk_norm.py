"""Q/K normalization and bounded length scaling for ICL attention."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ICLQKNorm(nn.Module):
    """Per-head RMSNorm plus a bounded positive softmax temperature."""

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        reference_length: int = 512,
        max_log_slope: float = 0.5,
        eps: float = 1e-6,
        trainable_length_scale: bool = True,
    ):
        super().__init__()
        if reference_length < 1:
            raise ValueError("reference_length must be positive")
        if max_log_slope <= 0:
            raise ValueError("max_log_slope must be positive")
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.reference_length = reference_length
        self.max_log_slope = max_log_slope
        self.eps = eps
        self.q_weight = nn.Parameter(torch.ones(head_dim))
        self.k_weight = nn.Parameter(torch.ones(head_dim))
        # ELU kernel attention uses RMS-normalized Q/K but has no softmax
        # temperature. Keep these tensors in the state dict for architecture-
        # independent initialization, while excluding them from DDP/optimizers
        # in pure ELU-global layers where scale_query is never called.
        self.base_log_scale = nn.Parameter(
            torch.zeros(num_heads), requires_grad=trainable_length_scale
        )
        # sigmoid(-4) ~= 0.018: long-context scaling starts close to identity.
        self.log_slope_logit = nn.Parameter(
            torch.full((num_heads,), -4.0), requires_grad=trainable_length_scale
        )

    def normalize_q(self, q: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(q, (self.head_dim,), self.q_weight.to(q.dtype), self.eps)

    def normalize_k(self, k: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(k, (self.head_dim,), self.k_weight.to(k.dtype), self.eps)

    def scale_query(self, q: torch.Tensor, visible_keys: int) -> torch.Tensor:
        log_ratio = math.log(max(visible_keys, 1) / self.reference_length)
        slope = self.max_log_slope * self.log_slope_logit.sigmoid()
        limit = math.log(2.0)
        log_scale = self.base_log_scale.tanh().mul(limit) + slope * log_ratio
        scale = log_scale.clamp(-limit, limit).exp().to(q.dtype).view(1, self.num_heads, 1, 1)
        return q * scale
