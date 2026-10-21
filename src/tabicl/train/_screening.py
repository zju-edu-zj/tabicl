"""Controlled initialization, restartable prior streams and fixed validation."""

import hashlib
import random
import types
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import IterableDataset, get_worker_info

from tabicl._model.icl_attention import SetRetrievalAttention


EVAL_ATTENTION_POLICIES = ("native", "global_only", "sparse_only")


class StepSeededPrior(IterableDataset):
    """One deterministic prior batch per global optimizer step and DDP rank.

    Worker count and prefetch do not change the batch sequence. Set start_step
    before iterating (after loading the optimizer checkpoint).
    """

    def __init__(self, dataset, np_seed, torch_seed, rank=0, world_size=1):
        self.dataset = dataset
        self.np_seed, self.torch_seed = np_seed, torch_seed
        self.rank, self.world_size = rank, world_size
        self.start_step = 0

    def __iter__(self):
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        step = self.start_step + worker
        while True:
            offset = step * self.world_size + self.rank
            np.random.seed((self.np_seed + offset) % 2**32)
            random.seed((self.np_seed + offset) % 2**32)
            torch.manual_seed((self.torch_seed + offset) % 2**63)
            yield self.dataset.get_batch()
            step += workers


def load_common_initialization(model, path, model_config, seed):
    """Copy every shared tensor from an untrained dense reference, strictly."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("purpose") != "icl_screen_initialization":
        raise ValueError("initial_weights must be a screening initialization, not a trained checkpoint")
    for key, value in model_config.items():
        if not key.startswith("icl_") and saved["config"].get(key) != value:
            raise ValueError(f"Initial model architecture differs: {key}")
    for key in ("icl_num_blocks", "icl_nhead", "icl_ssmax"):
        if saved["config"].get(key) != model_config[key]:
            raise ValueError(f"Initial model architecture differs: {key}")
    result = model.load_state_dict(saved["state_dict"], strict=False)
    allowed = (
        ".fusion_gate_weight",
        ".fusion_gate_bias",
        ".route_direction",
        ".inducing_points",
    )
    if result.unexpected_keys or any(not name.endswith(allowed) for name in result.missing_keys):
        raise ValueError(f"Incomplete common initialization: {result}")
    # Route buffers must also agree across sparse variants and DDP ranks.
    for name, module in model.named_modules():
        if hasattr(module, "route_direction"):
            value = int.from_bytes(hashlib.sha256(f"{seed}/{name}".encode()).digest()[:8], "little")
            gen = torch.Generator().manual_seed(value % 2**63)
            direction = torch.randn(module.head_dim, generator=gen)
            module.route_direction.copy_(F.normalize(direction, dim=0))
        if hasattr(module, "inducing_points"):
            value = int.from_bytes(
                hashlib.sha256(f"{seed}/{name}/inducing_points".encode()).digest()[:8],
                "little",
            )
            gen = torch.Generator().manual_seed(value % 2**63)
            points = torch.empty_like(module.inducing_points, device="cpu")
            points.normal_(std=0.02, generator=gen).clamp_(-0.04, 0.04)
            with torch.no_grad():
                module.inducing_points.copy_(points.to(module.inducing_points.device))


@contextmanager
def _attention_policy(model, policy):
    """Temporarily replace retrieval layers by one branch for a counterfactual."""
    if policy not in EVAL_ATTENTION_POLICIES:
        raise ValueError(f"Unknown validation attention policy: {policy}")
    modules = [m for m in model.icl_predictor.modules() if isinstance(m, SetRetrievalAttention)]
    if policy == "native" or not modules:
        yield
        return
    saved = []
    for module in modules:
        had_override = "_hybrid_attention" in module.__dict__
        saved.append((module, had_override, module.__dict__.get("_hybrid_attention")))

        def intervention(self, q, k, v, selected_policy=policy):
            diagnose = self.diagnostics_enabled
            if diagnose:
                self.last_diagnostics = {}
            if selected_policy == "global_only":
                heads = self._global_attention(q, k, v)
            elif self.mode == "fused_full" or k.shape[-2] <= (
                self.block_size * self.num_selected_blocks
            ):
                attention_q = (
                    self.qk_norm.scale_query(q, k.shape[-2])
                    if self.qk_norm is not None
                    else q
                )
                heads = F.scaled_dot_product_attention(attention_q, k, v)
                if diagnose:
                    self._diagnose(attention_q, k, v)
            else:
                heads = self._sparse_attention(q, k, v)
            combined = heads.transpose(-3, -2).contiguous().view(
                *q.shape[:-3], q.shape[-2], self.embed_dim
            )
            return self.out_proj(combined)

        module._hybrid_attention = types.MethodType(intervention, module)
    try:
        yield
    finally:
        for module, had_override, previous in saved:
            if had_override:
                module._hybrid_attention = previous
            else:
                del module._hybrid_attention


@torch.no_grad()
def evaluate_cases(model, data, device, amp_ctx, diagnostic_queries=16, policies=("native",)):
    """Held-out pretraining objective; no ensembling or inference calibration.

    Training execution path under no_grad intentionally keeps CE comparable to
    the training objective (all configured dropouts must be zero). Restore each
    module's mode, diagnostics and global RNG states even on failure.
    """
    if any(isinstance(m, torch.nn.Dropout) and m.p for m in model.modules()):
        raise ValueError("Screen validation requires dropout=0")
    policies = tuple(dict.fromkeys(policies))
    invalid = set(policies) - set(EVAL_ATTENTION_POLICIES)
    if not policies or invalid:
        raise ValueError(f"Invalid validation attention policies: {sorted(invalid)}")
    has_retrieval = any(isinstance(m, SetRetrievalAttention) for m in model.modules())
    if not has_retrieval:
        policies = ("native",)
    modes = {m: m.training for m in model.modules()}
    diagnostics = [(name, m) for name, m in model.icl_predictor.named_modules() if hasattr(m, "diagnostics_enabled")]
    states = {
        m: (m.diagnostics_enabled, m.diagnostic_query_start, m.diagnostic_queries, m.last_diagnostics)
        for _, m in diagnostics
    }
    np_state, py_state = np.random.get_state(), random.getstate()
    dev = torch.device(device)
    devices = [torch.cuda.current_device() if dev.index is None else dev.index] if dev.type == "cuda" else []
    records = []
    try:
        model.train()
        with torch.random.fork_rng(devices=devices), amp_ctx:
            for policy in policies:
                with _attention_policy(model, policy):
                    for case in data["cases"]:
                        x, y, s = case["x"].to(device), case["y"].to(device), case["train_size"]
                        for _, m in diagnostics:
                            m.diagnostics_enabled = diagnostic_queries > 0
                            m.diagnostic_query_start = s
                            m.diagnostic_queries = max(1, diagnostic_queries)
                            m.last_diagnostics = {}
                        pred = model(x, y[:, :s]).float()
                        if not torch.isfinite(pred).all():
                            raise FloatingPointError(
                                f"Nonfinite validation predictions: {case['id']}, policy={policy}"
                            )
                        record = {
                            "id": case["id"], "support": s, "features": x.shape[-1], "policy": policy
                        }
                        target = y[:, s:]
                        if model.max_classes > 0:
                            record["ce"] = F.cross_entropy(
                                pred.flatten(0, 1), target.long().flatten()
                            ).item()
                            record["accuracy"] = (pred.argmax(-1) == target).float().mean().item()
                        else:
                            alphas = torch.linspace(0, 1, pred.shape[-1] + 2, device=device)[1:-1]
                            err = target[..., None] - pred
                            record["pinball"] = torch.maximum(
                                alphas * err, (alphas - 1) * err
                            ).mean().item()
                            record["median_mse"] = (
                                pred[..., pred.shape[-1] // 2] - target
                            ).square().mean().item()
                        for name, m in diagnostics:
                            for key, value in m.last_diagnostics.items():
                                record[f"diag/{name}/{key}"] = float(value)
                        records.append(record)
    finally:
        for m, mode in modes.items():
            m.training = mode
        for m, state in states.items():
            m.diagnostics_enabled, m.diagnostic_query_start, m.diagnostic_queries, m.last_diagnostics = state
        np.random.set_state(np_state)
        random.setstate(py_state)
    panels = defaultdict(list)
    for record in records:
        panels[(record["policy"], record["support"])].append(record)
    metrics = {}
    for (policy, support), rows in panels.items():
        prefix = "val" if policy == "native" else f"val/{policy}"
        for key in rows[0]:
            if key not in ("id", "support", "features", "policy"):
                metrics[f"{prefix}/support_{support}/{key}"] = sum(r[key] for r in rows) / len(rows)
    return metrics, records
