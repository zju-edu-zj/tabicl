"""Explicit hybrid parameter routing and checkpoint compatibility checks."""

import math

import torch

from tabicl._model.layers import MultiheadAttentionBlock
from tabicl.train._muon import Muon


def build_optimizer(model, config):
    """Muon only for transformer projections; AdamW for every other parameter.

    Use module ownership, not dimensionality: slots, gates, embeddings and SSMax
    networks must not accidentally enter Muon merely because they are matrices.
    Frozen parameters are excluded. Parameter names make state routing auditable.
    """
    if not math.isfinite(config.lr) or config.lr <= 0:
        raise ValueError("lr must be finite and positive")
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters")
    if not config.muon:
        return torch.optim.AdamW(
            [p for _, p in trainable], lr=config.lr,
            betas=(config.beta1, config.beta2), weight_decay=config.weight_decay,
        )

    adamw_lr = getattr(config, "adamw_lr", None)
    adamw_lr = config.lr / 4 if adamw_lr is None else adamw_lr
    if not math.isfinite(adamw_lr) or adamw_lr <= 0:
        raise ValueError("adamw_lr must be finite and positive")
    muon_ids = set()
    for module in model.modules():
        if isinstance(module, MultiheadAttentionBlock):
            for p in (module.attn.in_proj_weight, module.attn.out_proj.weight,
                      module.linear1.weight, module.linear2.weight):
                if p.ndim != 2:
                    raise ValueError("Muon transformer projections must be 2D")
                muon_ids.add(id(p))

    groups = {
        "muon": dict(params=[], param_names=[], use_muon=True,
                     lr=config.lr, weight_decay=config.weight_decay),
        "adamw_decay": dict(params=[], param_names=[], use_muon=False,
                            lr=adamw_lr, weight_decay=config.weight_decay),
        "adamw_no_decay": dict(params=[], param_names=[], use_muon=False,
                               lr=adamw_lr, weight_decay=0.0),
    }
    for name, p in trainable:
        key = "muon" if id(p) in muon_ids else "adamw_no_decay" if p.ndim < 2 else "adamw_decay"
        groups[key]["params"].append(p)
        groups[key]["param_names"].append(name)
    routed = [dict(group, group_name=name) for name, group in groups.items() if group["params"]]
    ids = [id(p) for group in routed for p in group["params"]]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for _, p in trainable}:
        raise ValueError("Each trainable parameter must belong to exactly one optimizer group")
    return Muon(
        routed, lr=config.lr, weight_decay=config.weight_decay,
        matched_adamw_rms=0.2, momentum=config.beta1, nesterov=True, ns_steps=5,
        adamw_betas=(config.beta1, config.beta2), adamw_eps=1e-8,
        use_cautious_wd=config.use_cautious_wd,
        compile_ns=getattr(config, "compile_optimizer", False),
    )


def optimizer_spec(optimizer):
    """Stable routing identity, independent of scheduler position and LR."""
    return dict(
        kind="muon_hybrid" if isinstance(optimizer, Muon) else "adamw",
        version=1,
        groups=[dict(
            name=g.get("group_name", "adamw"), use_muon=g.get("use_muon", False),
            param_names=list(g["param_names"]) if "param_names" in g else None,
            shapes=[list(p.shape) for p in g["params"]],
        ) for g in optimizer.param_groups],
    )


def validate_optimizer_checkpoint(optimizer, checkpoint):
    """Reject legacy all-Muon state and changed routing before loading moments."""
    expected = optimizer_spec(optimizer)
    saved = checkpoint.get("optimizer_spec")
    if saved is not None:
        compatible = saved == expected
    else:
        # Legacy unpartitioned AdamW can still resume with unpartitioned AdamW.
        groups = checkpoint.get("optimizer_state", {}).get("param_groups", [])
        compatible = (
            expected["kind"] == "adamw" and len(groups) == 1
            and not groups[0].get("use_muon", False)
            and len(groups[0]["params"]) == len(optimizer.param_groups[0]["params"])
        )
    if not compatible:
        raise ValueError(
            "Checkpoint optimizer/routing is incompatible (legacy all-parameter Muon is not hybrid). "
            "Use --only_load_model True and a NEW checkpoint_dir to initialize hybrid training; "
            "resume hybrid checkpoints with the same parameter routing."
        )
