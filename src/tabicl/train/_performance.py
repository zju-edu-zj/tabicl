"""Training-only attention policy and update-free compilation warmup."""
from contextlib import nullcontext
import random
import time

import numpy as np
import torch
import torch.nn.functional as F


def resolve_attention_backend(policy, dtype, max_rows, device_name):
    if policy != "auto":
        return policy
    # Measured on the ZW810 PyTorch 2.8 vendor stack, not extrapolated to NVIDIA.
    if dtype != "float32" or "ZW810" not in device_name:
        return "native"
    return "fp32_math" if max_rows <= 2048 else "fp32_row_math"


def frontend_warmup_shapes(cfg):
    """Three admissible (rows, width, support) shapes, including one column.

    Long-table priors cap widths by rows. Warming 60k x 100 when the actual
    cap is 20 needlessly raises startup memory and can fail before training.
    The midpoint also exercises the wider, medium-length part of that prior.
    """
    from tabicl.prior._dataset import Prior

    rows = cfg.max_seq_len
    cap = cfg.max_features
    adaptive = (getattr(cfg, "seq_len_per_gp", False)
                and getattr(cfg, "adaptive_feature_limit", True))
    if adaptive:
        cap = max(cfg.min_features, Prior.adjust_max_features(rows, cfg.max_features))
    shapes = [(rows, cap), (rows, cfg.min_features),
              (rows, max(cfg.min_features, int(.37 * cfg.max_features)))]
    if cap < cfg.max_features:
        middle = max(2, getattr(cfg, "min_seq_len", None) or 2, rows // 2)
        middle_cap = max(cfg.min_features, Prior.adjust_max_features(middle, cfg.max_features))
        shapes[2] = (middle, middle_cap)
    result = []
    for (length, width), position in zip(shapes, ("middle", "low", "high")):
        low = int(cfg.min_train_size * length) if cfg.min_train_size < 1 else int(cfg.min_train_size)
        high = int(cfg.max_train_size * length) if cfg.max_train_size < 1 else int(cfg.max_train_size)
        low, high = max(1, low), min(length - 1, high)
        if low > high:
            raise ValueError("invalid warmup support interval")
        support = low if position == "low" else high if position == "high" else (low + high) // 2
        result.append((length, width, support))
    return result


def warmup_frontend(trainer):
    """Run three local shapes on every rank concurrently, without training steps.

    Covers common dynamic shapes and the one-column specialization, not every
    possible guard. Compilation caches are process-local. Startup includes the
    final barrier; none of these passes consume prior data or update weights.
    """
    cfg = trainer.config
    if not getattr(cfg, "compile_warmup", False) or cfg.compile_frontend == "none":
        return None
    model = trainer.model
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("warmup must run before gradient accumulation")
    device = torch.device(cfg.device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    modes = [(m, m.training) for m in model.modules()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    np_state, py_state = np.random.get_state(), random.getstate()
    shapes = frontend_warmup_shapes(cfg)
    batch = min(cfg.batch_size, cfg.micro_batch_size)
    started = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            model.train()  # Match run_batch; restore individual modes below.
            for rows, width, support in shapes:
                x = torch.randn(batch, rows, width, device=device)
                y = (torch.randn(batch, rows, device=device) if trainer.regression else
                     torch.randint(0, cfg.max_classes, (batch, rows), device=device))
                d = None if cfg.ignore_d else torch.full((batch,), width, device=device, dtype=torch.long)
                with model.no_sync() if trainer.ddp else nullcontext():
                    with trainer.amp_ctx, trainer._sdpa_ctx():
                        pred = model(x, y[:, :support], d)
                        if trainer.regression:
                            alphas = torch.linspace(0, 1, cfg.num_quantiles + 2, device=device,
                                                    dtype=pred.dtype)[1:-1].view(1, 1, -1)
                            error = y[:, support:].unsqueeze(-1) - pred
                            loss = torch.maximum(alphas * error, (alphas - 1) * error).mean()
                        else:
                            loss = F.cross_entropy(pred.flatten(end_dim=-2), y[:, support:].flatten())
                    trainer.scaler.scale(loss).backward()
                model.zero_grad(set_to_none=True)
                del x, y, pred, loss
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        for module, mode in modes:
            module.training = mode
        np.random.set_state(np_state)
        random.setstate(py_state)
    local_seconds = time.perf_counter() - started
    if trainer.ddp:
        torch.distributed.barrier()
    result = dict(rank=trainer.ddp_rank, rounds=3, local_seconds=local_seconds,
                  startup_seconds=time.perf_counter() - started, optimizer_updates=0,
                  shapes=shapes)
    print(f"Compilation warmup: {result}", flush=True)
    return result
