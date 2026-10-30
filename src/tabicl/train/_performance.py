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
    rows = cfg.max_seq_len
    low = int(cfg.min_train_size * rows) if cfg.min_train_size < 1 else int(cfg.min_train_size)
    high = int(cfg.max_train_size * rows) if cfg.max_train_size < 1 else int(cfg.max_train_size)
    low, high = max(1, low), min(rows - 1, high)
    if low > high:
        raise ValueError("invalid warmup support interval")
    batch = min(cfg.batch_size, cfg.micro_batch_size)
    widths = (cfg.max_features, cfg.min_features, max(cfg.min_features, int(.37 * cfg.max_features)))
    started = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            model.train()  # Match run_batch; restore individual modes below.
            for width, support in zip(widths, ((low + high) // 2, low, high)):
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
                  startup_seconds=time.perf_counter() - started, optimizer_updates=0)
    print(f"Compilation warmup: {result}", flush=True)
    return result
