# taken from https://github.com/fabian-sp/sda/blob/e2f95648ffdaf937adb6d68340b5e2dc9cf7e8a6/sda/optim/muon.py
""" Muon optimizer

Original code copied from:
    https://github.com/toothacher17/Megatron-LM/blob/moonshot/distributedmuon-impl/megatron/core/optimizer/muon.py

Then we apply the following changes:
    * Remove everything related to distributed training (as we run on single GPU)
    * Reshape all weights to 2D, before Newton-Schulz, and then reshape back.
    (ref: https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench94_muon.py)
"""
import math

import torch


# copy from https://github.com/KellerJordan/Muon/tree/master
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    if G.ndim not in (2, 3):
        raise ValueError("Newton-Schulz expects one matrix or a batch of matrices")
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.transpose(-2, -1)

    # Frobenius-normalize every matrix independently.  This also makes the
    # batched path exactly the same algorithm as the former per-parameter path.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if transposed:
        X = X.transpose(-2, -1)
    return X


# adjust LR based on: https://github.com/MoonshotAI/Moonlight
def adjust_lr_wd_for_muon(lr, matched_adamw_rms, param_shape):
    A, B = param_shape  # assumes 2D shape given (!)
    adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


# copy from https://github.com/KellerJordan/Muon/tree/master
class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Routing is controlled purely by the per-group ``use_muon`` flag, which is the caller's
    responsibility: groups with ``use_muon=True`` are optimized with Muon (each parameter is
    reshaped to 2D before the Newton-Schulz step), and any other groups use an internal AdamW.
    The trainer routes only transformer projection matrices to Muon; norms, biases,
    embeddings, slots, gates, SSMax networks and output heads use the AdamW groups.

    Arguments:
        param_groups: The parameters to be optimized.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        matched_adamw_rms: The AdamW Update RMS that Muon is designed to match. (0.2~0.4 recommended)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (5 is probably always enough)
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(self,
                 param_groups,
                 lr=2e-2,
                 weight_decay=0.1,
                 matched_adamw_rms=0.2,
                 momentum=0.95,
                 nesterov=True,
                 ns_steps=5,
                 adamw_betas=(0.95, 0.95),
                 adamw_eps=1e-8,
                 use_cautious_wd: bool = False,
                 compile_ns: bool = False):

        defaults = dict(lr=lr, weight_decay=weight_decay,
                        matched_adamw_rms=matched_adamw_rms,
                        momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_betas=adamw_betas, adamw_eps=adamw_eps, use_cautious_wd=use_cautious_wd)

        super().__init__(param_groups, defaults)
        self._zeropower = zeropower_via_newtonschulz5
        if compile_ns:
            version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
            if version < (2, 8):
                raise RuntimeError("Compiled Muon requires PyTorch >= 2.8")
            # Bucket sizes can change when a parameter has no gradient. Share
            # dynamic graphs across matrix shapes instead of exhausting Dynamo's
            # specialization cache. Keep state/step counters eager.
            self._zeropower = torch.compile(zeropower_via_newtonschulz5, fullgraph=True, dynamic=True)

    @torch.no_grad()
    def step(self):
        # update muon momentum first
        for group in self.param_groups:

            if not group.get('use_muon', False):
                continue

            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            use_cautious_wd = group["use_cautious_wd"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            matched_adamw_rms = group["matched_adamw_rms"]
            active = [p for p in group["params"] if p.grad is not None]
            if not active:
                continue

            grads = [p.grad for p in active]
            buffers = []
            for p, g in zip(active, grads):
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buffers.append(state["momentum_buffer"])
            torch._foreach_mul_(buffers, momentum)
            torch._foreach_add_(buffers, grads)

            # Stack equal matrix shapes so the five Newton-Schulz iterations use
            # batched GEMMs. Different transformer projections remain separate
            # buckets, and optimizer state stays per parameter/checkpoint-compatible.
            buckets = {}
            for p, g, buf in zip(active, grads, buffers):
                matrix_shape = (len(g), g.numel() // len(g))
                key = (g.device, g.dtype, matrix_shape)
                buckets.setdefault(key, []).append((p, g, buf))
            for (_, _, matrix_shape), entries in buckets.items():
                if nesterov:
                    ns_input = torch.stack([g.add(buf, alpha=momentum).reshape(matrix_shape)
                                            for _, g, buf in entries])
                else:
                    ns_input = torch.stack([buf.reshape(matrix_shape) for _, _, buf in entries])
                updates = self._zeropower(ns_input, steps=ns_steps).unbind(0)
                params = [p for p, _, _ in entries]
                if use_cautious_wd:
                    for p, update in zip(params, updates):
                        shaped = update.view_as(p)
                        p.add_(p * (shaped * p >= 0), alpha=-lr * weight_decay)
                elif weight_decay:
                    torch._foreach_mul_(params, 1 - lr * weight_decay)
                adjusted_lr = adjust_lr_wd_for_muon(lr, matched_adamw_rms, matrix_shape)
                torch._foreach_add_(params, [u.view_as(p) for p, u in zip(params, updates)],
                                    alpha=-adjusted_lr)

        # use adam for other params
        for group in self.param_groups:

            if group.get('use_muon', False):
                continue

            params = group["params"]
            lr = group['lr']
            weight_decay = group['weight_decay']
            use_cautious_wd = group['use_cautious_wd']
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']

            if not use_cautious_wd:
                # Most auxiliary parameters are small. Launching every moment,
                # bias correction and update separately costs more than their
                # arithmetic. Bucket by each parameter's own clock so missing
                # gradients and resumed optimizer states keep AdamW semantics.
                buckets = {}
                for p in params:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state['adamw_exp_avg'] = torch.zeros_like(p.grad)
                        state['adamw_exp_avg_sq'] = torch.zeros_like(p.grad)
                        state['step'] = 0
                    state['step'] += 1
                    key = (p.device, p.dtype, state['step'])
                    buckets.setdefault(key, []).append((p, p.grad, state))
                for (_, _, step), entries in buckets.items():
                    active_params = [p for p, _, _ in entries]
                    grads = [g for _, g, _ in entries]
                    averages = [s['adamw_exp_avg'] for _, _, s in entries]
                    squares = [s['adamw_exp_avg_sq'] for _, _, s in entries]
                    torch._foreach_lerp_(averages, grads, 1 - beta1)
                    torch._foreach_lerp_(squares, torch._foreach_mul(grads, grads), 1 - beta2)
                    corrected = torch._foreach_div(averages, 1 - beta1 ** step)
                    denominator = torch._foreach_sqrt(squares)
                    torch._foreach_div_(denominator, (1 - beta2 ** step) ** 0.5)
                    torch._foreach_add_(denominator, eps)
                    updates = torch._foreach_div(corrected, denominator)
                    torch._foreach_mul_(active_params, 1 - lr * weight_decay)
                    torch._foreach_add_(active_params, updates, alpha=-lr)
                continue

            for p in params:

                g = p.grad
                if g is None:  # parameter unused this step; skip like AdamW does
                    continue
                state = self.state[p]

                if len(state) == 0:
                    state['adamw_exp_avg'] = torch.zeros_like(g)
                    state['adamw_exp_avg_sq'] = torch.zeros_like(g)
                    state['step'] = 0

                # Like torch AdamW, advance a parameter's clock only when it has
                # a gradient (important for conditionally used task/SSMax paths).
                state['step'] += 1
                step = state['step']

                buf1 = state['adamw_exp_avg']
                buf2 = state['adamw_exp_avg_sq']
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                g = (buf1 / bias_correction1) / (buf2.sqrt() / bias_correction2 ** 0.5 + eps)
                # apply weight decay
                if use_cautious_wd:
                    wd_mask = (g * p >= 0)
                    p.data.add_(p * wd_mask, alpha=-lr * weight_decay)
                else:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr)
