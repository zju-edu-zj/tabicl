"""ELU kernel attention with a recomputing first-order backward.

Keep projected inputs and the small support state, rather than retaining the
FP32 ELU feature tensors and numerator produced by generic autograd. The
feature map, denominator floor and FP32 accumulation match the eager operator.
"""

import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


def _features(x):
    return (F.elu(x.float()) + 1.0).clamp_min(1e-6)


def _feature_grad(x, upstream):
    xf = x.float()
    derivative = torch.where(xf >= 0, 1.0, torch.exp(xf.clamp_max(0)))
    active = F.elu(xf) + 1.0 >= 1e-6
    return (upstream * derivative * active).to(x.dtype)


class _ELUAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        with torch.autocast(device_type=q.device.type, enabled=False):
            # Cap FP32 workspaces on long tables. The state remains shared by
            # all queries; neither forward nor backward forms an N*S matrix.
            max_rows = max(q.shape[-2], k.shape[-2])
            ctx.chunk_size = 8192 if max_rows > 16384 else max(1, max_rows)
            state = torch.zeros((*k.shape[:2], k.shape[-1], v.shape[-1]), device=q.device, dtype=torch.float32)
            normalizer = torch.zeros((*k.shape[:2], k.shape[-1]), device=q.device, dtype=torch.float32)
            for start in range(0, k.shape[-2], ctx.chunk_size):
                pk = _features(k[:, :, start:start + ctx.chunk_size])
                state.add_(pk.transpose(-1, -2) @ v[:, :, start:start + ctx.chunk_size].float())
                normalizer.add_(pk.sum(-2))
            out = torch.empty((*q.shape[:-1], v.shape[-1]), device=q.device, dtype=q.dtype)
            for start in range(0, q.shape[-2], ctx.chunk_size):
                pq = _features(q[:, :, start:start + ctx.chunk_size])
                den = (pq * normalizer.unsqueeze(-2)).sum(-1, keepdim=True).clamp_min(1e-6)
                out[:, :, start:start + ctx.chunk_size] = (pq @ state / den).to(q.dtype)
        ctx.save_for_backward(q, k, v, state, normalizer)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        q, k, v, state, normalizer = ctx.saved_tensors
        with torch.autocast(device_type=q.device.type, enabled=False):
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            grad_state, grad_normalizer = torch.zeros_like(state), torch.zeros_like(normalizer)
            for start in range(0, q.shape[-2], ctx.chunk_size):
                qc = q[:, :, start:start + ctx.chunk_size]
                pq = _features(qc)
                raw_den = (pq * normalizer.unsqueeze(-2)).sum(-1, keepdim=True)
                den = raw_den.clamp_min(1e-6)
                out = pq @ state / den
                grad = grad_out[:, :, start:start + ctx.chunk_size].float()
                grad_num = grad / den
                grad_den = -(grad * out).sum(-1, keepdim=True) / den * (raw_den >= 1e-6)
                grad_state.add_(pq.transpose(-1, -2) @ grad_num)
                grad_normalizer.add_((pq * grad_den).sum(-2))
                grad_pq = grad_num @ state.transpose(-1, -2) + grad_den * normalizer.unsqueeze(-2)
                dq[:, :, start:start + ctx.chunk_size] = _feature_grad(qc, grad_pq)
            # Free query workspaces before processing support chunks.
            if q.shape[-2]:
                del pq, out, grad, grad_num, grad_den, grad_pq, raw_den, den
            for start in range(0, k.shape[-2], ctx.chunk_size):
                kc, vc = k[:, :, start:start + ctx.chunk_size], v[:, :, start:start + ctx.chunk_size]
                pk = _features(kc)
                grad_pk = vc.float() @ grad_state.transpose(-1, -2) + grad_normalizer.unsqueeze(-2)
                dk[:, :, start:start + ctx.chunk_size] = _feature_grad(kc, grad_pk)
                dv[:, :, start:start + ctx.chunk_size] = (pk @ grad_state).to(v.dtype)
        return dq, dk, dv


def elu_kernel_attention(q, k, v):
    return _ELUAttention.apply(q, k, v)
