"""ELU kernel attention with a recomputing first-order backward.

Keep projected inputs and the small support state, rather than retaining the
FP32 ELU feature tensors and numerator produced by generic autograd. The
feature map, denominator floor and FP32 accumulation match the eager operator.
"""

import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["S"])
def _elu_state(K, V, STATE, NORMALIZER, S, D: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    bh = tl.program_id(2)
    state = tl.zeros((BM, BN), tl.float32)
    feature_sum = tl.zeros((BM,), tl.float32)
    for start in range(0, S, BK):
        row = start + tl.arange(0, BK)
        raw = tl.load(K + (bh * S + row[:, None]) * D + m[None, :],
                      (row[:, None] < S) & (m[None, :] < D), other=0.0)
        raw32 = raw.to(tl.float32)
        feature = tl.where(raw32 >= 0, raw32 + 1.0, tl.exp(raw32))
        feature = tl.where(row[:, None] < S, tl.maximum(feature, 1e-6), 0.0)
        value = tl.load(V + (bh * S + row[:, None]) * D + n[None, :],
                        (row[:, None] < S) & (n[None, :] < D), other=0.0).to(tl.float32)
        state += tl.dot(tl.trans(feature), value, input_precision="ieee")
        if tl.program_id(1) == 0:
            feature_sum += tl.sum(feature.to(tl.float32), axis=0)
    tl.store(STATE + (bh * D + m[:, None]) * D + n[None, :], state,
             (m[:, None] < D) & (n[None, :] < D))
    if tl.program_id(1) == 0:
        tl.store(NORMALIZER + bh * D + m, feature_sum, m < D)


@triton.jit(do_not_specialize=["N"])
def _elu_apply(Q, STATE, NORMALIZER, OUT, N, D: tl.constexpr,
               BT: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    bh = tl.program_id(2)
    d = tl.arange(0, BD)
    raw = tl.load(Q + (bh * N + row[:, None]) * D + d[None, :],
                  (row[:, None] < N) & (d[None, :] < D), other=0.0)
    raw32 = raw.to(tl.float32)
    feature32 = tl.maximum(tl.where(raw32 >= 0, raw32 + 1.0, tl.exp(raw32)), 1e-6)
    state = tl.load(STATE + (bh * D + d[:, None]) * D + n[None, :],
                    (d[:, None] < D) & (n[None, :] < D), other=0.0)
    numerator = tl.dot(feature32, state, input_precision="ieee")
    norm = tl.load(NORMALIZER + bh * D + d, d < D, other=0.0)
    denominator = tl.maximum(tl.sum(feature32 * norm[None, :], axis=1), 1e-6)
    tl.store(OUT + (bh * N + row[:, None]) * D + n[None, :],
             numerator / denominator[:, None],
             (row[:, None] < N) & (n[None, :] < D))


def _triton_forward(q, k, v):
    b, h, n, d = q.shape
    s = k.shape[-2]
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    state = torch.empty((b, h, d, d), device=q.device, dtype=torch.float32)
    normalizer = torch.empty((b, h, d), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    _elu_state[(triton.cdiv(d, 32), triton.cdiv(d, 32), b * h)](
        k, v, state, normalizer, s, d, 32, 32, 32, num_warps=4
    )
    _elu_apply[(triton.cdiv(n, 32), triton.cdiv(d, 32), b * h)](
        q, state, normalizer, out, n, d, 32, 32, triton.next_power_of_2(d), num_warps=4
    )
    return q, k, v, state, normalizer, out


@triton.jit(do_not_specialize=["N"])
def _elu_backward_query(Q, GRAD_OUT, STATE, NORMALIZER, DQ, GRAD_STATE,
                        GRAD_NORMALIZER, N, D: tl.constexpr, BT: tl.constexpr,
                        BD: tl.constexpr):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    bh = tl.program_id(1)
    d = tl.arange(0, BD)
    valid = (row[:, None] < N) & (d[None, :] < D)
    raw = tl.load(Q + (bh * N + row[:, None]) * D + d[None, :], valid, other=0.0).to(tl.float32)
    raw_feature = tl.where(raw >= 0, raw + 1.0, tl.exp(raw))
    feature = tl.where(valid, tl.maximum(raw_feature, 1e-6), 0.0)
    state = tl.load(STATE + (bh * D + d[:, None]) * D + d[None, :],
                    (d[:, None] < D) & (d[None, :] < D), other=0.0)
    normalizer = tl.load(NORMALIZER + bh * D + d, d < D, other=0.0)
    numerator = tl.dot(feature, state, input_precision="ieee")
    raw_den = tl.sum(feature * normalizer[None, :], axis=1)
    den = tl.maximum(raw_den, 1e-6)
    grad = tl.load(GRAD_OUT + (bh * N + row[:, None]) * D + d[None, :], valid, other=0.0).to(tl.float32)
    grad_num = grad / den[:, None]
    out = numerator / den[:, None]
    grad_den = -tl.sum(grad * out, axis=1) / den
    grad_den = tl.where((row < N) & (raw_den >= 1e-6), grad_den, 0.0)
    grad_feature = tl.dot(grad_num, tl.trans(state), input_precision="ieee")
    grad_feature += grad_den[:, None] * normalizer[None, :]
    derivative = tl.where(raw >= 0, 1.0, tl.exp(tl.minimum(raw, 0.0)))
    derivative = tl.where(raw_feature >= 1e-6, derivative, 0.0)
    tl.store(DQ + (bh * N + row[:, None]) * D + d[None, :],
             grad_feature * derivative, valid)

    partial_state = tl.dot(tl.trans(feature), grad_num, input_precision="ieee")
    tl.atomic_add(GRAD_STATE + (bh * D + d[:, None]) * D + d[None, :], partial_state,
                  (d[:, None] < D) & (d[None, :] < D))
    partial_normalizer = tl.sum(feature * grad_den[:, None], axis=0)
    tl.atomic_add(GRAD_NORMALIZER + bh * D + d, partial_normalizer, d < D)


@triton.jit(do_not_specialize=["S"])
def _elu_backward_support(K, V, GRAD_STATE, GRAD_NORMALIZER, DK, DV,
                          S, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    bh = tl.program_id(1)
    d = tl.arange(0, BD)
    valid = (row[:, None] < S) & (d[None, :] < D)
    raw = tl.load(K + (bh * S + row[:, None]) * D + d[None, :], valid, other=0.0).to(tl.float32)
    raw_feature = tl.where(raw >= 0, raw + 1.0, tl.exp(raw))
    feature = tl.where(valid, tl.maximum(raw_feature, 1e-6), 0.0)
    value = tl.load(V + (bh * S + row[:, None]) * D + d[None, :], valid, other=0.0).to(tl.float32)
    grad_state = tl.load(GRAD_STATE + (bh * D + d[:, None]) * D + d[None, :],
                         (d[:, None] < D) & (d[None, :] < D), other=0.0)
    grad_normalizer = tl.load(GRAD_NORMALIZER + bh * D + d, d < D, other=0.0)
    grad_feature = tl.dot(value, tl.trans(grad_state), input_precision="ieee")
    grad_feature += grad_normalizer[None, :]
    derivative = tl.where(raw >= 0, 1.0, tl.exp(tl.minimum(raw, 0.0)))
    derivative = tl.where(raw_feature >= 1e-6, derivative, 0.0)
    grad_value = tl.dot(feature, grad_state, input_precision="ieee")
    tl.store(DK + (bh * S + row[:, None]) * D + d[None, :], grad_feature * derivative, valid)
    tl.store(DV + (bh * S + row[:, None]) * D + d[None, :], grad_value, valid)


def _triton_backward(q, k, v, state, normalizer, grad_out):
    b, h, n, d = q.shape
    s = k.shape[-2]
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    grad_state = torch.zeros_like(state)
    grad_normalizer = torch.zeros_like(normalizer)
    block = 16 if d >= 128 else 32
    padded = triton.next_power_of_2(d)
    _elu_backward_query[(triton.cdiv(n, block), b * h)](
        q, grad_out.contiguous(), state, normalizer, dq, grad_state, grad_normalizer,
        n, d, block, padded, num_warps=4,
    )
    _elu_backward_support[(triton.cdiv(s, block), b * h)](
        k, v, grad_state, grad_normalizer, dk, dv, s, d, block, padded, num_warps=4,
    )
    return dq, dk, dv


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
            if q.is_cuda and q.dtype in (torch.bfloat16, torch.float16) and q.shape[-1] <= 128:
                q, k, v, state, normalizer, out = _triton_forward(q, k, v)
            else:
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
            if (q.is_cuda and q.dtype in (torch.bfloat16, torch.float16)
                    and q.shape[-1] == v.shape[-1] and q.shape[-1] <= 128):
                return _triton_backward(q, k, v, state, normalizer, grad_out)
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
