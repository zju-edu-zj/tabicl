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


@triton.jit(do_not_specialize=["S", "SPLITS", "CHUNK"])
def _elu_state(K, V, STATE, NORMALIZER, S, D: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               SPLITS, CHUNK):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    bh = tl.program_id(2) // SPLITS
    split = tl.program_id(2) % SPLITS
    state_offset = (split * tl.num_programs(2) // SPLITS + bh) * D
    state = tl.zeros((BM, BN), tl.float32)
    feature_sum = tl.zeros((BM,), tl.float32)
    for start in range(0, CHUNK, BK):
        row = split * CHUNK + start + tl.arange(0, BK)
        raw = tl.load(K + (bh * S + row[:, None]) * D + m[None, :],
                      (row[:, None] < S) & (m[None, :] < D), other=0.0)
        raw32 = raw.to(tl.float32)
        feature = tl.where(raw32 >= 0, raw32 + 1.0, tl.exp(raw32))
        feature = tl.where(row[:, None] < S, tl.maximum(feature, 1e-6), 0.0)
        value = tl.load(V + (bh * S + row[:, None]) * D + n[None, :],
                        (row[:, None] < S) & (n[None, :] < D), other=0.0).to(tl.float32)
        state += tl.dot(tl.trans(feature), value, input_precision="tf32x3")
        if tl.program_id(1) == 0:
            feature_sum += tl.sum(feature.to(tl.float32), axis=0)
    tl.store(STATE + (state_offset + m[:, None]) * D + n[None, :], state,
             (m[:, None] < D) & (n[None, :] < D))
    if tl.program_id(1) == 0:
        tl.store(NORMALIZER + state_offset + m, feature_sum, m < D)


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
    numerator = tl.dot(feature32, state, input_precision="tf32x3")
    norm = tl.load(NORMALIZER + bh * D + d, d < D, other=0.0)
    denominator = tl.maximum(tl.sum(feature32 * norm[None, :], axis=1), 1e-6)
    tl.store(OUT + (bh * N + row[:, None]) * D + n[None, :],
             numerator / denominator[:, None],
             (row[:, None] < N) & (n[None, :] < D))


def _triton_state(k, v):
    b, h, s, d = k.shape
    k, v = k.contiguous(), v.contiguous()
    # Bound private states to 32 MiB, including large micro-batches.
    max_splits = max(1, (32 * 1024**2) // (b * h * (d * d + d) * 4))
    splits = min(triton.cdiv(s, 256), max_splits)
    chunk = triton.cdiv(triton.cdiv(s, splits), 32) * 32
    states = torch.empty((splits, b, h, d, d), device=k.device, dtype=torch.float32)
    normalizers = torch.empty((splits, b, h, d), device=k.device, dtype=torch.float32)
    _elu_state[(triton.cdiv(d, 32), triton.cdiv(d, 32), b * h * splits)](
        k, v, states, normalizers, s, d, 32, 32, 32, splits, chunk, num_warps=4, num_stages=1
    )
    state = states.sum(0) if splits > 1 else states[0]
    normalizer = normalizers.sum(0) if splits > 1 else normalizers[0]
    return k, v, state, normalizer


def _triton_forward(q, k, v):
    b, h, n, d = q.shape
    q = q.contiguous()
    k, v, state, normalizer = _triton_state(k, v)
    out = torch.empty_like(q)
    _elu_apply[(triton.cdiv(n, 32), triton.cdiv(d, 32), b * h)](
        q, state, normalizer, out, n, d, 32, 32, triton.next_power_of_2(d), num_warps=4
    )
    return q, k, v, state, normalizer, out


def prepare_elu_state(k, v):
    """Inference-only derived support state; the caller owns its lifetime."""
    with torch.autocast(device_type=k.device.type, enabled=False):
        if k.is_cuda and k.shape[-1] == v.shape[-1] and 16 <= k.shape[-1] <= 128:
            return _triton_state(k, v)[2:]
        feature = _features(k)
        return feature.transpose(-1, -2) @ v.float(), feature.sum(-2)


def apply_elu_state(q, state, normalizer):
    with torch.autocast(device_type=q.device.type, enabled=False):
        b, h, n, d = q.shape
        if q.is_cuda and state.shape[-1] == d and 16 <= d <= 128:
            out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
            _elu_apply[(triton.cdiv(n, 32), triton.cdiv(d, 32), b * h)](
                q.contiguous(), state, normalizer, out, n, d, 32, 32,
                triton.next_power_of_2(d), num_warps=4,
            )
            return out
        feature = _features(q)
        den = (feature * normalizer.unsqueeze(-2)).sum(-1, keepdim=True).clamp_min(1e-6)
        return (feature @ state / den).to(q.dtype)


@triton.jit(do_not_specialize=["N", "CHUNK"])
def _elu_backward_query(Q, GRAD_OUT, STATE, NORMALIZER, DQ, GRAD_STATE,
                        GRAD_NORMALIZER, N, D: tl.constexpr, BT: tl.constexpr,
                        BD: tl.constexpr, CHUNK):
    split = tl.program_id(0)
    bh = tl.program_id(1)
    d = tl.arange(0, BD)
    state = tl.load(STATE + (bh * D + d[:, None]) * D + d[None, :],
                    (d[:, None] < D) & (d[None, :] < D), other=0.0)
    normalizer = tl.load(NORMALIZER + bh * D + d, d < D, other=0.0)
    partial_state = tl.zeros((BD, BD), tl.float32)
    partial_normalizer = tl.zeros((BD,), tl.float32)
    for start in range(0, CHUNK, BT):
        row = split * CHUNK + start + tl.arange(0, BT)
        valid = (row[:, None] < N) & (d[None, :] < D)
        raw = tl.load(Q + (bh * N + row[:, None]) * D + d[None, :], valid, other=0.0).to(tl.float32)
        raw_feature = tl.where(raw >= 0, raw + 1.0, tl.exp(raw))
        feature = tl.where(valid, tl.maximum(raw_feature, 1e-6), 0.0)
        precision: tl.constexpr = "ieee" if D >= 128 else "tf32x3"
        numerator = tl.dot(feature, state, input_precision=precision)
        raw_den = tl.sum(feature * normalizer[None, :], axis=1)
        den = tl.maximum(raw_den, 1e-6)
        grad = tl.load(GRAD_OUT + (bh * N + row[:, None]) * D + d[None, :], valid, other=0.0).to(tl.float32)
        grad_num = grad / den[:, None]
        out = numerator / den[:, None]
        grad_den = -tl.sum(grad * out, axis=1) / den
        grad_den = tl.where((row < N) & (raw_den >= 1e-6), grad_den, 0.0)
        grad_feature = tl.dot(grad_num, tl.trans(state), input_precision=precision)
        grad_feature += grad_den[:, None] * normalizer[None, :]
        derivative = tl.where(raw >= 0, 1.0, tl.exp(tl.minimum(raw, 0.0)))
        derivative = tl.where(raw_feature >= 1e-6, derivative, 0.0)
        tl.store(DQ + (bh * N + row[:, None]) * D + d[None, :],
                 grad_feature * derivative, valid)
        partial_state += tl.dot(tl.trans(feature), grad_num, input_precision=precision)
        partial_normalizer += tl.sum(feature * grad_den[:, None], axis=0)
    offset = (split * tl.num_programs(1) + bh) * D
    tl.store(GRAD_STATE + (offset + d[:, None]) * D + d[None, :], partial_state,
             (d[:, None] < D) & (d[None, :] < D))
    tl.store(GRAD_NORMALIZER + offset + d, partial_normalizer, d < D)


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
    precision: tl.constexpr = "ieee" if D >= 128 else "tf32x3"
    grad_feature = tl.dot(value, tl.trans(grad_state), input_precision=precision)
    grad_feature += grad_normalizer[None, :]
    derivative = tl.where(raw >= 0, 1.0, tl.exp(tl.minimum(raw, 0.0)))
    derivative = tl.where(raw_feature >= 1e-6, derivative, 0.0)
    grad_value = tl.dot(feature, grad_state, input_precision=precision)
    tl.store(DK + (bh * S + row[:, None]) * D + d[None, :], grad_feature * derivative, valid)
    tl.store(DV + (bh * S + row[:, None]) * D + d[None, :], grad_value, valid)


def _triton_backward(q, k, v, state, normalizer, grad_out):
    b, h, n, d = q.shape
    s = k.shape[-2]
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    block = 16 if d >= 128 else 32
    max_splits = max(1, min(128, (32 * 1024**2) // (b * h * (d * d + d) * 4)))
    splits = min(triton.cdiv(n, 256), max_splits)
    chunk = triton.cdiv(triton.cdiv(n, splits), block) * block
    partial_state = torch.empty((splits, *state.shape), device=q.device, dtype=torch.float32)
    partial_normalizer = torch.empty((splits, *normalizer.shape), device=q.device, dtype=torch.float32)
    padded = triton.next_power_of_2(d)
    _elu_backward_query[(splits, b * h)](
        q, grad_out.contiguous(), state, normalizer, dq, partial_state, partial_normalizer,
        n, d, block, padded, chunk, num_warps=4, num_stages=1,
    )
    grad_state = partial_state.sum(0) if splits > 1 else partial_state[0]
    grad_normalizer = partial_normalizer.sum(0) if splits > 1 else partial_normalizer[0]
    _elu_backward_support[(triton.cdiv(s, block), b * h)](
        k, v, grad_state, grad_normalizer, dk, dv, s, d, block, padded, num_warps=4, num_stages=1,
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
            if (q.is_cuda and q.dtype in (torch.float32, torch.bfloat16, torch.float16)
                    and q.shape[-1] == v.shape[-1] and 16 <= q.shape[-1] <= 128):
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
            if (q.is_cuda and q.dtype in (torch.float32, torch.bfloat16, torch.float16)
                    and q.shape[-1] == v.shape[-1] and 16 <= q.shape[-1] <= 128):
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
