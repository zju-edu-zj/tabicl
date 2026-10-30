from contextlib import nullcontext
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from tabicl._model.tabicl import TabICL
from tabicl._model.attention import sdpa_with_flattened_batch
from tabicl.train._performance import resolve_attention_backend, warmup_frontend


@pytest.mark.parametrize("dtype,rows,name,expected", [
    ("float32", 2048, "PPU-ZW810", "fp32_math"),
    ("float32", 4096, "PPU-ZW810", "fp32_row_math"),
    ("bfloat16", 2048, "PPU-ZW810", "native"),
    ("float32", 2048, "NVIDIA H100", "native"),
])
def test_auto_backend_is_hardware_and_precision_specific(dtype, rows, name, expected):
    assert resolve_attention_backend("auto", dtype, rows, name) == expected
    assert resolve_attention_backend("native", dtype, rows, name) == "native"


@pytest.mark.parametrize("regression", [False, True])
def test_warmup_restores_state_rng_and_has_no_updates(regression):
    torch.set_num_threads(2)
    model = TabICL(max_classes=0 if regression else 3, num_quantiles=5, embed_dim=16,
                   col_num_blocks=1, col_nhead=2, col_num_inds=4, row_num_blocks=1,
                   row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2,
                   zero_init=False).train()
    model.col_embedder.eval()  # Frozen-component modes must survive warmup.
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    np_state, py_state = np.random.get_state(), random.getstate()
    cfg = SimpleNamespace(compile_warmup=True, compile_frontend="all", device="cpu",
                          max_seq_len=24, min_train_size=.3, max_train_size=.9,
                          batch_size=2, micro_batch_size=2, max_features=7, min_features=1,
                          max_classes=3, num_quantiles=5, ignore_d=True)
    trainer = SimpleNamespace(config=cfg, model=model, regression=regression, ddp=False,
                              ddp_rank=0, amp_ctx=nullcontext(), _sdpa_ctx=nullcontext,
                              scaler=torch.amp.GradScaler("cpu", enabled=False))
    result = warmup_frontend(trainer)
    assert result["rounds"] == 3 and result["optimizer_updates"] == 0
    assert torch.equal(rng, torch.get_rng_state())
    assert np.array_equal(np_state[1], np.random.get_state()[1])
    assert py_state == random.getstate()
    assert not model.col_embedder.training
    assert all(p.grad is None for p in model.parameters())
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, state[k], rtol=0, atol=0)


def test_row_math_preserves_output_and_gradient_without_dtype_conversion():
    q = torch.randn(2, 2, 4, 8, requires_grad=True)
    k = torch.randn(2, 2, 9, 8, requires_grad=True)
    v = torch.randn(2, 2, 9, 8, requires_grad=True)
    a = sdpa_with_flattened_batch(q, k, v, fp32_math=True)
    b = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(a, b)
    for left, right in zip(torch.autograd.grad(a.square().sum(), (q,k,v), retain_graph=True),
                           torch.autograd.grad(b.square().sum(), (q,k,v))):
        torch.testing.assert_close(left, right)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA SDPA kernels required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_row_policy_preserves_precision_outputs_and_gradients(dtype, monkeypatch):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from tabicl._model import attention
    monkeypatch.setattr(attention, "_use_flash_attn3", False)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unsupported")
    torch.manual_seed(17)
    inputs = [torch.randn(3, 4, rows, 16, device="cuda", dtype=dtype, requires_grad=True)
              for rows in (17, 31, 31)]
    # Native vendor dispatch excludes cuDNN in Trainer as well.
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        expected = sdpa_with_flattened_batch(*inputs)
        actual = sdpa_with_flattened_batch(*inputs, fp32_math=True)
        assert actual.dtype == dtype
        tolerance = 3e-4 if dtype == torch.float32 else .02
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        grads_a = torch.autograd.grad(actual.float().square().mean(), inputs, retain_graph=True)
        grads_b = torch.autograd.grad(expected.float().square().mean(), inputs)
        for a, b in zip(grads_a, grads_b):
            torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
