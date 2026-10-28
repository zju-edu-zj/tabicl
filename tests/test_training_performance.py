"""Numerical and checkpoint contracts for whole-model training acceleration."""
from copy import deepcopy
from types import MethodType

import pytest
import torch
import torch.nn.functional as F

from tabicl._model.layers import SkippableLinear
from tabicl._model.tabicl import TabICL
from tabicl._model.ssmax import _logn


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def model(**kwargs):
    config = dict(embed_dim=16, col_num_blocks=1, col_nhead=2, col_num_inds=4,
                  col_feature_group="same", col_target_aware=True, col_ssmax=True,
                  row_num_blocks=2, row_nhead=2, row_num_cls=2, icl_num_blocks=3,
                  icl_nhead=2, icl_attention="global_sparse", icl_block_size=16,
                  icl_num_selected_blocks=1, icl_qk_norm=True, icl_ssmax=False,
                  zero_init=False, num_quantiles=9)
    return TabICL(**dict(config, **kwargs)).train()


def legacy_column(candidate):
    """Reference: encode labels for every column and process all CLS slots."""
    col = candidate.col_embedder
    compute = col._compute_embeddings
    forward = col.forward

    def expanded(self, features, train_size, y_train=None, embed_with_test=False):
        if self.target_aware and y_train is not None and features.ndim == 4:
            y_train = y_train.expand(features.shape[0], features.shape[1], -1)
        return compute(features, train_size, y_train, embed_with_test)

    def with_reserved(self, *args, **kwargs):
        kwargs["skip_reserved"] = False
        return forward(*args, **kwargs)

    col._compute_embeddings = MethodType(expanded, col)
    col.forward = MethodType(with_reserved, col)


def assert_gradients(actual, expected, rtol=2e-4, atol=2e-6):
    left, right = dict(actual.named_parameters()), dict(expected.named_parameters())
    assert left.keys() == right.keys()
    for name in left:
        a, b = left[name].grad, right[name].grad
        assert (a is None) == (b is None), name
        if a is not None:
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol, msg=name)


@pytest.mark.parametrize("group,affine", [(False, False), (False, True), ("same", False), ("valid", False)])
@pytest.mark.parametrize("target_aware,max_classes", [(False, 10), (True, 10), (True, 0)])
@pytest.mark.parametrize("recompute", [False, True])
def test_whole_model_reserved_slots_and_label_broadcast_match_legacy(group, affine, target_aware, max_classes, recompute):
    torch.manual_seed(121)
    actual = model(col_feature_group=group, col_affine=affine, col_target_aware=target_aware,
                   max_classes=max_classes, recompute=recompute)
    expected = deepcopy(actual)
    legacy_column(expected)
    x = torch.randn(2, 35, 7)
    x[0, :, -1] = -100  # Exercise sentinel columns, including target-aware activation.
    y = torch.randint(0, 3, (2, 23)).float() if max_classes else torch.randn(2, 23)
    a, b = actual(x, y), expected(x, y)
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    a.square().mean().backward()
    b.square().mean().backward()
    assert_gradients(actual, expected)


@pytest.mark.parametrize("dropout,ragged,frozen", [(0.2, False, False), (0, True, False), (0, False, True)])
def test_reserved_slot_fallback_and_frozen_column(dropout, ragged, frozen):
    torch.manual_seed(44)
    actual = model(col_feature_group=False, dropout=dropout,
                   icl_attention="dense" if dropout else "global_sparse")
    if frozen:
        actual.col_embedder.requires_grad_(False)
    expected = deepcopy(actual)
    legacy_column(expected)
    x, y = torch.randn(2, 35, 7), torch.randint(0, 3, (2, 23)).float()
    d = torch.tensor([4, 7]) if ragged else None
    torch.manual_seed(45)
    a = actual(x, y, d)
    torch.manual_seed(45)
    b = expected(x, y, d)
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    a.sum().backward()
    b.sum().backward()
    assert_gradients(actual, expected)


@pytest.mark.parametrize("classes", [3, 31])
def test_column_public_output_and_many_class_label_gradients(classes):
    torch.manual_seed(51)
    actual = model()
    expected = deepcopy(actual)
    legacy_column(expected)
    x, y = torch.randn(2, 17, 7), torch.randint(0, classes, (2, 11)).float()
    a, b = actual.col_embedder(x, y), expected.col_embedder(x, y)
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    a.square().mean().backward()
    b.square().mean().backward()
    assert_gradients(actual, expected)


def test_skippable_linear_matches_masked_reference_output_and_gradients():
    torch.manual_seed(11)
    actual = SkippableLinear(3, 16).double()
    expected = deepcopy(actual)
    x = torch.randn(2, 4, 3, dtype=torch.float64)
    x[0, 1] = -100
    a_input, b_input = x.requires_grad_(), x.clone().detach().requires_grad_()
    a = actual(a_input)
    b = F.linear(b_input, expected.weight, expected.bias)
    b[(b_input == -100).all(-1)] = -100
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().sum().backward()
    b.square().sum().backward()
    assert_gradients(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(a_input.grad, b_input.grad, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("recompute", [False, True])
def test_dynamic_frontend_compile_outputs_gradients_and_checkpoint_keys(dtype, recompute):
    torch.manual_seed(81)
    # Isolate compiler equivalence from the discontinuous sparse top-k router.
    # Sparse full-model backward/resume is exercised by the trainer tests.
    eager = model(recompute=recompute, icl_attention="dense").cuda()
    compiled = deepcopy(eager)
    names = list(compiled.state_dict())
    parameter_ids = [id(p) for p in compiled.parameters()]
    ddp_compiler_setting = torch._dynamo.config.optimize_ddp
    compiled.compile_frontend("all")
    assert list(compiled.state_dict()) == names
    assert [id(p) for p in compiled.parameters()] == parameter_ids
    for batch, rows, features, support in [(1, 35, 7, 23), (2, 67, 10, 41), (1, 129, 100, 77)]:
        x = torch.randn(batch, rows, features, device="cuda")
        y = torch.randint(0, 3, (batch, support), device="cuda").float()
        eager.zero_grad(set_to_none=True)
        compiled.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            a, b = compiled(x, y), eager(x, y)
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a.float(), b.float(), rtol=0.05 if dtype == torch.bfloat16 else 0.002,
                                   atol=0.005 if dtype == torch.bfloat16 else 0.0002)
        a.float().square().mean().backward()
        b.float().square().mean().backward()
        assert torch._dynamo.config.optimize_ddp == ddp_compiler_setting
        grads_a, grads_b = [], []
        for (name, p), (_, q) in zip(compiled.named_parameters(), eager.named_parameters()):
            assert (p.grad is None) == (q.grad is None), name
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name
                grads_a.append(p.grad.flatten().float())
                grads_b.append(q.grad.flatten().float())
        ga, gb = torch.cat(grads_a), torch.cat(grads_b)
        assert (ga - gb).norm() / gb.norm() < (0.025 if dtype == torch.bfloat16 else 0.002)
    restored = model(recompute=recompute, icl_attention="dense").cuda()
    restored.load_state_dict(compiled.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="already"):
        compiled.compile_frontend("row")


def test_compilation_rejects_unknown_mode_and_older_torch(monkeypatch):
    candidate = model()
    with pytest.raises(ValueError):
        candidate.compile_frontend("invalid")
    monkeypatch.setattr(torch, "__version__", "2.5.1+cu121")
    candidate.compile_frontend("none")
    with pytest.raises(RuntimeError, match="2.8"):
        candidate.compile_frontend("all")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_ssmax_log_length_is_dynamic_without_specialization(dtype):
    from torch._dynamo.testing import CompileCounter

    counter = CompileCounter()

    def scaling(x):
        return _logn(x.shape[-1], x.device, x.dtype)

    compiled = torch.compile(scaling, dynamic=True, fullgraph=True, backend=counter)
    for length in (7, 11, 19, 23, 37, 53, 67, 89, 113, 131, 173, 257):
        x = torch.empty(2, length, dtype=dtype)
        torch.testing.assert_close(compiled(x), scaling(x), rtol=1e-6, atol=0)
    assert counter.frame_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_column_many_support_sizes_do_not_exhaust_graph_cache(dtype):
    torch.manual_seed(89)
    candidate = model().cuda()
    candidate.compile_frontend("all")
    # More distinct supports than Dynamo's default cache permits: numerical
    # tests with just two shapes cannot catch SymFloat length specialization.
    with torch._dynamo.config.patch(fail_on_recompile_limit_hit=True):
        for index, support in enumerate((11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)):
            x = torch.randn(2, 67, 7 + index % 3, device="cuda")
            y = torch.randint(0, 3, (2, support), device="cuda").float()
            candidate.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                loss = candidate.col_embedder(x, y, skip_reserved=True).float().square().mean()
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.distributed.is_nccl_available(),
                    reason="CUDA/NCCL required")
def test_compiled_components_inside_ddp_backward_and_config_restoration(tmp_path):
    # A real DDP forward activates Dynamo's DDP graph splitter, unlike a raw
    # module test. Exercise RoPE scalar outputs and gradient accumulation.
    from torch.nn.parallel import DistributedDataParallel

    torch.distributed.init_process_group("nccl", init_method=(tmp_path / "rendezvous").as_uri(),
                                         rank=0, world_size=1)
    previous = torch._dynamo.config.optimize_ddp
    try:
        core = model(embed_dim=32, col_num_inds=8, row_num_blocks=3,
                     row_rope_interleaved=False).cuda()
        names = list(core.state_dict())
        core.compile_frontend("all")
        wrapped = DistributedDataParallel(core, device_ids=[torch.cuda.current_device()],
                                          broadcast_buffers=False)
        for rows, support in [(35, 23), (67, 41)]:
            x = torch.randn(2, rows, 10, device="cuda")
            y = torch.randint(0, 3, (2, support), device="cuda").float()
            for index in range(2):
                wrapped.require_backward_grad_sync = index == 1
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = wrapped(x, y).float().square().mean()
                loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in core.parameters() if p.grad is not None)
            wrapped.zero_grad(set_to_none=True)
        assert list(core.state_dict()) == names
        assert torch._dynamo.config.optimize_ddp == previous
    finally:
        torch.distributed.destroy_process_group()
