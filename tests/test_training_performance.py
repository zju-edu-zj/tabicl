"""Numerical and checkpoint contracts for whole-model training acceleration."""
from copy import deepcopy
from types import MethodType

import pytest
import torch
import torch.nn.functional as F

from tabicl._model.layers import InducedSelfAttentionBlock, SkippableLinear
from tabicl._model.tabicl import TabICL
from tabicl._model.ssmax import _logn
from tabicl._model.rope import RotaryEmbedding


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    # Tests intentionally create many different architectures / freeze modes.
    # Dynamo's cache is per code object, so isolate these independent runs.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
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
    float_compiler_setting = torch._dynamo.config.specialize_float
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
        assert torch._dynamo.config.specialize_float == float_compiler_setting
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


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_frontend_random_widths_classes_and_label_strides_have_bounded_graphs(monkeypatch, interleaved, recompute):
    """Stress Dynamo guards independently of GPU fusion/numerical tests above."""
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    counter = CompileCounter()
    original_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda *a, **kw: original_compile(*a, **kw, backend=counter))
    candidate = model(icl_attention="dense", row_rope_interleaved=interleaved, recompute=recompute)
    candidate.compile_frontend("all")
    # H=1 is legitimately specialized by Torch. Subsequent changes to H,
    # support and the *value* of max(y) must not add graphs or fall back eager.
    widths = [26, 1, 2, 4, 8, 13, 30, 64, 100] + list(range(1, 101))
    stabilized = None
    with torch._dynamo.config.patch(fail_on_recompile_limit_hit=True):
        for index, width in enumerate(widths):
            support = 5 + index * 7 % 23
            x = torch.randn(2, 35, width)
            # Trainer slices full labels; this stride differs from rand(B,S).
            y = torch.randint(1 + index % 10, (2, 35)).float()[:, :support]
            candidate.zero_grad(set_to_none=True)
            candidate(x, y).square().mean().backward()
            if index == 8:
                stabilized = counter.frame_count
                assert stabilized <= 4  # column regular/singleton + row
            if index > 8:
                assert counter.frame_count == stabilized
    assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)


@pytest.mark.parametrize("all_skipped", [False, True])
def test_fullgraph_isab_preserves_sentinel_outputs_and_zero_gradients(all_skipped):
    torch.manual_seed(771)
    eager = InducedSelfAttentionBlock(16, 2, 32, 4, zero_init=False)
    candidate = deepcopy(eager)
    compiled = torch.compile(candidate, backend="aot_eager", fullgraph=True, dynamic=True)
    x = torch.randn(2, 3, 17, 16)
    x[0] = -100
    if all_skipped:
        x[:] = -100
    a_input, b_input = x.clone().requires_grad_(), x.clone().requires_grad_()
    a, b = compiled(a_input, 11), eager(b_input, 11)
    torch.testing.assert_close(a, b)
    a.square().mean().backward()
    assert torch.count_nonzero(a_input.grad[0]) == 0
    if all_skipped:
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in candidate.parameters())
    else:
        b.square().mean().backward()
        torch.testing.assert_close(a_input.grad, b_input.grad)
        assert_gradients(candidate, eager)


@pytest.mark.parametrize("classes", [3, 31])
def test_compiled_column_class_dispatch_and_affine_match_eager(monkeypatch, classes):
    original_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda *a, **kw: original_compile(*a, **kw, backend="aot_eager"))
    torch.manual_seed(772)
    eager = model(col_affine=True, col_feature_group=False, icl_attention="dense")
    candidate = deepcopy(eager)
    candidate.compile_frontend("all")
    x = torch.randn(2, 17, 7)
    y = torch.randint(classes, (2, 11)).float()
    y[0, 0] = classes - 1
    a, b = candidate.col_embedder(x, y), eager.col_embedder(x, y)
    torch.testing.assert_close(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    assert_gradients(candidate, eager)
    if classes > 10:
        candidate.col_embedder.mixed_radix_ensemble = False
        with pytest.raises(ValueError, match="classes"):
            candidate.col_embedder(x, y)


def test_nonzero_dropout_column_compilation_keeps_eager_compaction(monkeypatch):
    original_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda *a, **kw: original_compile(*a, **kw, backend="eager"))
    torch.manual_seed(773)
    eager = model(dropout=.2, icl_attention="dense", col_target_aware=False, col_feature_group=False)
    candidate = deepcopy(eager)
    candidate.compile_frontend("all")
    x = torch.randn(2, 17, 7)
    x[0, :, 2] = -100
    y = torch.randint(3, (2, 11)).float()
    torch.manual_seed(774)
    a = candidate.col_embedder(x, y)
    torch.manual_seed(774)
    b = eager.col_embedder(x, y)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().mean().backward()
    b.square().mean().backward()
    assert_gradients(candidate, eager, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["regression", "unaware", "ragged", "valid", "with_test", "frozen"])
def test_fullgraph_frontend_alternative_paths_match_eager(monkeypatch, case):
    original_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda *a, **kw: original_compile(*a, **kw, backend="aot_eager"))
    torch.manual_seed(775)
    config = dict(icl_attention="dense")
    if case == "regression":
        config["max_classes"] = 0
    if case == "unaware":
        config["col_target_aware"] = False
    if case == "ragged":
        config["col_feature_group"] = False
    if case == "valid":
        config["col_feature_group"] = "valid"
    eager = model(**config)
    if case == "frozen":
        eager.col_embedder.requires_grad_(False)
    candidate = deepcopy(eager)
    candidate.compile_frontend("all")
    x = torch.randn(2, 17, 7)
    if case == "unaware":
        x[0, :, -1] = -100
    y = torch.randn(2, 11) if case == "regression" else torch.randint(3, (2, 11)).float()
    d = torch.tensor([4, 7]) if case == "ragged" else None
    kwargs = dict(d=d, embed_with_test=case == "with_test")
    a, b = candidate(x, y, **kwargs), eager(x, y, **kwargs)
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    a.square().mean().backward()
    b.square().mean().backward()
    assert_gradients(candidate, eager)


@pytest.mark.parametrize("xpos", [False, True])
def test_compiled_rope_does_not_guard_or_mutate_length_cache(xpos):
    from torch._dynamo.testing import CompileCounter

    rope = RotaryEmbedding(8, use_xpos=xpos)
    expected = RotaryEmbedding(8, use_xpos=xpos, cache_if_possible=False)
    seq = torch.arange(7)
    rope(seq, seq_len=7)
    if xpos:
        rope.get_scale(seq, seq_len=7)
    frequency_cache, scale_cache = rope.cached_freqs, rope.cached_scales
    counter = CompileCounter()

    def apply(positions):
        frequencies = rope(positions, seq_len=positions.shape[0])
        scale = rope.get_scale(positions, seq_len=positions.shape[0]) if xpos else 1.
        return frequencies, scale

    compiled = torch.compile(apply, backend=counter, fullgraph=True, dynamic=True)
    with torch._dynamo.config.patch(fail_on_recompile_limit_hit=True):
        for width in range(3, 25):
            positions = torch.arange(width)
            frequencies, scale = compiled(positions)
            torch.testing.assert_close(frequencies, expected(positions, seq_len=width))
            if xpos:
                torch.testing.assert_close(scale, expected.get_scale(positions, seq_len=width))
            assert rope.cached_freqs is frequency_cache
            assert rope.cached_scales is scale_cache
    assert counter.frame_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_float32_frontend_with_trainer_autocast_matches_disabled_autocast():
    # Trainer's --dtype float32 still enters an autocast(dtype=float32)
    # context; test this as well as the disabled context used above.
    torch.manual_seed(776)
    candidate = model(icl_attention="dense").cuda()
    candidate.compile_frontend("all")
    x = torch.randn(2, 35, 7, device="cuda")
    y = torch.randint(3, (2, 35), device="cuda").float()[:, :23]
    outputs, gradients = [], []
    for enabled in (False, True):
        candidate.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float32, enabled=enabled):
            col = candidate.col_embedder(x, y, skip_reserved=True)
            row = candidate.row_interactor(col)
            result = candidate.icl_predictor(row, y)
        assert col.dtype == row.dtype == result.dtype == torch.float32
        result.square().mean().backward()
        outputs.append(result.detach())
        gradients.append(torch.cat([p.grad.flatten() for p in candidate.parameters() if p.grad is not None]))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(gradients[0], gradients[1], rtol=2e-3, atol=2e-5)
