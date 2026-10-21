import torch
import pytest

from tabicl._model.icl_attention import SetRetrievalAttention
from tabicl._model.tabicl import TabICL
from scripts.diagnose_icl_checkpoint import POLICIES, _metrics, evaluate, resolve_checkpoints


@pytest.mark.parametrize("device,dtype", [
    ("cpu", torch.float32), ("cuda", torch.float32), ("cuda", torch.bfloat16),
])
def test_mechanism_diagnostic_policies_restore_model_and_return_records(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(7)
    model = TabICL(
        max_classes=3,
        embed_dim=32,
        col_num_blocks=1,
        col_nhead=4,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=4,
        row_num_cls=2,
        icl_num_blocks=2,
        icl_nhead=4,
        icl_ssmax=False,
        ff_factor=2,
        icl_attention="global_sparse",
        icl_sparse_every=2,
        icl_block_size=16,
        icl_num_selected_blocks=1,
    ).to(device).eval()
    modules = [(name, module) for name, module in model.icl_predictor.named_modules()
               if isinstance(module, SetRetrievalAttention)]
    assert modules
    for _, module in modules:
        module.use_triton = False
    x = torch.randn(1, 36, 3)
    # Reproduce the original failure: eval() returns only two classes, but the
    # query targets also contain class 2, which the support did not observe.
    y = torch.cat((torch.arange(32).remainder(2), torch.full((4,), 2))).unsqueeze(0)
    modes = {module: module.training for module in model.modules()}
    original_hybrid = [module._hybrid_attention for _, module in modules]
    records = evaluate(
        model, modules, [{"id": "tiny", "x": x, "y": y, "train_size": 32}],
        torch.device(device), dtype, POLICIES, query_count=2, seed=11, permutation=True,
    )
    assert len(records) == len(POLICIES)
    assert {r["policy"] for r in records} == set(POLICIES)
    assert all("cross_entropy" in r and "accuracy" in r for r in records)
    native = next(r for r in records if r["policy"] == "native")
    assert "support_permutation_logit_max_abs" in native
    assert any(key.endswith("head_0/recall_at_k") for key in native)
    assert any(key.endswith("head_0/fusion_sparse_gate_std") for key in native)
    assert all(module._hybrid_attention == original for (_, module), original in zip(modules, original_hybrid))
    assert all(not module.diagnostics_enabled for _, module in modules)
    assert all(module.diagnostic_query_start == 0 for _, module in modules)
    assert all(module.training == training for module, training in modes.items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_invalid_labels_fail_before_model_forward_and_restore_mode():
    model = torch.nn.Linear(2, 2).eval()
    model.max_classes = 3
    called = []
    handle = model.register_forward_pre_hook(lambda *_: called.append(True))
    try:
        with pytest.raises(ValueError, match=r"bad-label.*integers in \[0, 2\]"):
            evaluate(model, [], [{"id": "bad-label", "x": torch.zeros(1, 4, 2),
                                  "y": torch.tensor([[0., 1., 2., 3.]]), "train_size": 2}],
                     torch.device("cpu"), torch.float32, ["native"], 0, 11, False)
    finally:
        handle.remove()
    assert not called
    assert not model.training


def test_truncated_logits_fail_with_actionable_error():
    model = torch.nn.Linear(2, 2)
    model.max_classes = 3
    with pytest.raises(ValueError, match="requires 3 logits; got 2"):
        _metrics(model, torch.zeros(1, 1, 2), torch.tensor([[0, 2]]), support=1)


def test_checkpoint_directory_selection_includes_endpoints(tmp_path):
    for step in (5000, 10000, 20000, 25000, 30000):
        (tmp_path / f"step-{step}.ckpt").touch()
    (tmp_path / "unrelated.ckpt").touch()
    assert [p.name for p in resolve_checkpoints(tmp_path, 25000, None)] == [
        "step-5000.ckpt", "step-25000.ckpt", "step-30000.ckpt"
    ]
    assert [p.name for p in resolve_checkpoints(tmp_path, 0, None)] == [
        "step-5000.ckpt", "step-10000.ckpt", "step-20000.ckpt", "step-25000.ckpt",
        "step-30000.ckpt"
    ]
    assert [p.name for p in resolve_checkpoints(tmp_path, 25000, [10000, 30000])] == [
        "step-10000.ckpt", "step-30000.ckpt"
    ]
