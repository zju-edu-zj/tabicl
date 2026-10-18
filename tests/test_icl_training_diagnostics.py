from contextlib import nullcontext
import shlex
import subprocess
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from tabicl._model.tabicl import TabICL
from tabicl.train._run import Trainer
from tabicl.train._train_config import build_parser
from tabicl._model.ssmax import create_ssmax_layer


def test_screen_validation_bypasses_compiled_wrapper(monkeypatch):
    from tabicl.train import _run

    eager_model = object()
    compiled_wrapper = SimpleNamespace(_orig_mod=eager_model)
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(
        eval_data=True,
        device="cpu",
        icl_diagnostic_queries=0,
    )
    trainer.ddp = False
    trainer.master_process = True
    trainer.raw_model = compiled_wrapper
    trainer.eval_cases = {}
    trainer.amp_ctx = nullcontext()
    trainer.curr_step = 0
    trainer.local_log_dir = None
    trainer._sdpa_ctx = nullcontext
    trainer.log_local_metrics = lambda *args: None
    seen = []

    def evaluate(model, *args):
        seen.append(model)
        return {}, []

    monkeypatch.setattr(_run, "evaluate_cases", evaluate)
    trainer.validate_screen()
    assert seen == [eager_model]


def test_all_v2_launchers_expand_to_valid_default_and_ablation_arguments(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for mode in ("clf", "reg"):
        for stage in (1, 2, 3):
            env = dict(os.environ, DRY_RUN="1", CKPT_DIR=str(tmp_path / "ckpt"), LOG_DIR=str(tmp_path / "logs"))
            text = subprocess.check_output(
                ["bash", str(root / f"scripts/train_v2_{mode}_stage{stage}.sh")], env=env, text=True
            )
            tokens = shlex.split(text)
            config = build_parser().parse_args(tokens[tokens.index("--wandb_log") :])
            assert config.icl_attention == "global_sparse"
            assert config.icl_ssmax is False
    env.update(ICL_ATTENTION="global", ICL_DIAGNOSTICS_EVERY="0")
    text = subprocess.check_output(["bash", str(root / "scripts/train_v2_clf_stage1.sh")], env=env, text=True)
    tokens = shlex.split(text)
    config = build_parser().parse_args(tokens[tokens.index("--wandb_log") :])
    assert config.icl_attention == "global"
    assert config.icl_global_type == "elu"
    assert config.icl_diagnostics_every == 0
    env.update(ICL_GLOBAL_TYPE="induced", ICL_NUM_INDUCED_SLOTS="64")
    text = subprocess.check_output(["bash", str(root / "scripts/train_v2_clf_stage1.sh")], env=env, text=True)
    tokens = shlex.split(text)
    config = build_parser().parse_args(tokens[tokens.index("--wandb_log") :])
    assert config.icl_global_type == "induced"
    assert config.icl_num_induced_slots == 64


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_real_model_microbatch_logs_diagnostics_and_disables_sampling(regression, device):
    # Exercise real Trainer forward/loss/backward without launching priors,
    # external logging or optimizers.
    torch.manual_seed(23)
    trainer = Trainer.__new__(Trainer)
    trainer.config = build_parser().parse_args(
        ["--device", device, "--icl_diagnostics_every", "2", "--num_quantiles", "9", "--fail_fast", "True"]
    )
    trainer.ddp = False
    trainer.master_process = True
    trainer.regression = regression
    trainer.curr_step = 0
    trainer._disable_cudnn_sdp = False
    trainer.amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    trainer.scaler = torch.amp.GradScaler(device, enabled=False)
    trainer.model = TabICL(
        max_classes=0 if regression else 10,
        num_quantiles=9,
        embed_dim=16,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=3,
        icl_nhead=2,
        zero_init=False,
        icl_attention="global_sparse",
        icl_num_selected_blocks=2,
        icl_ssmax=False,
    ).to(device)
    y = torch.randn(1, 291) if regression else torch.randint(0, 3, (1, 291))
    batch = (torch.randn(1, 291, 6), y, torch.tensor([6]), torch.tensor([291]), torch.tensor([267]))
    metrics = trainer.run_micro_batch(batch, 0, 2)
    assert metrics["pinball" if regression else "ce"] > 0
    assert any(k.endswith("/recall_at_k") for k in metrics)
    assert any(k.endswith("/denominator_min") for k in metrics)
    assert all(v == 0 for k, v in metrics.items() if k.endswith("/nonfinite_fraction"))
    assert not any(m.diagnostics_enabled for m in trainer.model.modules() if hasattr(m, "diagnostics_enabled"))
    trainer.curr_step = 2  # next completed step=3, not a sampling step
    metrics = trainer.run_micro_batch(batch, 0, 2)
    assert not any(k.startswith("icl_diag/") for k in metrics)


@pytest.mark.parametrize(
    "variant", ["ssmax", "ssmax-mlp", "ssmax-mlp-elementwise", "qassmax-mlp", "qassmax-mlp-elementwise"]
)
def test_ssmax_per_row_lengths_preserve_scalar_dense_behavior(variant):
    torch.manual_seed(24)
    layer = create_ssmax_layer(variant, num_heads=2, embed_dim=16)
    q = torch.randn(2, 2, 3, 8, requires_grad=True)
    lengths = torch.tensor([[128, 256, 193], [99, 513, 1]])
    actual = layer(q, lengths)
    expected = torch.cat(
        [
            torch.cat([layer(q[b : b + 1, :, t : t + 1], int(lengths[b, t])) for t in range(3)], dim=2)
            for b in range(2)
        ],
        dim=0,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    gradients = torch.autograd.grad(actual.square().sum(), q, retain_graph=True)[0]
    reference = torch.autograd.grad(expected.square().sum(), q)[0]
    assert torch.allclose(gradients, reference, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "setting,value",
    [
        ("icl_router", "hierarchical"),
        ("icl_partition", "content_tree"),
        ("icl_block_size", 64),
        ("icl_num_selected_blocks", 8),
        ("icl_sparse_every", 2),
        ("icl_attention", "global_full"),
        ("icl_global_type", "induced"),
    ],
)
def test_resume_rejects_silent_architecture_changes(tmp_path, setting, value):
    trainer = Trainer.__new__(Trainer)
    trainer.config = build_parser().parse_args(["--device", "cpu", "--only_load_model", "True"])
    trainer.config.checkpoint_path = str(tmp_path / "attention.ckpt")
    trainer.raw_model = torch.nn.Linear(2, 2)
    original = {"icl_attention": "global_sparse", "icl_attention_impl_version": 4,
                "icl_router": "centroid", "icl_qk_norm": False}
    torch.save({"config": original, "state_dict": trainer.raw_model.state_dict()}, trainer.config.checkpoint_path)
    trainer.model_config = dict(original)
    trainer.load_checkpoint()
    trainer.model_config[setting] = value
    with pytest.raises(ValueError, match=setting):
        trainer.load_checkpoint()


def test_public_choices_and_removed_flags():
    from tabicl._model.icl_attention import ICL_ATTENTION_TYPES

    parser = build_parser()
    action = next(a for a in parser._actions if a.dest == "icl_attention")
    assert tuple(action.choices) == ICL_ATTENTION_TYPES
    for option in ("--icl_flex_qk_norm", "--icl_linear_num_features", "--icl_induced_num_slots"):
        with pytest.raises(SystemExit):
            parser.parse_args([option, "1"])


def test_dense_default_preserves_official_parameter_layout():
    from tabicl._model.encoders import Encoder

    m = Encoder(1, 32, 2, 64)
    assert set(m.blocks[0].attn.state_dict()) == {"in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias"}
