from contextlib import nullcontext
import shlex
import subprocess
import os
from pathlib import Path

import torch
import pytest

from tabicl._model.tabicl import TabICL
from tabicl.train._run import Trainer
from tabicl.train._train_config import build_parser
from tabicl._model.ssmax import create_ssmax_layer


def test_all_v2_launchers_expand_to_valid_default_and_ablation_arguments(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for mode in ("clf", "reg"):
        for stage in (1, 2, 3):
            env = dict(os.environ, DRY_RUN="1", CKPT_DIR=str(tmp_path / "ckpt"),
                       LOG_DIR=str(tmp_path / "logs"))
            text = subprocess.check_output(
                ["bash", str(root / f"scripts/train_v2_{mode}_stage{stage}.sh")], env=env, text=True)
            tokens = shlex.split(text)
            config = build_parser().parse_args(tokens[tokens.index("--wandb_log"):])
            assert config.icl_attention == "elu_linear_flex"
            assert config.icl_flex_qk_norm is True
            assert config.icl_flex_scale_mode == "bounded"
            assert config.icl_ssmax is False
    env.update(ICL_ATTENTION="elu_linear", ICL_DIAGNOSTICS_EVERY="0")
    text = subprocess.check_output(
        ["bash", str(root / "scripts/train_v2_clf_stage1.sh")], env=env, text=True)
    tokens = shlex.split(text)
    config = build_parser().parse_args(tokens[tokens.index("--wandb_log"):])
    assert config.icl_attention == "elu_linear"
    assert config.icl_diagnostics_every == 0


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_real_model_microbatch_logs_diagnostics_and_disables_sampling(regression, device):
    # Exercise real Trainer forward/loss/backward without launching priors,
    # external logging or optimizers.
    torch.manual_seed(23)
    trainer = Trainer.__new__(Trainer)
    trainer.config = build_parser().parse_args(["--device", device, "--icl_diagnostics_every", "2",
                                               "--num_quantiles", "9", "--fail_fast", "True"])
    trainer.ddp = False
    trainer.master_process = True
    trainer.regression = regression
    trainer.curr_step = 0
    trainer._disable_cudnn_sdp = False
    trainer.amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    trainer.scaler = torch.amp.GradScaler(device, enabled=False)
    trainer.model = TabICL(max_classes=0 if regression else 10, num_quantiles=9,
                          embed_dim=16, col_num_blocks=1, col_nhead=2, col_num_inds=4,
                          row_num_blocks=1, row_nhead=2, row_num_cls=2,
                          icl_num_blocks=3, icl_nhead=2, zero_init=False,
                          icl_attention="elu_linear_flex", icl_flex_num_kv_blocks=2,
                          icl_flex_qk_norm=True, icl_flex_scale_mode="bounded").to(device)
    y = torch.randn(1, 291) if regression else torch.randint(0, 3, (1, 291))
    batch = (torch.randn(1, 291, 6), y,
             torch.tensor([6]), torch.tensor([291]), torch.tensor([267]))
    metrics = trainer.run_micro_batch(batch, 0, 2)
    assert metrics["pinball" if regression else "ce"] > 0
    assert any(k.endswith("/recall_at_k") for k in metrics)
    assert any(k.endswith("/denominator_min") for k in metrics)
    assert all(v == 0 for k, v in metrics.items() if k.endswith("/nonfinite_fraction"))
    assert not any(m.diagnostics_enabled for m in trainer.model.modules()
                   if hasattr(m, "diagnostics_enabled"))
    trainer.curr_step = 2  # next completed step=3, not a sampling step
    metrics = trainer.run_micro_batch(batch, 0, 2)
    assert not any(k.startswith("icl_diag/") for k in metrics)


@pytest.mark.parametrize("variant", ["ssmax", "ssmax-mlp", "ssmax-mlp-elementwise",
                                      "qassmax-mlp", "qassmax-mlp-elementwise"])
def test_ssmax_per_row_lengths_preserve_scalar_dense_behavior(variant):
    torch.manual_seed(24)
    layer = create_ssmax_layer(variant, num_heads=2, embed_dim=16)
    q = torch.randn(2, 2, 3, 8, requires_grad=True)
    lengths = torch.tensor([[128, 256, 193], [99, 513, 1]])
    actual = layer(q, lengths)
    expected = torch.cat([
        torch.cat([layer(q[b:b+1, :, t:t+1], int(lengths[b, t]))
                   for t in range(3)], dim=2) for b in range(2)], dim=0)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    gradients = torch.autograd.grad(actual.square().sum(), q, retain_graph=True)[0]
    reference = torch.autograd.grad(expected.square().sum(), q)[0]
    assert torch.allclose(gradients, reference, atol=1e-6, rtol=1e-5)


def test_resume_rejects_silent_qk_norm_change_and_accepts_equivalent_auto(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.config = build_parser().parse_args(["--device", "cpu", "--only_load_model", "True"])
    trainer.config.checkpoint_path = str(tmp_path / "resume.ckpt")
    trainer.raw_model = torch.nn.Linear(2, 2)
    trainer.model_config = {"icl_attention": "elu_linear_flex", "icl_flex_qk_norm": True,
                            "icl_flex_scale_mode": "none", "icl_sparse_every": 3,
                            "icl_flex_block_size": 128, "icl_flex_num_kv_blocks": 4,
                            "icl_ssmax": False}
    stored = dict(trainer.model_config, icl_flex_qk_norm=False)
    torch.save({"config": stored, "state_dict": trainer.raw_model.state_dict()},
               trainer.config.checkpoint_path)
    with pytest.raises(ValueError, match="icl_flex_qk_norm"):
        trainer.load_checkpoint()
    stored.update(icl_flex_qk_norm=True, icl_flex_scale_mode="auto")
    torch.save({"config": stored, "state_dict": trainer.raw_model.state_dict()},
               trainer.config.checkpoint_path)
    trainer.load_checkpoint()
