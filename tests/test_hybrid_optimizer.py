from contextlib import nullcontext
from copy import deepcopy
import os
from pathlib import Path
import shlex
import subprocess

import pytest
import torch

from tabicl._model.tabicl import TabICL
from tabicl.train._muon import Muon, zeropower_via_newtonschulz5
from tabicl.train._optimizer import build_optimizer, optimizer_spec, validate_optimizer_checkpoint
from tabicl.train._run import Trainer
from tabicl.train._train_config import build_parser


def small_model(attention="global_sparse", global_type="elu"):
    return TabICL(
        embed_dim=16, col_num_blocks=1, col_nhead=2, col_num_inds=4,
        col_ssmax="qassmax-mlp-elementwise", row_num_blocks=1, row_nhead=2,
        row_num_cls=2, icl_num_blocks=3, icl_nhead=2, zero_init=False,
        icl_attention=attention, icl_global_type=global_type, icl_num_induced_slots=4,
        icl_block_size=16, icl_num_selected_blocks=1, icl_qk_norm=True, icl_ssmax=False,
    )


def config(*extra):
    return build_parser().parse_args([
        "--device", "cpu", "--muon", "True", "--lr", "0.0004", "--adamw_lr", "0.0001",
        "--weight_decay", "0.01", "--scheduler", "cosine_with_restarts",
        "--warmup_proportion", "-1", "--warmup_steps", "2", "--max_steps", "20",
        *extra,
    ])


@pytest.mark.parametrize("attention,global_type", [
    ("dense", "elu"), ("global_sparse", "elu"), ("global_sparse", "induced"),
])
def test_real_model_routing_excludes_sensitive_matrices_and_frozen_parameters(attention, global_type):
    model = small_model(attention, global_type)
    model.row_interactor.requires_grad_(False)
    opt = build_optimizer(model, config())
    routed = {name: g for g in opt.param_groups for name in g["param_names"]}
    assert set(routed) == {n for n, p in model.named_parameters() if p.requires_grad}
    assert sum(len(g["params"]) for g in opt.param_groups) == len(routed)
    assert routed["icl_predictor.tf_icl.blocks.0.attn.in_proj_weight"]["use_muon"]
    assert routed["col_embedder.tf_col.blocks.0.multihead_attn2.linear1.weight"]["use_muon"]
    sensitive = [n for n in routed if any(s in n for s in (
        "ind_vectors", "inducing_points", "fusion_gate", "qk_norm", "ssmax_layer",
        "icl_predictor.decoder", "y_encoder", "in_linear",
    ))]
    assert sensitive
    for name in sensitive:
        assert routed[name]["use_muon"] is False, name
    for name, p in model.named_parameters():
        if name in routed and p.ndim < 2:
            assert not routed[name]["use_muon"], name
            assert routed[name]["weight_decay"] == 0, name


def test_hybrid_adamw_matches_torch_with_tiny_and_missing_gradients():
    # Tiny gradients exercise epsilon placement; a missing gradient must not
    # advance that parameter's moment-bias correction or apply weight decay.
    params = [torch.nn.Parameter(torch.randn(4, dtype=torch.float64)) for _ in range(2)]
    refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
    hybrid = Muon([dict(params=params, use_muon=False)], lr=1e-3, weight_decay=0.01,
                  adamw_betas=(0.9, 0.999))
    adamw = torch.optim.AdamW(refs, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999))
    for step in range(6):
        for index, (p, ref) in enumerate(zip(params, refs)):
            grad = None if index == 1 and step % 2 == 0 else torch.randn_like(p) * 1e-9
            p.grad = grad
            ref.grad = None if grad is None else grad.clone()
        hybrid.step()
        adamw.step()
        for p, ref in zip(params, refs):
            torch.testing.assert_close(p, ref, rtol=1e-13, atol=1e-13)
    assert hybrid.state[params[1]]["step"] == 3


@pytest.mark.parametrize("shape,count", [((7, 5), 4), ((5, 7), 3), ((8, 8), 5)])
def test_batched_newton_schulz_matches_individual_matrices(shape, count):
    torch.manual_seed(sum(shape) + count)
    matrices = torch.randn(count, *shape, dtype=torch.float64)
    batched = zeropower_via_newtonschulz5(matrices, 5)
    individual = torch.stack([zeropower_via_newtonschulz5(matrix, 5) for matrix in matrices])
    torch.testing.assert_close(batched, individual, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
@pytest.mark.parametrize("nesterov,cautious", [(True, False), (True, True), (False, False)])
def test_batched_muon_matches_individual_step_with_missing_gradients(device, nesterov, cautious):
    torch.manual_seed(103)
    shapes = [(8, 5), (8, 5), (5, 8), (5, 8), (8, 5), (4, 4)]
    params = [torch.nn.Parameter(torch.randn(*shape, device=device)) for shape in shapes]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in params]
    optimized = Muon([dict(params=params, use_muon=True)], lr=4e-4, weight_decay=0.01,
                     momentum=0.9, nesterov=nesterov, use_cautious_wd=cautious)
    for step in range(4):
        for index, (p, ref) in enumerate(zip(params, reference)):
            p.grad = None if index == 1 and step % 2 == 0 else torch.randn_like(p)
            ref.grad = None if p.grad is None else p.grad.clone()
        optimized.step()
        for p, ref in zip(params, reference):
            g = ref.grad
            if g is None:
                continue
            state = optimized.state[p]
            # Recover the pre-step momentum; the reference update uses the old
            # independent 2D NS path and exactly the optimizer's weight decay.
            buf = state["momentum_buffer"]
            ns_input = (g + 0.9 * buf if nesterov else buf).reshape(len(g), -1)
            update = zeropower_via_newtonschulz5(ns_input, 5).reshape_as(ref)
            if cautious:
                ref.data.add_(ref * (update * ref >= 0), alpha=-4e-6)
            else:
                ref.data.mul_(1 - 4e-6)
            ref.data.add_(update, alpha=-4e-4 * (max(ns_input.shape) ** 0.5) * 0.2)
            torch.testing.assert_close(p, ref, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(params[1], reference[1], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
@pytest.mark.parametrize("batch_kind", ["dense", "compact", "nested"])
def test_trainer_hybrid_checkpoint_resumes_identical_next_update(tmp_path, device, batch_kind):
    torch.manual_seed(12)
    trainer = Trainer.__new__(Trainer)
    trainer.config = config("--batch_size", "1", "--micro_batch_size", "1",
                            "--icl_diagnostics_every", "0", "--fail_fast", "True")
    trainer.config.checkpoint_dir = str(tmp_path)
    trainer.config.device = device
    trainer.master_process = True
    trainer.ddp = False
    trainer.regression = False
    trainer.curr_step = 0
    trainer._disable_cudnn_sdp = False
    trainer.amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    trainer.scaler = torch.amp.GradScaler(device, enabled=False)
    trainer.model = trainer.raw_model = small_model().to(device)
    trainer.model_config = dict(icl_attention="global_sparse", icl_attention_impl_version=4,
                               icl_qk_norm=True, icl_block_size=16, icl_num_selected_blocks=1)
    trainer.configure_optimizer()
    batch = (torch.randn(1, 35, 3), torch.randint(0, 3, (1, 35)),
             torch.tensor([3]), torch.tensor([35]), torch.tensor([25]))
    if batch_kind == "compact":
        batch = ([batch[0][0]], [batch[1][0]], *batch[2:])
    elif batch_kind == "nested":
        batch = (torch.nested.nested_tensor([batch[0][0]]),
                 torch.nested.nested_tensor([batch[1][0]]), *batch[2:])
    for step in range(3):
        trainer.run_batch(batch)
        trainer.curr_step = step + 1
    trainer.save_checkpoint("step-3.ckpt")
    expected = trainer.run_batch(batch)
    expected_params = deepcopy(trainer.model.state_dict())
    trainer.config.checkpoint_path = str(tmp_path / "step-3.ckpt")
    trainer.configure_optimizer()
    trainer.load_checkpoint()
    assert trainer.curr_step == 3
    actual = trainer.run_batch(batch)
    assert actual["ce"] == expected["ce"]
    assert actual["lr/muon"] == expected["lr/muon"]
    assert actual["lr/adamw_no_decay"] == expected["lr/adamw_no_decay"]
    assert actual["lr/adamw_decay"] / actual["lr/muon"] == pytest.approx(0.25)
    for name, p in trainer.model.state_dict().items():
        torch.testing.assert_close(p, expected_params[name], rtol=0 if device == "cpu" else 1e-5,
                                   atol=0 if device == "cpu" else 1e-7)


def test_legacy_all_muon_rejected_but_weights_only_load_succeeds(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.config = config()
    trainer.config.checkpoint_path = str(tmp_path / "legacy.ckpt")
    trainer.raw_model = small_model()
    trainer.model_config = dict(icl_attention="global_sparse", icl_attention_impl_version=4,
                               icl_qk_norm=True, icl_block_size=16, icl_num_selected_blocks=1)
    old = Muon([dict(params=list(trainer.raw_model.parameters()), use_muon=True)])
    torch.save(dict(config=trainer.model_config, state_dict=trainer.raw_model.state_dict(),
                    optimizer_state=old.state_dict()), trainer.config.checkpoint_path)
    trainer.optimizer = build_optimizer(trainer.raw_model, trainer.config)
    with pytest.raises(ValueError, match="only_load_model True"):
        trainer.load_checkpoint()
    trainer.config.only_load_model = True
    trainer.curr_step = 0
    trainer.load_checkpoint()
    assert trainer.curr_step == 0 and len(trainer.optimizer.state) == 0
    spec = optimizer_spec(trainer.optimizer)
    spec["groups"][0]["param_names"].reverse()
    with pytest.raises(ValueError, match="routing"):
        validate_optimizer_checkpoint(trainer.optimizer, dict(optimizer_spec=spec))


def test_optimizer_gradient_diagnostics_are_grouped_and_preclip():
    trainer = Trainer.__new__(Trainer)
    first = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    second = torch.nn.Parameter(torch.tensor([1.0]))
    first.grad = torch.tensor([6.0, 8.0])
    second.grad = torch.tensor([2.0])
    trainer.optimizer = torch.optim.SGD([
        dict(params=[first], group_name="matrix", lr=0.1),
        dict(params=[second], group_name="sensitive", lr=0.1),
    ])
    metrics = trainer.optimizer_gradient_diagnostics()
    assert metrics["optimizer_diag/matrix/grad_rms"] == pytest.approx((50.0) ** 0.5)
    assert metrics["optimizer_diag/matrix/grad_max_abs"] == 8.0
    assert metrics["optimizer_diag/matrix/grad_nonfinite_fraction"] == 0
    assert metrics["optimizer_diag/matrix/grad_missing_fraction"] == 0
    assert metrics["optimizer_diag/sensitive/grad_rms"] == 2.0


def test_component_diagnostics_isolate_decoder_gradients_and_parameters():
    trainer = Trainer.__new__(Trainer)
    trainer.raw_model = small_model()
    for parameter in trainer.raw_model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.zeros_like(parameter)
    for parameter in trainer.raw_model.icl_predictor.decoder.parameters():
        parameter.grad.fill_(1)
    gradients = trainer.component_gradient_diagnostics()
    assert gradients["component_grad/decoder/grad_rms"] == 1
    assert gradients["component_grad/decoder/grad_max_abs"] == 1
    assert gradients["component_grad/decoder/grad_missing_fraction"] == 0
    assert gradients["component_grad/icl_attention/grad_rms"] == 0

    parameter_metrics, largest = trainer.parameter_diagnostics(trainer.raw_model)
    assert parameter_metrics["parameter_diag/decoder/numel"] > 0
    assert parameter_metrics["parameter_diag/icl_attention/numel"] > 0
    assert largest


def test_stage1_launcher_initializes_once_then_resumes_hybrid(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/train_v2_clf_stage1.sh"
    env = dict(os.environ, DRY_RUN="1", CKPT_DIR=str(tmp_path / "new run"),
               LOG_DIR=str(tmp_path / "logs"), INIT_CKPT=str(tmp_path / "old run/step-40000.ckpt"),
               LR="4e-4", ADAMW_LR="1e-4", MAX_STEPS="20000", WARMUP_STEPS="500")

    def launch():
        tokens = shlex.split(subprocess.check_output(["bash", str(script)], env=env, text=True))
        return build_parser().parse_args(tokens[tokens.index("--wandb_log"):])

    args = launch()
    assert args.only_load_model and args.checkpoint_path == env["INIT_CKPT"]
    assert args.muon and args.lr == 4e-4 and args.adamw_lr == 1e-4
    assert args.warmup_proportion == -1 and args.warmup_steps == 500
    Path(env["CKPT_DIR"]).mkdir()
    Path(env["CKPT_DIR"], "step-500.ckpt").touch()
    args = launch()
    assert not args.only_load_model and args.checkpoint_path is None
