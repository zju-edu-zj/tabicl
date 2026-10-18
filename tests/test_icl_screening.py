from contextlib import nullcontext
import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tabicl._model.tabicl import TabICL
from tabicl.prior._dataset import PriorDataset
from tabicl.train._screening import (
    StepSeededPrior,
    evaluate_cases,
    load_common_initialization,
)


def small_config(**overrides):
    return dict(
        embed_dim=16,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=3,
        icl_nhead=2,
        icl_ssmax=False,
        zero_init=False,
        **overrides,
    )


def test_formal_screen_compiles_training_model(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/screen_icl_attention.py"
    spec = importlib.util.spec_from_file_location("screen_icl_attention", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        "sys.argv",
        [
            str(script),
            "plan",
            "--output",
            str(tmp_path),
            "--steps",
            "20000",
            "--variants",
            "global_induced",
        ],
    )
    args = module.arguments()
    assert module.train_options(args, "global_induced", 42)["model_compile"] == "True"

    monkeypatch.setattr(
        "sys.argv",
        [
            str(script),
            "plan",
            "--output",
            str(tmp_path),
            "--steps",
            "20000",
            "--variants",
            "global_induced",
            "--no-model-compile",
        ],
    )
    args = module.arguments()
    assert module.train_options(args, "global_induced", 42)["model_compile"] == "False"


def test_all_screening_variants_share_every_common_parameter(tmp_path):
    config = small_config(icl_attention="dense")
    reference = TabICL(**config)
    path = tmp_path / "initial.pt"
    torch.save(
        dict(
            purpose="icl_screen_initialization",
            config=config,
            state_dict=reference.state_dict(),
        ),
        path,
    )
    routes, slots = [], []
    for attention, global_type, partition in [
        ("dense", "elu", "projection"),
        ("global", "elu", "projection"),
        ("global_sparse_alternating", "elu", "projection"),
        ("global_sparse", "elu", "projection"),
        ("global_sparse", "elu", "content_tree"),
        ("global", "induced", "projection"),
        ("global_sparse", "induced", "projection"),
    ]:
        changed = dict(
            config,
            icl_attention=attention,
            icl_global_type=global_type,
            icl_num_induced_slots=8,
            icl_partition=partition,
        )
        torch.manual_seed(983)  # Constructor RNG consumption differs across variants.
        model = TabICL(**changed)
        load_common_initialization(model, path, changed, seed=42)
        for name, value in reference.state_dict().items():
            assert torch.equal(value, model.state_dict()[name]), name
        routes.append(
            {
                name: value
                for name, value in model.state_dict().items()
                if name.endswith("route_direction")
            }
        )
        slots.append(
            {
                name: value
                for name, value in model.state_dict().items()
                if name.endswith("inducing_points")
            }
        )
    for name, value in routes[4].items():
        assert torch.equal(value, routes[3][name])
        assert torch.equal(value, routes[2][name])
    assert slots[-2]
    for name, value in slots[-1].items():
        assert torch.equal(value, slots[-2][name])
    bad = dict(changed, icl_ssmax=True)
    with pytest.raises(ValueError, match="icl_ssmax"):
        load_common_initialization(model, path, bad, 42)


def test_actual_prior_replays_across_workers_and_resume():
    prior = PriorDataset(
        batch_size=1,
        batch_size_per_gp=1,
        min_features=3,
        max_features=3,
        max_seq_len=64,
        min_train_size=0.4,
        max_train_size=0.8,
        prior_type="graph_scm",
        n_jobs=1,
    )

    def batches(workers, start, count):
        dataset = StepSeededPrior(prior, 43, 42, rank=1, world_size=2)
        dataset.start_step = start
        loader = DataLoader(dataset, batch_size=None, num_workers=workers)
        iterator = iter(loader)
        values = [next(iterator) for _ in range(count)]
        del iterator
        return values

    original = batches(0, 0, 4)
    parallel = batches(2, 0, 4)
    resumed = batches(2, 2, 2)
    for actual, expected in [(parallel, original), (resumed, original[2:])]:
        for a, b in zip(actual, expected):
            assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not torch.equal(original[0][0], original[1][0])


def test_trainer_resume_matches_uninterrupted_updates(tmp_path, monkeypatch):
    from tabicl.train import _run
    from tabicl.train._train_config import build_parser

    # Keep the real prior/trainer/optimizer/checkpoint path; CPU needs no pinned
    # CUDA loader. Worker-independent replay is exercised separately above.
    def cpu_loader(dataset, **kwargs):
        return DataLoader(dataset, batch_size=None, num_workers=0)

    monkeypatch.setattr(_run, "DataLoader", cpu_loader)

    def trainer(directory):
        options = dict(
            device="cpu",
            amp=False,
            muon=False,
            max_steps=4,
            batch_size=2,
            batch_size_per_gp=1,
            micro_batch_size=1,
            min_features=3,
            max_features=3,
            max_seq_len=64,
            min_train_size=0.7,
            max_train_size=0.8,
            prior_type="graph_scm",
            prior_seed_by_step=True,
            prior_device="cpu",
            n_jobs=1,
            embed_dim=16,
            col_num_blocks=1,
            col_nhead=2,
            col_num_inds=4,
            row_num_blocks=1,
            row_nhead=2,
            row_num_cls=2,
            icl_num_blocks=3,
            icl_nhead=2,
            icl_ssmax=False,
            icl_attention="global_sparse",
            icl_partition="content_tree",
            icl_block_size=16,
            icl_num_selected_blocks=2,
            zero_init=False,
            recompute=False,
            use_flash_attn3=False,
            fail_fast=True,
            wandb_log=False,
            tensorboard_log=False,
            checkpoint_dir=directory,
            save_temp_every=2,
            save_perm_every=100,
            icl_diagnostics_every=0,
        )
        tokens = [v for k, value in options.items() for v in ("--" + k, str(value))]
        return _run.Trainer(build_parser().parse_args(tokens))

    complete = trainer(tmp_path / "complete")
    complete.train()
    interrupted = trainer(tmp_path / "resumed")
    original = interrupted.run_batch

    def stop_after_two(batch):
        if interrupted.curr_step == 2:
            raise InterruptedError("simulated interruption")
        return original(batch)

    interrupted.run_batch = stop_after_two
    with pytest.raises(InterruptedError):
        interrupted.train()
    resumed = trainer(tmp_path / "resumed")
    assert resumed.curr_step == resumed.dataloader.dataset.start_step == 2
    resumed.train()
    assert resumed.scheduler.state_dict() == complete.scheduler.state_dict()
    for name, value in complete.raw_model.state_dict().items():
        assert torch.equal(value, resumed.raw_model.state_dict()[name]), name


@pytest.mark.parametrize("regression", [False, True])
def test_fixed_validation_restores_rng_modes_and_ignores_query_labels(regression):
    model = TabICL(
        **small_config(
            icl_attention="global_sparse",
            icl_partition="content_tree",
            max_classes=0 if regression else 10,
            num_quantiles=9,
            icl_block_size=16,
            icl_num_selected_blocks=2,
        )
    )
    model.eval()
    model.icl_predictor.train()  # Preserve mixed per-module mode, too.
    modes = [m.training for m in model.modules()]
    x = torch.randn(1, 64, 3)
    y = torch.randn(1, 64) if regression else torch.randint(0, 3, (1, 64))
    data = dict(cases=[dict(id="heldout", x=x, y=y, train_size=48)])
    outputs = []
    handle = model.register_forward_hook(
        lambda m, args, result: outputs.append(result.detach().clone())
    )
    state = torch.random.get_rng_state()
    np_state, py_state = np.random.get_state(), random.getstate()
    metrics, records = evaluate_cases(
        model, data, "cpu", nullcontext(), diagnostic_queries=4
    )
    assert torch.equal(state, torch.random.get_rng_state())
    assert all(np.array_equal(a, b) for a, b in zip(np_state, np.random.get_state()))
    assert py_state == random.getstate()
    assert modes == [m.training for m in model.modules()]
    assert all(p.grad is None for p in model.parameters())
    altered = y.clone()
    altered[:, 48:] = y[:, 48:] + (2 if regression else 1)
    evaluate_cases(
        model,
        dict(cases=[dict(id="heldout", x=x, y=altered, train_size=48)]),
        "cpu",
        nullcontext(),
        diagnostic_queries=0,
    )
    assert torch.equal(outputs[0], outputs[1])
    assert metrics["val/support_48/pinball" if regression else "val/support_48/ce"] > 0
    assert any(k.startswith("diag/") for k in records[0])
    handle.remove()

    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="heldout"):
        evaluate_cases(model, data, "cpu", nullcontext())
    assert torch.equal(state, torch.random.get_rng_state())
    assert modes == [m.training for m in model.modules()]
