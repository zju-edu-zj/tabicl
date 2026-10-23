from pathlib import Path
import os
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from tabicl.train import _run
from tabicl.train._run import Trainer
from tabicl.train._train_config import build_parser


def prior_config(**overrides):
    values = dict(
        prior_dir=None, regression_method=None, batch_size=4, batch_size_per_gp=4,
        min_features=1, max_features=4, max_classes=3, min_seq_len=None,
        max_seq_len=32, log_seq_len=False, log_n_features=False,
        seq_len_per_gp=True, min_train_size=0.3, max_train_size=0.9,
        replay_small=False, prior_type="dummy", prior_device="cpu", n_jobs=2,
        prior_prefetch_factor=7, prior_in_order=False, prior_seed_by_step=False,
        np_seed=42, torch_seed=43, load_prior_start=0, delete_after_load=False,
    )
    defaults = build_parser().parse_args([])
    values.update({key: getattr(defaults, key) for key in vars(defaults) if key not in values})
    values.update(overrides)
    return SimpleNamespace(**values)


def test_configure_prior_passes_ready_first_queue_settings(monkeypatch):
    captured = {}

    class Loader:
        def __init__(self, dataset, in_order=True, **kwargs):
            captured.update(kwargs)
            captured["in_order"] = in_order
            self.dataset = dataset

    monkeypatch.setattr(_run, "DataLoader", Loader)
    trainer = Trainer.__new__(Trainer)
    trainer.config = prior_config()
    trainer.regression = False
    trainer.master_process = False
    trainer.ddp_rank = 0
    trainer.ddp_world_size = 1
    trainer.configure_prior()
    assert captured["num_workers"] == 2
    assert captured["prefetch_factor"] == 7
    assert captured["persistent_workers"] is True
    assert captured["in_order"] is False


def test_step_seeded_prior_rejects_out_of_order_loading():
    trainer = Trainer.__new__(Trainer)
    trainer.config = prior_config(prior_seed_by_step=True)
    trainer.regression = False
    trainer.master_process = False
    trainer.ddp_rank = 0
    trainer.ddp_world_size = 1
    with pytest.raises(ValueError, match="prior_in_order=False"):
        trainer.configure_prior()


def test_stage1_prior_pipeline_defaults_match_128_core_node(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ, DRY_RUN="1", NUM_GPUS="16",
        CKPT_DIR=str(tmp_path / "checkpoints"), LOG_DIR=str(tmp_path / "logs"),
    )
    text = subprocess.check_output(["bash", str(root / "scripts/train_v2_clf_stage1.sh")], env=env, text=True)
    tokens = shlex.split(text)
    args = build_parser().parse_args(tokens[tokens.index("--wandb_log"):])
    assert args.n_jobs == 8
    assert args.prior_prefetch_factor == 8
    assert args.prior_in_order is False
    assert args.prior_backend == "node_pool"
    assert args.prior_pool_workers == 112


@pytest.mark.parametrize("task", ["clf", "reg"])
@pytest.mark.parametrize("stage", [1, 2, 3])
def test_all_curriculum_scripts_support_shared_pool_and_override(tmp_path, task, stage):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, DRY_RUN="1", PRIOR_BACKEND="node_pool", PRIOR_POOL_WORKERS="12",
               PRIOR_POOL_WINDOW="7", PRIOR_POOL_MB="128", CKPT_DIR=str(tmp_path), LOG_DIR=str(tmp_path))
    text = subprocess.check_output(["bash", str(root / f"scripts/train_v2_{task}_stage{stage}.sh")], env=env, text=True)
    tokens = shlex.split(text)
    args = build_parser().parse_args(tokens[tokens.index("--wandb_log"):])
    assert args.prior_pool_workers == 12
    assert args.prior_pool_window == 7
    assert args.prior_pool_mb == 128
    assert args.prior_backend == "node_pool"
