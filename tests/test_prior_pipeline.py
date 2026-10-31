from pathlib import Path
import os
import shlex
import subprocess
from types import SimpleNamespace

import pytest
import torch

from tabicl.train import _run
from tabicl.train._run import Trainer
from tabicl.train._train_config import build_parser
from tabicl.prior._genload import pad_prior_batch


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
    assert captured["collate_fn"] is pad_prior_batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for pinned memory")
def test_worker_padding_preserves_values_and_final_pinned_layout():
    features = torch.nested.nested_tensor([torch.randn(7, 3), torch.randn(11, 3)])
    labels = torch.nested.nested_tensor([torch.randn(7), torch.randn(11)])
    batch = (features, labels, torch.tensor([3, 3]), torch.tensor([7, 11]), torch.tensor([4, 8]))
    loader = torch.utils.data.DataLoader([batch], batch_size=None, num_workers=1,
                                       collate_fn=pad_prior_batch, pin_memory=True)
    actual = next(iter(loader))
    for left, right in zip(actual, batch):
        expected = right.to_padded_tensor(0.0) if right.is_nested else right
        torch.testing.assert_close(left, expected)
        assert not left.is_nested and left.is_pinned()


def test_step_seeded_prior_rejects_out_of_order_loading():
    trainer = Trainer.__new__(Trainer)
    trainer.config = prior_config(prior_seed_by_step=True)
    trainer.regression = False
    trainer.master_process = False
    trainer.ddp_rank = 0
    trainer.ddp_world_size = 1
    with pytest.raises(ValueError, match="prior_in_order=False"):
        trainer.configure_prior()


def test_stage1_defaults_adapt_workers_to_node_and_support_16_ranks(tmp_path):
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
    assert args.prior_backend == "dataloader"  # BF16 long-run default
    assert args.prior_pool_workers == 0
    assert args.prior_schedule == "none"  # one indivisible micro-batch per rank
    assert args.max_seq_len == 2048
    assert args.compile_frontend == "all" and args.compile_warmup
    assert args.icl_partition == "content_tree"


@pytest.mark.parametrize("stage,previous", [(2, "STAGE1_CKPT"), (3, "STAGE2_CKPT")])
def test_classifier_long_stage_defaults_are_fast_and_content_tree(tmp_path, stage, previous):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, DRY_RUN="1", NUM_GPUS="8", CKPT_DIR=str(tmp_path),
               **{previous: str(tmp_path / "input.ckpt")})
    script = root / f"scripts/train_v2_clf_stage{stage}.sh"
    text = subprocess.check_output(["bash", str(script)], env=env, text=True)
    tokens = shlex.split(text)
    args = build_parser().parse_args(tokens[tokens.index("--wandb_log"):])
    assert args.prior_backend == "node_pool"
    assert args.prior_schedule == "balanced"
    assert args.prior_pool_transport == "numpy"
    assert args.prior_pool_kernel_threads == 8
    assert args.compile_frontend == "all" and args.compile_warmup
    assert args.icl_partition == "content_tree"
    assert args.micro_batch_size == 1


@pytest.mark.parametrize("dtype,backend,schedule", [
    ("float32", "node_pool", "balanced"), ("bfloat16", "dataloader", "none"),
])
def test_stage1_default_backend_follows_long_run_measurement(tmp_path, dtype, backend, schedule):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, DRY_RUN="1", NUM_GPUS="8", TRAIN_DTYPE=dtype,
               CKPT_DIR=str(tmp_path), LOG_DIR=str(tmp_path))
    for key in ("PRIOR_BACKEND", "PRIOR_SCHEDULE"):
        env.pop(key, None)
    text = subprocess.check_output(["bash", str(root / "scripts/train_v2_clf_stage1.sh")], env=env, text=True)
    tokens = shlex.split(text)
    args = build_parser().parse_args(tokens[tokens.index("--wandb_log"):])
    assert (args.prior_backend, args.prior_schedule) == (backend, schedule)


@pytest.mark.parametrize("ranks,dtype,expected", [
    (8, "float32", "balanced"), (8, "bfloat16", "none"), (16, "float32", "none"),
])
def test_stage1_schedule_default_accounts_for_supply_and_micro_granularity(tmp_path, ranks, dtype, expected):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, DRY_RUN="1", NUM_GPUS=str(ranks), TRAIN_DTYPE=dtype,
               CKPT_DIR=str(tmp_path), LOG_DIR=str(tmp_path), PRIOR_BACKEND="node_pool")
    env.pop("PRIOR_SCHEDULE", None)
    text = subprocess.check_output(["bash", str(root / "scripts/train_v2_clf_stage1.sh")], env=env, text=True)
    tokens = shlex.split(text)
    args = build_parser().parse_args(tokens[tokens.index("--wandb_log"):])
    assert args.prior_schedule == expected


@pytest.mark.parametrize("task", ["clf", "reg"])
@pytest.mark.parametrize("stage", [1, 2, 3])
def test_all_curriculum_scripts_support_shared_pool_and_override(tmp_path, task, stage):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, DRY_RUN="1", PRIOR_BACKEND="node_pool", PRIOR_POOL_WORKERS="12",
               PRIOR_POOL_WINDOW="7", PRIOR_POOL_MB="128", PRIOR_SCHEDULE="window_bucketed",
               PRIOR_POOL_KERNEL_THREADS="8",
               PRIOR_POOL_TRANSPORT="mmap", PRIOR_BUCKET_STEPS="3",
               CKPT_DIR=str(tmp_path), LOG_DIR=str(tmp_path))
    text = subprocess.check_output(["bash", str(root / f"scripts/train_v2_{task}_stage{stage}.sh")], env=env, text=True)
    tokens = shlex.split(text)
    args = build_parser().parse_args(tokens[tokens.index("--wandb_log"):])
    assert args.prior_pool_workers == 12
    assert args.prior_pool_window == 7
    assert args.prior_pool_mb == 128
    assert args.prior_backend == "node_pool"
    assert args.prior_schedule == "window_bucketed"
    assert args.prior_pool_transport == "mmap" and args.prior_bucket_steps == 3
    assert args.prior_pool_kernel_threads == 8


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_precision_recipe_and_dataloader_escape_hatch(tmp_path, stage, dtype):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, DRY_RUN="1", TRAIN_DTYPE=dtype, PRIOR_BACKEND="dataloader",
               CKPT_DIR=str(tmp_path), LOG_DIR=str(tmp_path))
    env.pop("PRIOR_SCHEDULE", None)
    env.pop("USE_FLASH_ATTN3", None)
    text = subprocess.check_output(["bash", str(root / f"scripts/train_v2_clf_stage{stage}.sh")], env=env, text=True)
    tokens = shlex.split(text)
    args = build_parser().parse_args(tokens[tokens.index("--wandb_log"):])
    assert args.dtype == dtype and args.attention_backend == "auto"
    assert args.prior_backend == "dataloader" and args.prior_schedule == "none"
    assert not args.use_flash_attn3  # Compiled frontends use native SDPA.
