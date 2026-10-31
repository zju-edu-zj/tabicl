"""Profiling boundaries must preserve outputs/gradients and count ranks correctly."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tabicl._model.tabicl import TabICL
from tabicl.prior import _node_pool
from tabicl.train._run import Trainer
from tabicl.train._train_config import build_parser
from test_node_prior_pool import small_prior


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("recompute", [False, True])
def test_stage_tensor_hooks_preserve_real_model_gradients(monkeypatch, recompute):
    module = script("profile_training_pipeline")
    model = TabICL(max_classes=3, embed_dim=16, col_num_blocks=1, col_nhead=2,
                   col_num_inds=4, row_num_blocks=1, row_nhead=2, row_num_cls=2,
                   icl_num_blocks=3, icl_nhead=2, icl_attention="global_sparse",
                   icl_ssmax=False, icl_block_size=16, icl_num_selected_blocks=1,
                   zero_init=False, recompute=recompute).train()
    reference = deepcopy(model)
    count = []

    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self):
            self.index = len(count)
            count.append(self)

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(Trainer, "build_model", lambda self: setattr(self, "raw_model", model))
    trainer = module.ProfileTrainer.__new__(module.ProfileTrainer)
    trainer.active = None
    trainer.tracer = None
    trainer.stage_scopes = {}
    trainer.build_model()
    trainer.active = {}
    x, y = torch.randn(1, 48, 4), torch.randint(3, (1, 32))
    output = model(x, y)
    expected = reference(x, y)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.square().mean().backward()
    expected.square().mean().backward()
    trainer.event("end")
    for a, b in zip(model.parameters(), reference.parameters()):
        if b.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    names = ("col_start", "col_end", "row_start", "row_end", "icl_start", "icl_end",
             "icl_grad", "row_grad", "col_grad", "end")
    indices = [trainer.active[name].index for name in names]
    assert indices == sorted(indices)


def test_trace_component_wrappers_preserve_values_gradients_and_restore_on_error():
    module = script("profile_training_pipeline")
    from tabicl._model.layers import GlobalLinearAttention
    operator = GlobalLinearAttention(16, 2, qk_norm=False)
    inputs = [torch.randn(1, 2, n, 8, requires_grad=True) for n in (7, 11, 11)]
    original = GlobalLinearAttention._global_attention
    reference = operator._global_attention(*inputs)
    gradients = torch.autograd.grad(reference.square().sum(), inputs)
    with pytest.raises(RuntimeError, match="restore"):
        with module.trace_component_scopes():
            output = operator._global_attention(*inputs)
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
            for a, b in zip(torch.autograd.grad(output.square().sum(), inputs), gradients):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            raise RuntimeError("restore")
    assert GlobalLinearAttention._global_attention is original


def test_cpu_profile_wrappers_preserve_generated_tables():
    module = script("profile_graph_generation")
    prior = small_prior(batch_size=1)
    _node_pool._init_worker(prior)
    p = dict(seq_len=48, train_size=24, num_features=4, max_features=6,
             num_classes=3, regression=False, device="cpu", config=prior.config)
    record = module.profile((p, 0, 1, None))
    x, y, *_ = _node_pool._generate(p, record["seed"])
    assert record["sha256"] == hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
    assert record["graph_s"] > 0
    assert record["feature_cleanup_s"] > 0
    assert record["class_check_s"] > 0
    assert record["process_cpu_s"] > 0
    assert record["graph_s"] + record["filter_s"] <= record["total_s"]


def test_summary_uses_complete_steps_and_weighted_stage_totals(tmp_path):
    module = script("summarize_pipeline_profile")
    for rank in range(2):
        rows = []
        for step in range(3):
            rows.append(dict(step=step, rank=rank, prior_wait_s=rank,
                        prior_start_monotonic_s=step*10+rank*.5,
                        train_end_monotonic_s=step*10+rank*.5+rank+2,
                        train_wall_s=2, train_stream_ms=2000, optimizer_ms=1,
                        outside_micros_stream_ms=1, oom_micro_batches=0, loss=1.,
                        micros=[dict(micro_size=1, rows=10, width=rank+1, col_forward_ms=1, row_forward_ms=1,
                                     icl_forward_ms=2, col_backward_and_sync_ms=1,
                                     row_backward_ms=1, icl_backward_ms=2)]))
        (tmp_path / f"rank{rank}.jsonl").write_text("\n".join(map(json.dumps, rows))+"\n")
    result = module.summarize(tmp_path, 1)
    assert result["ranks"] == 2 and result["steps"] == 2 and result["tables"] == 4
    assert result["icl_forward_fraction"] == .5
    assert result["icl_forward_backward_fraction_including_col_sync"] == .5
    assert result["step_mean"]["wait_train_max_s"] == 3
    assert result["step_mean"]["aligned_step_span_s"] == 3.5
    assert result["workload_mean_per_table"]["rows"] == 10
    assert result["workload_mean_per_table"]["row_features"] == 15
    with pytest.raises(ValueError, match="missing ranks"):
        module.summarize(tmp_path, 1, expected_ranks=3)
    file = tmp_path / "rank0.jsonl"
    saved = file.read_text()
    rows = [json.loads(line) for line in saved.splitlines()]
    rows[1]["loss"] = float("nan")
    file.write_text("\n".join(map(json.dumps, rows))+"\n")
    with pytest.raises(ValueError, match="nonfinite"):
        module.summarize(tmp_path, 1)
    file.write_text(saved)
    # Detect a rank that exits early, instead of silently undercounting it.
    (tmp_path / "rank1.jsonl").write_text("")
    with pytest.raises(ValueError, match="missing ranks"):
        module.summarize(tmp_path, 1)


def test_kineto_summary_preserves_nested_attribution_and_overlapping_streams(tmp_path):
    module = script("summarize_kineto_trace")
    events = [dict(ph="X", cat="user_annotation", name="tabicl/icl_forward", pid=1, tid=1, ts=0, dur=50),
              dict(ph="X", cat="gpu_user_annotation", name="tabicl/icl_forward", pid=0, tid=7, ts=10, dur=40),
              dict(ph="X", cat="user_annotation", name="tabicl/icl/centroid_route", pid=1, tid=1, ts=10, dur=20),
              dict(ph="X", cat="cpu_op", name="aten::bmm", pid=1, tid=1, ts=15, dur=2,
                   args={"External id": 12}),
              dict(ph="X", cat="kernel", name="gemm", pid=0, tid=7, ts=20, dur=20,
                   args={"External id": 12, "device": 0}),
              dict(ph="X", cat="kernel", name="gemm", pid=0, tid=8, ts=30, dur=20,
                   args={"External id": 12, "device": 0}),
              dict(ph="X", cat="gpu_memcpy", name="HtoD", pid=0, tid=8, ts=10, dur=5,
                   args={"device": 0})]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(dict(traceEvents=events)))
    result = module.summarize(path)
    assert result["devices"]["0"] == dict(busy_union_ms=.035, activity_span_ms=.04)
    assert result["duration_sum_ms_by_kind"] == dict(kernel=.04, memcpy=.005)
    assert result["inclusive_components"]["tabicl/icl_forward"]["calls"] == 1
    assert result["inclusive_components"]["tabicl/icl_forward"]["cpu_range_sum_ms"] == .05
    assert result["inclusive_components"]["tabicl/icl_forward"]["top_gpu_activities"] == [
        dict(name="gemm", calls=2, duration_sum_ms=.04)]
    assert result["unlinked_gpu_activities"] == 1
    for name in ("tabicl/icl_forward", "tabicl/icl/centroid_route"):
        assert result["inclusive_components"][name]["gpu_duration_sum_ms"] == .04
    events.extend([dict(ph="X", cat="cuda_driver", name="cuLaunchKernelEx", pid=1, tid=1, ts=18, dur=1,
                        args={"correlation": 81}),
                   dict(ph="X", cat="kernel", name="triton_route", pid=0, tid=7, ts=22, dur=10,
                        args={"device": 0, "correlation": 81})])
    path.write_text(json.dumps(dict(traceEvents=events)))
    result = module.summarize(path)
    assert result["attribution_links"] == dict(external_id=2, launch_correlation=1)
    assert result["unlinked_gpu_duration_sum_ms"] == .005
    assert result["inclusive_components"]["tabicl/icl/centroid_route"]["gpu_duration_sum_ms"] == .05
    path.write_text(json.dumps(dict(traceEvents=events[:3])))
    with pytest.raises(ValueError, match="no supported GPU"):
        module.summarize(path)


def test_step_report_separates_prefetched_work_exposed_wait_and_gpu_busy(tmp_path):
    module = script("summarize_pipeline_profile")
    for rank in range(2):
        row = dict(step=4, rank=rank, prior_wait_s=2, prior_start_monotonic_s=10.,
                   train_start_monotonic_s=12., train_end_monotonic_s=15., train_wall_s=3,
                   train_stream_ms=3000, oom_micro_batches=0, loss=1.,
                   cpu_generation=dict(process_cpu_s=5, first_generation_monotonic_s=5.,
                                       generation_ready_monotonic_s=11., submitted_monotonic_s=4.,
                                       service_wait_s=1.),
                   micros=[dict(micro_size=1, col_forward_ms=1, row_forward_ms=1, icl_forward_ms=2,
                                icl_backward_ms=2, row_backward_ms=1, col_backward_and_sync_ms=1)])
        (tmp_path / f"rank{rank}.jsonl").write_text(json.dumps(row)+"\n")
        trace = dict(optimizer_steps={"4": dict(kernel_busy_union_ms=2000,
                                              communication_busy_union_ms=500, activity_busy_union_ms=2400)})
        (tmp_path / f"rank{rank}-trace-summary.json").write_text(json.dumps(trace))
    step = module.summarize(tmp_path, 0)["steps_detail"][0]
    assert step["cpu_generation_core_s"] == 10
    assert step["cpu_generation_span_s"] == 6
    assert step["cpu_submit_to_ready_s"] == 7
    assert step["generation_wait_rank_max_s"] == step["generation_wait_all_ranks_s"] == 1
    assert step["gpu_kernel_busy_union_rank_max_s"] == 2
    assert step["gpu_activity_busy_union_rank_max_s"] == 2.4
    # A balanced rank can have its own tables ready before the cohort finishes;
    # this wait is still caused by data generation, not transport.
    path = tmp_path / "rank0.jsonl"
    row = json.loads(path.read_text())
    row["cpu_generation"]["generation_ready_monotonic_s"] = 9.
    row["cpu_generation"]["dispatch_generation_ready_monotonic_s"] = 11.
    path.write_text(json.dumps(row)+"\n")
    step = module.summarize(tmp_path, 0)["steps_detail"][0]
    assert step["generation_wait_rank_mean_s"] == step["generation_wait_all_ranks_s"] == 1


def test_trace_attributes_kernel_and_communication_to_optimizer_step(tmp_path):
    module = script("summarize_kineto_trace")
    events = [dict(ph="X", cat="user_annotation", name="tabicl/optimizer_step/4", pid=1, tid=1, ts=0, dur=100),
              dict(ph="X", cat="cpu_op", name="aten::mm", pid=1, tid=1, ts=5, dur=2,
                   args={"External id": 5}),
              dict(ph="X", cat="kernel", name="gemm", pid=0, tid=7, ts=10, dur=40,
                   args={"External id": 5, "device": 0}),
              dict(ph="X", cat="cuda_runtime", name="launch", pid=1, tid=1, ts=7, dur=1,
                   args={"correlation": 9}),
              dict(ph="X", cat="kernel", name="lcclAllReduce", pid=0, tid=8, ts=30, dur=50,
                   args={"correlation": 9, "device": 0})]
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(dict(traceEvents=events)))
    result = module.summarize(path)["optimizer_steps"]["4"]
    assert result == dict(kernel_busy_union_ms=.04, communication_busy_union_ms=.05, activity_busy_union_ms=.07)


@pytest.mark.parametrize("stage,dtype,profile,reference", [
    (2, "bfloat16", False, False), (3, "float32", True, False),
    (3, "bfloat16", False, True), (2, "float32", True, True)])
def test_benchmark_launch_preserves_real_recipe_and_cohort_flags(tmp_path, monkeypatch, stage, dtype, profile, reference):
    module = script("benchmark_training_schedule")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, DRY_RUN="1", NUM_GPUS="16", PRIOR_BACKEND="node_pool", PRIOR_SCHEDULE="shape_cohort",
               ADAPTIVE_FEATURE_LIMIT="False", TRAIN_DTYPE=dtype)
    recipe = subprocess.check_output(["bash", str(root / f"scripts/train_v2_clf_stage{stage}.sh")],
                                     env=env, cwd=root, text=True)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: recipe)
    commands = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs:
                        commands.append(command) or SimpleNamespace(returncode=0))
    argv = ["benchmark", "--gpus", "16", "--stage", str(stage), "--dtype", dtype,
            "--steps", "30", "--output", str(tmp_path)]
    if profile:
        argv += ["--profile-pipeline", "--trace-all-ranks", "--trace-step", "20", "--shape-prior"]
    if reference:
        argv += ["--reference-centroid-topk", "--shape-prior"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 0
    command = commands[0]
    if profile:
        entry = command.index(str(root / "scripts/profile_training_pipeline.py"))
        own = script("profile_training_pipeline")
        options, flags = own.profile_parser().parse_known_args(command[entry+1:])
        assert options.trace_all_ranks and options.profile_shape_prior
        assert options.reference_centroid_topk == reference
    elif reference:
        entry = command.index(str(root / "scripts/train_reference_centroid_topk.py"))
        flags = command[entry+1:]
        assert flags.pop() == "--shape-prior"
    else:
        flags = command[command.index("tabicl.train")+1:]
    config = build_parser().parse_args(flags)
    assert config.prior_schedule == "shape_cohort" and not config.adaptive_feature_limit
    assert config.dtype == dtype and config.max_steps == 30 and config.micro_batch_size == 1
    assert config.max_seq_len == (10240 if stage == 2 else 60000)
    assert not config.checkpoint_path
