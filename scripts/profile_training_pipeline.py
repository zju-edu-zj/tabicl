"""Real training profile with per-rank CUDA stream boundaries, not CPU enqueue times.

Use via benchmark_training_schedule.py --profile-pipeline. No per-stage CUDA
synchronization and no backward-module hooks (which can alter compile/autograd).
Backward boundaries are tensor gradient hooks; column backward includes DDP's
tail and is explicitly labelled. Optional Kineto trace measures actual kernels,
copies and collective overlap; run that separately from throughput measurements.
"""
import argparse
from contextlib import contextmanager, ExitStack
from functools import wraps
import json
from pathlib import Path
import time

import torch

from tabicl.train._run import Trainer
from tabicl.train._train_config import build_parser


@contextmanager
def trace_component_scopes():
    """Trace-only labels around eager ICL components; restore even on error.

    Nested ranges are inclusive. No wrappers run during normal throughput
    measurements, warmup or the untraced profile steps.
    """
    from tabicl._model import _content_tree
    from tabicl._model.icl_attention import SetRetrievalAttention
    from tabicl._model.layers import GlobalLinearAttention

    targets = [(_content_tree, "content_tree_order", "icl/content_tree"),
               (SetRetrievalAttention, "_prepare_route", "icl/prepare_route"),
               (SetRetrievalAttention, "route", "icl/centroid_route"),
               (SetRetrievalAttention, "_sparse_attention", "icl/sparse_including_route"),
               (GlobalLinearAttention, "_global_attention", "icl/global_attention")]

    def scoped(original, label):
        @wraps(original)
        def call(*args, **kwargs):
            with torch.profiler.record_function(f"tabicl/{label}"):
                return original(*args, **kwargs)
        return call

    with ExitStack() as stack:
        for owner, name, label in targets:
            original = getattr(owner, name)
            stack.callback(setattr, owner, name, original)
            setattr(owner, name, scoped(original, label))
        yield


class ProfileTrainer(Trainer):
    def __init__(self, config, output, trace_step=-1, trace_steps=1, trace_all_ranks=False,
                 reference_centroid_topk=False):
        self.active = None
        self.trace_step, self.trace_steps = trace_step, trace_steps
        self.trace_all_ranks = trace_all_ranks
        self.reference_centroid_topk = reference_centroid_topk
        self.tracer = None
        self.trace_scopes = None
        self.stage_scopes = {}
        self.batch_events = None
        self.prior_wait_s = 0.
        super().__init__(config)
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.stream = (output / f"rank{self.ddp_rank}.jsonl").open("w", buffering=1)
        self.micro_records = []
        def optimizer_before(*args):
            if self.batch_events is not None:
                self.event("optimizer_start", self.batch_events)
        def optimizer_after(*args):
            if self.batch_events is not None:
                self.event("optimizer_end", self.batch_events)
        self.optimizer.register_step_pre_hook(optimizer_before)
        self.optimizer.register_step_post_hook(optimizer_after)

    def fetch_batch(self, iterator):
        # Preserve the loader type, resume and cleanup contracts.
        self.prior_started_s = time.perf_counter()
        result = super().fetch_batch(iterator)
        self.prior_finished_s = time.perf_counter()
        self.prior_wait_s = self.prior_finished_s - self.prior_started_s
        self.cpu_generation = getattr(self.dataloader, "last_batch_timing", {})
        return result

    def event(self, name, record=None):
        record = self.active if record is None else record
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        record[name] = event

    def build_model(self):
        super().build_model()
        if getattr(self, "reference_centroid_topk", False):
            from tabicl._model.icl_attention import SetRetrievalAttention
            for module in self.raw_model.modules():
                if isinstance(module, SetRetrievalAttention):
                    module.use_fast_centroid_topk = False
        for name, module in (("col", self.raw_model.col_embedder),
                             ("row", self.raw_model.row_interactor),
                             ("icl", self.raw_model.icl_predictor)):
            def before(module, args, name=name):
                if self.active is not None:
                    if self.tracer is not None:
                        scope = torch.profiler.record_function(f"tabicl/{name}_forward")
                        scope.__enter__()
                        self.stage_scopes[name] = scope
                    self.event(f"{name}_start")

            def after(module, args, output, name=name):
                if self.active is not None:
                    self.event(f"{name}_end")
                    if name in self.stage_scopes:
                        self.stage_scopes.pop(name).__exit__(None, None, None)
                    record = self.active
                    if output.requires_grad:
                        def gradient(grad):
                            self.event(f"{name}_grad", record)
                            return grad
                        output.register_hook(gradient)

            module.register_forward_pre_hook(before)
            module.register_forward_hook(after)

    def move_micro_batch(self, *tensors):
        if self.active is None:
            return super().move_micro_batch(*tensors)
        self.event("h2d_start")
        started = time.perf_counter()
        result = super().move_micro_batch(*tensors)
        self.active["h2d_host_s"] = time.perf_counter()-started
        self.event("h2d_end")
        return result

    def run_micro_batch(self, batch, index, count):
        self.active = dict(micro=index, rows=int(batch[3][0]), train_size=int(batch[4][0]),
                           width=int(batch[2].max()), micro_size=len(batch[2]))
        self.event("start")
        try:
            result = super().run_micro_batch(batch, index, count)
            self.event("end")
            self.micro_records.append(self.active)
            return result
        finally:
            self.active = None

    def run_batch(self, batch):
        # Exclude the possibly larger synthetic compilation warmup peak.
        torch.cuda.reset_peak_memory_stats()
        if (self.ddp_rank == 0 or self.trace_all_ranks) and self.curr_step == self.trace_step:
            self.tracer = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA],
                                                record_shapes=False)
            self.tracer.__enter__()
            self.trace_scopes = trace_component_scopes()
            self.trace_scopes.__enter__()
        self.micro_records = []
        batch_events = {}
        self.batch_events = batch_events
        self.event("start", batch_events)
        with torch.profiler.record_function(f"tabicl/optimizer_step/{self.curr_step}"):
            started = time.perf_counter()
            results = super().run_batch(batch)
            self.event("end", batch_events)
            # run_batch already copies scalar metrics to CPU once per step. The
            # event wait resolves only the final timestamp, never each stage.
            batch_events["end"].synchronize()
            wall = time.perf_counter() - started
        records = []
        for record in self.micro_records:
            row = {k: v for k, v in record.items() if not isinstance(v, torch.cuda.Event)}
            pairs = dict(host_prepare=("start", "h2d_start"), h2d=("h2d_start", "h2d_end"),
                         col_forward=("col_start", "col_end"),
                         frontend_preprocess=("h2d_end", "col_start"),
                         col_to_row=("col_end", "row_start"), row_to_icl=("row_end", "icl_start"),
                         row_forward=("row_start", "row_end"), icl_forward=("icl_start", "icl_end"),
                         loss_backward_setup=("icl_end", "icl_grad"),
                         icl_backward=("icl_grad", "row_grad"),
                         row_backward=("row_grad", "col_grad"),
                         col_backward_and_sync=("col_grad", "end"), micro=("start", "end"))
            for label, (a, b) in pairs.items():
                if a in record and b in record:
                    row[f"{label}_ms"] = record[a].elapsed_time(record[b])
            records.append(row)
        pool_stats = {}
        if hasattr(self.dataloader, "service"):
            if self.curr_step % 10 == 0:
                pool_stats = self.dataloader.stats()
        report = dict(step=self.curr_step, rank=self.ddp_rank, prior_wait_s=self.prior_wait_s,
                      prior_start_monotonic_s=self.prior_started_s, prior_end_monotonic_s=self.prior_finished_s,
                      train_start_monotonic_s=started, train_end_monotonic_s=started+wall,
                      train_wall_s=wall, train_stream_ms=batch_events["start"].elapsed_time(batch_events["end"]),
                      cpu_generation=self.cpu_generation,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(), micros=records,
                      pool=pool_stats, oom_micro_batches=results["oom_micro_batches"],
                      loss=float(results["pinball" if self.regression else "ce"]))
        if "optimizer_end" in batch_events:
            report["optimizer_ms"] = batch_events["optimizer_start"].elapsed_time(batch_events["optimizer_end"])
        report["outside_micros_stream_ms"] = report["train_stream_ms"] - sum(r["micro_ms"] for r in records)
        self.batch_events = None
        self.stream.write(json.dumps(report)+"\n")
        if self.tracer is not None and self.curr_step + 1 >= self.trace_step + self.trace_steps:
            self.trace_scopes.__exit__(None, None, None)
            self.trace_scopes = None
            self.tracer.__exit__(None, None, None)
            self.tracer.export_chrome_trace(str(self.output / f"rank{self.ddp_rank}-trace.json"))
            self.tracer = None
        return results

    def train(self):
        try:
            return super().train()
        finally:
            for scope in self.stage_scopes.values():
                scope.__exit__(None, None, None)
            self.stage_scopes.clear()
            if self.trace_scopes is not None:
                self.trace_scopes.__exit__(None, None, None)
            if self.tracer is not None:
                self.tracer.__exit__(None, None, None)
            self.stream.close()


def profile_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--profile-shape-prior", action="store_true")
    parser.add_argument("--trace-step", type=int, default=-1)
    parser.add_argument("--trace-steps", type=int, default=1)
    parser.add_argument("--trace-all-ranks", action="store_true")
    parser.add_argument("--reference-centroid-topk", action="store_true")
    return parser


def main():
    parser = profile_parser()
    options, args = parser.parse_known_args()
    if options.profile_shape_prior:
        from train_shape_prior import ShapeDataset
        from tabicl.train import _run
        _run.PriorDataset = ShapeDataset
    torch.multiprocessing.set_start_method("spawn", force=True)
    config = build_parser().parse_args(args)
    if not torch.cuda.is_available() or not str(config.device).startswith("cuda"):
        parser.error("CUDA required; CPU enqueue timings cannot replace GPU stage timings")
    ProfileTrainer(config, options.profile_output, options.trace_step, options.trace_steps,
                   options.trace_all_ranks, options.reference_centroid_topk).train()


if __name__ == "__main__":
    main()
