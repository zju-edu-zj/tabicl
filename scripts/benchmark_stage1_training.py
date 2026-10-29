"""Single-GPU Stage-1 Trainer benchmark, including first-use/compile latency.

Synthetic CPU batches exclude prior generation, checkpoints, logging and DDP.
run_batch includes transfers, loss checks, backward, clipping and hybrid updates.
Use separate processes (and PYTHONPATHs) for baseline/candidate comparisons.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import torch
from torch._dynamo.utils import counters

from tabicl._model.tabicl import TabICL
from tabicl.train._run import Trainer
from tabicl.train._train_config import build_parser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--mode", choices=["none", "row", "all"], default="all")
    parser.add_argument("--rows", type=int, nargs="+", default=[1024])
    parser.add_argument("--features", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--variable", action="store_true")
    parser.add_argument("--diagnostics", type=int, default=0)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--strict-fp32", action="store_true", help="Disable TF32 matmul")
    parser.add_argument("--profile", action="store_true", help="Profile one extra step, outside timings")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = not args.strict_fp32
    trainer = Trainer.__new__(Trainer)
    trainer.config = build_parser().parse_args([
        "--device", "cuda", "--dtype", args.dtype, "--batch_size", str(args.batch),
        "--micro_batch_size", str(args.micro_batch), "--muon", "True", "--lr", "0.0008",
        "--adamw_lr", "0.0001", "--weight_decay", "0.01", "--gradient_clipping", "10",
        "--warmup_steps", "5000", "--max_steps", "500000",
        "--icl_diagnostics_every", str(args.diagnostics), "--fail_fast", "True",
        "--ignore_d", "True", "--use_flash_attn3", "False",
    ])
    trainer.ddp = False
    trainer.master_process = True
    trainer.curr_step = 0
    trainer.regression = False
    trainer._disable_cudnn_sdp = True
    trainer.model = trainer.raw_model = TabICL(
        embed_dim=128, col_num_blocks=3, col_nhead=8, col_num_inds=128,
        col_feature_group="same", col_target_aware=True, col_ssmax="qassmax-mlp-elementwise",
        row_num_blocks=3, row_nhead=8, row_num_cls=4, row_rope_interleaved=False,
        icl_num_blocks=12, icl_nhead=8, icl_attention="global_sparse", icl_ssmax=False,
        icl_qk_norm=True, zero_init=False, recompute=args.recompute,
    ).cuda().train()
    trainer.model.compile_frontend(args.mode)
    trainer.configure_optimizer()
    trainer.configure_amp()
    rng = random.Random(917)
    widths = [4, 1, 2, 8, 13, 30, 64, args.features]
    widths += rng.sample(list(range(1, args.features + 1)), args.features)
    samples = []
    source = Path(__import__("tabicl._model.tabicl", fromlist=["__file__"]).__file__).parents[1]
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    report = dict(arguments=vars(args) | {"output": str(args.output)},
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  source=str(source), sha256=digest.hexdigest(),
                  tf32=torch.backends.cuda.matmul.allow_tf32,
                  inductor_cache=os.environ.get("TORCHINDUCTOR_CACHE_DIR", "default (shared)"),
                  scope=__doc__, samples=samples)
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch._dynamo.config.patch(fail_on_recompile_limit_hit=True):
            for step in range(args.steps):
                rows = args.rows[step % len(args.rows)]
                width = min(args.features, widths[step % len(widths)]) if args.variable else args.features
                support = int(rows * (.3 + .6 * ((step * 37) % 101) / 100)) if args.variable else int(rows * .6)
                labels = torch.randint(2 + step % 9 if args.variable else 10, (args.batch, rows)).float()
                batch = (torch.randn(args.batch, rows, width).pin_memory(), labels.pin_memory(),
                         torch.full((args.batch,), width), torch.full((args.batch,), rows),
                         torch.full((args.batch,), support))
                trainer.curr_step = step
                torch.cuda.synchronize()
                start = time.perf_counter()
                metrics = trainer.run_batch(batch)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                if not all(math.isfinite(v) for v in metrics.values()) or metrics["oom_micro_batches"]:
                    raise RuntimeError(f"Failed step: {metrics}")
                samples.append(dict(step=step, rows=rows, width=width, support=support,
                                    ms=elapsed, graphs=counters["stats"]["unique_graphs"], metrics=metrics))
                if step % 25 == 0:
                    print(json.dumps({k: v for k, v in samples[-1].items() if k != "metrics"}), flush=True)
        report["status"] = "ok"
        if args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                trainer.run_batch(batch)
                torch.cuda.synchronize()
            report["profile"] = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        values = [s["ms"] for s in samples]
        warm = values[5:] or values
        if values:
            report.update(mean_all_ms=statistics.mean(values), total_ms=sum(values),
                          median_after5_ms=statistics.median(warm), mean_after5_ms=statistics.mean(warm),
                          p95_ms=sorted(values)[math.ceil(.95 * len(values)) - 1], max_ms=max(values))
        report.update(peak_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      counters={k: dict(v) for k, v in counters.items()})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "profile")}, default=str))


if __name__ == "__main__":
    main()
