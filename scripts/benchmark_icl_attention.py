"""Reproducible CUDA benchmark; run with PYTHONPATH=src in the repository.

Timings include Python dispatch, synchronization and (train) forward/backward.
Compilation/warmup and correctness checks are outside measured samples.
"""

import argparse
import json
import statistics
import subprocess
import time
from types import SimpleNamespace
from pathlib import Path

import torch

from tabicl._model.encoders import Encoder
from tabicl._model.attention import HAS_FLASH_ATTN3
from tabicl._model.icl_attention import ICL_ATTENTION_TYPES, SetRetrievalAttention
from tabicl._model.kv_cache import KVCache
from tabicl._model.tabicl import TabICL
from tabicl.train._optimizer import build_optimizer


def compiled_variants():
    """Diagnostic only: counts cached Triton variants in this benchmark process."""
    from tabicl._model import _routing_triton, _sparse_triton

    names = {
        _routing_triton: (
            "_centroids",
            "_beam_search",
            "_content_pivots",
            "_content_scores",
        ),
        _sparse_triton: (
            "_forward",
            "_backward",
            "_edge_count",
            "_edge_scatter",
            "_backward_kv",
        ),
    }
    counts = {}
    for module, kernels in names.items():
        for name in kernels:
            cache = getattr(getattr(module, name), "device_caches", {})
            counts[name] = sum(len(entry[0]) for entry in cache.values())
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attention", choices=ICL_ATTENTION_TYPES, default="global_sparse")
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument(
        "--scope", choices=("encoder", "tabicl", "routing"), default="encoder"
    )
    p.add_argument("--phase", choices=("train", "prefill", "query"), default="train")
    p.add_argument("--router", choices=("centroid", "hierarchical"), default="centroid")
    p.add_argument("--router-beam", type=int, default=16)
    p.add_argument(
        "--partition", choices=("projection", "content_tree"), default="projection"
    )
    p.add_argument("--global-type", choices=("elu", "induced"), default="elu")
    p.add_argument("--induced-slots", type=int, default=128)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--selected-blocks", type=int, default=4)
    p.add_argument("--sparse-every", type=int, default=3)
    p.add_argument("--repeat", type=int, default=7)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16"
    )
    p.add_argument("--recompute", action="store_true")
    p.add_argument(
        "--optimizer-step",
        action="store_true",
        help="Include a hybrid Muon/AdamW update in train timings.",
    )
    p.add_argument("--compile", action="store_true", help="Compile the model with dynamic shapes.")
    p.add_argument(
        "--portable",
        action="store_true",
        help="Disable fused routing and sparse kernels.",
    )
    p.add_argument(
        "--vary-support",
        action="store_true",
        help="Cycle support lengths, including during warmup.",
    )
    p.add_argument("--memory-limit-gib", type=float, default=8)
    p.add_argument("--query-rows", type=int, default=256)
    p.add_argument(
        "--profile",
        type=Path,
        help="Request an untimed CPU/CUDA trace; report actual CUDA event coverage.",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.rows < 2 or args.batch_size < 1 or args.repeat < 1 or args.warmup < 1:
        p.error("rows >= 2, batch-size >= 1, repeat >= 1 and warmup >= 1 required")
    if args.optimizer_step and (args.phase != "train" or args.scope == "routing"):
        p.error("--optimizer-step requires encoder or tabicl training")
    if args.vary_support and (args.scope != "encoder" or args.phase != "train"):
        p.error("--vary-support requires encoder training")
    torch.manual_seed(123)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    total = torch.cuda.get_device_properties(0).total_memory
    if args.memory_limit_gib:
        torch.cuda.set_per_process_memory_fraction(
            args.memory_limit_gib * 2**30 / total
        )
    dtype = getattr(torch, args.dtype)
    s = args.rows * 4 // 5
    result = dict(
        vars(args),
        output=str(args.output),
        profile=str(args.profile) if args.profile else None,
        benchmark_version=6,
        support=s,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        memory_free_gib=torch.cuda.mem_get_info()[0] / 2**30,
        allow_tf32=True,
        seed=123,
        optimizer_step=args.optimizer_step,
        flash_attn3_available=HAS_FLASH_ATTN3,
    )
    result["gpu_state"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    records = []
    warmup_records = []
    model, x = None, None
    try:
        kw = dict(
            num_blocks=args.blocks,
            d_model=512,
            nhead=args.nhead,
            dim_feedforward=1024,
            attention_type=args.attention,
            sparse_every=args.sparse_every,
            block_size=args.block_size,
            num_selected_blocks=args.selected_blocks,
            router=args.router,
            router_beam=args.router_beam,
            partition=args.partition,
            global_type=args.global_type,
            num_induced_slots=args.induced_slots,
            qk_norm=True,
            recompute=args.recompute,
            zero_init=False,
        )
        if args.scope == "routing":
            model = SetRetrievalAttention(
                512,
                args.nhead,
                block_size=args.block_size,
                num_selected_blocks=args.selected_blocks,
                router=args.router,
                router_beam=args.router_beam,
                partition=args.partition,
                qk_norm=True,
            ).cuda()
            q = torch.randn(
                args.batch_size, args.nhead, args.rows, 512 // args.nhead, device="cuda", dtype=dtype
            )
            k = torch.randn(
                args.batch_size, args.nhead, s, 512 // args.nhead, device="cuda", dtype=dtype
            )

            def forward():
                return model.route(q, k)[1]

        elif args.scope == "tabicl":
            if args.phase != "train":
                p.error("tabicl scope currently measures training only")
            model = TabICL(
                embed_dim=128,
                row_num_cls=4,
                col_num_blocks=1,
                col_nhead=8,
                col_num_inds=16,
                row_num_blocks=1,
                row_nhead=8,
                icl_num_blocks=args.blocks,
                icl_nhead=args.nhead,
                icl_ssmax=False,
                icl_attention=args.attention,
                icl_sparse_every=args.sparse_every,
                icl_block_size=args.block_size,
                icl_num_selected_blocks=args.selected_blocks,
                icl_router=args.router,
                icl_router_beam=args.router_beam,
                icl_partition=args.partition,
                icl_global_type=args.global_type,
                icl_num_induced_slots=args.induced_slots,
                icl_qk_norm=True,
                recompute=args.recompute,
                zero_init=False,
            ).cuda()
            x = torch.randn(args.batch_size, args.rows, 8, device="cuda", requires_grad=True)
            y = torch.randint(0, 3, (args.batch_size, s), device="cuda")

            def forward():
                return model(x, y)

        else:
            model = Encoder(**kw).cuda()
            x = torch.randn(
                args.batch_size, args.rows, 512, device="cuda", requires_grad=args.phase == "train"
            )
            if args.phase in ("prefill", "query"):
                cache = KVCache()
                if args.phase == "query":
                    with (
                        torch.no_grad(),
                        torch.autocast(
                            "cuda", dtype=dtype, enabled=dtype != torch.float32
                        ),
                    ):
                        model.forward_with_cache(
                            x[:, :s], cache, train_size=s, store_cache=True
                        )
                    x = torch.randn(args.batch_size, args.query_rows, 512, device="cuda")

                    def forward():
                        return model.forward_with_cache(
                            x, cache, use_cache=True, store_cache=False
                        )

                else:
                    x = x[:, :s].contiguous()

                    def forward():
                        return model.forward_with_cache(
                            x, cache, train_size=s, store_cache=True
                        )

            else:

                def forward():
                    return model(x, train_size=s)

        if args.portable:
            for module in model.modules():
                if hasattr(module, "use_triton"):
                    module.use_triton = False
        training = args.phase == "train" and args.scope != "routing"
        optimizer = None
        if args.optimizer_step:
            optimizer = build_optimizer(
                model,
                SimpleNamespace(
                    muon=True, lr=4e-4, adamw_lr=1e-4, weight_decay=0.01,
                    beta1=0.9, beta2=0.999, use_cautious_wd=True,
                ),
            )
        if args.compile and args.scope != "routing":
            model = torch.compile(model, dynamic=True)
        model.train(training)
        for step in range(args.warmup + args.repeat):
            if args.vary_support:
                s = int(args.rows * (0.61, 0.67, 0.73, 0.79, 0.85)[step % 5])
            model.zero_grad(set_to_none=True)
            if x is not None:
                x.grad = None
            if step == args.warmup:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with (
                torch.set_grad_enabled(training),
                torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
            ):
                output = forward()
                if training:
                    target = output[:, s:] if args.scope == "encoder" else output
                    loss = target.float().square().mean()
            torch.cuda.synchronize()
            fwd = time.perf_counter()
            if training:
                loss.backward()
            torch.cuda.synchronize()
            backward_finished = time.perf_counter()
            if optimizer is not None:
                optimizer.step()
            torch.cuda.synchronize()
            finished = time.perf_counter()
            record = dict(total_s=finished - started, support=s)
            if args.vary_support:
                record["jit_variants"] = compiled_variants()
            if step >= args.warmup:
                records.append(dict(
                    record, fwd_s=fwd - started,
                    bwd_s=backward_finished - fwd, opt_s=finished - backward_finished,
                ))
            else:
                warmup_records.append(record)
            assert torch.isfinite(output).all(), "nonfinite output"
            if training:
                assert torch.isfinite(x.grad).all(), "nonfinite input gradient"
                assert all(
                    torch.isfinite(param.grad).all()
                    for param in model.parameters()
                    if param.grad is not None
                ), "nonfinite parameter gradient"
        result.update(
            status="ok",
            finite=True,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
        for key in ("fwd_s", "bwd_s", "opt_s", "total_s"):
            samples = [row[key] for row in records]
            result.update(
                {
                    "median_" + key: statistics.median(samples),
                    "min_" + key: min(samples),
                    "max_" + key: max(samples),
                }
            )
        if args.profile:
            model.zero_grad(set_to_none=True)
            if x is not None:
                x.grad = None
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                with (
                    torch.set_grad_enabled(training),
                    torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
                ):
                    output = forward()
                    if training:
                        target = output[:, s:] if args.scope == "encoder" else output
                        loss = target.float().square().mean()
                if training:
                    loss.backward()
                if optimizer is not None:
                    optimizer.step()
                torch.cuda.synchronize()
            args.profile.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(args.profile))
            result["profile_cuda_events"] = sum(
                event.device_type == torch.autograd.DeviceType.CUDA
                for event in prof.events()
            )
            args.profile.with_suffix(".txt").write_text(
                prof.key_averages().table(
                    sort_by=(
                        "self_device_time_total"
                        if result["profile_cuda_events"]
                        else "self_cpu_time_total"
                    ),
                    row_limit=35,
                )
            )
    except torch.OutOfMemoryError as error:
        result.update(
            status="oom",
            error=str(error),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
    result["records"] = records
    result["warmup_records"] = warmup_records
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
