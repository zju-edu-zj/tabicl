"""Run the five-branch latency/memory matrix sequentially on one CUDA GPU.

Includes 12-layer training, full TabICL, prefill, cached queries, variable
support, flat vs beam routing and portable vs fused implementations.
Subprocess isolation makes allocator peaks comparable and keeps OOM local.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "dense": ("dense", "projection"),
    "global": ("global", "projection"),
    "global_sparse_alternating": ("global_sparse_alternating", "projection"),
    "global_sparse_projection": ("global_sparse", "projection"),
    "global_sparse_content_tree": ("global_sparse", "content_tree"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, nargs="+", default=[1024, 4096, 8192, 16384, 65536])
    p.add_argument("--repeat", type=int, default=7)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--memory-limit-gib", type=float, default=8)
    p.add_argument("--matrix-only", action="store_true")
    p.add_argument("--compile", action="store_true", help="Compile every matrix model with dynamic shapes.")
    p.add_argument("--plan", action="store_true")
    args = p.parse_args()
    args.output = args.output.resolve()
    jobs = []

    def add(label, rows, attention, partition="projection", **options):
        stem = f"{label}_{rows}"
        cli = [
            "--attention",
            attention,
            "--partition",
            partition,
            "--rows",
            str(rows),
            "--repeat",
            str(args.repeat),
            "--warmup",
            str(args.warmup),
            "--memory-limit-gib",
            str(args.memory_limit_gib),
            "--output",
            str(args.output / (stem + ".json")),
        ]
        for key, value in options.items():
            if isinstance(value, bool):
                if value:
                    cli.append("--" + key.replace("_", "-"))
            else:
                cli.extend(("--" + key.replace("_", "-"), str(value)))
        jobs.append((stem, [sys.executable, str(ROOT / "scripts/benchmark_icl_attention.py"), *cli]))

    for rows in args.rows:
        for label, (attention, partition) in VARIANTS.items():
            add("train_" + label, rows, attention, partition, recompute=True, compile=args.compile)
    if not args.matrix_only:
        for label, (attention, partition) in VARIANTS.items():
            for phase in ("prefill", "query"):
                add(phase + "_" + label, 16384, attention, partition, phase=phase)
            for rows in (1024, 4096):
                add("tabicl_" + label, rows, attention, partition, scope="tabicl", recompute=True)
            add(
                "variable_" + label,
                16384,
                attention,
                partition,
                recompute=True,
                vary_support=True,
                warmup=max(5, args.warmup),
                repeat=max(10, args.repeat),
            )
            add("compiled_" + label, 4096, attention, partition, recompute=True, compile=True)
        for partition in ("projection", "content_tree"):
            for router in ("centroid", "hierarchical"):
                for rows in (16384, 65536):
                    add(
                        f"route_{partition}_{router}", rows, "global_sparse", partition, scope="routing", router=router
                    )
            for portable in (False, True):
                options = dict(blocks=3, recompute=True)
                if portable:
                    options["portable"] = True
                add(
                    f"kernel_{partition}_{'portable' if portable else 'fused'}",
                    4096,
                    "global_sparse",
                    partition,
                    **options,
                )
    if args.plan:
        print(json.dumps(jobs, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="4")
    sources = hashlib.sha256()
    for path in sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py")]):
        sources.update(str(path.relative_to(ROOT)).encode())
        sources.update(path.read_bytes())
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                command=sys.argv,
                source_sha256=sources.hexdigest(),
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                jobs=jobs,
            ),
            indent=2,
        )
        + "\n"
    )
    rows, errors = [], []
    for name, command in jobs:
        print(f"Benchmarking {name}", flush=True)
        with (args.output / (name + ".log")).open("w") as stream:
            done = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if done.returncode:
            errors.append(dict(job=name, returncode=done.returncode))
            print(f"FAILED: {name}; see its .log", flush=True)
            continue
        result = json.loads((args.output / (name + ".json")).read_text())
        rows.append(
            dict(
                job=name,
                status=result["status"],
                scope=result["scope"],
                phase=result["phase"],
                rows=result["rows"],
                attention=result["attention"],
                partition=result["partition"],
                router=result["router"],
                blocks=result["blocks"],
                portable=result["portable"],
                compiled=result["compile"],
                median_ms=result.get("median_total_s", 0) * 1000,
                min_ms=result.get("min_total_s", 0) * 1000,
                max_ms=result.get("max_total_s", 0) * 1000,
                peak_gib=result.get("peak_allocated_gib"),
            )
        )
        with (args.output / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    if errors:
        raise SystemExit(f"{len(errors)} benchmark subprocesses failed; inspect errors.json")


if __name__ == "__main__":
    main()
