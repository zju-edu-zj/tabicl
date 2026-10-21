"""Benchmark ICL attention branches concurrently across independent GPUs.

Each job uses one GPU. Multiple GPUs increase matrix throughput without adding
DDP communication to the measured model step. Results include raw JSON/logs, a
CSV summary, failures, and speedups relative to dense at the same row count.

Example:
  PYTHONPATH=src python scripts/benchmark_icl_branches.py \
    --output benchmarks/icl_branches --gpus 0 1 2 3 \
    --rows 1024 4096 16384 --batch-size 4 --optimizer-step
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "dense": ("dense", "elu", "projection"),
    "global_elu": ("global", "elu", "projection"),
    "global_induced": ("global", "induced", "projection"),
    "global_full_elu": ("global_full", "elu", "projection"),
    "global_sparse_elu": ("global_sparse", "elu", "projection"),
    "global_sparse_induced": ("global_sparse", "induced", "projection"),
    "global_sparse_alternating": ("global_sparse_alternating", "elu", "projection"),
    "sparse_all": ("sparse", "elu", "projection"),
    "global_sparse_content_tree": ("global_sparse", "elu", "content_tree"),
}
DEFAULT_VARIANTS = tuple(name for name in VARIANTS if name != "global_sparse_content_tree")


def assign_jobs(jobs, gpus, lengths):
    """Keep dense and alternatives for a sequence length on the same GPU."""
    assignments = [[] for _ in gpus]
    for index, length in enumerate(lengths):
        assignments[index % len(gpus)].extend(job for job in jobs if job["rows"] == length)
    return assignments


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=None, help="Physical GPU IDs; defaults to all visible GPUs.")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(DEFAULT_VARIANTS))
    parser.add_argument("--rows", nargs="+", type=int, default=[1024, 4096, 16384])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--scope", choices=("encoder", "tabicl"), default="encoder")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--memory-limit-gib", type=float, default=0)
    parser.add_argument("--recompute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizer-step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if min(args.rows) < 2 or min(args.batch_size, args.blocks, args.repeat, args.warmup) < 1:
        parser.error("positive sizes are required and rows must be >= 2")
    if args.gpus is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            args.gpus = [item.strip() for item in visible.split(",") if item.strip()]
        else:
            try:
                count = len(subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines())
            except (FileNotFoundError, subprocess.CalledProcessError):
                count = 0
            args.gpus = [str(index) for index in range(count)]
    if not args.gpus and not args.plan:
        parser.error("no GPUs detected; pass --gpus explicitly")
    return args


def main():
    args = parse_args()
    output = args.output.resolve()
    jobs = []
    for rows in args.rows:
        for label in args.variants:
            attention, global_type, partition = VARIANTS[label]
            name = f"{label}_r{rows}_b{args.batch_size}"
            result_path = output / f"{name}.json"
            command = [
                sys.executable, str(ROOT / "scripts/benchmark_icl_attention.py"),
                "--attention", attention, "--global-type", global_type,
                "--partition", partition, "--rows", str(rows),
                "--batch-size", str(args.batch_size), "--blocks", str(args.blocks),
                "--scope", args.scope, "--repeat", str(args.repeat),
                "--warmup", str(args.warmup), "--dtype", args.dtype,
                "--memory-limit-gib", str(args.memory_limit_gib),
                "--output", str(result_path),
            ]
            if args.recompute:
                command.append("--recompute")
            if args.optimizer_step:
                command.append("--optimizer-step")
            if args.compile:
                command.append("--compile")
            jobs.append(dict(name=name, label=label, rows=rows, command=command, result=result_path))
    assignments = assign_jobs(jobs, args.gpus, args.rows) if args.gpus else []
    if args.plan:
        print(json.dumps({gpu: [job["name"] for job in assigned]
                          for gpu, assigned in zip(args.gpus, assignments)}, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    for path in sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py")]):
        source_hash.update(str(path.relative_to(ROOT)).encode())
        source_hash.update(path.read_bytes())
    manifest = dict(
        command=sys.argv, gpus=args.gpus, source_sha256=source_hash.hexdigest(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        jobs=[{**job, "result": str(job["result"])} for job in jobs],
        assignments={gpu: [job["name"] for job in assigned]
                     for gpu, assigned in zip(args.gpus, assignments)},
    )
    (output / "manifest.json").write_text(json.dumps(manifest, default=str, indent=2) + "\n")

    def run_worker(gpu, assigned):
        completed = []
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="4")
        for job in assigned:
            print(f"[GPU {gpu}] {job['name']}", flush=True)
            log_path = output / f"{job['name']}.log"
            with log_path.open("w") as stream:
                process = subprocess.run(job["command"], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            completed.append((gpu, job, process.returncode))
        return completed

    completed = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(run_worker, gpu, assigned) for gpu, assigned in zip(args.gpus, assignments)]
        for future in futures:
            completed.extend(future.result())

    failures, rows = [], []
    for gpu, job, returncode in completed:
        if returncode or not job["result"].exists():
            failures.append(dict(job=job["name"], gpu=gpu, returncode=returncode))
            continue
        result = json.loads(job["result"].read_text())
        if result.get("status") != "ok":
            failures.append(dict(job=job["name"], gpu=gpu, status=result.get("status"), error=result.get("error")))
        rows.append(dict(
            job=job["name"], variant=job["label"], physical_gpu=gpu, status=result.get("status"),
            rows=job["rows"], support=result.get("support"), batch_size=args.batch_size,
            attention=result.get("attention"), global_type=result.get("global_type"),
            partition=result.get("partition"), median_fwd_ms=1000 * result.get("median_fwd_s", 0),
            median_bwd_ms=1000 * result.get("median_bwd_s", 0),
            median_opt_ms=1000 * result.get("median_opt_s", 0),
            median_total_ms=1000 * result.get("median_total_s", 0),
            peak_allocated_gib=result.get("peak_allocated_gib"), speedup_vs_dense=None,
        ))
    dense = {row["rows"]: row["median_total_ms"] for row in rows if row["variant"] == "dense" and row["status"] == "ok"}
    for row in rows:
        baseline = dense.get(row["rows"])
        if baseline and row["median_total_ms"]:
            row["speedup_vs_dense"] = baseline / row["median_total_ms"]
    rows.sort(key=lambda row: (row["rows"], args.variants.index(row["variant"])))
    if rows:
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "errors.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; inspect {output / 'errors.json'}")
    print(f"Completed {len(rows)} jobs; summary: {output / 'summary.csv'}")


if __name__ == "__main__":
    main()
