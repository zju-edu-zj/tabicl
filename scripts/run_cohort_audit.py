"""Run isolated throughput/profile pairs; use --dry-run to inspect the matrix.

Requires real CUDA/vendor devices. CPU profiles run separately, so they do not
compete with the online prior for cores. Never run cases concurrently.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def cases(stages, dtypes, seeds, schedules=("balanced", "window_fifo", "shape_cohort"),
          sources=("real", "shape"), crops=(True, False)):
    for stage in stages:
        for crop in (crops if stage == 3 else (True,)):
            for dtype in dtypes:
                for seed in seeds:
                    for source in sources:
                        shape_only = source == "shape"
                        for schedule in schedules:
                            yield dict(stage=stage, crop=crop, dtype=dtype, seed=seed,
                                       shape_only=shape_only, schedule=schedule)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--stages", type=int, nargs="+", choices=[2, 3], default=[2, 3])
    parser.add_argument("--dtypes", nargs="+", choices=["bfloat16", "float32"], default=["bfloat16"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--schedules", nargs="+", choices=["balanced", "window_fifo", "shape_cohort"],
                        default=["balanced", "window_fifo", "shape_cohort"])
    parser.add_argument("--sources", nargs="+", choices=["real", "shape"], default=["real", "shape"])
    parser.add_argument("--feature-limits", nargs="+", choices=["adaptive", "none"],
                        default=["adaptive", "none"], help="Stage3 crop cases; Stage2 cap is inactive")
    parser.add_argument("--phases", nargs="+", choices=["throughput", "profile"], default=["throughput", "profile"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--profile-steps", type=int, default=40)
    parser.add_argument("--skip", type=int, default=20)
    parser.add_argument("--trace", action="store_true", help="Add all-rank Kineto traces to each profile run")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--compile-cache", type=Path,
                        help="Optional shared warm cache; default measures independent cold startup")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.gpus < 1 or 64 % args.gpus or min(args.seeds) < 0 or args.skip < 0
            or min(args.steps, args.profile_steps) <= args.skip):
        parser.error("GPUs must divide 64; nonnegative seeds/skip and steps > skip required")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("use a fresh output directory to avoid automatic resume/mixed profiles")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
    matrix = list(cases(args.stages, args.dtypes, args.seeds, args.schedules, args.sources,
                        tuple(value == "adaptive" for value in args.feature_limits)))
    if not args.dry_run:
        (output / "matrix.json").write_text(json.dumps(matrix, indent=2)+"\n")
    for case in matrix:
        name = (f"s{case['stage']}-{case['dtype']}-crop{int(case['crop'])}-seed{case['seed']}-"
                f"{'shape' if case['shape_only'] else 'real'}-{case['schedule']}")
        env = dict(os.environ, PRIOR_BACKEND="node_pool", PRIOR_SCHEDULE=case["schedule"],
                   ADAPTIVE_FEATURE_LIMIT=str(case["crop"]), ICL_PARTITION="content_tree",
                   COMPILE_FRONTEND="all", COMPILE_WARMUP="True", RANK_TIMING_EVERY="1",
                   ICL_DIAGNOSTICS_EVERY="0")
        for phase in args.phases:
            profile = phase == "profile"
            directory = output / name / ("profile-run" if profile else "throughput")
            env["TORCHINDUCTOR_CACHE_DIR"] = str(
                args.compile_cache.resolve() if args.compile_cache else directory / "compile-cache")
            command = [sys.executable, str(root / "scripts/benchmark_training_schedule.py"),
                       "--gpus", str(args.gpus), "--stage", str(case["stage"]), "--dtype", case["dtype"],
                       "--steps", str(args.profile_steps if profile else args.steps), "--output", str(directory)]
            if args.initial_checkpoint:
                command += ["--initial-checkpoint", str(args.initial_checkpoint.resolve())]
            if case["shape_only"]:
                command += ["--shape-prior"]
            if profile:
                command += ["--profile-pipeline"]
                if args.trace:
                    command += ["--trace-step", str(args.skip), "--trace-all-ranks"]
            command += ["--train-args", "--np_seed", str(case["seed"]), "--fail_fast", "True"]
            settings = [f"{key}={env[key]}" for key in (
                "PRIOR_BACKEND", "PRIOR_SCHEDULE", "ADAPTIVE_FEATURE_LIMIT", "ICL_PARTITION",
                "COMPILE_FRONTEND", "COMPILE_WARMUP", "RANK_TIMING_EVERY", "ICL_DIAGNOSTICS_EVERY",
                "TORCHINDUCTOR_CACHE_DIR")]
            print(shlex.join(["env", *settings, *command]), flush=True)
            if args.dry_run:
                continue
            subprocess.run(command, cwd=root, env=env, check=True)
            if profile:
                from summarize_pipeline_profile import summarize
                summary = summarize(directory / "profile", args.skip, expected_ranks=args.gpus)
                if summary["steps"] != args.profile_steps - args.skip:
                    raise RuntimeError(f"incomplete profile measurement: {directory}")
            else:
                from summarize_training_schedule import summarize
                summary = summarize(directory, args.skip)
                if summary["oom_micro_batches"] or summary["nonfinite_loss"]:
                    raise RuntimeError(f"invalid throughput measurement: {directory}")
                if summary.get("rank_timing/step_max_s", {}).get("samples") != args.steps - args.skip:
                    raise RuntimeError(f"incomplete throughput measurement: {directory}")
            (directory / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
