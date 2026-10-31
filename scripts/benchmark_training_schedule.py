"""Run real curriculum scripts with a short, isolated DDP training workload.

Example: python scripts/benchmark_training_schedule.py --gpus 8 --steps 100
Uses this interpreter for torchrun, including vendor PyTorch environments.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape-prior", action="store_true",
                        help="Cheap synthetic tables isolate GPU scheduling; NOT real prior/quality training")
    parser.add_argument("--profile-pipeline", action="store_true", help="Per-rank GPU stage boundaries and input waits")
    parser.add_argument("--trace-step", type=int, default=-1, help="Optional rank-0 Kineto trace; adds overhead")
    parser.add_argument("--trace-steps", type=int, default=1)
    parser.add_argument("--trace-all-ranks", action="store_true")
    parser.add_argument("--initial-checkpoint", type=Path,
                        help="Optional common weights; default is isolated random initialization")
    parser.add_argument("--train-args", nargs=argparse.REMAINDER, default=[],
                        help="Additional Trainer flags, appended after curriculum defaults")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, NUM_GPUS=str(args.gpus), MAX_STEPS=str(args.steps), TRAIN_DTYPE=args.dtype,
               CKPT_DIR=str(args.output / "checkpoints"), LOG_DIR=str(args.output / "logs"),
               DRY_RUN="1", SAVE_TEMP_EVERY="1000000", SAVE_PERM_EVERY="1000000",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="2",
               TORCHINDUCTOR_COMPILE_THREADS="1", WANDB_LOG="False")
    text = subprocess.check_output(["bash", str(root / f"scripts/train_v2_clf_stage{args.stage}.sh")],
                                   env=env, cwd=root, text=True)
    tokens = shlex.split(text)
    command = [sys.executable, "-m", "torch.distributed.run", *tokens[1:],
               "--max_steps", str(args.steps),
               "--checkpoint_path", str(args.initial_checkpoint) if args.initial_checkpoint else "",
               "--save_temp_every", "1000000", "--save_perm_every", "1000000", *args.train_args]
    if args.profile_pipeline:
        entry = command.index("tabicl.train")
        command[entry-1:entry+1] = [str(root / "scripts/profile_training_pipeline.py"),
                                   "--profile-output", str(args.output / "profile"),
                                   "--trace-step", str(args.trace_step), "--trace-steps", str(args.trace_steps)]
        if args.shape_prior:
            command.append("--profile-shape-prior")
        if args.trace_all_ranks:
            command.append("--trace-all-ranks")
    elif args.shape_prior:
        entry = command.index("tabicl.train")
        command[entry-1:entry+1] = [str(root / "scripts/train_shape_prior.py")]
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    started = time.perf_counter()
    result = subprocess.run(command, env=env, cwd=root)
    report = dict(command=command, wall_seconds=time.perf_counter() - started, exit_code=result.returncode,
                  dtype=args.dtype, gpus=args.gpus, steps=args.steps, stage=args.stage,
                  shape_prior=args.shape_prior)
    report.update(profile_pipeline=args.profile_pipeline, trace_step=args.trace_step,
                  trace_steps=args.trace_steps, trace_all_ranks=args.trace_all_ranks)
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
