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
    command = [sys.executable, "-m", "torch.distributed.run", *tokens[1:]]
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    started = time.perf_counter()
    result = subprocess.run(command, env=env, cwd=root)
    report = dict(command=command, wall_seconds=time.perf_counter() - started, exit_code=result.returncode,
                  dtype=args.dtype, gpus=args.gpus, steps=args.steps, stage=args.stage)
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
