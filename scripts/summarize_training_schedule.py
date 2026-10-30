"""Summarize end-to-end wall time and measured rank stalls from real DDP runs."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(directory, skip):
    report = json.loads((directory / "run.json").read_text())
    rows = [json.loads(line) for line in (directory / "logs/metrics.jsonl").read_text().splitlines()]
    steady = rows[skip:]
    if not steady:
        raise ValueError(f"{directory}: no measurements after skipping {skip} steps")
    result = {k: report[k] for k in ("wall_seconds", "exit_code", "dtype", "gpus", "steps", "stage")}
    result.update(directory=str(directory), skip=skip)
    for key in ("prior_time", "train_time", "rank_timing/prior_max_s", "rank_timing/train_max_s"):
        values = np.asarray([row[key] for row in steady if key in row])
        if len(values):
            result[key] = dict(mean_ms=float(values.mean()*1000), p95_ms=float(np.quantile(values,.95)*1000),
                               max_ms=float(values.max()*1000), samples=len(values))
    totals = []
    for row in steady:
        pairs = [row[f"rank_timing/rank{r}_prior_s"] + row[f"rank_timing/rank{r}_train_s"]
                 for r in range(report["gpus"]) if f"rank_timing/rank{r}_train_s" in row]
        if pairs:
            totals.append(max(pairs))
    if totals:
        result["rank_timing/step_max_s"] = dict(mean_ms=float(np.mean(totals)*1000),
            p95_ms=float(np.quantile(totals,.95)*1000), samples=len(totals))
    result["steady_elapsed_per_step_ms"] = (
        (steady[-1]["elapsed_s"] - rows[skip-1]["elapsed_s"]) / len(steady) * 1000
        if skip else (steady[-1]["elapsed_s"] / len(steady) * 1000))
    result["oom_micro_batches"] = sum(row.get("oom_micro_batches", 0) for row in rows)
    result["nonfinite_loss"] = sum(not np.isfinite(row.get("ce", row.get("pinball", float("nan")))) for row in rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--skip", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.skip < 0:
        parser.error("skip must be nonnegative")
    text = json.dumps([summarize(p, args.skip) for p in args.directories], indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
