"""Summarize matched rank/step CUDA-boundary profiles, excluding startup steps.

Stream elapsed time includes launch/CPU gaps. It is not kernel active time.
Column backward includes DDP synchronization; use Kineto for communication and
kernel-only attribution. Rank maxima of wait+train are step diagnostics, not an
independent decomposition of exposed CPU wait and DDP wait.
"""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def summarize(directory, skip, expected_ranks=None):
    if skip < 0:
        raise ValueError("skip must be nonnegative")
    files = sorted(directory.glob("rank[0-9]*.jsonl"))
    if not files:
        raise ValueError("no per-rank profile files")
    rank_ids = {int(file.stem[4:]) for file in files}
    if expected_ranks is not None and rank_ids != set(range(expected_ranks)):
        raise ValueError("missing ranks or unexpected rank files")
    by_step = defaultdict(dict)
    for file in files:
        for line in file.read_text().splitlines():
            row = json.loads(line)
            if row["rank"] != int(file.stem[4:]):
                raise ValueError("rank/file mismatch")
            if row["step"] >= skip:
                if row["rank"] in by_step[row["step"]]:
                    raise ValueError("duplicate rank/step profile")
                by_step[row["step"]][row["rank"]] = row
    if not by_step or any(len(rows) != len(files) for rows in by_step.values()):
        raise ValueError("no complete profile or missing ranks/steps")
    totals = defaultdict(float)
    table_count = 0
    rank_steps = []
    workloads = defaultdict(float)
    gpu_steps = {}
    for path in sorted(directory.glob("rank[0-9]*-trace-summary.json")):
        rank = int(path.name.split("-", 1)[0][4:])
        gpu_steps[rank] = json.loads(path.read_text()).get("optimizer_steps", {})
    for rows in by_step.values():
        rank_steps.append(dict(step=next(iter(rows.values()))["step"],
                              wait_train_max_s=max(r["prior_wait_s"]+r["train_wall_s"] for r in rows.values()),
                              train_max_s=max(r["train_wall_s"] for r in rows.values()),
                              train_mean_s=float(np.mean([r["train_wall_s"] for r in rows.values()])),
                              prior_wait_max_s=max(r["prior_wait_s"] for r in rows.values())))
        if all("prior_start_monotonic_s" in r and "train_end_monotonic_s" in r for r in rows.values()):
            rank_steps[-1]["aligned_step_span_s"] = (
                max(r["train_end_monotonic_s"] for r in rows.values())
                - min(r["prior_start_monotonic_s"] for r in rows.values()))
        if all(r.get("cpu_generation") for r in rows.values()):
            timing = [r["cpu_generation"] for r in rows.values()]
            detail = rank_steps[-1]
            detail["cpu_generation_core_s"] = sum(t["process_cpu_s"] for t in timing)
            detail["cpu_generation_span_s"] = (max(t["generation_ready_monotonic_s"] for t in timing)
                                               - min(t["first_generation_monotonic_s"] for t in timing))
            detail["cpu_submit_to_ready_s"] = (max(t["generation_ready_monotonic_s"] for t in timing)
                                               - min(t["submitted_monotonic_s"] for t in timing))
            ready = [t.get("dispatch_generation_ready_monotonic_s", t["generation_ready_monotonic_s"])
                     for t in timing]
            waits = [max(0., end-r["prior_start_monotonic_s"])
                     for end, r in zip(ready, rows.values())]
            detail["generation_wait_rank_max_s"] = max(waits)
            detail["generation_wait_rank_mean_s"] = float(np.mean(waits))
            # A conservative exposed stall: all ranks have finished the previous
            # step, none has started this model, and every rank still lacks data.
            detail["generation_wait_all_ranks_s"] = max(0., min(
                min(ready),
                min(r["train_start_monotonic_s"] for r in rows.values()))
                - max(r["prior_start_monotonic_s"] for r in rows.values()))
            detail["input_service_wait_max_s"] = max(t["service_wait_s"] for t in timing)
        step = rank_steps[-1]["step"]
        if all(str(step) in gpu_steps.get(rank, {}) for rank in rows):
            for key in ("kernel_busy_union_ms", "communication_busy_union_ms", "activity_busy_union_ms"):
                rank_steps[-1][f"gpu_{key.removesuffix('_ms')}_rank_max_s"] = max(
                    gpu_steps[rank][str(step)].get(key, 0.) for rank in rows)/1000
        for row in rows.values():
            if row["oom_micro_batches"]:
                raise ValueError("OOM-skipped tasks invalidate the performance comparison")
            if not np.isfinite(row["loss"]):
                raise ValueError("nonfinite loss invalidates the performance comparison")
            for key in ("optimizer_ms", "train_stream_ms", "outside_micros_stream_ms"):
                totals[key] += row.get(key, 0.)
            for micro in row["micros"]:
                required = ("col_forward_ms", "row_forward_ms", "icl_forward_ms", "icl_backward_ms",
                            "row_backward_ms", "col_backward_and_sync_ms")
                if any(key not in micro for key in required):
                    raise ValueError("missing stage boundaries; use a model with all three stages trainable")
                table_count += micro["micro_size"]
                if "rows" in micro and "width" in micro:
                    weight, length, width = micro["micro_size"], micro["rows"], micro["width"]
                    workloads["tables"] += weight
                    workloads["rows"] += weight * length
                    workloads["width"] += weight * width
                    workloads["row_features"] += weight * length * width
                    workloads["row_features_squared"] += weight * length * width * width
                for key, value in micro.items():
                    if key.endswith("_ms"):
                        totals[key] += value
    if not table_count:
        raise ValueError("no profiled tables")
    forward = sum(totals[f"{s}_forward_ms"] for s in ("col", "row", "icl"))
    stages = forward + sum(totals[k] for k in ("icl_backward_ms", "row_backward_ms", "col_backward_and_sync_ms"))
    icl = totals["icl_forward_ms"] + totals["icl_backward_ms"]
    return dict(ranks=len(files), steps=len(by_step), tables=table_count,
                icl_forward_fraction=totals["icl_forward_ms"]/max(forward, 1e-9),
                icl_forward_backward_fraction_including_col_sync=icl/max(stages, 1e-9),
                mean_ms_per_table={k: v/table_count for k, v in totals.items()},
                workload_totals=dict(workloads),
                workload_mean_per_table={k: v/max(workloads["tables"], 1) for k, v in workloads.items()
                                         if k != "tables"},
                step_mean={k: float(np.mean([s[k] for s in rank_steps if k in s]))
                           for k in set().union(*(s.keys() for s in rank_steps)) if k != "step"},
                steps_detail=rank_steps,
                notes="CUDA stream boundaries include CPU submission gaps; column backward includes DDP tail; use trace for kernel-only costs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--skip", type=int, default=10)
    parser.add_argument("--expected-ranks", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path, help="write one row per optimizer step")
    args = parser.parse_args()
    result = summarize(args.directory, args.skip, args.expected_ranks)
    text = json.dumps(result, indent=2)+"\n"
    if args.output:
        args.output.write_text(text)
    if args.csv_output:
        rows = result["steps_detail"]
        with args.csv_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader()
            writer.writerows(rows)
    print(text)
