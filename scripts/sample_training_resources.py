"""Read-only Linux /proc CPU sampling for an existing serial benchmark matrix.

Run outside the training process. CPU cores are process CPU seconds / interval,
not worker task wall time. Short-lived processes that exit between samples are
undercounted. Rank descendants include the prior manager/workers AND any compile
children; use steady training samples, not warmup, to interpret that category.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time


def snapshot():
    result = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            result[int(directory.name)] = dict(parent=int(fields[1]), ticks=int(fields[11])+int(fields[12]),
                started=int(fields[19]), command=(directory / "cmdline").read_bytes().decode(errors="replace").split("\0"))
        except (OSError, ValueError, IndexError):
            continue  # normal process exit races
    return result


def descendants(processes, root):
    selected = {root}
    while True:
        more = {pid for pid, row in processes.items() if row["parent"] in selected}
        if more <= selected:
            return selected & processes.keys()
        selected |= more


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_pid", type=int)
    parser.add_argument("--interval", type=float, default=2.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.root_pid < 1 or args.interval <= 0:
        parser.error("positive PID and interval required")
    previous, previous_time = snapshot(), time.monotonic()
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    with args.output.open("w", buffering=1) as stream:
        while args.root_pid in previous:
            time.sleep(args.interval)
            current, now = snapshot(), time.monotonic()
            selected = descendants(current, args.root_pid)
            ranks = {pid for pid in selected
                     if any(value == "tabicl.train" or value.endswith(
                         ("/profile_training_pipeline.py", "/train_shape_prior.py"))
                         for value in current[pid]["command"])
                     and "torch.distributed.run" not in current[pid]["command"]}
            rank_descendants = set().union(*(descendants(current, pid)-{pid} for pid in ranks))
            cores, counts = defaultdict(float), defaultdict(int)
            case = None
            for pid in selected:
                row = current[pid]
                role = "rank" if pid in ranks else "rank_descendant" if pid in rank_descendants else "launcher"
                counts[role] += 1
                old = previous.get(pid)
                if old and old["started"] == row["started"]:
                    cores[role] += max(0, row["ticks"]-old["ticks"]) / ticks_per_second / (now-previous_time)
                command = row["command"]
                if any(value.endswith("/benchmark_training_schedule.py") for value in command) and "--output" in command:
                    case = command[command.index("--output")+1]
            step = None
            if case:
                try:
                    with (Path(case) / "logs/metrics.jsonl").open("rb") as metrics:
                        metrics.seek(0, 2)
                        metrics.seek(max(0, metrics.tell()-32768))
                        step = json.loads(metrics.read().splitlines()[-1]).get("step")
                except (OSError, ValueError, IndexError):
                    pass
            stream.write(json.dumps(dict(time_unix=time.time(), interval_s=now-previous_time,
                case=case, last_completed_step=step, cpu_cores=dict(cores), processes=dict(counts)))+"\n")
            previous, previous_time = current, now


if __name__ == "__main__":
    main()
