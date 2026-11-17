"""Separate GPU activity from CUDA-event intervals in a Chrome/Kineto trace.

Kernel sums can overlap across streams. Device busy time is an interval union;
it is distinct from the sum of kernel durations and from an optimizer step.
Forward/component attribution follows Kineto External ids to the enclosing CPU
record_function range. Component ranges are inclusive and must not be added.
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import heapq
import json
from pathlib import Path


def union_us(intervals):
    total, end = 0., float("-inf")
    for start, stop in sorted(intervals):
        total += max(0., stop - max(start, end))
        end = max(end, stop)
    return total


def summarize(path):
    events = json.loads(path.read_text())["traceEvents"]
    timed = [e for e in events if e.get("ph") == "X" and e.get("dur", 0) > 0]
    scopes = defaultdict(list)
    external = {}
    launches = {}
    for event in timed:
        # Kineto may also emit gpu_user_annotation mirrors of these ranges.
        # Those are device intervals, not additional CPU calls/enqueue scopes.
        if event.get("cat") == "user_annotation" and event.get("name", "").startswith("tabicl/"):
            scopes[event.get("pid"), event.get("tid")].append(event)
        if event.get("cat") == "cpu_op" and "External id" in event.get("args", {}):
            external[event["args"]["External id"]] = event
        if event.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in event.get("args", {}):
            launches[event["args"]["correlation"]] = event
    gpu, devices, scope_gpu, kinds, names = [], defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    scope_names = defaultdict(lambda: defaultdict(list))
    step_scopes = sorted((e for values in scopes.values() for e in values
                          if e["name"].startswith("tabicl/optimizer_step/")), key=lambda e: e["ts"])
    step_starts = [e["ts"] for e in step_scopes]
    step_gpu = defaultdict(lambda: defaultdict(list))
    # Build inclusive containment once by sweeping CPU source starts. Scanning
    # all scopes for every kernel is quadratic for multi-step/all-rank traces.
    sources = defaultdict(list)
    for event in timed:
        if event.get("cat") in ("cpu_op", "cuda_runtime", "cuda_driver"):
            sources[event.get("pid"), event.get("tid")].append(event)
    enclosing = {}
    for key, values in sources.items():
        ranges = sorted(scopes[key], key=lambda e: e["ts"])
        active, endings, cursor = {}, [], 0
        for source in sorted(values, key=lambda e: e["ts"]):
            while cursor < len(ranges) and ranges[cursor]["ts"] <= source["ts"]:
                scope = ranges[cursor]
                active[cursor] = scope
                heapq.heappush(endings, (scope["ts"]+scope["dur"]+5, cursor))
                cursor += 1
            while endings and endings[0][0] < source["ts"]:
                _, index = heapq.heappop(endings)
                active.pop(index, None)
            enclosing[id(source)] = [scope["name"] for scope in active.values()
                                     if source["ts"]+source["dur"] <= scope["ts"]+scope["dur"]+5]
    unlinked, unlinked_us = 0, 0.
    links = defaultdict(int)
    for event in timed:
        category = event.get("cat", "").lower()
        if category not in ("kernel", "gpu_memcpy", "gpu_memset", "memcpy", "memset"):
            continue
        start, stop = event["ts"], event["ts"] + event["dur"]
        gpu.append(event)
        device = str(event.get("args", {}).get("device", event.get("pid")))
        devices[device].append((start, stop))
        kind = "kernel" if category == "kernel" else "memcpy" if "memcpy" in category else "memset"
        kinds[kind].append(event["dur"])
        names[event.get("name", "")].append(event["dur"])
        # run_batch waits for its last CUDA event before ending the CPU range;
        # use that synchronized interval to include unlinked communication and
        # kernels launched from helper threads as well as the main thread.
        index = bisect_right(step_starts, start)-1
        if index >= 0:
            scope = step_scopes[index]
            if start < scope["ts"]+scope["dur"]:
                step = int(scope["name"].rsplit("/", 1)[1])
                comm = any(tag in event.get("name", "").lower()
                           for tag in ("nccl", "pccl", "lccl", "hccl", "allreduce"))
                label = "communication" if comm else kind
                step_gpu[step][label].append((start, min(stop, scope["ts"]+scope["dur"])))
        source = external.get(event.get("args", {}).get("External id"))
        method = "external_id"
        if source is None:
            # Direct Triton launches can omit External id. CUPTI's correlation
            # links the device kernel to its CPU runtime/driver launch instead.
            source = launches.get(event.get("args", {}).get("correlation"))
            method = "launch_correlation"
        if source is None:
            unlinked += 1
            unlinked_us += event["dur"]
            continue
        links[method] += 1
        # Keep inclusive nesting: route kernels can belong to both ICL forward
        # and the more specific route scope. The report labels that explicitly.
        for name in enclosing.get(id(source), []):
            scope_gpu[name].append(event["dur"])
            scope_names[name][event.get("name", "")].append(event["dur"])
    if not gpu:
        raise ValueError("trace contains no supported GPU activities; cannot report kernel timings")
    components = {}
    for ranges in scopes.values():
        for scope in ranges:
            row = components.setdefault(scope["name"], dict(cpu_range_sum_ms=0., calls=0))
            row["cpu_range_sum_ms"] += scope["dur"]/1000
            row["calls"] += 1
    for name, row in components.items():
        durations = scope_gpu[name]
        row.update(gpu_duration_sum_ms=sum(durations)/1000, gpu_activities=len(durations))
        row["top_gpu_activities"] = [dict(name=activity, calls=len(values), duration_sum_ms=sum(values)/1000)
            for activity, values in sorted(scope_names[name].items(), key=lambda item: -sum(item[1]))[:10]]
    return dict(
        file=str(path), gpu_activities=len(gpu), unlinked_gpu_activities=unlinked,
        unlinked_gpu_duration_sum_ms=unlinked_us/1000, attribution_links=dict(links),
        duration_sum_ms_by_kind={k: sum(v)/1000 for k, v in kinds.items()},
        devices={device: dict(busy_union_ms=union_us(intervals)/1000,
                             activity_span_ms=(max(e for _, e in intervals)-min(s for s, _ in intervals))/1000)
                 for device, intervals in devices.items()},
        inclusive_components=components,
        optimizer_steps={str(step): {**{f"{kind}_busy_union_ms": union_us(intervals)/1000
                                       for kind, intervals in kinds.items()},
                                    "activity_busy_union_ms": union_us([pair for values in kinds.values()
                                                                         for pair in values])/1000}
                         for step, kinds in sorted(step_gpu.items())},
        top_gpu_activities=[dict(name=name, calls=len(values), duration_sum_ms=sum(values)/1000)
                            for name, values in sorted(names.items(), key=lambda item: -sum(item[1]))[:30]],
        notes="Inclusive component sums overlap; GPU sums can overlap across streams. Busy union is only within recorded GPU activities, not an entire step. Attribution uses Kineto External ids or runtime/driver launch correlations.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(summarize(args.trace), indent=2)+"\n"
    if args.output:
        args.output.write_text(result)
    print(result)
