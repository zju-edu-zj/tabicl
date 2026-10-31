"""Separate GPU activity from CUDA-event intervals in a Chrome/Kineto trace.

Kernel sums can overlap across streams. Device busy time is an interval union;
it is distinct from the sum of kernel durations and from an optimizer step.
Forward/component attribution follows Kineto External ids to the enclosing CPU
record_function range. Component ranges are inclusive and must not be added.
"""
import argparse
from collections import defaultdict
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
        for scope in scopes[source.get("pid"), source.get("tid")]:
            if scope["ts"] <= source["ts"] and source["ts"]+source["dur"] <= scope["ts"]+scope["dur"]+5:
                scope_gpu[scope["name"]].append(event["dur"])
                scope_names[scope["name"]][event.get("name", "")].append(event["dur"])
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
