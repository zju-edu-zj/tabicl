"""Deterministic, fixed-cardinality assignment within a supplied step.

This allocator never rejects, splits or duplicates tasks. Cross-step window
composition belongs to _task_plan or _ready_micro; both retain intact micros.
Costs are an operation-count proxy, not a promise of measured GPU latency.
"""
import math
import os
from pathlib import Path


def available_cpu_cores():
    """Respect affinity AND hierarchical cgroup CPU quotas (v1/v2)."""
    available = float(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
    roots = [Path("/sys/fs/cgroup")]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            if not controllers:
                roots.append(Path("/sys/fs/cgroup") / relative.lstrip("/"))
            elif "cpu" in controllers.split(","):
                roots.append(Path("/sys/fs/cgroup/cpu") / relative.lstrip("/"))
    except OSError:
        pass
    for root in roots:
        # A parent slice can impose a tighter quota than the immediate group.
        for directory in (root, *root.parents):
            if not str(directory).startswith("/sys/fs/cgroup"):
                continue
            try:
                quota, period = (directory / "cpu.max").read_text().split()
                if quota != "max":
                    available = min(available, int(quota) / int(period))
            except (OSError, ValueError, ZeroDivisionError):
                pass
            try:
                quota = int((directory / "cpu.cfs_quota_us").read_text())
                period = int((directory / "cpu.cfs_period_us").read_text())
                if quota > 0:
                    available = min(available, quota / period)
            except (OSError, ValueError, ZeroDivisionError):
                pass
    return max(1, math.floor(available))


def micro_cost(params, results, indices, model):
    """Forward MAC proxy, NOT a latency predictor or generation-cost estimate.

    Includes projections/FFNs, CLS-only final ROW block, configured hybrid ICL
    layer counts, and centroid routing. Before generation callers supply the
    requested width; assignment after generation uses the accepted width.
    Backward, launch overhead, dtype/backend efficiency and sorting are not
    fitted here. Validate against device timings before interpreting gains.
    """
    rows, support = params[indices[0]]["seq_len"], params[indices[0]]["train_size"]
    if any((params[i]["seq_len"], params[i]["train_size"]) != (rows, support) for i in indices):
        raise ValueError("scheduling cannot mix lengths/supports within a micro-batch")
    width = max(results[i][2] for i in indices)
    e, cls = model.get("embed_dim", 128), model.get("row_num_cls", 4)
    # "same" uses circular feature groups and keeps one token per original
    # column; only "valid" reduces the token count by the group size.
    grouped = (math.ceil(width / model.get("col_feature_group_size", 3))
               if model.get("col_feature_group", "same") == "valid" else width)
    ff = model.get("ff_factor", 2)
    def dense(q, k, d):
        return (2*q + 2*k + 2*ff*q)*d*d + 2*q*k*d
    inds = model.get("col_num_inds", 128)
    col = model.get("col_num_blocks", 3) * grouped * (dense(inds, support, e) + dense(rows, inds, e))
    tokens = grouped + cls
    blocks = model.get("row_num_blocks", 3)
    row = rows * (max(0, blocks-1)*dense(tokens, tokens, e) + (dense(cls, tokens, e) if blocks else 0))
    dim = cls * e
    budget = model.get("icl_block_size", 128) * model.get("icl_num_selected_blocks", 4)
    attention = model.get("icl_attention", "global_sparse")
    nhead = model.get("icl_nhead", 8)
    # Global linear attention contracts per head (or through induced slots).
    global_keys = (model.get("icl_num_induced_slots", 128) if model.get("icl_global_type", "elu") == "induced"
                   else dim / nhead)
    global_work = 2 * (rows + support) * dim * global_keys
    retrieval = 2 * rows * dim * min(support, budget)
    leaves = math.ceil(support / model.get("icl_block_size", 128))
    if support > budget:
        routing_keys = (leaves if model.get("icl_router", "centroid") == "centroid" else
                        model.get("icl_router_beam", 16) * math.ceil(math.log2(max(2, leaves))))
        retrieval += rows * dim * routing_keys
    summary_slots = model.get("icl_summary_slots", 1)
    frontier = retrieval + 2 * rows * dim * max(0, leaves - model.get("icl_num_selected_blocks", 4)) * summary_slots
    hybrids = {"global_full": "fused_full", "global_sparse": "fused_sparse",
               "global_sparse_alternating": "sparse"}
    icl = 0.
    for layer in range(model.get("icl_num_blocks", 12)):
        kind = (hybrids[attention] if (layer+1) % model.get("icl_sparse_every", 3) == 0 else "global") if attention in hybrids else attention
        pair_work = {"dense": 2*rows*support*dim, "global": global_work,
                     "sparse": retrieval, "fused_sparse": global_work+retrieval,
                     "fused_full": global_work+2*rows*support*dim,
                     "dual_path": global_work+retrieval,
                     "adaptive_frontier": frontier}[kind]
        projection_work = (2*rows + 2*support + 2*ff*rows)*dim*dim
        if kind == "dual_path":
            projection_work += 2 * (rows + support) * dim * dim
        icl += projection_work + pair_work
    return float(len(indices) * (col + row + icl))


def balanced_partition(costs, ranks):
    """LPT with equal slot counts, then improving swaps; keep baseline if better.

    Equal datasets/micro-batches per rank preserve DDP loss averaging and the
    number of synchronization calls. Works for arbitrary rank counts, including
    one item per rank (where permutation cannot improve the maximum).
    """
    if ranks < 1 or not costs or len(costs) % ranks:
        raise ValueError("positive ranks must divide the number of micro-batches")
    if any(not math.isfinite(c) or c < 0 for c in costs):
        raise ValueError("costs must be finite and nonnegative")
    slots = len(costs) // ranks
    baseline = [list(range(r * slots, (r + 1) * slots)) for r in range(ranks)]
    groups, loads = [[] for _ in range(ranks)], [0.] * ranks
    for index in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        rank = min((r for r in range(ranks) if len(groups[r]) < slots), key=lambda r: (loads[r], r))
        groups[rank].append(index)
        loads[rank] += costs[index]
    # Strictly reduce the pair's maximum, so the global maximum never grows.
    for _ in range(min(len(costs), 64)):
        improved = False
        for a in sorted(range(ranks), key=lambda r: -loads[r]):
            for b in sorted(range(ranks), key=lambda r: loads[r]):
                if loads[a] <= loads[b]:
                    continue
                for i in range(slots):
                    for j in range(slots):
                        delta = costs[groups[a][i]] - costs[groups[b][j]]
                        if max(loads[a] - delta, loads[b] + delta) < loads[a] - 1e-9:
                            groups[a][i], groups[b][j] = groups[b][j], groups[a][i]
                            loads[a] -= delta
                            loads[b] += delta
                            improved = True
        if not improved:
            break
    if max(loads) > max(sum(costs[i] for i in g) for g in baseline):
        groups = baseline
    return [sorted(g, key=lambda i: (-costs[i], i)) for g in groups]


def schedule_cohort(params, results, local_batch, ranks, micro_size, model, balanced=True):
    if local_batch % micro_size or len(params) != local_batch * ranks:
        raise ValueError("balanced scheduling requires full, equal-size micro-batches")
    units = [list(range(i, i + micro_size)) for i in range(0, len(params), micro_size)]
    costs = [micro_cost(params, results, unit, model) for unit in units]
    slots = local_batch // micro_size
    groups = (balanced_partition(costs, ranks) if balanced else
              [list(range(r * slots, (r + 1) * slots)) for r in range(ranks)])
    assignments = [[i for u in group for i in units[u]] for group in groups]
    baseline = max(sum(costs[r * slots:(r + 1) * slots]) for r in range(ranks))
    maximum = max(sum(costs[u] for u in group) for group in groups)
    return assignments, dict(predicted_max_before=baseline, predicted_max_after=maximum,
                             predicted_gain=baseline / max(maximum, 1))
