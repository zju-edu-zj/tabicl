#!/usr/bin/env python3
"""Read-only statistical report for a TabICL training checkpoint and its logs.

Example:
  python scripts/analyze_checkpoint.py \
    /ossfs/workspace/tabicl/checkpoints/tabiclv2-clf/stage1/step-60000.ckpt

The checkpoint contains model/optimizer/scheduler state, but not gradients or
the training data. Runtime, learning-curve, and sampled ICL diagnostics are
read from the adjacent ``logs`` directory when available.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch


def _finite_stats(tensor):
    x = tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    finite = torch.isfinite(x)
    good = x[finite]
    if good.numel() == 0:
        return {
            "numel": x.numel(), "nonfinite": x.numel(), "zero_fraction": None,
            "mean": None, "std": None, "rms": None, "abs_max": None,
        }
    return {
        "numel": x.numel(),
        "nonfinite": int((~finite).sum()),
        "zero_fraction": float((good == 0).double().mean()),
        "mean": float(good.mean()),
        "std": float(good.std(unbiased=False)),
        "rms": float(good.square().mean().sqrt()),
        "abs_max": float(good.abs().max()),
    }


def _component(name):
    if name.startswith("col_embedder."):
        return "column_encoder"
    if name.startswith("row_interactor."):
        return "row_encoder"
    if name.startswith("icl_predictor."):
        return "icl_encoder"
    if name.startswith(("classifier.", "prediction_head.", "head.")):
        return "prediction_head"
    return name.split(".", 1)[0]


def _layer_tag(name):
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "blocks" and i + 1 < len(parts):
            return ".".join(parts[: i + 2])
    return ".".join(parts[:2]) if len(parts) > 1 else name


def analyze_model_state(state):
    by_component = defaultdict(list)
    by_layer = defaultdict(list)
    dtypes = Counter()
    total, nonfinite = 0, 0
    tensors = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        stats = _finite_stats(value)
        stats["shape"] = list(value.shape)
        stats["dtype"] = str(value.dtype)
        tensors[name] = stats
        by_component[_component(name)].append((name, stats))
        by_layer[_layer_tag(name)].append((name, stats))
        total += stats["numel"]
        nonfinite += stats["nonfinite"]
        dtypes[str(value.dtype)] += stats["numel"]

    def aggregate(entries):
        count = sum(s["numel"] for _, s in entries)
        bad = sum(s["nonfinite"] for _, s in entries)
        means = [(s["mean"], s["numel"] - s["nonfinite"]) for _, s in entries if s["mean"] is not None]
        squares = [(s["rms"] ** 2, s["numel"] - s["nonfinite"]) for _, s in entries if s["rms"] is not None]
        finite_count = sum(n for _, n in means)
        return {
            "tensors": len(entries), "parameters_or_buffers": count,
            "nonfinite": bad,
            "zero_fraction": (sum(s["zero_fraction"] * (s["numel"] - s["nonfinite"])
                                   for _, s in entries if s["zero_fraction"] is not None) / finite_count)
                              if finite_count else None,
            "mean": sum(v * n for v, n in means) / finite_count if finite_count else None,
            "rms": math.sqrt(sum(v * n for v, n in squares) / finite_count) if finite_count else None,
        }

    special = {}
    patterns = ("fusion_gate", "qk_norm", "route_direction", "inducing_points", "ssmax")
    for name, stats in tensors.items():
        if any(token in name.lower() for token in patterns):
            special[name] = stats
    return {
        "tensor_count": len(tensors),
        "total_numel": total,
        "nonfinite_total": nonfinite,
        "numel_by_dtype": dict(dtypes),
        "by_component": {k: aggregate(v) for k, v in sorted(by_component.items())},
        "by_layer": {k: aggregate(v) for k, v in sorted(by_layer.items())},
        "special_tensors": special,
        "tensor_stats": tensors,
    }


def analyze_optimizer(optimizer):
    if not isinstance(optimizer, dict):
        return {"available": False}
    groups = []
    state = optimizer.get("state", {})
    for i, group in enumerate(optimizer.get("param_groups", [])):
        groups.append({
            "index": i,
            "parameters": len(group.get("params", [])),
            "lr": group.get("lr"),
            "weight_decay": group.get("weight_decay"),
            "betas": group.get("betas"),
            "momentum": group.get("momentum"),
            "use_muon": group.get("use_muon"),
        })
    state_fields = defaultdict(list)
    for param_state in state.values():
        for key, value in param_state.items():
            if isinstance(value, torch.Tensor):
                state_fields[key].append(_finite_stats(value))
            elif isinstance(value, (int, float)):
                state_fields[key].append(value)
    fields = {}
    for key, vals in state_fields.items():
        if vals and isinstance(vals[0], dict):
            rms = [v["rms"] for v in vals if v["rms"] is not None]
            fields[key] = {
                "tensor_count": len(vals),
                "nonfinite": sum(v["nonfinite"] for v in vals),
                "mean_tensor_rms": sum(rms) / len(rms) if rms else None,
                "max_tensor_rms": max(rms) if rms else None,
            }
        else:
            fields[key] = {"count": len(vals), "min": min(vals), "max": max(vals)}
    return {
        "available": True,
        "parameter_state_entries": len(state),
        "parameter_group_count": len(groups),
        "parameter_groups": groups,
        "state_fields": fields,
    }


def _read_jsonl(path):
    if not path or not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def analyze_metrics(path):
    rows = _read_jsonl(path)
    if not rows:
        return {"available": False, "path": str(path) if path else None}
    steps = [r for r in rows if "elapsed_s" in r and "step" in r]
    numeric_keys = sorted({k for row in rows for k, v in row.items()
                           if k not in ("step", "elapsed_s") and isinstance(v, (int, float))})
    stats = {}
    for key in numeric_keys:
        vals = [(int(r["step"]), float(r[key])) for r in rows
                if key in r and isinstance(r[key], (int, float)) and math.isfinite(float(r[key]))]
        if not vals:
            continue
        values = [v for _, v in vals]
        window = values[-min(100, len(values)):]
        stats[key] = {
            "count": len(values), "first": values[0], "last": values[-1],
            "min": min(values), "max": max(values),
            "mean": sum(values) / len(values),
            "last_100_mean": sum(window) / len(window),
            "first_step": vals[0][0], "last_step": vals[-1][0],
        }
    train_times = [float(r.get("train_time", r.get("system/train_time"))) for r in steps
                   if isinstance(r.get("train_time", r.get("system/train_time")), (int, float))]
    prior_times = [float(r.get("prior_time", r.get("system/prior_time"))) for r in steps
                   if isinstance(r.get("prior_time", r.get("system/prior_time")), (int, float))]
    return {
        "available": True, "path": str(path), "records": len(rows),
        "training_step_records": len(steps),
        "first_step": min((int(r["step"]) for r in steps), default=None),
        "last_step": max((int(r["step"]) for r in steps), default=None),
        "elapsed_s": steps[-1].get("elapsed_s") if steps else None,
        "mean_train_time_s": sum(train_times) / len(train_times) if train_times else None,
        "mean_prior_time_s": sum(prior_times) / len(prior_times) if prior_times else None,
        "numeric_metrics": stats,
    }


def analyze_validation(path):
    rows = _read_jsonl(path)
    if not rows:
        return {"available": False, "path": str(path) if path else None}
    by_step = defaultdict(list)
    for row in rows:
        by_step[int(row["step"])].extend(row.get("cases", []))
    summaries = {}
    for step, cases in sorted(by_step.items()):
        metrics = defaultdict(list)
        for case in cases:
            for key, value in case.items():
                if key not in ("id", "support", "features") and isinstance(value, (int, float)):
                    metrics[f"support_{case.get('support', 'unknown')}/{key}"].append(float(value))
        summaries[str(step)] = {k: sum(v) / len(v) for k, v in sorted(metrics.items()) if v}
    return {"available": True, "path": str(path), "steps": summaries}


def _markdown(report):
    ckpt = report["checkpoint"]
    model = report["model"]
    lines = [
        f"# Checkpoint analysis: `{ckpt['path']}`", "",
        f"- Step: **{ckpt.get('curr_step')}**",
        f"- Parameters and buffers: **{model['total_numel']:,}** across {model['tensor_count']} tensors",
        f"- Non-finite model values: **{model['nonfinite_total']}**",
        f"- Task / ICL: **{ckpt.get('config', {}).get('max_classes')} classes / {ckpt.get('config', {}).get('icl_attention')}**",
        "", "## Architecture and training state", "",
        "```json", json.dumps({"config": ckpt.get("config"), "optimizer": report["optimizer"],
                               "scheduler": ckpt.get("scheduler_state")}, indent=2, default=str), "```",
        "", "## Parameter statistics by component", "",
        "| Component | Numel | RMS | Mean | Zeros | Non-finite |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in model["by_component"].items():
        lines.append(f"| {name} | {s['parameters_or_buffers']:,} | {s['rms']} | {s['mean']} | {s['zero_fraction']} | {s['nonfinite']} |")
    lines += ["", "## Training log", "", "```json", json.dumps(report["metrics"], indent=2), "```",
              "", "## Validation records", "", "```json", json.dumps(report["validation"], indent=2), "```",
              "", "## Interpretation limits", "",
              "Checkpoint weights do not contain gradients or per-step activations. Optimizer moments are not gradients.",
              "Router recall, retained attention mass, and fusion behavior are only available if the corresponding ICL diagnostics were enabled during training.",
              "Numerical parameter summaries are descriptive; they do not establish predictive quality. Evaluate the checkpoint on a fixed held-out suite and downstream tasks.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--log-dir", type=Path, help="Defaults to <checkpoint parent>/logs")
    parser.add_argument("--output", type=Path, help="Defaults to <checkpoint parent>/analysis-step-N")
    parser.add_argument("--no-tensor-stats", action="store_true", help="Omit per-tensor details from JSON")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint_path}")
    # weights_only avoids arbitrary pickle execution; never writes to checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        parser.error("Expected a TabICL training checkpoint containing state_dict")
    step = checkpoint.get("curr_step")
    log_dir = (args.log_dir or checkpoint_path.parent / "logs").expanduser().resolve()
    output_dir = (args.output or checkpoint_path.parent / f"analysis-step-{step or 'unknown'}").expanduser().resolve()
    model = analyze_model_state(checkpoint["state_dict"])
    if args.no_tensor_stats:
        model.pop("tensor_stats")
    report = {
        "checkpoint": {
            "path": str(checkpoint_path), "size_bytes": checkpoint_path.stat().st_size,
            "curr_step": step, "config": checkpoint.get("config", {}),
            "scheduler_state": checkpoint.get("scheduler_state", {}),
        },
        "model": model,
        "optimizer": analyze_optimizer(checkpoint.get("optimizer_state")),
        "metrics": analyze_metrics(log_dir / "metrics.jsonl"),
        "validation": analyze_validation(log_dir / "validation.jsonl"),
        "training_config": None,
    }
    config_path = log_dir / "config.json"
    if config_path.exists():
        report["training_config"] = json.loads(config_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = output_dir / "analysis.json", output_dir / "analysis.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + "\n")
    md_path.write_text(_markdown(report))
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Step: {step}; model tensors: {model['tensor_count']}; numel: {model['total_numel']:,}; nonfinite: {model['nonfinite_total']}")
    print(f"Training log available: {report['metrics']['available']}; validation log available: {report['validation']['available']}")
    print(f"Wrote {json_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
