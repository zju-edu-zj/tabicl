#!/usr/bin/env python3
"""Mechanism-level, post-hoc diagnostics for a trained TabICL checkpoint.

Pass either one Stage-1 checkpoint or its directory. If no cases file is given,
the script creates and reuses a deterministic Graph-SCM suite. All interventions
are temporary inference-time overrides; checkpoint weights and training recipes
are never modified.
"""

import argparse
import json
import math
import random
import re
import sys
import types
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

# Allow `python scripts/diagnose_icl_checkpoint.py` from a source checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn.functional as F
import numpy as np

from tabicl._model.icl_attention import SetRetrievalAttention
from tabicl._model.tabicl import TabICL
from tabicl.prior._dataset import PriorDataset
from tabicl.prior.graph_lib._config import PriorConfig


POLICIES = ("native", "global_only", "sparse_only", "random_candidates", "oracle_candidates", "dense_only")


def _float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().cpu())
    return float(value)


def _make_oracle_route(module):
    """Choose blocks with greatest exact dense-softmax mass (same block budget)."""
    original_route = module.route

    @torch.no_grad()
    def route(q, k):
        order, _ = original_route(q, k)
        sorted_k = k.gather(2, order.long()[..., None].expand_as(k)).float()
        block_size = module.block_size
        nk = (k.shape[-2] + block_size - 1) // block_size
        pad = nk * block_size - k.shape[-2]
        if pad:
            sorted_k = F.pad(sorted_k, (0, 0, 0, pad))
        # Chunk query rows to bound the temporary [B,H,Q,S] score tensor.
        chosen = []
        budget = min(module.num_selected_blocks, nk)
        scale = module.head_dim**-0.5
        for start in range(0, q.shape[-2], 16):
            query = q[:, :, start : start + 16]
            if module.qk_norm is not None:
                query = module.qk_norm.scale_query(query, k.shape[-2])
            logits = torch.matmul(query.float(), sorted_k.transpose(-1, -2)) * scale
            if pad:
                logits[..., -pad:] = -float("inf")
            probs = logits.softmax(-1)
            block_mass = probs.new_zeros((*probs.shape[:-1], nk))
            ids = (torch.arange(nk * block_size, device=q.device) // block_size)
            block_mass.scatter_add_(-1, ids.view(1, 1, 1, -1).expand_as(probs), probs)
            chosen.append(block_mass.topk(budget, dim=-1).indices)
        return order, torch.cat(chosen, dim=2).to(torch.int32)

    return route


def _make_random_route(module, generator):
    original_route = module.route

    @torch.no_grad()
    def route(q, k):
        order, _ = original_route(q, k)
        nk = (k.shape[-2] + module.block_size - 1) // module.block_size
        budget = min(module.num_selected_blocks, nk)
        shape = (*q.shape[:3], nk)
        # Generate on CPU so results are reproducible across CUDA devices.
        selected = torch.rand(shape, generator=generator, device="cpu").argsort(-1)[..., :budget]
        return order, selected.to(device=q.device, dtype=torch.int32)

    return route


def _install_policy(modules, policy, seed):
    saved = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for name, module in modules:
        saved.append((name, module, module._hybrid_attention, module.route))
        if policy == "native":
            original_hybrid = module._hybrid_attention

            def native_with_gate_diagnostics(self, q, k, v, _original=original_hybrid):
                diagnose = self.diagnostics_enabled
                output = _original(q, k, v)
                if diagnose and hasattr(self, "fusion_gate_weight"):
                    with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
                        gate = torch.sigmoid(
                            torch.einsum("bhtd,hd->bht", q.float(), self.fusion_gate_weight.float())
                            + self.fusion_gate_bias.float()[None, :, None]
                        )
                        gate = gate[:, :, self.diagnostic_query_start:]
                        for head, values in enumerate(gate.transpose(0, 1)):
                            values = values.reshape(-1)
                            self.last_diagnostics[f"head_{head}/fusion_sparse_gate_mean"] = values.mean()
                            self.last_diagnostics[f"head_{head}/fusion_sparse_gate_std"] = values.std(unbiased=False)
                            self.last_diagnostics[f"head_{head}/fusion_sparse_gate_min"] = values.min()
                            self.last_diagnostics[f"head_{head}/fusion_sparse_gate_max"] = values.max()
                return output

            module._hybrid_attention = types.MethodType(native_with_gate_diagnostics, module)
            continue
        if policy in ("random_candidates", "oracle_candidates"):
            module.route = (
                _make_random_route(module, generator)
                if policy == "random_candidates"
                else _make_oracle_route(module)
            )

        def intervention(self, q, k, v, _policy=policy):
            if _policy == "global_only":
                heads = self._global_attention(q, k, v)
            elif _policy == "dense_only":
                scaled_q = self.qk_norm.scale_query(q, k.shape[-2]) if self.qk_norm is not None else q
                heads = F.scaled_dot_product_attention(scaled_q, k, v)
                if self.diagnostics_enabled:
                    self._diagnose(scaled_q, k, v)
            else:
                heads = self._sparse_attention(q, k, v)
            combined = heads.transpose(-3, -2).contiguous().view(
                *q.shape[:-3], q.shape[-2], self.embed_dim
            )
            return self.out_proj(combined)

        module._hybrid_attention = types.MethodType(intervention, module)
    return saved


def _restore(saved):
    for _, module, hybrid, route in saved:
        module._hybrid_attention = hybrid
        module.route = route


def _enable_diagnostics(modules, query_count, query_start=0):
    states = []
    for _, module in modules:
        states.append((module, module.diagnostics_enabled, module.diagnostic_queries,
                       module.last_diagnostics, module._diagnose, module.diagnostic_query_start))
        original_diagnose = module._diagnose

        @torch.no_grad()
        def diagnose_by_head(self, q, k, v, selected=None, _original=original_diagnose):
            _original(q, k, v, selected)
            first = min(max(self.diagnostic_query_start, 0), q.shape[-2] - 1)
            sample = torch.linspace(
                first, q.shape[-2] - 1,
                min(self.diagnostic_queries, q.shape[-2] - first), device=q.device,
            ).long()
            with torch.autocast(device_type=q.device.type, enabled=False):
                logits = q[:1, :, sample].float() @ k[:1].float().transpose(-1, -2) / self.head_dim**0.5
                if selected is None:
                    mask = torch.ones_like(logits, dtype=torch.bool)
                else:
                    ids = torch.arange(k.shape[-2], device=q.device) // self.block_size
                    mask = (selected[:1, :, sample, :, None] == ids).any(-2)
                prob = logits.softmax(-1)
                sparse_prob = logits.masked_fill(~mask, -float("inf")).softmax(-1)
                exact = logits.topk(min(self.diagnostic_topk, k.shape[-2]), -1).indices
                dense_out, sparse_out = prob @ v[:1].float(), sparse_prob @ v[:1].float()
                for head in range(q.shape[1]):
                    dense_rms = dense_out[:, head].square().mean().sqrt().clamp_min(1e-6)
                    self.last_diagnostics[f"head_{head}/recall_at_k"] = (
                        mask[:, head].gather(-1, exact[:, head]).float().mean()
                    )
                    self.last_diagnostics[f"head_{head}/retained_mass"] = (
                        prob[:, head] * mask[:, head]
                    ).sum(-1).mean()
                    self.last_diagnostics[f"head_{head}/dense_output_relative_error"] = (
                        sparse_out[:, head] - dense_out[:, head]
                    ).square().mean().sqrt() / dense_rms
            self.diagnostics_enabled = False

        module._diagnose = types.MethodType(diagnose_by_head, module)
        module.diagnostics_enabled = query_count > 0
        module.diagnostic_queries = max(1, query_count)
        module.diagnostic_query_start = query_start
        module.last_diagnostics = {}
    return states


def _restore_diagnostics(states):
    for module, enabled, queries, diagnostics, diagnose, query_start in states:
        module.diagnostics_enabled = enabled
        module.diagnostic_queries = queries
        module.last_diagnostics = diagnostics
        module._diagnose = diagnose
        module.diagnostic_query_start = query_start


def _validate_labels(y, classes, context):
    """Check on CPU before any one-hot/CE CUDA kernel can assert."""
    labels = y.detach().cpu()
    if not labels.numel() or not torch.isfinite(labels).all():
        raise ValueError(f"{context}: classification labels must be non-empty and finite")
    if (labels != labels.long()).any() or (labels < 0).any() or (labels >= classes).any():
        raise ValueError(
            f"{context}: classification labels must be integers in [0, {classes - 1}]; "
            f"observed range [{labels.min().item()}, {labels.max().item()}]"
        )


def _metrics(model, pred, y, support):
    target = y[:, support:].to(pred.device)
    if pred.ndim != 3 or pred.shape[:2] != target.shape:
        raise ValueError(f"Prediction shape {tuple(pred.shape)} does not match target {tuple(target.shape)}")
    if model.max_classes > 0:
        if pred.shape[-1] != model.max_classes:
            raise ValueError(
                f"Pretraining evaluation requires {model.max_classes} logits; got {pred.shape[-1]}. "
                "Use the training forward path under no_grad, not the deployment eval path."
            )
        _validate_labels(target, pred.shape[-1], "Query targets")
    if not torch.isfinite(pred).all().item():
        raise FloatingPointError("Nonfinite diagnostic predictions")
    row = {
        "prediction_rms": _float(pred.square().mean().sqrt()),
        "prediction_nonfinite_fraction": _float((~torch.isfinite(pred)).float().mean()),
    }
    if model.max_classes > 0:
        row["cross_entropy"] = _float(F.cross_entropy(pred.flatten(0, 1), target.long().flatten()))
        row["accuracy"] = _float((pred.argmax(-1) == target).float().mean())
        row["confidence"] = _float(pred.softmax(-1).amax(-1).mean())
    else:
        median = pred.shape[-1] // 2
        row["median_mse"] = _float((pred[..., median] - target.float()).square().mean())
    return row


@contextmanager
def _pretraining_evaluation(model):
    # TabICL eval() truncates/remaps classification outputs for deployment.
    # Match train._screening.evaluate_cases: full classifier, no gradients.
    if any(isinstance(module, torch.nn.Dropout) and module.p for module in model.modules()):
        raise ValueError("Deterministic pretraining diagnostics require dropout=0")
    modes = {module: module.training for module in model.modules()}
    try:
        model.train()
        yield
    finally:
        for module, training in modes.items():
            module.training = training


@torch.inference_mode()
def evaluate(model, modules, cases, device, dtype, policies, query_count, seed, permutation):
    with _pretraining_evaluation(model):
        return _evaluate(model, modules, cases, device, dtype, policies, query_count, seed, permutation)


def _evaluate(model, modules, cases, device, dtype, policies, query_count, seed, permutation):
    records = []
    amp = dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"
    for case in cases:
        case_id = case.get("id", "unknown")
        if model.max_classes > 0:
            _validate_labels(case["y"], model.max_classes, f"Case {case_id}")
        x, y, support = case["x"].to(device), case["y"].to(device), int(case["train_size"])
        if x.ndim != 3 or y.shape != x.shape[:2] or not 0 < support < x.shape[1]:
            raise ValueError(f"Case {case_id}: invalid X/y shape or support split")
        for policy_index, policy in enumerate(policies):
            saved = _install_policy(modules, policy, seed + policy_index)
            diag_state = _enable_diagnostics(modules, query_count, support)
            try:
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                    pred = model(x, y[:, :support]).float()
                record = {"case_id": case.get("id", "unknown"), "support": support,
                          "features": int(x.shape[-1]), "policy": policy}
                record.update(_metrics(model, pred, y, support))
                for name, module in modules:
                    for key, value in module.last_diagnostics.items():
                        record[f"layer/{name}/{key}"] = _float(value)
                if policy == "native" and permutation:
                    perm = torch.randperm(support, generator=torch.Generator().manual_seed(seed), device="cpu").to(device)
                    xp = torch.cat((x[:, :support].index_select(1, perm), x[:, support:]), dim=1)
                    yp = y[:, :support].index_select(1, perm)
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                        perm_pred = model(xp, yp).float()
                    delta = (pred - perm_pred).abs()
                    record["support_permutation_logit_max_abs"] = _float(delta.max())
                    record["support_permutation_logit_mean_abs"] = _float(delta.mean())
                    if model.max_classes > 0:
                        record["support_permutation_prediction_flip_fraction"] = _float(
                            (pred.argmax(-1) != perm_pred.argmax(-1)).float().mean()
                        )
                records.append(record)
            except (ValueError, FloatingPointError) as exc:
                raise type(exc)(f"Case {case_id}, policy {policy}: {exc}") from exc
            finally:
                _restore_diagnostics(diag_state)
                _restore(saved)
    return records


def _summarize(records):
    groups = defaultdict(lambda: defaultdict(list))
    for record in records:
        key = (record["support"], record["policy"])
        for name, value in record.items():
            if name not in {"case_id", "support", "features", "policy"} and isinstance(value, (int, float)):
                if math.isfinite(float(value)):
                    groups[key][name].append(float(value))
    summary = []
    for (support, policy), metrics in sorted(groups.items()):
        row = {"support": support, "policy": policy, "cases": len({r["case_id"] for r in records
                                                                          if r["support"] == support and r["policy"] == policy})}
        for name, values in sorted(metrics.items()):
            values.sort()
            row[name] = {
                "mean": sum(values) / len(values),
                "std": (sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)) ** 0.5,
                "p50": values[len(values) // 2],
                "p90": values[min(len(values) - 1, int(0.9 * len(values)))],
            }
        summary.append(row)
    return summary


def _checkpoint_step(path):
    match = re.fullmatch(r"step-(\d+)\.ckpt", path.name)
    return int(match.group(1)) if match else None


def resolve_checkpoints(source, every, requested_steps):
    if source.is_file():
        return [source.resolve()]
    if not source.is_dir():
        raise ValueError(f"Checkpoint path does not exist: {source}")
    found = [(step, path.resolve()) for path in source.glob("step-*.ckpt")
             if (step := _checkpoint_step(path)) is not None]
    found.sort()
    if not found:
        raise ValueError(f"No step-*.ckpt files found in {source}")
    available = {step: path for step, path in found}
    if requested_steps:
        missing = sorted(set(requested_steps) - available.keys())
        if missing:
            raise ValueError(f"Requested checkpoint steps are absent: {missing}")
        return [available[step] for step in sorted(set(requested_steps))]
    if every == 0:
        return [path for _, path in found]
    selected = [(step, path) for step, path in found if step % every == 0]
    # Always retain endpoints so a partial run still has a before/after view.
    by_step = {step: path for step, path in ([(found[0][0], found[0][1]), *selected, (found[-1][0], found[-1][1])])}
    return [by_step[step] for step in sorted(by_step)]


def generate_cases(path, supports, case_count, queries, features, seed):
    meta = {"support": supports, "cases": case_count, "queries": queries,
            "features": features, "seed": seed, "task": "classification", "prior": "graph_scm"}
    if path.exists():
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("meta") != meta:
            raise ValueError(f"Existing auto-generated case configuration differs: {path}")
        return data
    cases = []
    for support in supports:
        for index in range(case_count):
            case_seed = seed + support * 1009 + index
            np.random.seed(case_seed % 2**32)
            random.seed(case_seed)
            torch.manual_seed(case_seed)
            feature_count = features[index % len(features)]
            prior = PriorDataset(
                regression=False, batch_size=1, batch_size_per_gp=1,
                min_features=feature_count, max_features=feature_count, max_classes=10,
                max_seq_len=support + queries, min_train_size=support,
                max_train_size=support + 1, prior_type="graph_scm", n_jobs=1,
                config=PriorConfig(filter_unpredictable_graphs=True, filter_unpredictable_datasets=True),
            )
            x, y, _, _, sizes = prior.get_batch()
            if int(sizes[0]) != support:
                raise RuntimeError(f"Generated support size {int(sizes[0])}, expected {support}")
            cases.append({"id": f"support_{support}/case_{index}/seed_{case_seed}",
                          "x": x.cpu(), "y": y.cpu(), "train_size": support})
            print(f"Prepared support={support}, case={index + 1}/{case_count}", flush=True)
    data = {"meta": meta, "cases": cases}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data


def diagnose_checkpoint(path, cases_data, device, dtype, policies, query_count, seed, permutation):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint must contain a TabICL model config: {path}")
    model = TabICL(**config).to(device).eval()
    state = checkpoint["state_dict"]
    if state and all(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)
    modules = [(name, module) for name, module in model.icl_predictor.named_modules()
               if isinstance(module, SetRetrievalAttention)]
    if not modules:
        raise ValueError(f"Checkpoint architecture has no SetRetrievalAttention modules: {path}")
    records = evaluate(model, modules, cases_data["cases"], device, dtype, policies,
                       query_count, seed, permutation)
    return {
        "checkpoint": str(path), "step": checkpoint.get("curr_step"), "config": config,
        "layers": [{"name": name, "mode": module.mode, "global_type": module.global_type,
                    "block_size": module.block_size, "selected_blocks": module.num_selected_blocks,
                    "router": module.router, "partition": module.partition} for name, module in modules],
        "records": records, "summary": _summarize(records),
    }


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path, nargs="?",
        default=REPO_ROOT / "checkpoints/tabiclv2-clf/stage1",
        help="One checkpoint or a directory containing step-*.ckpt "
             "(default: <repo>/checkpoints/tabiclv2-clf/stage1)",
    )
    parser.add_argument("cases", type=Path, nargs="?", help="Optional existing validation.pt")
    parser.add_argument("--output", type=Path, help="Default: <checkpoint directory>/icl-mechanism.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--diagnostic-queries", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-permutation", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=20000,
                        help="For a directory, sample this step interval plus endpoints; 0 evaluates all")
    parser.add_argument("--steps", type=int, nargs="+", help="Exact checkpoint steps to evaluate")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Merge with a compatible existing output and skip completed checkpoints")
    parser.add_argument("--eval-support", type=int, nargs="+", default=[384, 640, 896, 4096])
    parser.add_argument("--eval-cases", type=int, default=8)
    parser.add_argument("--eval-queries", type=int, default=128)
    parser.add_argument("--eval-features", type=int, nargs="+", default=[5, 20, 50])
    args = parser.parse_args()
    if args.diagnostic_queries < 0 or args.checkpoint_every < 0:
        parser.error("--diagnostic-queries and --checkpoint-every must be nonnegative")
    if min(args.eval_support + args.eval_features + [args.eval_cases, args.eval_queries]) < 1:
        parser.error("Case generation counts must be positive")
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    try:
        checkpoints = resolve_checkpoints(args.checkpoint, args.checkpoint_every, args.steps)
    except ValueError as exc:
        parser.error(str(exc))
    checkpoint_source = args.checkpoint.expanduser().resolve()
    default_output_dir = checkpoint_source if checkpoint_source.is_dir() else checkpoint_source.parent
    output = (args.output.expanduser().resolve() if args.output
              else default_output_dir / "icl-mechanism.json")
    cases_path = args.cases.resolve() if args.cases else output.with_suffix(".cases.pt")
    if args.cases and not cases_path.is_file():
        parser.error(f"Cases file does not exist: {cases_path}")
    cases_data = (torch.load(cases_path, map_location="cpu", weights_only=True)
                  if args.cases else generate_cases(cases_path, args.eval_support, args.eval_cases,
                                                    args.eval_queries, args.eval_features, args.seed))
    cases = cases_data.get("cases") if isinstance(cases_data, dict) else None
    if not cases:
        parser.error("Cases file must contain a non-empty 'cases' list")
    report = {
        "checkpoint_source": str(args.checkpoint.resolve()),
        "checkpoint_selection": {"every": args.checkpoint_every, "requested_steps": args.steps,
                                 "paths": [str(path) for path in checkpoints]},
        "cases_file": str(cases_path),
        "case_meta": cases_data.get("meta", {}),
        "device": str(device),
        "dtype": args.dtype,
        "policies": args.policies,
        "diagnostic_config": {"queries": args.diagnostic_queries, "seed": args.seed,
                              "support_permutation": not args.no_permutation,
                              "execution_mode": "pretraining_no_grad_v1"},
        "checkpoints": [],
        "interpretation_notes": [
            "global_only and sparse_only are controlled ablations of SetRetrievalAttention layers; other model layers remain unchanged.",
            "random_candidates preserves the router's key ordering but uniformly samples the same number of blocks.",
            "oracle_candidates chooses blocks by exact dense-softmax probability mass and is an upper bound for this block budget, not a deployable router.",
            "dense_only runs exact full-support softmax in retrieval layers and is a compute/quality reference, not a fallback path.",
            "Support permutation compares predictions after jointly permuting support rows and labels; it should be close to zero for a set-based ICL mechanism.",
        ],
    }
    if args.resume and output.exists():
        previous = json.loads(output.read_text())
        compatibility = ("checkpoint_source", "cases_file", "device", "dtype", "policies",
                         "diagnostic_config")
        mismatches = [key for key in compatibility if previous.get(key) != report.get(key)]
        if mismatches:
            parser.error(f"Cannot resume incompatible report; differing fields: {mismatches}")
        report["checkpoints"] = previous.get("checkpoints", [])
        old_paths = previous.get("checkpoint_selection", {}).get("paths", [])
        report["checkpoint_selection"]["paths"] = sorted(
            set(old_paths + report["checkpoint_selection"]["paths"]),
            key=lambda name: _checkpoint_step(Path(name)) or -1,
        )
    completed = {row["checkpoint"] for row in report["checkpoints"]}
    pending = [path for path in checkpoints if str(path) not in completed]
    print(f"Selected {len(checkpoints)} checkpoints: {[_checkpoint_step(p) for p in checkpoints]}; "
          f"pending={len(pending)}", flush=True)
    for index, path in enumerate(pending, 1):
        print(f"Diagnosing {path.name} ({index}/{len(pending)})", flush=True)
        result = diagnose_checkpoint(path, cases_data, device, dtype, args.policies,
                                     args.diagnostic_queries, args.seed, not args.no_permutation)
        report["checkpoints"].append(result)
        report["checkpoints"].sort(key=lambda row: row.get("step") or -1)
        # Keep completed checkpoints recoverable if a later one fails.
        _write_report(output, report)
    print(f"Report contains {len(report['checkpoints'])} checkpoints x {len(cases)} cases x "
          f"{len(args.policies)} policies")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
