"""Synthetic routing audit, not a trained model accuracy benchmark."""

import argparse
import json
from pathlib import Path

import torch

from tabicl._model.icl_attention import SetRetrievalAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", type=int, default=65537)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--beams", type=int, nargs="*", default=[4, 16, 32])
    parser.add_argument("--partitions", nargs="+", choices=("projection", "content_tree"),
                        default=["projection", "content_tree"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(8 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    results = []
    with torch.no_grad():
        for scenario in ("random", "clustered"):
            for seed in range(args.seeds):
                print(f"Auditing {scenario}, seed={seed}", flush=True)
                torch.manual_seed(seed)
                module = SetRetrievalAttention(512, 8).cuda()
                if scenario == "random":
                    k = torch.randn(1, 8, args.support, 64, device="cuda")
                    q = torch.randn(1, 8, args.queries, 64, device="cuda")
                    v = torch.randn_like(k)
                else:
                    centers = torch.randn(1, 8, 32, 64, device="cuda")
                    kid = torch.randint(32, (args.support,), device="cuda")
                    qid = torch.randint(32, (args.queries,), device="cuda")
                    k = centers[:, :, kid] + 0.2 * torch.randn(1, 8, args.support, 64, device="cuda")
                    q = centers[:, :, qid] + 0.2 * torch.randn(1, 8, args.queries, 64, device="cuda")
                    value_centers = torch.randn_like(centers)
                    v = value_centers[:, :, kid] + 0.2 * torch.randn_like(k)
                q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
                global_out = module._global_attention(q, k, v).float()
                for partition in args.partitions:
                    module.partition, module.router = partition, "centroid"
                    order, exact = module.route(q, k)
                    sorted_k = k.gather(2, order[..., None].expand_as(k))
                    sorted_v = v.gather(2, order[..., None].expand_as(v)).float()
                    logits = q.float() @ sorted_k.float().transpose(-1, -2) / 8
                    prob = logits.softmax(-1)
                    dense_out = prob @ sorted_v
                    denom = dense_out.square().mean().sqrt().clamp_min(1e-6)
                    top = logits.topk(32, -1).indices
                    block_ids = torch.arange(args.support, device="cuda") // 128
                    for beam in [None, *args.beams]:
                        print(f"  partition={partition}, beam={beam}", flush=True)
                        module.router = "centroid" if beam is None else "hierarchical"
                        module.router_beam = beam or 16
                        chosen = exact if beam is None else module.route(q, k)[1]
                        mask = (chosen[..., None] == block_ids).any(-2)
                        sparse_out = logits.masked_fill(~mask, -float("inf")).softmax(-1) @ sorted_v
                        record = dict(
                            scenario=scenario, seed=seed, partition=partition, router=module.router, beam=beam,
                            leaf_recall=float((chosen[..., None] == exact[..., None, :]).any(-1).float().mean()),
                            token_recall_at_32=float(mask.gather(-1, top).float().mean()),
                            retained_mass=float((prob * mask).sum(-1).mean()),
                            candidate_fraction=float(mask.float().mean()),
                            sparse_output_relative_error=float((sparse_out - dense_out).square().mean().sqrt() / denom),
                            global_output_relative_error=float((global_out - dense_out).square().mean().sqrt() / denom),
                            fused_half_relative_error=float(((sparse_out + global_out) / 2 - dense_out).square().mean().sqrt() / denom),
                        )
                        if scenario == "clustered":
                            labels = kid[order]
                            record["candidate_same_cluster_fraction"] = float(
                                (mask * (labels[:, :, None] == qid[None, None, :, None])).sum().float() / mask.sum())
                        results.append(record)
    output = dict(
        support=args.support,
        queries=args.queries,
        heads=8,
        head_dim=64,
        dtype="bfloat16",
        block_size=128,
        selected_blocks=4,
        results=results,
        limitation="Synthetic Q/K only; no trained model or downstream accuracy claims.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
