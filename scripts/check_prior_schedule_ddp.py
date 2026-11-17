"""Real model/DDP gradient contract for CPU-only 8/16-rank scheduling checks.

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=16 \
  scripts/check_prior_schedule_ddp.py

Small tables/architecture check correctness, not production GPU throughput.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import json

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from tabicl._model.tabicl import TabICL
from tabicl.prior._dataset import GraphPrior
from tabicl.prior._node_pool import NodePriorPool, iter_micro_batches
from tabicl.prior.graph_lib._config import PriorConfig


def loss(model, micro):
    x, y, d, _, support = micro
    n = int(support[0])
    return F.cross_entropy(model(x, y[:, :n]).flatten(end_dim=-2), y[:, n:].flatten())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", choices=["balanced", "round_robin", "window_fifo", "window_bucketed", "shape_cohort"], default="balanced")
    parser.add_argument("--transport", choices=["numpy", "mmap"], default="numpy")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--bucket-steps", type=int, default=2)
    parser.add_argument("--kernel-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    reports = []
    for stage in (1, 2, 3):
        torch.manual_seed(41)
        model = TabICL(max_classes=3, embed_dim=16, col_num_blocks=1,
                       col_nhead=2, col_num_inds=4, row_num_blocks=1,
                       row_nhead=2, row_num_cls=2, icl_num_blocks=3,
                       icl_nhead=2, icl_attention="global_sparse",
                       icl_ssmax=False,
                       icl_block_size=16, icl_num_selected_blocks=1,
                       zero_init=False).train()
        reference = deepcopy(model) if rank == 0 else None
        # Match Trainer: attention caches/buffers depend on each rank's shape.
        # Broadcasting differently sized buffers breaks the collective contract.
        ddp = DDP(model, broadcast_buffers=False)
        micro = 4 if stage == 1 else 1
        prior = GraphPrior(batch_size=4, batch_size_per_gp=micro,
                           min_features=2, max_features=6, max_classes=3,
                           min_seq_len=32, max_seq_len=64 if stage < 3 else 128,
                           seq_len_per_gp=True, min_train_size=.4, max_train_size=.6,
                           config=PriorConfig(min_n_nodes=2, max_n_nodes=4))
        with NodePriorPool(prior, workers=4, window=2*world, seed=93,
                           key=f"schedule-ddp-{__import__('os').environ['MASTER_PORT']}-{stage}",
                           schedule=args.schedule, micro_size=micro,
                           transport=args.transport, bucket_steps=args.bucket_steps,
                           kernel_threads=args.kernel_threads) as pool:
            opt = torch.optim.SGD(ddp.parameters(), lr=.01)
            ref_opt = torch.optim.SGD(reference.parameters(), lr=.01) if rank == 0 else None
            for step in range(args.steps):
                batch = next(pool)
                batches = [None] * world
                dist.all_gather_object(batches, batch)
                opt.zero_grad(set_to_none=True)
                units = list(iter_micro_batches(batch, micro))
                for i, unit in enumerate(units):
                    with ddp.no_sync() if i < len(units)-1 else nullcontext():
                        (loss(ddp, unit) / len(units)).backward()
                opt.step()
                if rank == 0:
                    ref_opt.zero_grad(set_to_none=True)
                    for candidate in batches:
                        for unit in iter_micro_batches(candidate, micro):
                            (loss(reference, unit) / (world * len(units))).backward()
                    ref_opt.step()
                    maximum = 0.
                    for (name, a), (_, b) in zip(model.named_parameters(), reference.named_parameters()):
                        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5, msg=name)
                        maximum = max(maximum, float((a-b).abs().max().detach()))
                    reports.append(dict(stage=stage, step=step, ranks=world, micro=micro,
                                        max_parameter_error=maximum, pool=pool.stats()))
                dist.barrier()
        del ddp, model, reference
    if rank == 0:
        print(json.dumps(reports, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
