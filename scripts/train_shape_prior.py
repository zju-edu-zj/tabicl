"""Benchmark-only training entry point: cheap tables with sampled prior shapes.

Used to isolate GPU scheduling from graph generation/rejection tails. Widths
are requested widths, not post-filter widths; random labels are NOT a training
or quality experiment. No production sampler/filter is changed.
"""
import torch

from tabicl.prior._dataset import GraphPrior, PriorDataset
from tabicl.train import _run
from tabicl.train._train_config import build_parser


class ShapePrior(GraphPrior):
    def generate_dataset(self, params, *, compact=False):
        rows, width = params["seq_len"], params["num_features"]
        x = torch.randn(rows, width)
        y = (torch.randn(rows) if params["regression"] else
             torch.randint(params["num_classes"], (rows,)))
        return x, y, torch.tensor(width)


class ShapeDataset(PriorDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior.__class__ = ShapePrior


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    _run.PriorDataset = ShapeDataset
    _run.Trainer(build_parser().parse_args()).train()
