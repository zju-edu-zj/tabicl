"""Benchmark-only entry point: use the original framework centroid selector."""
import argparse

import torch

from tabicl._model.icl_attention import SetRetrievalAttention
from tabicl.train._run import Trainer
from tabicl.train._train_config import build_parser


class ReferenceTrainer(Trainer):
    def build_model(self):
        super().build_model()
        for module in self.raw_model.modules():
            if isinstance(module, SetRetrievalAttention):
                module.use_fast_centroid_topk = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--shape-prior", action="store_true")
    options, args = parser.parse_known_args()
    if options.shape_prior:
        from train_shape_prior import ShapeDataset
        from tabicl.train import _run
        _run.PriorDataset = ShapeDataset
    torch.multiprocessing.set_start_method("spawn", force=True)
    ReferenceTrainer(build_parser().parse_args(args)).train()
