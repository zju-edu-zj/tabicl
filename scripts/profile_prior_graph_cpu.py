"""Single-worker complete-task cProfile, separate from throughput measurement."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import random

import numpy as np
import torch

from benchmark_prior_exact import initialize, generate
from tabicl.prior._dataset import GraphPrior
from tabicl.prior.graph_lib._config import PriorConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=int, default=128)
    p.add_argument('--rows', type=int, default=2048)
    p.add_argument('--seed', type=int, default=83)
    p.add_argument('--graph-execution', choices=['reference', 'optimized'], default='reference')
    p.add_argument('--filter-execution', choices=['reference', 'planned', 'sufficient'], default='reference')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    prior = GraphPrior(batch_size=8, batch_size_per_gp=4, n_jobs=1,
                       min_features=1, max_features=100, max_classes=10,
                       max_seq_len=a.rows, seq_len_per_gp=True,
                       min_train_size=.3, max_train_size=.9,
                       config=PriorConfig(categorical_sampling='marginal', graph_execution=a.graph_execution,
                           filter_execution=a.filter_execution,
                           filter_unpredictable_graphs=True, filter_unpredictable_datasets=True))
    params = []
    while len(params) < a.tasks:
        params.extend(prior.sample_parameters())
    initialize(prior, a.seed)
    profiler = cProfile.Profile()
    profiler.enable()
    records = [generate((i, param)) for i, param in enumerate(params[:a.tasks])]
    profiler.disable()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(str(a.output.with_suffix('.pstats')))
    stats = pstats.Stats(profiler)
    functions = [dict(file=k[0], line=k[1], function=k[2], primitive_calls=v[0], calls=v[1],
                      self_s=v[2], cumulative_s=v[3]) for k, v in stats.stats.items()]
    a.output.write_text(json.dumps(dict(records=records, functions=functions), indent=2))
    stats.strip_dirs().sort_stats('cumulative').print_stats(45)
    stats.sort_stats('tottime').print_stats(30)


if __name__ == '__main__':
    main()
