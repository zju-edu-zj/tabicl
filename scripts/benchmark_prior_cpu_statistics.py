"""Sequential ABBA microbenchmarks; full-prior tests determine adoption."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from tabicl.prior.graph_lib import _distance as dist


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    assert dist._native_available()
    ops = torch.ops.tabicl_prior_exact
    results = []
    for n in (2048, 4096):
        for d in (1, 2, 3, 8, 32, 100):
            for stride in (1, 2):
                x = torch.randn(n, d*stride, generator=torch.Generator().manual_seed(n+d))[:, ::stride]
                def old_stats():
                    return x.mean(0, keepdim=True), x.std(0, correction=0, keepdim=True)
                def new_stats():
                    return ops.column_std_mean_cpu(x)
                for repeat in (False, True):
                    v = x if not repeat else x[:7].repeat((n+6)//7, 1)[:n]
                    def old_unique():
                        import os
                        before = os.environ.get('TABICL_PRIOR_COMPACT')
                        os.environ['TABICL_PRIOR_COMPACT'] = 'sort'
                        try:
                            return dist._compact_rows(v, 16)
                        finally:
                            if before is None: os.environ.pop('TABICL_PRIOR_COMPACT')
                            else: os.environ['TABICL_PRIOR_COMPACT'] = before
                    for component, old, new in [('compact', old_unique, lambda: ops.compact_rows_cpu(v))] + (
                            [('statistics', old_stats, new_stats)] if not repeat else []):
                        timings = {'old': [], 'new': []}
                        for label, f in [('old', old), ('new', new), ('new', new), ('old', old)]:
                            for _ in range(10): f()
                            start = time.perf_counter()
                            for _ in range(200): f()
                            timings[label].append((time.perf_counter()-start)/200)
                        old_s, new_s = (statistics.mean(timings[k]) for k in ('old', 'new'))
                        results.append(dict(component=component, rows=n, width=d, stride=stride,
                            repeated=repeat, old_s=old_s, new_s=new_s, gain=old_s/new_s-1))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
