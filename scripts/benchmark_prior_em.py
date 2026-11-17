"""ABBA microbenchmark for the complete EM posterior AND output projection.

This explains kernel behavior; use benchmark_prior_exact for overall speedup.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from tabicl.prior.graph_lib._distance import compact_cdist, compact_em_probabilities, _native_available


def timed(function, minimum):
    count, started = 0, time.perf_counter()
    while count == 0 or time.perf_counter() - started < minimum:
        function()
        count += 1
    return (time.perf_counter() - started)/count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum', type=float, default=.03)
    parser.add_argument('--center-sweep', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    assert _native_available() and hasattr(torch.ops.tabicl_prior_exact, 'em_probabilities_cpu')
    records = []
    shapes = ([(d, k) for d in (8, 100) for k in (2, 3, 5, 7, 8, 9, 12, 15, 16, 17, 31, 32, 33, 64)]
              if args.center_sweep else [(1, 2), (8, 5), (32, 16), (100, 64)])
    for n in ([2048] if args.center_sweep else (2048, 4096)):
        for d, k in shapes:
            for p in ([1.7] if args.center_sweep else (.8, 1.7, 3.5)):
                for repeated in ([False] if args.center_sweep else (False, True)):
                    gen = torch.Generator().manual_seed(n+d+k)
                    x = torch.randn(n, d, generator=gen)
                    if repeated:
                        x = x[:4].repeat(n//4, 1)
                    centers = torch.randn(k, d, generator=gen)
                    stds = torch.exp(torch.randn(k, generator=gen))
                    constants = -torch.log(2*torch.pi*stds[None, :]**2)/2
                    q, weights = (1.63 if args.center_sweep else 1.7), torch.randn(k, 8, generator=gen)
                    scale = stds.pow(-q)
                    def old():
                        distances = compact_cdist(x, centers, p, vectorized=True)
                        return torch.softmax(constants-(distances/stds).clamp_min(0.)**q, -1) @ weights
                    def new():
                        probs, inverse = compact_em_probabilities(x, centers, stds, constants, p, q, scale)
                        output = probs @ weights
                        return output if inverse is None else output[inverse]
                    a, b = old(), new()
                    # Include sharp, high-dimensional p<1 stress cases. Record
                    # both float32 paths against a double-precision oracle,
                    # rather than treating old float32 rounding as ground truth.
                    oracle = torch.softmax(constants.double() - (
                        torch.cdist(x.double(), centers.double(), p=p)/stds.double())**q, -1) @ weights.double()
                    torch.testing.assert_close(b.double(), oracle, atol=1e-3, rtol=1e-3)
                    times = [timed(f, args.minimum) for f in (old, new, new, old)]
                    records.append(dict(n=n, d=d, k=k, p=p, repeated=repeated,
                        old_seconds=(times[0]+times[3])/2, new_seconds=(times[1]+times[2])/2,
                        speedup=(times[0]+times[3])/(times[1]+times[2])-1,
                        max_abs_error=(a-b).abs().max().item(),
                        old_oracle_error=(a-oracle).abs().max().item(),
                        new_oracle_error=(b-oracle).abs().max().item()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2))
    print(json.dumps(dict(cases=len(records), min_speedup=min(r['speedup'] for r in records),
        max_speedup=max(r['speedup'] for r in records))), flush=True)


if __name__ == '__main__':
    main()
