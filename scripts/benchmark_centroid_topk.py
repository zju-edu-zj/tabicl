"""Route selection correctness and synchronized wall latency, including guards."""
import argparse
import json
from pathlib import Path
import time

import torch

from tabicl._model._routing_triton import centroid_scores, centroid_topk


def reference(q, centers, budget, elements):
    chunk = max(1, elements // (q.shape[0]*q.shape[1]*centers.shape[-2]))
    return torch.cat([centroid_scores(part, centers).topk(budget, -1).indices
                      for part in q.split(chunk, dim=2)], 2).to(torch.int32)


def measure(fn, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-started)*1000)
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()-allocated
    del output
    return dict(mean_ms=sum(samples)/len(samples), min_ms=min(samples), samples_ms=samples,
                peak_temporary_and_output_bytes=peak)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--rows', nargs='+', type=int, default=[2048, 16384, 60000])
    parser.add_argument('--leaves', nargs='+', type=int, default=[13, 64, 128, 375])
    parser.add_argument('--chunk-elements', type=int, default=8*1024**2)
    args = parser.parse_args()
    torch.manual_seed(721)
    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        for rows in args.rows:
            for leaves in args.leaves:
                q = torch.randn(1, rows, 8, 64, device='cuda', dtype=dtype).transpose(1, 2)
                centers = torch.randn(1, 8, leaves, 64, device='cuda')
                elements, budget = args.chunk_elements, 4
                old = lambda: reference(q, centers, budget, elements)
                new = lambda: centroid_topk(q, centers, budget, elements)
                expected, actual = old(), new()
                assert torch.equal(expected, actual), (dtype, rows, leaves, (expected != actual).sum())
                case = dict(dtype=str(dtype), rows=rows, leaves=leaves,
                            reference=measure(old, args.repeats), optimized=measure(new, args.repeats))
                cases.append(case)
                print(json.dumps(case), flush=True)
    q = torch.randn(1, 129, 8, 64, device='cuda').transpose(1, 2)
    for kind in ('zero', 'duplicate', 'nan', 'inf'):
        centers = torch.randn(1, 8, 17, 64, device='cuda')
        if kind == 'zero':
            centers.zero_()
        elif kind == 'duplicate':
            centers[:, :, 3:] = centers[:, :, :1]
        else:
            centers[:, :, 0, 0] = float(kind)
        assert torch.equal(reference(q, centers, 4, 8192), centroid_topk(q, centers, 4, 8192)), kind
    args.output.write_text(json.dumps(dict(device=torch.cuda.get_device_name(), cases=cases,
                                          tie_and_nonfinite_checks=True), indent=2)+'\n')


if __name__ == '__main__':
    main()
