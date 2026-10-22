# Prior generation bottleneck audit

## Root cause

The Stage 1 recipe runs 16 DDP ranks and previously gave every rank 16
DataLoader workers. On a 128-core node this creates 256 graph generators before
counting the training processes. Each DataLoader item is one local batch of four
graph_scm datasets.

graph_scm generation has a heavy-tailed cost. A profile of 30 Stage 1 batches
found that GraphSCM itself used 78.4% of wall time and the ExtraTrees
predictability filter used 20.1%. Producing 120 accepted datasets required 181
GraphSCM calls; invalid splits and predictability filtering rejected the rest.
Within GraphSCM, arbitrary-p `torch.cdist` calls from random discretizers and
categorical conversion are the largest tensor-operation cost. Expensive samples
are therefore part of the configured prior, rather than a single accidental
sleep or I/O operation.

With an ordered DataLoader, a slow assigned item blocks already-completed items
behind it. DDP amplifies that local head-of-line stall: the late rank blocks all
other ranks at the next gradient collective. Rank-zero `prior_time` can remain
near zero while the same delay appears in another rank's `train_time`.

## Implemented policy

The classifier Stage 1 launcher now defaults to:

- 8 workers per rank, matching 128 workers across 16 local ranks;
- prefetch factor 8, for 64 queued local batches per rank;
- `in_order=False`, allowing the next completed graph batch to bypass a slow
  worker assignment;
- one Torch thread inside each worker, avoiding nested CPU oversubscription.

These settings do not discard or regenerate slow samples. They change the order
in which independently generated online batches are consumed. Deterministic
`prior_seed_by_step` streams reject `in_order=False`, because ready-first
delivery would invalidate the optimizer-step to seed mapping.

Environment overrides for Stage 1 are:

```bash
PRIOR_WORKERS=8 PRIOR_PREFETCH_FACTOR=8 PRIOR_IN_ORDER=False \
  bash scripts/train_v2_clf_stage1.sh
```

At startup rank zero prints the effective workers per rank/node, prefetch depth,
and ordering policy. On multi-node jobs, size `PRIOR_WORKERS` against the CPU
cores and `LOCAL_WORLD_SIZE` of each node, not the global world size.

## Measurements

All local tests used the exact Stage 1 graph_scm settings: four datasets per
rank, 1,024 rows, up to 100 features and 32 graph nodes, both graph and dataset
predictability filters enabled.

Single-rank maximum-throughput test, 80 measured batches:

| Configuration | Mean wait | p95 | p99 | Maximum | waits >1 s |
|---|---:|---:|---:|---:|---:|
| 16 workers, prefetch 2, ordered | 120 ms | 845 ms | 2,125 ms | 2,962 ms | 5.0% |
| 8 workers, prefetch 8, ready-first | 145 ms | 309 ms | 375 ms | 390 ms | 0% |

The optimized configuration deliberately trades a small amount of peak average
throughput for a much tighter tail and half as many processes.

Four-rank pressure test with 200 ms of simulated GPU work, comparing equal four
workers per rank:

| Configuration | Mean max-rank wait | p95 max-rank wait | Maximum | steps >1 s |
|---|---:|---:|---:|---:|
| prefetch 2, ordered | 749 ms | 2,732 ms | 3,136 ms | 30% |
| prefetch 8, ready-first | 451 ms | 878 ms | 1,112 ms | 3% |

With 550 ms simulated GPU work, the optimized pipeline's max-rank wait was 10
ms mean, 14 ms p95 and 97 ms maximum, so generation was effectively hidden by
prefetch on this machine. The full 16-rank result must still be measured on the
128-core training node:

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=16 \
  scripts/benchmark_prior_pipeline.py \
  --workers 8 --prefetch 8 --in-order False \
  --consumer-ms 550 --warmup 32 --steps 500 \
  --output prior_pipeline_stage1.json
```

The relevant number is `max_rank_wait`, not a single rank's median.

## Deeper generator changes

Further GraphSCM work is justified for Stage 2/3, where sequence length makes
the generator's intrinsic tail much larger, or if the 16-rank benchmark still
shows material `max_rank_wait`. The next candidates are:

1. A node-wide bounded producer queue, decoupled from individual DDP ranks.
2. Batched implementations of random discretization/categorical distance work.
3. A cheaper learned or vectorized predictability filter validated against the
   current ExtraTrees accept/reject decisions.
4. Enabling `use_corrected_cat_meta_sampling` for a new training run only. It
   improved local generation mean by about 23%, but changes categorical
   hyperparameter correlation and must not be silently introduced mid-run.

Reducing maximum features, graph nodes, function families or filter strength is
not a pure performance optimization. Those changes alter the prior and require
an accuracy-controlled training experiment.
