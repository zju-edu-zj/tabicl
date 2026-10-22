Now very interesting: in exp14, var + randn + ddp shows NO jitter (max 266ms, 0% spikes), unlike exp12's var (max 18.9s)!

  The difference between exp12 and exp14 var randn ddp: exp12 used torch.randn(1, L, H) generated inside fetch() on CPU, and exp14 also does... Actually the difference
  is exp12's fetch_fixed/fetch_real — exp12's var mode used the real DataLoader. Let me confirm: exp12 base used fetch_real (DataLoader). exp12 fixedseq used
  fetch_fixed (randn). So exp12's "var" jitter included the DataLoader. exp14 var randn has NO DataLoader → no jitter.

  So the jitter requires the real DataLoader (CPU worker processes) even with fixed padding? No — exp13 padfixed (real DataLoader) still spiked. Let me wait for exp14's
  var real ddp and fixed real ddp to disambiguate.

• Ran cd /ossfs/workspace/tabicl && cat > experiments/exp14_matrix.py <<'PY'
  │ """Exp 14: 2x2x2 controlled matrix: {varlen, fixedlen} x {realdata, randn} x {ddp, nosync}.
  │ All in one harness so nothing else differs."""
  │ … +71 lines
  └ --- seq=var data=randn sync=ddp ---
    {"seq": "var", "data": "randn", "sync": "ddp", "med": 247.05695919692516, "mean": 243.59014629386365, "p95": 257.3240593075752, "mx": 266.109824180603, "sp": 0.0}
    … +10 lines (ctrl + t to view transcript)
    {"seq": "fixed", "data": "real", "sync": "ddp", "med": 138.5978665202856, "mean": 407.2696727886796, "p95": 1032.5950662605392, "mx": 20460.132632404566, "sp":
    0.055}

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Now the picture is definitive. The critical variable is data=real (the DataLoader with CPU worker processes):

   config                  median    mean     p95       max        spike%
  ━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━  ━━━━━━━  ━━━━━━━━  ━━━━━━━━━  ━━━━━━━━
   var + randn + ddp       247ms     244ms    257ms     266ms      0%
  ──────────────────────  ────────  ───────  ────────  ─────────  ────────
   fixed + randn + ddp     259ms     260ms    264ms     275ms      0%
  ──────────────────────  ────────  ───────  ────────  ─────────  ────────
   var + randn + nosync    245ms     242ms    255ms     257ms      0%
  ──────────────────────  ────────  ───────  ────────  ─────────  ────────
   var + real + ddp        140ms     411ms    1370ms    11535ms    6%
  ──────────────────────  ────────  ───────  ────────  ─────────  ────────
   fixed + real + ddp      139ms     407ms    1033ms    20460ms    5.5%

  The jitter is caused by the real DataLoader (CPU generation), NOT by variable length and NOT by DDP! Both var+real and fixed+real jitter identically (~6%, max 11-
  20s). randn has zero jitter regardless of var/fixed/ddp.

  So my earlier exp11/exp12 conclusion was wrong — those steps had the real DataLoader, and what I attributed to "bwd" was actually the GPU waiting for data that the
  CPU hadn't produced yet, because in exp11/exp12 the timer wrapped fetch().

  Wait — in exp11 I did time data separately and it showed data max 10.25s too. Let me re-examine: the issue is that when CPU generation stalls, the DataLoader worker
  blocks, and next(it) blocks — that's the data phase. But then how does it show in bwd?

  Let me look again: in exp11, bwd max was 10.4s and data max was 10.25s — both spiked at the same steps. That's because when one rank's DataLoader is slow, that rank's
  next(it) blocks (data), and then all other ranks block in the DDP all-reduce of the previous step waiting for it — no. Let me just directly verify: is the spike in
  data-wait or in the compute, NOW that I have clean randn vs real contrast?

  Real answer: it's the CPU DataLoader. Let me confirm precisely by re-running exp11's exact-per-phase profiler but separating data-blocking from compute, using the
  fixed vs real contrast.

• Ran cd /ossfs/workspace/tabicl && python3 -c "
  │ import numpy as np
  │ A=np.load('/tmp/jitter_exact.npy')  # exp11: (world,steps,6) data,fwd,bwd,opt,seq,mem
  │ … +13 lines
  └ exp11 (real data) spike steps: 16
    data.max(0) at spikes: mean=3.98s
    … +8 lines (ctrl + t to view transcript)
    step 150: data max rank= 1 (2.31s) | bwd max rank= 9 (2.43s)
    step 157: data max rank=11 (10.25s) | bwd max rank= 1 (10.38s)

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• This is the final piece. Look at the pattern:

  - At each spike, one rank has a huge data wait (e.g., step17 rank12 = 8.47s) and a different rank has a huge bwd (rank4 = 8.59s), with the values nearly equal (8.47 ≈
    8.59).

  - The spike propagates: rank 12's data stall (8.47s) → rank 12 finishes late → all other ranks block in the NEXT step's DDP all-reduce (bwd) waiting for rank 12
    (8.59s).

  So the root cause IS the CPU DataLoader stall after all — but the mechanism is: one rank's CPU generation stalls → that rank is late → all 16 ranks stall in the DDP
  all-reduce. This is the DDP-coupling amplifying a single rank's CPU tail, exactly the mechanism I originally described, and it's why randn (no CPU gen) has zero
  jitter regardless of DDP.

  So both effects are real and connected:

  1. CPU generation has a heavy tail (one rank occasionally takes 8-10s).
  2. DDP couples it — one straggler stalls all 16 ranks.

  My seq_len fix removed the systematic straggler but not the stochastic CPU tail. And workers=8 isn't enough. Let me now find why the CPU tail is 8-10s and fix it. Let
  me instrument the generator directly under the real concurrent load.

• Ran cd /ossfs/workspace/tabicl && cat > experiments/exp15_cpu_tail.py <<'PY'
  │ """Exp 15: find what causes the multi-second CPU generation tail under the real
  │ 16-rank x N-workers concurrency. Records per-get_batch wall time and the time
  │ … +39 lines
  └ --- procs=64 ---
    {"procs": 64, "n": 1600, "med": 0.6345561593770981, "mean": 1.107543714098865, "p90": 2.364089008048177, "p99": 6.821697057057168, "p999": 16.61657853868977, "mx":
    … +2 lines (ctrl + t to view transcript)
    {"procs": 128, "n": 3200, "med": 2.4497255962342024, "mean": 3.0878194604458984, "p90": 5.467115161567926, "p99": 13.599513320699327, "p999": 33.249697730451146,
    "mx": 42.002298375591636, "frac_gt1s": 0.9209375, "frac_gt3s": 0.3396875}

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Now the CPU tail is fully characterized:

  - 64 procs (≈ real training load: 16 ranks × 4 workers): med 0.63s, mean 1.11s, 30% of batches >1s, 6.8% >3s, max 38.6s.
  - 128 procs: med 2.45s (degraded), 92% >1s, max 42s.

  The CPU generation has a catastrophic tail — single get_batch calls taking up to 38-42 seconds! This is the true root cause. Since each rank's DataLoader holds ≥1 in-
  flight batch, and 16 ranks share CPU, a single rank's 8-40s generation stalls everyone via DDP.

  Now let me find what inside get_batch takes 8-40s. It's a n_jobs=4 inner Pool + GraphSCM. Under heavy oversubscription (64+ processes each spawning a 4-pool), the CPU
  is thrashed. Let me test the hypothesis: oversubscription — reduce inner n_jobs and see if the tail shrinks.

• Ran cd /ossfs/workspace/tabicl && sed -i 's/n_jobs=4,device="cpu",/n_jobs=int(os.environ.get("INNER_JOBS","4")),device="cpu",/' experiments/exp15_cpu_tail.py && sed
  │ -i 's/^def worker(seed,q):/def worker(seed,q):\n    import os/' experiments/exp15_cpu_tail.py && export TABICL_ALLOW_ANY_PYTHON=1 && for IJ in 1 4; do echo "---
  │ 64 procs, inner n_jobs=$IJ ---"; INNER_JOBS=$IJ timeout 600 python3 experiments/exp15_cpu_tail.py --procs 64 --n 25 2>&1 | grep -E "^\{" || echo FAILED; done
  └ --- 64 procs, inner n_jobs=1 ---
    {"procs": 64, "n": 1600, "med": 0.4638238428160548, "mean": 0.9194588070444297, "p90": 2.099419446103275, "p99": 6.406663788389414, "p999": 18.048090497832767,
    … +2 lines (ctrl + t to view transcript)
    {"procs": 64, "n": 1600, "med": 0.630033653229475, "mean": 1.123512939158827, "p90": 2.3432577351108197, "p99": 7.062636411972344, "p999": 16.030863231370297, "mx":
    38.81407658196986, "frac_gt1s": 0.306875, "frac_gt3s": 0.07}

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• inner n_jobs=1 slightly helps the median (0.46 vs 0.63s) but the tail persists (max 24s vs 39s, still 6% >3s). So oversubscription isn't the main cause of the tail.

  The tail is inherent to the GraphSCM generation + filtering work being heavy and variable. Let me directly profile where a single get_batch spends its time and what
  causes the 8-40s outliers — measure SCM vs filter vs pool under concurrency, capturing the slowest calls.

• Ran cd /ossfs/workspace/tabicl && python3 -c "
  │ import numpy as np
  │ # From exp15: median single get_batch ~0.46-0.63s at 64 procs.
  │ … +6 lines
  └ At 64 procs: med=0.63s, p99=6.8s, max=38s
    => a single rank waits up to 6.8s (p99) / 38s (max) for one batch
       each rank holds ~2 prefetched batches => exposed stall up to ~14s

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Now let me determine what makes a single get_batch take 6-38s (vs its 0.46s median). The generate_dataset retry loop (for idx in range(100)) can regenerate up to 100×
  if filters keep rejecting. Under a bad HP draw (e.g., huge feature count + hard graph), each retry is expensive. Let me instrument a single-process run capturing slow
  batches and the retry count.

• Ran cd /ossfs/workspace/tabicl && cat > experiments/exp16_slowbatch.py <<'PY'
  │ """Exp 16: capture what makes individual get_batch calls slow (retry count, seq,
  │ feature count, filter time) at n_jobs=4, 1 process, many calls."""
  │ … +27 lines
  └ slowest 8 get_batch (time_s, scm_calls): [(7.5, 3), (5.22, 1), (5.02, 2), (4.88, 2), (4.71, 1), (3.2, 1), (2.82, 1), (2.81, 2)]
    fastest 8: [(0.102, 1), (0.1, 1), (0.094, 2), (0.09, 1), (0.087, 1), (0.069, 1), (0.055, 1), (0.055, 1)]
    time med=0.344 max=7.50 | scm_calls med=1 max=3
    corr(time, scm_calls)=0.28

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Crucial finding:

  - The slowest batch took 7.5s with only 3 SCM calls; another took 5.22s with 1 SCM call.
  - The fastest batch took 0.055s with 1 SCM call.
  - So the same single GraphSCM call can take 0.05s to 5+ seconds — it's the per-call cost that has the huge variance, not the retry loop.

  The slowness correlates weakly with retries (0.28) but the per-SCM-call time is the dominant variable. Why would one GraphSCM call take 5s when another takes 0.05s?
  The difference must be the sampled graph/HP: large max_features (up to 100), high num_features, large max_n_nodes (32), which makes the causal graph + functions
  expensive. And log-uniform seq_len means sometimes seq≈8192 × 100 features.

  Let me confirm the driver is num_features and seq_len.

• Ran cd /ossfs/workspace/tabicl && cat > experiments/exp17_scaling.py <<'PY'
  │ """Exp 17: which factor drives GraphSCM cost: seq_len, num_features, or n_nodes?
  │ Time cold SCM calls across a grid (single process, n_jobs=1)."""
  │ … +29 lines
  └ === seq_len effect (feat=100, nodes=32) ===
      seq= 1024 : 0.988s
    … +11 lines (ctrl + t to view transcript)
      nodes= 16 : 1.072s
      nodes= 32 : 4.039s

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Now I have the full cost model for GraphSCM (single cold call):

   factor          range        cost range
  ━━━━━━━━━━━━━━  ━━━━━━━━━━━  ━━━━━━━━━━━━━━━
   seq_len         1024→8192    0.99s → 4.15s
  ──────────────  ───────────  ───────────────
   num_features    5→100        0.48s → 2.98s
  ──────────────  ───────────  ───────────────
   n_nodes         2→32         1.57s → 4.04s

  GraphSCM cost is driven mainly by seq_len (up to ~4s at 8192), num_features (up to ~3s at 100), and max_n_nodes (up to ~4s at 32). These are the worst-case combos →
  5-7s single calls, and under 64-way concurrency that's 8-40s waits. Note these cold numbers include some first-call warmup; the medians in-context are lower, but the
  variance is real and driven by these factors.

  Root cause confirmed: the CPU prior generator has a heavy, factor-driven tail (long sequences × many features × large graphs), and DDP couples it across all 16 ranks.

  Now the fix. Options:

  1. Increase DataLoader workers (already 8) — helps throughput but tail persists.
  2. Parallelize within get_batch — currently the per-rank n_jobs=4 inner pool, but batch=1 means only 1 dataset → no inner parallelism to exploit. This is the key
     inefficiency: batch=1 per rank means each rank generates exactly 1 dataset, using only 1 CPU effectively.

  3. Generate on more CPU cores with a shared buffer — decouple generation from consumption.

  The cleanest high-leverage fix: shard the generation across more workers and prefetch deeper. Since each rank generates 1 dataset/step but has 8 workers, the workers
  already prefetch. The problem is the tail.

  Let me test the most effective knobs for the tail: more workers + higher prefetch. Let me measure the DataLoader data-wait under different (workers, prefetch) at the
  real 16-rank scale.

  Actually, the real question: can we shorten the tail itself? Let me test reducing the generator's worst case:

  - max_n_nodes 32→16 (4.0s→1.1s cost)
  - This changes data distribution (model diversity), so it's a semantic change.

  Better: increase workers so the tail is absorbed by prefetch depth. Let me test workers=16/32 with prefetch=6 at 16-rank scale.

• Waited for background terminal · cd /ossfs/workspace/tabicl && cat > experiments/exp18_workers.py <<'PY'
"""Exp 18: does more DataLoader workers/prefetch absorb the CPU tail?
Real 16-rank DDP, shared seq_len, measure the exposed data-wait per step."""
import sys, os, json, time, argparse
import numpy as np, torch, torch.distributed as dist
from types import SimpleNamespace
from torch.utils.data import DataLoader
sys.path.insert(0,"/ossfs/workspace/tabicl/src")
ap=argparse.ArgumentParser()
ap.add_argument("--workers",type=int,required=True); ap.add_argument("--prefetch",type=int,default=4)
ap.add_argument("--steps",type=int,default=200); a=ap.parse_args()
rank=int(os.environ["RANK"]); world=int(os.environ["WORLD_SIZE"]); local=int(os.environ["LOCAL_RANK"])
dist.init_process_group("nccl"); torch.cuda.set_device(local); dev=f"cuda:{local}"
torch.manual_seed(0)
from tabicl._model.tabicl import TabICL
from tabicl.train._optimizer import build_optimizer
from tabicl.prior._dataset import PriorDataset
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.train._screening import StepSeededPrior
from tabicl.prior._genload import seed_worker
from torch.nn.parallel import DistributedDataParallel as DDP
torch.manual_seed(0)
m=TabICL(embed_dim=128,row_num_cls=4,col_num_blocks=1,col_nhead=8,col_num_inds=16,row_num_blocks=1,
  row_nhead=8,icl_num_blocks=12,icl_nhead=8,icl_ssmax=False,icl_attention="dense",icl_qk_norm=True,
  ff_factor=2,norm_first=True).to(dev).to(torch.bfloat16)
cfg=SimpleNamespace(muon=True,lr=4e-4,adamw_lr=1e-4,beta1=0.9,beta2=0.95,weight_decay=0.01,use_cautious_wd=False)
opt=build_optimizer(m,cfg); m=DDP(m,device_ids=[local],broadcast_buffers=False)
scaler=torch.amp.GradScaler("cuda",enabled=True); H=100
os.environ["TABICL_SHARED_SEQ_LEN"]="1"
ds=PriorDataset(regression=False,batch_size=1,batch_size_per_gp=4,min_features=1,max_features=100,max_classes=10,
  min_seq_len=1024,max_seq_len=8193,log_seq_len=True,seq_len_per_gp=True,min_train_size=0.79,max_train_size=0.81,
  prior_type="graph_scm",n_jobs=1,device="cpu",
  config=PriorConfig(filter_unpredictable_graphs=True,filter_unpredictable_datasets=True))
prior=StepSeededPrior(ds,42,42,rank,world,shared_seq_len=True)
dl=DataLoader(prior,batch_size=None,num_workers=a.workers,prefetch_factor=a.prefetch,persistent_workers=True,worker_init_fn=seed_worker)
it=iter(dl)
def fetch():
    x,y,d,seq_lens,tr=next(it); return x.to_padded_tensor(0.0),y.to_padded_tensor(0),int(tr[0])
def sync(): torch.cuda.synchronize()
def step():
    x,y,tr=fetch(); xt=x.to(dev); yt=y.to(dev)
    with torch.amp.autocast("cuda",dtype=torch.bfloat16):
        p=m(xt,yt[:,:tr],None); loss=torch.nn.functional.cross_entropy(p.reshape(-1,10),yt[:,tr:].reshape(-1))
    scaler.scale(loss).backward(); opt.step(); opt.zero_grad(set_to_none=True)
for _ in range(5): step()
sync(); dat=np.zeros(a.steps); tot=np.zeros(a.steps)
for s in range(a.steps):
    t0=time.perf_counter(); x,y,tr=fetch(); t1=time.perf_counter()
    xt=x.to(dev); yt=y.to(dev)
    with torch.amp.autocast("cuda",dtype=torch.bfloat16):
        p=m(xt,yt[:,:tr],None); loss=torch.nn.functional.cross_entropy(p.reshape(-1,10),yt[:,tr:].reshape(-1))
    scaler.scale(loss).backward(); opt.step(); opt.zero_grad(set_to_none=True); sync()
    dat[s]=t1-t0; tot[s]=time.perf_counter()-t0
rt=torch.tensor(np.stack([dat,tot]).T,device=dev)
g=[torch.zeros_like(rt) for _ in range(world)]; dist.all_gather(g,rt)
if rank==0:
    A=torch.stack(g).cpu().numpy()
    d=A[:,:,0]; t=A[:,:,1]; sm=t.max(0)
    print(json.dumps(dict(workers=a.workers,prefetch=a.prefetch,steps=a.steps,
      data_maxrank_med=float(np.median(d.max(0)))*1000,data_maxrank_p95=float(np.percentile(d.max(0),95))*1000,
      step_med=float(np.median(sm))*1000,step_mean=float(sm.mean())*1000,step_p95=float(np.percentile(sm,95))*1000,step_max=float(sm.max())*1000,
      frac_gt1s=float((sm>1).mean()))))
dist.barrier(); dist.destroy_process_group()
PY
export TABICL_ALLOW_ANY_PYTHON=1 PYTHONPATH=src && for W in 8 16 24; do echo "--- workers=$W ---"; timeout 900 python3 -m torch.distributed.run --standalone
--nproc_per_node=16 experiments/exp18_workers.py --workers $W --prefetch 6 --steps 200 2>&1 | grep -E "^\{" || echo FAILED; done

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Both failed — likely OOM (16 or 24 workers × 16 ranks = 256-384 generator processes, each holding memory). Let me check the error.

• Ran cd /ossfs/workspace/tabicl && cat > experiments/exp18_workers.py <<'PY'
  │ """Exp 18: does more DataLoader workers/prefetch absorb the CPU tail?
  │ Real 16-rank DDP, shared seq_len, measure the exposed data-wait per step."""
  │ … +63 lines
  └ --- workers=8 ---
    FAILED
    … +2 lines (ctrl + t to view transcript)
    --- workers=24 ---
    FAILED

• Ran cd /ossfs/workspace/tabicl && export TABICL_ALLOW_ANY_PYTHON=1 PYTHONPATH=src && timeout 400 python3 -m torch.distributed.run --standalone --nproc_per_node=16
  │ experiments/exp18_workers.py --workers 8 --prefetch 6 --steps 30 2>&1 | grep -iE "error|traceback|memory|kill|cannot|fail" | head -15
  └ [rank7]: Traceback (most recent call last):
    [rank7]: ValueError: Tensors must be contiguous
    … +11 lines (ctrl + t to view transcript)
    [rank2]: ValueError: Tensors must be contiguous
    [rank9]: Traceback (most recent call last):

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Just a bug in my exp18 harness (to_padded_tensor contiguity for the gather tensor). Let me fix it quickly.

• Ran cd /ossfs/workspace/tabicl && sed -i 's/rt=torch.tensor(np.stack(\[dat,tot\]).T,device=dev)/
  │ rt=torch.tensor(np.stack([dat,tot],axis=1),device=dev,dtype=torch.float32).contiguous()/' experiments/exp18_workers.py && export TABICL_ALLOW_ANY_PYTHON=1
  │ PYTHONPATH=src && timeout 600 python3 -m torch.distributed.run --standalone --nproc_per_node=16 experiments/exp18_workers.py --workers 8 --prefetch 6 --steps 150
  │ … +1 lines
  └ {"workers": 8, "prefetch": 6, "steps": 150, "data_maxrank_med": 17.04641990363598, "data_maxrank_p95": 183.59322845935822, "step_med": 149.7325301170349,
    "step_mean": 442.31751561164856, "step_p95": 412.4961793422699, "step_max": 20455.120086669922, "frac_gt1s": 0.04}

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Hmm — workers=8/prefetch=6 still spikes (max 20.4s, 4% >1s). But wait, data_maxrank_med is only 17ms, data_maxrank_p95 184ms — the median data wait is tiny. So the
  DataLoader usually keeps up, but occasionally one rank's generation blocks for many seconds.

  Note also step_max=20.4s but step_p95=412ms. So it's a rare, extreme event (4%). Let me test whether more workers actually reduces the frequency, using a valid run.
  Let me also reconsider: given n_jobs=1 in these tests but the real training used n_jobs=4, the real config spawns inner pools which may be the real tail source. Let
  me test the exact real config (n_jobs=4 inner) vs workers.

  Let me make the experiment robust and test workers 8 vs 16 with inner n_jobs=1 (to avoid nested-pool explosion).

• Ran cd /ossfs/workspace/tabicl && export TABICL_ALLOW_ANY_PYTHON=1 PYTHONPATH=src && for W in 16 24; do echo "--- workers=$W prefetch=8 ---"; timeout 700 python3 -m
  │ torch.distributed.run --standalone --nproc_per_node=16 experiments/exp18_workers.py --workers $W --prefetch 8 --steps 150 2>&1 | grep -E "^\{" || echo FAILED;
  │ done
  └ --- workers=16 prefetch=8 ---
    {"workers": 16, "prefetch": 8, "steps": 150, "data_maxrank_med": 26.263412088155746, "data_maxrank_p95": 207.0639580488205, "step_med": 171.36462032794952,
    … +2 lines (ctrl + t to view transcript)
    {"workers": 24, "prefetch": 8, "steps": 150, "data_maxrank_med": 19.577860832214355, "data_maxrank_p95": 241.72866344451904, "step_med": 176.36092007160187,
    "step_mean": 247.9628026485443, "step_p95": 382.44399428367615, "step_max": 4616.509914398193, "frac_gt1s": 0.013333333333333334}

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Progress on workers:

   workers/prefetch    step_med    step_mean    step_max    frac>1s
  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━  ━━━━━━━━━━━  ━━━━━━━━━━  ━━━━━━━━━
   8/6                 150ms       442ms        20.5s       4.0%
  ──────────────────  ──────────  ───────────  ──────────  ─────────
   16/8                171ms       315ms        18.3s       0.7%
  ──────────────────  ──────────  ───────────  ──────────  ─────────
   24/8                176ms       248ms        4.6s        1.3%

  More workers reduces the frequency (4% → 0.7%) and mean (442 → 248ms), but doesn't eliminate the extreme tail (still 4-18s occasionally). This is because a single
  GraphSCM call can genuinely take 5-40s under load, and no amount of prefetch fully hides a 20s stall.

  Conclusion: The CPU generator's inherent heavy tail (driven by seq_len×features×nodes) cannot be fully hidden by more workers. The robust fixes are:

  1. Reduce per-call cost (bound max_n_nodes/features/seq) — semantic change.
  2. Decouple generation via a large shared prefetch buffer / separate generation process pool so a slow batch doesn't block training.
  3. Increase workers + prefetch (partial).

  Let me quantify option 2's potential: if generation were fully decoupled (infinite prefetch), step time would be just the compute (~0.12-0.25s). Let me verify the
  best achievable by testing workers=32/prefetch=16 and also check what limits it. But first, let me confirm the mechanism is single-call cost by measuring with a huge
  prefetch (workers=32, prefetch=16).

• Ran cd /ossfs/workspace/tabicl && export TABICL_ALLOW_ANY_PYTHON=1 PYTHONPATH=src && timeout 700 python3 -m torch.distributed.run --standalone --nproc_per_node=16
  │ experiments/exp18_workers.py --workers 32 --prefetch 12 --steps 150 2>&1 | grep -E "^\{" || echo FAILED
  └ {"workers": 32, "prefetch": 12, "steps": 150, "data_maxrank_med": 14.502263627946377, "data_maxrank_p95": 180.2487075328827, "step_med": 181.1647266149521,
    "step_mean": 217.34857559204102, "step_p95": 356.25016689300537, "step_max": 3224.621534347534, "frac_gt1s": 0.006666666666666667}

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• workers=32/prefetch=12: step_mean=217ms, max 3.2s, 0.7%>1s. Diminishing returns — the tail persists at ~3s even with 512 generator processes.

  Let me now step back and design the definitive fix, then validate. The analysis shows two independent issues both need addressing:

  1. Systematic: DDP couples one rank's slow data → seq_len fix already handles the systematic part.
  2. Stochastic CPU tail: inherent to GraphSCM (5-40s single calls under load).

  The robust engineering fix for #2 is to not let data generation block the training loop's critical path. The cleanest: dedicated persistent generation with a large
  prefetch buffer AND enough workers that the tail is buffered. Since prefetch can't fully hide 20s stalls, the best practical fix is a combination:

  - More workers (16) + higher prefetch → cuts spike rate 4%→0.7%, mean 442→315ms.
  - Plus optionally cap the worst-case generation cost.

  Let me verify the fully-decoupled ceiling and finalize the recommendation. Let me measure the "compute only" floor with the same harness (randn data, real model).

• Ran cd /ossfs/workspace/tabicl && export TABICL_ALLOW_ANY_PYTHON=1 PYTHONPATH=src && timeout 700 python3 -m torch.distributed.run --standalone --nproc_per_node=16
  │ experiments/exp14_matrix.py --seq var --data randn --sync ddp --steps 200 2>&1 | grep -E "^\{" || echo FAILED
  └ {"seq": "var", "data": "randn", "sync": "ddp", "med": 246.62021826952696, "mean": 243.40172110125422, "p95": 256.6134567372501, "mx": 260.805981233716, "sp": 0.0}/
    opt/conda/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `.
    Using the current device set by the user.

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Compute floor confirmed: ~245ms/step, zero jitter when data is free (randn).

  So the achievable target is ~245ms/step if data generation is fully decoupled. Current best (workers=32): 217ms mean but with residual 3s spikes.

  Now I have everything to design the fix. Let me write the final solution: increase workers + prefetch in the staged script, plus document the deeper option (bounded
  generation). Let me test one more promising fix — prefetch the data much earlier via a background thread isn't available in DataLoader, but I can test whether
  persistent_workers + larger prefetch + more workers with the real config (inner n_jobs=1) reaches the floor.

  Let me finalize by testing the recommended config (workers=16, prefetch=8) for more steps to get a stable estimate, and also compute the expected training-time
  savings.

• Ran cd /ossfs/workspace/tabicl && python3 -c "
  │ import numpy as np
  │ # Summary of the jitter investigation
  │ … +13 lines
  └ === Per-step wall time summary (16-rank DDP, real model) ===
      original (workers=4, indep seq_len)    mean=1010ms p95=2.07s  [baseline]
    … +4 lines (ctrl + t to view transcript)

    Spike rate: workers 8 -> 4.0%%   workers 16 -> 0.7%%   workers 32 -> 0.7%%

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Now let me apply the fix (workers=16, prefetch up from 4→8) to the repo, then verify. Let me update the staged script and check where prefetch_factor is set.

• Explored
  └ Search prefetch_factor in _run.py

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Let me make prefetch_factor configurable via env (to allow tuning without touching the recipe), defaulting to the current values.

• Explored
  └ Read _run.py
