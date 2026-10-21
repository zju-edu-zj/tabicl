# Stage-1 ICL mechanism diagnosis

This is a read-only post-hoc evaluation for checkpoints produced by
`scripts/train_v2_clf_stage1.sh`. No attention screening run is required. The
script does not alter checkpoint weights, optimizer state, or the training
recipe.

Evaluation uses the pretraining forward path with gradients disabled, matching
the held-out objective in `tabicl.train._screening`. This keeps all
`max_classes` logits even when a query class is absent from the support. The
deployment `eval()` path truncates classification outputs based on support
classes and cannot be used with the original pretraining labels. Labels are
validated on CPU before any CUDA label-indexing kernel; output dimensions are
checked before cross-entropy. Model modes are restored after evaluation.

## Direct use on a Stage-1 directory

```bash
cd /home/zj/table_foundation/sparse/tabicl
python scripts/diagnose_icl_checkpoint.py --dtype bf16
```

The checkpoint directory defaults to
`<repository>/checkpoints/tabiclv2-clf/stage1`, and the report defaults to
`<checkpoint directory>/icl-mechanism.json`. 

With no cases argument, a deterministic Graph-SCM suite is generated beside
the output as `icl-mechanism.cases.pt` and then reused. Defaults cover support
sizes 384, 640, 896, and 4096. The first three probe the Stage-1 distribution:
384 is below the 512-key sparse budget, while 640 and 896 activate retrieval.
4096 probes length extrapolation before Stage 2.

For a checkpoint directory, the default evaluates every 20,000 steps plus the
first and last available checkpoints. All results go into the one output JSON.
Use exact milestones for a cheaper first pass:

```bash
  --steps 5000 25000 50000 100000 200000 300000 400000 500000
```

Use `--checkpoint-every 0` only when every saved 5,000-step checkpoint is
really needed. A dense sweep is expensive because every checkpoint runs every
case under six interventions. A sensible workflow is a sparse milestone sweep,
then an exact `--steps` sweep around intervals where the diagnostics change.
The output is rewritten atomically after each completed checkpoint, preserving
partial progress if a later checkpoint fails. Reusing the command or adding new
`--steps` resumes a compatible report by default and skips finished checkpoints;
pass `--no-resume` to recompute it from scratch.

Reports created before the pretraining-forward fix have incompatible evaluation
semantics. Recompute those with `--no-resume`; their saved `.cases.pt` can still
be reused. After a CUDA device-side assert, launch a fresh Python process before
rerunning the corrected script.

## Interventions

The diagnostic evaluates six policies on exactly the same cases:

* `native`: the checkpoint's actual global+sparse architecture and learned
  fusion gate.
* `global_only`: remove the sparse contribution in retrieval layers.
* `sparse_only`: remove the global contribution in retrieval layers.
* `random_candidates`: keep the block budget and partition, but uniformly
  sample candidate blocks instead of using the query-dependent router.
* `oracle_candidates`: choose the same block budget using exact dense-softmax
  mass. This is an upper bound for candidate selection, not a deployable router.
* `dense_only`: exact full-support softmax in retrieval layers. This is a
  compute/quality reference, not a fallback path used by the trained model.

Other layers and all learned weights remain unchanged. Comparisons therefore
answer distinct questions: branch ablations test whether both branches carry
useful information; random versus native tests whether routing is better than
chance; native versus oracle tests how much room remains in the router at the
same compute budget; dense-only estimates the remaining loss from limited
retrieval capacity.

## Recorded statistics

For every checkpoint, case, support length, policy, and retrieval layer, the
report stores task cross-entropy, accuracy, confidence, prediction RMS and
non-finite rate. Router diagnostics include top-k key recall, retained dense
attention mass, candidate fraction, visible-key range, and sparse-versus-dense
attention-output relative error. Recall, retained mass, output error, and native
fusion-gate distribution are additionally retained per head, so a good layer
average cannot hide inactive or failed heads. Router samples and per-head gates
use query rows. Global diagnostics include ELU denominator,
feature range and output RMS (or induced-slot statistics). Native fusion also
records its mean sparse gate.

The native model is rerun after jointly permuting support rows and labels. The
maximum/mean logit change and prediction-flip fraction detect violations of the
intended set semantics. Summary entries are grouped by checkpoint, support and
policy and report mean, standard deviation, median and p90 across cases.

Run the same cases again with `--dtype fp32` for a numerical precision audit.
Use `--no-permutation` to omit the invariance rerun. Oracle routing computes
dense scores in small query chunks and is intentionally slower; it can be
excluded with `--policies ...` for very large contexts.
