"""Plan/run controlled 20K–50K-step ICL screening in the current Python environment.

Default: 20K steps, log-uniform 1K–8K screening of two global memories and
their same-layer sparse residuals, plus dense and a full-retrieval diagnostic.
The stage1 preset retains the original fixed-1024 data distribution.
The explicit pilot preset uses a small model and is NOT a pretraining ranking.
"""

import argparse
import hashlib
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from tabicl._model.tabicl import TabICL
from tabicl.prior._dataset import PriorDataset
from tabicl.prior.graph_lib._config import PriorConfig
from tabicl.train._train_config import build_parser

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "dense": ("dense", "elu", "projection"),
    "global_elu": ("global", "elu", "projection"),
    "global_induced": ("global", "induced", "projection"),
    "global_full_elu": ("global_full", "elu", "projection"),
    "global_sparse_elu": ("global_sparse", "elu", "projection"),
    "global_sparse_induced": ("global_sparse", "induced", "projection"),
    # Optional follow-ups. They are excluded from the default causal screen.
    "global_sparse_alternating": ("global_sparse_alternating", "elu", "projection"),
    "global_sparse_content_tree": ("global_sparse", "elu", "content_tree"),
}
DEFAULT_VARIANTS = (
    "dense",
    "global_elu",
    "global_induced",
    "global_full_elu",
    "global_sparse_elu",
    "global_sparse_induced",
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "prepare", "run", "summarize"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preset", choices=("stage1", "routing", "pilot"), default="routing")
    p.add_argument("--steps", type=int)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(DEFAULT_VARIANTS))
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument(
        "--model-compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compile formal training by default; use --no-model-compile for eager/DDP diagnosis.",
    )
    p.add_argument("--batch-size", type=int)
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--eval-support", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    p.add_argument("--eval-cases", type=int, default=16, help="Independent tables per support panel.")
    p.add_argument("--eval-queries", type=int, default=128)
    p.add_argument("--eval-features", type=int, nargs="+", default=[5, 20, 50])
    p.add_argument("--eval-seed", type=int, default=100003)
    p.add_argument("--task", choices=("classification", "regression"), default="classification")
    p.add_argument(
        "--memory-limit-gib",
        type=float,
        default=8,
        help="Pilot worker allocator cap; formal runs use this cap too (0 disables).",
    )
    args = p.parse_args()
    args.output = args.output.resolve()
    args.steps = args.steps if args.steps is not None else (200 if args.preset == "pilot" else 20000)
    args.batch_size = (
        args.batch_size
        if args.batch_size is not None
        else (4 if args.preset == "pilot" else 16 if args.preset == "routing" else 64)
    )
    args.eval_every = args.eval_every if args.eval_every is not None else (100 if args.preset == "pilot" else 2000)
    if args.preset != "pilot" and not 20000 <= args.steps <= 50000:
        p.error("Formal screening requires 20000–50000 steps; use --preset pilot for a short functional run")
    if (
        min(
            args.steps,
            args.batch_size,
            args.micro_batch_size,
            args.workers,
            args.eval_every,
            args.eval_cases,
            args.eval_queries,
            args.num_gpus,
            *args.eval_support,
            *args.eval_features,
        )
        < 1
    ):
        p.error("Counts must be positive")
    if args.batch_size % args.num_gpus or (args.batch_size // args.num_gpus) % args.micro_batch_size:
        p.error("Global batch must divide evenly across GPUs and microbatches")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.variants)) != len(args.variants):
        p.error("Seeds and variants must be unique")
    if args.memory_limit_gib < 0:
        p.error("--memory-limit-gib must be nonnegative; 0 disables the cap")
    if not str(Path(sys.executable).resolve()).startswith("/home/zj/miniconda3/envs/stdplm/"):
        p.error("Run Python from the stdplm conda environment")
    return args


def train_options(args, variant, seed):
    attention, global_type, partition = VARIANTS[variant]
    job = args.output / f"seed_{seed}" / variant
    pilot = args.preset == "pilot"
    options = dict(
        device="cuda",
        dtype="bfloat16",
        amp=True,
        np_seed=seed,
        torch_seed=seed,
        max_steps=args.steps,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        lr=8e-4,
        muon=True,
        beta1=0.9,
        weight_decay=0.01,
        use_cautious_wd=False,
        scheduler="cosine_with_restarts",
        warmup_proportion=0.01,
        cosine_num_cycles=1,
        cosine_amplitude_decay=1,
        cosine_lr_end=1e-7,
        gradient_clipping=10,
        prior_type="graph_scm",
        prior_device="cpu",
        n_jobs=args.workers,
        batch_size_per_gp=4,
        prior_seed_by_step=True,
        min_features=1,
        max_features=8 if pilot else 100,
        max_classes=10,
        max_seq_len=8193 if args.preset == "routing" else 1024,
        min_train_size=0.79 if args.preset == "routing" else 0.3,
        max_train_size=0.81 if args.preset == "routing" else 0.9,
        seq_len_per_gp=True,
        graph_noise=False,
        filter_unpredictable_graphs=True,
        filter_unpredictable_datasets=True,
        allow_act_warping=False,
        min_n_nodes=2,
        max_n_nodes=32,
        cauchy_dag_offset=0,
        embed_dim=32 if pilot else 128,
        col_num_blocks=1 if pilot else 3,
        col_nhead=8,
        col_num_inds=16 if pilot else 128,
        col_affine=False,
        col_feature_group="same",
        col_feature_group_size=3,
        col_target_aware=True,
        col_ssmax=True,
        col_ssmax_type="qassmax-mlp-elementwise",
        row_num_blocks=1 if pilot else 3,
        row_nhead=8,
        row_num_cls=4,
        row_rope_base=100000,
        row_rope_interleaved=False,
        icl_num_blocks=3 if pilot else 12,
        icl_nhead=8,
        icl_attention=attention,
        icl_global_type=global_type,
        icl_num_induced_slots=128,
        icl_partition=partition,
        icl_router="centroid",
        icl_sparse_every=3,
        icl_block_size=128,
        icl_num_selected_blocks=4,
        icl_ssmax=False,
        icl_qk_norm=True,
        ff_factor=2,
        norm_first=True,
        zero_init=False,
        use_flash_attn3=False,
        model_compile=(not pilot if args.model_compile is None else args.model_compile),
        recompute=not pilot,
        fail_fast=True,
        icl_diagnostics_every=200,
        icl_diagnostic_queries=16,
        icl_diagnostic_topk=32,
        wandb_log=False,
        tensorboard_log=False,
        log_dir=job / "logs",
        checkpoint_dir=job,
        save_temp_every=args.eval_every,
        save_perm_every=5000,
        max_checkpoints=2,
        initial_weights=args.output / f"initial_seed_{seed}.pt",
        eval_data=args.output / "validation.pt",
        eval_every=args.eval_every,
    )
    if args.preset == "routing":
        options.update(min_seq_len=1024, log_seq_len=True)
    if args.task == "regression":
        options.update(regression_method="quantile", num_quantiles=999, norm_type="layernorm_nobias")
    return {k: str(v) for k, v in options.items()}


def cli_options(options):
    return [v for key, value in options.items() for v in ("--" + key, value)]


def initialization_config(options):
    c = build_parser().parse_args(cli_options(options))
    keys = (
        "num_quantiles",
        "embed_dim",
        "col_num_blocks",
        "col_nhead",
        "col_num_inds",
        "col_affine",
        "col_feature_group",
        "col_feature_group_size",
        "col_target_aware",
        "row_num_blocks",
        "row_nhead",
        "row_num_cls",
        "row_rope_base",
        "row_rope_interleaved",
        "icl_num_blocks",
        "icl_nhead",
        "ff_factor",
        "dropout",
        "activation",
        "norm_first",
        "zero_init",
        "recompute",
    )
    config = {key: getattr(c, key) for key in keys}
    config.update(
        max_classes=10 if c.regression_method is None else 0,
        bias_free_ln=c.norm_type == "layernorm_nobias",
        col_ssmax=c.col_ssmax_type if c.col_ssmax else False,
        icl_ssmax=False,
        icl_qk_norm=c.icl_qk_norm,
        icl_attention="dense",
        icl_attention_impl_version=4,
        icl_sparse_every=3,
        icl_block_size=128,
        icl_num_selected_blocks=4,
        icl_query_chunk_size=64,
        icl_router="centroid",
        icl_router_beam=16,
        icl_partition="projection",
        icl_global_type="elu",
        icl_num_induced_slots=128,
    )
    return config


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    val_path = args.output / "validation.pt"
    meta = dict(
        support=args.eval_support,
        cases=args.eval_cases,
        queries=args.eval_queries,
        features=args.eval_features,
        seed=args.eval_seed,
        task=args.task,
        prior="graph_scm",
    )
    if val_path.exists():
        if torch.load(val_path, map_location="cpu", weights_only=True)["meta"] != meta:
            raise ValueError("Existing validation configuration differs; use a separate output directory")
    else:
        cases = []
        for support in args.eval_support:
            for index in range(args.eval_cases):
                seed = args.eval_seed + support * 1009 + index
                np.random.seed(seed % 2**32)
                random.seed(seed)
                torch.manual_seed(seed)
                features = args.eval_features[index % len(args.eval_features)]
                prior = PriorDataset(
                    regression=args.task == "regression",
                    batch_size=1,
                    batch_size_per_gp=1,
                    min_features=features,
                    max_features=features,
                    max_classes=10,
                    max_seq_len=support + args.eval_queries,
                    min_train_size=support,
                    max_train_size=support + 1,
                    prior_type="graph_scm",
                    n_jobs=1,
                    config=PriorConfig(filter_unpredictable_graphs=True, filter_unpredictable_datasets=True),
                )
                x, y, _, _, sizes = prior.get_batch()
                assert int(sizes[0]) == support
                cases.append(
                    dict(id=f"support_{support}/case_{index}/seed_{seed}", x=x.cpu(), y=y.cpu(), train_size=support)
                )
                print(f"Prepared validation support={support}, case={index}", flush=True)
        torch.save(dict(meta=meta, cases=cases), val_path)
    for seed in args.seeds:
        options = train_options(args, "dense", seed)
        config = initialization_config(options)
        path = args.output / f"initial_seed_{seed}.pt"
        if path.exists():
            if torch.load(path, weights_only=True, map_location="cpu")["config"] != config:
                raise ValueError("Existing initialization configuration differs")
        else:
            torch.manual_seed(seed)
            model = TabICL(**config)
            torch.save(dict(purpose="icl_screen_initialization", config=config, state_dict=model.state_dict()), path)


def summarize(args):
    rows = []
    for seed in args.seeds:
        by_variant = {}
        for variant in args.variants:
            path = args.output / f"seed_{seed}" / variant / "logs/validation.jsonl"
            if path.exists():
                items = [json.loads(line) for line in path.read_text().splitlines()]
                # Never compare checkpoints trained for different step budgets.
                found = [x for x in items if x["step"] == args.steps]
                if found:
                    by_variant[variant] = {c["id"]: c for c in found[-1]["cases"]}
        dense = by_variant.get("dense", {})
        for variant, cases in by_variant.items():
            for support in args.eval_support:
                panel = [c for c in cases.values() if c["support"] == support]
                if len(panel) != args.eval_cases:
                    raise ValueError(f"Incomplete validation panel: {variant}, support={support}")
                objective = "ce" if args.task == "classification" else "pinball"
                row = dict(
                    seed=seed,
                    variant=variant,
                    step=args.steps,
                    support=support,
                    tables=len(panel),
                    **{objective: float(np.mean([c[objective] for c in panel]))},
                )
                if args.task == "classification":
                    row["accuracy"] = float(np.mean([c["accuracy"] for c in panel]))
                if dense:
                    delta = np.array([c[objective] - dense[c["id"]][objective] for c in panel])
                    rng = np.random.default_rng(123)
                    boot = delta[rng.integers(0, len(delta), (2000, len(delta)))].mean(1)
                    row.update(
                        delta_vs_dense=float(delta.mean()),
                        paired_table_bootstrap_ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                    )
                rows.append(row)
    aggregates = []
    objective = "ce" if args.task == "classification" else "pinball"
    for variant in args.variants:
        for support in args.eval_support:
            selected = [r for r in rows if r["variant"] == variant and r["support"] == support]
            if not selected:
                continue
            values = [r[objective] for r in selected]
            item = dict(
                variant=variant,
                support=support,
                completed_seeds=len(selected),
                requested_seeds=len(args.seeds),
                **{objective: float(np.mean(values))},
                objective_std_across_seeds=float(np.std(values, ddof=1)) if len(values) > 1 else None,
            )
            if all("delta_vs_dense" in r for r in selected):
                item["mean_delta_vs_dense"] = float(np.mean([r["delta_vs_dense"] for r in selected]))
            if args.task == "classification":
                item["accuracy"] = float(np.mean([r["accuracy"] for r in selected]))
            aggregates.append(item)
    output = dict(
        preset=args.preset,
        task=args.task,
        steps=args.steps,
        results=rows,
        seed_aggregate=aggregates,
        caveat=(
            "Pilot is a functional/learning check, not a convergence ranking. "
            if args.preset == "pilot"
            else "Screening does not replace full pretraining and downstream evaluation. "
        )
        + "CI resamples paired tables within each seed; seed std is reported separately.",
    )
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


def main():
    args = arguments()
    torch.set_num_threads(4)
    commands = []
    for seed in args.seeds:
        for variant in args.variants:
            options = train_options(args, variant, seed)
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={args.num_gpus}",
                "-m",
                "tabicl.train",
                *cli_options(options),
            ]
            build_parser().parse_args(cli_options(options))  # Validate even a dry plan.
            commands.append((seed, variant, options, command))
    if args.action == "plan":
        for seed, variant, _, command in commands:
            print(f"# {args.preset} / seed={seed} / {variant}")
            print(shlex.join(command))
        return
    if args.action == "summarize":
        summarize(args)
        return
    prepare(args)
    if args.action == "prepare":
        return
    env = dict(
        os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), OMP_NUM_THREADS="4"
    )
    # Applied in each trainer process, not to unrelated GPU processes.
    env["TABICL_CUDA_MEMORY_LIMIT_GIB"] = str(args.memory_limit_gib)
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
    source_hash = hashlib.sha256()
    for path in sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("scripts/*.sh")]):
        source_hash.update(str(path.relative_to(ROOT)).encode())
        source_hash.update(path.read_bytes())
    for seed, variant, options, command in commands:
        job = Path(options["checkpoint_dir"])
        job.mkdir(parents=True, exist_ok=True)
        manifest = dict(
            options=options,
            num_gpus=args.num_gpus,
            eval_sha256=sha256(options["eval_data"]),
            initial_sha256=sha256(options["initial_weights"]),
        )
        recipe = job / "recipe.json"
        if recipe.exists() and json.loads(recipe.read_text()) != manifest:
            raise ValueError(f"Run recipe changed: {recipe}; use a new output directory")
        recipe.write_text(json.dumps(manifest, indent=2) + "\n")
        with (job / "invocations.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        command=command,
                        git_head=git_head,
                        source_sha256=source_hash.hexdigest(),
                        tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
                    )
                )
                + "\n"
            )
        print(f"Running {variant}, seed={seed}; logs: {job / 'console.log'}", flush=True)
        with (job / "console.log").open("a", buffering=1) as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    summarize(args)


if __name__ == "__main__":
    main()
