# CPU fallback 预算与 sparse 质心路由审计

实验机器为 8 张 96 GiB PPU-ZW810，CPU quota 为 69 核；PyTorch 2.8 vendor build、Triton 3.4。远程仓库 `/ossfs/workspace/tabicl-training-perf`，实验隔离副本 `/tmp/tabicl-cohort-target-20261031`。所有整模实验串行使用 8 张卡；没有把其他训练进程停掉，也没有用算子加速比推算训练加速比。

## CPU 预算的修复

`nearest_center` 的有限值多维路径已经使用共享预算，但 NaN/Inf 会直接返回完整 `cdist(...).argmin()`，绕过借核。这正是部分慢图即使设置 `PRIOR_POOL_KERNEL_THREADS=8` 仍很慢的原因。

现在非有限值 fallback 和超大有限一维 fallback 都进入同一个 `parallel_distance`。仍保留完整 cdist/argmin，不替换 NaN/Inf 的语义；普通有限一维排序路径、p=2 GEMM、小距离计算不额外开线程。借核阈值为 `rows * centers * width >= 10,000,000`。

共享 semaphore 的每个 permit 代表一条生成计算线程：worker 开始生成时取得一个基础 permit；大距离 kernel 非阻塞地借用当前剩余 permit，最多共 8 线程；结束或异常时恢复线程数并归还 permit。没有空闲 permit 就单线程执行，不等待凑齐 8 线程。worker 的基础 permit 本身也计入预算。

本机预算为 `69 - 8 local ranks = 61`。默认 56 workers 全部繁忙时只剩 5 个额外 permit，因此一个 kernel 最多能拿到 6 线程；其他 worker 闲下来后才可能拿到 8。这里的“空闲”是预算中尚未占用的 permit，不是对 OS 实际空闲核的保证。训练 host 线程、驱动和 RPC 的保留量是启发式，不是整台机器的硬 CPU 配额。

脚本共享 recipe、Stage2/3 显式默认以及训练 CLI 的 node_pool 默认均改为 56。`PRIOR_POOL_WORKERS=0` 仍可恢复按 quota/affinity 减 local ranks 的自动选择。16 张卡位于同一节点时，本机同样 quota 下预算降为 53，56 个 worker 中最多 53 个同时取得基础 permit；这里没有声称测过 16 张实体卡性能。Stage1 BF16 原来的 DataLoader 路线保持其默认 worker 配置。

### 固定慢图复现

单位为秒；生成时间不含 worker 启动，CPU 核秒包括该 worker 的计算线程。两条路线的输出 X/y SHA256 完全相同。

| task | 行数 / 请求特征 | 单线程生成 | 修复后最多 8 线程生成 | 单线程 CPU 核秒 | 借核 CPU 核秒 |
|---|---:|---:|---:|---:|---:|
| 106 | 21,115 / 6 | 30.798 | 7.338 | 30.667 | 31.726 |
| 29 | 35,022 / 32 | 49.044 | 14.362 | 48.823 | 48.971 |

task106 在修复前虽然已开启 kernel_threads=8，仍需约 24.86 秒；其中两次大型非有限 fallback 合计约 19.88 秒。修复后对应调用约 1.292、1.079 秒。task29 是对照：修复前借核约 14.37 秒，修复后 14.36 秒。收益主要在漏掉的长尾分支，不是全图都突然变快。

task106 哈希：`4c9c0c2b2110da8fc64ddda4a769ffcc1223a975c03d4a6fc0435ac8b5363b43`。
task29 哈希：`f4ec7a22cd7afbeb78c64cfa7968d22eaf1e874636e3741899fee25c42e87c95`。

### 56 workers 真实生成 512 张表

| Stage | 含启动总 wall | 单表 P50 | P95 | P99 | 最慢表 | 总 CPU 核秒 | 图生成占比 | 过滤占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 16.608 | 0.478 | 2.927 | 5.865 | 7.393 | 484.488 | 79.49% | 19.75% |
| 3 | 28.312 | 0.690 | 6.179 | 15.071 | 21.090 | 993.437 | 80.16% | 19.23% |

Stage3 修复前同样 56 workers 的 512 表 wall 约 32.621 秒、最慢 26.54 秒。该 CPU-only 对照不能直接解释为训练提速 13%；训练还取决于预取覆盖、GPU 消费速度和 DDP 关键路径。

512 表恰好对应 8 个 global batch=64 的 step。用总核秒除以 `8*61`，Stage2/3 的理想供给下界分别约 0.993、2.036 秒/step；这是假定所有 permit 完全用于有效生成、没有尾部或排队的下界，不是实测 stall。Stage3 已接近 GPU 消费的量级。若仍是同一节点、同样 CPU 资源和 global batch，却增加至 16 GPU，GPU 消费变快不会使每步需要生成的 64 张表减少；也不能仅靠重新平衡 rank 分配消除这部分 CPU 需求。

## 每步计时口径

node_pool 为每张表记录提交、实际开始生成、生成完成、编码完成以及 `process_time()`，通过现有 get_batch RPC 返回本 rank 汇总，不额外增加 RPC。

- `cpu_generation_core_s`：一个全局 optimizer step 的所有表消耗的 CPU 核秒之和，不能当作墙钟秒。
- `cpu_generation_span_s`：这批表中最早开始到最后生成完成的跨度，可以发生在之前几个 step 的预取期间。
- `cpu_submit_to_ready_s`：包含队列等待的提交到生成完成跨度。
- `generation_wait_rank_max_s`：最慢 rank 从请求本 step 到数据生成依赖完成的暴露等待；balanced 必须等整个 cohort 完成，不能只看该 rank 自己的表。
- `generation_wait_all_ranks_s`：所有 rank 都已结束上一轮、又都缺少下一轮生成数据的共同空闲前缀，是保守的 CPU 供给 stall 下界。它不把 RPC、解码和训练 host 开销算成图生成。
- GPU compute / communication / activity：Kineto 中每个 optimizer step、每个 rank 的计算 kernel、通信 kernel、全部设备活动的时间区间并集，最后取 rank 最大值。通信可能和计算重叠，三个数不能相加。kernel 运行时间也可能包含硬件内部等待。
- `wait_train_max_s`：rank 的取数据与训练 wall 之和取最大值。CUDA event 的 stream 时间包含提交空洞和 DDP 尾部，不能冒充纯 GPU 计算时间。

CPU 生成和 GPU 执行存在重叠，不能用 `CPU + GPU = step`。完整逐步数据由 `summarize_pipeline_profile.py --csv-output` 导出；本次四组实验的原始 110 行逐步数据已随仓库保存于 [`cpu_budget_topk_20261031_steps.csv`](performance_results/cpu_budget_topk_20261031_steps.csv)。全 rank trace 有额外开销，整模吞吐必须另用不带 trace 的训练对照。

CSV 的 `step` 为零起算的 optimizer step；例如 `step=10` 是第 11 次参数更新。

## Sparse 路由审计与实现

现有 `content_tree` 先按内容构造 support 分区，再算叶子质心。centroid router 对所有叶子计算精确 FP32 IEEE 点积，选 4 个 block；默认每个 block 128 个 support token。`global_sparse` 还保留覆盖全部 support 的 ELU 全局分支。这里没有减少候选叶子、改变采样分布、改为近似 beam search，或改变 QK norm 可见 token 数。

旧路径每个 query chunk 产生完整 scores，调用框架 topk，保存 int64 indices 列表，最后 cat 并转 int32。新路径保留原 score kernel，Triton 在每行执行最多 8 次并行最大值选择，把候选按照框架 gatherTopK 的顺序排列，只对小 K 使用框架 sort，最后直接写入连续 int32 输出。NaN/Inf、阈值并列和最终排序顺序都在测试里逐索引比对。

scores、候选 values/IDs、排序 values/positions 使用可复用 workspace；32 MiB 上限现在把这些辅助缓冲一起计入，最后的输出单独计。最低粒度为每个 batch/head 一行；极端 batch/head 大于上限的情况仍需这个最小单元。大于 1024 个叶子或 K>8 使用原路径。checkpoint 的离散 route 仍然记录后重放，不在反向重新选路。

并列值的索引顺序对齐的是本次验证的框架/设备实现。框架 topk 对相等元素没有跨版本稳定索引契约；升级 PyTorch 或更换设备后，应重新运行 `tests/test_centroid_topk.py`，不能据此宣称所有未来后端都保证并列索引相同。`use_fast_centroid_topk=False` 保留完整参考路径，便于验证与对照。

一个被否决的原型曾用 `nonzero()` 压缩异常行，再回退框架 topk。尽管算子微基准变快，它引入了逐 chunk 的 D2H 同步。8 卡 Stage2 的 trace 对照从 1.667 秒/step 回退到 2.656 秒，而计算 kernel 约 0.657→0.651 秒，通信活动约 0.423→1.171 秒。最终实现去掉了该同步点；不能把那个原型的微基准当作可交付训练收益。

grouped sparse backward 的反向路由使用原子插入，累加顺序本来就可能变化。测试要求 route IDs 和前向逐位相同；FP32 梯度保留严格小误差，BF16 梯度允许原有归约顺序造成的舍入差异。这里没有修改 sparse attention 的前后向数学运算。

还需区分路由 workspace 和 attention workspace：`_sparse_grouped.grouped_attention` 仍有 `[B,H,N,K,D]` 的 FP32 partial，再按 query 做 logsumexp merge。`B=1,H=8,N=60000,K=4,D=64` 时单个 partial 就是 468.75 MiB，远大于本次限制为 32 MiB 的路由 workspace。它用于按叶子分组的 tensor-core 计算，不能简单删除，也不能把修改它的收益算到本次 top-k 优化中。下一轮 GPU 内存/带宽优化应单独评估 partial 写回与 merge；改变 partial 精度或用原子归约需要重新验证前向、梯度与可重现性。

## 复现实验

以下命令在仓库根目录、已安装依赖的训练环境执行。整模对照必须串行运行，不能同时在同一节点执行 CPU 生成或其他 GPU 压测。

```bash
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PRIOR_BACKEND=node_pool PRIOR_POOL_WORKERS=56
export PRIOR_POOL_KERNEL_THREADS=8 PRIOR_SCHEDULE=balanced
export COMPILE_FRONTEND=all COMPILE_WARMUP=True RECOMPUTE=True
export RANK_TIMING_EVERY=1 ICL_DIAGNOSTICS_EVERY=0

# 无 trace 的整模吞吐；参考路径再加 --reference-centroid-topk。
python scripts/benchmark_training_schedule.py \
  --stage 3 --dtype bfloat16 --gpus 8 --steps 80 \
  --output /tmp/stage3-fast \
  --train-args --np_seed 42 --fail_fast True
python scripts/summarize_training_schedule.py /tmp/stage3-fast --skip 10

# 独立做逐步 CPU/GPU 分解。前 10 步不记录 trace。
python scripts/benchmark_training_schedule.py \
  --stage 3 --dtype bfloat16 --gpus 8 --steps 40 \
  --output /tmp/stage3-profile --profile-pipeline \
  --trace-step 10 --trace-steps 30 --trace-all-ranks \
  --train-args --np_seed 42 --fail_fast True
for rank in 0 1 2 3 4 5 6 7; do
  python scripts/summarize_kineto_trace.py \
    "/tmp/stage3-profile/profile/rank${rank}-trace.json" \
    --output "/tmp/stage3-profile/profile/rank${rank}-trace-summary.json"
done
python scripts/summarize_pipeline_profile.py /tmp/stage3-profile/profile \
  --skip 10 --expected-ranks 8 --output /tmp/stage3-profile/summary.json \
  --csv-output /tmp/stage3-profile/steps.csv

# 单独验证质心路由延迟、峰值分配、索引一致性。
python scripts/benchmark_centroid_topk.py \
  --output /tmp/centroid-topk.json --repeats 20

# 真实 CPU-only 生成；不要和上述整模测量同时执行。
python scripts/profile_graph_generation.py --stage 3 --tasks 512 \
  --workers 56 --pool-kernel-threads 8 --cpu-reserve 8 \
  --output /tmp/stage3-cpu-512.json
```

取消裁列时在整模命令的 `--train-args` 后增加 `--adaptive_feature_limit False`。固定 60k×20 的 GPU 消费对照可使用 `--shape-prior`，并在训练参数增加 `--min_seq_len 60000 --max_seq_len 60000 --log_seq_len False --min_features 20 --max_features 20`；该组合本轮未成功运行，因 node_pool prior 在该固定范围下拒绝采样（`low >= high`）。它只是后续排除真实图生成的诊断建议，不是已完成实验，也不代表真实 prior 的训练吞吐。

## 最终逐步 profile 结果

以下为同一远程节点、8 卡、trace 从第 10 步开始的 30 个 optimizer step；trace 本身有额外开销，数值用于拆分瓶颈，不作为无 trace 吞吐承诺。`GPU activity` 是设备活动区间并集，`GPU kernel` 是 kernel 区间并集；二者均取最慢 rank，通信与计算可能重叠。

| 场景 | step wall 均值 | GPU activity | GPU kernel | CPU 生成核秒 | CPU 共同 stall | CPU 生成 wall span |
|---|---:|---:|---:|---:|---:|---:|
| Stage2 BF16，裁列，fast | 1.570 s | 0.902 s | 0.651 s | 54.4 / step | 0 | 5.62 s |
| Stage3 BF16，裁列，reference | 4.359 s | 2.030 s | 1.735 s | 158.2 / step | 1.714 s | 25.13 s |
| Stage3 BF16，裁列，fast | 4.235 s | 2.189 s | 1.591 s | 176.1 / step | 1.546 s | 28.40 s |
| Stage3 BF16，不裁列，fast | 5.543 s | 2.591 s | 1.911 s | 210.4 / step | 2.400 s | 33.72 s |

逐步分布比均值更重要：Stage3 裁列 fast 的 step wall P50/P95/max 为 2.745/12.124/20.684 s，CPU stall P50/P95/max 为 0/9.784/18.522 s；这正是 rank barrier 下长尾表决定 step 的证据。Stage3 不裁列不仅 GPU activity 增加，CPU 核秒和共同 stall 也分别增加约 19% 和 55%，因此主要问题仍是 CPU 图供给长尾，而不是 centroid 路由本身。

无 trace 的 80-step 对照中，Stage3 BF16 seed42 fast/reference 为 3.431/3.437 s/step；seed43 fast/reference 为 3.191/3.564 s/step。Stage2 BF16 fast/reference 为 1.346/1.360 s/step。Stage2 FP32 fast/reference 为 1.618/1.571 s/step，差异方向不稳定，不能归因于路由优化。路由 microbenchmark（24 cases、20 repeats）在 60k rows/375 leaves 上约 9.44 ms vs 33.45 ms（BF16），在 2048 rows/128 leaves 上约 0.224 ms vs 0.427 ms；实际训练收益被 CPU 长尾、预取和其他 ICL kernel 稀释。

本轮针对性测试为 `202 passed, 2 warnings`（centroid top-k、CPU budget、node pool、pipeline profile、prior pipeline）。完整仓库套件曾在远程长时运行中输出失败标记但未在控制器超时前保留 traceback，因此不将其宣称为全量通过；失败定位不影响上述 202 个相关测试的结果，提交前应在有更长作业时限的环境补跑全量套件。
