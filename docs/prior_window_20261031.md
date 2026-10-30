# 节点供给与有限窗口调度审计（2026-10-31）

本次代码在独立 worktree `/ossfs/workspace/tabicl-training-perf` 中验证。
实验节点为 8 × 96 GiB PPU-ZW810，vendor PyTorch 2.8、Triton 3.4；
CPU quota 约 69.76 核。16 rank 检查使用 CPU/Gloo，不是 16 张物理 GPU 的吞吐测试。
实验使用独立 `/tmp/tabicl-perf-venv`：Python 3.12.11、Torch `2.8.0+ppu1.7.0.oe`、
Triton `3.4.0+ppu1.7.0.oe`、NumPy 2.5.4、SciPy 1.16.3、scikit-learn 1.9.1、
threadpoolctl 3.7.0、pytest 7.4.3。Torch/Triton 为节点自带 vendor 构建。

本次确认了两类可修复成本：数组 RPC 的主机拷贝，以及大规模非整数 p 距离计算的
单核长尾。后者已有真实 8 卡 Stage2/3 短训收益；mmap 在纯数据面有明显收益，
但窗口分桶尚未证明 8 卡训练更快。各项保持显式可选，未以预测收益替换默认策略。

## 先明确两个不同的问题

节点共享 worker 可以减少各卡独立生成造成的资源碎片，但不会消除图生成、
可预测性过滤、rejection loop 或固定 step 的长尾。旧 `none` 和
DataLoader `in_order=False` 都会消费完成顺序前缀；固定 cohort 则必须包含指定的慢表。
不同随机前缀、不同 worker 数的训练对比不能单独证明共享池实现更慢。

新的受控比较分别隔离真实图生成、输出传输和 GPU 调度。所有性能数字都必须区分
包含启动的进程 wall time、稳态 elapsed/step、取数等待和含 DDP 同步的训练时间。
某卡提前拿到数据后，在反向同步中等待另一卡，会把 prior time 转移到 train time。
不能把 prior time 下降直接解释为整步加速，也不能将不同 rank 的 prior/train 最大值相加。

## 架构与保证

```mermaid
flowchart LR
    A[原 prior 参数采样层次] --> B[TaskPlanner：固定 task ID 和 seed]
    B --> C[有限窗口：按 GPU 成本组完整 step]
    C --> D[步内：等数量分配完整 micro-batch]
    D --> E[生成服务：输出字节准入、CPU worker、结果归属]
    E --> F[数据面：NumPy RPC 或 mmap 描述符]
    F --> G[rank 映射并确认，micro-batch padding，GPU 训练]
```

`_task_plan.py` 只管理参数和身份，不生成表、不做 IPC。
`_scheduling.py` 提供成本函数与等数量分配；`_node_pool.py` 负责生成生命周期、
准入和 `(local_rank, step_sequence)` 交付；`_pool_transport.py` 独立管理输出存储和映射。
调度策略与传输策略可以分别选择。

| 选项 | 作用 |
|---|---|
| `PRIOR_SCHEDULE=balanced` | 原固定 step；全 cohort 生成后按实际接受列数分卡 |
| `PRIOR_SCHEDULE=window_fifo` | 有限窗口采样但保持原 step 成员；按请求列数提前分卡 |
| `PRIOR_SCHEDULE=window_bucketed` | 同一有限窗口内按成本重新组成 step，然后分卡 |
| `PRIOR_BUCKET_STEPS=4` | 不重叠窗口包含的 optimizer step 数 |
| `PRIOR_POOL_TRANSPORT=numpy/mmap` | 独立的数据传输实现；默认保持 numpy |
| `PRIOR_POOL_WINDOW=128` | 输出 pipeline 的 local batch 数量上限，与元数据窗口不同 |
| `PRIOR_POOL_MB=2048` | 预留最终输出预算；不含生成临时内存或 rank 已映射的数据 |
| `PRIOR_POOL_KERNEL_THREADS=1` | 默认单线程；大于 1 时，大距离 kernel 可借用节点共享 CPU permits |

先完整采样原 local batch 的 GP/subgroup 参数，再确定身份和 seed，最后重排。
窗口内每个任务恰好使用一次，不按完成快慢挑选、不超时丢弃、不改变过滤规则。
完整 micro-batch 不拆开，Stage1 的 micro=4 保留组内长度和 split；
Stage2/3 的 micro=1 可以逐表重组。跨 micro 的原分组相关性可能跨越 optimizer step。

每 rank 的数据集数、micro-batch 数和梯度累积权重保持相同。
一个任务最多移动 `W-1` 个 optimizer step；这是步数界，不是墙钟等待界。
每个窗口完整消费时，FIFO 与分桶的 X/y 多重集合相同；若训练在窗口中间停止，
未消费后缀不同。重排改变每步的数据相关性和 optimizer 更新轨迹，不保证与 FIFO 的
训练更新或收敛逐步等价。梯度检查验证的是**重排后同一步**的 DDP 全局均值合同。

分桶先将 micro 单元按估计成本排序，按完整全局 step 容量切桶，再等数量分配给 rank。
优化目标为一个窗口内 `sum(step 的 max_rank_cost)`；若该预测目标变差则回退。
这是启发式，不能保证真实毫秒数降低。桶的 step 顺序用独立 RNG 打乱，避免总是短到长；
采样 RNG 不受影响。多节点共用桶序号的置换，使各节点大致对应相同成本分位，
但不做跨节点任务迁移或全局精确均衡。

global batch=64 时，8 rank 的 Stage2/3 每 rank 有 8 个 micro 单元，16 rank 为 4 个。
Stage1 的 16 rank 每卡只有一个 micro=4 单元；步内置换无法降低最大值，
跨 step 的窗口重组仍有空间。实现不依赖“每卡至少两个任务”。

## 成本模型的边界

新 proxy 包括 COL 的两段 ISAB、ROW 最后一个仅 CLS query 的 block、projection/FFN、
实际 hybrid ICL 层数、全局 attention、稀疏检索和 centroid routing。
默认 12 层、sparse_every=3 对应 8 个 global 与 4 个 fused-sparse 层。
`feature_group=same` 不减少列 token，`valid` 才减少。

该数值是 forward MAC proxy，未拟合 backward、kernel 启动、sorting、dtype 或 backend 效率。
生成前使用请求列数；生成后列清理可能使实际工作量下降。
CPU 生成代价不能用同一个 GPU proxy 代替：图结构、过滤和重试次数会造成不可预知长尾。
当前 CPU 队列仍在一个 cohort 内优先提交 `rows × requested_features` 大的表，
这不是经过校准的生成耗时预测器。

## 数据面与生成服务改进

mmap 模式由 worker 将紧凑 X/y 写入私有 tmpfs 文件，经 manager 只传描述符。
rank 建立 copy-on-write 映射后 ACK，服务 unlink 文件；Tensor 持有映射，ACK 和服务关闭后仍有效。
使用 `posix_fallocate` 尽早报告空间不足，避免写映射时 SIGBUS。
它仍有 worker 写入和 CPU→GPU 拷贝，不是 GPU 零拷贝，也没有新增 pin-memory/H2D 预取流。
Stage2/3 的 micro=1 直接返回视图，避免原先无必要的 pad/stack 主机拷贝。

输出预算改按 `requested_features` 预留；compact 输出只能删除列，原来使用
`max_features` 会把长窄表的预算显著高估。分类标签按 int64 计费。
保留一个超预算 cohort 单独执行的防死锁例外；预算不是整个作业的严格 RSS 上限。

窗口策略按元数据提前分卡，一个 rank 自己的表生成完即可取走，无需先等完整 cohort。
这减少服务端不必要的全节点交付 barrier，但最后的 DDP 同步仍受最慢 rank 限制。
新增生成延迟分位数、编码/结果传递延迟、客户端 RPC/映射/ACK 时间、lease 和队列状态。
`worker_busy_seconds / workers / uptime` 仅是已完成任务占用的近似，不能当作 CPU 利用率。
启用 kernel 预算时，该时间还包含等待 base permit 的时间；新增的
`parallel_distance_extra_thread_seconds` 也是分配线程数乘墙钟，不是实际 CPU 使用量。

## 复现

先保留原默认设置，仅显式启用候选路径：

```bash
PRIOR_BACKEND=node_pool PRIOR_SCHEDULE=window_bucketed \
PRIOR_BUCKET_STEPS=4 PRIOR_POOL_TRANSPORT=mmap \
PRIOR_POOL_WORKERS=56 PRIOR_POOL_WINDOW=128 PRIOR_POOL_MB=2048 \
RANK_TIMING_EVERY=1 bash scripts/train_v2_clf_stage2.sh
```

Stage3 换为 stage3 脚本并提供真实 Stage2 checkpoint。
`window_fifo + mmap` 是隔离“跨 step 分桶”的对照；
`window_fifo + numpy/mmap` 隔离传输。`balanced → window_fifo` 同时改变了提前分配和交付时机，
不能只把差异归因于去掉 barrier。多节点时 worker 数按每节点 CPU quota 设置。
checkpoint 不保存 in-flight 窗口，恢复会启动新 prior 流，不提供精确数据重放。

诊断工具：

- `benchmark_training_schedule.py`：真实 curriculum 的隔离短训；`--shape-prior` 仅隔离形状计算成本。
- `benchmark_prior_pipeline.py --fixed-task-stream --in-order True --hash-tasks`：相同任务的供给对照。
  DataLoader 诊断路径使用紧凑 Tensor IPC，不是原训练 loader 的整 local batch padding/pinning 路径。
- `benchmark_prior_pipeline.py --payload-only`：廉价固定形状输出，只测传输和消费开销。
- `profile_graph_generation.py`：分别统计图、过滤、重试；`--start-task` 可重放指定慢 cohort。
- `inspect_prior_pool.py <local-rank-pid>`：Linux 下读取运行中服务的队列统计。
- `check_prior_schedule_ddp.py --schedule window_bucketed --transport mmap --steps 4 --bucket-steps 2`：
  使用 torchrun 检查跨窗口的真实模型多步 DDP/串行均值更新一致性。

## 真实短训：没有证明窗口策略加速

8 卡、global batch=64、Stage2/3 micro=1、56 个共享 worker、输出窗口 128 local batches、
2 GiB 预留预算，BF16，全尺寸生产模型。每项 24 步，稳态统计去掉前 4 步。
关闭 frontend compile 和 FA3 以隔离供给/调度；保留其余真实 stage 的形状采样。
各项串行运行；Stage3 使用同一个短 Stage1 权重做计算实验，**不是正式 curriculum 或收敛实验**。
短训沿用 5000 步 LR warmup，日志中的 LR 为 0；以下用于计算/供给对照，
不用于证明模型质量。非零学习率的更新合同另外由多步 DDP/串行对照检查。

| 策略 / 数据面 | Stage2 全程 wall (s) | Stage2 稳态 (ms/step) | Stage3 全程 wall (s) | Stage3 稳态 (ms/step) |
|---|---:|---:|---:|---:|
| balanced / numpy | 82.99 | 1404.7 | 259.72 | 8001.6 |
| window_fifo / numpy | 82.24 | 1452.3 | 264.22 | 8197.2 |
| window_fifo / mmap | 82.53 | 1434.0 | 267.85 | 8135.2 |
| window_bucketed / mmap | 84.89 | 1578.0 | 276.24 | 8324.8 |

全部 8 次无 OOM micro-batch、无 nonfinite loss，退出码均为 0。
这是每项一次、包含真实长尾的短实验，不足以对几个百分点的差异做统计显著性判断；
但结果明确不支持把窗口分桶或 mmap 宣传为现有 Stage2/3 的端到端加速方案。
保留 opt-in，未改变默认数据面/调度。
完整统计见 [training-summary](benchmarks/prior_window_20261031/tabicl-window-training-summary.json)。

Stage2 的 `window_fifo` 中，rank0 取数均值从 numpy 的 42.90 ms 降至 mmap 的 2.24 ms；
整步却是 1452.3 → 1434.0 ms，差异很小。Stage3 则没有整步收益。
同一组各 rank 的数据完成时间不同，先拿到数据的 rank 会在反向同步等待，
因此不能只看 rank0 取数或 `train_max` 判断 GPU 计算是否变慢。

## 生成长尾证据

分阶段 profiler 不改变 GraphSCM 或过滤器的输入/输出，每次调用恢复包装。
32 个 worker 分别生成 Stage2、Stage3 的头 128 个任务；再重放 Stage3 第 18 步
原 cohort 的 task 1088–1151。这里的比例是**各任务时间总和的比例**，不是整个进程的墙钟占比。

| 任务集合 | 图生成占比 | 过滤占比 | 中位生成时间 (s) | p99 (s) | 最慢 (s) |
|---|---:|---:|---:|---:|---:|
| Stage2，0–127 | 83.45% | 16.24% | 0.411 | 7.50 | 9.01 |
| Stage3，0–127 | 89.02% | 10.76% | 0.701 | 34.73 | 48.73 |
| Stage3，1088–1151 | 93.31% | 6.54% | 0.511 | 63.70 | 150.93 |

最慢 task 1096 请求 55,880 行 × 13 列，最终接受 8 列；
仅生成一次、过滤一次且通过，图生成 150.31 s，过滤 0.61 s。
运行中栈采样落在 `RandomDiscretizationFunction → nearest_center → torch.cdist`。
这不是“反复 rejection”可以解释的个例，也不能靠 GPU MAC 打分准确预测。

真实训练暂停时，服务曾报告 2110/2112 个任务已完成，16 个 buffered cohort 中
14 个完全 ready，预算占用约 1.4 GB、无准入阻塞、无未确认 lease。
后续表已生成而当前 step 等少数慢表：增加输出预算或换 IPC 不能消除这段等待。
当时的结果传递均值约 3.8 ms，相比秒至百秒的单图长尾不是主要矛盾。
该快照和栈只用于定位，不能代替逐阶段计时；profile JSON 已保留全部任务记录。

资料：[Stage2 profile](benchmarks/prior_window_20261031/tabicl-window-generation-profile-2.json)、
[Stage3 profile](benchmarks/prior_window_20261031/tabicl-window-generation-profile-3.json)、
[慢 cohort](benchmarks/prior_window_20261031/tabicl-window-generation-tail-3.json)、
[慢 worker 栈](benchmarks/prior_window_20261031/tabicl-window-worker-stack-tail.txt)。

## 同任务供给对照与纯数据面

CPU/Gloo 8 rank，56 个生成 worker，local batch=8，先 4 步预热、再测 8 步。
`round_robin` 固定 step 成员；DataLoader 用 7 worker/rank、prefetch=2、in_order=True，
共享池用 56 worker、112 local batch 窗口、2 GiB 预算。
测量段 task 256–767 的完整 X/y 多重集合 SHA256 在三个实现间分别一致。
这是短前缀、供给实验，不能直接外推长训吞吐；`warmup` 结束时后台预取进度也可能不同。

| Stage | 实现 | 启动+预热 (s) | 测量段 (s) | 测量段表/秒 | 两段合计 (s) |
|---|---|---:|---:|---:|---:|
| 2 | DataLoader compact Tensor IPC | 46.22 | 15.22 | 33.63 | 61.45 |
| 2 | node_pool numpy | 18.94 | 12.77 | 40.09 | 31.71 |
| 2 | node_pool mmap | 18.95 | 12.63 | 40.55 | 31.57 |
| 3 | DataLoader compact Tensor IPC | 98.75 | 28.10 | 18.22 | 126.85 |
| 3 | node_pool numpy | 58.39 | 31.15 | 16.44 | 89.54 |
| 3 | node_pool mmap | 57.70 | 32.29 | 15.86 | 89.99 |

共享池逐表分发，DataLoader 一个 worker 顺序生成一个 local batch，调度粒度不同。
Stage3 DataLoader 测量段略快，但其预热明显更久；只报任一口径都不充分。
这些结果否定了“node_pool 天生比 DataLoader 慢”的笼统结论，也没有证明真实训练应该默认切换。
完整 JSON 位于本目录的 `benchmarks/prior_window_20261031/tabicl-window-replay-*.json`。
初版 DataLoader 基准在计时/哈希完成后出现 vendor Tensor-IPC worker 退出警告。
直接调用 `_shutdown_workers()` 的补充实验也在未计时的后台图生成中触发 SIGABRT。
诊断脚本因此增加共享 stop event：完成测量后，在表之间停止未来生成，
消费完正在传递的结果，再关闭 persistent workers，最后销毁 Gloo。
这个排空只影响未计时的预取后缀，不改变已测量的任务或哈希。
8 rank Stage3 的完整重放与清理已复测退出 0，无上述 worker 退出错误；
测量段哈希仍为 `d71cd9f40cf466a10aefa6d8ce48cdf72ca7372d648a0c3c88abc0afbd4de50d`。
复测测量段 27.07 s、启动/预热 98.24 s，与原对照量级一致。
这些分段时间都不含退出后的排空；见
[退出清理复测](benchmarks/prior_window_20261031/tabicl-window-replay-drain-3.json)。

纯 payload 实验使用固定行数、约 100 列、ones/zeros，无图生成/过滤，
8 步预热后测 32 步，仍做主机有限值检查。DataLoader 对照未启用 pinning。

| 行数 | node_pool numpy (表/秒) | node_pool mmap (表/秒) | DataLoader Tensor IPC (表/秒) |
|---|---:|---:|---:|
| 2048 | 493.6 | 3798.9 | 4873.1 |
| 16384 | 87.9 | 949.7 | 980.7 |

两种长度下 mmap 分别约为 numpy 的 7.7/10.8 倍，证明旧的数组传输确有开销，
修复后接近 DataLoader 的 Tensor IPC。这个倍数只适用于廉价表的供给微基准，
不能解释为真实 prior 或训练也能获得同等倍数的提升。
完整 JSON 为 `tabicl-window-payload-*.json`，包含启动、payload、各 rank 等待与池状态。

## 窗口分桶的计算收益与 16 rank 边界

`--shape-prior` 保留真实 stage 的请求形状，使用廉价随机 X/y 替代图与过滤。
这隔离 GPU 形状调度，不是质量训练：生成后的真实接受列数通常更少。
8 卡、global batch=64、micro=1、mmap、8 个廉价生成 worker；
32 步，去掉前 4 步，其他模型和 dtype 与上述短训一致。

| Stage | FIFO (ms/step) | 分桶 (ms/step) | FIFO step p95 (ms) | 分桶 step p95 (ms) |
|---|---:|---:|---:|---:|
| 2 | 748.3 | 757.7 | 869.4 | 1399.4 |
| 3 | 1669.1 | 1665.0 | 2074.6 | 4019.0 |

4 次实验全部退出 0，无 OOM/nonfinite。分桶将昂贵表集中到少数重 step，
因此跨 step 的耗时方差增大；单步 p95 不降不等于总计算量增加。
现有 8 micro/rank 的步内均衡已经很接近均摊，实测没有证明额外窗口分桶有平均收益。
见 [形状实验](benchmarks/prior_window_20261031/tabicl-window-shapes-summary.json)。

再用 `audit_prior_windows.py --windows 32` 分析 128 个 step 的参数，
global batch=64、W=4，比较 **已有步内均衡** 与 **窗口分桶+步内均衡** 的预测总 makespan。
这里使用请求列数和默认生产模型的 MAC proxy，完全不运行 GPU，也不计 CPU 长尾。

| Stage | 8 rank 每卡 micro 数 | 8 rank 预测比值 | 16 rank 每卡 micro 数 | 16 rank 预测比值 |
|---|---:|---:|---:|---:|
| 1 | 2 | 1.052× | 1 | 1.359× |
| 2 | 8 | 1.00036× | 4 | 1.147× |
| 3 | 8 | 1.00024× | 4 | 1.037× |

因此 8 rank 的负结果不能推出 16 rank 也没有空间，尤其 Stage2 的预测改善更大；
Stage1 的“一卡一个 micro”也可通过跨 step 组桶改善预测目标。
这些数字是**模型预测，不是物理 16 GPU 加速结果**；仍需在目标节点验证 kernel 效率、
动态编译、共享池供给和跨节点最长 rank。完整结果见
[元数据审计](benchmarks/prior_window_20261031/tabicl-window-metadata-headroom.json)。
表中的 16 rank 按一个节点内的 16 rank 计算；两个节点各 8 rank 时，实际只有节点内的
32 张表参与每步重组，不能直接套用这一预测比值。

## CPU 长尾的定点改进

对 task 1096 单独重放，保持全部随机种子、prior 和过滤参数：

| 大规模非欧氏距离线程数 | 完整生成 (s) | 距离函数时间总和 (s) | X/y SHA256 |
|---|---:|---:|---|
| 1 | 151.35 | 140.12 | 一致 |
| 8 | 28.64 | 17.61 | 一致 |

该任务含 55,880 行、537 维中间节点与 255 个中心的距离调用，
输出只有 13 个请求列并不能代表图中间层宽度。距离占其原耗时约 92.6%。
非整数 p 的 CPU `cdist` 很贵，固定所有 worker 为单线程会让最后一张慢表独占一核，
即使其他核已空闲也无法加速。此处保留原 `cdist` 算法、分块与归约，
只并行独立距离，未引入近似计算、改 p、放宽过滤或截断任务。
1/8 线程的单表结果见 `tabicl-window-distance-{1,8}.json`。

再次使用生产共享 permit 路径而不是诊断强制线程数，完整生成为 **29.08 s**，
距离总时间 18.04 s，最大借用线程数为 8；完整 X/y 哈希仍完全相同。
见 [生产预算重放](benchmarks/prior_window_20261031/tabicl-window-distance-budget-8.json)。

再用 32 个 worker 串行运行两种配置，完整重放 task 1088–1151 的 64 表 cohort，
避免只优化一个脱离竞争环境的任务：

| kernel 上限 | 进程 wall (s) | 单表 p99 (s) | 最慢单表 (s) |
|---|---:|---:|---:|
| 1 | 156.36 | 63.62 | 150.66 |
| 8，共享 permits | 34.75 | 16.44 | 29.03 |

64/64 表的完整 X/y SHA256、行数、请求/接受列数、生成尝试数和过滤结果逐项一致，
两轮都退出 0。该 profile 是无 GPU 消费者的独立供给实验，预算预留一个客户端核；
后面的 8 卡训练按 8 个 local rank 预留 CPU，并使用 56 个生成 worker，配置不同。
完整记录见 [单线程基线](benchmarks/prior_window_20261031/tabicl-window-generation-tail-baseline-3.json)
与 [共享预算](benchmarks/prior_window_20261031/tabicl-window-generation-tail-budget-3.json)。

生产实现为独立 `_cpu_budget.py`：每个生成任务先持有一个 base permit，
足够大的多维、非 p=2 距离调用才尝试**非阻塞**借空闲 permits，
退出后恢复线程数并归还。总 permits 为 `max(1, available_cpu_cores - local_ranks)`，
受 affinity/cgroup quota 限制；不能给每个 worker 各自放大一套未计费线程。
该预算由同一作业的 local ranks 共享，不是节点上多个独立训练作业的全局 CPU 调度器。
没有空闲 permits 时继续单线程；p=2 GEMM、1-D 优化与小距离计算保持原路径。
任务失败也归还 permits，worker 硬退出由服务报错并整体清理，不在损坏的池上继续。

`PRIOR_POOL_KERNEL_THREADS=8` 是可选的每 kernel 上限；默认仍为 1。
借用不是抢占式的：一次已经启动的距离调用不能在别的 worker 随后空闲时动态扩线程，
也可能让刚到达的新任务短暂等待 base permit，必须用完整训练对照评估净效果。

同前缀、相同 8 卡/56 worker、24 步设置，将 kernel 上限从 1 改为 8，结果如下。
稳态仍去掉前 4 步；除线程预算外配置与对应旧行一致。

| Stage | 策略 / 数据面 | 单线程 wall (s) | 预算 wall (s) | 单线程稳态 (ms/step) | 预算稳态 (ms/step) |
|---|---|---:|---:|---:|---:|
| 2 | balanced / numpy | 82.99 | 74.10 | 1404.7 | 1174.4 |
| 2 | window_bucketed / mmap | 84.89 | 81.06 | 1578.0 | 1476.9 |
| 3 | balanced / numpy | 259.72 | 195.60 | 8001.6 | 4913.1 |
| 3 | window_bucketed / mmap | 276.24 | 193.23 | 8324.8 | 4888.4 |

4 次均退出 0、无 OOM/nonfinite。原 balanced 路径 Stage2/3 稳态分别缩短约 16.4%/38.6%，
包含启动的全程分别缩短约 10.7%/24.7%。这支持 CPU 长尾定点修复，
不等于长训平均收益保证：各项仅一次、共享 CPU 预算的竞争随预取状态变化，
且前缀含已定位的极慢图。完整数据见
[CPU 预算短训](benchmarks/prior_window_20261031/tabicl-window-kernel-summary.json)。

同样启用 CPU 预算后，Stage2 分桶仍慢于 balanced，Stage3 二者接近。
因此不能把预算带来的收益算成分桶收益，也不能据此默认切换数据面/分桶策略。

## 正确性验证与适用范围

针对性测试覆盖窗口内身份/参数/种子、完整 micro 边界、任务无丢失重复、
步数位移界、相等梯度权重、提前交付、预算小于单个 cohort 的进展、
传输失败传播、mmap ACK/关闭后的生命周期、单表 kernel 结果一致与异常归还 permits。
窗口属性测试覆盖 1/2/8/16 rank、每卡一个 micro，以及 W=1/2/4/8。
真实生成两窗口对照比较 FIFO/分桶和 numpy/mmap 的完整 X/y 多重集合哈希。

最终针对性回归 **175 passed**；完整仓库回归 **1144 passed、260 skipped、40 warnings**，
耗时 718.88 s，退出码 0。没有修改精度容差。跳过项主要为本机无 XPU/MPS、
无 Docker，以及 sklearn 已声明的非确定性/未启用 array-api 检查。
完整回归使用本机 venv、`TABICL_CHECKPOINT_DIR` 中的两个既有 estimator checkpoint，
并在同一 Python 进程先禁用 FA3，再调用 `pytest.main(["-q", "-rs"])`；
清除了短训脚本注入的 prior/compile/stage 环境覆盖。
见 [完整回归日志](benchmarks/prior_window_20261031/tabicl-window-cpu-full-tests.log)
和 [针对性回归日志](benchmarks/prior_window_20261031/tabicl-window-cpu-unit.log)。

16 rank CPU/Gloo 检查使用真实 TabICL 小模型、mmap、window_bucketed、kernel cap=8，
Stage1 micro=4、Stage2/3 micro=1；每个 stage 跨两个窗口做 4 次非零 LR SGD 更新。
DDP 与相同重排后数据的串行全局均值比较，12 次更新参数最大误差
**1.1921e-7**，未放宽原 `atol=2e-6, rtol=2e-5`。
这验证正确性，不是 16 GPU 性能或原 FIFO optimizer 轨迹等价性。
日志见 [16 rank 更新检查](benchmarks/prior_window_20261031/tabicl-window-ddp16-budget.log)。

当前更有证据的 8 卡试用方式是先保留 balanced，单独打开 CPU 预算：

```bash
PRIOR_BACKEND=node_pool PRIOR_SCHEDULE=balanced \
PRIOR_POOL_WORKERS=56 PRIOR_POOL_KERNEL_THREADS=8 \
RANK_TIMING_EVERY=1 bash scripts/train_v2_clf_stage2.sh
```

56 只对应本机配额和 8 rank 的实验配置；其他节点可先用 `PRIOR_POOL_WORKERS=0`
按 quota/affinity 和 local rank 自动计算，再量测 worker 数与内存。
窗口分桶建议作为目标 16 GPU 布局的对照项，分别测 `window_fifo`、`window_bucketed`，
保持传输、CPU 预算和任务前缀一致，记录真实整步和全程耗时。
没有 16 张物理 GPU 测量，也未验证正式 Stage2→3 长训质量、动态编译后的吞吐
或跨节点全局均衡；这些不能由 CPU DDP 正确性与 MAC 预测替代。

若继续降低 Stage3 等待，应先区分「没有空闲 CPU permit」和「已经有空闲核，
但当前单线程 kernel 不能中途扩展」。观察 generation p99、borrow 次数/最大线程和
逐 rank 整步时间，比单独加大输出窗口更有信息。当前 CPU 优先级只看请求形状，
不能代表图中间层的代价；更好的 CPU 预测需要 DAG 中间宽度、中心数、距离 p 等元数据。
若将这些图结构也提前采样，必须另外维护随机流合同并验证哈希，不能直接用 GPU MAC
代替生成耗时，也不能以跳过慢表的方式提升测量吞吐。
