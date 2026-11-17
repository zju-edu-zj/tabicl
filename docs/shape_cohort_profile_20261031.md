# 同形状 cohort 与训练流水线 profile（2026-10-31）

## 目标机器与测量边界

通过更新后的 `ssh -p 65484 127.0.0.1` 已恢复远程实验。目标为
8 × 96 GiB PPU-ZW810、约 69 核 CPU quota、约 1 TiB RAM，Python 3.12.11、
厂商 PyTorch 2.8 / Triton 3.4。隔离实验目录为 `/tmp/tabicl-cohort-target-20261031`，
环境为 `/tmp/tabicl-perf-venv`；不覆盖其他实验的 checkpoint。
非交互 shell 先 `source /etc/profile.d/dockerenv.sh`，否则厂商 SDPA 缺少 PPU SDK 环境。

CPU profiler 与 GPU 实验串行，避免抢占生成核。GPU 使用完整 27,055,194 参数模型、
content_tree、compile all、逐 rank 三轮前反向预热、global batch=64、micro=1。
短跑使用隔离随机初始化，不能用于判断预训练收敛或验证集质量。
本机只有 8 张卡；16 rank 的 CPU/Gloo 正确性检查不能替代 16 卡吞吐实验。

此前远程审计见 [有限窗口与供给审计](prior_window_20261031.md)。其中旧配置出现过
55,880 行、请求 13 列、图内 537 维中间节点的单表 150 秒长尾；共享 CPU permits
将该表降至约 29 秒，但那不等于最新编译默认配置的 Stage3 总时间仍主要受 CPU 限制。

## 为什么裁剪与不裁剪必须分别测

现有 adaptive limit 是采样时限制请求列数及过滤器的 padded 宽度，不仅是输出后删列：

| 行数上界 | 10,240 | 20,000 | 30,000 | 40,000 | 50,000 | 60,000 | 65,000 | 更长 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 列数 cap | 100 | 80 | 60 | 40 | 30 | 20 | 15 | 10 |

本次补充 `ADAPTIVE_FEATURE_LIMIT=False` / `--adaptive_feature_limit False`，用于关闭该规则，
保留其他过滤规则和 `max_features=100`。Stage2 最大 10,240 行，正常配置下两者没有差异。
Stage3 的差异取决于长度；短表不变，60,000 行的列数 cap 从 20 变成 100。

不裁剪会同时增加 CPU 图生成／过滤与 GPU frontend 的工作。COL 随列 token 数增长，
ROW 的一部分 attention 随列 token 数呈平方增长；当前 `col_feature_group=same`
保留每个原列对应的 token，不能误按 `ceil(width/3)` 估算。
ICL 的输入维度由 `row_num_cls × embed_dim` 固定，请求列数对它的直接形状影响较小。
因此裁剪后 ICL **相对占比可能提高**，但不能从层数或 MAC 直接得到毫秒占比。

CPU 是否成为瓶颈，要比较有效供给与 GPU 需求：每个节点每步需要
`64 × local_world_size / world_size` 张表，若 GPU 计算需要 T 秒，所需供给约为该数量除以 T。
平均供给不足会持续积累等待；平均足够但指定 cohort 内有慢表，也会暴露长尾。
行数增大时 CPU 与 GPU 工作都增长，二者比例不是固定的；裁剪、kernel、过滤、线程竞争
都会改变这个比例。取数等待还能转移成别的 rank 的反向同步等待。

## 目标机器 CPU profile

56 个生成 worker、61 个共享 permits、kernel 上限 8 线程、预留 8 个客户端核，
OMP/BLAS 均为 1；全部使用真实 GraphSCM、正式过滤规则。
正式训练的 auto workers 为 61，而这里为 56；后者给大 kernel 留出更多可借 permits。
这些独立 CPU profile 用于定位热点，不能直接当成在线 61-worker 池的稳态供给率。

| Stage3 场景 | 表数 | 总 wall (s) | 单表 P50 (s) | P99 (s) | 最大 (s) | 累计任务时间 (s) |
|---|---:|---:|---:|---:|---:|---:|
| 原长度分布，adaptive | 512 | 32.62 | 0.80 | 16.71 | 26.54 | 970.84 |
| 原长度分布，无 cap | 512 | 55.94 | 0.78 | 16.83 | 40.58 | 1097.90 |
| 固定 60k 行，adaptive | 128 | 46.60 | 2.64 | 33.78 | 40.23 | 690.41 |
| 固定 60k 行，无 cap | 128 | 53.28 | 8.55 | 39.19 | 48.62 | 1507.77 |

60k 不裁剪的累计工作量约 2.18 倍，而总 wall 只有约 1.14 倍；并行批次的末尾由最慢任务支配。
混合长度的中位数接近不说明两者代价相同，应同时看长尾、累计工作与节点供给。
同一参数采样前缀保留 rows/split/classes/task seed；改变宽度会改变图 RNG 消耗和重试，
不是给同一个已生成图追加列。

原长度分布 adaptive 的累计任务耗时中，图构建／生成占 **80.29%**，过滤占 **19.12%**，
全部 `cdist` 占 **41.45%**；无 cap 分别为 **74.82% / 24.47% / 38.61%**。
`cdist` 是嵌套时间，不能与图生成相加，也不代表训练整步的 41%。
`RandomEMAssignmentFunction._transform` 的直接距离调用尚未进入 nearest-center 借核路径，
adaptive 合计约 64.94 秒、占任务总时间约 6.69%；因此不能把它当成唯一或最大的剩余瓶颈。
已借核的多维 nearest-center 距离仍占主要距离成本。
混合 adaptive 样本中，大型非欧氏、多维 nearest-center 的单线程 `cdist` 调用累计约
191.94 秒，多线程调用累计约 40.16 秒（这不是同一 kernel 的配对加速比）。
61 worker 配 61 permits 持续满载时难有可借额度；应把 48/56/61 worker 的在线吞吐与
长尾分布作为联合调参对象，不能假定更多 worker 一定更快。

外部 rows × 请求列数与任务耗时的 Spearman 相关约 **0.666**；最慢的一张请求只有 6 列、
21,115 行，却生成 26.54 秒并尝试两次。应关注图内中间宽度、函数类型和拒绝重试，
不能仅用最终表宽预测 CPU 长尾。
这里的“尝试次数”是外层 `GraphSCM.__call__` 次数；`RandomDataset.sample` 内部还会
重新采样不满足祖先关联约束的 DAG，这部分包含在图生成时间中，但没有另算进该计数。
512 表中有 157 表需要两次以上外层尝试，最多 6 次；距离调用的最大中间宽度为 762、
最大中心数为 577。宽度大于 256 的距离调用只有 49 次，却累计约 32.59 秒。
这提供了内部形状的优化目标，但限制内部宽度／重试次数会改变 prior，不能当作等价优化。

生成器已经在 `RandomGraphFunction._fit` 中剪掉不通向输出的节点，不能再把“跳过无用节点”
列成尚未实施的主要优化。更准确的 CPU 调度预测需要图内函数类型、父节点拼接宽度和
中心数等信息；当前 `RandomTransformer` 是依赖输入、逐节点惰性 fit，结构采样、张量参数
采样和数据计算交错。把它拆成便宜的完整计算计划需要保留 RNG 消耗及拒绝语义的重构，
不是在现有 rows/width 上再乘一个经验系数就能得到可靠预测。

在线 `/proc` 采样进一步区分了两种状态。Stage2 已采到的稳态窗口中，生成子进程的
CPU 中位数约 60 核，训练 rank 合计约 7.6–7.8 核，接近节点约 69 核配额；
这时 61 worker 占满 61 个基础 permit，大距离 kernel 通常无额外额度可借。
Stage3 seed42 balanced 的两段超过 36 秒的采样无进展区间，生成子进程平均却只用
约 4.9／5.3 核；说明暴露的尾部也会发生在 CPU 大量空闲时，不能全部解释成总吞吐不足。
它可能是已在执行的单线程热点、无法借核的直接距离调用，或先前已确定线程数的长 kernel，
需要结合慢任务调用栈判断，不能从节点 CPU 总量直接定位到一个算子。

采样使用进程 user+system CPU 增量除以两秒间隔，按间隔加权；不把任务 wall time
当 CPU 利用率。`rank_descendant` 包括生成 manager/worker、资源跟踪进程及可能的编译子进程，
只选训练期解释；采样间退出的短命进程会漏算。无进展区间只是日志 step 不变的采样下界，
不是 GPU idle 的精确时间。早交付时部分 rank 可能已经进入 collective 轮询，
因此 rank 消耗 CPU 也不证明 GPU 在做有效模型计算。

### 慢图配对调用栈：有空闲核仍然慢的具体原因

在目标节点空闲后，单 worker、相同 task seed、真实过滤，逐表 cProfile 复跑：

| task | 行×请求列 | 全部单线程 (s) | 正式共享借核，上限 8 (s) | 剩余主要距离成本 |
|---|---|---:|---:|---|
| 29 | 35,022×32 | 49.18 | 14.37 | EM 直接调用 2.85s；常规 nearest-center 5.13s |
| 106 | 21,115×6 | 30.79 | 24.86 | 非有限值 fallback 19.88s，仍为 1 线程 |

每对完整 X/y 的 SHA256、两次外层尝试及接受宽度均一致。task29 的累计进程 CPU 时间
都是约 48.96 秒，wall 的下降来自并行；task106 的正式借核只消除了常规路径部分。
这里的 cProfile 有插桩且 CPU 竞争条件与 56-worker 批次不同，不能拿绝对秒数逐表相减解释在线停顿。

task106 在 `_distance.py:nearest_center` 检测到非有限中间值后，直接调用 `torch.cdist`，
绕过后面的 `parallel_distance`。其中 21,115×332 对 255 个中心，p≈0.5505，单次约
10.08 秒；另一次 21,115×263 对 246 个中心，p≈0.5402，约 8.51 秒。
它们不因最后只接受 6 列而变小；输出图仍可能通过原过滤规则。
因此“61 个 worker 把 permits 用满”不是所有长尾的充分解释：这条路径即使有额度也不会借。

额外的隔离诊断 `--distance-threads 8` 把借核入口提到整个 nearest-center 调用外，
task106 降至 **7.39 秒**，完整 X/y 哈希仍一致，进程 CPU 时间约 30.89 秒。
这不是正式配置的吞吐结果，也没有把无预算强制多线程带入生产；它证明下一项应优先
将大型非有限值 fallback 纳入同一个共享预算，再处理 EM 的直接距离调用。
不能因全体任务中 fallback 平均只约 2% 就忽略它：固定 step 的最长等待由个别任务决定。

## 本地 CPU 补充实验（非目标吞吐）

环境：macOS arm64，10 个可用 CPU 核，16 GiB RAM；Python 3.12.15，
Torch 2.8.0，NumPy 2.5.4，SciPy 1.18.1；无 CUDA。
独立 venv 为 `/tmp/tabicl-cohort-venv`。4 个生成 worker，kernel 上限 8 线程，
9 个共享 permits（独立 CPU 实验只预留一个客户端核）。这不是远程 8/16 GPU 的资源配置。

Stage3 原长度分布的 128 张真实图，adaptive limit 开启：生成器内部图生成占
82.33%，过滤占 17.47%；单表 P50 0.361 秒，P99 12.756 秒，最大 23.031 秒。
图构建／生成中的 nearest-center 包装耗时占任务总时间 43.25%，这是嵌套计时，不能与图生成相加。
`rows × requested_features` 与任务耗时的 Spearman 相关系数只有约 0.639，
不能把该乘积当成可靠 CPU 长尾预测器。

同一参数采样前缀关闭 limit 后，128 表 wall 为 40.01 秒，开启时为 40.97 秒；
但单表 P50 从 0.361 增至 0.426 秒、过滤占比从 17.47% 增至 27.22%。
这一次混合长度短样本的总时间被个别长尾和并发竞争支配，不能解释为“不裁剪更快”。
配对保持行数、split、类别数和 task seed；改变宽度后图内 RNG 消耗及重试轨迹可能变化，
并不是同一个图仅增加若干列。

固定 60,000 行的 16 表对照更直接：初次测量开启／关闭 limit 的单表 P50 为
1.77／3.82 秒，完整生成 wall 为 21.57／36.55 秒。样本小、仅本地供给实验，
不用于宣称远程训练加速。完整 kernel 插桩后的独立复测为：

| 60k 行，16 表 | 单表 P50 (s) | 单表 P99 (s) | 生成总 wall (s) | 图生成 | 过滤 | 所有 cdist |
|---|---:|---:|---:|---:|---:|---:|
| adaptive cap=20 | 1.756 | 13.550 | 16.627 | 81.88% | 17.84% | 30.27% |
| 无 cap，max=100 | 3.826 | 20.048 | 33.833 | 61.71% | 37.90% | 20.51% |

两次相同场景的 adaptive wall 仍有 21.57／16.63 秒波动，正说明不能用一次短跑给出稳定加速倍数。
原始参数、逐表耗时、哈希、完整环境及 MAC 审计的紧凑记录见
[CPU 结果](performance_results/shape_cohort_cpu_20261031.json)；完整 kernel 明细仍在本地 `/tmp/tabicl-cohort-60k-*-v2.json`。

慢 task 29 的单 worker cProfile 显示约 7.21 秒总时间中，124 次 `torch.cdist`
累计约 4.20 秒；nearest-center 累计约 3.27 秒。审计发现 `RandomEMAssignmentFunction._transform`
直接调用 `torch.cdist`，**没有经过当前 nearest-center 的共享借核路径**。
因此仅观察 `parallel_distance_calls` 会漏算剩余距离成本。
profiler 已单独记录所有 `torch.cdist` 的 caller、形状、p、线程数和时间，
同时记录图构建、尝试次数、过滤拒绝、列清理、类别检查、permit 等待及 worker RSS。

这些百分比的分母是各任务耗时之和，不是整步训练墙钟，也不是 CPU 使用率。
生成过程中最大中间节点宽度可以远大于最终列数，裁剪不能彻底消除图内距离长尾。

## 目标机器 GPU profile：裁剪版首轮

BF16、8 rank、完整模型，40 步去前 10 步，每个 case 统计 1,920 张表。
下表为逐 rank 的 `wait + train` 配对后取最大值，再跨步求均值；与单机对齐墙钟差异小于 0.01 秒。
这轮只含一个 seed，是诊断对照，吞吐结论另用无 profiler 的多 seed 测量。

| 生成器 / 调度 | 每步 (s) | 最长训练区间 (s) | 最长取数等待 (s) | 平均行数 | 平均实际列数 |
|---|---:|---:|---:|---:|---:|
| 真实图 / balanced | 4.557 | 1.662 | 2.916 | 12,031 | 40.53 |
| 真实图 / cohort | 3.540 | 3.460 | 1.654 | 13,959 | 43.45 |
| 廉价形状图 / balanced | 1.625 | 1.578 | 0.066 | 12,031 | 41.71 |
| 廉价形状图 / cohort | 1.825 | 1.722 | 0.133 | 13,959 | 44.53 |

不能将第二、第三列的最大值相加；它们可能来自不同 rank。cohort 的平均
`rows × 实际列数` 比 balanced 高约 29%，短跑的独立形状数也更少。
真实图 cohort 在此窗口每步缩短约 22.3%，但廉价生成器下反而略慢，
尚不能归因于更好的 GPU 均衡。window_fifo 对照保留原全局任务和 step 组成，
同时改为按请求宽度预分配 rank、允许本地 batch 提前交付；它用于区分这组实现差异与
共享形状条件的影响，不能将其差值严格归因于提前交付这一项。

真实图 balanced 的第 19 步取数最长约 **60.71 秒**，第 37 步约 13.03 秒、
第 17 步约 11.72 秒；其余 27/30 步的最慢 rank 取数均小于 0.2 秒。
这轮均值受到少量极慢图的强烈影响，主要问题是暴露的生成长尾／有序交付等待，
不能描述成每一步都在持续缺数据。cohort 也有约 25.07、16.51、5.42 秒的取数长尾。
多数步取数快也不证明长期供给充足：一次长 stall 会给未来多个 step 留出预生成时间，
接下来又能连续取到数据。裁剪版廉价生成器约 1.63 秒/步，对应约 39 张表/秒的 GPU 需求；
独立 CPU 的 15.7 张/秒包含启动和末尾长尾，不能直接作为在线容量，但足以提醒不能
只看取数中位数。应同时审计较长窗口的累计消费、生成进度、预算准入与关键慢任务。
cohort 某一步 rank1 取数约 5.42 秒、训练约 4.21 秒，而 rank2 取数仅 0.07 秒、
训练却约 9.55 秒：后者的反向同步在等前者。因此训练区间增加不等于 kernel 变慢。

| 真实图，每表平均 CUDA stream 边界 (ms) | balanced | cohort |
|---|---:|---:|
| H2D | 0.211 | 0.233 |
| frontend 预处理 | 1.619 | 1.222 |
| COL 前向 | 11.027 | 12.950 |
| ROW 前向 | 12.373 | 14.643 |
| ICL 前向 | 42.074 | 48.486 |
| ICL 反向 | 48.498 | 53.311 |
| ROW 反向 | 39.715 | 48.168 |
| COL 反向及同步尾部 | 46.253 | 217.760 |

ICL 占三个前向模块之和约 **64.3% / 63.7%**；前后向分母包含 COL 同步尾部时，
ICL 比例约 **45.3% / 25.8%**。后一个下降主要受等待污染，不能据此判断 ICL 更便宜。
如果只把 ICL 前向加速一倍、其他路径不动，也消不掉当前秒级的取数或同步长尾。
H2D CUDA 边界是亚毫秒量级；host `.to()` 偶尔较长可能在等此前 GPU 工作，不能直接解释成传输带宽不足。

节点池每十步的快照里经常已经有 6–14 个完整未来 batch，仍可在较早的指定 step
出现长尾等待。这提示应区分平均供给不足和有序交付的队头阻塞；
`ready_batches` 是快照，不能当成整个等待期间始终有多少可用数据。
worker → manager 的平均结果 transit 约 4–7 ms、编码约 0.03 ms，
它们与客户端等生成的 RPC 总等待不是同一指标。

在线 CPU 采样另外读取训练 launcher 的整个 `/proc` 子进程树，以进程 user+system
CPU 时间增量除以两秒采样间隔，区分训练 rank 与 rank 的子进程。
后者包括 manager、生成 worker 和资源追踪器；编译时也可能含编译子进程，
所以只解释完成第十步后的训练样本。进程在两次采样之间退出会少算，RSS 不作为独占内存相加。
这是 CPU 实际执行时间，不使用 `worker_busy_seconds` 冒充 CPU 核利用率。

未裁剪的原始配置在预热 60k × 100 时出现 OOM（96 GiB 卡已用约 95.7 GiB，
还需约 1.49 GiB），没有产生有效训练比较。需要启用重计算，并在裁剪／不裁剪两组同时启用，
以免将重计算额外成本误算为列数成本。实验不通过降低最大列数来隐藏此限制。

后续对照继续使用同一设备，`RECOMPUTE=True`，40 步去前 10 步；
无 profiler 吞吐另以两个 seed 各跑 80 步、去前 10 步。
实际矩阵复用 `/tmp/torchinductor_root`，用于评估暖 cache 下的运行；
各次 `run.json` 的总 wall 包括启动、编译／预热和最终保存，但试验顺序会影响
cache 命中，不可将这些总 wall 当成相互独立的冷启动成本对照。

### 相同重计算设置的完整裁剪对照

以下全部 `RECOMPUTE=True`，其余与前轮相同；单位为秒。

| 列数规则 | 生成器 / 调度 | 平均每步 | 每步 P50 | 每步 P95 | 最长取数的均值 | ICL 前向占比 |
|---|---|---:|---:|---:|---:|---:|
| adaptive | 真实 / balanced | 3.906 | 2.249 | 12.450 | 1.777 | 67.6% |
| adaptive | 真实 / cohort | 3.552 | 1.701 | 14.297 | 1.178 | 65.7% |
| adaptive | 廉价 / balanced | 2.071 | 2.012 | 2.434 | 0.075 | 65.7% |
| adaptive | 廉价 / cohort | 2.247 | 1.403 | 5.386 | 0.106 | 64.1% |
| 无 cap | 真实 / balanced | 4.721 | 2.840 | 18.207 | 2.025 | 62.4% |
| 无 cap | 真实 / cohort | 7.838 | 1.576 | 27.809 | 4.795 | 54.4% |
| 无 cap | 廉价 / balanced | 2.418 | 2.323 | 3.085 | 0.106 | 57.6% |
| 无 cap | 廉价 / cohort | 3.091 | 1.374 | 9.793 | 0.236 | 52.4% |

balanced 不裁剪时，实际 `rows × features` 从 319,014 增至 584,023，约 1.83 倍；
COL / ROW / ICL 前向每表从 10.36 / 11.80 / 46.31 ms 变成
16.37 / 14.66 / 51.59 ms。ICL 的前向占比降低，不代表 ICL 绝对耗时降低。
排除预热后的训练期峰值显存从约 17.66 增至 45.57 GiB。
无 cap 廉价生成器的实际工作量略有不同，所以 4.72 与 2.42 的差不能严格当成
可消除的 CPU 时间，但可以排除“这些等待全是模型纯计算”的解释。
也不能从真实图两轮的 4.56／3.91 秒得出“重算提高计算速度”：裁剪版廉价生成器
启用重算后从 1.63 增至 2.07 秒/步，GPU 工作确实增加。真实流水线受到运行时 CPU
长尾和预取进度影响，GPU 算得慢一些也可能掩盖更多生成时间；这两轮不是重算加速的证据。

cohort 的无 cap 第 27 步为 56,496 行、请求 76 列：rank6 取数 82.51 秒、训练
11.33 秒，rank4 取数仅 0.25 秒、训练区间却为 93.59 秒；整步约 93.84 秒。
这是供给长尾转移成其他 rank 的 DDP 等待的直接记录。
该 cohort 窗口的实际 `rows × features` 也比 balanced 高约 57%，因此不能把
7.84/4.72 的全部差异归因于调度机制。即使廉价生成器，cohort 的长表步也会一起变重。
两 seed、无 profiler 的吞吐结果在下文单列，避免将这个单 seed 的短窗口当成稳定结论。

### 无 profiler 的正式流水线吞吐

8 卡 BF16、adaptive、recompute=False，真实图、content_tree、compile all、正常预热；
每组运行 80 步，去掉前 10 步，剩余 70 步／4,480 表。两 seed、三调度、两 stage 共 12 组。
单位为秒/optimizer step，mean 先逐 rank 配对 wait+train 后取最大，再跨步求平均。

| Stage | seed | balanced 均值 / P95 | window_fifo 均值 / P95 | shape_cohort 均值 / P95 |
|---|---:|---:|---:|---:|
| 2 | 42 | 1.169 / 1.119 | 1.190 / 1.853 | 1.308 / 4.229 |
| 2 | 43 | 1.038 / 1.064 | 1.034 / 1.456 | 0.975 / 1.731 |
| 3 | 42 | 3.679 / 15.873 | 3.658 / 17.458 | 2.782 / 7.893 |
| 3 | 43 | 2.558 / 8.755 | 2.797 / 6.480 | 2.598 / 12.001 |

全组 OOM=0、nonfinite loss=0，并执行了正常 optimizer 更新；四舍五入的进度条 LR=0
不表示未更新。Stage2 cohort 两 seed 合并均值为 1.142s，对照 1.104s，耗时增加约 3.4%。
Stage3 合并为 2.690s，对照 3.119s，耗时减少约 13.7%、吞吐提高约 15.9%；
但 seed42 快约 24.4%，seed43 慢约 1.6%，**尚不能称为稳定普遍加速**。
seed43 cohort 的 P95 还从 8.75s 增至 12.00s。

同形状条件会降低短样本的独立形状数。重建实际采样元数据后，cohort 相对 balanced 的
平均请求 MAC 在 Stage2 seed42/43 分别为 -7.36%/-0.82%，Stage3 为 -13.46%/+1.47%。
因此不能把 Stage3 seed42 的全部改善归因于调度。window_fifo 与 balanced 保持同一全局任务
集合和 step 组成，但预分配宽度依据及提前交付也有差异；它在这次 Stage3 两 seed 合并中
没有提高吞吐，说明仅提前交付完整本地 batch 尚不能消除慢表和 DDP 等待。

Stage3 两 seed、三个调度的在线生成子进程平均实际 CPU 约 36–48 核；
seed42 balanced 是 36.29 核，cohort 40.05 核；seed43 分别是 48.41/48.00 核。
均值既包含满负载片段也包含长尾，不表示存在可随意占用的固定空闲核数。
全部逐组均值、P50/P95/max、启动加保存 wall、请求形状直方图、在线 CPU 及慢图调用栈见
[目标机完整汇总](performance_results/shape_cohort_target_20261031.json)。
原始逐 rank 记录及 metrics 保留在目标机 `/tmp/tabicl-target-*`。

### 固定 60k×20 的实际 GPU kernel trace

另跑独立长表诊断：8 卡、廉价形状数据、同形状条件、recompute=True、8 micro/rank，
只对 rank0 的一个完整 optimizer step 记录 Kineto；不用其有插桩时间代替上表吞吐。
CPU condition 为 59,999 行、47,769 support、20 列（从 trace 对应的逐 micro 记录读取）。
GPU activity span 约 7.108s，区间并集 busy 约 7.014s；kernel 累计 6.968s，
memcpy 累计 45.90ms（含各种方向复制，并非全部 H2D），memset 1.56ms。

| 归属 | 累计 kernel/activity 时间 |
|---|---:|
| COL 前向 | 216.25ms |
| ROW 前向 | 414.59ms |
| ICL 前向 | 1,673.72ms |
| 质心 route，含下层算子 | 1,191.59ms |
| 其中框架 gatherTopK | 878.13ms |
| centroid scores kernel，全 trace | 226.73ms |
| content_tree，全 trace | 54.63ms |
| ELU backward query，全 trace | 377.63ms |
| sparse backward KV，全 trace | 314.88ms |

前向三个模块的真实活动累计中，ICL 约 **72.6%**，是长表 GPU 端的重点。
内层 scope 是 inclusive、可能包含 checkpoint replay，不能与父 scope 或前向相加；
反向没有全部按模块 attribution，不能声称上述比例就是整个训练 step 的 ICL 占比。
框架 gatherTopK 单独约占全部 kernel 累计的 12.6%，明显大于 content_tree 的约 0.8%。
下一项 GPU kernel 优化更应审查 `_centroid_scores` 到 top-k 的中间张量／多次 selection，
尝试保持完全相同 top-k/tie 规则的融合或高效选择，而非优先重写树排序。
此前两种 top-k 中位数树分割方案已经比现有 segmented radix sort 慢，不能重复作为已知收益。
近似层级路由、跨层复用树或减少检索预算会改变算法，不当作等价优化。

汇总器识别并排除 `gpu_user_annotation` 镜像 CPU call 计数，使用 External id 及
CUDA driver/runtime correlation 关联直接 Triton launch。35,009 个活动中只剩 7 个无法关联，
共约 2.05ms；未归属时间明确保留。组件累计和 busy union 口径不同，不能相加成整步耗时。

## 同形状 cohort 的实现

```mermaid
flowchart LR
    S[公共 seed + 绝对 optimizer step] --> C[采样 rows / split / 请求宽度 / cap]
    C --> P[TaskPlanner：条件采样各表参数、固定 task ID]
    P --> G[生成服务：CPU permits、准入、独立图和重试]
    G --> T[独立传输层：numpy 或 mmap]
    T --> R[固定 rank 归属、相同数量 micro、DDP]
```

新增 `PRIOR_SCHEDULE=shape_cohort`。它在生成前为整个全局 optimizer step
共享 rows、train_size、请求 num_features 与 max_features；不是对已有随机表重新排序。
所有节点用同一个 `np_seed + absolute_step` 派生的隔离 RNG 得到条件，
无需增加形状广播 collective；图种子仍带 node offset，类别数、图结构和拒绝重试独立。
跨节点要求配置、公共 seed 和 step 一致，正如正常 DDP 对训练配置的要求。

架构职责没有混合：`_dataset.py` 定义单表原始形状分布及条件参数采样，
`_task_plan.py` 决定 step 条件和身份，`_scheduling.py` 分配完整 micro，
`_node_pool.py` 管理生成与固定归属交付，`_pool_transport.py` 继续负责存储／映射。
cohort 使用生成前分配，一个 rank 的全部本地表完成即可交付；**尚未实现逐 micro 流式交付**。

global batch=64 时，Stage2/3 的 8 rank 每卡累积 8 个 micro，16 rank 每卡累积 4 个；
每个 micro 一张表，平均梯度权重保持不变。Stage1 的 micro=4 也受支持。
服务不按完成顺序选择前缀，不丢慢表、不复制表；输出预算／lease／传输合同继续生效。
`prior_pool_window` 的单位是节点上的 local batch 数：128 在 8 local rank 时最多覆盖
16 个 optimizer step，在 16 local rank 时最多覆盖 8 个，实际还受 2 GiB 预算限制。
因此 8 卡的预取深度／性能不能原样推算为 16 卡结果；跨节点仍按各自 local rank 数准入，
同形状条件则由公共绝对 step 决定。

这里是“共享生成形状条件”：生成后删除常数列会使实际接受列数略有不同，
所以不能承诺最终 GPU 宽度完全一致。为强求相同宽度而重试或丢表会造成选择偏差，
补造特征也改变输入语义，本实现没有采用这些操作。

原始单表请求形状的边际分布保持，但 step 内联合分布明显改变：以前最多有 64 个形状样本，
现在整个 step 只有一个形状样本。因此不与原流逐表／逐步等价，也不承诺相同优化轨迹。
形状种子独立于图 RNG；恢复可复现绝对 step 的形状条件，in-flight 图流仍沿用原节点池的
“恢复后新流”语义，**不是精确数据重放**。收敛与验证集质量需要单独验收。

更具体地，固定模型参数，令单表梯度的条件均值为 `mu(shape)`、条件协方差为
`Sigma(shape)`。忽略表间其他依赖，原独立 B 表的平均梯度协方差为
`[Var(mu) + E(Sigma)] / B`，一步共享形状后为 `Var(mu) + E(Sigma) / B`。
所以保持单表边际只保证固定参数处的平均梯度，不保证梯度噪声或优化器轨迹等价。
这也是本轮没有直接替换正式默认调度的原因。

若后续质量评估发现相关性过强，可扩展为 K 个形状 cohort：每个 micro 位置跨 rank
共享形状，一步保留 K 个独立形状。在 global64、micro1 下，8 rank 可取 K=8，
16 rank 可取 K=4，各 rank 仍处理完全相同的一组请求形状；不必让整步都只有一种形状。
这会把形状梯度项改成约 `Var(mu)/K`，也能减轻整个长表 step 同时变重的突发。
当前提交实现的是 K=1；没有把这个后续方案标成已实现或已验证。

## 为什么不能预先承诺显著加速

对 128 个 step、global batch=64 的同一 MAC proxy，已有步内 balanced 到理想均摊的剩余比例为：

| Stage | 8 rank | 16 rank |
|---|---:|---:|
| 2 | 0.067% | 13.05% |
| 3 | 0.057% | 3.76% |

这是请求宽度上的计算模型，未拟合 content-tree 构建、实际 kernel 效率、反向、通信或编译；
16 rank 按同一节点内全局分配计算，不能当成两个 8 卡节点的结果。
它说明 Stage3 的 8 micro/rank 已能把大部分形状差异摊平，继续消除同一 proxy 的不均衡空间很小。
真实耗时模型不准确时，cohort 仍可能减少 DDP 等待和形状分支差异，必须实测。

首轮目标设备的廉价生成器 profile 对该模型给出了更具体的边界。去掉每个 rank 最后一个
可能含 DDP 尾部的 micro，balanced 的 1,680 个样本中，MAC 与单表训练 stream 时间的
Spearman 相关为 0.9925、Pearson 为 0.9770，说明排序大体合理；仅用一个吞吐常数换算
MAC 的平均绝对百分比误差仍约 24%，短表固定开销尤其明显。这不是 CPU 图生成成本模型。
对每步各 rank 的 COL/ROW/ICL 前向及 ICL/ROW 反向时间求和，再比较 max/mean，
balanced 均值约 1.153、cohort 约 1.041。这个子区间不含 COL 反向，亦非纯 kernel active time，
不能用来直接计算整步加速上限；但说明 MAC 上的 0.05% 不均衡并不等于实测不均衡。
下一版分派模型应拟合设备上的分模块延迟、短表启动成本和实际宽度，而不是任意扩大某个 MAC 系数。

cohort 也可能变慢：长行数条件会让整个 step 的所有 CPU 图一起变重，
GPU 需求与生成需求同时突增；有限预算无法无限预取；图内结构长尾不因共享外部形状消失。
因此正式默认仍保持 `balanced`，新增路径显式开启；不存在已验证的“最快 cohort 默认”。

## GPU profile 怎样判断真正瓶颈

`profile_training_pipeline.py` 输出逐 rank / step / micro 的：

- 等数据墙钟、单表 rows/split/实际列数、H2D 的 host 与 CUDA stream 时间；
- COL、ROW、ICL 前向，以及基于 Tensor 梯度 hook 的反向边界；
- frontend 前处理和模块间空隙、loss/backward setup、optimizer、整个训练 stream 与墙钟；
- 每步 CUDA peak（排除较大 synthetic warmup 的历史峰值）、OOM、loss、周期性池状态。

这些 CUDA events 记录 stream 上的边界，**包含 CPU 提交间隙，不等于 kernel active time**。
COL backward 的区间含 DDP 尾部及指标处理；异步通信可能在其他区间重叠。
可选全 rank Kineto trace 分离真实 kernel、memcpy、collective 和 GPU idle。
每步只在最后事件同步，不逐模块 synchronize；性能基准另跑无 profiler 的版本。
编译仍发生在正常方法内部，不给 backward module 添加 full hook。

汇总器分别报告 ICL 前向占三个前向模块之和、ICL 前反向占三个模块前反向之和；
后者明确标注包含 COL sync 分母，不拿它冒充整个 step 的 ICL kernel 占比。
逐 rank 的 wait+train 先配对再取最大值，不能把不同 rank 的 wait 最大值与 train 最大值相加。
单机 monotonic 时间可对齐；跨节点 monotonic 时钟不能直接拼接。
缺失 rank/step、缺少阶段事件、OOM 或 nonfinite loss 都拒绝成为有效 profile。

`summarize_kineto_trace.py profile/rank0-trace.json` 对实际 GPU activity 做区间并集，
另外报告 kernel / memcpy / memset 的累计时间；跨 stream 重叠不能简单相加成 GPU busy。
通过 Kineto External id 或 runtime/driver correlation 将 activity 关联到 COL/ROW/ICL 前向和 ICL 内部的 CPU scope，
组件范围是 inclusive，父子时间同样不能相加。没有 GPU activity 时明确报错，
不会把只有 CPU enqueue 的 trace 冒充 GPU kernel profile。

判断顺序：先看无插桩总吞吐和最长 rank，再看等数与训练的相关性，最后用 trace 找关键路径。
真实 prior 与廉价 `ShapePrior` 对照可以定位供给成本，但后者使用请求列数且没有真实重试，
不是完全相同 GPU 工作量的因果对照，更不能用于训练质量实验。

## 复现命令

正式脚本显式启用 cohort（checkpoint 与此前 stage 启动方式相同）：

```bash
PRIOR_SCHEDULE=shape_cohort bash scripts/train_v2_clf_stage3.sh
ADAPTIVE_FEATURE_LIMIT=False PRIOR_SCHEDULE=shape_cohort bash scripts/train_v2_clf_stage3.sh
```

CPU profiler 应独立串行运行，避免和训练抢核；在 8 local rank 的目标节点预留 8 核：

```bash
PYTHONHASHSEED=42 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/profile_graph_generation.py --stage 3 --tasks 512 --workers 56 \
  --pool-kernel-threads 8 --cpu-reserve 8 --feature-limit adaptive --output /tmp/cpu-crop.json
# 分别改为 --feature-limit none；再分别追加 --rows 60000 做固定长表对照。
# --cprofile-dir /tmp/cprofile 额外生成逐任务 Python 调用栈，但有插桩开销。
```

Linux 上可对已运行的矩阵 launcher 做低频、只读 CPU 采样，不修改训练代码：

```bash
python scripts/sample_training_resources.py LAUNCHER_PID --interval 2 --output /tmp/online-cpu.jsonl
```

先检查完整 GPU 矩阵，再启动（以下 Stage3 单 seed 为 12 个 case，各跑吞吐和 profile，共 24 次）：

```bash
RECOMPUTE=True python scripts/run_cohort_audit.py --gpus 8 --stages 3 --seeds 42 --steps 300 \
  --output /tmp/cohort-audit --trace --dry-run
RECOMPUTE=True python scripts/run_cohort_audit.py --gpus 8 --stages 3 --seeds 42 --steps 300 \
  --output /tmp/cohort-audit --trace
```

矩阵包括裁剪／不裁剪、真实图／廉价形状图、balanced／window_fifo／shape_cohort，
使用 content_tree、compile all、正常逐 rank warmup。profile 默认为 40 步、去前 20 步；
`--trace` 取一整步所有 rank 的 trace。吞吐另跑 300 步，记录包含编译启动与最终 checkpoint
的 wall 以及稳态时间。每次独立编译 cache，防止试验顺序改变冷启动成本。
默认 seed 为 42/43/44，Stage2/3 均运行；可加 `--dtypes bfloat16 float32` 和 `--gpus 16`。
每次输出必须是新目录，所有 case 串行执行。

默认独立随机初始化以避免 stage checkpoint 不存在导致实验无法启动；要评估真实权重，
用 `--initial-checkpoint /path/to/common.ckpt` 对所有 case 使用同一权重。
不同 stage 正式质量实验应分别给各自正确的前阶段 checkpoint。
cohort 每步只有一个独立形状，长尾形状在 300 步里仍可能覆盖不足；应检查形状直方图，
必要时延长并报告多 seed 置信区间，而非从一次短前缀的 p95 下结论。

## 正确性测试与优化优先级

新增测试覆盖 1/2/8/16 rank 的全局形状条件、分类／回归、跨节点公共条件、绝对 step 恢复、
8 组 5,000 样本的边际分布检查、独立类别 RNG、非法条件、真实图 numpy/mmap 传输、
极小输出预算、lease 清理、8/16 逻辑 rank 的真实服务交付。
逻辑 rank 测试的客户端在线程中运行；它不是 16 个 DDP 进程或 16 张 GPU 实验。
16 个真实 Gloo 进程、三 stage、各两次 optimizer 更新的 DDP／串行全局均值对照通过，
最大参数误差 5.96e-8。CPU 模型的 profile hooks 与原模型输出／梯度逐位一致。
最终调度／传输专项为 **202 passed、1 skipped**（CUDA pinning 不适用于本地）；
真实 stage 脚本参数与 profile 入口组合检查为 **12 passed、2 skipped**。

扩展 CPU 回归为 418 passed、181 skipped、7 failed；未修改基线同顺序复跑为
383 passed、181 skipped、**相同 7 项失败**，未把它们记作通过。
失败涉及缺失 Triton 的长 support attention、checkpoint context 重用及旧数据重放；
另两个必须导入 Triton 的测试模块无法在 macOS 收集。
目标 CUDA/vendor 环境最终完整回归为 **1202 passed、260 skipped、40 warnings，680.65 秒**。
skip 主要为不可用的 XPU/MPS（各 120 项）和明确条件跳过，未计为通过；
使用原测试要求的 checkpoint，禁用本机不支持编译的外部 FA3 扩展，仍运行 native CUDA 测试。
随后修正 trace 镜像计数和 Triton launch attribution 后，8 项 profiler 测试在本地及目标机单独复跑。
集成回 `/ossfs/workspace/tabicl-training-perf` 后，29 个交付文件逐一校验 SHA256，
cohort、stage recipe、prior pipeline 与 profiler 的 88 项专项测试再次全部通过。
16 卡性能及收敛验证尚未完成。

后续优化按实测关键路径决定：

1. 优先将大型非有限值 fallback 纳入同一 CPU 预算：task106 的配对实验已经定位到
   两个 1 线程距离调用占约 18.6 秒，正式借核路径覆盖不到。隔离强制 8 线程能从
   24.86s 进一步降至 7.39s，尚需生产预算实现、位级结果／重试一致性和在线多 seed 验证。
   EM 的直接距离调用也应复用预算，其总体占比约 6.7–7.9%，并非唯一热点。
   CPU 长尾与未来 step 的生成会竞争同一预算。再联合比较 48/56/61 worker、
   kernel 上限 8/16 的在线吞吐及长尾，判断为大型距离算子留出 permits 是否有益；
   不预先把减少 worker 或增大线程数称为加速。进一步可限制向 executor 提交的未来任务，
   为当前 step 的关键慢图优先留出预算；已经运行的图不能靠调整队列顺序直接抢占。
   对中间维度和 p 分布定点优化时检查完整 X/y、过滤与尝试次数，不能只比单个 kernel。
   再看 `std`／activation 与过滤器，不能把全部 graph 时间都算成 nearest-center。
2. GPU 端优先优化质心路由的 top-k／分块中间张量，长表 trace 的 gatherTopK 为 878ms，
   树本身只有约 55ms；保持路由结果、tie 与梯度规则，先比 kernel 再比全流水线。
   如果多数 rank 已有数据而 DDP 尾部很长，校准真实设备的 frontend/ICL 时间模型；
   比较 cohort 与原 balanced 的总吞吐。形状 proxy 上已均衡不代表 kernel 时间已均衡。
3. 若等待的是某 rank 最后一张表，考虑按固定 task ID 逐 micro 交付并重叠 CPU/GPU。
   保持各 rank micro 数、同步位置与梯度权重，不能改成谁先完成就训练谁。
4. H2D 预取、pinning 或 mmap 应在 trace 证明复制暴露在关键路径后做；
   毫秒级传输无法消除秒级图长尾，增加 worker/预算也不保证收益。
5. 已修复 warmup 使用不可生成的大形状：裁剪路径现在预热 60k×20、60k×1、30k×60，
   同时覆盖长表与中等长度宽表；不裁剪仍保留 60k×100，不降低模型数据规格来掩盖显存问题。
   三轮在所有 rank 并行，不是顺序执行 24 个 optimizer step，且没有参数更新。
   实测这一轮首次冷编译／预热约 253 秒，复用 cache 仍约 70 秒，都是实际启动成本；
   三个预热形状不覆盖所有编译 guard，不能保证后续没有编译 stall。短作业必须把
   启动、稳态和最终保存分别计入总时间，不能只报去掉 warmup 的 step 时间。
   后续充分命中 cache 的无重算裁剪路径约 27 秒，说明 70 秒也不是固定启动常数。

目标机器另发现已安装 FA3 扩展缺少 FakeTensor 实现，不能用于编译 frontend。
脚本在启用 compile 时默认 `USE_FLASH_ATTN3=False`，使用 native SDPA；这仍允许 SDPA 内部
选择可用的 fused 后端。有兼容外部扩展的环境可显式设置 `USE_FLASH_ATTN3=True`。

共享 permits 只限制本作业 worker 的并发，既不是 OS 空闲核探测，也不限制其他作业。
无空闲 permit 就继续单线程；不能保证有核可借。CPU worker、临时图 RSS、
编译进程与其他实验都应使用实际 affinity/quota 配置；输出预算不是整个作业的 RSS 上限。
