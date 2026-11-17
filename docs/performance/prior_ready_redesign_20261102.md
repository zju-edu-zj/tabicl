# 完整 micro 就绪调度与 categorical 潜变量边缘化

## 修改的边界

本轮在已有节点生成池内实现 `ready_micro`，没有增加 rank 接收线程、第二个生成池或额外预取层。仍由原 TaskPlanner 产生任务、原 ProcessPoolExecutor 执行、原 NumPy/mmap 传输、原输出字节预算控制驻留。

生成算法新增 `categorical_sampling=marginal`。图结构、全部函数族、转换器、树深度/数量、GP frequencies、完整 N 行拟合和正式过滤都保留。它消去的是只影响一次类别选择、之后不再使用的随机权重向量，不是用 cheap 函数替代 SCM。

## 就绪调度：任务身份、执行、交付分开

TaskPlanner 先抽取完整 source step 的元数据并分配连续 task ID。每个任务的 seed 仍由 `[pool_seed, task_id]` 决定，先于完成顺序；池内执行顺序也未改动。

一个 unit 是 source step 中原本连续的完整 micro-batch。例如 global batch=64、8 ranks、micro=4 时，每个 source step 有 16 个 unit，每个 rank 最终拿两个 unit。只有 unit 内四张表全部完成，它才可交付；不拼接来自不同 micro 的表，不改变 micro 内共享 N/support 的关联。

每次组成下一训练步：

1. 先找必须在本步交付的 unit。
2. 如果其中有未完成的 unit，等待原任务，不丢弃、不重采样、不降低复杂度。
3. 其余槽位由最老的完整就绪 unit 补足，允许从后续 source step 借用。
4. 用实际生成宽度估算 GPU cost，在这 16 个完整 unit 之间做原有等槽位 LPT 分配，每 rank 仍是两个 micro。
5. 所有 rank 按同一 dispatch step 索引领取自己的固定结果；快 rank 无法抢走慢 rank 的分配。

并不是“挑最快的任务永远训练”。设 source step 为 s，交付 step 为 t，等待上界为 D，则必须 `t <= s + D`。早交付受已有 source window 限制。D=0 退化为 source-step 保序；D=2 允许暂时绕开慢 unit，但它到期仍没完成就阻塞。

这是训练步计的最大延期，不是墙钟超时。如果任务本身一直不返回，保留它和保证墙钟等待上界不能同时成立；本实现选择保留任务。worker 报错沿用池的异常传播机制，不把失败伪装成成功数据。

### 公平性证明与有限训练的边界

每个 source step 恰有 K 个 unit，每个 dispatch step 也消耗 K 个。第 t 步开始时，之前 deadline 已到的 unit 已由归纳假设交付；本步新增到期来源最多只有一个，因此到期 unit 不超过 K 个。算法先保留全部到期 unit，再填其他就绪 unit，保证所有已完成/最终完成的任务在其 deadline 前交付，且每个 unit 只从待交付集合移除一次。

在固定训练步数的末尾，最近 D 个 source step 可能只交付了部分任务；较早来源已全部交付。生成池本来就会在退出时留下未消费的预取工作。本实现没有把“最终都交付”夸大成“有限前缀恰好等于原始前缀”；变化是有界的训练顺序和跨 micro 的同一步组合，优化轨迹不会逐位一致。

不同 source 的 micro 在同一步出现，因此跨 micro 的相关性改变；每张表及 micro 内的生成过程不变。DDP 每 rank 的样本数、micro 大小、累积次数和 loss 权重不变。不能由此推出最终模型精度已被长训验证。

### 资源与所有权

source window 同时计算生成中、就绪、部分交付的来源批次。ACK 只释放本 rank 已映射的表及其预留字节；源批次要等所有表 ACK 后才移除。mmap 文件解除名字后，客户端仍持有的映射继续有效。

跨来源组合会出现“一个来源已经全部回收，但同一步另一个 rank 尚未领取”的情况，因此本步全局生成就绪时间在组装时缓存，后续不能再读取已回收来源。相关回归测试覆盖这一交错。

`PRIOR_READY_MAX_DELAY` 必须非负，且小于有效 source window `max(1, PRIOR_POOL_WINDOW // LOCAL_WORLD_SIZE)`。池窗口不足一个 source 时，沿用已有的一次整 cohort 入场规则，允许 D=0。字节预算很小可能使实际 lookahead 只有一个 source，此时功能仍正确，但自然没有绕开慢任务的空间。输出预算不是图计算临时内存的总预算。

## 生成算法：积分掉只使用一次的类别权重

### 原算法实际生成的分布

`GlobalSampler.categorical(name, K, mode='meta')` 在默认 legacy 模式中查询 `globals[name]`，却把权重写到 `globals[name + '__meta_weights']`。因此同一个名字的下次调用仍会构造全新权重，并没有复用它。这个历史行为在仓库中已有 `use_corrected_cat_meta_sampling` 开关。

每次 fresh `SimpleRandomWeights(Context())` 会：

1. 建立独立 Context/GlobalSampler。
2. 抽 polynomial decay 和 Gaussian noise 的参数，包含 numerical meta 参数与 beta 抽样。
3. 构造 K 个 logits，加 Gaussian noise。
4. 对 logits 做均匀随机排列。
5. 做 softmax、L2 归一化，再按总和归一化为类别概率。
6. 依照这组概率只抽一次类别，权重之后被下一次覆盖。

排列使概率向量 P 可交换。因此对任意 k，`E[P_k]` 相同，又因为 `sum(P_k)=1`，所以 `Pr(C=k)=E[P_k]=1/K`。fresh 权重与同一主 Context 的 numerical meta 参数独立，且不跨次复用；每次调用可以直接均匀抽类别。这是将不可观察的单次潜变量积分掉，保留类别序列的联合分布，而非凭经验把一个非均匀 prior 改为均匀 prior。

仓库内搜索确认权重缓存只在 GlobalSampler 内使用，没有其他组件读取该 `globals` 权重，也没有把 `__meta_weights` 当作显式采样名称。外部代码如果直接读取这些内部缓存，边缘化模式不保证该内部状态兼容。

### 必须保留的相关性

`use_corrected_cat_meta_sampling=True` 下，第一次按名字抽权重，以后重复使用，类别之间存在共享概率造成的相关性。即使每次单独观察仍是均匀边缘，整个序列也不是独立均匀。因此新实现完整保留该路径，不做边缘化。

`mode=global`、`mode=local`、配置为非 meta 的委派路径、以及预先存在的 legacy 名字缓存也保持原行为。数值 meta 的 beta 超参数仍按名字缓存；真正用于矩阵、节点 importance、聚合等计算的 RandomWeights 仍完整抽样。

去掉潜变量会改变 RNG 消耗和同 seed 的数据内容，但不缩小函数族、图复杂度或理想随机源下的任务分布。正式过滤也没有改变；既然候选联合分布相同，拒绝采样得到的接受分布也相同。有限 PRNG、浮点和有限样本审计不构成逐位一致或最终模型精度证明。

## 验证方法

策略层用随机完成排列检查完整 micro、每个 task 恰好一次及最大延期；构造慢任务验证既能绕开，又会在 deadline 强制等待。真实 spawn 池覆盖多 rank、NumPy/mmap、小于单批输出的预算、错误传播、慢 rank 交错回收。D=0 与旧 balanced 逐步比较实际生成数据内容集合。

采样器检查均匀频率和重复名字的相邻类别相关性；缓存/修正模式与旧实现做同 seed 完全一致对照；用禁止构造权重的替身检查 fresh 分支确实消除了工作。

完整 prior 基准包括全部拒绝重试与过滤，以独立进程做 ABBA；GPU 基准使用真实 Stage-1 脚本、全部模型、8 ranks、COMPILE_FRONTEND=all、逐 rank 前后向预热，丢弃前 20 步，按同步日志的 elapsed 及逐步 max-rank 时长汇总。不能把算子加速或 CPU 表/秒当作 GPU 训练加速。

分布审计使用完整接受任务，比较两组独立 seed 下 legacy 的自然波动，再与 marginal 比较；指标含接受图节点/边/深度、原函数族覆盖、类别数/少数类/标签熵、Ridge 与 64-tree ExtraTrees 的实际 train/test 划分留出 balanced accuracy。bootstrap 区间以任务为单位。这些是分布及难度代理，没有代替下游长训验证。

审计的 `attempts` 统计实际物化的 `RandomGraphFunction` 次数，不包含在物化前已被祖先关联检查拒绝的 DAG。CPU 基准的完整 wall/CPU 时间则包含这些结构重采样的成本。逐 rank 的 `train_s` 包含 DDP 通信及等待，不能直接解释成该 rank 独占的 GPU 算子时间。

## 复现实验

```bash
export PYTHONPATH=src
export TABICL_PRIOR_DISTANCE_LIBRARY=/path/to/tabicl_prior_exact_cpu.so
python scripts/run_prior_ready_experiments.py --phase cpu --tasks 1024 --output /tmp/prior-results/cpu
# 训练前加载本机 SDK 环境；CPU 基准与训练串行运行。
python scripts/run_prior_ready_experiments.py --phase training --steps 120 --output /tmp/prior-results/training
python scripts/run_prior_ready_experiments.py --phase delay --dtypes bfloat16 --steps 240 --output /tmp/prior-results/training
python scripts/audit_prior_categorical.py --tasks 512 --workers 8 --rows 2048 --seeds 61 62 --output /tmp/prior-results/audit/n2048.json
python scripts/audit_prior_categorical.py --tasks 512 --workers 8 --rows 4096 --seeds 61 62 --output /tmp/prior-results/audit/n4096.json
python scripts/summarize_prior_ready_experiments.py --root /tmp/prior-results --output /tmp/prior-summary.json
```

原始逐任务、逐步日志保留在 `/tmp/tabicl-ready-results-20261102`，提交汇总 JSON，避免把大量训练日志加入源码。机器为 8×PPU-ZW810，CPU quota 约 69.76 cores；完整 CPU prior 基准使用 8 workers，训练节点池使用 56 workers。对照基线为 `7dc336f`，所有运行使用 full 25-tree dataset filter 和已有 native CPU kernels。

## 完整 prior 的 CPU 实测

每格 1,024 个最终接受任务；每种长度按 legacy → marginal → marginal → legacy 顺序串行运行，包含启动、拒绝重试和正式过滤。N=2048 使用 seed=54，N=4096 使用 seed=55；同模式重复的任务流固定，新旧模式不要求逐表相同。

| 行数 | legacy 两次耗时 (s) | marginal 两次耗时 (s) | 合并吞吐提升 | 每任务 CPU 工作量下降 |
|---|---|---|---|---|
| 2048 | 64.060 / 64.031 | 52.555 / 52.649 | 21.75% | 19.55% |
| 4096 | 95.616 / 95.096 | 87.048 / 87.557 | 9.22% | 7.62% |

N=2048 两次平均 graph 工作时间之和从约 367.8 s 降到 271.6 s，filter 时间从 107.8 s 变为 110.5 s。收益来自图生成，不是少做过滤。N=4096 新模式反而多触发了图/过滤重试（1,409→1,449 次图生成，1,263→1,295 次正式过滤），仍然更快。这些是完整任务流的实测收益，不能直接当成 GPU 训练提升，也不能保证每个 seed 都有相同提升。

新模式 N=2048 的每个接受任务平均消耗约 0.373 CPU 秒。粗略按 56 个单线程生成 worker 满负荷估算，生产 64 张表已经需要约 0.426 s，尚未计入管理、传输及长尾造成的空闲。这只是由 8-worker 基准外推的容量估计，不是 56-worker 扩展效率的测量；但它解释了为什么调度不能无限消除约 0.45–0.50 s/step 的供给瓶颈。当前 graph 仍占 graph+filter 计时的约 71%，下一轮算法工作应优先继续消除图生成中可积分、可合并的工作；不能仅凭过滤次数多就认定减少过滤是最大收益来源。

## 分布与任务难度审计

每种长度、每种模式各 1,024 个最终接受任务（seed=61/62 各 512），总计 4,096 个任务。所有输出有限、训练/测试划分均覆盖实际类别，去常数列后的常数列比例均为 0。所有八个原始完整函数族均有覆盖，没有减少函数族或启用新的 cheap 替代。原实现内部既有的 `cheap_random_function_fct_type` 调用也保持原比例机制，不是本轮新增。

下表为 marginal − legacy 的按任务配对差值，括号是对任务重采样 2,000 次的 95% bootstrap 区间。它检验明显分布漂移；区间包含 0 不等于证明分布等价，等价判断来自前述潜变量分析。

| 指标 | N=2048 差值 [95% CI] | N=4096 差值 [95% CI] |
|---|---|---|
| 图节点数 | −0.532 [−1.201, 0.104] | +0.192 [−0.414, 0.818] |
| 图边数 | −1.970 [−7.988, 3.722] | −0.330 [−6.623, 5.737] |
| DAG 最大深度 | −0.169 [−0.584, 0.227] | +0.018 [−0.390, 0.406] |
| 接受表宽度 | −0.022 [−0.361, 0.310] | −0.109 [−0.443, 0.227] |
| 最少类比例 | +0.0003 [−0.0073, 0.0075] | +0.0040 [−0.0030, 0.0113] |
| 归一化标签熵 | +0.0080 [−0.0058, 0.0227] | +0.0088 [−0.0063, 0.0237] |
| 图尝试次数 | +0.042 [−0.031, 0.114] | −0.023 [−0.083, 0.034] |
| Ridge 留出 balanced accuracy | +0.0142 [−0.0062, 0.0340] | +0.0132 [−0.0078, 0.0344] |
| ExtraTrees 留出 balanced accuracy | +0.0191 [−0.0038, 0.0417] | +0.0137 [−0.0109, 0.0378] |

没有发现这些指标的明确漂移，但统计功效有限，尤其没有排除几个百分点的难度变化。这不是最终 TabICL 模型的长期泛化验证；短训练只检验可运行性、数值稳定性与吞吐。

八类完整函数的使用比例也作了描述性比较：N=2048 时旧模式两 seed 间的 total variation 为 0.0116，同 seed 新旧模式分别为 0.0130/0.0162；N=4096 对应 0.0126 与 0.0104/0.0142。每个函数族的比例均约 12%，与旧模式的随机波动处于相近量级。图内多次选择存在相关性，因此没有把节点数当作独立任务数来夸大统计显著性。

## 真实 8-rank Stage-1 训练

本组为 N=2048、global batch=64、micro=4、每 rank 两个 micro，未缩小模型。每次独立启动 120 个 optimizer step，稳态统计去掉前 20 步。两次 baseline 与两次 ready/legacy 按 ABBA 顺序，随后独立重复两次 ready/marginal。节点池 56 workers、window=128、输出预算 2 GiB；D=2。BF16 的既有默认 DataLoader 另外实测，不能把节点池 `none` 冒充 BF16 默认配置。

| 路线与配置 | 两次稳态 ms/step | 两次 step p95 (ms) | 两次完整 wall (s) |
|---|---|---|---|
| BF16 node_pool/none/legacy | 652.38 / 680.44 | 971 / 1,009 | 144.79 / 148.59 |
| BF16 ready_micro/legacy | 583.52 / 607.17 | 1,641 / 2,100 | 141.17 / 143.78 |
| BF16 ready_micro/marginal | 440.47 / 557.76 | 1,202 / 1,820 | 118.57 / 134.18 |
| FP32 node_pool/balanced/legacy | 614.39 / 617.69 | 1,374 / 1,637 | 149.94 / 152.79 |
| FP32 ready_micro/legacy | 614.07 / 638.61 | 1,380 / 1,952 | 149.10 / 155.43 |
| FP32 ready_micro/marginal | 515.50 / 499.65 | 580 / 562 | 134.25 / 129.32 |

合并两次的稳态结果：

| 比较 | 原 ms/step | 新 ms/step | 吞吐变化 | 120-step 完整 wall 缩短 |
|---|---|---|---|---|
| BF16 节点池只改交付 | 666.41 | 595.34 | +11.94% | 2.88% |
| BF16 节点池组合方案 | 666.41 | 499.11 | +33.52% | 13.85% |
| FP32 只改交付 | 616.04 | 626.34 | −1.64% | −0.60% |
| FP32 组合方案 | 616.04 | 507.58 | +21.37% | 12.94% |

完整 wall 包括启动、编译/预热、训练和退出回收，不是只报 GPU 执行段。各轮是独立训练进程，但没有人为清空现有编译缓存，因此 wall 也不代表全冷缓存的首次编译成本。三个前后向预热在每个 rank 上执行，没有 optimizer 更新，也未消耗 prior；8 ranks 各三次不等于串行消耗 24 个训练 step。上述完整 wall 已包含这项成本。

BF16 默认 DataLoader（8 workers/rank，in_order=False）单次对照：legacy 670.96 → marginal 513.37 ms/step，稳态吞吐 +30.70%；wall 192.34 → 168.64 s。单次结果的证据弱于 ABBA，不能据此认定 ready 比 DataLoader 稳定快多少。ready/marginal 的两次均值仅比 DataLoader/marginal 这一次快约 2.86%，小于运行间波动。

### 公平期限确实可能造成 stall

D=2 只改调度时，BF16 平均吞吐提高，但两次最慢 step 达 4.35/3.88 s，明显差于节点池旧模式的 1.21/1.16 s。组合 marginal 后改善，但仍有 2.17/3.05 s 的最慢 step。这里没有承诺“预取让 stall 消失”：到期的慢任务必须保留，消费之前积累的快任务后仍会等待。

因此额外用 BF16、完整组合方案做 240-step 的 D=2→8→8→2 对照，检查更长 lookahead 是否把提前消耗的工作量转成可用的后台时间，同时记录第 201 步的真实池诊断。

| 运行顺序 | D | 稳态 ms/step | step p95 (ms) | 最慢 step (ms) | 完整 wall (s) |
|---|---|---|---|---|---|
| a1 | 2 | 457.45 | 1,629 | 3,587 | 175.47 |
| b1 | 8 | 488.43 | 682 | 2,392 | 183.06 |
| b2 | 8 | 562.60 | 816 | 6,284 | 200.49 |
| a2 | 2 | 515.63 | 1,825 | 3,904 | 191.18 |

合并均值 D=2 为 486.54 ms，D=8 为 525.52 ms，D=8 吞吐反而低 7.42%，完整 wall 多 4.61%。两次 step p95 的均值虽然从 1,727 ms 降到 749 ms，但最差 step 反而变长。结论是 D=8 在这一配置中改变了等待形态，没有获得整体吞吐收益，不将它作为推荐默认。

第 201 步的实际最大延期分别达到设定的 2/8/8/2，均未越界；这四轮最大提前量均为两步。D=2 的公平等待检查约 3.2–3.4 万次，D=8 为 266–559 次，这只是条件检查次数，不是停顿事件数或毫秒数。池内有效 source window 为 16，峰值输出预留约 0.43–0.47 GB，低于 2 GiB 字节预算；因此本组不能简单归因于输出字节上限。D=8 仍有公平上限，但调度无法创造额外的 CPU 生成能力。

本轮共 18 次真实八 rank 训练、2,640 个 optimizer step，全部正常退出，记录中 OOM micro-batch 和非有限 CE 均为 0。它们是短训练性能/数值检查，不是最终模型质量评估。

### 逐 rank 时间应如何解释

FP32 baseline 两次逐步 `max_rank_prior` 的均值为 132.93/135.99 ms，组合方案为 37.11/35.17 ms；`max_rank_train` 仍约 476–490 ms。因而本组主要是消除了 prior 暴露在训练关键路径上的等待，模型执行开始成为下一瓶颈。

BF16 `none` 允许不同 rank 先取先训练，因此一个 rank 的 prior 等待可能表现成另一个 rank 在 backward/DDP 内等待。例如 baseline a1 中各 rank 的平均 prior 时间分布在 56–227 ms，而 ready_micro 的 b1 变成约 253–260 ms。不能把“rank prior 更整齐”直接视为总耗时减少，也不能把独立的 `max(prior)` 和 `max(train)` 相加：它们可能来自不同 rank，且通信等待发生重叠。本报告同时记录同一步 `max_i(prior_i + train_i)`，最终以同步 elapsed/step 为吞吐依据；汇总 JSON 保留所有 rank 的均值、p95 和最大值。

## 测试覆盖

远端 SDK/native 环境通过 331 个用例，覆盖本轮两个新测试文件及执行、native kernels、训练 recipe、CPU budget、prior pipeline 的回归。另在本地 CPU 环境通过 166 个既有节点池、window、scheduling 和 filter cascade 用例。合计 497 个测试用例通过，来自两组环境，不能误写为全部在远端通过。

最终代码同步到 `/ossfs/workspace/tabicl-training-perf` 后，再在该目录的远端 SDK/native 环境复跑两个新增测试文件：142 passed，80.28 s；同时通过 `git diff --check`。重复验证不计入上述 497 个不同用例。

其中新测试共 142 个用例：120 组随机完成排列；慢任务绕行及 deadline 强制等待；全部 task ID 恰好一次；完整 micro；小字节预算；真实 spawn 多 rank 的 NumPy/mmap；部分 ACK 回收；源批次先释放、慢 rank 后领取的交错；错误传播；D=0 与旧 balanced 的逐步生成内容集合相同；恢复参数；类别频率/相关性及修正缓存路径兼容。另通过源码编译检查和 shell 语法检查。

## 启用与选择

本轮没有修改各阶段既有的默认后端/调度，也没有默认启用 marginal；两项均可独立选择。`PRIOR_CATEGORICAL_SAMPLING=marginal` 同时适用于 DataLoader 和节点池。`PRIOR_SCHEDULE=ready_micro` 只对节点池有效。

FP32 可以使用本轮实测的组合方案。以下命令为真实训练，需按计划设置独立 `CKPT_DIR`，避免实验意外恢复旧目录；不是 benchmark 脚本：

```bash
cd /ossfs/workspace/tabicl-training-perf
source /usr/local/PPU_SDK/envsetup.sh
export PATH=/opt/conda/bin:$PATH
export TABICL_PRIOR_DISTANCE_LIBRARY=/tmp/tabicl-redesign-build-20261101/tabicl_prior_exact_cpu.so

TRAIN_DTYPE=float32 PRIOR_BACKEND=node_pool PRIOR_SCHEDULE=ready_micro \
PRIOR_READY_MAX_DELAY=2 PRIOR_CATEGORICAL_SAMPLING=marginal \
COMPILE_FRONTEND=all COMPILE_WARMUP=True \
CKPT_DIR=/ossfs/workspace/tabicl/checkpoints/stage1-ready-marginal-fp32 \
bash scripts/train_v2_clf_stage1.sh
```

上述 native 库路径是本机已有构建。换机器应先在对应 SDK/PyTorch 环境执行 `python scripts/build_prior_exact.py --build-directory /tmp/tabicl-prior-build`，再将环境变量指向该次输出的 `.so`。不要在每个生成 worker 中编译。

BF16 希望最少改变现有数据路径时，可先只启用 marginal：

```bash
TRAIN_DTYPE=bfloat16 PRIOR_BACKEND=dataloader PRIOR_SCHEDULE=none \
PRIOR_CATEGORICAL_SAMPLING=marginal COMPILE_FRONTEND=all COMPILE_WARMUP=True \
CKPT_DIR=/ossfs/workspace/tabicl/checkpoints/stage1-dataloader-marginal-bf16 \
bash scripts/train_v2_clf_stage1.sh
```

若需要节点池的完整 micro 就绪交付，则使用第一条命令并将 `TRAIN_DTYPE` 和独立 checkpoint 目录改为 BF16。它已通过真实训练，但目前证据不能证明其稳定优于 DataLoader/marginal，更不能承诺每个 step 都消除 stall。
