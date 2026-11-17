# 训练加速与每卡任务分配：2026-10-30

后续审计、有限窗口实现和受控对照见 [2026-10-31 报告](prior_window_20261031.md)。
下文保留当时结果；新报告补充了同任务回放、数据面隔离和 Stage2/3 实验。

本次在远程 ZW810 节点运行实际训练，不以成本估计的改善代替训练收益。
节点有 8 张 96 GiB PPU-ZW810，vendor PyTorch 2.8、Triton 3.4；宿主可见
192 逻辑 CPU，但容器 CPU quota 约 69.76 核。实验池使用 56 个单线程 worker。
16-rank 检查使用 CPU/Gloo；该节点没有 16 张物理卡，不能把它称为 16 卡吞吐测试。

## Stage1 与精度路径

`train_v2_clf_stage1.sh` 默认 `MAX_SEQ_LEN=2048`、`COMPILE_FRONTEND=all`、
`COMPILE_WARMUP=True`。FP32/BF16 共享脚本，通过 `TRAIN_DTYPE` 选择。
共享 `performance_recipe.sh` 传递策略，Trainer 根据设备、实际计算精度和最大行数解析：

| 硬件/精度 | 最大行数 | auto 策略 |
|---|---:|---|
| ZW810 / FP32 | ≤2048 | MATH SDPA |
| ZW810 / FP32 | >2048 | 仅 ROW MATH，其余保留原分派 |
| BF16，或其他硬件 | 任意 | 原生分派 |

显式覆盖为 `ATTENTION_BACKEND=native/fp32_math/fp32_row_math`。
FP32 脚本默认禁用外部 FA3，避免其把 FP32 attention 转为 FP16；显式冲突配置会报错。
MATH 仍然是 FP32，但内核及浮点累加顺序不同，不保证逐位相同或训练轨迹完全相同。
旧整模实验在 2048 行时 all-MATH 约 720 ms、native 约 1016 ms；4096 行 all-MATH
在 96 GiB 卡上 OOM，ROW-only MATH 约 1615 ms、native 约 1978 ms。
这些是特定 micro-batch/模型/软件栈的结果，不能外推到所有设备。

这里的“修改分派”是限制 `scaled_dot_product_attention` 可选的实现，不改 attention 公式。
此安装版本公开的后端是 MATH、FLASH_ATTENTION、EFFICIENT_ATTENTION 和 CUDNN_ATTENTION；
硬件、dtype、mask、stride 与 head dimension 决定哪些实际可用，枚举存在不代表该卡支持。
外部 FA3 是项目包装函数的另一条路径，不等于这些原生枚举中的一个。
MATH 将 attention 拆成矩阵乘法/softmax 等算子；融合后端旨在减少中间张量和访存，
但在该 vendor 的 FP32 内核、这些形状上，实测融合路径反而较慢。
因此这里采用硬件限定策略，而不把“融合一定快”或“MATH 一定快”当作通则。
2048→4096 不只是算力问题：MATH 会显式物化部分 attention 中间张量，
全模型峰值显存已经越过本机容量。FP32 路线保持 dtype；累加顺序差异与
原先 FA3 的 FP32→FP16 转换是两种不同的数值影响，本次将后者禁用。

每 rank 独立持有编译缓存，因此所有 rank 同时做三次前向和反向，最后 barrier。
它们没有 optimizer/scheduler 更新，不取 prior，不计入训练步数，并恢复 RNG、buffer 和模块模式。
8 rank × 3 不等于串行执行 24 个训练步。此节点已观察到约 30–31 秒启动预热时间；
预热不是免费的，短作业必须比较包含启动的总时间。动态形状/stride/分支仍可能产生后续编译。
stage2/3 的长表范围很宽，保留各自原来的编译/预热默认值；可显式启用再测。

## 为什么固定每步任务集合

共享池从 `sparse_prior` 引入，并保留单数据集生成、紧凑输出、micro-batch 才 padding
以及既有图/数据过滤。生成 worker 在节点内共享，不再为每张卡创建独立的 worker 池。

支持三种模式：

| `PRIOR_SCHEDULE` | 生成与交付方式 | 适用目的 |
|---|---|---|
| `none` | 完整 local batch 按完成顺序交付任意 rank | 原共享池、供给吞吐对照 |
| `round_robin` | 固定每步节点任务集合，保持原连续 local batch 分配 | 隔离均衡效果的对照 |
| `balanced` | 同一集合内重新分配完整 micro-batch | 均衡每卡训练工作量 |

Classifier Stage1 根据实测选择默认值：FP32 使用 node_pool，且每卡有多个 micro-batch 时为
`balanced`，只有一个时为 `none`；BF16 默认 DataLoader。
BF16 显式设 `PRIOR_BACKEND=node_pool` 时默认 `none`。始终可以显式覆盖。
共享 recipe 在其他 curriculum 脚本默认 `balanced`，DataLoader 则默认 `none`。

先按原 local batch 的 GP/subgroup 层次完整采样，采样本节点所有 rank 的任务组成一个 cohort。
每个数据集的 identity/seed 在执行顺序调整之前确定；生成优先启动预计较重的 `rows × requested_features`
任务。生成后用实际接受的列数、行数、support 数及模型配置估计 COL/ROW/ICL 工作量。
`feature_group="same"` 保留列数；只有 `valid` 才减少列 token 数。

均衡器先按成本降序、带等数量限制的 LPT 分配，再做能降低负载峰值的成对交换。
若结果比原连续分配的成本峰值更差则回退。它适用于任意 rank 数，既不是两卡特例，
也不是只把最重和最轻任务配对。成本是 operation proxy，没有假装是精确 GPU 毫秒数。

只移动完整 micro-batch，不拆 GP、不增删/重复任务、不跨 optimizer step 搬任务。
每卡始终获得相同数量的数据集和 micro-batch，因此 loss accumulation 和 DDP rank mean 的
全局权重保持一致。先取完数据的 rank 通过 `(rank, cohort_sequence)` 请求下一步，不能抢走别卡任务。
不同顺序会改变浮点累加；若启用 dropout，任务与 rank RNG 的对应关系也会改变，不能声称逐步逐位续训等价。

| global batch=64 | micro-batch | 8 卡每卡单元 | 16 卡每卡单元 |
|---|---:|---:|---:|
| Stage1 | 4 | 2 | 1 |
| Stage2/3 | 1 | 8 | 4 |

Stage1 的 16 卡配置每卡只有一个不可拆单元，单纯置换不能降低最大值。
如需在此配置继续均衡，应另做 `micro_batch=2/1` 的 kernel 效率和吞吐对照，或增加 global batch；
后者会改变优化配置，需要另行验证。本次保持用户原来的 batch 语义。

固定 cohort 也会产生供给队头阻塞：必须等待这一步的全部生成结果才能按实际宽度均衡。
增加预取能隐藏部分等待，但不能保证抹平过滤/rejection 长尾。`none` 允许完成顺序不同，
因此它与另外两种模式的有限训练前缀并不是同一组任务；只有 `round_robin` 对 `balanced`
可以严格隔离“相同每步任务、仅分卡不同”的影响。

## 资源限制与复现

`PRIOR_POOL_WORKERS=0` 自动取 `max(1, floor(affinity 与层级 cgroup quota 的最小值) - local_world_size)`。
本机 8 卡约为 61，16 rank 约为 53；不是根据宿主 192 CPU 直接启动 176 个 worker。
所有 worker 的 Torch/BLAS 线程数为 1，且在 spawn import NumPy/SciPy 前就设置环境变量。
`PYTHONHASHSEED` 也在 spawn 前固定：只 seed NumPy/Torch/Python random 不足以稳定哈希容器的遍历顺序。
测试比较了不同 worker/调度执行顺序下完整 X/y 的 SHA256，未只比较 shape。

`PRIOR_POOL_WINDOW=128` 以 local batch 为单位；固定模式使用 `max(1, window // local_ranks)`
个 cohort。`PRIOR_POOL_MB=2048` 限制预留最终输出字节，分类标签按 int64 计费。
为避免死锁，允许一个超预算 cohort 单独执行；它不限制 GraphSCM/ExtraTrees 临时内存、
解释器、RPC 副本或 rank 已取走的数据。长表仍需结合节点内存控制 worker 数。
固定模式不为内存 admission 跳过前面的 cohort，否则会改变 optimizer step 的任务集合。

多节点时每节点各一个池，只做节点内均衡，不是跨节点全局调度。
恢复 checkpoint 时使用恢复步数偏移生成 seed，不保存 in-flight 队列，不能逐步重放 prior。
模型/优化器/scheduler checkpoint 格式保持兼容。`prior_seed_by_step=True` 与 node_pool 显式不兼容。

实际训练比较示例（每次使用新目录，避免自动 resume）：

```bash
PRIOR_BACKEND=node_pool PRIOR_SCHEDULE=balanced PRIOR_POOL_WORKERS=56 \
RANK_TIMING_EVERY=1 TORCHINDUCTOR_CACHE_DIR=/tmp/tabicl-inductor-ddp-bf16 \
/tmp/tabicl-perf-venv/bin/python scripts/benchmark_training_schedule.py \
  --gpus 8 --steps 160 --dtype bfloat16 --output /tmp/tabicl-example-balanced
```

只改 `PRIOR_SCHEDULE=round_robin/none` 做池内对照，或设 `PRIOR_BACKEND=dataloader`。
后者自动选择 `PRIOR_SCHEDULE=none`。16 卡运行设 `NUM_GPUS=16`；benchmark 工具则是 `--gpus 16`。
stage2/3 使用 `--stage 2/3`，并设置真实前阶段的 `STAGE1_CKPT/STAGE2_CKPT`。
工具的 `--train-args` 可在脚本默认值后追加 Trainer 参数。

`RANK_TIMING_EVERY=100` 抽样记录各 rank 的 prior/train host wall time；分析实验设为 1。
训练时间包含 DDP 等待、拷贝和 optimizer，并非独立 GPU kernel 时间。
不能把不同 rank 的 prior_max 与 train_max 直接相加当作单步时间；汇总工具额外按 rank
先相加再取最大值，并报告稳态 elapsed/step 和完整进程 wall time。
逐步 all_gather 和日志也有开销；对照双方必须一致。

## 测量解释

首轮 8 卡 Stage1 配对，含编译和三次预热、最终 checkpoint；省略前 20 步统计训练段：

| 精度/步数 | 分配 | 最慢 rank 训练段均值 ms | 全程 s |
|---|---|---:|---:|
| FP32 / 120 | round_robin | 873.1 | 192.45 |
| FP32 / 120 | balanced | 717.4 | 173.79 |
| BF16 / 160 | round_robin | 371.9 | 206.87 |
| BF16 / 160 | balanced | 311.1 | 200.41 |

这轮使用修正 feature-group token 估计前的 proxy。FP32 训练段约减少 17.8%，
端到端约减少 9.7%；BF16 训练段约减少 16.4%，端到端只有约 3.1%。
BF16 首轮有短暂 CPU 测试干扰，因此还需要独占顺序复测，不据此承诺长期加速幅度。
全程时间不能与省略前 20 步的训练段比例混用。相同任务分配对照也不是收敛实验。

修正 proxy 后，完整测试结束再顺序运行 FP32 120 步对照，省略前 20 步：

| 分配 | 最慢 rank 训练段均值 ms | 稳态 elapsed/step ms | 全程 s |
|---|---:|---:|---:|
| round_robin | 879.3 | 885.3 | 195.59 |
| balanced | 717.9 | 725.7 | 176.15 |

训练段减少 18.4%，稳态每步减少 18.0%，全程减少 9.9%；两个重复对照的方向一致。
这是相同步任务集合的实际 8 卡训练，并非仅按 proxy 预测；仍然不是长期收敛试验。

修正 proxy 后的 BF16 顺序复测（160 步，统计省略前 20 步）：

| 供给/分配 | 最慢 rank 训练段均值 ms | 稳态 elapsed/step ms | 全程 s |
|---|---:|---:|---:|
| DataLoader | 669.3 | 657.9 | 217.20 |
| node_pool / none | 685.0 | 695.2 | 181.70 |
| node_pool / round_robin | 340.0 | 699.2 | 196.05 |
| node_pool / balanced | 302.1 | 685.4 | 194.80 |

均衡在相同任务集合的比较中使训练段减少 11.2%，但稳态只减少约 2.0%，
完整耗时只减少 0.6%。固定 cohort 的 prior 等待 p95 仍约 1.9 秒，最大约 7 秒。
共享池 none 的完整耗时比 DataLoader 少 16.3%，但此短测稳态并未改善；
供给方式间的有限前缀任务也不同，不能据此宣称长期吞吐改善 16.3%。
默认 BF16 不为这点均衡收益付固定 cohort 的队头阻塞代价。

随后将 DataLoader 和 node_pool/none 都延长到 640 步，省略前 100 步统计：

| 供给 | 最慢 rank 训练段均值 ms | 稳态 elapsed/step ms | 全程 s |
|---|---:|---:|---:|
| DataLoader | 663.7 | 657.7 | 537.82 |
| node_pool / none | 809.6 | 820.9 | 601.16 |

共享池全程反而增加 11.8%，稳态每步增加 24.8%；160 步时的启动优势没有转化为长期收益。
因此最终 Stage1 BF16 默认改回 DataLoader，保留共享池用于显式实验；
不能因为共享池架构或较短测试好看就把它设为默认。
这是不同有限任务前缀的供给路线比较，不是相同任务的纯调度因果对照。
DataLoader 为每 rank 8 个 worker（节点共 64），池为节点 56 个；这是实际 recipe 对照，
不是等 worker 数的架构隔离实验。16 卡若选 DataLoader，原 `PRIOR_WORKERS=8` 会启动
128 个生成进程，应按该节点 CPU quota 显式调整，不能把本机 8 卡最佳路线直接外推。
所有 Stage1 短/长测沿用 5000 步学习率 warmup，未修改优化训练配方；
这里只验证训练执行和吞吐，不代表模型完成预训练或质量等价。

Stage2/3 的短测使用原范围和 micro_batch=1，BF16、无额外编译、禁用外部 FA3，
验证原生 vendor SDPA 路径；都从同一架构的短 Stage1 checkpoint 只加载权重，
用于性能/执行正确性，不能当作已完成 curriculum 的模型质量验证。

| Stage / 实际步数 | 分配 | 最慢 rank 训练段均值 ms | 稳态 elapsed/step ms | 全程 s |
|---|---|---:|---:|---:|
| 2 / 24 | round_robin | 1094.3 | 1406.9 | 89.20 |
| 2 / 24 | balanced | 990.5 | 1399.1 | 84.39 |
| 3 / 12 | round_robin | 2505.3 | 2977.1 | 125.45 |
| 3 / 12 | balanced | 1778.7 | 2335.0 | 117.76 |

分别省略前 4/2 步。Stage2 的训练段改善被供给等待抵消，稳态基本持平；
Stage3 短测稳态减少 21.6%，方向明确但只有 10 个测量步，不作为长期幅度保证。
全部完成最终 checkpoint，均无 OOM micro-batch 或非有限 loss。

## 正确性检查范围

单元测试覆盖任意 rank 数与每卡单元数、等数量/不丢不重、micro-batch 边界、
相同全局梯度权重、CPU quota、固定任务 seed、慢客户端、极小 byte budget 下的 FIFO admission。
真实 16-rank CPU/Gloo 检查使用小型 TabICL 和实际 GraphPrior，比较 DDP accumulation
后的一次 SGD 参数更新与完整全局任务串行参考，覆盖三个 stage 的 micro-batch 形态。
长表供给检查与小模型梯度检查分开报告，不能互相替代。

实际 16-rank/Gloo 三阶段检查已运行 `balanced` 和 `round_robin` 两种模式。
使用真实 GraphPrior、真实小型 TabICL、DDP `no_sync` accumulation，
与串行全局平均参考的一次 SGD 更新比较，六项最大参数绝对误差均不超过 `2.98e-8`。
它验证等权重、collective 次数与执行正确性，不代表完整架构的 16 卡显存或吞吐。

另用 16 个 rank、32 个生成 worker、真实 Stage2/3 行数范围、micro-batch=1，
预热 2 步后检查 8 步长表供给。两阶段均完成；每步最慢 rank 取数等待均值约
1.65/3.09 秒，p95 约 8.38/15.67 秒。消费者仅用 550 ms 延迟模拟，
所以这是长尾/队列压力检查，不能当作实际训练速度。它说明固定 cohort 的
过滤与生成长尾仍是下一步优先优化的对象。

编译/FP32 参考测试显式禁用外部 FA3，匹配 Stage1 的运行配置。
否则安装了 FA3 的环境会在直接构造模型的测试中把 FP32 attention 转为 FP16，
而该 vendor 扩展也缺少 fullgraph 所需的 fake-tensor 支持。
本次保持原有数值阈值，并未通过放宽误差断言让测试通过。

完整测试在上述 vendor 环境执行 `python -m pytest -q -rs`，结果为
**1067 passed、260 skipped、40 warnings、0 failed，640.20 秒**。
跳过项为不可用的 XPU/MPS（249 项）、未安装 Docker（3 项）、未启用
`SCIPY_ARRAY_API`（4 项），以及 sklearn 对非确定性估计器原有的跳过（4 项）。
测试使用官方 classifier/regressor v2 checkpoint，本地缓存并通过
`TABICL_CHECKPOINT_DIR` 提供；没有用随机权重替代需要预训练模型的检查。
640 步 BF16 对照后，仅调整 Stage1 的 backend 默认选择；再运行 prior pipeline
与 attention/warmup recipe 检查，结果为 **30 passed、0 failed，22.65 秒**。

机器可读汇总见 `docs/benchmarks/training_schedule_20261030.json`。
远程保留原始 run.json、config、逐步 metrics 与验证日志于
`/ossfs/workspace/tabicl-perf-audit-20261030/current/`（归档不复制大型 checkpoint）。

进一步收益应优先看 prior 长尾与供给/训练重叠，再看成本 proxy 的 GPU 实测校准。
16 卡 Stage1 的一单元限制需要改变执行粒度才能突破；不能靠排序算法承诺解决。

后续实验应按下面顺序推进（这些是后续方向，未作为本次已实现收益）：

1. 记录单数据集生成时间、过滤/rejection 次数和算子耗时，区分 CPU 总产能不足与
   个别长尾。优先继续优化距离计算、ExtraTrees 过滤的执行与临时内存；保持
   采样分布、过滤阈值和接受规则，不通过丢弃慢任务换取速度。
2. 用相同已生成任务做 GPU replay，采集每个 micro 的 CUDA event 时间和实际列宽/
   行数/support，校准 FP32/BF16 各自的成本模型。现有 host `train_time` 包含 DDP
   等待，不能直接作为单 micro 的独立计算成本训练模型。
3. 分别测试 worker 数与输出预算，报告全程、稳态、内存、等待 p95/p99。
   当前自动值是尊重 quota 的起点，不是证明过的最优值；同节点其他作业也会影响资源。
4. 若固定 cohort 仍有显著队头阻塞，可研究“先按请求形状分配、结果到达后只做
   安全交换”的早交付策略，或生成/过滤与传输的流水线。必须继续保证每步任务集合、
   等数量、DDP collective 次数与 loss 权重，且实际计入调度/IPC 开销。
5. 16 卡 Stage1 另测 `MICRO_BATCH_SIZE=2/1`、global batch 不变；用更细分配带来的
   均衡收益对照更多 kernel launch、较低 batch 效率和 accumulation 开销。
   Stage2/3 已是 micro=1，不能再依赖拆 micro；需要解决供给或单任务计算长尾。
