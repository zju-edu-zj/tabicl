# 节点共享池、单数据集调度与长表算子优化

日期：2026-10-22。实现适用于在线 CPU `graph_scm`。本次优化对象是数据生成和供给，模型的稀疏/线性 attention 架构及 hybrid 参数分组没有改变。

## 结论与适用边界

三个改动已经接入分类、回归的全部六个 stage 脚本：节点只创建一组生成 worker；每个任务生成一个数据集；输出保持紧凑，到训练 micro-batch 才 padding。另优化了常量列检测、bootstrap、随机树和离散化距离算子。

本地同步双 rank 测试中，共享池的稳态供给吞吐提高约 2%–5%，p95 最慢 rank 等待降低约 11%–31%。不能据此宣称训练速度有同等提升：模拟每步 550 ms 计算时，两条管线均已基本隐藏生成耗时。完整生成器的同种子配对测试收益约 3%–7%，也远低于部分独立算子的倍数。

本地是 i9-14900K（32 逻辑核）、RTX 5090、PyTorch 2.9.1+cu130、Python 3.12 的共享机器。没有在用户的 128 核、16 GPU 训练节点上执行测试；112 worker 是针对该机器的起始配置，尚不是实测最优值。

## 调度方式及其语义

实现位于 `src/tabicl/prior/_node_pool.py`，训练入口在 `src/tabicl/train/_run.py`。

```mermaid
flowchart LR
    A[节点中央参数采样器] --> B[有界 batch 窗口]
    B --> C[单数据集任务队列]
    C --> D[节点共享的 CPU workers]
    D --> E[按原分组重组完整 batch]
    E --> F[先完成的 batch 队列]
    F --> G[本节点任意 DDP rank]
    G --> H[切 micro-batch 后 padding]
    H --> I[GPU 前向、反向、hybrid 更新]
```

旧 DataLoader 的一个 item 是一个 rank 的完整 local batch，worker 顺序生成其中的数据集。一个慢图会占住 worker，也会拖住该 item 的其余工作。`in_order=False` 只能绕过尚未完成的 item，不能让某个 rank 借用其他 rank 已生成的数据。

新管线由 local rank 0 启动 manager，所有本地 rank 连接同一个 Unix socket。manager 创建固定数量的 spawn worker；CUDA 初始化后也不 fork 训练进程。任务粒度变成单个数据集，先完成的完整 local batch 可交给任意本地 rank，减少 worker 和 rank 间负载不均。这里“共享池”指共享生成服务和 worker；数组通过 NumPy/pickle IPC 传输，并非零拷贝共享内存实现。

保留了原 `GraphPrior` 的参数采样层次：group 共享长度及 train/test split，subgroup 共享特征数，各数据集分别采样类别数。先完整采样原 local batch 的参数，再拆成任务，完成后按原位置重组，不能把独立采样的数据集任意拼成 batch。图检查、分类 split 检查、ExtraTrees 检查及其 rejection loop 均保留，不设置超时过滤，也不在运行中丢弃慢图。

执行顺序和随机数映射会改变，因此它不是旧 DataLoader 的逐步数据重放。ready-first 在有限训练前缀内也可能暂时偏向先完成的数据；不能把“不丢慢图”表述为每个 optimizer step 的数据分布完全相同。训练正常进行时慢任务继续执行，最终完整 batch 仍会入队；结束时尚在窗口内的任务会被清理。

### 紧凑输出与延迟 padding

`GraphSCM(defer_padding=True)` 只省略最终零列填充。常量列清理后只传输 `[seq_len, actual_features]`，不在 manager 内构造 NestedTensor。训练端先切 micro-batch，再对该 micro-batch 的最大行数、最大实际特征数 padding。

特别保留了 ExtraTrees 的旧输入宽度：送入过滤器前临时补到原 `max_features`。零列会参与 ExtraTrees 的随机特征选择，直接删除会改变接受/拒绝决策。这部分不能作为无语义变化的省内存手段。原 `align_micro_batch` 本来就按实际特征数裁剪，新的延迟 padding 与其最终模型输入一致。

例如一个 batch 包含 `[32, 32, 60000]` 行，micro-batch 大小为 2。前两个数据集只 pad 到 32 行，不再为后面的 60K 表创建无用的整批中间张量。legacy dense 输入保留切片视图，避免破坏 pinned storage。

### 限流、失败与续训

- `prior_pool_workers`：每节点 worker 总数，不能再乘 GPU 数。
- `prior_pool_window`：同时保留的 ready + in-flight local batch 数；它不是 worker 数或数据集数。
- `prior_pool_mb`：按 `4 * seq_len * (max_features + 1)` 预留最终 float32 输出空间。允许一个超预算 batch 单独执行，以免卡死。它不限制 worker 的解释器、GraphSCM/ExtraTrees 临时内存、RPC 副本和 rank 已取走的 batch。
- worker 固定一个 Torch/BLAS 线程，避免 112 个进程再次各自创建多线程计算池。
- worker 异常附带 traceback 传到 rank；正常结束等待各 rank 释放服务，随后清理进程和 socket；manager finalizer 也会清理 worker。
- 模型、优化器、scheduler checkpoint 格式不变。现有 hybrid checkpoint 可正常续训，无需只加载模型。新池启动前用当前恢复步数偏移 seed，避免每次恢复重新消费 seed 42 的同一段流；不保存队列/RNG 快照，不保证数据逐步重放。`prior_seed_by_step=True` 与该后端不兼容，入口显式拒绝。

每 200 步 rank 0 的 metric 增加 `prior_pool/ready_batches`、`buffered_batches`、`reserved_bytes`、`peak_reserved_bytes`、`completed_datasets`、`delivered_batches`。这是节点服务状态，不能把 rank 0 的取数等待当成全部 rank 的等待。

## 算子方案与数值边界

1. 常量列检测：每列排序求 unique 改成与首行比较后沿行 `any`，复杂度从排序转为线性扫描。保留单行、NaN、inf 的原语义。
2. bootstrap：保留 `default_rng(0)` 的相同抽样顺序和每个 bootstrap 样本的相同归约。小索引缓存最多两项，大数组按约 8 MiB 索引/聚集预算分块；不减少 bootstrap 次数，不近似 p-value。
3. 随机树：只沿行分块，约束每个 worker 的比较、索引、叶值 gather 中间张量；树维度归约顺序不变。避免 60K 行一次 gather 上 GiB 的叶值。兼容真实参数采样返回的 NumPy 整数。
4. 最近中心：一般维度沿行分块计算 `cdist`，不保留整张长表距离矩阵。一维正 p 范数可按 `abs(x-c)` 选最近中心，去掉每个点对的幂运算。p=2 保留原矩阵乘法路径；不支持的 dtype/device/p、极端值及接近舍入平局的行使用原 `cdist`。

第 4 项不能只凭数学等价替换：真实图中遇到约 `1e-18` 的数，`cdist` 的幂运算会下溢，将不同距离都算成 0，按首索引选中；绝对值选取可能得到不同索引。实现已补上极小非零距离检测，并加入回归用例。随机 Stage 1 压测的 100 个完整数据集、3941 次距离调用与原索引一致；完整生成器配对验证如下。它是已覆盖边界的浮点优化，测试不能构成对所有输入/所有 PyTorch 版本逐值等价的数学证明。

暂不改过滤强度、图节点数、函数族、categorical meta-sampling 或接受策略。这些改变确实可能更快，但会改变 prior，不能与本次算子优化混合后把收益归因于实现。

## 性能结果

### 独立 CPU 算子

单线程，reference/optimized 分别在全新 spawn 进程运行；先验证输出，再预热一次、计时五次取中位数。RSS 每 1 ms 采样，是进程增量峰值，不等于张量精确分配量；短暂峰值可能漏采。

| 算子 | 行数 | 原实现 ms | 优化后 ms | 加速 |
|---|---:|---:|---:|---:|
| 常量列检测，100 列 | 1,024 | 3.298 | 0.399 | 8.26× |
| 常量列检测，100 列 | 10,240 | 39.263 | 7.298 | 5.38× |
| 常量列检测，100 列 | 60,000 | 311.207 | 43.198 | 7.20× |
| bootstrap，200 次 | 1,024 | 0.660 | 0.173 | 3.81× |
| bootstrap，200 次 | 10,240 | 9.773 | 7.759 | 1.26× |
| bootstrap，200 次 | 60,000 | 51.510 | 33.284 | 1.55× |
| 随机树，128 棵 | 1,024 | 5.628 | 5.197 | 1.08× |
| 随机树，128 棵 | 10,240 | 126.166 | 56.121 | 2.25× |
| 随机树，128 棵 | 60,000 | 681.753 | 475.734 | 1.43× |
| 一维最近中心，128 中心 | 1,024 | 2.655 | 0.708 | 3.75× |
| 一维最近中心，128 中心 | 10,240 | 29.679 | 8.262 | 3.59× |
| 一维最近中心，128 中心 | 60,000 | 161.343 | 47.065 | 3.43× |

60K 随机树的进程增量峰值由约 1082 MiB 降至 34.8 MiB，bootstrap 从 183.2 MiB 降至 7.4 MiB。一维 fast path 的舍入保护会分配辅助布尔张量，小表未必比原路径省内存；不要把所有算子都描述成同时降低延迟和内存。

原始数据：[prior_operators_20261022.json](benchmarks/prior_operators_20261022.json)。

### 完整生成器同种子配对

每个长度 40 个接受后的数据集；使用 Stage 1 的全部图/过滤参数，显式将行数固定为 1K/10K/60K。reference 恢复旧的 unique/bootstrap/tree/cdist，双方均用 compact 输出以隔离算子差异；每个 seed 交替运行顺序，完整执行所有 rejection loop。x、y、d 均逐值一致（`rtol=atol=0`）。这不是 Stage 2/3 长度混合分布的配对测试。

| 行数 | 原生成器总秒数 | 优化后总秒数 | 加速 |
|---|---:|---:|---:|
| 1,024 | 18.61 | 18.02 | 1.032× |
| 10,240 | 126.24 | 118.33 | 1.067× |
| 60,000 | 248.49 | 234.13 | 1.061× |

因此独立算子很快，并不意味着整个生成器快数倍。剩余图函数、图结构、采样和 ExtraTrees 仍消耗大部分时间。原始数据：[prior_generation_20261022.json](benchmarks/prior_generation_20261022.json)。

### 同步 rank 供给矩阵

2 个 CPU/Gloo rank，总计 8 个生成 worker。DataLoader 每 rank 4 worker，`spawn`、prefetch 8、`in_order=False`；共享池 8 worker、window 32、输出预算 512 MiB。每 rank batch=4，预热 16 步，测量 100 步。每次 fetch 前和模拟消费前均 barrier，末尾再次 barrier，比较相同步数的总耗时；两后端均包含上述算子优化。DataLoader 也是 ready-first，不能把表中收益归为从 ordered 切换至 unordered。

| 配置 | DL 测量总秒数 | 池测量总秒数 | 吞吐增益 | DL / 池 p95 最慢 rank 等待 ms |
|---|---:|---:|---:|---:|
| Stage 1，纯供给 | 38.47 | 37.51 | 2.6% | 1093 / 759 |
| Stage 2，纯供给 | 67.53 | 64.47 | 4.7% | 1482 / 1314 |
| Stage 3，纯供给 | 141.75 | 138.86 | 2.1% | 3613 / 2992 |
| Stage 1，每步消费 550 ms | 55.59 | 55.40 | 0.3% | 5.8 / 3.8 |

含启动和预热的总时间分别为：Stage 1 `46.65 / 46.46 s`，Stage 2 `79.55 / 80.84 s`，Stage 3 `172.78 / 171.52 s`，550 ms 消费 `66.99 / 68.11 s`。小样本短跑中启动成本会抵消收益，Stage 2 的含启动总时间甚至略慢；长训练只启动一次。没有重复实验置信区间，2%–5% 的小幅收益不能视作稳定硬件结论。

两后端消费的是各自随机流和完成前缀，而不是相同数据集；例如 Stage 2 rank 0 平均行数约 3230 / 2989。长尾和有限前缀差异限制了吞吐比较的因果解释，不能据此推算 16 rank 的确定加速倍数。JSON 中保留 `length_mean`、`features_mean`、payload、等待分位数以及含预热总时间，以便审计。

Stage 1/2/3 rank 0 平均 IPC payload 分别约从 `1.59/5.03/10.51 MiB` 降到 `0.80/2.41/5.49 MiB`，但新池增加一个约 640 MiB 的 manager 进程。整个作业的逐 rank 峰值 RSS 求和反而约多 0.6–0.7 GiB；它含共享页重复计数，不是 USS/PSS，也不能证明节点总内存下降。

额外固定 60K 行测试：双方各 rank 预热 2、测量 5 步，均正常完成；总测量时间 `7.37 / 6.05 s`，p95 等待 `2128 / 2247 ms`。样本太少且特征数不同，只作为长表稳定性检查，不用于声称吞吐或尾延迟改善。

全部十组原始 JSON：[prior_pipeline_20261022/](benchmarks/prior_pipeline_20261022/)。

## 正确性与训练测试

已运行：

- 原始管线改动回归：722 passed、256 skipped、40 warnings，215 秒。后续算子/准入修改后再次运行：723 passed、256 skipped、40 warnings，238.6 秒。两次均因当前环境缺少 `statsmodels` / `shap`，显式排除了两个不相关的 forecast/SHAP 测试文件；不是无条件的“全部测试通过”。
- spawn 双客户端访问同一组 worker PID，检查参数分组、输出尺寸、队列/字节预算、异常传播、关闭后 PID 消失和 socket 清理。
- compact/full 分类及回归、过滤开启/关闭的同种子逐值比较；dense/nested/list 三种输入的 micro-padding 比较，包含 60K 与短表混合。
- CPU/CUDA 上真实模型的 compact/nested/dense 前向、反向、hybrid 更新、checkpoint 保存及恢复后的下一步更新一致性。
- 单卡 CUDA 实际 CLI：node_pool + bf16 + global_sparse + hybrid 完成 4 步并从 checkpoint 恢复到第 6 步；验证了非零学习率和实际参数更新（第 1 至 4 步最大参数差约 `3.80e-4`）。小模型 smoke run 不能用作正式 Stage 1 吞吐。
- 1K/10K/60K 独立算子和完整图生成配对测试，另有下溢平局等数值边界回归。
- 六个 curriculum 脚本参数 dry run，以及 shell 语法检查。

复现回归套件：

```bash
conda activate stdplm
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  --ignore=tests/test_datetime_encoder.py --ignore=tests/test_shap_categorical.py
```

## 训练节点运行命令

以下均在 `tabicl` 仓库根目录执行。分类 Stage 1 沿用现有 hybrid 目录即可从最新 checkpoint 恢复优化器/scheduler，不要设置仅加载模型：

```bash
conda activate stdplm
NUM_GPUS=16 \
CKPT_DIR=/ossfs/workspace/tabicl/checkpoints/tabiclv2-clf/stage1_hybrid_from20k \
PRIOR_BACKEND=node_pool PRIOR_POOL_WORKERS=112 \
PRIOR_POOL_WINDOW=128 PRIOR_POOL_MB=2048 \
bash scripts/train_v2_clf_stage1.sh
```

实际目录不同则替换 `CKPT_DIR`。六个 stage 脚本均读取 `scripts/prior_recipe.sh`；Python CLI 的 backend 默认仍为 dataloader，防止其他调用方隐式换后端。需要旧后端作对照时：

```bash
NUM_GPUS=16 CKPT_DIR=/your/existing/hybrid/run \
PRIOR_BACKEND=dataloader PRIOR_WORKERS=7 \
PRIOR_PREFETCH_FACTOR=8 PRIOR_IN_ORDER=False \
bash scripts/train_v2_clf_stage1.sh
```

这只是后端切换对照命令；不要同时启动两个进程写同一个 checkpoint 目录。

训练节点 CPU 供给矩阵（无需占 GPU，模拟 16 个同步消费者；两个后端总 worker 均为 112）：

```bash
conda activate stdplm
PYTHON="$CONDA_PREFIX/bin/python" NUM_RANKS=16 \
WORKERS_PER_RANK=7 POOL_WORKERS=112 POOL_WINDOW=128 POOL_MB=2048 \
STEPS=500 WARMUP=32 LONG_STEPS=20 \
OUTPUT_DIR=prior_benchmarks/node128 \
bash scripts/benchmark_prior_matrix.sh
```

回归版矩阵增加 `REGRESSION=True`。本地本次命令使用 `NUM_RANKS=2 WORKERS_PER_RANK=4 POOL_WORKERS=8 POOL_WINDOW=32 POOL_MB=512 STEPS=100 WARMUP=16 LONG_STEPS=5`。

独立算子/生成器测试：

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmark_prior_operators.py --output prior_operators.json
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/benchmark_prior_generation.py --datasets 40 --output prior_generation.json
```

## 后续算子与准入调度复核

在 2026-10-22 的后续复核中，先尝试了多维的 PyTorch 广播距离 + 在线 argmin。虽然独立随机张量的部分配置略快，但真实固定 60K 图生成的 8 个样本中该路径耗时 66.95 秒；profile 显示 93 次多维调用耗时 44.4 秒，其中 `pow` 占 33.5 秒。原 `cdist` 在同一 profile 只占 6.6 秒。因此没有保留多维手写距离归约；多维继续用 `cdist`，按行分块以限制输出矩阵峰值。这个结果说明单算子工作量/形状分布不同，不能用均匀张量微基准代替完整 prior 配对。

保留的一维路径按中心排序，每个输入只检查插入点左右的两个候选，复杂度从 O(NC) 降为排序中心加 O(N log C) 查找。距离接近平局或数值极端时回退至 `cdist`，以减少浮点舍入改变离散中心选择的机会。60K、128 个中心的单算子基准：`cdist` 165.7 ms，排序查找 2.8 ms，索引匹配率 100%。多维 8/32 特征分别约 407/1194 ms，与 reference `cdist` 基本持平；收益主要是分块控制距离输出峰值，而非缩短计算。

固定 60K 分类 prior 的 12 个同种子数据集配对（包含 rejection loop）均逐值一致：reference 91.39 秒，当前 82.65 秒，约 1.11x。样本数仍不足以给长尾收益置信区间；此数字仅代表该次 CPU 测试。4 个样本的小基准波动较大，不作为结论。

共享池准入现在维护一个有界参数 batch 候选窗口。在当前输出预算放不下队首 batch 时，会扫描后续候选，先准入能放入预算的 batch；候选均无法放入时仍等待预算释放，池完全空时允许单个超预算 batch 保证进展。每个原始 local batch 内的参数相关性与成员顺序不变；不同 batch 之间可能越序，随机流不再对应 optimizer-step 顺序。没有丢弃、超时替换或拆散 batch。

池统计增加准入候选数、绕过次数、受预算阻塞次数，以及最近最多 4096 个 worker 生成时长的 P50/P95/P99。生成时长不含参数队列等待、worker 执行队列等待、manager IPC 或 rank 取数等待。最终准入逻辑的 2-rank、4-worker Stage 3、window 16、128 MiB、50 测量步压力测试记录 79 次候选绕过；但 ready queue 最终耗尽，最慢 rank 取数等待 P95/P99 为 4.92/5.52 秒，worker 生成时长 P99 为 12.62 秒。另一次同配置短测为 5.17/6.37 秒和 16.82 秒，但消费的数据流不同，不能据此认定调度优化带来确定的尾延迟降幅。该测试刻意只有 4 个 worker，不与此前 8-worker 基线作速度比较；它证明准入逻辑在工作，也证明准入不能解决生成能力不足和单任务长尾。

新的尾延迟复测应在训练节点比较足够长的测量窗，并同时保存逐步最慢 rank 等待和池状态。worker 时长分位数可以辨别长任务，但不能代替实际 GPU idle 时间；若 P99 仍高，需进一步 profile 难生成图、过滤拒绝率及真实 worker 并发，而非继续扩大候选窗口。

## 后续阶段的选择

先在实际训练节点测量 64/96/112 worker 和不同窗口，比较相同步数含启动总时间、稳态吞吐、最慢 rank 等待及节点内存，不以某一步 `prior_time=0` 判断成功。window 的目的在于跨慢任务保留调度余地，不是无限预生成数据；输出预算过低会压缩并发。112 个 worker 的 Python/Torch 基础内存就可能达到几十 GiB，尚未计图中间量。Stage 3 应按实际内存适当减少 worker，而不是简单扩大队列。

若稳态 `ready_batches` 长期为 0 且所有 worker 忙，池已供给不足，应继续 profile GraphSCM 和 ExtraTrees；加 manager 层无法消除这些计算。若队列常满且等待已接近零，进一步 CPU 微优化不会改善当前训练速度，应转向真实训练的 attention/优化器 profile。若长表 IPC 已成为明显占比，再实现有容量及生命周期管理的共享内存环形缓冲；当前证据不足以承担零拷贝复杂度。

对于 GPU attention，本次没有引入架构切换、dense 阈值或新的近似算法；需要用实际长表的前向+反向 profile 单独决策，不能从本报告的生成器数据推断收益。
