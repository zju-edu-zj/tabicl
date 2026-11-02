# Prior CPU 优化与 Stage 1 默认配置

后续完整森林专用实现、独立质量探针及逐任务长尾诊断，见 [完整森林与长尾报告](prior_native_forest_20261103.md)。本页的旧实验数字保留其原始配置，不能与后续实现的收益重复累计。

当前 Stage 1 的 `optimized/sufficient/full` 入口默认还启用 `TABICL_PRIOR_FOREST=native`，保留完整 25-tree/depth-six/bootstrap/OOB 机制。它减少完整 prior CPU 工作约 2.3%，但新增 800-step×8 轮 ABBA 尚未确认稳定训练加速（FP32 −0.11%、BF16 +0.90%，均在波动范围内）；不要将森林热点快 18%–22% 当成整步加速。使用 `TABICL_PRIOR_FOREST=sklearn` 可回退；其他 curriculum stage 不切换。native 森林依赖当前 Torch C++ ABI，不依赖 sklearn 私有树 ABI，也无需新 Python 包。完整数据与限制见后续报告。

## 一键启动与安装

在已配置的远端进入 `/ossfs/workspace/tabicl-training-perf`：

```bash
bash scripts/train_v2_clf_stage1_fast.sh
```

包装脚本自动激活现有 `/usr/local/PPU_SDK/envsetup.sh`，优先使用 `/opt/conda/bin/python`，然后执行原 Stage 1 脚本。普通 CUDA 机器没有该 SDK 文件时直接继续；可用 `PYTHON` 和 `PPU_SDK_SETUP` 指定自己的解释器与 SDK。首次启动在 torchrun 之前编译 CPU 扩展，后续使用仓库内 `.cache/prior-native/` 的缓存。缓存按源码、Torch 构建、C++ ABI、Python SOABI、CPU 能力和编译选项隔离，不携带其他机器的预编译二进制。

新环境先安装适配硬件的 PyTorch/SDK，再安装其余依赖：

```bash
/opt/conda/bin/python -m pip install -r requirements-prior-performance.txt
# 编译器未安装时，Debian/Ubuntu 使用：apt-get install build-essential
```

依赖文件包括 ninja、NumPy/SciPy/sklearn、训练与测试所需 Python 包，故意不指定或安装 torch，以保留厂商版本。sklearn 私有调用适配稳定 1.3–1.9，其他版本回退公开接口；详见上一轮充分统计量报告。编译器、ninja 或依赖缺失时，启动在创建训练 worker 之前明确报错。安装命令应在 SDK 已激活的同一个解释器环境执行。

扩展构建使用 [PyTorch 2.8 官方 cpp_extension 接口](https://docs.pytorch.org/docs/2.8/cpp_extension.html)：`load(..., is_python_module=False)` 注册 dispatcher 算子，`CXX` 选择编译器、`MAX_JOBS=1` 限制编译并发。这里使用常规 ATen/C++ ABI，按 Torch 构建隔离缓存，不把一个二进制宣称为跨 Torch 版本通用。

主目录首次启动验证还发现并修复了冷缓存问题：`redirect_stdout` 只重定向 Python 输出，不能阻止 ninja 子进程向 fd 1 写编译命令；shell 捕获的库路径因此混入日志。现在构建期间同时重定向底层 fd，stdout 只输出一个真实库路径。新增测试在含空格的独立缓存目录实际编译、再次复用，并分别核验 stdout、动态库和 manifest；编译日志保留在 stderr。

环境变量覆盖仍然有效，例如：

```bash
TRAIN_DTYPE=float32 bash scripts/train_v2_clf_stage1_fast.sh
CKPT_DIR=/path/to/new/run MAX_STEPS=1000 bash scripts/train_v2_clf_stage1_fast.sh
DRY_RUN=1 bash scripts/train_v2_clf_stage1_fast.sh
```

默认 checkpoint 路径沿用原 Stage 1 脚本；已有 `step-*.ckpt` 时自动恢复。做独立实验应指定新的 `CKPT_DIR`。`DRY_RUN=1` 只输出命令，不导入 Torch、不编译扩展、不启动训练。直接运行原脚本也会自动准备扩展，但调用方应先激活 SDK 并选好 Python。

不要继续导出旧 `TABICL_PRIOR_DISTANCE_LIBRARY`。默认自动选择当前源码的缓存；显式指定库时启动器会检查新算子并执行统计计算，旧库缺符号会要求取消该变量后重建。支持显式使用原路径进行消融：`TABICL_PRIOR_STATISTICS=torch TABICL_PRIOR_COMPACT=sort`。

Stage 1 的配置以这台节点的真实 prior 训练为依据：默认 BF16、8 ranks、global batch=64、micro=4、2048 行，56 个节点共享生成 worker；`node_pool/ready_micro`，窗口 128 个 rank-local batch、内存预算 2048 MiB。BF16 公平性最大延后默认 8 步，FP32 保留 2 步；小窗口或更多 ranks 时自动限制为 cohort 窗口减一，显式 `PRIOR_READY_MAX_DELAY` 仍优先。两种精度都采用 `COMPILE_FRONTEND=all` 和每 rank 三种形状的前后向预热；生成池内核线程为 1，训练 OMP/MKL 线程为 2。参数仍可通过环境变量覆盖，不把该节点测量的最优值当作所有硬件的通用最优值。

## 本轮实现与分布约束

完整行去重在多列输入上从 Torch 排序改为原生哈希；单列保留更快的 Torch unique。先对小块完整坐标的子集做成本探测，再按每行全部坐标建立哈希；哈希冲突逐坐标核对，正负零按相等处理。只复用真正相同输入行的独立距离/EM 计算，最后通过 inverse 恢复原行序；没有近似检索、删除维度、减少中心或采样训练行。重复不足时停止并返回原输入。p=2 的尺寸相关 GEMM 路径、非有限输入及其他不支持的形状保留原路径。

标准化使用两遍全表统计：第一遍按列累计均值，第二遍围绕该均值累计平方偏差；累加器使用 double，输出仍为 float32，保留 `correction=0` 和 `std+1e-4`。它避免逐列跨行的 strided reduction，适合多列行主序矩阵。单列采用原 Torch reduction，因为微基准显示新实现反而慢；float64、需要梯度、非 CPU 和 reference 图执行也保留原路径。拟合均值/方差只计算一次，后续 `ensure_iid` 调用复用已拟合状态。

没有更换随机函数或降低图复杂度。full 25-tree 正式过滤、200 次 bootstrap、拒绝重试和全部请求保留。新统计的舍入差异可能经离散化或筛选阈值改变少数最终任务；不要求生成逐位一致。验证包括完整任务分布、拒绝序列和独立 holdout 探针，不能用短训练 CE 比较 prior 质量，也不能把这些检查当作大模型长期泛化证明。

默认启用此前已验证的 `categorical_sampling=marginal`、`graph_execution=optimized` 和 `filter_execution=sufficient`。前者积分掉每次重抽、只用一次的可交换类别权重；共享缓存与 corrected 模式仍保留原关系。后两者保留所有可观察机制和正式筛选目标。`reference/legacy` 显式覆盖仍可用于对照。

## 实验口径与复验

实验位于 `/ossfs/workspace/tabicl-training-experiments/prior-fast-default-20261103/`。CPU 吞吐、真实八卡训练、带探针的分布审计、profile 和回归均串行执行，不互相争用 CPU。测试环境为 8×PPU-ZW810、约 69.76 CPU cores quota，Python 3.12、厂商 Torch 2.8、NumPy 1.26、sklearn 1.7.2。

CPU 对照保持 marginal/optimized/sufficient/full；A 为 Torch statistics＋排序去重，B 为本轮 native statistics＋哈希去重。每格按 A→B→B→A 新建进程；generation 区间开始前等待所有 worker 完成依赖/库初始化，不提前生成或丢弃任务。另报包含进程启动/退出的 wall 和各 worker `process_time` 之和，三者不能混用。ABBA 重复任务不计为新增独立质量样本。

训练使用真实 prior、真实模型、8 ranks、global batch=64、micro=4、`COMPILE_FRONTEND=all` 和逐 rank 前后向预热。稳态统计排除前 40 步；保留完整启动 wall，但共享编译缓存与不同启动状态会影响该数值，不能将其解释为严格冷启动收益。逐步分别取 prior/train 的最大 rank 不一定是同一 rank，不能把两者均值直接相加当作 step；端到端速度按日志 elapsed 的 optimizer step 计算。

```bash
export PYTHONPATH=src PYTHONHASHSEED=42
export TABICL_PRIOR_DISTANCE_LIBRARY=$(python scripts/prepare_prior_runtime.py)
python -m pytest -q tests/test_prior*.py tests/test_node_prior_pool.py tests/test_training_recipe.py
python scripts/run_prior_runtime_experiments.py --workers 8 --tasks 512 --seeds 96 97 \
  --output /path/to/new/cpu8
python scripts/summarize_prior_runtime_experiments.py --root /path/to/new/cpu8 \
  --output /path/to/new/cpu8-summary.json
python scripts/run_prior_training_matrix.py --cases ready loader balanced \
  --steps 240 --skip 40 --output /path/to/new/training-screen
python scripts/audit_prior_graph_cpu.py --comparison runtime --seeds 98 99 \
  --tasks 256 --rows 2048 --workers 8 --output /path/to/new/quality.json
python scripts/summarize_prior_runtime_experiments.py --quality --root /path/to/new/quality.json \
  --output /path/to/new/quality-summary.json
```

各脚本保留原始 JSON/log；吞吐脚本拒绝覆盖已有运行。复验应使用新目录，并把 CPU 与训练分开运行。

最终在主目录直接调用 `train_v2_clf_stage1_fast.sh`，取消手工指定的原生库路径，由入口自动准备扩展和环境，串行完成两条路线：

| dtype / 实际默认配置 | 训练步数 / 丢弃前几步 | 稳态 ms/step | 包含检查、预热、池启动与退出的 wall(s) | OOM / 非有限 loss |
|---|---:|---:|---:|---:|
| BF16 / ready_micro / 最大延后 8 | 320 / 40 | 323.20 | 241.72 | 0 / 0 |
| FP32 / ready_micro / 最大延后 2 | 160 / 40 | 488.04 | 207.79 | 0 / 0 |

这两次是实际入口的完整启动验证，使用新的 checkpoint 目录；构建缓存已存在。独立冷缓存构建由前述回归另测，不能将此 wall 称为从零安装/冷编译耗时，也不能直接用这两个不同长度运行的 wall 比较精度格式。

## 配置筛选与增量训练对照

单线程算子 ABBA 微基准示例，连续 float32、连续存储，单位 μs/调用（这些倍数不能用于预测整步收益）：

| 算子 / 形状 | 原实现 | 新实现 |
|---|---:|---:|
| mean/std，2048×8 | 113.43 | 15.28 |
| mean/std，2048×100 | 1346.71 | 102.84 |
| mean/std，4096×100 | 2685.04 | 207.38 |
| 连续行去重成本探测，2048×8 | 135.41 | 7.46 |
| 7 种重复行去重，2048×8 | 3743.09 | 175.87 |

单列统计和单列去重反而是 Torch 更快，最终分派均保留 Torch；微基准包括这些失败点，不据此隐藏较慢形状。

下面是单次 240-step 筛选，去掉前 40 步，单位 ms/optimizer step。三个方案使用相同完整 prior，均启用此前的 marginal/optimized/sufficient 和本轮 native/hash；变化仅为加载和调度策略。它用于筛选候选，不是重复实验的最终速度承诺。

| dtype | node_pool ready_micro | node_pool balanced | DataLoader |
|---|---:|---:|---:|
| FP32 | 491.09 | 487.58 | 595.80 |
| BF16 | 315.79 | 353.87 | 361.49 |

在未增加“单列哈希回退”前，另做了两种 dtype 各四次 320-step ABBA（每次去掉前 40 步）。A 已经采用 marginal/optimized/sufficient、56-worker node_pool 和 ready_micro；B 仅再增加 native/hash。因此它衡量本轮原生算子的**增量**，不是旧默认配置到新默认配置的全部收益。

| dtype | A1 | B1 | B2 | A2 | B 相对 A 吞吐 |
|---|---:|---:|---:|---:|---:|
| FP32 | 490.49 | 493.82 | 489.80 | 490.91 | −0.23% |
| BF16 | 367.50 | 351.30 | 390.07 | 371.56 | −0.31% |

这些重复没有确认增量端到端收益。BF16 的 B1 单次看似提升约 5%，B2 又回落，不能挑选较好的一次作为结论。CPU 减少工作仍有资源价值，但现有预取和训练计算可以掩盖它；任务尾部、调度公平性和运行波动也会改变实际等待。`worker_busy_wall` 包含进程被抢占的时间，不能当成 CPU core-seconds；只有吞吐实验记录的 `process_time` 才用于比较实际 CPU 工作。

再以修改前脚本的默认算法/加载策略运行各 240 步：FP32 为 reference graph＋legacy category＋balanced node_pool，BF16 为 reference graph＋legacy category＋DataLoader。完整 sufficient 过滤保持不变。旧路径分别为 **655.27 / 550.63 ms/step**，wall 为 238.52 / 245.41 s，均无 OOM 或非有限 loss。相对于前述新路径 FP32 约 490–492 ms、BF16 800-step 长测 344.37 ms，累计改进明显；但这是单次旧路径对照、运行长度也不同，不能作为严格配对的精确收益百分比，更不能全部归因于本轮 native/hash。

## 最终完整 prior 与分布审计

最终代码含单列 statistics/compaction 回退。56 workers，每次 2048 个完整任务，每格四次 ABBA，共生成 32768 个任务；下表每格有 2048 个独立配对任务。A/B 内部重复的完整输出和筛选序列均完全可复现，全部正式 forest 调用仍为 25 棵树。

| seed / 行数 | 生成吞吐变化 | 含启动/退出的 wall 吞吐变化 | 实际 CPU 工作减少 |
|---|---:|---:|---:|
| 98 / 2048 | +5.53% | +3.29% | 5.18% |
| 98 / 4096 | +0.86% | +0.14% | 2.40% |
| 99 / 2048 | +3.36% | +2.23% | 3.47% |
| 99 / 4096 | +4.55% | +2.70% | 1.42% |

合并各格的实际时间后，生成吞吐 +3.35%、wall 吞吐 +1.96%、CPU 工作 −2.83%。图生成区间减少约 5.9%–8.6%，过滤区间反而增加约 3.5%–8.2%；后者可能包含任务数/任务数值变化和进程调度影响，不是过滤算法本身得到加速。8192 个独立任务中 8169 个筛选接受/拒绝序列一致；其余仍正常完成正式重试，没有忽略任何原始请求。

为避免默认叠加互相抵消的优化，另做了 seed=98、56 workers、每次 2048 个任务的独立 ABBA 消融：

| 开启项 | 2048 行生成吞吐 / CPU 工作减少 | 4096 行生成吞吐 / CPU 工作减少 |
|---|---:|---:|
| 只改 statistics | +1.03% / 4.58% | +1.16% / 3.82% |
| 只改 compaction | +4.10% / 3.92% | +2.82% / 2.88% |
| 两者同时，前述最终实验 | +5.53% / 5.18% | +0.86% / 2.40% |

2048 行的默认保留两者。4096 行没有证据证明两者组合一定最快，可使用 `TABICL_PRIOR_STATISTICS=torch` 只保留哈希去重。上述实验各自重新运行 A 对照，不能把不同实验的绝对秒数直接排序；小幅差距也没有足够重复来作硬件通用承诺。

另在两个行数各做 512 个独立配对任务的完整质量审计（两组种子；ABBA 重复不加入样本量）。以下为 B−A 的均值变化；balanced accuracy 以百分点计，其余为原量纲。

| 指标 | 2048 行 | 4096 行 |
|---|---:|---:|
| 特征数 | −0.018 | −0.025 |
| 类别数 | +0.002 | 0 |
| 归一化类别熵 | −0.000011 | +0.000269 |
| 少数类占比 | −0.000154 | +0.000268 |
| 常数特征比例 | 0 | 0 |
| 图节点数 / 活跃节点数 | +0.008 / −0.002 | +0.027 / +0.021 |
| 图边数 / 深度 | +0.080 / +0.010 | +0.092 / +0.008 |
| Ridge holdout balanced accuracy | +0.203 pp | −0.073 pp |
| ExtraTrees holdout balanced accuracy | +0.308 pp | −0.101 pp |

2048 行两个探针变化的配对 bootstrap 95% 区间分别约 [+0.005,+0.498]、[+0.073,+0.637] pp；4096 行分别约 [−0.250,+0.091]、[−0.272,+0.023] pp。这是有限样本的实际变化，不能宣称严格同分布或“所有差异都不显著”。这些结果未显示图复杂度下降或明显探针退化，结合保持所有机制的实现约束，支持启用此工程优化；不能代替长期 TabICL 泛化实验。原始摘要保存全部逐字段区间与函数混合计数。

最终主目录相关回归：**934 passed，6 warnings，482.95 s**（`tests/test_prior*.py`、`tests/test_node_prior_pool.py`、`tests/test_training_recipe.py`）。覆盖完整行冲突核验、正负零、strided 输入、float64 oracle、极大/极小值及非有限统计、拟合状态复用、autograd 回退、老库回退、单列成本分派、完整任务/RNG 兼容路径、加载器默认值、延后 8 步、小窗口和 16-rank 公平调度，以及实际冷缓存编译/缓存复用的 stdout 路径约束。最终脚本语法检查与 `git diff --check` 亦通过；测试训练均已退出，没有恢复原来的长训练。

## 4096 行与生成池大小

4096 行使用真实 8-rank 训练，各运行 240 步并去掉前 40 步。global batch 均为 64：

| dtype | micro | 稳态 ms/step | 完整运行 wall(s) | OOM / 非有限 loss |
|---|---:|---:|---:|---:|
| FP32 | 2 | 886.72 | 299.06 | 0 / 0 |
| BF16 | 4 | 518.47 | 204.29 | 0 / 0 |

FP32 的 micro=4 曾在逐 rank 编译预热的 row-interactor 前向中显存不足，该失败日志也保留；不能把它计入有效吞吐，或宣称全部试验都没有 OOM。需要运行这一长度的 FP32 时使用：

```bash
TRAIN_DTYPE=float32 MAX_SEQ_LEN=4096 MICRO_BATCH_SIZE=2 \
  bash scripts/train_v2_clf_stage1_fast.sh
```

两个成功配置的 micro 不同，因此其速度差包括显存约束和累积次数影响，不能当成纯精度格式对照。默认 Stage 1 仍为 2048 行 / micro=4，两种 dtype 均已验证。

另外单次比较了 48 / 64 个池 worker：BF16、2048 行、320 步去掉 40 步，分别为 356.24 / 361.63 ms/step。它们均没有明确超过 56-worker 重复运行的范围（351.30–390.07 ms/step），不据单次差距将默认改成更大的池。56 为此节点的已测折中；其他 CPU quota 可覆盖 `PRIOR_POOL_WORKERS`，不是所有机器的通用最优值。

## 长测与仍然存在的尾部

BF16 / 2048 行 / 56 workers / ready_micro / 最大延后 2 步，再运行 800 步，去掉前 80 步：720 个有效 optimizer steps 的均值 **344.37 ms**，中位数 **295.76 ms**，p95 **391.00 ms**，p99 **1437.05 ms**，最大 **5503.29 ms**。完整运行 wall 为 351.94 s；所有 800 步均无 OOM、无非有限 loss。

这里均值显著大于中位数，原因不能归结为普通计时噪声。在 720 步中，26 步的最慢 rank 取数区间超过 100 ms。最慢的 step=333、554 分别等待 prior 5226.95、5211.59 ms，而 train 区间最大 rank 分别为 301.36、272.76 ms。另有 step=215 的 prior 仅 9.02 ms、train 却为 1539.09 ms，说明也存在训练侧长尾。不同 rank 的 prior/train 最大值不能直接相加，以上按对齐 step 查看，属于瓶颈诊断。

`ready_micro` 的公平性不是超时丢弃：到期而未完成的完整 micro 会阻塞发放。因此既能绕过一般慢任务，也会在公平性边界等待极慢任务。提高 worker 数没有解决这一机制。需要继续区分“全部池子耗尽”与“仍有 ready micro 但到期任务未完成”；前者需要降低生成 CPU 工作，后者可以在有界公平性下调节延后窗口。无论如何，不重新采样慢任务、不降低其最终进入训练的概率。

随后按 2→8→8→2 的顺序做四次 800-step BF16 长测，均去掉前 80 步。两次 8 步配置分别为 **325.77 / 328.72 ms/step**，最慢 rank 的平均取数区间分别为 26.62 / 38.62 ms，最大 1954.89 / 4177.39 ms。8 步配置明显减轻公平性边界的尾部，但没有消除长尾。第二次 8 步运行的完整 wall 为 381.90 s，比第一次 336.79 s 长，说明启动/编译状态不能从稳态吞吐推算。原本的中间复测在 step=390 因 SSH 断开收到 SIGHUP，未用于比较；重新完成的两次有效运行均从独立目录启动，所有任务仍保留。

最后一次 2 步回测为 **343.33 ms/step**，wall 390.93 s，最慢 rank 平均取数 35.12 ms、最大 3830.17 ms；也无 OOM 或非有限 loss。四次有效结果如下：

| 顺序 / 公平性最大延后 | 稳态 ms/step | 最慢 rank 平均取数 ms | 最大取数 ms |
|---|---:|---:|---:|
| A1 / 2 步 | 344.37 | 53.83 | 5226.95 |
| B1 / 8 步 | 325.77 | 26.62 | 1954.89 |
| B2 / 8 步 | 328.72 | 38.62 | 4177.39 |
| A2 / 2 步 | 343.33 | 35.12 | 3830.17 |

各运行稳态步数相同，A 平均 343.85 ms，B 平均 327.24 ms，吞吐提高约 **5.07%**。两个 B 都快于两个 A，支持将 BF16 默认改为 8 步；取数尾部仍有波动，B2 的最大等待甚至高于 A2，因此不承诺每次最大 stall 都减少。

本次只将 BF16 默认公平性延后窗口调大；FP32 没有这个参数的长测对照，保留原值。16 ranks、128 个 local batch 只有 8 个 cohort，因此默认限制为 7 步；32 ranks 的 local batch 无法组成 micro=4，则保留前述 FIFO 回退。随机乱序完成测试涵盖延后 0/1/2/5/8 和 30 个顺序种子，逐 micro 完整性、截止步和全部任务唯一交付都必须成立。性能矩阵脚本固定 2 步用于前述消融；复验新默认 8 步可直接使用 Stage 1 包装脚本，或在 `benchmark_training_schedule.py` 外设置 `PRIOR_READY_MAX_DELAY=8`。

## 最终流水线与 CPU profile

真实 prior 的 8-rank 流水线 profile 各运行 120 步，去掉 40 步，各覆盖 5120 张训练表。两种 dtype 平均宽度均约 49.36，行数 2048；形状工作量相近。profile 有插桩开销，不用它替代无插桩吞吐。

| 对齐 step 诊断，均值 ms | FP32 | BF16 |
|---|---:|---:|
| 最慢 rank train wall | 480.55 | 368.43 |
| 最慢 rank prior fetch wall | 21.25 | 26.98 |
| 真正等待生成完成，最慢 rank | 0.00 | 4.22 |
| 所有 rank 都还缺输入的保守暴露等待 | 0.00 | 4.04 |
| 输入服务等待 | 2.05 | 6.42 |

prior fetch 包括 RPC、解码、主线程调度等，并非全是生成工作。真实生成时间戳显示 FP32 在这一稳定窗口已完全被预取覆盖，BF16 窗口也仅暴露约 4 ms 的生成等待；这解释了为什么 CPU 工作减少而训练 step 未必立即下降。120-step profile 没覆盖 800-step 长测里所有极端任务，因此不能据此宣称 CPU 瓶颈已消失。

单步的生成 CPU 工作总和约为 17.23 / 18.93 core-seconds。它是分配到该 step 的任务合计，可跨越多个训练 step 的时间范围；不能直接把生成 span 与 train 相加。BF16 以 0.3 秒级消费一批 64 张表时，56 个 worker 的供给已经比较紧，减少核时仍有价值。

设备 stream 边界的每表均值如下（ms）。它们包含 CPU 提交空隙，col backward 还包含 DDP 同步尾部，不能称为纯 kernel active time：

| 区间 | FP32 | BF16 |
|---|---:|---:|
| col forward | 9.32 | 3.12 |
| row forward | 4.82 | 2.68 |
| ICL forward | 5.05 | 5.62 |
| ICL backward | 9.45 | 8.72 |
| row backward | 8.23 | 7.39 |
| col backward + sync | 20.11 | 12.03 |

BF16 在 col/row 前向明显更快，ICL 前向并没有相同收益；优化器、ICL、同步和 CPU 长尾仍会限制总收益。现有路径不宜笼统描述为“始终只有 CPU 瓶颈”。

最终代码再做单 worker cProfile，seed=99，每长度 128 个完整任务，和吞吐测试分开执行。总插桩时间为 28.690 / 46.911 s：

| 累计区间占比（有嵌套，不能相加） | N=2048 | N=4096 |
|---|---:|---:|
| should_filter | 32.09% | 37.64% |
| 实际完整森林计划 / Cython 构树 | 30.81% | 36.20% |
| nearest_center | 15.39% | 18.56% |
| compact_em_probabilities | 5.66% | 5.87% |
| 完整行 compaction | 0.72% | 0.60% |

2048 行的原生列统计自身仅 0.188 s（约 0.66%），原生哈希自身 0.177 s。继续打磨这些小项已经难以带来大幅整体改善。下一轮更有价值的是保留完整 25-tree/OOB 目标，进入实际构树的分裂扫描和 OOB 遍历融合，减少逐树接口与中间输出；以及对 nearest-center 的全中心/全坐标 SIMD 内核按真实输入重放优化。需要用固定候选配对审计约束筛选变化，再做完整任务 ABBA，不能用减少树数、提前接受或偏向快任务来换吞吐。

下一轮应先记录固定候选的树构建、OOB 索引准备、遍历和 bootstrap 四段时间，再决定是否值得进入 sklearn 的实际 split 内核。充分统计量已经消掉过滤评分中的大量重复工作；若大头在构树，继续改评分接口的上限很小。单独把过滤时间减半，按本次单 worker profile 估计生成加速上限约 19% / 23%；这只是 Amdahl 估算，不能当作已有实现收益。nearest-center 应捕获真实形状、p、中心数及重复率重放，逐形状比较“沿中心向量化”和“沿行向量化”，保留全中心、全坐标计算，再进入完整任务回归。当前没有足够证据直接替换正式森林算法或启用额外筛选捷径。
