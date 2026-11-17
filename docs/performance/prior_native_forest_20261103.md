# 完整森林专用实现与逐任务长尾诊断

本轮优化针对正式 prior 过滤器的树构建，不减少树、特征、样本、类别、图节点或任务，也不增加新的生成池或预取队列。旧实现仍可用 `TABICL_PRIOR_FOREST=sklearn` 显式选择。原有生成优化及启动依赖见 [Stage 1 性能指南](prior_fast_default_20261103.md)。

## 实现与分布约束

通用 ExtraTrees 的随机分割器在每个节点上遍历特征、抽随机阈值，并为每个候选重排样本以计算左右统计。prior 的固定配置只需要深度六、全部特征、bootstrap 权重和 OOB 预测，因此可以专门实现：

1. 每张表把 float32 特征转置一次，整片森林共享；target 和统计量用 double。
2. 每棵树使用独立 bootstrap 权重，仍拟合完整 25 棵树，没有共享已拟合树。只缓存不依赖 X/y 的 bootstrap 计划。
3. 每个节点累计目标总量；逐个随机排列的非恒定特征计算极值、抽均匀阈值，扫描累加左侧权重及目标充分统计量，右侧由总量相减得到。
4. 二分类/回归保留 MSE，多分类保留全类别 Gini；后者与 one-hot MSE 的实数切分排序等价。全部候选评估完，只按最优切分重排一次样本。
5. 恒定特征发现向两个子节点传递；深度、叶子大小、纯节点和最小增益约束保留。每棵树只预测其 OOB 样本，按每行真实投票数归一化；无投票行仍补零并警告。

设节点的目标加权总量为 S，候选左侧为 L，左右权重为 Wl/Wr。回归的左右 SSE 总和等于节点不变的平方和减去 `L²/Wl + (S−L)²/Wr`；多输出逐列求和。分类的加权 Gini 也只需各类计数及左右权重。因此逐候选不必重排样本，也不必再扫描右侧目标。仍扫描全部候选特征、使用全体有效 bootstrap 样本、抽连续随机阈值；没有直方图量化或降采样。

渐近复杂度仍是约 `O(树数 × 深度 × 行数 × 特征数)`。收益来自减少内存写入、树间重复转置及 Python/Cython 接口，不是数量级改变。特征副本约 `4×N×D` 字节，bootstrap 权重约 `8×N×25` 字节；树逐棵构建、预测后释放，没有保存一整个森林供下一张表复用。

原生森林通过 Torch dispatcher 注册，依赖本机 Torch 的 C++ ABI，**不依赖 sklearn 的私有树 ABI**。sklearn 仍用于公开模型及对照路径，不需要替换厂商 Torch。运行库缓存同时哈希 `.cpp` 与新 `.h` 文件，防止修改头文件后复用旧二进制；显式 native 模式会在启动阶段检查新符号。

这不是逐位或逐 seed 复刻 sklearn。特征遍历状态和样本分区顺序改变后，具体随机阈值及近似平局的选择会变化，可能改变少量过滤决策，随后影响该任务的拒绝重试轨迹。保留的是完整森林机制与复杂度；不能仅凭机制相同声称最终接受分布严格相同。下面的独立探针用于检查明显漂移，也不能代替长期训练后的真实数据评测。

## 验证口径

实验根目录：`/ossfs/workspace/tabicl-training-experiments/prior-split-20261103/`。环境为 8×PPU-ZW810，约 69.76 cores CPU quota，Python 3.12、厂商 Torch 2.8、NumPy 1.26、sklearn 1.7.2。

- 切分正确性：独立加权暴力 oracle，直接计算左右子集误差，不复用 C++ 的 total−left 算法；另遍历完整树核验叶子概率/均值、样本覆盖、深度与全 25-tree OOB 聚合。超出原生支持范围的特征/输出/类别/树数在分配 bootstrap 计划前回退公开 estimator。
- 真实候选回放：各行数 128 个真实过滤候选，A/B 交替重复三轮，只测过滤森林，不宣称整步倍数。
- 完整 CPU prior：56 workers，固定参数及任务 seed，A→B→B→A，新建池并等待每个 worker 初始化完成后计 generation；同时记录冷启动 wall、任务 process CPU 总量、拒绝数和每次森林大小。CPU 测试、质量探针和训练串行。
- 接受分布：配对任务测特征数、类别数、类别熵、少数类比例、唯一行数，以及独立 Ridge/HistGradientBoosting 的 holdout accuracy/balanced accuracy。bootstrap 保持 A/B 任务配对，并整组重采样原始四任务参数组，避免忽略共享参数相关性；重复运行不算新独立样本。
- 八卡训练：真实模型、真实 prior、global batch 64、micro 4、COMPILE_FRONTEND=all、每 rank 前后向预热。两种 dtype 各 ABBA，按 elapsed 计算 optimizer step 耗时，排除启动阶段；逐 rank prior/train 的最大值不能直接相加。

初次 2048/seed98 的部分 CPU 格子与轻量回归重叠，因此后续增加独占、更长的完整任务复测。该初测保留作原始记录，不把所有数字都当作独占测量。

## CPU 结果

同一组真实候选的三次回放，2048 行森林 CPU 时间 22.013→18.098 s，吞吐 +21.6%；4096 行 43.897→37.206 s，吞吐 +18.0%。每种行数均有 2/128 个候选过滤决定改变。它只说明森林热点变快，不能外推为完整 prior 或训练快 18%–22%。

完整 prior 初测（每格四次、每次 2048 个完整任务）：

| 行数 / seed | generation 吞吐变化 | 任务 CPU 工作减少 | filter 区间减少 |
|---|---:|---:|---:|
| 2048 / 98，部分重叠 | +7.95% | 7.68% | 20.98% |
| 2048 / 99 | +4.78% | 5.13% | 18.45% |
| 4096 / 98 | +4.27% | 5.88% | 17.42% |
| 4096 / 99 | +6.36% | 7.72% | 18.61% |

这些结果与 Amdahl 限制一致：森林热点约占原 prior 三分之一，热点快约两成只能带来几个百分点的完整吞吐提升，图生成仍占主要部分。filter 区间包含额外统计和任务拒绝轨迹，不能完全等同于单次树构建成本。

随后独占运行的 2048/seed98 长复测，每次 8192 个成功任务：

| 指标 | sklearn A1 | native B1 | native B2 | sklearn A2 |
|---|---:|---:|---:|---:|
| generation 秒 | 41.385 | 43.472 | 41.610 | 45.451 |
| 任务 process CPU 总秒 | 2197.63 | 2306.37 | 2201.83 | 2417.51 |
| filter 区间总秒 | 670.59 | 595.05 | 572.94 | 733.83 |
| 含池启动/退出 wall 秒 | 78.901 | 81.582 | 80.413 | 82.745 |

ABBA 合并 generation 吞吐 **+2.06%**，任务 CPU 工作 **−2.32%**，filter 区间 **−16.83%**；含启动 wall 基本相同（80.823→80.998 秒）。A 的图区间均值 1609.60 秒，B 为 1671.84 秒，存在速度波动和少量重试轨迹变化，不能把热点收益直接套到总量。A/B 的过滤次数分别为 10792/10815，拒绝分别为 2600/2623，所有森林仍为 25 棵，成功任务均为 8192 个。短测的 5%–8% 不能作为长测承诺。

## 接受数据质量检查

每个长度 256 对任务，共 512 对，四任务参数组配对 bootstrap 2000 次。2048 行有 250/256 对、4096 行有 252/256 对接受表完全相同；不同的表来自过滤决定和随后拒绝重试轨迹变化。完整机制保留，但不要求生成数组相同。

| 长度 | 线性 accuracy 变化，百分点 [95% CI] | 提升树 accuracy 变化，百分点 [95% CI] | 提升树 balanced accuracy 变化，百分点 [95% CI] |
|---|---:|---:|---:|
| 2048 | +0.145 [−0.683, +0.958] | −0.002 [−0.896, +0.814] | −0.100 [−1.089, +0.890] |
| 4096 | +0.053 [−0.261, +0.478] | +0.066 [−0.444, +0.716] | −0.054 [−0.692, +0.711] |

宽度、类别数、熵、少数类比例和唯一行数也记录在结果中。未观察到独立探针明显变容易或变难，但这只是有限样本的漂移检查；不能由“CI 包含零”推断统计等价，更不代表最终模型在真实数据上已验证无损。

测试的 nearest 初始上界算法已从生产代码移除：真实热点回放中，2048 行约快 6.7%，4096 行约慢 3.1%。额外四维投影扫描在普通维度上抵消收益，不满足稳定启用条件。原始实验记录保留，最终代码不带该开关或分派路径。

## 八卡训练

每轮 800 步、排除前 80 步，保持已有 ready_micro、公平性延后 FP32=2 / BF16=8、56 workers 和全部 prior 机制。A 为 sklearn 完整森林，B 仅替换森林内核；两者都已包含此前 native statistics/hash 等优化。

| dtype | A1 ms/step | B1 | B2 | A2 | B 相对 A 吞吐 |
|---|---:|---:|---:|---:|---:|
| FP32 | 491.47 | 494.05 | 491.40 | 492.89 | −0.11% |
| BF16 | 329.21 | 312.87 | 323.35 | 312.73 | +0.90% |

FP32 中 A 均值 492.178 ms、B 均值 492.725 ms，未测到端到端收益。每步最慢 rank 的 prior 区间均值 A 为 18.333 ms、B 为 17.920 ms；train 区间 A 为 484.543 ms、B 为 485.297 ms。后者包含搬运、主机提交、前后向、优化器与同步，不是纯 GPU kernel 时间。训练计算和现有生成窗口掩盖了小幅 prior 资源收益，且剩余差距处于轮次波动范围。四轮均无 OOM 或非有限 loss。

BF16 中 A 均值 320.970 ms、B 均值 318.111 ms，合并吞吐约 +0.90%，小于 A/B 各自轮次波动，尚未确认稳定端到端收益。逐步最大 rank 的 prior 均值 29.516→22.608 ms，train 均值 304.616→307.028 ms；两个区间的最大 rank 可能不同，不能相加当作 step。A 两轮最大 prior 等待分别 4.685 / 0.701 秒，B 为 1.320 / 3.006 秒，不能挑选一轮或删除慢步来宣称长尾已消除。全部八轮、合计 6400 个 optimizer step 均正常结束，OOM=0、非有限 loss=0。

默认只在 Stage 1 的 `optimized/sufficient/full` 配方导出 `TABICL_PRIOR_FOREST=native`，依据是完整生成 CPU 工作减少约 2.3%、完整生成吞吐约 +2.1%，而不是已经证明训练快 5%。显式 `TABICL_PRIOR_FOREST=sklearn` 优先；兼容配方和其他 curriculum stage 不自动切换。Python 层未显式设置时仍默认 sklearn。benchmark 驱动同步解析该 Stage 1 环境默认，避免 DRY_RUN 只返回 argv 导致实测运行环境与入口不同。

一键入口仍为 `bash scripts/train_v2_clf_stage1_fast.sh`；源码和依赖沿用原安装流程，新头文件随运行库一起编译。已有厂商 Torch、C++ 编译器及 ninja 时无需新 Python 依赖。native 森林不依赖 sklearn 私有树版本；显式 sklearn 对照路径仍按已审计版本适配或回退公开接口。取消旧的 `TABICL_PRIOR_DISTANCE_LIBRARY` 后启动会按新源码自动重建；显式 native 配旧库则启动前明确报错。

## 长尾诊断

`TABICL_PRIOR_TRACE_DIR=/path/to/new/trace` 启用有界 JSONL 诊断。每个 worker 默认最多记录 2048 个任务，可用 `TABICL_PRIOR_TRACE_TASKS_PER_WORKER` 修改；service 最多记录 500000 个事件。关闭时不安装组件计时 wrapper。只记录身份、seed、形状、阶段和耗时，不保存 X/y。

目录按实验和节点隔离，分析器一次分析一个节点服务；不要把多节点的同名 task_id 合并。检查 worker/service 是否达到记录上限，再解释未覆盖任务的统计。该诊断有计时、JSON 写入和调度扰动，不能把诊断轮的步耗时当作无插桩吞吐。

worker 记录每个任务的外层 graph/sanity/filter 尝试及拒绝原因，组件计时同时给 inclusive/exclusive，避免嵌套重复求和。GraphSCM 内部还可能因祖先无交集重采图结构，这些计入 graph 时间，不逐次单列为外层 attempt。service 记录 task_id 对应的提交、worker 启动、生成、编码、收到时刻，以及公平性等待的具体缺失任务、dispatch 的全部任务及 rank 分组。没有改变池数量、计划 seed、任务顺序或有界公平性规则。

```bash
python scripts/analyze_prior_trace.py /path/to/trace --skip-steps 80 --output /path/to/new-summary.json
```

分析对并发 rank 等待区间求并集，而不是把八个 rank 的等待相加。queued/running/transit blocker 也分别求并集，三者可能因不同任务重叠，不可相加；另给已知 blocker 的总并集。running 截止到编码完成，transit 截止服务收到，收到后的唤醒/锁等待不硬归因成 worker 工作。`--skip-steps` 按实际 dispatch step 选择任务与等待事件。只有 due_micro 等待才归因到公平性 blocker，普通 ready 不足单列。重复 task_id 或空目录明确报错，防止混入不同节点/轮次或将错误路径解读为零等待。

只有 trace 显示关键任务长期排队才值得改变 executor 提交策略；若任务已在运行，单纯改优先级不会加速它。当前保持已有调度语义，避免用没有证据的 EDF 等替换重复已有源顺序。

独立运行 native/BF16 800 步、排除前 80 步，完整覆盖 **46,080 个任务、720 次 dispatch**。56 个 worker 各记录 839–1082 个任务，service 记录 80,008 个事件，均未达到上限。带诊断的数据仅用于归因，不加入上面的 ABBA 吞吐结果。

| 稳态指标 | 结果 |
|---|---:|
| 任务 wall p50 / p95 / p99 / max | 0.179 / 0.822 / 1.432 / 8.687 秒 |
| 任务 CPU p50 / p99 / max | 0.177 / 1.418 / 8.629 秒 |
| due_micro 等待并集 | 13.569 秒 |
| due 等待中存在 queued blocker 的并集 | 0 秒 |
| due 等待中存在 running blocker 的并集 | 13.415 秒 |
| due 等待中存在 transit blocker 的并集 | 0.018 秒 |
| 普通 insufficient_ready 等待并集 | 1.490 秒 |

普通任务排队 p50 为 2.593 秒，这是有界生成窗口的正常积压；不能仅凭它判断关键路径需要 EDF。实际公平性等待的约 98.9% 存在正在执行的 blocker，未观察到尚未取得 worker 的 blocker。CPU 与 wall 相近也说明极慢任务主要确实在做生成工作。这个结果不支持本轮增加调度或预取机制。

所有外层尝试累计 graph/sanity/filter wall 为 9328.21 / 108.15 / 3273.00 秒；其中最终被拒绝的尝试分别占 3094.74 / 41.74 / 845.91 秒。接受 46,080 次、filter 拒绝 14,280 次、类别 sanity 拒绝 6802 次、无特征拒绝 899 次；每任务尝试 p95=3、p99=5、max=10。已经被拒绝的图成本约占全部 graph 成本三分之一，说明更早获得可靠拒绝证据有价值，但这些累计时间不是一种提前判定算法已能省下的时间。不能跳过完整过滤或因任务昂贵而拒绝它。

组件 exclusive wall 中，nearest-center 为 2442.25 秒、linear 904.37、compact EM 778.37、GP 758.45、quadratic 714.20、tree 593.49。nearest-center 约占 graph 区间的 26.2%。最慢任务 task_id=21155 总共 8.687 秒，其中 nearest-center 为 7.731 秒，经历 filter 拒绝、无特征、接受三次尝试；另一任务 17970 没有重试，仍耗时 6.626 秒，其中 nearest-center 5.592 秒。因此不能只优化过滤或重试次数。

下一轮优先捕获这些慢任务中每次 nearest 调用的实际 N/D/K、p、去重前后行数，针对高成本形状比较全中心/全坐标的分块 SIMD 距离扫描与现有实现；对 p=2 单独比较数值可接受的 GEMM 路径。普通形状保留现有快路径，不能因这次被移除的四维上界方案在 2048 上略快就再次全局开启。第二方向是 compact EM 的重复计算/内存布局。进一步减少拒绝成本，应先证明哪些零特征或类别 sanity 判定能在必要输出完成后立即得到，且不会因尚未生成的依赖而误拒；本轮未提交没有这种证据的提前筛选代码。

## 回归与证据

最终原生运行库上的完整回归为 **1141 passed**，包括独立切分 oracle、完整树/OOB、真实 spawned pool、配方和训练数值契约；边界保护与分析器修正追加 **148 passed**。默认入口、benchmark 环境及配方/编译前后向复验 **94 passed / 1 deselected**，不重复执行完整回归已通过的冷编译用例；另有 1 条 Torch TF32 性能提示，不改变当前 FP32 策略。后续分析器阶段汇总追加运行 **7 passed**，其中包含新增的拒绝工作计时检查。各轮有重复测试，不把这些数字相加当作独立用例数。

可审计的汇总数值与测试口径保存在 [实验汇总 JSON](prior_native_forest_20261103.json)，大体积候选、接受表探针记录和训练日志留在上述实验根目录。原始 trace 保留 task_id/seed，可复现昂贵任务；不提交机器相关二进制、checkpoint 或临时无效原型。

另从一键入口实际启动独立的八卡 BF16 32 步短训练，移除显式运行库、森林模式和 trace 环境覆盖，使用新 checkpoint 目录；自动加载 `64c269d3831a8ba30ee5` 运行库，torchrun 进程环境确认 `TABICL_PRIOR_FOREST=native`、OMP/MKL=2。8 个 rank 各完成三次前后向预热，最长 startup 30.661 秒，预热 optimizer 更新为零。为覆盖真实参数更新，短训练设 `WARMUP_STEPS=4`，32 步中 31 步学习率大于零，正常退出、OOM=0、非有限 loss=0。这是默认入口功能验证，不用它估计稳态加速或质量。未恢复长期训练。

## 复验命令

先激活现有 SDK/PyTorch，设置 `PYTHONPATH=src PYTHONHASHSEED=42`，构建并导出运行库：

```bash
export TABICL_PRIOR_DISTANCE_LIBRARY=$(python scripts/prepare_prior_runtime.py)
python -m pytest -q tests/test_prior*.py tests/test_node_prior_pool.py tests/test_training_recipe.py tests/test_training_performance.py
python scripts/benchmark_prior_matrix.py --workers 56 --tasks 8192 --rows 2048 --seeds 98 --output /path/to/new/cpu
python scripts/audit_prior_quality.py --workers 8 --tasks 256 --rows 2048 --seed 99 --output /path/to/new/quality2048.json
python scripts/benchmark_forest_training.py --steps 800 --skip 80 --output /path/to/new/training
```

所有吞吐实验使用新路径；训练驱动拒绝复用输出目录，防止误读旧指标或 checkpoint。诊断 trace 必须单独运行，不能把带 instrumentation 的速度混入默认对照。
