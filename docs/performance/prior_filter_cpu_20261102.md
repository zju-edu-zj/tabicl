# 完整 prior 过滤器：复用执行计划，保留所有树

后续实现已扩展 sklearn 支持版本并将默认过滤执行器改为 `sufficient`，见[充分统计量与 EM 优化报告](prior_sufficient_cpu_20261102.md)。本文保留当时的配置和实验结论；下文的 1.7.2 版本限制、默认 reference 与未启用建议均属于该历史版本。

## 本轮范围

基于 `05c11f80e4e4e6ae1cd146c4a0b9e3642eaead60` 已优化的图生成路径，新增 `filter_execution=reference|planned`。两组均使用 `graph_execution=optimized`、`categorical_sampling=marginal` 和 `dataset_filter_mode=full`；本轮收益是相对于上一轮优化后的增量，不能再次计入上一轮的图执行收益。

每个候选数据集仍完整生成，正式过滤仍新拟合全部 25 棵深度上限 6 的 ExtraTrees，保留原 bootstrap、OOB、Brier/MSE 改善、200 次 bootstrap p-value 与接受阈值。没有替换函数族、降低复杂度、缩短行数、提前接受或丢弃慢任务。ready_micro 的有界等待与公平性不变。

## 执行方法与正确性边界

原 sklearn 高层接口对每张表反复 clone 25 个 estimator、反射和校验参数、构建随机状态、生成 bootstrap 索引，并在 OOB 预测后计算过滤器不会读取的 R²。新专用执行器直接调用同一版本 sklearn 的 Cython 树构建器，省掉这些固定框架成本。

LRU 只缓存依赖行数、树数和固定 seed=1 的不可变计划：每棵树的 bootstrap 计数权重、OOB 行号、最终投票数及 splitter 初始种子。缓存上限为 4 个计划，单计划按 `N * n_estimators * 16 <= 2 MiB` 限制主体数组，另有投票数组和少量对象开销。训练常见 N=2048/4096、25 棵树均覆盖。缓存不包含数据、标签、拟合后的树或预测。

原过滤器本来就对每张表使用同一个 random_state=1，因此同长度下这些 bootstrap 计划已经相同；缓存没有额外引入跨表的随机耦合。树的分裂仍依赖每张表的新数据和标签。

每张新表仍创建全新的 criterion、splitter、Tree 和 DepthFirstTreeBuilder，并完整拟合全部树，再按原顺序汇总 OOB。所有原特征列（包括 padding）都参与原随机特征选择；不能随意删掉常数 padding 来加速树，否则会改变分裂规律。

额外审计了目标机安装的 sklearn 1.7.2 `_splitter.pyx`：splitter 初始化只从 NumPy RandomState 读取一次整数，随后使用自己的 C uint32 随机流。缓存这次固定读取结果，避免每棵树重建 MT19937；没有共享或固定后续分裂随机数，也没有复用拟合树。

由于使用私有 Cython ABI，专用路径只支持经过审计的 **scikit-learn 1.7.2**。其他版本、非有限数据、超大计划等返回普通 ExtraTrees 路径；progressive 模式也沿用原实现。Python/CLI/shell 默认均为 reference，显式设置 `PRIOR_FILTER_EXECUTION=planned` 才启用。没有宽泛捕获异常来隐藏实现错误。

本轮不是以逐位一致作为 prior 质量要求，但执行复用本身可以做到一致：单元测试的完整 OOB 预测与 sklearn 数组完全相等；完整任务对照的输出 digest、过滤决策、宽度、类别数和完整森林拟合次数完全相同。这比仅观察分类比例相近更直接地排除了这次执行改动引入的筛选偏移，但不证明整个 prior 已最优或模型下游收敛不变。

## 完整 CPU 任务实测

目标环境：8×PPU-ZW810，CPU quota 约 69.76 cores，Python 3.12.11、vendor PyTorch 2.8.0、NumPy 1.26.0、sklearn 1.7.2。每组 512 个最终接受任务、8 个单线程 worker、seed=83、`PYTHONHASHSEED=42`。顺序 reference → planned → planned → reference，各次使用新进程，包含启动、拒绝重试、图生成及完整正式过滤；没有和训练、profile 或回归并发计时。

| 行数 | reference 平均 wall | planned 平均 wall | 完整任务吞吐提升 | process CPU 工作量下降 |
|---|---:|---:|---:|---:|
| 2048 | 21.971 s | 20.121 s | **9.20%** | 9.86% |
| 4096 | 37.094 s | 35.363 s | **4.89%** | 5.95% |

N=2048 每次都进行 668 次完整森林过滤、741 次图调用；N=4096 每次都是 653 次完整过滤、726 次图调用。全部任务逐项记录完全相同。重复 ABBA 的同 seed 表不能算作独立质量样本：这里共 1024 个不同的请求任务身份，重复运行用于计时复验。

过滤区间的 worker wall 之和从 50.966 → 37.332 s（2048，下降 26.75%）、87.190 → 71.911 s（4096，下降 17.52%）；graph 区间基本不变。这里的 `filter_s`/`graph_s` 是 perf_counter 区间之和，不是 process CPU 时钟；CPU 工作量另由任务内部的 process_time 记录，不含 worker 导入/初始化。进程池启动和长尾也影响总体 wall，所以不能把过滤下降比例直接当作整体吞吐提升。

移除高层框架后，实际构树、预测和图数值工作仍占主导，因此收益是约 5%–9%，不是倍数级。4096 行时真实构树和图数值成本更大，固定框架成本占比下降，增益随之缩小。

长尾也没有消失。seed=83 两次运行的单任务 p95 均值为 0.782 → 0.740 s（2048）和 1.492 → 1.437 s（4096）；p99 为 1.334 → 1.232 s 和 2.131 → 2.118 s。尤其 4096 的 p99 几乎不变，说明仅消去过滤框架成本不能解决最慢图生成任务。这里是各次分位数的平均，不是合并样本重新求的分位数。

用独立 seed=84 重复全部 ABBA，结果如下；不能将不同 seed 的 wall 差异归因于执行器，因为图规模、类别和重试次数也不同。

| 行数 | reference 平均 wall | planned 平均 wall | 完整任务吞吐提升 | process CPU 工作量下降 |
|---|---:|---:|---:|---:|
| 2048 | 22.118 s | 20.405 s | **8.40%** | 9.08% |
| 4096 | 34.359 s | 32.485 s | **5.77%** | 6.05% |

两种 seed、两种长度共 **2048 个不同请求任务**，每个任务均进行四次对照；全部输出 digest、过滤决策、宽度、类别和完整森林拟合次数无不一致。此处仍然不是 8192 个独立质量样本。

## 优化后 CPU 成本在哪里

分别对 128 个完整接受任务作单 worker cProfile，与上面的计时实验分开。self 总计为 32.915 s（2048）和 56.292 s（4096）。下列 self 区间互不重叠，但 Cython 构树主要记在 `planned_oob_prediction` 的 self 中；不是说这些时间都花在 Python。

| self 成本 | 2048 | 4096 |
|---|---:|---:|
| 完整森林专用执行器 | 9.510 s（28.9%） | 18.537 s（32.9%） |
| nearest 距离内核 | 4.164 s（12.7%） | 7.744 s（13.8%） |
| cdist 距离内核 | 1.527 s（4.6%） | 3.040 s（5.4%） |
| std | 1.742 s（5.3%） | 3.537 s（6.3%） |
| unique_dim | 1.257 s（3.8%） | 2.220 s（3.9%） |

实际森林约占三成，距离数值约占 17%–19%，二者优先级高于继续打磨 DAG 拓扑管理。即便将 std 或 unique_dim 各自完全消除，整体上限也只有几个百分点，而且当前 std_mean 替换已经证实不合适。profile 会扰动时间，不能把这里的比例当作无插桩吞吐的精确份额；cumulative 列还存在父子重叠，不能相加。

## 真实八卡训练

使用真实 stage1 模型和完整 prior，8 ranks、batch=64、micro-batch=4、56 个节点级 worker、ready_micro/D=2、marginal/full、optimized graph、COMPILE_FRONTEND=all 与逐 rank 前后向预热。每种 dtype 进行 240-step ABBA，剔除前 40 步，各次保留 200 步。

| BF16 指标 | reference a1 | planned b1 | planned b2 | reference a2 |
|---|---:|---:|---:|---:|
| steady optimizer step (ms) | 359.03 | 392.68 | 386.19 | 407.46 |
| 每步最慢 rank prior 等待均值 (ms) | 65.10 | 47.51 | 52.57 | 84.39 |
| 每步最慢 rank prior 等待最大值 (ms) | 2274.88 | 1479.35 | 2358.16 | 2685.78 |
| 每步最慢 rank train 均值 (ms) | 314.93 | 372.68 | 354.38 | 350.34 |
| OOM / 非有限 CE | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

BF16 合并平均 step 为 383.24 → 389.43 ms，对应吞吐 **−1.59%**，未显示整体训练收益。最慢 rank prior 等待均值约 74.75 → 50.04 ms，但 train 区间约 332.64 → 363.53 ms，抵消了它。不能据此声称生成器让 GPU 算术变慢，也不能反过来忽略整体变慢的结果：就绪重排改变有限窗口内的 micro 组合及最大 padding；train 计时还含 DDP 等待，缺少按步输入形状与设备事件时不能精确归因。生产默认仍保持 reference，planned 作为有明确 CPU 收益的可选执行器，尚不推荐声称它能提升 BF16 端到端吞吐。

FP32 合并平均 step 为 **485.89 → 483.51 ms，吞吐 +0.49%**，同样不足以证明稳定的端到端收益。这轮明确建立的是完整 prior 的 CPU 工作量与独立生成吞吐改善；没有建立显著的 BF16 或 FP32 训练加速。

| FP32 指标 | reference a1 | planned b1 | planned b2 | reference a2 |
|---|---:|---:|---:|---:|
| steady optimizer step (ms) | 481.03 | 482.88 | 484.13 | 490.76 |
| 每步最慢 rank prior 等待均值 (ms) | 19.17 | 21.64 | 21.30 | 19.10 |
| 每步最慢 rank prior 等待最大值 (ms) | 46.61 | 120.98 | 60.57 | 53.64 |
| 每步最慢 rank train 均值 (ms) | 469.52 | 469.59 | 471.71 | 480.38 |
| OOM / 非有限 CE | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

FP32 的 prior 等待已很小，计算区间明显更长，额外 CPU 产能可以被既有预取吸收，因此生成端提速并不自动缩短 optimizer step。BF16 更容易暴露供给长尾，但仍需处理 train/DDP/输入形状波动。这里的 prior 与 train 是分别对 rank 取最大值后的统计，且与 wall step 的计时边界不同，不能相加来计算精确瓶颈占比。

共享编译缓存、有限窗口输入形状、ready 完成顺序与同步等待会影响计时。只报告稳态 optimizer step 和按 step 对齐的最慢 rank 指标，不把冷启动 wall 当成受控收益，不用各组 CE 比较收敛质量。长期训练保持停止。

CPU 测试用 8 个 worker，而真实训练用 56 个 worker 并同时驱动 8 个设备；内存带宽、调度和同步压力不同，不能把 CPU 小池子的增益比例直接外推。后续训练归因应按 `(step, rank, micro)` 同时记录实际行数、有效列数、padding 后最大列数、任务身份和设备执行时间；先区分输入组合变化、真正的设备计算与 DDP 等待，再决定是否调整 worker 数和 shape 调度。只看 prior 均值或只看最慢 rank 的一个累计区间不足以作此判断。

审计现有计时代码确认，`Timer` 使用主机 `default_timer`，而 `run_batch` 末尾通过 `.cpu().tolist()` 回传指标会等待设备工作完成。因此 train 时间包含模型执行、指标回传和 DDP 等待，是有用的吞吐边界，但不是 CUDA/PPU 事件测到的纯 kernel 时间。后续应使用异步事件并延后读取，避免为诊断每个 micro 额外同步、反过来改变性能。

## 没有保留的实验

实测用 `torch.std_mean(correction=0)` 合并原来的分开 mean/std。对 N=2048/4096、宽度 1/8/32/100、连续/跨步输入做 ABBA 微基准。多列仅约 1%–8% 局部收益，一维反而变慢：分开/合并耗时比约 0.76–0.81（2048）和 0.47–0.52（4096）。因此没有改生产 Standardize，也不提交这个无收益原型；原始测量保留在实验目录，摘要记录在结果 JSON。

## 后续算法审计

下一步更值得研究的是**分类标签的充分统计量**，而不是继续压缩 estimator 构造。当前多分类过滤把类别变成 C 维 one-hot，再做多输出 MSE 回归；对一个节点，平均 MSE 等于 `(1 - sum_c p_c²) / C`，即 Gini/C。二分类只保留一列，方差则是 Gini/2。因而在实数算术下，完整类别计数可以表达同一个分裂目标和同一叶子概率，不需要逐样本累加 C 个 one-hot 值。

目标机 `_criterion.pyx` 还显示，原 MSE 的 children_impurity 会再扫描左侧样本的平方和；类别计数已经提供这些充分统计量。这可能比框架复用有更大机会。需要先测完整森林，再审计离散/常数输入的分裂并列、纯叶停止、bootstrap、全部特征及二分类输出布局，最后做完整 prior 对照。数学上的目标等价不表示 sklearn 两种实现浮点并列时逐树相同，更不直接证明完整任务吞吐会提高。此项先作为调查，不混入本次已验证的 planned 生产路径。

本轮另外做了 **36 组完整森林探针**：N=2048/4096、d=1/8/100、C=2/5/10、连续/取整离散 X，随机标签，每组 ABBA、每段三次完整拟合。结果相对于当前 planned MSE：

| 类别数 | d=1 森林吞吐变化 | d=8 | d=100 |
|---|---:|---:|---:|
| 2 | −5.5%～−11.3% | −2.0%～−3.2% | −0.6%～−1.9% |
| 5 | +60.2%～+72.9% | +23.3%～+27.4% | +3.2%～+4.9% |
| 10 | +132.3%～+165.9% | +53.2%～+63.6% | +10.2%～+12.1% |

这说明值得设计**多分类专用充分统计量路径，二分类继续走单列 MSE**。31/36 组预测完全一致，另 5 组最大概率差为约 0.032～0.167；目标等价并不保证数值并列时走同一树，后续分裂可能放大差异。这不是为了追求逐位一致而否定它，而是明确完整任务中要测什么：固定候选表上配对比较正式 p-value、接受率及边界翻转，并按 C、d、图规模、离散比例和任务可预测性分层；还要测重试后的输出分布、完整任务吞吐与长尾，最后看训练。当前探针使用随机标签和合成 X，不代表真实 prior 的类别/宽度权重，也没有正式过滤与重试，因此不能据此宣称整体倍数加速或质量已验证。原型未进入提交，测量记录保留在实验目录与汇总 JSON。

另一个可严格定义的方向是**完整判据的提前认证**，与已有四树 z-score 提前接受不同。计划已知每行最终 OOB 投票数 v 和剩余投票数 r；前缀概率和为 s 时，最终概率必在 `s/v + (r/v)*simplex` 中。对每行分别求最终平方误差的上下界，再对原来的 200 个 bootstrap 索引求均值界。只有所有剩余树的可能输出都给出相同接受/拒绝结论时才停止，否则继续完整森林。它不要求弱 prefix 预测完整结果，但界可能很松，重复计算界也有成本；未经过完整任务实测之前，不宣称它有收益，也没有把它加入本次实现。`remove_trivial_datasets` 还需要独立的误差比例界，不能直接复用默认分类条件。

距离内核与重复行压缩仍是图内重点。单纯消除约几个百分点的排序/去重开销，其总吞吐上限有限；应同时记录去重率和中心/维度规模，避免压缩成本倒挂。label-first 提前排除无效标签也不能直接重排现有 lazy transformer：共享 meta 参数的首次抽样边界、fit 时数据统计和 ensure_iid replay 都可能依赖调用顺序，需要先明确联合生成计划。没有把任意“少生成一些”当成免费的分布保持优化。

## 测试与复现

远端 prior/pool 主回归 **677 passed，258.04 s**；另有训练配方 **15 passed，16.71 s**，共 **692 个不同用例通过**。新增 64 项覆盖完整 OOB 与 sklearn 一致、连续/离散/常数与 padding、二分类/多分类/回归、多个长度与输出维度、非连续输入、无 OOB 投票、所有树重新构建、缓存只读及上限、全局 RNG 不变、版本/非有限输入回退与 CLI。

```bash
cd /ossfs/workspace/tabicl-training-perf
source /usr/local/PPU_SDK/envsetup.sh
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=src PYTHONHASHSEED=42
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# planned 专用执行器的支持版本；其他版本会自动回退 reference。
# python -m pip install scikit-learn==1.7.2
# 使用上一轮已构建的目标机 native 库；也可以用 build_prior_exact.py 重建。
export TABICL_PRIOR_DISTANCE_LIBRARY=/ossfs/workspace/tabicl-training-experiments/prior-graph-cpu-recovery-20261102/tabicl_prior_exact_cpu.so

python scripts/run_prior_filter_cpu_experiments.py --tasks 512 --workers 8 --output /tmp/filter-results/abba-final
python scripts/run_prior_filter_cpu_experiments.py --seed 84 --tasks 512 --workers 8 --output /tmp/filter-results/abba-seed84
python scripts/run_prior_filter_training.py --steps 240 --skip 40 --dtypes bfloat16 float32 --output /tmp/filter-results/training
python scripts/profile_prior_graph_cpu.py --tasks 128 --rows 2048 --graph-execution optimized --filter-execution planned --output /tmp/filter-results/profile2048-final.json
python scripts/profile_prior_graph_cpu.py --tasks 128 --rows 4096 --graph-execution optimized --filter-execution planned --output /tmp/filter-results/profile4096-final.json
python -m pytest -q tests/test_prior*.py tests/test_node_prior_pool.py tests/test_training_recipe.py
```

启用完整优化组合：

```bash
PRIOR_SCHEDULE=ready_micro PRIOR_READY_MAX_DELAY=2 \
PRIOR_CATEGORICAL_SAMPLING=marginal PRIOR_GRAPH_EXECUTION=optimized \
PRIOR_FILTER_MODE=full PRIOR_FILTER_EXECUTION=planned \
COMPILE_FRONTEND=all COMPILE_WARMUP=True TRAIN_DTYPE=bfloat16 \
bash scripts/train_v2_clf_stage1.sh
```

这只是复现命令，本轮没有启动长期训练。原始逐任务记录、训练日志和 profile 保存在持久目录 `/ossfs/workspace/tabicl-training-experiments/prior-filter-cpu-20261102`。汇总见 [结果 JSON](prior_filter_cpu_20261102.json)。另一个已有未提交修改的本地仓库未作改动。
