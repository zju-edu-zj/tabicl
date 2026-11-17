# 完整 prior 的多分类充分统计量与 EM 执行融合

## 范围与默认配置

基于 `74b3db2`，继续优化完整 CPU prior。本轮先隔离测量多分类过滤器，再测 EM，最后测试生产规模的 worker 并发。训练保持停止；本文 CPU 数字不能视为 FP32/BF16 训练 step 收益。

`filter_execution` 的 Python、CLI、shell 默认统一改为 `sufficient`：多分类用完整类别计数森林；二分类、回归用上一轮的完整 MSE 执行计划。`planned` 保留上一轮 MSE 实现，`reference` 使用公开 sklearn estimator。不支持的版本或输入走原回退。`dataset_filter_mode` 仍默认 `full`。

没有减少函数族、图规模、样本行数、中心、维度、树数或 bootstrap 次数。没有放宽过滤阈值、提前接受、复用拟合树或丢弃慢任务。此次并未再次增加预取层。

EM 优化归入已有 `graph_execution=optimized`；它不改变 `categorical_sampling` 的默认值。原生库需要预先重建，再在新 worker 中加载；不在训练 worker 内编译。旧原生库没有新符号时继续使用距离和 softmax 的组合路径，仍能把重复行的恢复推迟到输出映射之后。

## 多分类过滤器：改变计算表示，保留完整判据

原实现将类别编码成 C 维 one-hot，作为多输出回归目标。设节点总权重 W，各类别权重 Wc，pc=Wc/W。该节点每个 one-hot 输出的平均平方误差相加是 `1-sum(pc²)`，对 C 个输出再平均便是 Gini/C。固定 C 下，分裂增益排序与完整类别计数的 Gini 相同。

因此，多分类用 sklearn 本身的 `Gini(1, n_classes)`，而非给回归 MSE 逐样本累计 C 个输出及平方和。每张表仍创建全新的 criterion、splitter、Tree、builder，完整拟合 25 棵深度上限 6 的树，使用相同 bootstrap 权重、所有特征（包括常数 padding）、随机种子与 OOB 行集合。OOB 预测仍保留 C 个类别概率。这里节省的是目标编码和 criterion 计算，不是把完整任务换成廉价任务。

二分类原先已经只用一列标签，继续使用这一条 MSE 路径；上一轮测量显示改用分类器反而会慢。回归也不改 criterion。

预测完成后，通过标签编码直接计算原来的多分类 Brier 判据，避免构造 one-hot 和多份差分矩阵。设全表类别频率为 pi，某行真实类别为 y，OOB 预测为 p：

```
baseline_error = 1 - 2*pi[y] + sum(pi²)
model_error    = 1 - 2*p[y]  + sum(p²)
improvement    = baseline_error - model_error
```

仍对逐行 improvement 做原来的 200 次 bootstrap；`p >= .05` 的不可预测拒绝条件，以及可选的 trivial 误差比条件都不变。无 OOB 投票的行继续按 sklearn 的零预测处理；如有无效预测行，baseline 仍来自原完整表，不能用过滤后的子表重算。

实数算术下目标等价，但 sklearn 两种 criterion 的浮点并列分裂可能不同。验证重点是概率误差、正式筛选翻转、重试后的最终任务与分层分布；不把逐位一致作为 prior 质量要求。

## 已完成的真实候选审计与过滤器计时

seed=93/94，N=2048/4096，每组 512 个最终请求。配对审计中，两种森林看到同一完整候选表，并以原 MSE 决策推进重试，避免因两种路径各自接受的样本不同而误判筛选偏移。计时在另外的新进程完成，配对审计耗时不用于吞吐报告。

共覆盖 2,048 个请求中的 **2,257 个多分类候选**。接受/拒绝翻转为 0；同时打开 trivial 条件进行旁路检查也没有翻转。包含 C=3…10、有效宽度 1…100、离散/连续输入与常数 padding。落在旧 p-value [.02,.08] 的边界候选只有 14 个，因此这不是对罕见边界或下游收敛的穷尽证明。按请求 bootstrap 的净拒绝率差区间为 [0,0]，这是当前样本零翻转导致的退化区间，不能解释成未来严格零差异。

另做完整任务 ABBA：planned → sufficient → sufficient → planned，8 个单线程 worker；包含图生成、正式过滤、拒绝重试、进程启动/退出。所有组均记录完整 25 树拟合次数。每个 seed/行数组的 512 个最终任务 digest、宽度、类别及筛选重试序列均相同；ABBA 重复运行也各自完全可重复。

| N | seed | 完整任务 wall 吞吐提升 | process CPU 工作量下降 | 过滤区间耗时下降 |
|---|---|---:|---:|---:|
| 2048 | 93 | 0.14% | 1.91% | 6.82% |
| 2048 | 94 | 1.04% | 2.06% | 7.19% |
| 4096 | 93 | 3.06% | 2.68% | 7.60% |
| 4096 | 94 | 1.22% | 2.08% | 7.76% |

过滤区间是 worker 中的 wall time，不单独等同于 CPU core-seconds。总 process CPU 更直接反映本轮节省的 CPU 工作量。随机任务大部分时间仍在图生成，且本轮基线已经省掉 estimator 框架成本；不能把个别低维森林 2 倍以上的局部收益直接当成完整 prior 收益。

## EM：保留完整函数，改写执行流程

每个 EM 函数仍按原分布抽取 K 个带扰动的中心、每中心尺度 sigma、距离指数 p、外层指数 q，以及后续随机输出矩阵。没有改变中心采样、噪声、尺度、p/q 或随机数消费顺序。原计算为：

```
distance[n,k] = (sum_d abs(x[n,d]-center[k,d])**p)**(1/p)
logit[n,k] = constant[k] - (distance[n,k]/sigma[k])**q
probability = softmax(logit, dim=centers)
output = probability @ random_output_matrix
```

新内核直接计算 `constant - (sum abs(diff)**p)**(q/p) * sigma**(-q)`，以行方向 SIMD 覆盖少量中心的情况，融合稳定 softmax。不再把 N×K 距离和 logits 分别写回内存；保留 N×K 完整概率输出及每 SIMD block 的 K 个临时值。不是把 softmax 换成最近中心，也不删除概率小的中心。

对离散祖先产生的重复完整输入行，距离、softmax、线性输出均只计算一次，最后恢复行顺序。旧实现虽已压缩距离计算，却在 softmax 和线性映射前就恢复了全部行，后两段仍重复工作。原输出 RandomLinear 的懒初始化只依赖 K 和输出维度，所以仍在原调用时点采样同一个矩阵；没有根据去重行重新拟合数据依赖的参数。

p=2 保留原有与输入规模有关的 GEMM 分派。非有限输入、无新原生符号等情况回退。完整输入连续、高维且中心数恰好填满原 AVX2 lane 时，原有沿中心 SIMD 的距离内核更合适，继续走该路径；有重复行或中心存在 scalar 尾部时，新沿行 SIMD 通常更快。这个成本分派只选择计算方式，不改变任务参数。

48 组完整 posterior＋输出映射微基准覆盖 N=2048/4096、p=.8/1.7/3.5、连续与重复行。以下是各形状跨 N/p 的吞吐提升中位数，基线为上一轮 optimized 路径：

| 输入维度 D / 中心 K | 连续输入 | 重复行输入 |
|---|---:|---:|
| 1 / 2 | 260.9% | 259.9% |
| 8 / 5 | 154.1% | 158.8% |
| 32 / 16 | 0.3% | 8.9% |
| 100 / 64 | 0.3% | 17.6% |

另做 28 组中心数扫描，p=1.7、q=1.63，避免 p=q 恰好消去幂运算造成偏乐观。高维 D=100、K=2/3/5/7 的局部提升约 80%/111%/127%/137%；K=8/16/32/64 经分派后基本持平，单次短测仍有约百分之几的噪声。微基准只衡量该算子，不能代表完整 prior 吞吐。

数值检查以 FP64 公式作 oracle，并直接检查完整后验的非负性与行和、strided 输入、零距离、尾部行、训练后新输入、输出矩阵及随机状态。FP32 极尖锐 softmax 会放大距离舍入；保留完整公式，不要求旧浮点求根再乘方与新合并幂逐位相等。

### 完整任务中的 EM 增量

保持 sufficient 过滤不变，只开关 EM 新执行流程。每格 512 个完整接受任务，8 workers，ABBA，新进程全员初始化后开始 generation 计时；warm-up 只加载依赖/原生库，不提前生成或丢弃任务。

| N | seed | generation 吞吐提升 | process CPU 工作量下降 | graph 区间耗时下降 |
|---|---|---:|---:|---:|
| 2048 | 93 | 1.26% | 1.20% | 1.85% |
| 2048 | 94 | 1.24% | 1.32% | 2.18% |
| 4096 | 93 | 0.94% | 0.94% | 1.71% |
| 4096 | 94 | 1.33% | 1.34% | 2.10% |

各模式的 ABBA 重复运行输出及筛选序列完全一致。跨模式 2,048 个请求中，2,044 个请求的正式过滤序列相同，2,037 个请求的最终宽度与类别数相同。输出 digest 不要求相同：概率舍入经下游离散化、重试可能改变少数完整任务。四组的平均标签熵变化约为千分之几或更小；完整配对统计与按请求 bootstrap 均值差区间保存在汇总 JSON。单个任务可因重试而出现较大变化，均值接近并不意味着逐任务相同，也不代替下游模型质量评估。

EM 局部快很多但完整 prior 只增加约 1% 吞吐，是因为大多数工作属于其余函数与完整过滤，不能按最好的微基准外推。这里没有把 200 次 bootstrap 减少或让更便宜任务优先被接受。

### 56-worker 组合确认

单节点 quota 约 69.76 cores，56 个单线程 worker，seed=95，N=2048/4096 各做 ABBA，每次 2,048 个最终接受任务。A 为上一轮 planned MSE＋原 EM，B 为 sufficient＋新 EM，均保留已优化的其余图执行和 marginal sampling。这是本轮相对上一轮 optimized/planned 的增量，不是相对最早 reference 的累计提升。

| N | A generation 两次(s) | B generation 两次(s) | 稳态生成吞吐提升 | process CPU 工作量下降 | 含进程启动/退出的吞吐提升 |
|---|---|---|---:|---:|---:|
| 2048 | 12.873 / 12.271 | 12.123 / 11.984 | 4.30% | 5.43% | 3.66% |
| 4096 | 23.869 / 25.141 | 23.026 / 23.730 | 4.82% | 5.41% | 2.46% |

两种长度各有 2,047/2,048 个请求的正式过滤序列相同，最终宽度/类别一致分别为 2,041 和 2,039 个。所有正式森林仍为完整 25 树；没有丢弃任务。ABBA 同模式重复输出完全可复现，跨模式的差异来自上述 EM 数值变化经生成流程传播。

该测试覆盖生产规模的 CPU 并发，但不是包含节点池组装、数据搬运、设备训练和 DDP 的端到端训练测试。单个 seed 的高并发幅度比 8-worker 多 seed 增量更大，不能据此保证所有任务组合都有 4%–5%。N=4096 的 p99 任务耗时基本不变，本轮没有解决所有慢任务尾部。

## 剩余工作量与下一轮方向

56-worker 组合路径中，worker 的 graph/filter 区间之和约为 72%/28%（N=2048）、69%/31%（N=4096）。这是区间 wall 计时的构成，不是纯 CPU 核秒的精确分账；足以说明单独进一步优化 EM 或 bootstrap 不会带来数量级收益。

最终源码及新库另做单 worker cProfile，seed=95，每长度 128 个完整接受任务；与所有吞吐和回归串行执行。profile 总时间为 34.414 / 55.149 s：

| 区间 | N=2048 占 profile 总时间 | N=4096 占 profile 总时间 |
|---|---:|---:|
| 完整 should_filter（累计） | 29.53% | 31.65% |
| `_planned_prediction`（累计，包含实际 Cython 构树） | 28.46% | 30.41% |
| nearest_center（累计） | 15.96% | 18.86% |
| compact_em_probabilities（累计） | 7.73% | 9.32% |
| `_compact_rows`（累计，已包含在距离/EM 内） | 4.07% | 4.80% |
| Tensor.std（自身，包括整条生成流程的调用） | 5.17% | 5.84% |

累计项有嵌套，不能逐行相加；cProfile 插桩也改变开销构成。这些数字用于优先级判断，完整吞吐结论仍以前面的独立 ABBA 为准。原始 profile JSON、pstats 和文本结果均在实验目录。

下一轮优先复核完整任意 p 最近中心执行、重复完整行检测与布局转换，再考虑标准化统计/仿射输出的融合。保留全部中心和坐标，用真实输入重放决定成本分派；避免让 unique 或有限性检查在便宜形状上比计算本身还贵。过滤器若想继续获得大收益，需要进入构树/分裂扫描内核，而不是继续缓存几个 Python 对象。任何替代实现都应维持完整类别计数、25 棵新树、OOB、全特征与原阈值，并沿用本轮的固定候选配对检查。

本轮没有恢复训练，尚未测量这些增量对 FP32/BF16 optimizer step 的影响。后续训练测量应记录逐步任务形状和最慢 rank 等待，避免将就绪重排后的有限窗口形状差异误当成设备计算加速。

## sklearn 调用约定与版本边界

严格地说，这里适配的是 Python 调用已安装 sklearn Cython 扩展的私有接口，不是把一个版本的二进制 tree 对象拿到另一版本使用。sklearn 官方不为私有 API 提供兼容承诺，所以仍需要明确范围、源码审计、独立进程实测与公开接口回退。

适配了两个分支：

| 分支 | RandomSplitter 构造参数 | 分类 Tree.predict 返回值 | 本轮运行验证 |
|---|---|---|---|
| 1.3.x | criterion, max_features, min_samples_leaf, min_weight_leaf, RNG | 叶类别计数，需要按行归一化 | 1.3.2 |
| 1.4–1.9 | 上述参数 + monotonic_cst=None | 叶类别概率 | 下列 14 个版本 |

实际安装 wheel 验证：**1.3.2、1.4.0、1.4.1.post1、1.4.2、1.5.0、1.5.1、1.5.2、1.6.0、1.6.1、1.7.0、1.7.1、1.7.2、1.8.0、1.9.0、1.9.1**。每版 45 组回归 OOB 与公开 ExtraTreesRegressor 完全相等，31 组分类 OOB 与公开 ExtraTreesClassifier 在 1e-14 容差内相等，共 1,140 组森林比较。包含 N=16/133/257/2048/4096、单/多输出、连续/离散/常数与 padding、全局缺失类别及只有一棵树时无 OOB 投票的行。

1.3.0/.1 的相应构造与 build 调用已核对源码，与 1.3.2 相同；本轮 Python 3.12 环境没有为这两版做 wheel 运行验证。分派支持稳定 1.3–1.9，明确区分这两版的源码证据与其余 15 版的运行证据。rc/dev、1.2 及更早、1.10 及以后的未知版本使用公开路径。NumPy/Scipy 与 sklearn wheel 自己的二进制兼容要求仍由安装环境满足，本适配并不豁免它们。

1.9 新增 `narwhals>=2.0.1` 运行依赖；初次隔离检查因漏装依赖而在 sklearn import 时失败，补装到独立目录后 1.9.0/.1 均通过。未更改训练环境的 sklearn 或 NumPy。所有测试 wheel 安装在节点本地临时目录，结果和日志持久化到实验目录，避免网络挂载上大量小文件安装影响时间。

来源：

- [官方私有 API 兼容政策](https://blog.scikit-learn.org/updates/dev-api/)
- [1.3.0 构造与 build 调用](https://github.com/scikit-learn/scikit-learn/blob/1.3.0/sklearn/tree/_classes.py)、[1.3.1](https://github.com/scikit-learn/scikit-learn/blob/1.3.1/sklearn/tree/_classes.py)、[1.3.2](https://github.com/scikit-learn/scikit-learn/blob/1.3.2/sklearn/tree/_classes.py)
- [1.4.2 splitter](https://github.com/scikit-learn/scikit-learn/blob/1.4.2/sklearn/tree/_splitter.pyx)、[1.4.2 分类节点值](https://github.com/scikit-learn/scikit-learn/blob/1.4.2/sklearn/tree/_criterion.pyx)
- [1.9.1 构造与 build 调用](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/tree/_classes.py)、[森林 bootstrap](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/ensemble/_forest.py)、[运行依赖](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/pyproject.toml)

## 实验目录与脚本

综合回归在最终源码和新原生库上执行：全部 `test_prior*.py`、`test_node_prior_pool.py`、`test_training_recipe.py`，**815 passed，0 failed / 0 skipped，459.31 s**。六条 warning 来自依赖弃用、刻意少树 OOB、测试输入非连续以及已有 nested/fork 路径。新过滤器覆盖 67 项，新 EM 覆盖 54 项（其中包含形状成本分派和混合 dtype 回退），另外补充 2 项消融完整性检查。另有上述 15 版本共 1,140 组独立森林比较，以及 48＋28 组 EM 微基准数值检查。`git diff --check` 通过。

原始结果：`/ossfs/workspace/tabicl-training-experiments/prior-sufficient-cpu-20261102/`。配对候选在 `quality/`，完整过滤器 ABBA 在 `filter-throughput/`，安装版本比较在 `api-local/`。

`check_prior_forest_api.py` 可独立检查一个安装版本，不导入 torch 或整个 TabICL；`run_prior_sufficient_experiments.py --phase api` 自动在独立目录安装并逐版检查。`audit_prior_sufficient_filter.py` 负责候选配对质量检查；`benchmark_prior_exact.py` 负责完整任务计时和完整森林次数；`summarize_prior_sufficient_experiments.py` 校验 ABBA 重复性并汇总，不把重复 seed 当作更多独立质量样本。

远端环境先激活原 PPU SDK，并使用 `/opt/conda/bin/python`。重建库和复验示例：

```bash
export PYTHONPATH=src PYTHONHASHSEED=42
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MAX_JOBS=1 python scripts/build_prior_exact.py \
  --build-directory /ossfs/workspace/tabicl-training-experiments/prior-sufficient-cpu-20261102/native
export TABICL_PRIOR_DISTANCE_LIBRARY=/ossfs/workspace/tabicl-training-experiments/prior-sufficient-cpu-20261102/native/tabicl_prior_exact_cpu.so
python scripts/run_prior_sufficient_experiments.py --phase throughput \
  --comparison em --warm-workers --output /path/to/new-em-results
python scripts/run_prior_sufficient_experiments.py --phase throughput \
  --comparison combined --warm-workers --workers 56 --tasks 2048 --seeds 95 \
  --output /path/to/new-scale-results
python scripts/run_prior_sufficient_experiments.py --phase api --output /path/to/new-api-results
```

正式脚本已默认 `PRIOR_FILTER_EXECUTION=sufficient`。启用本文的 EM 及上一轮完整图执行优化，还需显式设置 `PRIOR_GRAPH_EXECUTION=optimized`；本文完整生成计时同时设置了 `PRIOR_CATEGORICAL_SAMPLING=marginal`。这些图/类别开关的默认值没有在本轮更改。加载旧 `.so` 不会报新符号错误，但也得不到新融合内核收益；必须使用重建后的库，并重新启动 worker。
