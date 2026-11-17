# 图生成 CPU 吞吐：距离剪枝、公共二次基与不可观察潜变量

## 范围与结果

在 `2ccc09d` 的 `ready_micro + categorical_sampling=marginal` 路径上继续优化。图结构、八类完整函数、各函数的原有参数分布、全行数生成、类别转换和完整 25-tree 正式过滤均保留。本轮没有增加预取层，没有用 cheap component 替代任何原有函数，也没有减少过滤或丢弃慢任务。

新增 `graph_execution=reference|optimized`，Python 和 shell 默认仍为 `reference`；使用本轮优化时应显式启用 `optimized`。这一开关同时适用于 DataLoader 和节点池，独立于调度与 categorical sampling 开关。没有新库时保留 Torch/旧 native 回退，但取得本文距离加速需要重新构建本机 native 库。

首轮 CPU ABBA、两 seed 消融、4,096 张表审计及 native 回归完成后，远端容器重建，`/tmp` 原始文件丢失，已下载的汇总和持久化检查点保留。恢复连接后重新构建 native 库并安装缺失依赖，在新环境中重新执行完整八卡 FP32/BF16 ABBA、最终实现 profile、回归及 CPU 复验。两次环境的计时分开报告，不跨环境拼接对照。按用户要求，长期训练保持停止。

容器重建前，完整接受任务的 CPU ABBA 实测：

| 行数 | reference 两次 wall (s) | optimized 两次 wall (s) | 合并吞吐提升 | CPU 工作量下降 |
|---|---|---|---|---|
| 2048 | 52.548 / 52.512 | 40.933 / 40.801 | 28.54% | 24.27% |
| 4096 | 86.830 / 86.912 | 67.514 / 67.561 | 28.63% | 23.50% |

每格 1,024 个最终接受任务，8 个单线程 worker，seed=83，`PYTHONHASHSEED=42`。顺序为 reference → optimized → optimized → reference；包含进程启动、拒绝重试、图生成和完整过滤。机器为 8×PPU-ZW810，CPU quota 约 69.76 cores。计时基准与训练、分布审计、回归测试分开运行。CPU 表/秒不是训练 step/秒。

恢复环境重新执行 CPU ABBA，每格 **512 个最终接受任务**，其余上述 CPU 参数一致，且不与训练/profile/回归并发：

| 行数 | reference 两次 wall (s) | optimized 两次 wall (s) | 合并吞吐提升 | CPU 工作量下降 |
|---|---|---|---|---|
| 2048 | 28.364 / 28.317 | 21.772 / 21.878 | 29.85% | 24.13% |
| 4096 | 48.051 / 47.849 | 37.084 / 37.099 | 29.27% | 23.28% |

这是新环境内的独立确认，不把它与旧环境 1024-task 的 wall 相除。N=2048 每次 reference/optimized 都调用 668 次完整森林；N=4096 分别为 632/653 次，optimized 没有靠少过滤取得收益。新结果见 [恢复后的汇总](prior_graph_cpu_recovery_20261102.json)。环境为 Python 3.12.11、PyTorch 2.8.0（PPU vendor build）、NumPy 1.26.0、SciPy 1.16.3、scikit-learn 1.7.2；完整依赖锁定记录保存在实验目录的 `pip-freeze.txt`。

N=2048 时，各 worker 的 graph 区间计时和从约 277.5 s 降至 185.3 s，filter 仍约 106–108 s；N=4096 graph 从约 480.0 s 降至 320.3 s。这里分项是 `perf_counter` 区间之和；CPU 工作量下降另由 `process_time` 测量，不能混为同一个时钟。后者 optimized 反而调用了更多过滤（1331 对 1285）和图生成（1477 对 1455），依然更快，收益并非减少正式验证或挑选更便宜的任务。

补充消融每种配置在 seed=83/84 下各生成 512 个完整任务，仍使用 8 workers 和 full 过滤：

| 单独启用的优化 | seed=83 wall (s) | seed=84 wall (s) | 两 seed 合并吞吐变化 |
|---|---|---|---|
| 无，reference | 28.153 | 30.334 | 基线 |
| 无用 embedding/gather | 27.036 | 27.063 | +8.11% |
| 距离 | 24.588 | 26.216 | +15.12% |
| 二次基 | 27.016 | 28.189 | +5.94% |
| 全部 | 21.709 | 21.830 | +34.33% |

消融用于判断改动是否值得保留，不把单项收益相加。无用值删除改变 RNG 消耗，二次式和距离改变浮点求值，所以接受任务流与重试次数也会变化；单项百分比包含有限任务流的波动，并非严格逐算子归因。主要吞吐结论使用上面的 1024-task 重复 ABBA。最终清理过的 native 库与 ABBA 所用库，在 512 个完整任务上逐表 digest、过滤决策和形状完全一致，清理没有改变算法。

## 真实八卡训练：恢复后的完整对照

真实模型、真实 prior、8 ranks、batch=64、micro-batch=4、N=2048，node_pool/56 workers、ready_micro/D=2、marginal/full、COMPILE_FRONTEND=all、逐 rank 前后向预热全部保持一致。两种 dtype 各执行 reference → optimized → optimized → reference，每组 160 个 optimizer steps；剔除前 20 步后，每组计 140 步。只在同一恢复环境内比较。

| FP32 指标 | reference a1 | optimized b1 | optimized b2 | reference a2 |
|---|---|---|---|---|
| steady optimizer step (ms) | 515.73 | 491.99 | 493.83 | 549.37 |
| 最慢 rank prior 等待均值 (ms) | 22.35 | 22.09 | 18.09 | 65.55 |
| 最慢 rank prior 等待最大值 (ms) | 89.19 | 61.16 | 62.29 | 2,610.43 |
| 最慢 rank train 均值 (ms) | 512.97 | 480.07 | 485.19 | 496.73 |
| 进程总 wall，含启动/预热/退出 (s) | 410.17 | 184.55 | 158.29 | 169.28 |
| OOM / 非有限 CE | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

FP32 两次合并 steady 吞吐提高 **8.04%**，合并平均 step 从 532.55 ms 降到 492.91 ms，step 时间下降 7.44%。reference 第二次有 2.6 s prior 等待尖峰；两次 optimized 都没有同等尖峰。首次 reference 承担约数分钟的冷 Inductor/SymPy 编译，而后续运行复用编译缓存，因此表中 launch wall 不是受控的收益比较，不能据此宣称 40.8% 的启动或总训练提速。用于主要结论的是稳态 step；160 步的短窗口仍不能保证长期尾延迟幅度。

最慢 rank 的 prior/train 是每一步分别取最大后统计，二者可能来自不同 rank，且 DDP train 内含通信同步等待，不能把两个均值直接相加当作 step，也不能把 train 下降全归因于 GPU 计算变快。改变就绪顺序也会改变有限窗口内的任务形状与难度，各组 CE 不能用来比较收敛质量。断连前 FP32 第一对 13.76% 的观察仅保留在旧汇总中，不作为恢复后的收益结论。

| BF16 指标 | reference a1 | optimized b1 | optimized b2 | reference a2 |
|---|---|---|---|---|
| steady optimizer step (ms) | 530.75 | 392.61 | 427.63 | 435.84 |
| 最慢 rank prior 等待均值 (ms) | 178.99 | 49.12 | 87.03 | 157.68 |
| 最慢 rank prior 等待 p95 (ms) | 1,245.02 | 73.77 | 713.81 | 964.34 |
| 最慢 rank prior 等待最大值 (ms) | 2,411.34 | 1,335.93 | 1,431.02 | 2,321.51 |
| 最慢 rank train 均值 (ms) | 365.58 | 356.57 | 352.87 | 291.72 |
| 进程总 wall (s) | 378.41 | 159.79 | 142.10 | 142.29 |
| OOM / 非有限 CE | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

BF16 合并 steady step 从 483.29 ms 降至 410.12 ms，吞吐提高 **17.84%**，step 时间下降 15.14%。两组 reference 的 prior 等待均值都高于两组 optimized，但 optimized 仍存在 1.3–1.4 s 尖峰，第二次 p95 也较高，供给尾部没有解决。reference a2 的 train 明显更快，不能解释为 graph_execution 对 GPU 核的加速；有限窗口输入、运行时状态和同步等待没有逐项隔离，故这个百分比应称为本次完整 ABBA 的观察值，而不是严格可复现的长期保证。首次 BF16 也承担冷编译，不比较 launch wall 百分比。

优化后 BF16 的合并 step 时间比 FP32 短约 16.8%，对应吞吐高约 20.2%；这与旧路径两者差不多快并不矛盾。FP32 约 480–485 ms 的 train 较易掩盖后台生成成本，BF16 的计算更短、任务消耗更快，暴露出更多 prior 等待。减少生成成本使 BF16 更能利用计算优势，但不会令训练吞吐等比例提升于 CPU 表/秒。

## 当前剩余瓶颈与下一步优先级

完整 CPU ABBA 的单 worker 区间计时显示：optimized 的 graph 约占 graph+filter 工作时间的 63%–64%，完整正式 filter 约占 36%（N=2048：约 185/106 s；N=4096：约 320/180 s，每次 1024 个接受任务）。原来 filter 约占 27%–28%，所以图执行变快后，它在总工作中的比例上升。这些是 8 个单线程 worker 的区间 wall 之和，适合在 CPU 不饱和的隔离实验中估计工作构成，不能直接与 DDP 等待相加。

恢复后在最终 native 实现上，各长度单 worker profile 128 个完整接受任务，总计 167 次完整过滤。下表分母是分别 39.00 / 62.21 s 的 cProfile 总时间；累计项存在嵌套，不能逐行相加，也不能直接当作无插桩吞吐比例。

| 区间或操作 | N=2048 | N=4096 | 解释 |
|---|---|---|---|
| 完整 should_filter（累计） | 40.91% | 41.72% | 每次仍拟合 25 棵树 |
| 单树 `_fit`（自身） | 23.78% | 28.72% | 包含树构建的 Cython 工作 |
| 森林 `_make_estimator`（累计） | 5.56% | 3.53% | 4175 次 clone/参数设置/随机状态初始化，属于 filter 内部 |
| nearest_center（累计） | 15.16% | 16.51% | 含压缩、检查和 native；nearest native 自身为 10.74% / 12.15% |
| compact_cdist（累计） | 4.88% | 6.01% | EM 完整距离；native 自身为 3.91% / 4.92% |
| `_compact_rows`（累计） | 3.34% | 3.65% | 已含在上面两条距离路径中 |
| Standardize fit + transform（累计之和） | 5.07% | 6.13% | 分列 mean/std，以及两次逐元素操作 |
| RandomWeights.sample（累计） | 6.03% | 4.03% | 含 sampler、逐行排列、softmax 和归一化 |
| `_bootstrap_pvalue`（累计） | 0.85% | 1.08% | bootstrap 索引缓存不是当前最大机会 |
| RandomCauchyDAG.sample（累计） | 0.18% | 0.12% | 图结构构造很便宜，重写它不足以显著提速 |

下一轮建议按下面顺序执行，每个候选先做隔离消融，只有完整接受任务变快且通过审计的实现才进入生产。

1. **完整森林的执行计划复用。** 固定 `random_state=1`、25 棵树、depth=6 的过滤器每个任务反复 clone、反射参数、初始化随机状态、产生 bootstrap/OOB 索引。先在独立的专用 filter executor 中预生成只依赖长度/树 seed 的不可变计划，复用输入校验和 bootstrap/OOB 索引；每张新表必须重新构建全部树和 OOB 预测，不能复用已拟合树。sklearn 私有实现有版本差异，需固定支持版本、特征探测并回退。以原 filter 的 OOB、p-value、接受决策和完整任务分布作验证。仅去掉 `_make_estimator` 所有成本的乐观上限也只有约 5.9% 的 N=2048 CPU 吞吐提升；若要显著超过它，必须进一步优化占总时间约四分之一的树构建/预测执行，而不是只缓存构造器。不得改变 max_features、padding、bootstrap 权重或树分裂规律。
2. **距离执行与重复行检测。** 两条距离路径累计约 20%–23%，是最大的图内热点。现有 `torch.unique(...,dim=0)` 会排序并复制整行；可设计原生哈希去重，以完整行比较处理碰撞，只对真正相同行复用结果，并按 N/C/D 和抽样重复率决定是否压缩。保留全部中心、坐标和任意 p；不得用近似检索替代。核本身可继续合并布局转换、有限性检查和下界访问。消融要分别记录去重时间、去重后行数、距离时间以及每任务尾部，防止便宜输入上哈希开销倒挂。
3. **融合标准化，再检查随机权重。** 先实测 `torch.std_mean(correction=0)`，再考虑原生单次统计+仿射输出；保留 fit 时统计、`ensure_iid` replay、原 eps、所有参与归一化的潜通道及非连续输入支持。随机权重可用批量 Fisher–Yates 替代逐行 Python `randperm`，仍保留每行均匀排列、meta 参数与噪声关系。以占比计算，单独将约 5% 的标准化工作减半只对应约 2.6% 的总吞吐提升；这属于补充收益，不是数倍加速方案。
4. **把 CPU 收益与训练收益分开验证。** FP32 optimized 主要已受约 480–485 ms 的 train 阶段限制；BF16 仍有明显 prior 尾部，继续优化生成更有机会。逐 rank 统计没有显示固定 rank 的巨大计算差距，且同步会传播放大等待，不能仅按 rank 均值重新分配任务。后续对有希望的 CPU 实现增加窗口长度、任务身份/形状和完成时间记录，再跑真实 BF16 重复对照。继续盲目增加 worker 或预取没有本轮证据支持。

以上是依据新 profile 形成的下一轮方案，不表示已经实现或测得收益。若要减少过滤次数，必须区分“完整判据的等价执行”与“修改接受分布”：已经存在的 progressive 提前接受路径会改变判据，本轮始终关闭；不能凭小 prefix 的强预测性就宣称与 25-tree 正式过滤等价。

本轮保留的是降低同类完整任务执行成本的改动。没有为了减少拒绝次数而取消原过滤，也没有把候选任务的成本加入接受条件。

## 从热点到执行设计

完整 prior profile 和真实输入重放显示，主要可改进工作在任意 p 的多维最近中心搜索、EM 的距离矩阵，以及二次函数的重复收缩。原有一维最近中心路径、p=2 的 Torch 路径、重复行压缩、树函数和正式过滤沿用原实现。

### 最近中心：行方向 SIMD，加上非负部分和下界

需要的结果是每行最近中心的编号，不需要完整的 `[N,C]` 距离矩阵。对于 p>0，最小化距离等价于最小化 `sum_d abs(x_d-c_d)^p`，最后的 `1/p` 次幂可以省去。保留原始中心顺序及严格小于的更新规则。

处理方式：

1. 将输入转为按坐标连续的布局，float32 同时处理 8 行、float64 同时处理 4 行。所有 lane 使用同一个中心，因此即使中心只有 2–3 个也能利用 SIMD。
2. 按各坐标在中心之间的 range 降序访问坐标。排序只决定访问顺序；没有取部分坐标，也没有修改距离权重。
3. 对当前中心累加非负项。某行的部分和达到当前最优总和后，后续项不能让它重新胜出；整个 SIMD block 均不能胜出时终止该中心的扫描。
4. 任意 p 使用向量 `exp(p*log(abs(diff)))`；p=1/2 有专门表达式。native 内部支持 p=2，但生产分派仍保留原 Torch p=2 路径。

这是实际距离下界驱动的确定性剪枝，不是近似最近邻，不改变中心数、维数或 p 的分布。收益依赖真实坐标尺度和中心分离程度；range 排序让高区分度坐标更早贡献下界。剩余未剪枝工作仍按完整距离计算。

浮点归约顺序、幂实现以及省去最终 root 可改变极近距离的离散获胜编号。本轮接受这些数值差别，不以逐位一致为目标；测试比较选中中心在原距离下是否达到最小值，而非把所有浮点 tie 编号当成质量指标。对于 EM 不能只保留最近编号，也不能省 root：它继续生成完整距离矩阵，供原有尺度、q 次幂、softmax 和输出函数使用。

### 被否决的实现

最初尝试沿中心方向 SIMD 计算所有距离，再取最小值。它失去了旧标量内核的提前终止，真实输入中的小中心数又不能填满 SIMD。完整 prior 在 N=2048 慢约 8%，N=4096 慢约 14%，因此没有保留这一最近中心实现。

最终最近中心沿行 SIMD，并恢复且加强部分和下界。EM 必须返回全部距离，它仍适合沿中心 SIMD；它与上述被否决的“全部算完再取最近中心”不是同一个消费场景。无收益的最近中心原型不进入提交。

### 二次函数：先形成公共单项式，再做一次 GEMM

原函数对每个输出 o 计算 `sum_ij Q[o,i,j] * x_i * x_j`。不同输出反复计算相同的乘积；`x_i*x_j` 与 `x_j*x_i` 也相同。

新实现仍抽取原来的完整随机 Q，保持其抽样过程及系数相关性。然后建立上三角单项式基 `phi_ij=x_i*x_j`，对角系数为 `Q_ii`，非对角系数为 `Q_ij+Q_ji`，输出为 `phi @ coefficients`。没有改成独立抽上三角系数，那样一般不能保留原矩阵分布。

原有最多 20 个输入特征的机制不变，加常数后基宽最多 231。native 直接填基矩阵，避免 Torch 高级索引的两个临时矩阵，再用 GEMM 合并所有输出。以 D 表示加常数后的宽度，O 表示输出宽度：主要工作由各输出重复收缩变为一次 O(ND²) 形成公共基，加一次 O(ND²O/2) 的规则矩阵乘法。保留一次项、交叉项、平方项和常数项。

### 跳过没有观察者的 embedding 与 gather

`softmax_id` 最终返回原输入与类别编号，`softmax_int` 最终返回编号的浮点表示；两者都不读取 softmax discretizer 生成的 `RandomPoints(K,K)` embedding。但原实现会为该 embedding 进一步构造随机函数，随后按编号 gather，再丢弃。

optimized 给 discretizer 传递 `return_values=False`：保留 Standardize、bias 的 RandomWeights、factor、softmax、多项式类别抽样及所有可观察输出，只不生成这些 embedding。`softmax_disc` 会读取 embedding，仍完整生成。neighbor 的 id/int 模式也跳过无用中心 gather；neighbor_disc/neighbor_func 保留它。

保护条件：`meta_sampling_mode=global` 或 `use_corrected_cat_meta_sampling=True` 时，仍生成 embedding，以保留首次调用形成的共享缓存、边界和类别维度。普通 meta/local 路径的死 embedding 不产生可观察值；其数值 meta 超参数可在首次活跃使用时按原先分布建立，未观察的调用不增加有关这些超参数的条件信息。原有 legacy categorical 的 fresh 权重或 marginal 均匀选择也不形成跨调用共享类别概率，因此不需要保留这部分工作。删除它会改变 RNG 消耗和同 seed 的后续样本，但不需要复制这些已被丢弃的样本值。

保留现有 corrected/global 路径的相关性比追求最大删除范围更合适。没有对所有 RandomPoints 或所有 RandomWeights 做普遍删减。

## 数据复杂度与任务难度审计

每种长度、每种执行模式各 1,024 个最终接受任务，seed=83/84 各 512；两种长度合计 4,096 张表。全部输出有限，实际类别同时出现在训练集和测试集，清理后的常数列比例为 0。八类完整函数均有覆盖，比例约 12%–13%。记录 accepted graph 的节点、边、活跃节点、深度与所有组件选择，也单独标记 reference 中不可观察 embedding 的内部组件，避免把删除死组件误报为删除活跃函数族。

下表为 optimized − reference 的任务级配对均值和 2,000 次 bootstrap 95% 区间。相同 task ID 共用请求元数据，不保证候选图逐个相同。

| 指标 | N=2048 差值 [95% CI] | N=4096 差值 [95% CI] |
|---|---|---|
| 节点数 | −0.394 [−0.955, 0.191] | −0.312 [−0.926, 0.272] |
| 边数 | −4.532 [−10.193, 1.080] | −3.633 [−9.480, 2.071] |
| 活跃节点数 | −0.169 [−0.688, 0.362] | −0.097 [−0.594, 0.420] |
| DAG 深度 | −0.181 [−0.551, 0.187] | −0.191 [−0.554, 0.179] |
| 表宽度 | +0.108 [−0.146, 0.368] | −0.014 [−0.265, 0.237] |
| 类别数 | +0.072 [−0.021, 0.162] | −0.024 [−0.100, 0.044] |
| 最少类比例 | +0.0018 [−0.0045, 0.0078] | +0.0002 [−0.0061, 0.0069] |
| 归一化标签熵 | +0.0107 [−0.0032, 0.0235] | +0.0024 [−0.0116, 0.0169] |
| 图尝试次数 | −0.0127 [−0.0693, 0.0498] | −0.0098 [−0.0586, 0.0410] |
| Ridge 留出 balanced accuracy | −0.0095 [−0.0245, 0.0061] | +0.0096 [−0.0051, 0.0247] |
| 64-tree ExtraTrees 留出 balanced accuracy | −0.0079 [−0.0244, 0.0090] | +0.0056 [−0.0111, 0.0235] |

N=2048 两 seed reference 活跃函数比例的 total variation 为 0.00647；同 seed reference/optimized 为 0.00838/0.00664。N=4096 分别为 0.00621 与 0.00736/0.00909，处于相近量级。图内函数选择存在相关性，未把每个节点当作独立任务做显著性检验。

这些审计没有显示明确的系统性退化，但区间包含 0 不构成分布等价证明，也不能排除几个百分点的任务难度变化。保持复杂度的依据首先是执行算法和潜变量分析；审计用于发现实现错误与明显偏移。最终模型泛化仍需后续训练和下游评估。

## 测试与复现

恢复后的远端 SDK/PyTorch/native 环境，prior 主回归为 **613 passed，5 warnings，247.51 s**；另在生产目录补齐 `test_training_recipe.py` 的 **15 passed，27.84 s**，本次累计 **628 个不同用例通过**。新增文件中的 131 项已包含在主回归中，且单独通过。覆盖任意 p、float32/float64、非连续输入、重复中心/行、零宽度/空输入、跨 12 个数量级的坐标、坐标排列、SIMD 尾行、数值最小距离、完整二次式恒等关系、无 native 回退、20 特征上限、死 embedding 确实不生成、global/corrected 共享状态保护、七种类别转换器的完整生成与 split 覆盖、CLI/shell 配置传递，以及既有 pool/scheduling/window/filter/execution 和训练配方/预热回归。warnings 涉及 sklearn OOB 覆盖、PyTorch nested tensor 和 fork 提示，未出现测试失败。

本地初次回归为 502 passed、119 skipped，补充 7 项也全部通过；skip 来自没有目标 native/GPU 的环境条件，不能把它当成远端 native 验证。消融工具曾因闭包引用覆盖而失败，已修复并验证全部组合；生产完整 ABBA 路径没有执行该测试替换。训练对照使用完整模型和真实 prior，另行报告 optimizer steps、非有限值与 OOM，不用 shape prior 替代。

```bash
cd /ossfs/workspace/tabicl-training-perf
source /usr/local/PPU_SDK/envsetup.sh
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=src PYTHONHASHSEED=42
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/build_prior_exact.py --build-directory /tmp/tabicl-graph-build
export TABICL_PRIOR_DISTANCE_LIBRARY=/tmp/tabicl-graph-build/tabicl_prior_exact_cpu.so

python scripts/run_prior_graph_cpu_experiments.py --tasks 1024 --workers 8 --output /tmp/graph-results/ordered-cpu
python scripts/run_prior_graph_followups.py --phase ablation --tasks 512 --output /tmp/graph-results/ablation
python scripts/audit_prior_graph_cpu.py --tasks 512 --workers 8 --rows 2048 --output /tmp/graph-results/audit2048.json
python scripts/audit_prior_graph_cpu.py --tasks 512 --workers 8 --rows 4096 --output /tmp/graph-results/audit4096.json
python scripts/run_prior_graph_followups.py --phase training --steps 160 --output /tmp/graph-results/training
python scripts/summarize_prior_graph_cpu.py --root /tmp/graph-results --output /tmp/graph-summary.json
```

native 库在启动 worker 前构建一次。AVX2/AVX512 主机使用 AVX2 SIMD 构建，其他主机使用 Torch 默认 vector capability；不要直接跨不兼容硬件/PyTorch 环境复制 `.so`。构建保留 `-fno-fast-math -ffp-contract=off`，性能来自执行组织和向量数学，并非全局开启不受控 fast-math。

重建前 `/tmp/tabicl-graph-cpu-results-20261102` 的原始文件已随容器丢失；已下载的 CPU/审计汇总保存在 `prior_graph_cpu_20261102.json`。恢复后的全部逐任务输出、profile、训练日志、依赖版本、源码快照和 native 库直接写入持久目录 `/ossfs/workspace/tabicl-training-experiments/prior-graph-cpu-recovery-20261102`，其汇总单独保存。逐任务输出和训练日志不加入源码。可以用 `profile_prior_graph_cpu.py` 做完整任务 profile，或用 `diagnose_prior_distances.py` 从真实任务捕获耗时距离输入重放；二者会增加测量开销，不能用其 wall 时间代表无插桩吞吐。

旧汇总保留断连前已经下载的完整 CPU/审计数据；消融和 FP32 第一对还保留了已观察到的概要数值，但未恢复已丢失的原始逐步文件。源码基线为远端 `2ccc09d3687027013df0e131a19f6a925844020b`。本地另一个已有未提交修改的仓库未作改动。

## 启用

在原先实测的 ready/marginal 配置上增加：

```bash
# 在本次相同的远端 SDK/PyTorch 环境可直接使用已验证的持久化库。
# 更换环境后，应按上方命令重新构建并修改此路径。
export TABICL_PRIOR_DISTANCE_LIBRARY=/ossfs/workspace/tabicl-training-experiments/prior-graph-cpu-recovery-20261102/tabicl_prior_exact_cpu.so
PRIOR_GRAPH_EXECUTION=optimized \
PRIOR_BACKEND=node_pool PRIOR_SCHEDULE=ready_micro PRIOR_READY_MAX_DELAY=2 \
PRIOR_CATEGORICAL_SAMPLING=marginal PRIOR_FILTER_MODE=full \
TRAIN_DTYPE=float32 COMPILE_FRONTEND=all COMPILE_WARMUP=True \
CKPT_DIR=/path/to/the/intended/checkpoints \
bash scripts/train_v2_clf_stage1.sh
```

`CKPT_DIR` 指向已有检查点时自动恢复模型、优化器、scheduler 和 step；不要用 `INIT_CKPT` 代替完整恢复，后者只载入权重并重置训练进度。若已有训练仍在运行，必须先让它完成检查点并停止，再启用新 worker。

本次原 FP32 长期训练已先完成并加载验证 `curr_step=71500` 的完整检查点，然后停止原进程。检查点为 `/ossfs/workspace/tabicl-training-runs/stage1-fp32-ready-marginal-20261102-bhsi_47p/checkpoints/step-71500.ckpt`，文件大小 219,622,140 bytes，包含模型、optimizer、optimizer_spec、scheduler 与 step。连接恢复后确认持久化文件仍在。本轮只运行短对照实验，遵照用户最新要求不恢复长期训练；未来需要恢复时再显式使用该检查点目录。
