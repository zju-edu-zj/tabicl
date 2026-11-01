# Prior 生成流程重审与可执行的加速方案（2026-11-01）

后续完整 micro 就绪调度、类别潜变量边缘化及其端到端结果见
[2026-11-02 实现报告](prior_ready_redesign_20261102.md)。本页保留上一轮实验的设置与结论。

## 本轮纠正

已有预取仍保留；新增 RankPrefetch、pilot→tail、histogram gate、校准配置和相关训练参数已移除。失败原型不作为“默认关闭”的生产分支保留。旧实验的 JSON 和历史报告保留用于说明为何撤回，不能再按旧命令启用。

此前不能由 histogram 筛选失败推断“提前筛选无效”。它既改变了首次拟合的样本数，又使用了有限的边际/和积探针，还把校准样本分散到 63 个层。这只能否证那一版实现及部署策略。用户允许浮点和输出不一致；需要保护的是任务覆盖、复杂度和训练价值，而不是原数组。

本轮真正实现的流程改动是：**完整 N 行生成 + 4→25 棵树的可复用渐进过滤，仅提前接受，困难任务继续正式过滤**。另实现全部树/全部输出的融合执行。默认仍为 full；progressive 是有独立审计的显式选项，不宣称已证明最终模型精度不变。

## 为什么又加预取不解决问题

训练的数据路径已有两种提前生产机制：

- DataLoader 后台 worker、persistent workers、prefetch_factor；
- node_pool 的节点共享进程池、任务窗口、结果队列和输出字节预算。

它们属于不同 prior backend，不能把两个 backend 的容量相加。原新增线程只在 rank 端提前做 RPC 接收、反序列化和 ACK，没有增加生成 worker，也没有降低每张成功表的 CPU 工作量。

当生成速度小于消费速度，现有队列最终耗尽，再叠一层接收队列只能短暂挪动等待位置。Python 对象解码还受 GIL、内存带宽和临时副本影响。队列非空证明线程工作，并不证明最慢 rank 变快。

旧 BF16 短测：step 620.2→618.9 ms，最慢 prior 等待均值 288.6→300.8 ms，尾部没有改善。rank0 的改善不能代替每步 max-rank。故撤回新增接收队列，而不删除原有生成预取。

## 当前生成到底做什么

### 任务与图计划

TaskPlanner/Prior 先确定 N、特征数、类别数、train_size 及任务 seed。batch_size_per_gp 的任务可能共享形状，做开发/验证划分时应把共享参数组放在同一侧。worker 数是节点级 CPU 资源分配，不是每个 rank 都再创建同样多的 worker。

GraphPrior 对同一个任务形状进行拒绝采样。每次尝试重新抽图、函数和数据。Context 包含 PriorConfig 与 GlobalSampler；global 参数每个名字只抽一次，meta 参数先抽分布参数再抽局部值，local 独立抽样。这会让同一任务内的宽度、权重、函数等具有相关性。

RandomDataset 抽节点数，RandomCauchyDAG 以节点输入/输出 importance 和 sigmoid 连边概率抽有向无环图。随后给 x/y 各组抽可放置节点，再把数值/类别列放到节点。结构过滤检查 x 与 y 是否存在公共祖先；不满足时在生成张量之前拒绝。这已经是有效的便宜早筛，应保留。

RandomGraphFunction 反向标记所有观测列需要的祖先，不计算完全无关节点；计算 last-use，在最后一个活跃子节点之后释放父节点引用。不存在简单“把无用半张图删掉”而尚未实现的免费收益。

### 每个节点及每个函数

根节点首先选择 uniform、circle、Gaussian、covariance 等基础 points，再通过 RandomFunction。非根节点走 concat 或 aggregation：后者分别变换各父节点，再 sum/product/max/logsumexp，不能用一份简化变换替代。

RandomFunction 当前 default 含 MLP、随机树、离散化、线性、二次、GP、EM assignment、product。这里的 CheapRandomFunction 是仓库原有的有限子函数集合，用于 product/转换器防递归；本轮没有新增或扩大这种替代。

- MLP：随机宽度与层数、起止激活，逐层随机线性映射。
- 树：随机抽全部 T 棵 oblivious trees；全表标准差决定分裂维度抽样权重，从全表行抽阈值，保留全部深度和全部随机叶向量，对 T 棵求均值。
- 离散化：从输入行抽全部 centers，抽 p，最近中心映射到随机线性变换后的目标向量。
- 线性：抽随机矩阵；二次项保留原先的最多 20 个输入维度及完整随机二次张量。
- GP：保留 256 个随机 Fourier frequencies、核分支、各维尺度、随机相位及完整输出投影。
- EM assignment：抽带噪 centers、尺度、p/q，计算全部中心距离及 softmax 软分配，再随机线性映射。
- product：保留两个原有子函数及逐元素乘积。

节点随后可按全表标准化、抽特征 importance、做全局 RMS 归一化。这里 L2Normalize 的 fitted factor 使用全表平均平方范数，不能当作纯逐行归一化。

每个观测列的 converter 从部分 latent features 提取数据，还可能把转换后结果写回 latent：

- numerical：id 或 Kumaraswamy/activation；历史兼容分支会改变继续传播的 latent。
- neighbor categorical：从输入抽 centers，nearest-center 得到类别；id/disc/func/int 分别决定 latent 保持、替换中心、再变换或替换整数。
- softmax categorical：完整输入先 standardize、加随机 bias/scale，再逐行 multinomial；id 保留 latent，disc/int 用随机抽到的类别改变 latent。

可选 Gaussian noise 是每行独立噪声，node importance 再按节点缩放。随机类别和噪声会打破此前相同状态的行，任何压缩执行必须识别这个边界。

### 全表整理及正式过滤

GraphSCM 拼接观测特征，做 outlier 处理、全表标准化、特征置换和类别标签置换。外层去常量列、检查类别及训练/测试覆盖、按需要 padding。正式过滤依然看到原先的 padding 宽度；删掉补零列会改变 ExtraTrees 的特征选择过程。

正式过滤用 25 棵 depth=6、bootstrap+OOB 的 ExtraTreesRegressor；分类目标 one-hot，二分类只保留一列。逐行比较模型 OOB Brier/MSE 与常数预测误差，对 improvement 做 200 次 bootstrap，p≥0.05 拒绝。**不是训练 200 个森林**。可选 remove_trivial 是另一个拒绝条件。

被拒绝后从头重抽候选，直到生成一个成功任务。吞吐必须包括这些重试，不能只测某个算子或成功候选的最后一次尝试。

## 什么冗余值得消除

### 已验证保留：融合全部随机树

过去先生成 [N,T,depth] 分裂张量，再 gather [N,T,O] 叶向量，最后沿 T 归约。新 C++ 循环对每行遍历每棵树的完整路径，直接把全部 O 维叶向量累加到 [N,O]，避免两个中间张量。

没有减少 T、depth、O、函数类型或随机参数。归约顺序允许变化。无 native 库、非 CPU 或其他 dtype 仍走原 Torch 路径。

N=2048、8 workers、512 个成功任务、seed=46 的两轮各自对照：Torch tree 34.721/34.668 s，native tree 32.956/33.013 s，合并吞吐约 +5.18%。这是在已有 native nearest-center 之后的增量，不是 GPU 训练收益。

### 未保留：重复状态的两个执行原型

逐 GP 节点执行 unique，完整吞吐从 15.536 降到 15.159 tables/s。更完善的原型直接从离散化取得类别编码，在确定性节点间传播；保留 full-N fitting，随机噪声/softmax latent 改变时废弃编码，并尝试覆盖 GP、树/二次项及 distance/EM。

编码版本的多轮 wall 差异约 0%～0.8%，没有稳定净收益，故没有保留运行时代码。原因不只是实现细节：免费的状态发现也仅能覆盖部分后继；多父节点合并和随机转换经常破坏低 cardinality，且 GP/融合树在此 CPU 上已经很快。

audit_prior_execution.py --row-reuse 可在原完整候选上测“假如已免费知道重复行，最多能省多少”。128 个成功任务的诊断中，GP 全 transform 约 1.22 CPU 秒，重新 unique 约 4.14 秒；EM/discretization 更值得关注。这个 oracle 含额外执行和缓存效应，不能当作实测吞吐或严格上界。通过观测再挑下一个项目，优于对每个算子一律套去重。

### 未保留：标签优先生成

在完整候选上记录 y 的祖先闭包与各节点时间。若 y 只有一类、非有限，或某类不足两行，那么无论如何排列都不能同时覆盖 train/test；可以设想先算 y，再跳过其余节点。

512 个成功任务、762 次候选、90 次此类无效：即使免费知道无效标签，乐观可省 8.40/227.79 秒，即 3.69%。实际重排、提前拟合和检查都有成本；因此没有为此引入复杂的新执行器。这不能否定“预测性早筛”或其他拒绝策略。

## 已实现：完整表上的渐进过滤

### 流程

1. 按当前流程完整生成 N 行、全部图函数与 converter，执行已有合法性检查。
2. 用完整表拟合正式森林的前 4 棵树，所有超参数与原过滤器一致。
3. 得到逐行 OOB improvement：
   imp_i = ||y_i - mean(y)||² - ||y_i - prediction_4_i||²。
4. 计算筛选分数 z = mean(imp) / (std(imp, ddof=1)/sqrt(N) + 1e-30)。
5. z≥2 且有限：提前接受，省去剩余 21 棵及最终 bootstrap。
6. 其他情况：warm_start 保留前 4 棵，再补到 25 棵，运行原 p-value 决策。没有弱分数早拒绝。

z 是经过任务级留出审计的筛选分数，**不是独立同分布理论保证的 z-test**；共享训练数据使 OOB 行误差相关。零次观察误判也不能证明未来零误判。已有测试验证补齐后的 25 棵树、随机种子与 OOB prediction 等同直接训练 25 棵。

还有一个少树森林的限制：4 棵 bootstrap 树下，一行没有任何 OOB vote 的概率约为 16%，sklearn 对这些行的 OOB prediction 填 0；当前分数沿用这个行为，没有把它伪装成有效的独立留出预测。二分类只保留一个 one-hot 分量，因此缺票的影响会随标签编码变化，不能声称理论保守或标签翻转不变。另补测 N=2048/4096、16 seeds、2% 左右少数类、强信号/纯噪声及标签翻转，共 128 例：与完整过滤无冲突；8 个代表案例进入回归测试。它们检验已知风险，不构成任意不平衡分布的保证。

仅支持已审计的 N=2048/4096、2–10 类、1–100 输入列、classification、25-tree/200-bootstrap unpredictable filter。regression、其他长度/类别数/宽度、自定义树数或 bootstrap 次数、remove_trivial 均走完整过滤。remove_trivial 不能提前接受，因为少量树的分数不能判断补齐后是否接近完美。

统计证据针对当前默认 GraphSCM recipe。更换图、函数族或噪声配置后需要重新审计；形状门槛本身不提供跨分布保证。

该方案改变的是过滤工作的分配，没有使用 cheap SCM，没有删交互，也不拿边际低相关判定 XOR/多跳任务无用。新模式仍有经验性的接受分布风险，因此显式设置 dataset_filter_mode，旧 full 路径保留。

### 为什么它比原 pilot 合理

旧 pilot 首先在 256 行拟合完整 graph；这会改变 centers、标准化、树阈值及类别分布。新方案先完整生成 N 行，完全避免这个问题。

旧小森林筛选后又从头训练正式森林；新方案保留前 4 棵，困难候选只需补 21 棵。weak histogram 用有限边际/低阶探针代替模型；新方案使用正式过滤器自身的前缀，仍可识别其能够识别的非线性关系。

最关键是把“明显有信号”用于**省过滤**，把“不确定”交给完整森林，而不是武断地丢掉弱边际、多跳、稀有类别任务。它不节省图生成，也不声称解决所有拒绝浪费。

### 实验与收益范围

开发审计使用 seed=48、2048 个成功任务、2693 次过滤。旧 parity 划分的开发集 1325 次、验证集 1368 次；z=2 在验证中提前接受 749 次，观察误接受 0，单侧 95% 二项上界约 0.40%。后续工具按参数共享组划分，并支持固定门槛/新 seed 的独立验证，避免回看验证集调参。

N=2048、seed=49、512 个成功任务、8 workers 的 ABBA：

| 模式 | A1 full | B1 progressive | B2 progressive | A2 full |
|---|---:|---:|---:|---:|
| wall s | 32.244 | 29.517 | 29.434 | 32.190 |
| CPU s | 222.911 | 201.174 | 200.680 | 222.223 |
| filter CPU s | 53.204 | 31.292 | 31.129 | 53.135 |
| tables/s | 15.879 | 17.346 | 17.395 | 15.905 |

合并吞吐约 +9.30%，过滤工作约 -41%，总体 CPU 约 -9.72%。每轮都生成 734 次候选、调用 661 次过滤、拒绝 149 次。

N=4096、seed=52、512 个成功任务、8 workers 的独立 ABBA：

| 模式 | A1 full | B1 progressive | B2 progressive | A2 full |
|---|---:|---:|---:|---:|
| wall s | 53.108 | 47.357 | 47.261 | 52.906 |
| CPU s | 371.114 | 327.576 | 327.454 | 370.439 |
| filter CPU s | 88.868 | 46.828 | 46.937 | 88.915 |
| tables/s | 9.641 | 10.811 | 10.833 | 9.677 |

合并吞吐约 +12.04%。635 次过滤中，390 次只拟合 4 棵，245 次补齐 25 棵；候选 704 次、最终拒绝 123 次。两个长度的所有 ABBA 轮次逐任务比较，digest、过滤决定、有效宽度、类别数、熵、最少类比例全部一致。这是这些样本上的观测，并非算法对所有输入保证等价。

阈值冻结为 2 后，另用新 seed 各验证 1024 个成功任务：

| N / seed | 完整候选过滤数 | 提前接受数 | 与完整过滤冲突数 | 名义单侧 95% 误接受率上界 |
|---|---:|---:|---:|---:|
| 2048 / 50 | 1350 | 778 | 0 | 0.384% |
| 4096 / 51 | 1243 | 763 | 0 | 0.392% |

上界是以候选独立为假设的二项 Clopper–Pearson 统计；共享参数和重试会产生相关性，不能把它当作分布无关保证。它衡量与旧过滤器冲突的比例，也不等于最终训练精度损失。完整汇总、运行设置和配对检查见 [结果 JSON](prior_redesign_20261101.json)。

p99 仍约 2.24 秒；这个优化主要提升产能，没有显著消除图生成长尾。CPU 吞吐与训练 step 时间不是同一个指标。

当前 node_pool 的 balanced 路径还存在明确的交付边界：一个 cohort 的全部表完成后才按实际宽度分配；early_dispatch 虽可提前分配，仍等待固定分配给该 rank 的表。DataLoader 则以完整 local batch 为输出单位。下一轮系统改动应研究以完整 micro-batch 为就绪单位，从已有生成池的完成结果中组装下一步，同时设有界等待、保留全部任务和独立 seed，避免慢任务长期饥饿。这是改变交付流程，不是再加接收预取，也不能直接从平均 CPU 收益推算其效果。

## 实际八卡训练：FP32 有收益，BF16 尚无净收益

使用当前 stage1 的完整 prior、8 张 PPU-ZW810、N=2048、batch64/micro4、COMPILE_FRONTEND=all 和每 rank 三次前后向预热。没有使用 shape-prior。FP32 沿用 node_pool/56 workers/balanced，BF16 沿用 DataLoader/每 rank 8 workers/in_order=False。每个 dtype 按 full→progressive→progressive→full 运行，每轮 80 步，跳过前 20 步比较后 60 步。

| dtype / 指标 | full A1 | progressive B1 | progressive B2 | full A2 |
|---|---:|---:|---:|---:|
| FP32 实际步间隔均值 ms | 630.38 | 511.47 | 543.69 | 588.85 |
| FP32 每步 max-rank prior 均值 ms | 139.37 | 37.39 | 57.30 | 110.54 |
| FP32 整轮 wall s | 125.82 | 118.31 | 119.81 | 124.86 |
| BF16 实际步间隔均值 ms | 626.07 | 645.47 | 561.53 | 565.02 |
| BF16 每步 max-rank prior 均值 ms | 276.83 | 285.69 | 211.73 | 220.87 |
| BF16 整轮 wall s | 157.97 | 158.77 | 150.56 | 153.77 |

FP32 合并稳定步 609.61→527.58 ms，吞吐约 +15.55%；max-rank prior 均值 124.96→47.35 ms。BF16 为 595.54→603.50 ms，吞吐约 -1.32%，max-rank prior 248.85→248.71 ms，**没有观察到端到端净收益**。八轮均正常退出，无 OOM/nonfinite loss。不能只挑 BF16 第二组的微小正收益，或将 CPU 的 +9.3% 直接套到 BF16 训练。

wall 包括启动、预热、保存和 worker 退出；它与跳过启动后的步间隔是不同指标。已有编译缓存下，FP32 每轮最慢 rank 的预热启动约 33.81–34.18 s，BF16 约 29.81–30.15 s；所有 rank 并行预热，每 rank 三轮、零 optimizer update，并非串行 24 个训练步。冷缓存首次编译仍可能更久，应计入总训练时间。

这是 80 步短测，预取队列初始状态及长尾有较大影响；BF16 还会因 in_order=False 改变进入相同步号的任务组合。每步 rank 的 train_time 包含等待其他 rank 的同步，不能把它全部解释为本卡算力。完整原始摘要及预热范围存入结果 JSON。当前证据支持显式试用 progressive；不支持自动替换 BF16 默认配置，也不证明长训练/最终模型质量。

当前测量确认应继续看 max-rank，但尚未证明它完全来自“算得慢的 rank”：BF16 的输入等待仍大，可能来自完整 local batch 的生成长尾与交付边界。下一步应对齐 task seed/完成时间、逐步核对各 rank 的到达时间及 GPU busy，再验证前述基于完整 micro-batch 的就绪组装；把同步等待算进 train_time 后再优化 GPU 算子会误判方向。

## 仍要更大提升，下一版该怎样重新设计

### 首选执行器：Plan → full-data fit → typed execution → progressive validation

把图结构、随机函数计划和执行分开，明确每个算子的依赖：

- 参数只依赖维度：随机矩阵、GP frequencies，可以提前生成和分组执行；
- 参数依赖经验分布：标准化、树阈值、centers，必须看到完整数据，或使用保留频数的加权统计/抽样；
- fitted 后逐行确定性：可融合、分块、按编码状态计算；
- 按行随机：必须保留全部行级随机性，不能对同类只抽一次；
- 观测列/过滤边界：只在需要时 materialize，并尽快释放中间状态。

真正的压缩表示应为 values[U,D] + codes[N] + counts[U]。统计按 counts 加权，从原经验分布抽 center/threshold 时也按 counts 抽；多父节点用联合类别编码，独立噪声后回 dense。不能直接对 U 个代表行 uniform fit，这会把频繁状态与稀有状态同等对待，改变任务难度。

不过本轮编码原型及 cost oracle 不支持把“全图编码执行”直接包装成巨大收益。下一步应先以原真实图上的可覆盖计算时间作为投资门槛；可覆盖不足就不重写整个执行器。另一个值得量化的目标是 Python 小对象/采样/fit 调度，将连续的确定性算子融合为执行块、将独立任务按算子形状批处理；必须保留各任务独立随机参数，而不是让很多表共享同一个图来凑吞吐。

本轮额外对 64 个成功任务做 cProfile：总计约 38.77 CPU 秒，nearest native 4.50 秒、torch.cdist 3.98 秒、森林 tree fit 内部 5.58 秒、SimpleRandomWeights.sample 包含子调用 5.13 秒，categorical 调度包含子调用 7.21 秒。后两者有重叠，不能相加。cProfile 会放大小函数开销；这些数字用于定位下一步，不作为无仪器吞吐占比。尤其不能直接缓存 meta categorical 权重：默认历史兼容分支可能每次重新抽权重，缓存会改变跨调用相关性，应作为独立 prior 改动审计。

### 提案分布重设计：降低失败率，不能靠粗暴筛掉难任务

若追求比当前更大的算法收益，应直接研究“高接受率的完整复杂度 proposal”，而不是继续叠加 prefetch：

1. 记录当前完整候选的图、converter、噪声、可观测信号路径、实际拒绝原因和 CPU 成本。
2. 学习/设计一个可观测信号预算：保持原节点数、宽度、全部函数族及方向混合，在观测变量与 y 的共享祖先/路径上分配结构化信号与独立扰动。不能仅保证图上连通；converter 和噪声会消灭实际信号。
3. 显式保留直接预测、共同原因、反向因果、纯交互、多跳、稀有类别等任务桶；不要把“强制 x→y”当成全部方案，也不要把 labels 全改成简单线性分位数。
4. 原完整过滤继续作教师和不确定候选后备；用真实接受率、每成功任务 CPU 与任务覆盖优化 proposal，而不是优化筛选 AUC。
5. 留出新的图 seed 和真实任务；比较同 token/同训练步预算的模型。成功后将其作为新 prior 版本启用。

这会改变提案分布。用户允许这种改变，目标也不应是 KS 全部贴合旧 prior；但“函数数量没少”不等于质量提升。不能在没有训练评估时宣称新 proposal 至少等价。此方案尚未作为运行时代码提交，也不声称已经实现了 2 倍提升。

### 早拒绝的正确优化目标

对一个早拒绝策略，净收益由覆盖率 × 真正可跳过的后续成本，减去 screening、状态保存、重算和额外重试成本决定。AUC 很高也可能没有收益。

先用原完整候选的观测做成本 oracle，再验证冻结 full-data fitting 之后真正的可执行路径；按任务/参数组独立划分开发、验证集，使用 pooled policy 加分层覆盖审计，而不是把所有稀有层切到无法学习。错误预算应关联任务质量，不能设置一个事后任意收紧的门槛，再把全 defer 当成算法失败的普遍结论。

## 怎样衡量生成质量

建议使用两个独立 baseline 样本集合估计自然波动，再与新算法比较。至少报告：

- 覆盖：实际活跃图深度/边数、各函数与 converter、纯交互/共同原因/反向因果、多跳、宽度与 cardinality；
- 数据：有效列、类别数/熵/最少类比例、常量/离群、数值秩和相关结构；
- 任务：linear 与非线性 probe 的留出损失、改善幅度、难度分桶，单独看 XOR、稀有类别和小特征数；
- 资源：每成功任务 CPU、候选/重试数、accepted tables/s、内存、实际 56-worker 负载；
- 训练：相同初始化/数据预算，真实与 synthetic holdout 的 accuracy/NLL/校准，多个 seed；以及每步 max-rank 输入等待、真实步间隔和总 wall。

当前 CPU 测试和任务摘要足以支持“执行加速已测到”，不能单独证明最终模型精度相同。

## 测试及提交范围

目标机器上的生产目录重新验证：`test_prior_execution.py`、`test_prior_filter_cascade.py`、`test_prior_exact.py`、`test_prior_pipeline.py` 共 **170 passed**。覆盖 native 树完整路径/输出、边界与非连续张量、full fallback、4→25 棵对象复用、补齐森林一致性、XOR/多分类、稀有类别/标签翻转、RNG 和训练参数接线。

更广的相关测试累计 416 个选定案例通过（包含上述重复执行的案例，不把重复跑算成新案例）。第一次广测有 10 个失败：8 个是未 source PPU SDK 导致 CUDA 编译环境缺失，2 个是原来的 bitwise Torch 分块测试误调用了浮点归约次序不同的 native 树。补齐环境并把两个旧测试限定回其所验证的 Torch fallback 后，10 个均通过；native 树另由容差/结构/RNG 测试覆盖。编译检查通过，八次真实 DDP 均正常退出。本地补充测试 45 passed、1 个 CUDA-only skip。

新增接收预取、短表 pilot、histogram 校准及编码状态执行原型不进入最终提交；保留诊断脚本与历史结果说明，用于复现成本归因及撤回依据。没有把尚未验证的 signal-budget proposal 包装成完成的运行时算法。

## 使用与复现

先在目标 PyTorch 环境构建一次 native 库，不要在 worker 内 JIT build：

```bash
export PATH=/opt/conda/bin:$PATH
python scripts/build_prior_exact.py --build-directory /tmp/tabicl-prior-redesign-build
export TABICL_PRIOR_DISTANCE_LIBRARY=/tmp/tabicl-prior-redesign-build/tabicl_prior_exact_cpu.so
export PYTHONHASHSEED=42 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/benchmark_prior_exact.py --tasks 512 --workers 8 --rows 2048 \
  --seed 49 --filter-mode progressive --output /tmp/prior-progressive.json
python scripts/audit_prior_filter_cascade.py --tasks 1024 --workers 8 --rows 4096 \
  --seed 51 --fixed-z 2 --output /tmp/prior-progressive-validation.json
```

两条训练 dtype 都可使用同一个 CPU prior 选项：

```bash
PRIOR_FILTER_MODE=progressive bash scripts/train_v2_clf_stage1.sh
```

脚本默认 full。该选项独立于 FP32/BF16、COMPILE_FRONTEND 和原有 prefetch；它不添加接收线程。需要真实训练性能时按相同编译缓存、相同 node_pool 设置、足够长的稳定窗口比较，不能用冷/热启动差异报加速。
