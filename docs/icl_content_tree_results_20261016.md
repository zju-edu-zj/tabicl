# content_tree 实现与实测（2026-10-16）

本次交付可直接运行的 20K–50K 筛选脚本、content_tree CUDA 路由和完整的测量入口。已经完成的是实现正确性、短程学习和性能测试；**尚未完成原尺寸模型的 20K–50K 训练，不能据此认定任何近似 attention 已逼近 dense。** 配方和操作见 [筛选文档](icl_attention_screening.md)。本文性能数字是优化前历史基线；QK norm、direct-index kernel 和最新 1K–64K 结果见 [优化报告](icl_attention_optimization_20261016.md)。

原始结果目录：`/home/zj/table_foundation/exploration/results/icl_content_tree_20261016/`。其中 `pilot200` 是本报告的分类学习实验，`performance` 是性能矩阵。此前 `smoke20` 仅用于调通流程，未用于下面的表现对比。

## 实现取舍与审计

五个主要对比分支为 dense、global、global_sparse_alternating、global_sparse/projection、global_sparse/content_tree；attention 类型列表不再增加一种配方，分区作为正交参数 `icl_partition`。两种 sparse 分区都固定 B=128、k=4，每第三层检索；主对比使用精确 centroid top-k。hierarchical beam 单独测，避免把分区变化和搜索近似混在一起。

content_tree 在每个检索层、每个 head 上只读取当前 support K：先以固定方向投影建立内容顺序，再逐节点从 16 个内容顺序分位点估计远点对，以远点差向量作分割方向，中位划分直到叶大小不超过 B。叶表示是当前原始 K 的均值；query 选择叶后，对选中的原始 K/V 做一次统一 softmax。没有额外可训练码本、路由蒸馏、辅助 loss 或 query 参与建树。

GPU 的节点打分直接按索引读取原 K，避免每层重排整段 FP32 K；只排序标量和索引。分割仍使用 PyTorch GPU stable argsort，建树仍有多级排序成本。根投影仍有一次 K 的 FP32 转换，centroid 搜索仍有 O(N S/B) 项。因此这是高效的内容分区实现，不能声称整条路径严格线性，也不能把 content_tree 和 hierarchical 搜索视为同一件事。

另一个实际问题是在线 prior 每步改变 support 长度：原 Triton 把 N/S 等长度视作编译常量，会持续生成长度特有 kernel。本次把稀疏前反向、反向边表和中心计算中的运行长度改为动态参数；head 维度、块大小、树的几何形状仍作编译常量。动态 support benchmark 同时记录每次 warmup/测量的 kernel cache 变体数，检查是否继续编译。

审计还修复了异常输入的索引安全：NaN 分数在排序中可能落到 +inf padding 后面，导致截断结果包含 sentinel 索引。CPU/CUDA 均将 NaN 分数规范为 +inf，并保持合法索引在 padding 前；训练和验证仍会对非有限输出 fail fast，修复并不掩盖数值失败。

边界：稳定排序在投影碰撞、完全相同 K 跨叶边界时仍可能依赖输入顺序，当前连续随机输入的置换测试不能证明任意重复样本的严格集合不变性。缓存只保存原始 K/V，cached query 目前重新构建派生索引。树路由不对离散决策求导；稀疏 CUDA autograd 只支持一阶梯度。

## 正确性与训练流程

286 项测试通过，9 项跳过。覆盖 FP32/BF16/FP16 与掩码 dense 的输出和 Q/K/V、投影及融合门梯度，CPU/CUDA 路由对齐，尾块及非 2 次幂 beam，热门块反向、非有限输入索引、重计算、多层缓存、query 标签与 query 分批隔离。额外验证了真实 graph_scm prior 在不同 worker 数和断点恢复后的样本一致性，以及真实 Trainer 连续 4 步与 2 步中断恢复后继续 2 步的权重逐位一致性。

正式筛选配置为原尺寸 3/3/12 层，ICL d=512、8 heads，默认 30K steps，可选 20K–50K。默认 routing preset 从随机初始化训练，总行数 log-uniform 1K–8K，support 约 80%；另有保留 fixed-1024 的 stage1 对照。五组使用相同公共权重、按 step/rank 固定的训练任务流、固定验证表和相同学习率计划。脚本可恢复模型、optimizer、scheduler 和数据位置，记录配置/源码/输入文件哈希，拒绝静默改变架构与配方。

## 实际学习表现：三种子 × 五分支 × 200 steps

这是明确标记的 pilot：column/row 各 1 层、ICL 3 层，ICL d=128、8 heads，batch=4，训练总行数固定 1024，features≤8。随机种子 42/43/44；每个长度面板 8 张独立表，每张 64 queries，features 为 5/8；各 seed 使用同一验证集。验证不做集成或校准，CE 使用全部 10 类 logits。16K 是该 pilot 的长度外推，不能替代长上下文训练。

下表是最终 CE 的三个训练种子均值，越低越好。全部 15 次训练完成，原始逐表结果和每 seed 的配对 bootstrap 区间在 `pilot200/summary.json`。

| 分支 | support 256 | 1K | 4K | 16K |
| --- | ---: | ---: | ---: | ---: |
| dense | 1.6489 | 1.5425 | 1.7460 | 1.5344 |
| global | 1.6470 | 1.5470 | 1.7378 | 1.5454 |
| global_sparse_alternating | 1.6597 | 1.5532 | 1.7519 | 1.5426 |
| global_sparse/projection | 1.6581 | 1.5585 | 1.7402 | 1.5394 |
| global_sparse/content_tree | 1.6415 | 1.5476 | 1.7265 | 1.5459 |

16K 的 accuracy 分别为 35.48%、35.35%、35.61%、35.55%、35.16%。以 seed42 为例，dense 的四面板平均 CE 从 2.3135 降到 1.6269，content_tree 从 2.3141 降到 1.6344，确认模型在学习。跨 seed 的 16K CE 标准差分别为 0.0173/0.0162/0.0291/0.0249/0.0204；不能据这些微小差值给出稳定排序，更不能从 200 steps 推断收敛表现。

对真实 pilot 模型的最后一个检索层，取每张表前 16 个 query 做诊断，三个 seed × 八张表均值如下。这里的 relative error 是**稀疏输出**相对相同 Q/K/V 下 dense 输出的误差，并非最终预测误差，也不是融合后的误差。

| support | 分区 | token-top32 recall | retained mass | sparse output relative RMSE |
| --- | --- | ---: | ---: | ---: |
| 4K | projection | 46.13% | 16.10% | 0.464 |
| 4K | content_tree | 79.23% | 17.85% | 0.549 |
| 16K | projection | 31.29% | 4.40% | 0.598 |
| 16K | content_tree | 62.86% | 4.96% | 0.693 |

这个结果很关键：content_tree 确实提高了高分 token 覆盖，但在 16K 上 512 候选只覆盖约 5% 的 softmax 质量，候选内重新归一化仍可产生更大的输出误差。不能用 recall 上升替代“逼近 dense”的证据。融合门均值约 0.505–0.507，200 steps 也不足以判断它能否学会有效纠偏。下一轮正式筛选应同时以 CE/accuracy、retained mass 和输出误差判断，而非只看路由命中率。

回归另做五分支各 20 steps 的链路检查，采用 999 quantiles、无偏置 LayerNorm，验证 support=256/1024/4096、每面板两张表。4K pinball 分别为 dense 0.395915、global 0.396072、alternating 0.395665、projection 0.395891、content_tree 0.396161。该预算仅验证分类以外的训练、验证和 checkpoint 路径可运行，不用于判断回归优劣；原始结果在 `regression20/summary.json`。分类与回归共 20 次运行，日志中未出现非有限值或 OOM microbatch。

## 性能与合成路由结果

性能矩阵 62 个任务全部成功，未出现 OOM 或子进程失败。设备为共享 RTX 5090，进程分配器上限 5 GiB；BF16、ICL d=512/8 heads、12 层、batch=1，support=总行数的 80%，训练启用 activation checkpoint。每个静态任务 warmup=3、测量=7；计时包含前反向和同步，不包含 optimizer/prior。显存是进程 peak allocated，不是整张 GPU 的占用。

| 分支 | 1K ms | 4K ms | 8K ms | 16K ms | 64K ms | 64K peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 18.86 | 24.91 | 90.25 | 234.03 | 3152.46 | 3.022 |
| global | 26.63 | 32.29 | 51.94 | 62.19 | 492.14 | 3.020 |
| global_sparse_alternating | 28.42 | 54.15 | 74.88 | 132.74 | 772.23 | 3.203 |
| global_sparse/projection | 37.04 | 36.49 | 64.19 | 113.62 | 978.30 | 3.457 |
| global_sparse/content_tree | 38.41 | 47.74 | 66.85 | 164.24 | 890.86 | 3.457 |

64K 上 content_tree 约为 dense 的 3.54 倍速度，但其显存略高；dense 本身使用 memory-efficient SDPA，并不物化完整注意力矩阵。短序列 dense 更快。共享 GPU 的外部负载在测量过程中改变：64K 每次样本范围分别为 dense 2796–3508 ms、global 340–531 ms、alternating 609–915 ms、projection 696–1010 ms、content_tree 696–1035 ms。两种分区的范围重叠，不能从单次中位数断言 content_tree 比 projection 快。独占设备上的部署结论需要复测。

完整 TabICL 路径使用 column/row 各 1 层、ICL 12 层、8 features，而正式筛选是 3/3/12，不能混同。缓存测量使用 ICL 12 层，support=13107、cached query=256，包含重建派生索引。

| 分支 | 完整模型训练 1K ms | 完整模型训练 4K ms | prefill ms | cached query ms |
| --- | ---: | ---: | ---: | ---: |
| dense | 34.16 | 65.07 | 31.35 | 3.09 |
| global | 47.46 | 51.98 | 25.10 | 8.89 |
| global_sparse_alternating | 54.53 | 53.90 | 34.04 | 11.61 |
| global_sparse/projection | 36.61 | 39.64 | 46.37 | 13.03 |
| global_sparse/content_tree | 41.50 | 65.16 | 34.84 | 16.46 |

因此本轮优化主要证明长上下文预训练的加速潜力，**没有证明少量 query 的缓存推理加速**。后者当前 dense 更快；缓存中保存全局摘要、排序索引和树，避免每次 query 重建，是独立的后续优化点。

动态 support 实验在 N=16384 上轮换 S=9994/10977/11960/12943/13926，warmup=5、测量=10。content_tree 的 sparse 前向、dQ、dK/dV、边表、centroid 各只缓存一个变体；分割轴和打分各缓存 7 个树深变体。首次调用后改变 S，计数均保持不变，确认没有按这些 support 长度继续生成 kernel。该项的五分支中位耗时分别为 251.59/92.81/109.43/99.59/139.82 ms。

与 portable gathered-SDPA 参考比较，3 层 ICL、4K、recompute 的完整前反向：projection 301.27→12.34 ms，content_tree 361.26→18.56 ms；peak allocated 均为 0.453→0.240 GiB。这里比较的是三层模型路径，不能把倍数写成单个纯 attention kernel 的速度。

路由阶段本身包含分区、中心构建和搜索，下面均取 7 次中位数，hierarchical beam=16：

| 分区/搜索 | N=16K ms | N=64K ms |
| --- | ---: | ---: |
| projection / centroid | 0.91 | 20.07 |
| projection / hierarchical | 4.83 | 21.22 |
| content_tree / centroid | 1.30 | 20.47 |
| content_tree / hierarchical | 6.09 | 26.63 |

当前范围下 hierarchical 没有表现出稳定速度优势，主实验继续使用精确 centroid top-k。全部逐样本耗时、GPU 状态与配置见 `performance/*.json`，汇总见 `performance/summary.csv` 和 `timings.csv`，图见 `scaling.png`/`scaling.svg`。

合成路由测试使用 support=16385/65537、每个 head 64 queries、8 heads、d=64、BF16、三个随机种子；分别测试独立高斯 Q/K/V 和 32 个内容簇、簇内相关 V。下表均是精确 centroid，取种子均值；它衡量路由适合哪种几何结构，不是任务准确率。

| support / 场景 | 分区 | token-top32 recall | retained mass | sparse relative RMSE |
| --- | --- | ---: | ---: | ---: |
| 16K / random | projection | 6.99% | 3.83% | 5.786 |
| 16K / random | content_tree | 12.90% | 5.08% | 5.482 |
| 16K / clustered | projection | 26.19% | 24.55% | 0.287 |
| 16K / clustered | content_tree | 89.26% | 82.91% | 0.059 |
| 64K / random | projection | 2.62% | 1.05% | 11.714 |
| 64K / random | content_tree | 6.65% | 1.57% | 11.016 |
| 64K / clustered | projection | 8.84% | 7.15% | 0.341 |
| 64K / clustered | content_tree | 52.97% | 26.50% | 0.061 |

64K clustered 下，content_tree 候选的同簇比例为 100%，projection 为 29.33%，符合内容分区的设计目标。但 random 下 attention 本来分散，global 单独的相对误差约 0.778，初始 0.5 融合后 content_tree 的误差约 5.481；clustered 下 sparse-only 误差约 0.061，0.5 融合后则为 0.468。固定半半融合并不总是合适，需要训练决定门控，不能把“摘要＋检索”当作未经训练就成立的 dense 近似。

beam=16 在 64K random 上相对精确 centroid 的 leaf recall 为 projection 84.54%、content_tree 94.32%；在 clustered 上分别为 96.78%、99.89%。beam=32 可提高到 93.00%/99.27% 和 99.59%/100%，但当前速度没有支持默认改用树搜索。原始数据见 `quality_16k.json`、`quality_64k.json`。

测试刻意保留尾块：k=4 是最多 512 个原始样本，并非每次恰好 512；当最后叶不足 B 且被选中，候选数会减少，小叶均值的方差也会影响 centroid 分数。因此比较中同时记录 candidate fraction 和真实 visible keys，不能仅根据 k×B 宣称精确相同的有效候选数。

综合本轮证据，保留五个分支进入正式筛选；content_tree 是有实际召回改进的候选，而不是已经验证的默认赢家。优先跑文档中的 routing preset，避免 fixed-1024 训练掩盖检索差异；先以任务指标决定是否保留稀疏增强，再决定是否值得优化缓存索引或继续近似搜索。正式 20K–50K 和下游评估尚未完成。
