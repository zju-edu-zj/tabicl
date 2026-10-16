# ICL attention 审计与性能测试（2026-10-15）

## 结论与默认配置

第一版保留 `dense / global / global_full / global_sparse_alternating / global_sparse` 五种配方，使用同一层组织规则，删除原有 Performer、ICL induced、窗口、池化 MoBA、Flex 等入口及遗留参数。column encoder 的 induced attention 不属于此次清理范围。

建议质量实验默认 `global_sparse + centroid`，块大小 128、每 query/head 选 4 块、每 3 层增强一次。训练显式使用 BF16；长序列启用 recompute。`hierarchical` 保留为近似路由实验，beam 默认 16，最终候选仍为 4 块。不要把 beam 改大误当作读取更多原始样本，也不要根据复杂度直接认定树搜索更快。

本次完成实现正确性、训练前后向、长序列扩展、峰值显存、路由开销、合成路由覆盖率、完整小型 TabICL、缓存推理与 fallback 测试。**没有进行预训练质量实验，不能据此断言摘要＋512 行已经逼近 dense。**

## 审计发现及对应修改

| 发现 | 本次处理 | 仍需注意 |
| --- | --- | --- |
| 同一巨型实现混合多种不同 attention、归一化与路由语义 | 拆分配方、attention、路由 kernel、稀疏 kernel；统一 CLI 和六个训练启动脚本 | 旧实验参数不做静默别名转换 |
| 全局与原始样本融合需要明确归一化语义 | 共享 Q/K/V；全局为 ELU+1，原始样本为一次候选集 softmax；独立可训练融合门初始 0.5 | parameter-less 指块路由，不指融合门 |
| 逐 token、逐维累积 dK/dV 的原子更新在热门块上竞争 | FP16/BF16 反向先反转 query→block 路由边，再由块独占计算 dK/dV；tensor-core GEMM、FP32 累积；dQ 单独重算 | 需要额外紧凑边表，显存并不一定下降；FP32/小输入保留直接反向 |
| 中心计算物化整段排序后的 FP32 K | Triton 从原始 K 与排序索引直接归约中心 | 投影排序本身仍包含 K 的 FP32 转换 |
| centroid 的小 query 分片产生大量 GEMM/top-k 调度 | 自动按最多 32 MiB 分数工作区分片 | 长度增长后仍有 N×ceil(S/B) 路由成本 |
| hierarchy 每层 materialize N×beam×d 候选 | 每 query/head 一个 Triton program，寄存器中逐层 beam search | 这是打分与排序，不是高吞吐大 GEMM；不保证快于 centroid |
| 树的尾块和 padding 需要保持中心语义 | 保留按真实行数归约及按 token 数加权的定义，补充尾块、空子树和非整树测试 | 父均值分数不是后代分数上界，beam search 不是精确搜索 |
| beam 与读取预算耦合 | 分开 `router_beam` 和 `num_selected_blocks` | 为和旧实现比较，配对测试特意使用 beam=4；新默认是 16 |
| 候选预算覆盖全部 support 仍排序/搜索 | 直接调用 full SDPA，跳过路由 | global_sparse 的全局分支仍正常计算 |
| 融合分支诊断状态可能在全局路径提前消耗 | 修复诊断开关传播；覆盖 checkpoint 重放 | 开启诊断会额外读取少量 query 对全部 support；性能表均关闭诊断 |
| 不同配方或 router 可以在参数形状相同下错误恢复 checkpoint | 非 dense 实现版本升为 3；resume 检查配方、层周期、块预算、router 和有效 beam | 旧实验 checkpoint 要开启新的训练目录 |

树搜索没有新增学习参数、teacher、蒸馏或辅助损失。排序方向是固定随机 buffer，存入 checkpoint；中心来自本次 K，query 点积决定硬 top-k。排序/top-k 不反传，实际读取的 Q/K/V 正常学习。

## 测试条件与证据范围

- 单张 NVIDIA RTX 5090，PyTorch 2.9.1+cu130，stdplm 环境，4 个 CPU 线程，TF32 允许，FlashAttention 3 未安装；dense 使用 PyTorch 自动选择的 SDPA。
- GPU 有其他任务运行，显存与利用率波动。未停止或修改其他进程。每个 benchmark 独立进程串行运行，PyTorch allocator 上限 8 GiB。
- 默认 B=1，ICL d_model=512，8 heads，head_dim=64，FFN=1024，support=floor(0.8×总行数)。总行数包含 support 和 query；attention 的 K/V 仅来自 support。
- 每个新测试预热 3 次、测量 7 次，排除首次编译。同步 GPU 后用 wall clock 测量，包含 Python 调度；训练为前向＋反向，不含优化器 step、数据生成、验证和日志。计时外检查输出及梯度有限。
- 报告 median，原始 JSON 保留每个样本、min/max、allocated/reserved 显存及 GPU 状态。单次运行内的范围也无法消除不同进程之间的资源竞争，下面的速度比仅描述这些实测。
- 数值测试使用固定候选集合与 masked dense/reference 比较；不能用 dtype 改变后 top-k 恰好一致作为数值正确性的前提。

原始数据、CSV、图像和优化前源码快照在 [结果目录](../../../results/icl_cleanup_20261015/)，五分支扩展曲线如下。阴影为本次重复样本的最小至最大值。

![ICL 训练时间与显存](../../../results/icl_cleanup_20261015/scaling.png)

## 五分支训练扩展

12 层，BF16，recompute=True，centroid，128×4 候选。每个单元格为前向＋反向中位数，单位 ms。

| 分支 | 1,024 行 | 4,096 行 | 16,384 行 | 65,536 行 | 131,072 行 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 16.21 | 32.54 | 226.01 | 2,860.60 | 10,690.93 |
| global | 23.50 | 28.06 | 83.29 | 361.82 | 728.15 |
| global_full | 25.58 | 33.25 | 145.72 | 1,271.23 | 4,237.98 |
| global_sparse_alternating | 25.51 | 29.52 | 135.91 | 643.37 | 1,290.67 |
| global_sparse | 29.33 | 45.94 | 149.46 | 712.26 | 1,531.30 |

同一组测试的峰值 allocated 显存，单位 GiB，包含参数、激活、梯度和临时工作区，不包含其他 GPU 进程。

| 分支 | 1,024 行 | 4,096 行 | 16,384 行 | 65,536 行 | 131,072 行 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.232 | 0.312 | 0.845 | 3.022 | 5.925 |
| global | 0.250 | 0.360 | 0.976 | 3.020 | 5.921 |
| global_full | 0.253 | 0.382 | 1.017 | 3.276 | 6.433 |
| global_sparse_alternating | 0.250 | 0.360 | 0.951 | 3.203 | 6.288 |
| global_sparse | 0.253 | 0.390 | 1.016 | 3.457 | 6.796 |

这些测量支持长上下文计算量改善：64K、128K 的 global_sparse 分别约为 dense 的 4.0×、7.0× 速度。但短表存在路由、摘要与 kernel launch 开销，1K/4K 不应承诺加速。128K 的 global_sparse 测量范围为 1,374–1,738 ms，dense 为 10,320–10,866 ms；16K 的 global_sparse 为 103–152 ms，跨运行噪声更明显。

显存结论与时间结论不同：dense SDPA 没有保存完整 N×S attention 矩阵；本次稀疏实现还需排序后的 K/V、路由索引和反向边表。在 recompute 下，FFN、残差及输入激活也占很大比例。因此没有测得 global_sparse 相对 dense 的显存优势，128K 反而高约 0.87 GiB。不能用 attention 的理论稀疏度替代整个模型的峰值显存。

## 优化前后配对

原有 `elu_linear_flex_moba` 与新的 `global_sparse`，3 层、BF16、recompute=False、无 Q/K normalization、无额外温度缩放、无额外 dense threshold，128×4，hierarchy 两侧均 beam=4。新默认 beam=16 不用于这组前后比较。每个条件相邻运行旧版和新版，均 3 次预热＋7 次测量。

| router | 总行数 | 旧版总耗时 ms（范围） | 新版总耗时 ms（范围） | 旧版反向 ms | 新版反向 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| centroid | 16,384 | 47.87（45.21–50.16） | 29.21（26.21–30.64） | 36.22 | 18.23 |
| centroid | 65,536 | 171.17（139.89–287.76） | 129.57（128.73–135.28） | 131.05 | 77.27 |
| hierarchical | 16,384 | 36.11（35.85–38.35） | 28.50（27.80–29.04） | 24.48 | 17.81 |
| hierarchical | 65,536 | 229.38（201.78–236.71） | 114.92（112.08–115.35） | 149.90 | 73.32 |

两个方向均有改善，64K 尤其体现在反向。64K 峰值 allocated 从旧版约 3.696 GiB 增至新版约 3.758 GiB，符合用紧凑边表换反向速度的实现选择。

这是相同形状和数学配置的架构性能比较，不是固定同一套投影权重及逐元素梯度的对照实验；数值等价由单独的 reference 测试覆盖。旧 harness 将 zero_grad 计入时间，新 harness 在计时外清空梯度，存在少量测量口径差异。旧数据的 forward、backward 中位数相加不必等于总时间中位数。

`paired_before_*.json` 和 `paired_after_*.json` 是该表依据；更早的 `before_*.json/after_*.json` 仅保留为探索记录。优化前完整 src 与 benchmark 在 `before_source.tar.gz` 中，SHA256：`aaad809a751efee23eb8b8aa9a7282e0676cfb63fd259c074ef5eabe3725ed77`。该快照来自本次修改前工作区，包含当时已有的用户修改。

## centroid 与 hierarchy 的实际权衡

12 层 global_sparse、BF16、recompute=True，最终候选同为 512 行，hierarchy beam=16：

| 总行数 | centroid 总耗时 ms（范围） | hierarchy 总耗时 ms（范围） |
| --- | ---: | ---: |
| 16,384 | 152.68（147.25–155.26） | 167.69（164.71–169.60） |
| 65,536 | 550.63（486.73–976.07） | 586.93（559.71–1,039.79） |
| 131,072 | 1,434.43（1,293.97–1,436.92） | 1,360.78（1,171.57–1,641.41） |

另补测 CLI 默认的 4 heads、head_dim=128，其他设置相同，总行数 65,536：

| 分支 / router | 总耗时 ms（范围） | 峰值 allocated GiB |
| --- | ---: | ---: |
| dense | 3,005.98（2,671.65–3,751.19） | 3.021 |
| global_sparse / centroid | 732.68（723.31–744.86） | 3.445 |
| global_sparse / hierarchy，beam=16 | 838.63（719.98–858.29） | 3.445 |

这组更大 head dimension 的长序列训练也通过有限性检查，没有改变当前默认 centroid 的选择。

单独调用 route、包含投影排序、中心计算以及树构建/搜索，不含 attention 或反向：

| 总 query 行数 | support 行数 | centroid ms | hierarchy beam=16 ms |
| --- | ---: | ---: | ---: |
| 16,384 | 13,107 | 2.93 | 4.56 |
| 65,536 | 52,428 | 13.61 | 30.06 |
| 131,072 | 104,857 | 39.36 | 65.65 |
| 262,144 | 209,715 | 129.74 | 114.19 |

整层时间和 route-only 是不同输入、不同时间段的独立测量，不应直接相减推断 kernel 成本。当前 centroid 的大 GEMM 很高效，树搜索的低阶复杂度要到更长序列才可能抵消访问和排序常数。131K 以下没有稳定的 hierarchy 优势；262K route-only 才测到约 12% 的降低，尚未覆盖 262K 完整训练。默认继续使用 centroid。

还需要看搜索误差。以下用 65,537 support、64 query、8 heads、d=64、BF16、128×4，3 个随机种子测试；clustered 是 32 个随机中心加 0.2 标准差噪声。全部为合成未训练 Q/K。单位 %：

| 数据 | router / beam | 相对精确 centroid 的叶 top-4 recall | dense token top-32 recall | 保留 dense softmax mass |
| --- | --- | ---: | ---: | ---: |
| random | centroid | 100.00 | 2.62 | 1.05 |
| random | hierarchy / 4 | 59.13 | 2.42 | 1.01 |
| random | hierarchy / 16 | 84.54 | 2.59 | 1.04 |
| random | hierarchy / 32 | 93.00 | 2.62 | 1.05 |
| clustered | centroid | 100.00 | 8.84 | 7.15 |
| clustered | hierarchy / 4 | 65.02 | 8.40 | 6.91 |
| clustered | hierarchy / 16 | 96.78 | 8.82 | 7.12 |
| clustered | hierarchy / 32 | 99.59 | 8.83 | 7.15 |

候选平均占 support 约 0.71%–0.77%，不是每次都恰好 512 行，因为选中尾块时只有真实行有效。该结果说明两件事：beam=4 的叶搜索偏差不可忽略；即使精确 centroid，单投影排序＋均值中心也不等价于 dense token 检索。合成结果没有证明这种分块足以抓住训练后重要样本。首轮预训练必须一起观察检索 recall、mass、融合 gate 与下游指标，再决定是否需要更好的分块；不应先堆路由蒸馏。

## 完整模型与推理

完整小型 TabICL：8 列、embed_dim=128、row_num_cls=4、column/row encoder 各 1 层、ICL 3 层，BF16，recompute=True，分类输入；测量模型前后向，不含优化器。它覆盖特征到 ICL 的连接，但不代表实际大模型整步成本。

| 总行数 | dense ms | global ms | global_sparse ms | 峰值 allocated GiB（各分支约） |
| --- | ---: | ---: | ---: | ---: |
| 4,096 | 17.36 | 21.46 | 19.75 | 0.74 |
| 16,384 | 73.25 | 48.20 | 59.25 | 2.75 |

16K dense、global_sparse 的重复范围分别为 69–141、57–106 ms，噪声较大。完整模型的非 ICL 组件分摊成本，不能把 encoder 的加速比直接套用到整个训练。

独立 ICL encoder 3 层，BF16，无反向：prefill **只处理 support 并写 K/V cache**；query 测量已经预填充的 cache 上 256 行 query，预填充时间不计入 query。

| support 行数 | 分支 | prefill ms | cached 256 query ms |
| --- | --- | ---: | ---: |
| 13,107 | dense | 15.50 | 0.60 |
| 13,107 | global | 4.50 | 3.39 |
| 13,107 | global_sparse | 8.74 | 2.73 |
| 52,428 | dense | 174.87 | 1.02 |
| 52,428 | global | 19.81 | 4.09 |
| 52,428 | global_sparse | 29.36 | 9.47 |

目前 cache 只存原始 K/V，每次调用仍重建摘要、排序与中心/树；小 query 任务下 dense SDPA 反而更快。训练加速与 cached-query 延迟是两个问题。如果后续优化推理，应首先把每层 ELU 状态、排序后 K/V、中心/树作为 support 派生缓存保存，再测 query batching，而不是继续更换 attention 名称。

## 精度、预算、fallback 与 profiler

以下均 global_sparse + centroid；不同测试运行时间段不同，用来发现配置敏感性，不作为严格的倍数对照。

| 配置 | 条件 | 总耗时 ms | 峰值 allocated GiB |
| --- | --- | ---: | ---: |
| FP32 | 16K，3 层，无 recompute | 58.21 | 1.471 |
| FP16 | 16K，3 层，无 recompute | 19.73 | 1.058 |
| BF16 | 16K，3 层，无 recompute，配对新版 | 29.21 | 1.058 |
| BF16，候选 1024 行 | 16K，3 层，无 recompute | 42.36 | 1.058 |
| BF16，head_dim=128、4 heads | 16K，12 层，recompute | 165.60 | 1.015 |
| BF16，禁用 Triton 的 portable 路径 | 4K，3 层，无 recompute | 213.69 | 0.652 |

FP16 某次样本最高 43.63 ms；不能据其 19.73 ms 中位数宣布 FP16 稳定优于 BF16。BF16/FP16 才启用 tensor-core 块反向；FP32 保留直接反向。训练脚本原有 float32 默认未改变，性能实验要显式设置 `TRAIN_DTYPE=bfloat16`。portable 路径用于正确性、CPU 与不支持形状，不能用于外推长上下文速度。

已导出两个 router 的 profiler trace 和 CPU operator 表；**当前环境实际采集到的 CUDA profiler events 为 0**，JSON 明确记录 `profile_cuda_events=0`。这些 trace 只能审计 CPU 调度，不支持 kernel 耗时占比或带宽结论。报告的 GPU 同步 wall-clock 计时独立于 profiler，以上表格均未把 profiler 时间计入。

## 正确性验证和已知边界

针对本次改动及相邻推理模块，运行 `test_icl_attention.py`、`test_linear_attention.py`、`test_icl_attention_optimized.py`、`test_icl_training_diagnostics.py`、`test_inference.py`、`test_inference_memory.py`。覆盖：

- CPU/CUDA、FP32/BF16，与相同候选的 masked dense 比较 forward、输入梯度及全部 attention 参数梯度；全局路径另与显式 N×S ELU 核比较。
- CUDA 中心/树与 portable reference、多 batch/head、非满尾块、非二次幂叶数、非二次幂 beam；beam 覆盖全部叶时等于 centroid。
- FP16/BF16、d=16/32/64/128 的新反向；热门块竞争、随机路由、仅一行的尾块，query 数超过快路径阈值；BF16/FP16 梯度相对范数容差分别为 0.008/0.001。
- 五配方、两种 norm 顺序、多层 support/query 可见性、query 分批、cache 等价与连续随机输入置换；全预算直接 SDPA。
- 梯度 checkpoint 重放、长 support、诊断在重算中保留；分类/回归 Trainer microbatch、CPU/CUDA、恢复配置冲突、公开参数集和六个启动脚本。

最终结果为 **177 passed、9 skipped**；跳过项是本机不可用的 XPU/MPS 后端。另通过改动文件的 Ruff F 检查、Python 编译检查、六个启动脚本及共享脚本的 bash 语法检查和 `git diff --check`。未运行需要下载官方权重的整个项目测试集。

非 dense 要求 dropout=0、SSMax=False；稀疏自定义 autograd 只支持一阶梯度；fused 路径支持 head dimension 16/32/64/128、候选行数≤2048，其他形状回退。B 必须是至少 16 的二次幂。

投影完全相等但内容不同的样本使用稳定排序，跨块边界时可能受输入顺序影响；因此当前并没有对任意浮点碰撞输入证明严格集合不变性。连续随机置换测试不能覆盖该问题。近似相等分数也可能因低精度或归约顺序改变离散 top-k。以上是明确边界，没有用较宽输出容差掩盖候选集合不同。

## 复现

在仓库目录执行，所有 Python 使用 stdplm：

```bash
# 五配方主矩阵；建议串行，避免自行制造 GPU 竞争。
for attention in dense global global_full global_sparse_alternating global_sparse; do
  for rows in 1024 4096 16384 65536 131072; do
    OMP_NUM_THREADS=4 PYTHONPATH=src conda run -n stdplm python \
      scripts/benchmark_icl_attention.py --attention "$attention" --rows "$rows" \
      --blocks 12 --recompute --output "/tmp/matrix_${attention}_${rows}.json"
  done
done

PYTHONPATH=src conda run -n stdplm python scripts/benchmark_icl_router_quality.py \
  --output /tmp/router_quality.json

PYTHONPATH=src conda run -n stdplm python scripts/summarize_icl_benchmarks.py /tmp

OMP_NUM_THREADS=4 PYTHONPATH=src conda run -n stdplm python -m pytest -q \
  tests/test_icl_attention.py tests/test_linear_attention.py \
  tests/test_icl_attention_optimized.py tests/test_icl_training_diagnostics.py \
  tests/test_inference.py tests/test_inference_memory.py
```

其他子实验通过 `--scope routing|tabicl`、`--phase prefill|query`、`--router hierarchical --router-beam 16`、`--dtype`、`--nhead`、`--selected-blocks`、`--portable` 复现，完整命令参数保存在对应 JSON。正式选型下一步应固定训练 token、数据与种子完成五分支质量对比；本次结果只决定实现路径、可行长度和需要关注的性能瓶颈。
