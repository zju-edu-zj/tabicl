# ICL attention 第一版实验

公开接口只保留五种 `icl_attention`。support 与 query 都只读取 support；query 之间不做 attention。

| 分支 | 普通层 | 每第 3 层（可配置） | 目的 |
| --- | --- | --- | --- |
| `dense` | full softmax | full softmax | 基准 |
| `global` | ELU+1 全局摘要 | ELU+1 全局摘要 | 只压缩全局信息是否足够 |
| `global_full` | 全局摘要 | 摘要与 full softmax 融合 | 混合结构的完整读取对照 |
| `global_sparse_alternating` | 全局摘要 | selected-row softmax | 同层融合是否必要 |
| `global_sparse` | 全局摘要 | 摘要与 selected-row softmax 融合 | 全局摘要＋少量原始样本 |

12 层、`icl_sparse_every=3` 时，增强层是第 3、6、9、12 层。训练 CLI 默认 `global_sparse`；`TabICL()` 默认仍为 `dense`，保留官方构造方式。实验配方固定 column 为 `qassmax-mlp-elementwise`，ICL 为 `icl_ssmax=False, icl_qk_norm=True`；二者不共享归一化选项。global_full 是完整读取对照，不是数学意义上的质量上界。

## 计算与路由

全局分支使用 `phi(x)=max(ELU(x)+1, 1e-6)`，在 FP32 中累积 `phi(K)^T V` 与 `sum(phi(K))`，query 读取归一化核 attention。长 support 使用分块和重算反向，避免保存整段 FP32 特征映射计算图。

稀疏分支按 head 组织 support，读取选中叶块中的**原始 K/V**，在整个候选集合执行一次 softmax attention。ICL Q/K 分别在 head dimension 上做 RMSNorm；每个 head 有可学习的基础温度和非负 log-length slope，最终相对温度被限制在 `[0.5, 2]`。dense 以全部可见 support 长度缩放 Q，sparse 以候选预算缩放 Q，global ELU 只使用归一化 Q/K、不使用 softmax 温度。默认每块 128 行，每 query/head 选 4 块，最多读取 512 行；尾块只读取有效行。不加入邻近行、当前位置或 query 自身。当全部 support 装得下候选预算时，直接调用 full SDPA，跳过排序与路由。

融合层共享 Q/K/V，用 `g=sigmoid(q·w_head+b_head)` 得到每 query/head 的门，输出 `g*sparse+(1-g)*global`，再执行共享输出投影。w、b 初始化为 0，两路初始各占 0.5。融合门可训练；**块路由没有可训练参数**。不需要 teacher、蒸馏或辅助路由损失。硬排序/top-k 不求导，Q/K/V 通过实际 attention 路径学习。

`icl_partition` 决定叶块；`icl_router` 决定检索方法，两者独立组合：

1. `projection`：每 head 对 support K 在一个固定随机方向的 FP32 投影排序，按块分组。方向作为 checkpoint buffer 保存。
   `content_tree`：从上述内容排序开始，在每个节点抽取 16 个内容顺序上的等距代表点，用两次最远点选择估计分割轴，按该轴投影平衡二分，直到叶块容量为 B。每个检索层/head 都用当前 support K 重建，query 和 query 标签不参与构建；没有可训练码本、k-means 迭代、蒸馏或路由辅助 loss。它没有按输入行号分块。
2. 叶中心为块内原始 K 的 FP32 均值，尾块除以真实行数。
3. `centroid` 对全部叶中心精确计算 query 点积并 top-k。它不是 token-level 精确 top-k。
4. `hierarchical` 对相邻叶块构建二叉树，父中心按真实 token 数加权；逐层保留独立的 `icl_router_beam` 个节点，最后选 k 叶。默认 beam=16、k=4。

父中心分数不是后代分数上界，树搜索可能漏掉高分叶；beam 覆盖所有叶才退化成精确 centroid 搜索。beam 与最终候选预算不再绑定。两者都需要排序建索引：设 query 数 N、support 数 S、维度 d、块大小 B、候选块 k、beam w，排序为 O(S log S)，centroid 路由为 O(N ceil(S/B)d)，hierarchical 路由为 O(Nwd log ceil(S/B))，候选 attention 为 O(NkBd)。centroid 在超长序列仍有二次项；包含排序后 hierarchy 也不是严格线性。

上述 O(S log S) 是 projection 的建块成本。content_tree 有 L=ceil(log2(ceil(S/B))) 层，额外投影约 O(SdL)，逐层局部排序的比较排序上界 O(S log(S)L)，并非严格线性构建。实际 CUDA 使用分段的非稳定 radix argsort；树深和补齐容量为二次幂，只有标量分数与索引被重排。尾叶不足 B 行、补齐空叶均不参加检索。这里只在 16 个代表点中近似选择分割轴，没有扫描全节点寻找精确最远点。

## 高性能路径和明确边界

- CUDA 中心 kernel 从原始 K 和排序索引直接归约，避免完整排序后的 FP32 K。centroid 分数按最多 32 MiB 工作区分片。
- projection 与 centroid 打分直接读取低精度 Q/K、以 FP32 累积，不再先生成整段 `q.float()`/`k.float()` 副本；路由顺序和叶 ID 全部使用 int32。
- content_tree 的节点代表点/分割轴与投影打分使用 Triton kernel，直接读取原始 K，避免逐层产生完整 N×d FP32 重排张量；局部标量排序保留 PyTorch 实现。CPU/CUDA 都有参考结果检查。
- 树搜索每 query/head 一个 Triton program，在寄存器内逐层搜索，避免构造全量 N×beam×d gather。
- 稀疏 forward 不物化 N×kB×d gather。FP16/BF16、N≥256、B≥32 的反向反转紧凑路由边，由叶块独占计算 dK/dV，以 tensor-core GEMM 和 FP32 累积替代逐 token 原子竞争；dQ 独立重算。小输入、FP32 保留直接反向。
- global ELU 的 CUDA forward 融合特征映射、support state 和 query apply，不保存整段 FP32 `phi(Q/K)`；融合分支也把 gate、两路混合和 BHND→BND 布局变换合成一个 kernel。
- non-reentrant activation checkpoint 首次 forward 记录 int32 离散路由，backward 重放同一选择，不再重建树、排序和 top-k；连续张量仍按 checkpoint 语义重算。
- fused sparse 支持 head dimension 16/32/64/128、kB≤2048；其他配置和 CPU 使用分块 gathered SDPA 参考路径，其长序列性能不具代表性。B 必须为 ≥16 的二次幂，hierarchy beam≥k。
- 非 dense 配方当前要求 dropout=0，避免不同分支使用不一致的 dropout 语义；自定义稀疏反向只支持一阶梯度。低精度反向按 FP16/BF16 容差验证，不保证逐 bit 一致。
- query 独立，可切分推理；缓存保存原始每层 K/V。当前每次 cached-query 调用仍重建排序、摘要和索引，尚未缓存派生状态。
- **投影完全相等的不同样本没有定义 tie-break，跨叶边界时可能受 kernel 与原输入顺序影响。**连续随机输入置换测试不能证明任意碰撞/重复输入的严格不变性。
- 离散 top-k 对近似相等分数敏感，dtype 或归约顺序可能改变选择。masked dense 数值测试固定同一候选集合比较输出与梯度。
- 稀疏 forward/backward、反向路由边和中心 kernel 的 N/S/相关 stride 使用运行时参数，避免在线 prior 每步长度变化导致逐长度重新编译。content_tree 树深、节点宽度仍使用少量二次幂特化。

## 配置、代码和复现

保留路由参数 `icl_sparse_every`、`icl_block_size`、`icl_num_selected_blocks`、`icl_partition`、`icl_router`、`icl_router_beam`；`icl_query_chunk_size` 只控制参考路径工作区。诊断采样配置独立存在。旧 Performer、ICL induced slots、窗口/池化 MoBA/Flex 分支及其 normalization、scale、threshold 参数已删除，不提供静默别名。column encoder 原有 induced attention 保留。

代码职责：`icl_attention.py` 定义路由与融合；`_content_tree.py` 编排平衡内容分割；`_routing_triton.py` 实现内容分割打分、中心和树搜索；`_sparse_triton.py` 实现 selected-row softmax；`layers.py` 保留共享投影与全局摘要；`encoders.py` 组织逐层配方。CLI 和训练脚本使用同一套配置。

在仓库目录运行，所有 Python 使用 stdplm 环境：

```bash
TRAIN_DTYPE=bfloat16 RECOMPUTE=True \
  ICL_ATTENTION=global_sparse ICL_ROUTER=centroid \
  conda run --no-capture-output -n stdplm bash scripts/train_v2_clf_stage1.sh

PYTHONPATH=src conda run -n stdplm python scripts/benchmark_icl_attention.py \
  --attention global_sparse --router hierarchical --router-beam 16 \
  --rows 65536 --blocks 12 --recompute --output /tmp/icl_hierarchy.json

OMP_NUM_THREADS=4 PYTHONPATH=src conda run -n stdplm python -m pytest -q \
  tests/test_icl_attention.py tests/test_linear_attention.py \
  tests/test_icl_attention_optimized.py tests/test_icl_training_diagnostics.py
```

训练脚本保留原有 `TRAIN_DTYPE=float32` 默认值；上述性能实验应显式设为 bfloat16，才能使用新的 tensor-core 反向路径。hierarchical 是可选近似加速实验，当前实测并不始终比 centroid 快。

非 dense checkpoint 实现版本现在是 4，旧实验 checkpoint 不允许直接 resume 为新配方。恢复训练检查分支、ICL QK norm、SSMax、partition 及相关路由语义，避免形状相同而含义变化。官方 dense 权重投影参数名称与形状保留；启用 QK norm 会新增参数，因此不能加载成旧结构后静默续训。

先固定数据、训练 token 数和种子比较五分支，再在 global_sparse 内比较 router。记录 validation metric、训练步耗时、峰值显存及抽样 token-top-k recall、retained softmax mass、候选覆盖率。随机输入性能测试不能证明逼近 dense；质量结论必须由预训练与下游评估给出。性能结果见 [性能测试报告](icl_attention_performance.md)。

五分支 20K–50K step 的受控运行、短程学习验证和 content_tree 性能测试见 [筛选实验说明](icl_attention_screening.md)。本轮 kernel/QK-norm 优化及 1K–64K 结果见 [优化报告](icl_attention_optimization_20261016.md)。global_full 保留为可选结构对照，不占用默认五组的训练预算。
