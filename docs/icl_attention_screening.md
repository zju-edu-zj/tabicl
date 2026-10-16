# ICL attention 筛选与 content_tree

实现入口是 `scripts/screen_icl_attention.py`。默认运行 dense、global、global_sparse_alternating、global_sparse/projection、global_sparse/content_tree 五组，默认 30K optimizer steps；正式预算限制为 20K–50K。分类和回归分别运行、分别建立 checkpoint。

实现审计、正确性检查和实际测量结果见 [2026-10-16 测试报告](icl_content_tree_results_20261016.md)。

## 运行方式

在 tabicl 仓库目录运行，所有 Python 使用 stdplm：

```bash
# 查看全部参数，不创建训练任务。
PYTHONPATH=src conda run -n stdplm python scripts/screen_icl_attention.py plan \
  --output /home/zj/table_foundation/exploration/results/icl_screen30k \
  --steps 30000 --seeds 42 43 44

# 生成共同初始化、固定验证集，随后顺序运行五组 × 三个种子。
OMP_NUM_THREADS=4 PYTHONPATH=src conda run --no-capture-output -n stdplm \
  python scripts/screen_icl_attention.py run \
  --output /home/zj/table_foundation/exploration/results/icl_screen30k \
  --steps 30000 --seeds 42 43 44 --memory-limit-gib 0

# 原始 fixed-1024 stage-1 分布对照，使用单独目录。
OMP_NUM_THREADS=4 PYTHONPATH=src conda run --no-capture-output -n stdplm \
  python scripts/screen_icl_attention.py run --preset stage1 \
  --output /home/zj/table_foundation/exploration/results/icl_screen30k_1024 \
  --steps 30000 --seeds 42 43 44 --memory-limit-gib 0
```

`--num-gpus` 控制每组 DDP 进程数；`--batch-size` 是全局 batch，默认 routing=16、stage1=64。`--micro-batch-size` 默认 1，可按显存调整，但每个对比分支必须一致；要求全局 batch 能被 GPU 数和 microbatch 整除。`--workers` 默认 4。同一条 run 命令可恢复 optimizer/scheduler 和 step 对应的数据序列；改变训练预算、初始化或验证集会拒绝复用现有 recipe。`prepare` 只生成共享输入，`summarize` 只重算对比报告，`--variants` 支持单独运行未完成的分支。

默认每进程显存分配器上限 8 GiB；上面专用 GPU 命令用 0 关闭上限。这个选项不限制其他进程。当前的共享 RTX 5090 测试使用较低上限，不应把该上限理解成完整配方的显存需求。

## 为什么默认不是只训 1024

默认 `routing` preset 从随机初始化开始，采用总行数 log-uniform [1024,8193)、support 比例 [0.79,0.81)，最大 100 features。512 行候选预算在这种分布下会真正排除大量 support。它是架构筛选分布，不是官方 stage2 checkpoint 续训，也没有宣称修改后的完整 stage1 curriculum 已被验证。

`stage1` preset 保留总行数 1024、support 比例 [0.3,0.9)、batch=64。约三分之一的表 support≤512，直接走 full SDPA；其余也只有 5–8 个叶块，选 4 块。这种分布适合检查短上下文学习能力，但不能单独判断长上下文路由优劣。建议先以 routing 的结果淘汰明显失效结构，再结合 stage1 对照选择进入完整 curriculum 的模型。

两种正式 preset 均采用原尺寸网络：column 3 层、row 3 层、ICL 12 层、ICL d=512/8 heads、B=128/k=4、每第三层检索；BF16 + activation checkpoint + dynamic `torch.compile`，Muon、LR=8e-4、1% warmup、完整筛选预算内单周期 cosine。所有组关闭 ICL SSMax、dropout、FlashAttention-3，启用 bounded ICL QK norm；column 固定 `qassmax-mlp-elementwise`。两个 preset 的 batch/长度不同，不能跨 preset 直接比较原始 wall time 或 loss。不同预算也必须各自重新运行，不能用 30K cosine 的中途点冒充 20K 完整实验。

## 控制条件与记录

公共权重来自同一份未训练 dense 初始化，包括 column/row、QKV、输出投影、FFN 和预测头；混合模型只新增初始化为 0 的融合门和固定 route buffer。sparse 两种 partition 的投影方向也按模块名和种子固定。禁止把已训练 dense checkpoint 当共同初始化，以免比较变成不同程度的权重迁移。

在线 prior 的种子由 `(seed, optimizer_step, DDP rank)` 决定，每个 step 包括完整本 rank batch；DataLoader worker 数、预取和断点恢复不改变相同 world size 下的样本流。改变 GPU 数会改变批次分片，recipe 会拒绝该变更。硬路由及 GPU 归约存在正常浮点差异，不保证模型训练逐 bit 相同。

默认固定验证集每个 support=256/1024/4096/16384 面板 16 张独立 graph_scm 表，每张 128 个 query，features 循环 5/20/50，种子与训练独立。step 0、每 2000 step、最终 step 都验证。验证使用无梯度的预训练执行路径，直接计算全部 10 类 logits 的 CE/accuracy，或回归 pinball/median MSE；不使用推理集成、类别裁剪或校准。验证前后恢复各模块模式、RNG 和诊断状态；不会让评估改变训练随机流。

每组保存 `recipe.json`、源码 SHA256/命令/HEAD 的 `invocations.jsonl`、`logs/metrics.jsonl`、逐表 `logs/validation.jsonl` 和模型/优化器 checkpoint。共享验证集与初始化都记录文件 SHA256。根目录 `summary.json` 只比较恰好达到所选最终预算的 checkpoint，并输出逐 seed/逐长度的目标差值与配对表 bootstrap 95% CI；此 CI 只反映表间波动，不是跨训练种子不确定性。

重点检查 4K/16K 的目标差值、跨 seed 一致性和学习曲线，同时看 sampled retained mass、token-top32 recall、融合门、非有限值、梯度、OOM 和时间。`fail_fast=True`，不允许某个分支跳过 OOM microbatch 后继续参加比较。质量判定容忍度应在正式筛选前按可接受退化定义；不要拿路由 recall 或 200 step 的暂时 CE 顺序代替最终任务表现。

## 实现与性能复现

`content_tree` 在每个检索层/head 上根据当前 support K 重建平衡树，节点从 16 个按内容排序的代表点估计远点分割轴，CUDA 直接索引原 K 做打分，只排序标量和行索引。它没有行号窗口、训练码本、路由蒸馏和辅助 loss。候选仍是原始 K/V，候选内只做一次标准 softmax；global、融合门、预算与 projection 对照完全相同。

建树和搜索分开：正式五组均使用精确 centroid top-k，之后再比较 hierarchical beam。centroid 对所有叶打分仍有 O(N S/B) 项；content_tree 的多层分割也有排序成本，因此不宣称整个算法严格线性。分割采用精确的非稳定 radix argsort；投影碰撞、完全相同 K 的样本跨叶边界时仍可能依赖输入顺序，现有连续输入置换测试不能证明任意重复样本的严格集合不变性。缓存当前保存原始 K/V，cached query 会重新建派生索引。

```bash
# 12 层 ICL、完整模型路径、prefill/query、动态 support、路由与参考实现。
OMP_NUM_THREADS=4 PYTHONPATH=src conda run --no-capture-output -n stdplm \
  python scripts/benchmark_icl_suite.py --output /tmp/icl_perf \
  --memory-limit-gib 8 --repeat 7 --warmup 3 --compile

# 合成路由质量，不能当作下游准确率。
OMP_NUM_THREADS=4 PYTHONPATH=src conda run --no-capture-output -n stdplm \
  python scripts/benchmark_icl_router_quality.py --support 65537 \
  --queries 64 --seeds 3 --output /tmp/icl_route_quality.json
```

性能矩阵逐子进程运行并记录 GPU 占用、可用显存、每次耗时、median/min/max、峰值分配与 OOM。训练时间包括前反向和 Python 调度，不含 optimizer/prior；完整模型测试使用 8 features、column/row 各 1 层、ICL 12 层，以区别于正式训练的 3/3/12 配方。prefill/query 使用 12 层 ICL；query=256 且包含重建派生索引。动态 support 测试轮换五种长度，warmup 和正式样本分别保存。共享 GPU 的 min/max 可能很宽，数字只能用于当前条件下比较，部署决策需在独占设备复测。
