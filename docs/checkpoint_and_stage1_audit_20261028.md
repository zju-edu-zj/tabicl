# Checkpoint 数值兼容性、运行环境与全模型训练审计

> 后续已升级 Torch 2.8 / Triton 3.4 并实施全模型优化，最终测试与数值诊断见
> [全模型训练优化与验证](torch28_whole_training_optimization_20261028.md)。下文保留当时的审计结论与环境记录。

日期：2026-10-28。此次是诊断与审计：新增了诊断脚本与测试产物，没有修改生产模型、Trainer、训练配方或已有 checkpoint。以下将“已测量”“代码事实”“待实施方案”分开。

## 1. 结论

1. **不是只有 BF16 会改变旧 checkpoint 的预测，也不能认定复用就一定降质。** 同一份权重在新旧数值实现下，FP32/BF16 都有预测翻转；这组数据上 FP32 平均准确率略升，BF16 平均准确率不变，但长表和个别表有回退。参数加载兼容，不代表数值函数兼容。
2. **主要放大器是离散路由。** 固定旧实现的路由、只换算术内核后，FP32 翻转率从 1.9368% 降到 0.0326%，BF16 从 2.5391% 降到 0.0814%。这不是实际部署方案，而是定位因果的 oracle 对照。
3. **Stage 1 的下一优化对象应是前端和训练调度，而不是继续只做长表 ICL kernel。** 1024 行、100 列、microbatch=4 时，列+行模块占 FP32 约 92%、BF16 约 88%；已有 ICL 加速在这个整模型运行点明显受 Amdahl 上限约束。
4. 当前可直接运行的验证组合为 **Python 3.12.16 / PyTorch 2.5.1+cu121 / Triton 3.1.0**。驱动修复后不再需要旧用户态库包装器。Trainer 默认配置还存在 `DataLoader(in_order=False)` 与 Torch 2.5 不兼容的问题。

## 2. Checkpoint：验证对象和归因实验

### 2.1 实验口径

对照源码：`74d0285c32c2dbad61fc9f43c6c589dde272913a`，只读 worktree `/home/wbq/sparse/tabicl-baseline`。新旧两套包同时加载，各自 strict 加载相同权重。

检查的 checkpoint：

```text
checkpoints/icl_screen_round1_staged/seed_42/global_sparse_elu/step-50000.ckpt
checkpoints/icl_screen_round1_staged/validation.pt
```

3 个 column block + 3 个 row block + 12 个 ICL block；每三层一个 global+sparse，其余为 ELU global。48 张保存的案例，每张 128 个 query，即每种精度 6144 个预测；support=1024/4096/16384，各 16 张，特征数为 5/20/50。不是 100 列质量评测，也不是所有分支/seed 的泛化结论。

使用 `.train()` + `no_grad()`，dropout=0，以复现训练数值路径；不是 sklearn estimator 全流程评测。前端输出在两种精度的首个案例验证逐位一致，随后共享旧前端表示以隔离 ICL。每个案例的 `all_legacy` 对照均断言 logits **逐位一致**，96 次全部通过。

### 2.2 TF32 开启：同 checkpoint，新旧实现

| 精度/路径 | 平均 CE，越低越好 | 准确率 | 相对旧实现预测翻转率 |
|---|---:|---:|---:|
| FP32 旧实现 | 0.504754 | 79.2318% | — |
| FP32 新实现 | 0.503449 | 79.4596% | 1.9368% |
| FP32 新算术 + 固定旧路由 | 0.504751 | 79.2480% | 0.0326% |
| BF16 旧实现 | 0.504938 | 79.4108% | — |
| BF16 新实现 | 0.505440 | 79.4108% | 2.5391% |
| BF16 新算术 + 固定旧路由 | 0.504956 | 79.4596% | 0.0814% |

16K support 子集：FP32 准确率 74.0234%→74.6582%，BF16 为 74.7559%→74.4141%（下降 0.3418 个百分点）。BF16 该子集 CE 却从此前记录的 0.605869 降为 0.604916，说明 CE 和 argmax 准确率不能混作一个结论。

最坏单表默认翻转率：FP32 25.7813%、BF16 27.3438%。BF16 最坏表在固定路由后翻转降至 0.7813%。因此“平均基本不变”不能代替逐表迁移检查；另一方面，翻转不等于错误增加，原来的错误预测也可能被修正。

以表为重采样单位的配对bootstrap（NumPy `default_rng(123)`，按下表顺序连续采样，每项20,000次，95% percentile区间；每张表内128个query不拆开）：

| 精度 / 子集 | 新减旧准确率，百分点 | 95%区间，百分点 | 新减旧CE | CE的95%区间 |
|---|---:|---:|---:|---:|
| FP32 / 全48表 | +0.2279 | [-0.0163, +0.5208] | -0.001305 | [-0.003209, +0.000304] |
| FP32 / 16K的16表 | +0.6348 | [-0.0488, +1.4160] | -0.003495 | [-0.008265, +0.000129] |
| BF16 / 全48表 | 0 | [-0.3255, +0.3092] | +0.000502 | [-0.001158, +0.002112] |
| BF16 / 16K的16表 | -0.3418 | [-0.9766, +0.2930] | -0.000953 | [-0.005029, +0.002989] |

区间均跨0，因此这批案例**不能证明总体质量显著提高或下降**，更不能证明BF16必然退化；但逐表输出变化是真实且可重复的，兼容性风险也是真实的。bootstrap仍假定表能代表任务分布，不能代替多seed、外部真实任务验证。

9 种算子消融的平均翻转率：

| 新实现中替换/控制的部分 | FP32 | BF16 |
|---|---:|---:|
| 无，默认新实现 | 1.9368% | 2.5391% |
| 仅旧 RMSNorm | 1.7741% | 2.5065% |
| 仅旧 ELU global | 1.8717% | 2.2949% |
| 仅旧 sparse kernel | 1.9857% | 2.3438% |
| 旧 RMSNorm + 旧 global | 1.4811% | 1.7741% |
| 旧 RMSNorm + 旧 sparse | 1.8392% | 2.3275% |
| 三者全旧 | 0 | 0 |
| 固定旧路由 | 0.0326% | 0.0814% |
| 固定旧路由 + 旧 RMSNorm | 0.0326% | 0.0488% |

不能将某一消融“没有单调改善”解释成该算子无影响：路由是非连续函数，多处误差不是简单可加关系。

### 2.3 数值差异具体来自哪里

**第一层：RMSNorm 数值与输出 dtype。** 旧 `F.rms_norm` 在 BF16 autocast 下将 Q/K 提升为 FP32；新融合 RMSNorm 保持 FP32 归约，但输出 BF16。此行为已实测，不是根据 AMP 的名称推测。FP32 也会因归约、FMA 顺序改变而出现微小差异。

**第二层：ELU global 的数值路径改变。** 因为旧 Q/K norm 输出 FP32，旧 ELU 在 FP32 和 BF16 模式下通常都走 Torch FP32 fallback；Trainer 又设置 `allow_tf32=True`。新实现让 FP32 也走 Triton，采用分段 state 累加与 `tf32x3`。这会改变数值，不是修改 ELU 注意力定义。

固定完全相同的 FP32 Q/K、projection dtype 的 V，使用 FP32 ELU 特征、FP64 矩阵乘和归约作为参考，输出 relative L2：

| V dtype / support | 旧，TF32 开 | 新，TF32 开/关相同 | 旧，TF32 关 |
|---|---:|---:|---:|
| FP32 / 1024 | 2.0803e-4 | 2.7635e-6 | 1.624e-7 |
| FP32 / 16384 | 2.2458e-4 | 2.7192e-6 | 4.219e-7 |
| BF16 / 1024 | 6.4341e-5 | 1.7261e-6 | 1.652e-7 |
| BF16 / 16384 | 8.2394e-5 | 1.7724e-6 | 4.313e-7 |

这是随机、归一化的同输入局部测试，不包含新 norm 的 BF16 量化，也不证明所有训练输入都更准确。它证明：新 kernel 相对启用单项 TF32 的旧 fallback 可以**更接近数学参考，同时不复现旧模型**；不能把 checkpoint 漂移笼统归咎于“新 kernel 计算更差”。关闭 Torch TF32 也不会把 Triton 代码里的显式 `tf32x3` 改成 IEEE。

**第三层：稀疏 softmax 的计算顺序。** 原路径逐 query 标量归约；新路径按 leaf 组成矩阵 tile，分别算局部 softmax/LSE 后合并。BF16 路径在 AV 矩阵乘前会把概率转成低精度。固定候选集时是同一个数学算子，但不是逐位相同的浮点计算。

**第四层：projection 排序、leaf 分组和 centroid top-k 将差异离散放大。** score 的小变化可使 token 穿过 leaf 边界，进而改变候选集合、后续表示和下一层路由。仅加稳定排序不能解决不同 score 的相对顺序改变。

逐层平均 hidden relative L2：

| 层号（0-based） | FP32 | BF16 | 说明 |
|---:|---:|---:|---|
| 0 | 0.000139 | 0.001239 | global |
| 1 | 0.000224 | 0.002060 | global |
| 2 | 0.015044 | 0.031207 | 首个 sparse，误差跃升 |
| 5 | 0.063460 | 0.087597 | 第二个 sparse |
| 8 | 0.108926 | 0.132020 | 第三个 sparse |
| 11 | 0.119504 | 0.138176 | 第四个 sparse |

最后一层候选 support-token 集合的平均保留比例为 FP32 62.54%、BF16 60.44%（每表采样 32 个均匀位置的 query，跨 head 汇总）。不能把排序位置变化直接当候选变化；诊断同时比较了 token 所属 leaf 和实际候选 token 集合。

### 2.4 关闭 Torch TF32 的交叉验证

重跑相同 48 张表和全部消融：FP32 默认翻转 1.1882%，固定旧路由后 **0**；平均概率 RMSE 约 1.49e-7。BF16 默认翻转 2.0020%，固定路由后 0.0651%，概率 RMSE 约 6.32e-4。

这进一步支持“硬路由是主要放大器”。但不是关闭 TF32 即可无损复用：FP32 默认仍有超过 1% 翻转，BF16 仍有 norm 量化、AV 舍入等差异；关闭 TF32还会改变旧模型和普通 GEMM 的速度/数值。

### 2.5 复用建议

- 如果要求旧 checkpoint 严格复现，应保留旧算子、dtype、TF32/SDPA 设置的完整数值策略，而不是只退回旧 norm。当前诊断脚本能组合回旧实现，但生产接口**尚无**完整 legacy policy 开关。
- 如果允许质量在容差内变化，可复用权重，但必须按 dtype、support 长度、任务、seed 分层评测，观察 CE、准确率、最坏表和置信区间。6144 个 query 来自 48 张相关表，不等于 6144 个独立任务，应按表 bootstrap。
- 从新 Stage 1 开始训练可避免“已训练模型切换算术”的兼容问题，但仍需验证训练质量；不能由局部误差测试承诺最终精度。
- 已有 Stage 1/2 checkpoint 续训，应在同一 prior 序列上对照短程 loss、梯度、路由覆盖、验证集趋势，再决定完整 Stage 2/3；权重可加载不代表优化器状态、学习率阶段也应无条件沿用。此次没有运行长程适配训练，不承诺少量微调就能恢复。
- 固定旧路由是用旧网络提供 oracle 的诊断，不是“免费修复”：真正复现这些路由可能需要旧上游表示，成本不可忽略。只把 router 改为 FP32 也不足以严格复现旧上游数值。

这里讨论的是 **checkpoint 权重复用**。KV/support 派生缓存是另一件事；缓存命中/失效不解释上述 fresh-forward 的预测变化。

## 3. 已有优化，以及版本要求

| 优化 | 实质 | 依赖/注意事项 |
|---|---|---|
| leaf-grouped sparse | query→leaf 反向索引，矩阵 tile 复用 K/V，LSE 合并；leaf 独占 dK/dV | 自定义 CUDA Triton JIT；Tensor Core 路径，不要求开启 `torch.compile` |
| FP32 sparse/ELU | `tf32x3`、FP32 累积 | 不等于 IEEE 逐位 FP32；该实现的加速目标为支持 TF32 的 CUDA GPU，不能外推到旧架构 |
| 融合 Q/K RMSNorm | FP32 reduction、保留 BF16、融合 head-major 布局 | Triton；明确改变旧 BF16 norm 的输出 dtype |
| ELU 分段 state / 私有梯度 | support 并行归约，减少共享原子竞争，限制工作区 | Triton；D=128 backward 已有 IEEE 特殊路径处理共享内存限制 |
| support 派生缓存 | 缓存 ELU state / norm、排序、centroid 等 | 仅 eval+no_grad 的可验证失效场景；不加速普通训练 |
| 动态长度编译边界 | 路由/动态 kernel 区域 `torch.compiler.disable`，长度 `do_not_specialize` | 减少重编译；是允许 graph break 的混合执行，不是 fullgraph |

### 3.1 本机实际可用环境

| 项目 | 已验证值 | 含义 |
|---|---|---|
| Python | 3.12.16 | 当前 conda 环境 |
| PyTorch | 2.5.1+cu121 | wheel CUDA runtime 12.1 |
| Triton | 3.1.0 | 随该 Torch 配套安装；此次 kernel 基线 |
| Triton 自带 ptxas | 12.4.99 | Triton 使用的汇编工具 |
| PATH 中 nvcc | 11.5.119 | 不是本轮运行时版本；这些 Triton 测试不依赖它编译 CUDA 扩展 |
| GCC | 11.2.0 | 本轮 Inductor 主机代码编译环境 |
| 驱动/KMD | 615.71.09 | 驱动已修复，直接 CUDA 可用 |
| nvidia-smi 的 CUDA UMD | 13.4 | 不是 Torch wheel 的 runtime 版本 |

`pip check` 通过。GPU 型号等完整信息保存在原始 JSON，不做硬件代际优劣外推。

**不需要为了这些优化把系统 CUDA Toolkit 升到与 nvidia-smi 同一版本。** 需要的是驱动支持所用 runtime 和目标架构、Torch/Triton 配套、Triton 自带或指定的 ptxas 能生成对应代码。自编译 FlashAttention 等扩展则另有 nvcc、ABI、架构要求，不能与当前 Triton 路径混为一谈。

代码的 `torch>=2.2` 是项目通用下限，**不是新加速路径所有特性的已验证下限**。建议首先锁定上述已测试组合；不能凭一次成功测试保证任意 Torch>=2.2/Triton>=某版本都兼容，也不要只独立升级 Triton 而忽略 Torch 的依赖锁。

### 3.2 两类“动态编译”要区分

1. **Triton JIT**：eager 也会首次编译 kernel；按 dtype/head_dim/block/tile 等产生变体。`do_not_specialize` 只是减少长度变体，不意味着所有 shape 只编译一次。需要可写编译缓存和临时目录、主机编译工具链/驱动开发接口；不要求系统 nvcc 版本等于 wheel CUDA。
2. **Torch Dynamo/Inductor**：用户显式 `torch.compile(..., dynamic=True)` 后才开启。本项目 Trainer 当前没有调用 `torch.compile` 或提供训练编译开关；已有 ICL 动态长度通过不代表 Stage 1 整网已经编译通过。需要兼容的 Torch/Triton、C++ 编译器、足够 CPU 内存/磁盘和可写缓存；首次编译成本必须单列，不计入稳定态速度。

当前 ICL 有主动禁编译边界，因此不能承诺 `fullgraph=True`。前端还有 `.item()`、GPU tensor bool、布尔动态索引和 view 原地更新；`dynamic=True` 不能自动消除这些问题。F、support 和 dtype 变化也可能触发 guard/recompile。简单开启 `capture_scalar_outputs` 并不能处理全部数据依赖分支。

### 3.3 两个当前运行陷阱

**旧驱动包装器。** `scripts/run_tabicl312.sh` 默认注入 `/home/wbq/opt/nvidia-545-user` 旧用户态库。驱动修复后不应再按历史报告默认运行它；本轮全部直接调用：

```bash
PYTHONPATH=src /home/wbq/anaconda3/envs/tabicl312/bin/python ...
```

本轮未改包装器。历史报告中的旧修复命令仅用于当时环境。

**DataLoader 的版本门槛。** Torch 2.5.1 的 DataLoader 没有 `in_order` 参数，Trainer 能检测但默认 Stage 1 脚本设 `PRIOR_IN_ORDER=False`，因而当前默认命令会直接报错。先用 `PRIOR_IN_ORDER=True`；如必须使用 out-of-order 供数，应使用提供该接口的 Torch（2.6 起），并重新验证配套 Triton/训练。`benchmark_prior_pipeline.py` 也直接传此参数，不能原样在 2.5 跑。不能为速度把 `prior_seed_by_step=True` 和 out-of-order 混用，代码已有互斥检查。

## 4. 全模型 Stage 1 的真实耗时

### 4.1 测量方法

新增 `scripts/profile_training_stages.py` 使用真实 3+3+12、embed=128、8 heads、128 inducing points、4 CLS、ICL dim=512、FFN factor=2、target-aware、same grouping、QASSMax、QK norm。随机输入已在 GPU，support=60%，训练 CE，完整连通前向+反向；没有 detach 阶段来制造不同的计算图。

3 次预热，7 次测量，中位数；阶段 CUDA events 与总 wall 均记录。反向用 tensor hook 标边界。RowInteraction 会原地写 CLS，因此 hook 必须在该写操作后注册；编译可能消掉边界 view hook，届时只报合并的前端反向。每次计时结束后检查 loss/全部梯度有限，不把检查开销计入 kernel 计时。峰值为 allocated 显存，不含 optimizer state 和 prior。

这些是**全模型算子**计时，不含 H2D/prior/DDP、梯度裁剪、指标日志或真实 optimizer step，不能冒充最终训练 step。额外测了零 LR 的 Muon+AdamW step，单列而不混入模型计时。参数没有被训练更新。

### 4.2 100 列：列、行已占主要部分

以下每段均为 forward+backward；总 wall 中位数与各段中位数之和允许有微小区别。

| 精度 | 行数 | microbatch | 列 ms | 行 ms | ICL ms | 总 wall ms | 峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 128 | 1 | 20.44 | 9.70 | 8.35 | 38.81 | 1.22 |
| FP32 | 400 | 1 | 42.42 | 27.12 | 8.48 | 78.15 | 2.23 |
| FP32 | 1024 | 1 | 95.68 | 67.80 | 17.66 | 181.13 | 4.63 |
| FP32 | 1024 | 4 | 371.03 | 262.81 | 54.16 | 688.09 | 18.10 |
| BF16 | 128 | 1 | 11.39 | 5.04 | 12.96 | 29.44 | 0.83 |
| BF16 | 400 | 1 | 19.82 | 13.33 | 10.48 | 43.71 | 1.50 |
| BF16 | 1024 | 1 | 36.91 | 31.65 | 14.51 | 83.33 | 3.05 |
| BF16 | 1024 | 4 | 139.95 | 124.11 | 34.88 | 299.18 | 11.61 |

Stage 1 配方总行数固定 1024；128/400 行是额外短表覆盖。这里用 100 列是关注的最宽运行点，不等于 prior 所有表都有 100 列。

1024×100、B=4：即使把 ICL 成本完全降到零，FP32 也最多再快约 1.085×，BF16 约 1.132×。相反，列+行若能减半，理论总加速约 1.85×/1.79×。这些是上限估算，不是已实现的收益承诺。

10 列时，模型更偏启动/调度瓶颈：

| 精度 | 128 行 | 400 行 | 1024 行 |
|---|---:|---:|---:|
| FP32，B=1 | 19.16 ms | 24.29 ms | 45.71 ms |
| BF16，B=1 | 22.86 ms | 23.90 ms | 31.46 ms |

128×10 的 BF16 在本次测试还慢于 FP32。不能承诺 BF16 对所有短表都更快；这个量级需要更长重复测试、启动开销分析和 batching。

### 4.3 已有 ICL 优化带来多少整网收益

1024×100、B=1，同分支 global_sparse 前后对照：

| 精度 | 旧 ICL ms | 新 ICL ms | 旧整网 ms | 新整网 ms | 整网加速 |
|---|---:|---:|---:|---:|---:|
| FP32 | 49.23 | 17.66 | 215.44 | 181.13 | 1.19× |
| BF16 | 41.81 | 14.51 | 111.25 | 83.33 | 1.34× |

此前 16K ICL 的多倍加速是真的，但不能代入 Stage 1 整网。这也解释“ICL 很快但训练还是耗时”。

同一尺寸 dense 对照：FP32 整网 187.16 ms，BF16 79.92 ms；BF16 dense ICL 10.49 ms，比 global_sparse 的 14.51 ms 快。两者架构不同，不能静默在短表把 ELU/global_sparse 改成 dense 并声称 checkpoint 等价。已有逻辑会在 support<=512 时让 sparse 子路走全 support SDPA，但仍保留 global 和 gate，这个数学等价边界是合理的；不是整个 hybrid 都可删掉。

### 4.4 重计算、optimizer 和两个否定性对照

| 精度，1024×100，B=4 | 不重计算 ms / GiB | 全量重计算 ms / GiB |
|---|---:|---:|
| FP32 | 688.09 / 18.10 | 881.73 / 4.73 |
| BF16 | 299.18 / 11.61 | 383.18 / 3.48 |

全量重计算约慢 28%，因此 Stage 1 显存够时不应开启；Stage 2/3 或更大 batch 再考虑按模块选择性重计算，而不是全局布尔一刀切。

Muon+AdamW 零 LR step 稳定约 24 ms。生产全局 batch=64；单卡 microbatch=4 时每步应累计 16 次，因此不能把“每 microbatch +24 ms”当每步开销。16 rank、每 rank=4 时则每 rank 一次 microbatch，optimizer 占比会变大。DDP 通信未测。

只去除 ISAB skip 检查的诊断：FP32 184.70 ms、BF16 83.97 ms，没有可见加速。去掉 column SSMax 的架构消融：181.85/82.27 ms，也没有显著整网收益。后者不是等价优化，本轮随机初始化的 query MLP 末层为零，不能代表已训练 checkpoint 的实际 SSMax 功能。**不推荐据此删 SSMax，也不能将消融后的速度写成已落地收益。**

## 5. 前端代码审计：应该具体优化什么

### 5.1 FP32 优先：column/row attention 训练 kernel

Profiler 在 1024×100、B=1 的 FP32 前端观察到 9 次 memory-efficient SDPA（column 6 次、row 3 次）：forward 约 22.17 ms，backward 约 57.51 ms。它们已是融合 attention，不是 naive N² 概率矩阵，但占全模型约 44%。BF16 相同调用走 FlashAttention，约 2.28+10.05 ms。

不能靠把 FP32 输入强转 BF16 来宣称优化了 FP32。若必须保留 FP32，应单独评估 D=16 的 TF32x3/FP32 累积 Flash-style 前后向，针对 column 的 128 inducing queries / support 和 row 的约104 tokens 特化，避免中间注意力矩阵，明确数值策略。需测 IEEE 参考误差、完整参数梯度、target support 截断、mask、RoPE、SSMax、短表、tail、非连续布局，并再跑 checkpoint 漂移评测。

风险和优先级都高；建议先做可独立开关的 kernel A/B，不能直接替换通用 SDPA。启用 cuDNN/FA3 或升级 Torch 也需要重新测试后端选择与精度，不能假设必然更快。

### 5.2 BF16 优先：布局、mixed precision 残差、LayerNorm

相同 BF16 profiler：`aten::copy_` 共 1073 次、self CUDA 约 14.59 ms；54 次 LayerNorm forward 约 4.83 ms，backward self CUDA 约 10.91 ms；这些与 GEMM/FlashAttention 同级重要。profiler 的 operator 和 kernel 行有嵌套包含关系，不能把各行百分比相加；用未插 profiler 的 events 做总耗时比较。

前端并非所有激活都是 BF16：inducing/CLS 参数保持 FP32，残差 `q + attn` 可将后续残差流提升到 FP32。应先统计每个边界 dtype/stride，区分：必要的 FP32 数值稳定性、重复 autocast、transpose后 contiguous、KV 输入重复 normalize、列→行布局转换。优先融合 residual+LayerNorm（FP32 归约及合适的 weight-grad 归约）、bias/GELU、布局写回，避免先全局改变残差 dtype。

Row RoPE 的 [`apply_rotary_emb`](../src/tabicl/_model/rope.py:57) 关闭 autocast，FP32 频率使 BF16 Q/K 参与 FP32 中间计算，再转回；多次 cos/sin、cat/rotate/mul/add 带来临时张量。可缓存频率/cos/sin、融合旋转与最终 cast；必须保持 non-interleaved/offset/CLS query 与全 K 的对应关系。不能为了省 copy 把不同 RoPE 模式混用。

### 5.3 无效 column 工作与标签编码：相对低风险的第一批改动

- `feature_group="same", size=3` **仍有 F 个组**，不是把100列缩到34列。当前 column 处理104组；改 `valid` 会改变模型与 checkpoint 语义。
- `_train_forward_with_feature_group` 先插4个 `-100` CLS reserved slots，再做 column transformer；target-aware 标签加法可能让 reserved 列不再被整列 skip。RowInteraction 随后用 learned CLS 原地覆盖这些结果。column 是按列独立处理的，故可设计不计算这4列、直接在行模块插入 CLS 的等价路径。最多省去 column 的4/104（100列）或4/14（10列）左右工作，不是整个模型同比例；需要检验梯度、padding、affine/inference/cache 分支和无标签时的 skip 行为。
- 标签当前先扩展为 `B×(F+4)×support` 再 one-hot+linear，重复同一编码。可先做 `B×support×E` 标签编码，再 broadcast 加到各列。进一步以 embedding gather 代替 one-hot GEMM时，要保留 bias 和 AMP 舍入；前者数学等价，后者不一定与旧浮点路径逐位一致。权重梯度的广播归约也必须对照。
- `num_classes=int(y_train.max().item())+1` 每 microbatch 在 GPU 同步，回归分支也先执行。Stage 1 class upper bound已知，可给经过验证的训练专用 metadata/fast path，保留通用 many-class 推理检测，而不是把检测从所有路径删除。
- `SkippableLinear` 先算全量 linear 再 mask；ISAB 每层对 `B×F×T×E` 扫 `-100`，`if skip_mask.any()` 触发同步。可由输入元数据维护列有效性并传递显式 mask，避免重复扫描/动态 ragged gather。但本次单独去检查没有加速，优先级应低于布局/attention；必须保留 sentinel 的公共语义。

### 5.4 部分图编译，而不是整网盲编译

本轮在当前 **Torch 2.5.1 / Triton 3.1.0** 实测到了明确的正确性门槛，不能直接推荐 `torch.compile(model, dynamic=True)`：

| BF16、1024×100、B=1，无计时钩子的独立对照 | loss | 全参数梯度有限性 |
|---|---:|---|
| eager | 2.342505 | 通过 |
| 仅 column，dynamic=True | NaN | 334 个参数梯度非有限 |
| 仅 row，dynamic=True | 2.342537 | 通过 |
| column+row，dynamic=True | NaN | 334 个参数梯度非有限 |
| 仅 column，dynamic=False | 2.342442 | 通过 |

进一步定位：动态 column 删除 skip 检查、删除 SSMax、仅编译 `tf_col` 都仍产生 NaN；仅动态编译 `in_linear` 通过。失败时不只是 reserved CLS，实际100列的 13,107,200 个输出元素也全部非有限。因此可定位到**动态 column transformer 编译路径**，不是计时 hook、标签 `.item()` 或 SSMax 单独解释的问题；尚未定位到某条编译器 lowering/生成 kernel，不能断言升级某个版本就必然修复。

保留真实计算图，分别测量通过有限性检查的编译组合：

| 精度、1024×100 | B | eager ms | 仅 row 动态编译 ms | column 静态 + row 动态 ms |
|---|---:|---:|---:|---:|
| FP32 | 1 | 181.13 | 163.23 | 未测 |
| BF16 | 1 | 83.33 | 67.13 | 未测 |
| BF16 | 4 | 299.18 | 232.76 | 172.01 |

BF16 B4 最后一种组合相对 eager 约 **1.74×**，延迟减少42.5%，峰值 allocated 从11.61降到9.27 GiB；仅 row 编译为10.87 GiB。FP32 仅 row 编译约1.11×，显存4.68 GiB，并没有同样大的收益。编译可能消掉 column/row 边界 view hook，因此这些记录只报合并前端 backward，没有插入 clone 或断图来强行拆段。

这些是固定随机输入、固定 shape、排除 prior/H2D/optimizer 的**稳态前后向**，不是生产整step或已验证的checkpoint迁移配置。BF16 B4静态column组合首轮含编译17.51秒，后续约172ms；缓存状态会影响首轮，不能视作通用冷编译成本。Stage1真实F和support长度都变化，`dynamic=False` 可能产生大量变体，必须记录 guard、recompile/cache limit 和全程摊销成本；不能把单shape的1.74×直接乘500K步。短表128×10、B1、BF16仅row编译为21.05ms，相对eager22.86ms仅约1.09×，收益没有宽表大。

还进行了无计时钩子的输出/全部梯度对照。下面relative L2以eager为分母，梯度按所有有梯度参数拼接的全局L2计算；参数名去除compile的`_orig_mod`包装，梯度参数集合均一致。

| 1024×100、B1 | row输出 relative L2 | logits relative L2 | 全参数梯度 relative L2 |
|---|---:|---:|---:|
| FP32，仅row动态编译 | 1.052e-4 | 2.437e-4 | 2.426e-4 |
| BF16，仅row动态编译 | 0.005455 | 0.006001 | 0.005829 |
| BF16，仅column静态编译 | 0.006170 | 0.006252 | 0.006088 |
| BF16，column静态+row动态 | 0.006310 | 0.005975 | 0.006450 |

这张随机表各路径argmax翻转都为0，但不代表旧checkpoint质量等价。少数很小的路由/门控梯度相对误差可达数十个百分点，例如BF16组合的ICL末层QK norm权重梯度relative L2约0.39、max abs约2.7e-6。应结合绝对误差和按参数分组的指标，不用全局L2掩盖小梯度，也不能由相对误差大就推断必然降质。compile融合同样会改变舍入，仍需训练轨迹与已训练模型的分层质量评估；本轮没有把编译开关加进生产Trainer。

宜先将纯 tensor 的 FFN、norm/残差、RoPE、静态 attention block 分段编译，绕开数据依赖验证；再移出 `.item()`、GPU bool、动态布尔切片。Stage 1 T固定，按 F/support 的实际分布规划少量图和编译预算，观察 guard日志、编译次数和稳态性能。

**不能盲目把任意 F pad 到100以凑同一张图**：same grouping 的 circular index 使用真实 F，padding会改变首尾列邻居，且当前 ignore_d 会把 padded列视作普通列。分桶需要保持分组和mask的原始语义。

最后一个 row block 已经只计算4个 CLS queries over全部K/V，前两个处理全部feature tokens；不要把这个已存在的优化重新列为新增收益。全新的 sparse/DSA 主要针对行数轴，不应直接套到104个feature token或128个inducing point的小 attention上。

## 6. 模型之外：Trainer、optimizer、供数

### 6.1 Trainer 的明确同步点

[`run_micro_batch`](../src/tabicl/train/_run.py:909) 的 CPU metadata `.item()` 本身不是 GPU同步，不应一概删除。真正值得处理的是：

- pinned-memory DataLoader 后，`micro_X/micro_y/micro_d.to(device)` 没有 `non_blocking=True`；先验证 trim/nested padding 后张量是否仍 pinned，再加入异步拷贝。独立copy stream预取下一microbatch需维护事件、storage lifetime和stream ownership。
- 每 microbatch `if not isfinite(loss)`、CE `.item()`、accuracy `.item()`，以及每step将grad norm转float。这些读回会中断CPU/GPU重叠。可在GPU累计指标、每step一次搬运；有限性检查应设计异步标记并在安全边界检查，不能为提速默默去掉 fail-fast。
- 诊断只在rank0每200step，但该rank可能拖住collective，需分别记录诊断step与普通step的p50/p95；不把诊断关闭后的数字与开启的基准混比较。
- `run_batch` 先将整个nested batch pad，再切microbatch，Stage2/3会被最长表放大内存；可按microbatch pad。Stage1固定T时不是第一瓶颈。
- 已有 `require_backward_grad_sync` 只在最后一个microbatch同步，不应重复建议“增加梯度累积/no_sync”作为尚未实现的技巧。DDP需实测bucket、allreduce尾部和prior最慢rank。

### 6.2 Optimizer

Muon 已按等矩阵形状做batched Newton–Schulz，momentum也用foreach，并非每矩阵全部独立小GEMM。仍有两个方向：

1. [`zeropower_via_newtonschulz5`](../src/tabicl/train/_muon.py:18) 实际是 `X=G`，FP32 参数梯度在BF16 AMP下仍为FP32，因此NS主要是FP32矩阵乘。可先编译/融合这一小段；改成BF16 NS属于数值策略改变，要验证optimizer更新误差和长期质量。
2. 混合optimizer中的AdamW目前逐参数做多个elementwise操作。可增加foreach/fused路径，保持参数分组、per-parameter step、条件未使用参数、weight decay、cautious WD和state_dict迁移。把它简单换成默认AdamW会丢掉这些语义。

先按每optimizer-step的有效batch/累积数计算占比，再决定投入优先级；当前24ms不是每个microbatch的固定税。

### 6.3 graph_scm prior

Stage1有500K步，且graph/dataset过滤会产生CPU尾延迟。worker不足、按序head-of-line blocking、多rank最慢者都会让GPU idle。默认8 workers/rank×16rank=128 workers，prefetch8还会占CPU/RAM；workers越多不必然越快。

实际调用已有`benchmark_prior_pipeline.py`：graph_scm、1024行、1–100列、local batch4、graph/dataset过滤开启，CPU生成。Torch2.5用诊断期兼容类**仅去掉不支持的`in_order=True`实参**，底层仍是真正按序DataLoader，没有模拟out-of-order。OMP/MKL/OpenBLAS线程各限制为1，worker内Torch线程也为1。每次8批预热、24批记录、prefetch=8，consumer delay=300ms，对应未编译100列BF16 B4的消费量级：

| workers，单rank | fetch mean ms | p50 ms | p95 ms | max ms | 等待>100ms的批次 |
|---:|---:|---:|---:|---:|---:|
| 1 | 896.53 | 661.16 | 2695.56 | 3991.25 | 91.7% |
| 4 | 119.37 | 0.24 | 779.79 | 1233.53 | 12.5% |
| 8 | 7.58 | 0.30 | 42.36 | 75.33 | 0% |

又以8 workers、172ms consumer做16批预热、64批记录：平均等待5.99ms、p50=0.33ms、p95=34.62ms、max=135.88ms，有1/64批等待超过100ms。这里只改变消费延迟，不运行GPU编译模型。

这证实供数可能抵消kernel加速，尤其worker不足时；也说明仅看平均GPU用时遗漏了什么。但这些是短程单rank、固定消费延迟测试：实际prior特征宽度变化，FP32/BF16及compile会改变消费速度；不能把这张表当最优workers搜索，或外推16rank的最慢rank尾延迟。24个记录不足以可靠估计长期p99，故正文不据此推断p99。prefetch先填队列也可能美化短程结果，应扩大生产观测窗口。尤其该脚本local batch固定为4，不是单卡global batch64时Trainer每次获取64表再累计16个microbatch的供数口径；后者需使用对应batch重新测量。

应同时记录：fetch wait p50/p95/p99、max-rank wait、CPU占用、GPU idle、H2D和模型时间；用接近实际消费速度的consumer delay跑供数测试。预生成prior会改变随机性、磁盘开销和数据多样性，需要可复现设计；不是免费优化。

## 7. 推荐实施顺序与验收

1. **先保证启动和可观测性**：不注入旧驱动库；处理 `PRIOR_IN_ORDER`；建立生产3+3+12、1024×100、B4、FP32/BF16和10列/短表的整step基线，拆普通step/诊断step，计入真实prior/H2D/optimizer/DDP。
2. **等价前端改造**：标签只编码一次、去掉被覆盖的reserved列计算、显式有效性metadata；用相同输入验证输出/梯度及已训练checkpoint，而不是仅随机模型loss有限。
3. **BF16 主线**：布局/cast和fused residual-LN/FFN/RoPE；每项单独A/B，记录数值策略，不先全局把残差流降精度。
4. **FP32 主线**：优化前端D16 attention前后向，独立保留参考fallback；必要时引入显式IEEE与TF32x3配置，不把“FP32参数”误当“所有算术IEEE”。
5. **短表主线**：减少launch和主机同步、按语义正确的shape分组、合适的microbatch，再评估局部compile/CUDA graph；不能直接capture带 `.item()`、动态索引或改变分组语义的整训练循环。
6. **随GPU提速复测prior/DDP/optimizer**：先保证供数不饥饿，再看真实step的剩余占比优化；不要等到所有kernel做完才发现GPU在等数据，也不要只看孤立微基准。

每个改动的验收至少包括：FP32/BF16、回归/分类、same/valid/no-group、target-aware on/off、padding/mask、support30/60/90%、T=128/400/1024与Stage2/3长表、B=1/4、重计算、eager/部分compile、输出与全部相关参数梯度、optimizer state连续性、checkpoint分层质量、热/冷编译和峰值显存。短表不回退、宽表整step真正变快，才算完成；仅kernel更快不够。

## 8. 复现与产物

以下在 `/home/wbq/sparse/tabicl` 执行；所有 checkpoint 只读：

```bash
PYTHONPATH=src /home/wbq/anaconda3/envs/tabicl312/bin/python scripts/diagnose_checkpoint_drift.py \
  --baseline-root ../tabicl-baseline \
  --checkpoint checkpoints/icl_screen_round1_staged/seed_42/global_sparse_elu/step-50000.ckpt \
  --cases checkpoints/icl_screen_round1_staged/validation.pt \
  --output ../perf_results/drift_diagnosis_tf32_on --tf32 on

PYTHONPATH=src /home/wbq/anaconda3/envs/tabicl312/bin/python scripts/audit_elu_precision.py \
  --baseline-root ../tabicl-baseline --output ../perf_results/elu_precision_audit.json

PYTHONPATH=src /home/wbq/anaconda3/envs/tabicl312/bin/python scripts/profile_training_stages.py \
  --output ../perf_results/stage1_audit/replay_bf16_b4.json \
  --dtype bfloat16 --rows 1024 --features 100 --batch 4 --profile

# 固定shape局部compile试验；不是当前生产Trainer开关
PYTHONPATH=src /home/wbq/anaconda3/envs/tabicl312/bin/python scripts/profile_training_stages.py \
  --output ../perf_results/stage1_audit/replay_bf16_b4_compile.json \
  --dtype bfloat16 --rows 1024 --features 100 --batch 4 --compile-front --static-col

PYTHONPATH=src /home/wbq/anaconda3/envs/tabicl312/bin/python scripts/audit_frontend_compile.py \
  --output ../perf_results/replay_frontend_compile_equivalence.json \
  --dtype bfloat16 --modes eager row col_static both_static
```

当前Torch2.5上复现按序prior测试（只是诊断进程内兼容，不修改Trainer；参数可改workers/consumer/steps）：

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/wbq/anaconda3/envs/tabicl312/bin/python - <<'PY'
import runpy, sys, torch
from torch.utils.data import DataLoader as OriginalLoader
class OrderedCompatLoader(OriginalLoader):
    def __init__(self, *args, **kwargs):
        if not kwargs.pop('in_order', True):
            raise ValueError('This compatibility probe supports ordered loading only')
        super().__init__(*args, **kwargs)
torch.utils.data.DataLoader = OrderedCompatLoader
sys.argv = ['scripts/benchmark_prior_pipeline.py', '--workers', '8',
            '--prefetch', '8', '--in-order', 'True', '--warmup', '16',
            '--steps', '64', '--consumer-ms', '172',
            '--output', '../perf_results/replay_prior_ordered.json']
runpy.run_path('scripts/benchmark_prior_pipeline.py', run_name='__main__')
PY
```

原始产物在 `/home/wbq/sparse/perf_results/`：

- `drift_diagnosis_tf32_on/diagnosis.json`、`drift_diagnosis_tf32_off/diagnosis.json`：逐表9种消融、分层hidden误差、路由与候选集合变化。
- `elu_precision_audit.json`：同Q/K/V的FP64-reference误差和norm dtype。
- `stage1_audit/`：12种T/F/dtype矩阵、B4、重计算、新旧源码、dense和两个局部消融；`.trace.json`为Chrome/Perfetto trace，`.profile.txt`为聚合表。
- `frontend_compile_correctness_bf16.json`、`frontend_compile_localization_bf16.json`：独立无计时钩子的compile有限性检查与NaN定位；动态column失败、静态column和动态row通过。
- `stage1_audit/*compile_row.json`、`*compile_front_static_col.json`：通过有限性检查的局部编译稳态计时和首轮成本。
- `frontend_compile_equivalence_bf16.json`、`frontend_compile_equivalence_combined_bf16.json`、`frontend_compile_equivalence_fp32.json`：无计时钩子的eager/局部compile输出、逐参数梯度误差与预测翻转。
- `prior_audit_ordered_workers_{1,4,8}.json`：真实graph_scm、单rank、按序加载、300ms consumer的供数等待。
- `prior_audit_ordered_workers_8_fast_consumer.json`：8 workers、172ms consumer、16批预热和64批记录。

驱动修复后再次直接用tabicl312运行相关13个测试文件（与2026-10-27报告所列范围相同）：**562 passed、9 skipped、5 warnings，45.46秒**。跳过项为不可用的MPS/XPU；警告为一条Inductor TF32建议和4条现有multiprocessing fork弃用提示。测试通过不覆盖上文额外诊断主动发现的动态column编译NaN；该错误已单独记录，不能用总通过数掩盖。四个新增诊断脚本的`py_compile`、`git diff --check`也通过。没有运行需要外部下载权重的整个sklearn评测全集。

这轮没有开展500K步预训练或多seed外部真实任务质量评估。任何对最终质量、跨GPU/跨版本性能的外推都应保留这个限制。
