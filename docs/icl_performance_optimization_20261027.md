# ICL 性能优化与验证（2026-10-27）

> 2026-10-28更新：驱动已修复。本文末尾 `run_tabicl312.sh` 注入旧545用户态库的复现命令仅适用于当时环境，**现在请直接使用** `PYTHONPATH=src /home/wbq/anaconda3/envs/tabicl312/bin/python ...`。最新checkpoint漂移归因、版本要求、Stage1全模型性能与编译NaN诊断见[后续审计](checkpoint_and_stage1_audit_20261028.md)。本文的ICL加速倍数不能作为Stage1整训练加速倍数。

本轮已经实现按 leaf 聚合的稀疏计算、FP32 加速路径、保持 BF16 的 Q/K RMSNorm、ELU 分块归约和推理派生缓存。现有 checkpoint 的参数名、形状和分支配置不变，无需转换。没有实现新的 DSA 架构，也没有用长训练来验证它的质量。

## 测量口径

对照为 Git `74d0285c32c2dbad61fc9f43c6c589dde272913a`，只读 worktree 在 `/home/wbq/sparse/tabicl-baseline`。同一个 Python/CUDA 环境中串行启动优化前、优化后的独立进程，固定 seed=123，预热 3 次、测量 7 次，记录中位数、逐次样本和峰值 allocated/reserved 显存。计时包含 CPU 调度、前后向及同步；默认不含 optimizer step。首次 JIT 不计入稳定态中位数，原始 warmup 数据保留。

默认测试是整个 ICL encoder，包括投影、归一化、FFN、残差和路由，并非单独 attention kernel。hidden=512、8 heads、head_dim=64、batch=1、块大小 128、选 4 块（512 个候选），support 为总行数的 80%。`global_sparse` 每三层有一个 global+sparse 层，其余是 global；`sparse` 每层都是检索层。不同架构的质量不能由速度推出。

FP32 使用 FP32 参数、输入、输出和累积；自定义矩阵乘采用三项 TF32（`tf32x3`），不是逐位等价的 IEEE FP32，也不是单项 TF32。基准的普通矩阵乘启用 `allow_tf32=True`。BF16 是 FP32 参数加 CUDA autocast，归约、softmax、LSE 和 ELU state 仍为 FP32。BF16 Q/K 不再被 RMSNorm 隐式提升成 FP32；因此旧 checkpoint 的低精度舍入和部分 top-k 边界可能变化。

## 主要结果

3 层 ICL encoder、16,384 总行数，训练前向+反向：

| 精度 | 分支 | 优化前 ms | 优化后 ms | 同分支加速 | 峰值 allocated GiB 前→后 |
|---|---|---:|---:|---:|---:|
| FP32 | dense | 679.920 | 676.198 | 1.01× | 1.756→1.586 |
| FP32 | global | 51.836 | 37.970 | 1.37× | 1.663→1.567 |
| FP32 | sparse | 502.653 | 75.499 | 6.66× | 1.932→1.768 |
| FP32 | global_sparse | 206.528 | 52.493 | 3.93× | 1.840→1.691 |
| FP32 | global_sparse_alternating | 201.192 | 49.953 | 4.03× | 1.751→1.634 |
| BF16 | dense | 100.686 | 92.975 | 1.08× | 1.422→1.082 |
| BF16 | global | 42.282 | 25.775 | 1.64× | 1.510→1.025 |
| BF16 | sparse | 352.768 | 37.803 | 9.33× | 1.645→1.132 |
| BF16 | global_sparse | 149.541 | 33.045 | 4.53× | 1.580→1.089 |
| BF16 | global_sparse_alternating | 144.016 | 31.276 | 4.60× | 1.490→1.060 |

在这个运行点，优化后 global_sparse 相对优化后 dense，FP32 快约 12.88×，BF16 快约 2.81×。两者的注意力机制不同，此处只比较执行成本。

短表不能照搬长表的倍数：

| 总行数 | 精度 | 优化后 dense ms | 优化后 sparse ms | 优化后 global_sparse ms |
|---:|---|---:|---:|---:|
| 1,024 | FP32 | 7.440 | 6.990 | 5.259 |
| 1,024 | BF16 | 5.081 | 4.839 | 4.244 |
| 4,096 | FP32 | 53.341 | 20.532 | 15.017 |
| 4,096 | BF16 | 12.611 | 10.416 | 9.538 |

1K BF16 dense 本次中位数从 4.238 ms 变为 5.081 ms，存在短表回退；不能声称每一种形状都加速。该区间有明显的启动与 CPU 调度占比。12 层 1K BF16 下 sparse 17.501 ms，仍慢于 dense 13.199 ms。

完整 12 层、16,384 总行数、无重计算：

| 精度 | 分支 | 优化前 ms | 优化后 ms | 加速 | 峰值 GiB 前→后 |
|---|---|---:|---:|---:|---:|
| FP32 | dense | 2698.893 | 2699.066 | 1.00× | 6.413→5.734 |
| FP32 | sparse | 1967.135 | 309.014 | 6.37× | 7.121→6.465 |
| FP32 | global_sparse | 812.434 | 213.566 | 3.80× | 6.750→6.157 |
| BF16 | dense | 407.394 | 376.594 | 1.08× | 5.127→3.767 |
| BF16 | sparse | 1427.203 | 152.772 | 9.34× | 6.023→3.968 |
| BF16 | global_sparse | 606.544 | 128.084 | 4.74× | 5.755→3.796 |

这里 BF16 global_sparse 比优化后 dense 快约 2.94×，显存接近 dense。稀疏路由元数据和 leaf 局部输出需要工作区，所以稀疏不保证比高效 dense 更省显存。

60,000 总行数、3 层 encoder 的训练前向+反向也已验证：

| 精度 | 分支 | 优化前 ms | 优化后 ms | 加速 |
|---|---|---:|---:|---:|
| FP32 | dense | 9149.955 | 9208.368 | 0.99× |
| FP32 | sparse | 2189.295 | 328.878 | 6.66× |
| FP32 | global_sparse | 869.659 | 203.584 | 4.27× |
| BF16 | dense | 1070.956 | 1046.334 | 1.02× |
| BF16 | sparse | 1548.976 | 184.953 | 8.37× |
| BF16 | global_sparse | 628.798 | 126.825 | 4.96× |

其他集成测试均检查了数值有限性，扩展矩阵共 38 组前后对照全部成功：

| 测试场景 | FP32 global_sparse 前→后 ms | BF16 global_sparse 前→后 ms |
|---|---:|---:|
| 12 层、16K、开启重计算 | 940.078→264.251 | 698.302→158.883 |
| 12 层、4K、包含 Muon+AdamW 更新 | 209.619→79.458 | 177.371→58.347 |
| 整个 TabICL、4K、3 个 ICL 层 | 81.522→50.006 | 57.222→27.752 |
| 3 层、batch=4、4K | 180.869→51.678 | 148.603→31.933 |

整模型测试另含 1 个 col block、1 个 row block、8 个特征，包含前面的特征提取，因此收益小于只测 ICL encoder。它不是生产 3+3+12 配置的整模型耗时。

## 推理与缓存

3 层、16K 总行数对应的 support、每批 256 query，固定 support KV cache，热缓存 query 延迟：

| 精度 | 分支 | 优化前 ms | 优化后 ms | 加速 |
|---|---|---:|---:|---:|
| FP32 | global | 2.217 | 0.630 | 3.52× |
| FP32 | sparse | 2.728 | 1.672 | 1.63× |
| FP32 | global_sparse | 2.993 | 1.091 | 2.74× |
| BF16 | global | 2.489 | 0.753 | 3.31× |
| BF16 | sparse | 2.346 | 1.621 | 1.45× |
| BF16 | global_sparse | 3.105 | 1.081 | 2.87× |

BF16 dense 同形状约 1.033 ms，比 sparse 的 1.621 ms 更快；固定短 query 时稀疏元数据开销仍然重要。不要把大批 query 的训练加速当成每批预测都能获得的倍数。

缓存新增内容为 ELU 的 `phi(K).T @ V`、`sum(phi(K))`，以及 sparse 的 support order、centroids 和块长度。只在 eval+no_grad、关闭诊断时复用。K/V 的 tensor identity/version、模块参数/buffer version、分块与算子配置改变会重建派生数据。slice、to、concat 不继承派生数据，赋值会清空；显存统计包含派生数据。`inference_mode` 产生的 K/V 没有 version counter，使用安全的重算路径。尚未缓存层次 router 的完整 tree，也未缓存 induced slot values。

## 实现细节与边界

1. `_sparse_grouped.py` 将 query→leaf 索引反转，按 32-query tile 复用 leaf K/V，使用矩阵乘计算每个 leaf 的局部输出与 LSE，再按 logsumexp 合并为原来的稀疏 softmax。反向复用同一份索引，dK/dV 由 leaf 独占归约，避免 token 级 FP32 原子散射。短 query、小块等情况保留直接索引内核。非标准块大小需要让 support tile 整除 leaf；已覆盖 16/32/64/96/128/256 块的输出和梯度。
2. `_norm_triton.py` 将 RMSNorm、dtype 保持和 head-major 布局转换融合。FP32 归约后直接输出 projection dtype，不再把 BF16 Q/K 扩成 FP32；权重名与梯度返回 dtype 不变。非 CUDA 和不支持维度走显式 fallback。
3. `_elu_kernel.py` 并行计算 support 各段的局部 state，再归约；query backward 写私有 state 梯度后归约，替换共享 state 原子竞争。前后向私有工作区各按约 32 MiB 上限规划（至少一个 state）；长表不形成 N×S 概率矩阵。不等 Q/V 维度正确走 fallback。D=128 backward 用 IEEE dot，避免本机共享内存溢出。
4. `_routing_triton.py` 将 Triton 3.1 不支持的 `tl.gather` 改为小型寄存器查找；硬 top-k 的 centroid 打分仍使用 IEEE，避免在本轮引入近似检索策略。
5. 移除了 screening 脚本硬编码的他人 Python 路径检查，使用用户选择的环境即可运行。
6. 修复 Torch 2.5 动态编译中 RMSNorm symbolic stride 追踪及 Triton symbolic constexpr 代码生成失败。归一化和自定义 attention 内核作为显式编译边界，外围投影/FFN 仍交给 Inductor；不支持 `fullgraph=True`。FP32/BF16 的动态 support 训练均已跑通，2K 中位数分别为 8.647/7.339 ms，同条件 eager 为 8.408/5.705 ms，当前不建议为了速度默认启用编译。编译预热首轮约 9 秒，后续新长度仍有约 0.7–1.2 秒开销，不能算作稳定态毫秒延迟。这组编译计时早于下面的随机长度特化修复；最终源码的动态编译输出及全部梯度由回归测试验证，未重新测量这组编译耗时。
7. 将 grouped route 的 leaf 数、ELU split/chunk 长度和 RMSNorm 长度相关 stride 改为运行时参数，避免 Stage 2/3 随机长度无必要地生成新 Triton kernel。新增测试确认三种不同 query/support 长度在预热后复用这些 kernel。基准工具也修复了 Triton 3.1 cache 计数接口，否则旧工具会一直报告 0，不能据此推断没有重编译；修改 block/head/dtype 等结构配置仍会产生特化。

## 已有 checkpoint 的数值与质量回归

读取 `global_sparse_elu/step-50000.ckpt`，严格加载原参数，在固定的保存案例上对比原实现和新实现；没有训练或修改权重。验证覆盖 support=1,024/4,096/16,384，每个长度 16 张表，特征数 5/20/50，每表 128 个 query，共 48 张表、6,144 个预测/精度。模式为预训练验证使用的 `.train()+no_grad`，同时对一个 1K 案例运行整模型 backward，两个精度的梯度均有限。

| 精度 | 平均 CE 原→新 | 平均准确率原→新 | 预测翻转率 | 单表最大翻转率 |
|---|---:|---:|---:|---:|
| FP32 | 0.504754→0.503449 | 79.2318%→79.4596% | 1.9368% | 25.7813% |
| BF16 | 0.504938→0.505440 | 79.4108%→79.4108% | 2.5391% | 27.3438% |

16K 子集的 BF16 准确率为 74.7559%→74.4141%，CE 为 0.605869→0.604916。平均数接近不表示每张表都稳定：FP32/BF16 单个预测的最大概率差达到 0.438/0.510。这是实际的兼容性限制，不能将本次修改描述为旧 checkpoint 的逐预测等价替换，也不能用这一个 checkpoint 的小验证集推断所有分支和完整下游质量。

为排查是否仅由新 RMSNorm 引起，在最初 12 张表上额外恢复原 RMSNorm/原 autocast 路径，但保留新 attention 内核：预测翻转率 FP32 为 2.0833%、BF16 为 2.4089%，差异仍存在；默认新实现同一子集为 1.4974%/1.8229%。这说明仅恢复旧 Q/K dtype 不能恢复原模型预测。矩阵乘精度、归约次序、硬排序/top-k 边界及跨层传播都需要纳入进一步归因；本次没有完成逐层路由反事实分析，不能认定唯一原因。固定路由的 FP64 kernel 参考验证、全部输入梯度验证和整网 checkpoint replay 已通过，独立于这项整模型漂移测量。

本轮解决的是执行性能；48 张表上的整体质量没有显示明显退化，但长时间继续训练、全部 checkpoint 和下游真实数据仍须另行评估。

## 验证与复现

本地环境为 Python 3.12.16、PyTorch 2.5.1+cu121、Triton 3.1.0、Transformers 4.57.6。起初 PyTorch CUDA 运行时与宿主驱动不兼容，且 NVIDIA 用户态库与内核模块版本不一致；已在 `tabicl312` 安装可运行的依赖，用本地解包的匹配库修复当前进程加载路径，没有安装或替换系统驱动。`pip check` 通过。

从仓库根目录执行：

```bash
# 包装器使用 tabicl312，设置项目 src 和本地匹配的 CUDA 驱动库路径。
bash scripts/run_tabicl312.sh -c 'import torch; print(torch.__version__, torch.cuda.is_available())'

bash scripts/run_tabicl312.sh -m pytest -q --tb=short \
  tests/test_icl_kernel_regression.py tests/test_icl_attention.py \
  tests/test_icl_attention_optimized.py tests/test_linear_attention.py \
  tests/test_icl_mechanism_diagnostics.py tests/test_icl_training_diagnostics.py \
  tests/test_icl_screening.py tests/test_benchmark_icl_branches.py \
  tests/test_hybrid_optimizer.py tests/test_prior_pipeline.py \
  tests/test_inference_memory.py tests/test_checkpoint_analysis.py tests/test_inference.py

bash scripts/run_tabicl312.sh scripts/benchmark_icl_regression.py \
  --baseline-root /home/wbq/sparse/tabicl-baseline --output ../perf_results/reproduce \
  --rows 1024 4096 16384 --variants dense global sparse global_sparse global_sparse_alternating \
  --phases train prefill query --blocks 3 --warmup 3 --repeat 7

bash scripts/run_tabicl312.sh scripts/run_icl_extended_validation.py \
  --baseline-root /home/wbq/sparse/tabicl-baseline \
  --output ../perf_results/extended_reproduce --include-compile

bash scripts/run_tabicl312.sh scripts/validate_icl_checkpoint_regression.py \
  --baseline-root /home/wbq/sparse/tabicl-baseline \
  --checkpoint checkpoints/icl_screen_round1_staged/seed_42/global_sparse_elu/step-50000.ckpt \
  --cases checkpoints/icl_screen_round1_staged/validation.pt \
  --output ../perf_results/checkpoint_validation_reproduce \
  --support 1024 4096 16384 --cases-per-length 16
```

包装器可用 `TABICL_PYTHON`、`TABICL_DRIVER_ROOT` 覆盖本机路径；正常驱动环境不需要本地库修复。基准脚本运行两棵树时会分别设置 `PYTHONPATH`，不会误用同一份实现。

新增 GPU 回归覆盖 FP32/BF16/FP16 输出和 Q/K/V 梯度、非连续布局、热点路由、未使用 leaf、尾块只有一个 token、不同维度/块大小、ELU 饱和与 denominator floor、Q/V 不等维、混合 QK/V dtype、RMSNorm 输入/权重梯度、缓存失效以及六种分支的 checkpoint replay 和 optimizer step。稀疏参考使用 FP64，并在 CPU 上计算 softmax，以规避当前 Torch CUDA FP64 odd-width softmax 的归一化问题；近零真实梯度同时检查绝对误差，非零梯度仍检查相对误差。

最终回归结果为 **562 passed、9 skipped、4 warnings，13.44 秒**；跳过项为不可用的 MPS/XPU，警告为现有 multiprocessing fork 弃用提示。额外验证了 content_tree+hierarchical 路由、不同 support 长度、动态编译训练，以及长度变化时的 Triton kernel 复用。编译与 eager 的梯度比较固定同一张检索图，避免将 BF16 编译融合引起的硬 top-k 跳变误判为内核梯度错误；真实 router 的动态编译另由集成测试验证数值有限性，不宣称其预测与 eager 完全一致。

现有训练入口直接使用新内核，无需改 checkpoint 或新增训练 flag。FP32 使用 `--dtype float32 --amp False`；BF16 使用 `--dtype bfloat16 --amp True`。仅指定 BF16 dtype 而关闭 amp 时，训练仍是 FP32。原三个 stage 的长度课程、重计算和优化器均可继续使用；这些测试没有运行三个 stage 的完整长训练。

原始结果在 `/home/wbq/sparse/perf_results/`：`final_matrix/` 保存 90 组前后对照和每次样本、manifest、日志；`extended/` 保存更大规模的验证；`profile_final.json` 是 Chrome trace，`profile_final.txt` 是热点表。checkpoint 只读取，不写入训练权重。

`tests_final.log` 保存最终测试；`checkpoint_validation_final/` 保存最终源码在 48 张表上的 logits、逐表指标及源码 hash（结果与较早的 `checkpoint_validation_all/` 一致）；`checkpoint_legacy_norm/` 保存归一化消融。`extended/jobs.json` 保留最初两项 compile 失败记录，不做事后覆盖；修复后的成功记录分别是 `compile_float32_fixed.json`、`compile_bfloat16_fixed.json`。

基础 90 组使用的源代码 hash 早于后续非默认块 dispatch/tile 修复；该修复不改变测量所用的默认 BS=128、N≥256 路径。12 层扩展矩阵使用修复后的源码，其 manifest 可用于核对。

最终动态长度特化修复后再次独立复测 12 层、16K：FP32 sparse 1954.455→307.301 ms（6.36×）、global_sparse 811.587→213.622 ms（3.80×）；BF16 sparse 1432.568→153.308 ms（9.34×）、global_sparse 608.278→128.702 ms（4.73×），显存与前表一致。记录及最终源码 hash 在 `dynamic_final_confirmation/`。

## 下一步应优化什么

16K BF16 global_sparse 的 profiler 已经从原来的标量稀疏 kernel 转向投影/FFN GEMM、ELU backward、copy、layernorm 和 top-k。单独 dK/dV 约 1.1 ms，继续只打磨 sparse kernel 的整网收益会受其他部分限制。profiler 含真实 CUDA 事件，但各层级聚合时间有包含关系，不能把表内各行百分比相加。

当前 centroid router 依然有 `O(H*N*S*d/block_size)` 二次项；chunking 只限制工作显存。下一步优先考虑 score+top-k 融合、避免重复 contiguous 拷贝、层次 tree 的缓存，以及极热 leaf 的 dK/dV 分段归约。更改路由精度、层次近似检索、跨层候选复用均需单独验证召回与任务质量。

DSA 的逐行 indexer 仍会支付全 support 低维打分成本。当前执行内核优化并不能证明 DSA 更优；建议先做抽样 query 的逐行 oracle 上限，再在既有 checkpoint 上预热一个受 attention 分布监督的轻量 indexer，随后稀疏适应与长表适应。项目原来的三个 stage 是长度课程，可以承载这三个实验步骤，但二者不是同一种阶段划分。详细方案保留在工作区的 `analysis.md`。
