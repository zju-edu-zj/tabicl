# ICL attention 实验配置、归一化与诊断

六个 `scripts/train_v2_{clf,reg}_stage{1,2,3}.sh` 共用 `scripts/icl_recipe.sh`，预训练默认选择 `elu_linear_flex`。分类与回归、三个 curriculum stage 必须使用相同的 ICL 架构配置。`TabICL(...)` 的模型构造器仍默认 dense，以保持已发布 dense checkpoint 的加载行为；训练 CLI 的默认值则是实验架构。

旧 `linear_sparse` / `hybrid_linear_topk` 实现、入口和对应旧测试已删除，`icl_exact_topk`、`icl_sparse_chunk_size` 也不再接受。没有删除任何已有 checkpoint。旧架构 checkpoint 不能直接续训新架构；请使用独立 checkpoint 目录。加载器会校验架构、Flex 归一化、缩放模式、层间隔和块预算，避免参数形状相同但行为悄悄改变。改架构不是普通的 curriculum resume。

## 1. 当前模型的约束与九种选项

ICL 每层的 Q 是所有行，K/V 只能来自 labeled support 前缀。测试行不进入任何 support summary，也不能通过 attention 互相影响。这里不使用因果扫描或依赖原始行顺序的局部窗口；否则表格集合语义、查询分批与 KV cache 会改变。

以下复杂度仅描述 attention 主计算，不包含投影、FFN、排序和块索引。设总行数为 N、support 行数为 S、每头维度为 d、随机特征数为 m、slot 数为 M、Flex 每行候选数为 C。

| `--icl_attention` | 每层/混合方式 | 主 attention 成本（每头） | 用途与限制 |
|---|---|---|---|
| `dense` | support-only softmax | O(NSd) | 对照基线 |
| `elu_linear` | 正值 ELU+1 归一化 kernel | O((N+S)d²) | 纯 linear 路径；不是 softmax 的等价变换 |
| `linear` | Performer 风格正值随机特征 | O((N+S)md) | softmax kernel 近似实验；见下文数值限制 |
| `elu_linear_flex` | ELU 层，每第 E 层替换为 Flex | 两类成本按层相加 | 默认；全局混合与显式候选 softmax 交替 |
| `linear_flex` | Performer 层，每第 E 层替换为 Flex | 同上，linear 使用 m | 比较两种全局 kernel |
| `flex_sparse` | 全部 Flex | O(NCd)，另有块索引成本 | 单独检查 sparse 的收益与信息损失 |
| `induced` | learned slots 读 support，行再读 slots | O((N+S)Md) | 全局 softmax 压缩；slot 是信息瓶颈 |
| `linear_induced` | Performer 层，每第 E 层替换为 induced | 两类成本按层相加 | 全局 kernel 与全局压缩交替 |
| `induced_flex` | induced 层，每第 E 层替换为 Flex | 两类成本按层相加 | 用 slots 替代 linear 全局通道 |

混合是**按层替换**，不是单层中两个输出相加，也没有门控残差分支。默认 12 层、E=3，即第 3/6/9/12 层 Flex，其余 8 层 ELU。E=1 时所有层都成为替换算子；E 大于层数时没有替换层。

ELU 路径计算 `phi(x)=max(ELU(x)+1, 1e-6)`，先累积 `sum(phi(k) vᵀ)`、`sum(phi(k))`，再用 query 读取并除以正值 normalizer。特征映射、summary 与分母使用 FP32，避免长 support 的 BF16 累加误差。它的特征维度就是 d，`icl_linear_num_features` 对 ELU 无效。当前 KV cache 接口仍保存 projected support K/V，分批调用时重新聚合 summary；算子计算是线性的，但缓存存储还没有压缩成固定大小 summary。

Performer 路径使用固定、每头独立的 Gaussian orthogonal random features；超过 d 的特征使用独立正交块，增加 m 不再重复同一组方向。当前指数输入有数值 clamp，因此是 FAVOR+ 风格的有限精度实验实现，不能宣称完全复现原论文的无偏近似性质。不要直接把 ELU 的 QK norm/softmax temperature 套到 kernel 上：归一化或改变输入幅值会改变 kernel 本身。

## 2. Flex 做了什么，什么仍未得到证明

FlexAttention 是可编译的 attention kernel 接口；BlockMask 跳过没有可见连接的块，块内 `mask_mod` 仍可精确过滤连接。它不替你发现相关 support，也不自动降低一个 dense mask 的复杂度。参见 [PyTorch FlexAttention 文档](https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html)。

当前路由流程：

1. 对各头 Q/K 做可选 RMS 归一化，头均值投影到固定随机一维方向。
2. support 按投影值排序；每个 query 独立选择该排序上的连续 KV block 窗口。
3. query 按窗口起点排序，仅为了把相同/接近窗口打包。
4. 一个 query tile 的 BlockMask 覆盖该 tile 的窗口并集，`mask_mod` 再保留每行自己的窗口；padding key 始终不可见。
5. 编译的 Flex kernel 在可见连接上执行普通 softmax，输出恢复原 query 顺序。

修复了原来 query tile 共同决定候选的问题：现在单个 query 的逻辑候选只依赖它自身和 support。添加其他查询、改变查询分批不会改变其预测语义。KV cache、分批推理与 support permutation 已做回归验证；投影分数完全相同时的排序 tie 仍可能影响稀疏边界，浮点误差也可能在路由边界改变选择。

默认 block size=128、每行 4 个 KV blocks，通常 C=512；support 不足时退化为所有 support，尾块不满时 C 小于 512。窗口并集使 kernel 真正访问的块比例可能大于逻辑候选比例。当前 `from_kv_blocks` 为 backward 转置保留完整 KV block 索引容量，元数据为 O(B·ceil(N/128)·ceil(S/128))，不是严格线性的端到端内存；support 排序还有 O(S log S) 成本。

**这个单投影路由仍是实验启发式，不是成熟的 ANN 检索器，没有高召回保证，也没有直接的路由学习目标。** 默认将它与全局 ELU 交替，给未选中的 support 保留全局信息路径。但“存在全局路径”不意味着一定能抵消 sparse 的信息损失，必须用预训练与下游验证确认。

CUDA、每头 d≥16 时运行 `torch.compile(flex_attention)`。CPU 和极小 head 的测试走 eager 参考实现，会物化 dense scores；它们只用于正确性测试。Flex attention dropout 尚未支持，应保持 dropout=0。首次编译开销不代表稳定训练步耗时；变长输入可能触发额外编译。

## 3. 为什么 sparse 不必强制 SSMax

标准 sparse attention 仍是 `softmax(qkᵀ/sqrt(d))v`，只是分母在可见集合 A(q) 上求和。kernel 不需要 SSMax 才能工作。SSMax 处理的是 softmax 随可见 key 数变化而变平的问题，而不是“使用 sparse kernel”的问题。

对固定候选数 C=512 的 Flex，support 从 4K 增至 60K 并没有让该 softmax 分母多出几十倍项。如果继续用 support 总长度 S 放大 logits，长度补偿就与实际归一化集合不匹配，可能不必要地使注意力变尖锐。现在所有 Flex 长度缩放都使用**逐 query 的实际可见 key 数 C(q)**，排除 padding；原 dense 的 scalar 长度行为保持不变。

QK norm 和长度缩放不是二选一：

- QK norm 控制投影向量范数，减少 Q/K 范数增长导致的 logits/梯度异常。
- 长度缩放控制给定可见集合上的集中程度，不会改善正值标量缩放前后的 top-k 排序。
- QASSMax 的 elementwise 版本还能改变各维相对权重，可能改变相似度排序；它比单一 temperature 更有表达力，但不是纯粹的长度补偿。当前 MLP 基础缩放没有正值约束，不能凭名称假定始终是正温度。

[LimiX-2 最新报告（2026-09，§2.3）](https://arxiv.org/html/2609.17488v1) 结合 QK RMS normalization 与每头长度缩放；[官方缩放实现（检查的 commit）](https://github.com/limix-ldm-ai/LimiX/blob/516bf396333feb3198cf7aff8a6c10421f218e24/model/v2_0/softmax_scaling_mlp.py) 使用正值、有界系数。我们借鉴这一参数化，而没有声称迁移其完整模型或已验证其训练收益。[TabICLv2](https://arxiv.org/html/2602.11139v1) 的 QASSMax 仍保留为对照。

默认 Flex 路径：

```text
q_hat = q / sqrt(mean(q²) + 1e-5)
k_hat = k / sqrt(mean(k²) + 1e-5)
w_h    = 0.25 * (tanh(a_h) + 1)       # 0 < w_h < 0.5
beta_h = 2.5  * (tanh(b_h) + 1)       # 0 < beta_h < 5
s_h(q) = beta_h * (1 + w_h * log(C(q)))
logit  = s_h(q) * dot(q_hat, k_hat) / sqrt(d)
```

RMS 统计和缩放计算为 FP32，随后转回 attention dtype；V 不变。QK norm 没有 affine gain，由 beta 控制可学习幅值。初始化 beta=1、w≈0.00899，C=512 时 s≈1.056，接近无额外缩放。这里 `bounded` 指 w、beta 系数有界，不是 s 与长度无关，也不是对每个 logit 做 softcap。诊断名称 `temperature_*` 表示上述 **logit 倍率 s**，数值越大注意力越集中。

这套设计仍不能保证 attention 不塌缩：QK norm 后 `|dot(q_hat,k_hat)/sqrt(d)|≤sqrt(d)`（忽略 dtype 舍入），但学习到较大 s 后仍可能非常尖锐。因此必须监控熵、top1 概率与缩放系数。目前不额外叠加 logit softcap；它会进一步改变相似度间距和梯度。若训练中出现集中度失控，可单独消融 `cap*tanh(logit/cap)`，或先比较 RMS/none 与 RMS/bounded，避免同时改变多个机制后无法判断原因。

| `--icl_flex_scale_mode` | 行为 |
|---|---|
| `bounded` | 上述正值每头缩放；默认；忽略旧 ICL SSMax |
| `none` | 只有 1/sqrt(d)，可与 QK norm 独立组合 |
| `ssmax` | 使用 `icl_ssmax=True` 与 `ssmax_type`，传入实际 C(q)；缺少 SSMax 时直接报错 |
| `auto` | 根据 `icl_ssmax` 是否启用选择 ssmax 或 none；用于旧配置兼容 |

不会同时叠加 bounded 和旧 SSMax。Linear 层没有 softmax，也不创建 SSMax 参数。`icl_ssmax` 对 dense/induced 仍有效；column embedder 的 `col_ssmax` 独立，六个 v2 脚本仍启用它。

## 4. 参数与脚本设置

下表是六个 v2 脚本的统一默认值。所有环境变量在启动 shell 脚本前设置，对应参数也可直接传给 `python -m tabicl.train`。

| 环境变量 | CLI | 默认 | 作用 |
|---|---|---|---|
| `ICL_ATTENTION` | `--icl_attention` | `elu_linear_flex` | 架构选择 |
| `ICL_SPARSE_EVERY` | `--icl_sparse_every` | 3 | 混合层替换间隔；纯路径忽略 |
| `ICL_FLEX_BLOCK_SIZE` | `--icl_flex_block_size` | 128 | 块大小，必须为 128 的正整数倍 |
| `ICL_FLEX_NUM_KV_BLOCKS` | `--icl_flex_num_kv_blocks` | 4 | 每行 KV 块预算，正整数；与块大小相乘决定候选预算 |
| `ICL_FLEX_QK_NORM` | `--icl_flex_qk_norm` | True | 仅 Flex 的 Q/K RMS normalization |
| `ICL_FLEX_SCALE_MODE` | `--icl_flex_scale_mode` | bounded | 仅 Flex 的 logits 缩放 |
| `ICL_LINEAR_NUM_FEATURES` | `--icl_linear_num_features` | 64 | 仅 Performer 的 m；ELU 忽略 |
| `ICL_INDUCED_NUM_SLOTS` | `--icl_induced_num_slots` | 64 | 仅 induced 的 M；更多 slots 增加容量和成本 |
| `ICL_SSMAX` | `--icl_ssmax` | False | 旧 ICL softmax 缩放开关；不能覆盖显式 bounded/none |
| `SSMAX_TYPE` | `--ssmax_type` | qassmax-mlp-elementwise | 共用于启用 SSMax 的 column/dense/induced/Flex-ssmax 层 |
| `ICL_DIAGNOSTICS_EVERY` | `--icl_diagnostics_every` | 200 | 第 1 步及每 200 步采样；0 关闭 |
| `ICL_DIAGNOSTIC_QUERIES` | `--icl_diagnostic_queries` | 16 | 每个 Flex 层采样的 query 数 |
| `ICL_DIAGNOSTIC_TOPK` | `--icl_diagnostic_topk` | 32 | dense logits top-k 的 k；最多为 support 数 |
| `FAIL_FAST` | `--fail_fast` | True | OOM、非有限 loss/梯度/诊断即停止，避免静默跳过问题 |
| `TRAIN_DTYPE` | `--dtype` | float32 | 保留原 curriculum 的默认 dtype；5090 可显式设 bfloat16 |
| `RECOMPUTE` | `--recompute` | False | 重算 attention/FFN 节省 activation，增加计算 |
| `USE_FLASH_ATTN3` | `--use_flash_attn3` | stage1=False，stage2/3=True | 原 dense SDPA/column 路径设置，不是 Flex 开关 |
| `DRY_RUN` | 无 | 0 | 设 1 只输出展开的训练命令 |

CLI 自身的 `ssmax_type` 默认 `ssmax`、`fail_fast` 默认 False；shell 统一实验 recipe 使用上表值。布尔值使用 True/False。`ssmax_type` 改变也会影响启用了 SSMax 的 column 层，所以做纯 ICL 缩放消融时保持它不变。

主脚本还接受 `CKPT_DIR`、`LOG_DIR`、`NUM_GPUS`；stage2/3 分别用 `STAGE1_CKPT` / `STAGE2_CKPT` 指向上个 stage 的同架构 checkpoint。stage1 classifier 另支持 `MAX_STEPS`、`WANDB_LOG`、`WANDB_MODE`。所有脚本写 JSONL；TensorBoard/W&B 使用原脚本的开关。`DRY_RUN=1` 不运行预训练，classifier stage1 会创建日志目录。

```bash
# 核对展开后的命令
DRY_RUN=1 bash scripts/train_v2_clf_stage1.sh

# 5090：从新目录启动默认混合架构，BF16 + activation recompute
CKPT_DIR=/path/to/new/elu_flex_stage1 \
TRAIN_DTYPE=bfloat16 RECOMPUTE=True USE_FLASH_ATTN3=False \
bash scripts/train_v2_clf_stage1.sh

# 纯 linear 路径；不需要 Flex 的设置
ICL_ATTENTION=elu_linear CKPT_DIR=/path/to/new/elu_stage1 \
bash scripts/train_v2_clf_stage1.sh

# 四组归一化/缩放消融：各用独立 CKPT_DIR，保持其余配置一致
ICL_FLEX_QK_NORM=False ICL_FLEX_SCALE_MODE=none ...
ICL_FLEX_QK_NORM=True  ICL_FLEX_SCALE_MODE=none ...
ICL_FLEX_QK_NORM=True  ICL_FLEX_SCALE_MODE=bounded ...
ICL_FLEX_QK_NORM=True  ICL_FLEX_SCALE_MODE=ssmax ICL_SSMAX=True ...
```

最后四行是参数示意，`...` 需替换为脚本启动命令与独立目录。5090 用 Flex 不依赖外部 FlashAttention-3；`USE_FLASH_ATTN3=False` 避免将 Hopper 专用的原 FA3 设置当成 Blackwell 的必选项。

## 5. 诊断如何读

采样仅在 rank 0 的第一个 microbatch，Flex 对其中第一张表、每个 head，在测试 query 区间均匀抽取至多 16 行，与所有 support 比较。它只构造 `[1,H,16,S]` 的 FP32 logits，不构造训练规模 `[B,H,N,S]`，没有参与反向传播。Linear 的范围统计覆盖该 microbatch。checkpoint recompute 不重复诊断。

| 指标 | 含义与用途 |
|---|---|
| `recall_at_k` / `recall_worst_head` | 当前候选覆盖该层 dense logits top-k 的比例，均值与最差 head；应与 `candidate_fraction` 比较 |
| `retained_mass` / `_min` | 候选保留的 dense softmax 概率质量；通常比仅看 top-k 更能说明丢失的信息 |
| `candidate_fraction` / `visible_keys_min/max` | 逻辑候选占 support 的比例/实际数量，检查预算与尾块 |
| `kernel_block_fraction` | tile 窗口并集带来的真实块访问比例；大于候选比例表示 packing 有额外工作 |
| `dense_output_relative_error` | 同样 Q/K/V、缩放下，稀疏 softmax 与完整 softmax 的输出相对 RMS 差；在 out projection 前比较 |
| `logit_std` / `logit_abs_max` / `_selected` | 所有 support 与候选 logits 的数值范围；关注随训练的突增 |
| `entropy_normalized` / `sparse_top1_prob` | 候选 softmax 熵除以 log C、最大概率均值；前者约 1 表示均匀，约 0 表示尖锐 |
| `q_rms` / `k_rms` | 实际进入 logits 的采样 Q/K RMS；Q 已包含缩放，不必等于 1 |
| `temperature_min/max` / `length_weight_max` / `base_scale_max` | bounded 模式倍率与系数，检查是否逼近上界 |
| `denominator_min` / `denominator_clamp_fraction` | Linear 归一化分母下界、触发 1e-6 clamp 的比例；持续 clamp 值得排查 |
| `feature_min/max` / `output_rms` | Linear 特征与 attention 输出幅值 |
| `nonfinite_fraction` | 非有限 logits 或 linear attention 输出比例；应为 0 |
| `grad_in_proj_rms` | 采样 microbatch 反向后的 projection 梯度 RMS，已去 AMP scale，仍含梯度累积的 loss 除数；只在 rank0 记录 |
| `grad_norm` / `grad_clipped` | 原训练主循环记录的总体裁剪前梯度范数、是否触发裁剪 |

日志键为 `icl_diag/<layer path>/<metric>`，写入 `LOG_DIR/metrics.jsonl`，启用时也写 TensorBoard 的 `train/icl_diag/...` 和 W&B。诊断指标不做多个 microbatch 平均，也不是 DDP 全局平均；换 batch/sequence length 时应按长度分组比较。目前 dense/induced 未实现这些逐层诊断。

Recall 的参考是**当前层的完整 softmax**，不是任务标签，也不证明训练后的精度。若当前架构主动学习不同于 dense 的表示，低 recall 本身不能作为准确率失败的充分证据；但若希望把 Flex 解释为 dense 的高质量稀疏近似，低 recall 与低 retained mass 就是直接的反证。不要设一个脱离候选比例、概率分布和验证 loss 的统一“合格召回阈值”。

建议一同观察：验证 CE/pinball、按 S 分桶的 recall/mass、logit/entropy/gradient 曲线、候选预算改变后的性能。如果 logits 稳定但 mass 一直接近候选比例，优先检查/改进路由；QK norm 和标量 temperature 无法凭空产生更相关的候选。

## 6. 5090 实测与验证边界

2026-10-12，RTX 5090，PyTorch 2.9.1+cu130。测的是 ICL Encoder：B=1、N=4096、S=3276、12 层、D=512、8 heads、FFN=1024、BF16 autocast、FP32 参数、无 recompute。每项先 warmup 2 次（包含首次 Flex 编译），测量 5 次 forward+backward；无 optimizer step、无 prior、无 column/row encoder。诊断另跑一次，不计入正常迭代时间。

GPU 同时有其他训练进程，运行前仅约 12.67 GiB 空闲；以下是实际观测值，**不是隔离 GPU 上可信的速度排名**。峰值 allocated 是本进程 PyTorch 统计，不含其他进程、驱动或所有编译器内存。

| Flex QK/scale | median fwd+bwd | 5 次范围 | peak allocated |
|---|---:|---:|---:|
| raw / none | 34.94 ms | 28.45–39.84 ms | 1.326 GiB |
| RMS / none | 26.38 ms | 19.70–26.81 ms | 1.449 GiB |
| RMS / bounded（默认） | 35.21 ms | 27.34–44.88 ms | 1.520 GiB |
| RMS / QASSMax-elementwise | 38.87 ms | 37.40–52.97 ms | 1.584 GiB |

同条件另测 pure ELU 为 48.50 ms（40.59–52.90 ms）、1.458 GiB，dense SDPA 为 34.38 ms（27.82–38.72 ms）、0.886 GiB。此长度下没有观察到默认混合路径相对 dense 的明确速度/内存优势。Linear 的 FP32 features 与通用 autograd、Flex 的归一化/排序/gather/索引有成本，dense fused SDPA 本来就不保存完整 attention 矩阵；不能用渐近 FLOPs 推断实测峰值内存或短序列速度。对应文件 `elu_linear_4096.json`、`dense_4096.json`。

默认配置随机初始化、尚未更新参数时：四个 Flex 层候选占比约 15.1%–15.6%，top-32 recall 约 15.0%–16.0%，retained mass 约 15.3%–15.8%；与随机覆盖接近，**当前随机单投影没有显示有效检索能力**。logit std 约 1.05–1.08，abs max 约 4.36–4.85，entropy 约 0.91–0.94；linear 分母无 clamp、输出与梯度有限。额外诊断 iteration 约 150 ms，其中含正常 forward/backward；共享 GPU 上单次值不能精确估算诊断增量。

原始 JSON 与复现脚本在工作区 `exploration/results/icl_stability/`、`exploration/benchmark_icl_attention.py`。`qknorm_bounded_4096.json` 是上述默认配置最新记录，另外三份为 `raw_no_scale_4096.json`、`qknorm_no_scale_4096.json`、`qknorm_qassmax_4096.json`。

默认配置再检查 N=16384、S=13107：中位数 123.17 ms（范围 84.86–158.13 ms），peak allocated 5.524 GiB，输出/输入梯度有限，linear 分母无 clamp。候选占比约 3.87%–3.91%，recall 约 3.61%–4.37%、retained mass 约 3.89%–3.97%，再次接近随机覆盖；扩大 support 没有让这个路由自动变好。记录为 `qknorm_bounded_16384.json`。

16K 的 dense SDPA 对照为 213.71 ms（185.19–287.24 ms）、3.048 GiB，记录 `dense_16384.json`。该共享设备观测中混合路径耗时较低但峰值 allocated 较高；不能宣称同时提升速度和节省内存，也不能把此比值作为隔离硬件加速比。

16K 的 pure ELU 为 126.06 ms（106.91–177.50 ms）、5.284 GiB，记录 `elu_linear_16384.json`。与默认混合路径的耗时范围明显重叠，这组数据不足以判断 Flex 替换本身是否带来速度收益。

30 项正确性检查通过，包括正值 kernel 与显式 kernel 的输出/梯度对照、Flex 与 masked dense 的输出/参数梯度对照、CUDA BF16 编译 Flex、support permutation、query batching、KV cache、gradient checkpointing、CPU/CUDA 真实 TabICL 分类及分位数回归 microbatch 的 loss/backward/诊断、六个 launcher 参数展开、五种 SSMax 的逐行长度兼容，以及 resume 配置校验。这些检查验证实现语义和有限步数值健康；完整预训练稳定性、泛化精度和 sparse 路由学习效果仍需在目标训练机器验证。
