# ICL attention QK norm 与性能优化（2026-10-16）

本轮把 column 与 ICL 的归一化彻底分开：column 保持 TabICLv2 的 `qassmax-mlp-elementwise`，五个 ICL 筛选分支统一使用 bounded QK norm，不再让 ICL 继承 column 的 SSMax。Q/K 在每个 head 的 feature dimension 上独立 RMS normalize；可学习温度由每 head 基础项和非负 log-length slope 组成，相对初始温度硬限制为 `[0.5, 2]`。dense 使用 support 长度，sparse 使用可见候选预算；ELU global 没有 softmax，只使用归一化 Q/K。

## 已实现优化

1. 正常训练循环删除逐 step `torch.cuda.empty_cache()`；仅 OOM 恢复仍清缓存。allocator 的 reserved memory 是可复用缓存，不是活跃 tensor；逐 step 清空会强迫后续重新分配并同步，不能降低模型的 peak allocated。
2. sparse Triton forward、dQ、dK/dV 根据 int32 `order` 直接索引原始 K/V，不再生成完整排序 K/V；selected leaf 也为 int32。长序列反向先反转 query→leaf 的紧凑边，再由每个 leaf 独占归约 K/V 梯度。
3. projection 和 centroid 路由 kernel 直接读取 BF16/FP16 Q/K，以 FP32 累积，消除了路由入口的全量 `k.float()`/`q.float()`；centroid 按 128 leaves 分片，64K 不超过 shared-memory 限制。
4. activation checkpoint 的原始 forward 保存离散 `(order, selected)`，recompute 重放同一路由；soft attention、global 和 FFN 仍重算。这样既省掉第二次排序/top-k，也避免临界分数在两次 forward 选择不同路径。
5. global ELU CUDA forward 融合 `ELU+1`、support state/normalizer 和 query apply，不物化完整 FP32 feature tensor；backward 保持分块重算与 FP32 状态。
6. gate kernel 融合 per-head dot、sigmoid、global/sparse 混合和 BHND→BND 布局转换；正式筛选与性能矩阵启用 dynamic `torch.compile`。

content tree 的“segmented selection”做了两版精确原型：每层双 top-k 取中位两侧，以及单 top-k 后补集 compaction。16K 纯路由的双 top-k 为 2.14–4.31 ms（中位 2.60），补集版为 6.99–7.08 ms；最终的 CUDA segmented radix argsort 为 1.25–3.39 ms（中位 2.26）。瓶颈是每层 top-k/补集与临时张量，而不是移动完整 K/V。因此最终保留标量 score/index 的非稳定局部 radix argsort，Triton 继续负责 pivot、projection 和原始 K 读取。分数完全相等时不承诺 tie 顺序；stable argsort 实测为 3.34–3.40 ms（中位 3.36），却仍不能提供严格的集合 tie-break，所以未保留。这是基于测量撤销负优化，不把较慢实现留在主路径。

## 训练路径性能

条件：RTX 5090（共享设备）、PyTorch 2.9.1/CUDA 13.0、batch=1、12 个 ICL blocks、d=512/8 heads、support=总行数的 80%、BF16、activation checkpoint、dynamic `torch.compile`，计时包含完整 forward+backward，不含 optimizer/prior。每格为多次运行中位数 ms；共享 GPU 干扰明显，原始 min/max 必须与中位数一起看。

| 分支 | 1K | 4K | 8K | 16K | 64K | 64K peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 8.51 | 23.29 | 51.91 | 222.49 | 2681.65 | 3.428 |
| global | 6.77 | 17.60 | 22.81 | 46.63 | 224.73 | 3.040 |
| global_sparse_alternating | 24.85 | 30.69 | 54.50 | 103.85 | 623.10 | 3.350 |
| global_sparse/projection | 32.58 | 51.75 | 61.18 | 126.89 | 585.73 | 3.465 |
| global_sparse/content_tree | 28.15 | 54.76 | 96.57 | 202.57 | 617.93 | 3.467 |

64K 相对 dense 的中位加速为 global 11.9x、alternating 4.3x、projection 4.6x、content_tree 4.3x。短表上 sparse 的固定排序、top-k 和自定义 backward 成本仍使其慢于 dense/global；这不是 dispatcher 的理由，而是后续 kernel 优化应直接针对的成本。当前五分支都保留为实验对象，不依据速度裁剪模型路径。

route-only 的 projection/centroid 在 16K 为 0.777 ms，在 64K 为 14.19 ms；content_tree/centroid 64K 为 20.19 ms。64K 的完整范围分别为 dense 2643–2813 ms、global 199–369 ms、alternating 434–742 ms、projection 494–830 ms、content_tree 491–830 ms，说明共享负载足以改变相邻方案排序。1K projection 的 `torch.compile` on/off 最快样本为 26.76/27.45 ms，约 2.5% 改善；受外部负载影响，其中位数反而为 34.15/27.91 ms，因此目前只证明 compile 链路可用，不能宣称稳定提速。原始数据在 `benchmarks/icl_optimized_short_retry/summary.csv`、`benchmarks/icl_optimized_64k_retry/summary.csv`、`benchmarks/train_content_tree_final_*.json` 与 `benchmarks/train_projection_final_1k_*.json`。

## 正确性与边界

测试覆盖原始 K/V 非恒等 permutation 的 sparse 输出及所有梯度、FP16/BF16 grouped backward、fused ELU 输出/梯度、fused gate/布局输出/梯度、CPU/CUDA content-tree 路由一致性、int32 metadata、五分支 bounded QK norm、cache 与 checkpoint 路由只计算一次。正式质量结论仍必须来自 20K–50K screening；这些 kernel 基准只证明速度、内存和数值一致性，不能证明 sparse attention 已逼近 dense。

性能复现：

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src conda run --no-capture-output -n stdplm \
  python scripts/benchmark_icl_suite.py --output /tmp/icl_optimized \
  --rows 1024 4096 8192 16384 65536 --repeat 7 --warmup 3 \
  --memory-limit-gib 8 --matrix-only --compile
```
