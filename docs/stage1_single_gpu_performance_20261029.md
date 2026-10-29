# Stage 1 单卡 FP32 / BF16 优化与动态形状实验

本次基线为 `4a0cd58e1fac7c55e002b945309d3e3a8fd88572` 的 tracked 源码。
所有 GPU 计时串行执行：RTX 3090 24 GiB，PyTorch 2.8.0+cu128，Triton 3.4。
模型保持 Stage 1 的 3 COL + 3 ROW + 12 ICL、E=128、128 inducing points、
8 heads、4 CLS、global_sparse/ELU/QK norm，关闭 recompute、FA3。

## 测量范围

`scripts/benchmark_stage1_training.py` 调用真实 `Trainer.run_batch`，包含 H2D、
前后向、loss/nonfinite 检查、梯度裁剪、hybrid Muon/AdamW、scheduler、指标回传。
输入为预先准备的 pinned CPU 合成表；不包含 prior 生成、DataLoader 等待、日志落盘、
checkpoint 或 DDP。不能把下面的模型步骤收益直接当作完整作业收益。

计时在每步前后同步 GPU。所有首调用和编译尖峰均保留；同时报告剔除前5步后的稳态
统计，以区分两种问题。FP32 保持 FP32 张量和计算路径，但同训练器一样允许 TF32
matmul；它不是禁用 TF32 的严格 IEEE FP32 benchmark。脚本提供 `--strict-fp32` 对照。
BF16 使用 autocast，模型参数/优化器状态仍为 FP32，不偷偷将 FP32 attention 转成半精度。

共有44个对照进程、2210个计时训练步骤。汇总、逐步耗时、图数变化、源码hash、
冷缓存标识和CLI smoke指标保存在
[`performance_results/stage1_single_gpu_20261029.json`](performance_results/stage1_single_gpu_20261029.json)。
这些是单次串行对照，不是多次独立重复后的置信区间；1–2%的差异应保守解释。

## 性能根源：大激活前端及精度对应的 attention 后端

对修改后的 eager 再做 CUDA event 分段，B=micro=4、T1024、H100、S614，
3步warmup后7步取中位数，统计前后向、不含优化器/H2D：

| 路径 | COL前后向 | ROW前后向 | ICL前后向 | 完整模型前后向 |
|---|---:|---:|---:|---:|
| FP32 / TF32 matmul | 324.4 ms | 222.2 ms | 51.8 ms | 597.6 ms |
| BF16 autocast | 112.8 ms | 98.3 ms | 35.5 ms | 246.6 ms |

分段中位数相加与总时间中位数会有少量差异。COL+ROW分别占约91%/86%，
优化重点应放在这些大激活路径。12层ICL并不意味着它就是当前主要耗时来源。
hybrid优化器独立零LR计时约20.6 ms，两种精度相近；参数/优化器状态仍为FP32。

设B为micro-batch、T为行数、H为列数、E为embedding维度、I为inducing points：

- COL的induced attention主要工作量约为`B × H × (T+S) × I × E`，
  加上投影/FFN的`B × H × T × E²`。固定I=128时不是完整的`T²`列自注意力。
- ROW逐行做列间attention，主要为`B × T × H² × E`，再加`B × T × H × E²`。
- LayerNorm、激活、residual和布局拷贝都要反复读写`B × T × H × E`量级的激活。
  只算B4/T1024/H100/E128的一份激活就有52,428,800个元素，FP32约200 MiB，
  BF16约100 MiB；反向保存的是多层、多份张量。

单步profiler还确认了两条精度路径的实质差异：FP32的9次SDPA使用
`_efficient_attention_forward/backward`，累计约75.9/236.0 ms；BF16使用
`_flash_attention_forward/backward`，约10.5/40.5 ms。这里分别统计对应算子，
不叠加其父级SDPA时间。该GPU上BF16同时受益于不同attention内核和Tensor Core，
不能期待仅靠编译把保持FP32语义的路径变成BF16速度。

因此有三类独立问题：持续支付的大激活计算/访存，偶发的编译和诊断阻塞，
以及GPU等待prior的供给瓶颈。优化必须分别测量；只盯重编译会漏掉前者，
只看稳态GPU耗时则会漏掉后两者。

## 已落地的修改及原理

### 复用 support K/V

ISAB 原先把 `src[..., :S, :]` 写了两次。两次 slice 数值相同，但 Python 对象不同。
`MultiheadAttentionBlock` 按对象共享关系复用 LayerNorm，PyTorch packed projection
也据此选择合并 K/V 的 GEMM。因此原先重复了大 support 上的归一化，并分别投影 K/V。
现在只切一次，同一个 support 同时传给 K 和 V；普通训练和 cache 存储路径都已修复。

数学运算保持不变；合并 GEMM 以及梯度累加顺序可能改变浮点末位。

### QKV 拆分使用视图

PyTorch 的通用 `_in_projection_packed` 会为了标准 MHA 的 unpack 布局做
transpose/contiguous。当前代码随后又把投影变成 SDPA 所需的多头布局，形成多余的
整块激活拷贝。新的 `_project_qkv` 对共享 QKV/KV 使用一次 packed linear 后沿最后
一维 `chunk`。每个 head 的最后一维依旧连续，SDPA 可直接接受这些带 stride 的视图。
不同 K/V 的通用情形继续使用 PyTorch helper。

这主要减少显存读写和临时激活，不改变 attention 的 token、mask、softmax 或模型结构。
编译器本来就能消除一部分拷贝，因此 eager 收益大于 compiled 是合理结果。

### 有界动态编译

仍只编译 COL 的 tensor common path 和 ROW aggregation，不编译整个 Trainer。
`dynamic=True` 本身不会保证任意输入只用一张图。默认 duck shaping 会在首次输入
中把“数值恰好相等”的不同维度合成同一个符号。例如 B=H，或者 H=S；以后独立变化
就违反 equality guard，即使两个维度本来都允许动态。

本次只在前端 compiled callable 的局部关闭 `use_duck_shape`，避免这种首次样本依赖。
退出后恢复配置，不全局修改 Torch 设置，不提高 recompile limit，不用 Python 宽度
字典维护无穷变体。该开关是 Torch 私有 API，升级 Torch 时应重跑 guard 和 CUDA 回归。

H=1 仍是合法的独立 COL 变体：Torch 对 singleton 的广播/stride 语义有特殊处理，
不应该以“必须永远只有一张图”作为正确性目标。dtype、是否有梯度、batch=1、dropout、
冻结模式和不同分支也可能合法地产生变体；本次实验固定训练模式和精度分别测量。

### 数据管线与诊断

原先 NestedTensor 在 DataLoader 中被 pin，随后主进程 `to_padded_tensor` 分配的新
张量却是 pageable memory。`non_blocking=True` 无法凭空把它变为 pinned memory。
现在 collate 在 worker 中先完成相同的 padding，再由 DataLoader pin 最终布局，
把分配/拷贝移出训练主线程，并恢复异步传输所需条件。直接调用 `run_batch` 的 nested
输入兼容路径仍保留。H2D 与 GPU 计算完全重叠还需要独立 stream/双缓冲，本次未声称
已经实现；最终 wall time 也仍取决于 prior 供给速度。

梯度诊断原先对 model、component、optimizer group 多次重复扫同一个参数，并在
每个参数的 finite count / any / sum / max 上读 CUDA scalar。现在每个参数统计一次，
不用 `grad[finite]` 的动态 compaction，统一回传小型摘要，再在 CPU 聚合重叠类别。
保留缺失梯度、NaN/Inf 比例、finite-only RMS/max 的定义，以及诊断频率和 fail-fast。

专门把诊断间隔缩短为3步进行15步对照，B=micro=4/T1024/BF16/compile all。
第6步起，基线普通步/诊断步中位数为197.35/339.01 ms，修改后为191.35/219.68 ms。
诊断的额外成本约142→28 ms，减少约114 ms的尖峰。真实默认间隔200时，
仅这项的平均收益约0.57 ms/step，不能把高频诊断实验的20%均值改善外推到默认配置。

## 固定形状结果

以下为20步实验中第6步起的中位数，单位 ms；B=micro-batch=1，H=100，S=floor(0.6T)。
这样三种行数使用完全相同 batch，可在 24 GiB 上公平比较。

| 路径 | T | 基线 eager | 修改后 eager | 基线 compile all | 修改后 compile all |
|---|---:|---:|---:|---:|---:|
| FP32 / TF32 matmul | 1024 | 192.24 | 182.27 | 167.63 | 165.93 |
| FP32 / TF32 matmul | 2048 | 359.11 | 341.30 | 312.11 | 309.87 |
| FP32 / TF32 matmul | 4096 | 736.66 | 701.41 | 649.14 | 642.78 |
| BF16 autocast | 1024 | 101.08 | 94.57 | 77.78 | 79.68 |
| BF16 autocast | 2048 | 158.37 | 146.72 | 117.58 | 115.56 |
| BF16 autocast | 4096 | 279.29 | 257.03 | 204.95 | 199.95 |

BF16/T1024/B1 的 compiled 个例稍慢，不能声称所有形状都加速。
这组有共享磁盘编译缓存，首调用冷暖不同，不能直接用它比较启动优化。

更接近默认 micro-batch 的25步实验（B=micro=4、T1024、H100、S614）：

| 路径 | 基线 eager | 修改后 eager | 基线 compile all | 修改后 compile all |
|---|---:|---:|---:|---:|
| FP32 / TF32 matmul | 659.65 | 629.61 | 557.26 | 552.01 |
| BF16 autocast | 294.90 | 271.45 | 197.92 | 191.52 |

这里每个optimizer step只包含一个micro-batch。脚本默认global batch=64，
在单卡micro=4时每步有16个micro-batch；上表不是默认完整optimizer step的毫秒数。
eager的FP32/BF16耗时分别降低4.6%/8.0%；compiled分别降低0.9%/3.2%。
本次数据管线修改没有计入这个收益，因为基准输入本身已经是pinned dense tensors。

BF16 eager 峰值 allocated 从11.37降至10.84 GiB；compiled 约9.49 GiB。
T4096/B1 的 BF16 eager 从10.59降至10.05 GiB；FP32 eager 从16.33降至15.98 GiB。
显存不是只由参数决定：COL/ROW 要保存 B×T×H×E 级别的中间激活。
不要把 T 增大4倍同时还假定原 micro-batch=4 一定放得下。

## 行数、列数和 support 同时变化

每个进程150步，T在1024/2048/4096轮换，H包含1和1–100的随机排列，
S在0.3T–0.9T变化，类别数2–10变化，labels保留从完整序列切片的真实stride。
B=micro=1。两种精度均：第0步产生COL/ROW两张图；第1步H=1增加一张COL图；
第2步开始到结束均为3张图。第一次进入T4096以及后续所有行数切换没有继续加图。
所有步骤loss/训练指标有限，没有OOM、没有达到重编译上限。

| 路径 | 基线 eager 全150步均值 | 修改后 eager 全150步均值 | 基线 compiled 全150步均值 | 修改后 compiled 全150步均值 |
|---|---:|---:|---:|---:|
| FP32 | 253.95 | 244.41 | 349.58 | 344.15 |
| BF16 | 130.97 | 120.29 | 235.70 | 237.73 |

修改后第6步起的均值：FP32 eager/compiled 为246.58/225.25 ms，
BF16为118.12/97.93 ms。这里应看均值，混合形状的中位数容易被较短序列左右。
compiled稳态更快，但150步总时间仍输给eager，不能隐藏开头约17–21秒的编译成本。
基线和修改后的磁盘缓存命中不同，这张表描述各次完整实测，不把小的启动差异归因于代码。

## 独立冷缓存下的真实启动成本

另外为基线/修改后各创建全新`TORCHINDUCTOR_CACHE_DIR`，BF16、B=micro=4、
T1024/H100/S614，分别运行150步，保留首调用：

| 指标 | 基线 | 修改后 |
|---|---:|---:|
| 首步（含编译） | 42.03 s | 41.28 s |
| 第6步起均值 | 195.66 ms | 192.57 ms |
| 全150步均值 | 474.49 ms | 466.47 ms |
| 全150步总时间 | 71.17 s | 69.97 s |

两次都只生成COL/ROW两张图，此后没有新增图。固定形状仍有约41秒首调用成本，
所以“大尖峰”不能一律诊断为shape变化引起的重编译。首调用差异只有一次样本，
不声称这次修改稳定缩短了编译本身；确定的目标是减少稳态冗余并限制无谓变体。

## 应如何选择运行方案

短实验、经常切换架构或冻结模式：保留 `COMPILE_FRONTEND=none`。本次 eager 的
5–8% 改善直接生效，没有Dynamo/Inductor重编译尖峰。

长时间训练、配置固定且监测显示图数已经稳定：可用 `COMPILE_FRONTEND=all`。
稳定后的平均节省可以摊销有限编译成本。判断应使用

`总时间 ≈ prior未被隐藏的等待 + 编译总成本 C + N × 稳态训练均值 + checkpoint/log成本`

相对 eager 的粗略盈亏点为 `N > C / (t_eager - t_compiled)`。
上述变长实验约节省20–21 ms/step；17–21秒启动成本对应约800–1100步，
但它不是对其他硬件、batch、缓存状态或输入分布的保证。原始500K步配置更有机会受益；
不断出现新图、silent fallback或重新启动会改变这个估算。
冷缓存固定B4的启动成本更高，但每步相对eager也节省约79 ms，
粗估需约520步摊销41秒启动；该估计同样不能替代真实作业测量。

可用`TORCH_LOGS=recompiles`检查实际guard失败。如果图数已稳定却仍出现尖峰，
应对照诊断步、checkpoint写盘、`prior_time`和GPU/CPU事件，而不是继续扩大Dynamo缓存。
建议保留磁盘编译缓存以减少重复启动成本，但缓存命中不等于无需guard检查或保证零编译。

不建议先做原始X的宽度padding：`same` grouping沿真实H取模，先padding会改变特征邻居。
即使在grouping后分桶，ROW仍需要正确mask，dummy列还会增加注意力和FFN工作量；
若图数已经有界，为消除少数启动编译引入每步padding开销，通常不利于平均耗时。
也不建议 `dynamic=False` 固化整个前端：随机support长度会产生大量静态版本。
还实际试过统一设置`specialize_float=True`，变长实验稳态出现约5%的回退，因此未采用；
只保留原有ROW recompute确实需要的局部设置。

无需编译尖峰又要继续追求compiled的稳态速度，下一步应考虑经过数值验证的独立融合
LayerNorm/residual/GELU kernels，以及更少的跨COL/ROW布局物化。这能把形状相关的编译
边界缩小为固定head/channel维度的算子，而不是为整个训练图不断建版本。实现成本和
梯度验证成本明显高于本次移除冗余操作，不能把尚未实现的收益计入本报告。

## 使用和复现

Stage1 classifier已支持 `MAX_SEQ_LEN` 环境变量。例如24 GiB单卡长训练可从以下保守配置开始，
保持global batch=64，通过micro=1累积梯度；是否增大micro应按最坏H/S测量显存：

```bash
NUM_GPUS=1 MAX_SEQ_LEN=4096 MICRO_BATCH_SIZE=1 \
TRAIN_DTYPE=bfloat16 COMPILE_FRONTEND=all \
bash scripts/train_v2_clf_stage1.sh
```

FP32路径将`TRAIN_DTYPE`改成`float32`；短实验将`COMPILE_FRONTEND`改为`none`。
`RECOMPUTE=False`仍是显存允许时较快的选择。checkpoint和日志目录沿用原脚本配置，
实际使用前设置为本机路径。prior workers/prefetch应结合`prior_time`和CPU/内存调整，
不要把GPU空等误判为模型计算不足。

对于供给侧，prefetch主要吸收生成时间的抖动，不能提高长期生成吞吐。
假设平均生成一批需`g`秒、有效worker数为`W`、GPU消费需`t`秒，
稳态至少需要`g/W <= t`才能持续供给。BF16或编译加速后`t`下降，原先够用的worker
可能不再够用；应重新观察完整作业的均值和`prior_time`，避免无边界增大队列占用。
本次worker padding恢复了pinning条件，并未以短CLI smoke估算prior吞吐提升。

```bash
PYTHONPATH=src /home/wbq/anaconda3/envs/tabicl312/bin/python \
  scripts/benchmark_stage1_training.py --dtype bfloat16 --mode all \
  --rows 1024 2048 4096 --variable --steps 150 --output /tmp/stage1-variable.json
```

基线用`git archive <基线SHA>`解包至独立临时目录，再将PYTHONPATH改为该目录的src；
两个进程顺序执行。冷启动对照应分别设置全新的`TORCHINDUCTOR_CACHE_DIR`，而不是
只清Dynamo内存缓存。基准JSON含源码hash、环境、每步形状/计时/图数、均值、P95、
峰值显存和compiler counters。backend/AOT进入次数可能包含内部restart，不能直接当作
可执行图数；这里使用Dynamo的`unique_graphs`。

数值验证包含投影输出/全部梯度、真实CUDA SDPA的FP32/BF16视图路径、
前端编译与eager输出/梯度、checkpoint keys/parameter身份、dropout/sentinel/ragged/
many-class/freeze/recompute，以及首次H=B/S/T偶然相等的shape回归。
这些验证不意味着长期训练轨迹逐位相同：BF16、融合运算和离散top-k路由会放大末位差异，
长期质量仍需用固定评测集比较。

实际执行的针对性验证共103项通过：attention/prior/hybrid三组41项，
performance的CPU路径54项，真实CUDA编译/梯度/recompute/多support路径6项。
另有2项ISAB缓存回归，验证新的K/V视图在保存support后分块查询仍与原布局一致。
按单卡任务范围跳过DDP验证。另用真实脚本、完整graph_scm prior、2个worker、
B4/micro1分别完成FP32/T2048和BF16/T4096的step1–2，保存checkpoint后重新启动恢复至step3。
两次都正确加载model/hybrid optimizer/scheduler，355个state_dict参数键无编译包装前缀，
参数和训练指标有限、OOM计数为0。此smoke仅用于流程正确性；恢复时把max_steps从2调到3，
不能作为固定schedule下的长程训练质量或吞吐对照。
