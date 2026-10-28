# Torch 2.8 全模型训练优化与验证

日期：2026-10-28。本文件是本地审计记录，不进入性能优化 commit。

## 环境与对照

已更新现有 conda `tabicl312`：Python 3.12.16、Torch 2.8.0+cu128、
Triton 3.4.0、CUDA wheel runtime 12.8、cuDNN 9.10.2.21，`pip check` 通过。
测试进程直接使用该环境，不经注入旧 NVIDIA 用户态库的 `run_tabicl312.sh`。
驱动为 615.71.09，工具实际枚举的 GPU 为 RTX 3090。本文只报告当前运行环境的结果。
系统 nvcc 11.5 不参与现有 Torch/Triton/Inductor 编译，GCC/G++ 11.2 可用。

分支：`perf/stage1-torch28`。ICL 优化独立提交 `d0cdd57`。
前端、Trainer 与优化器提交为 `26cfb3a`；审计脚本、审计报告、结果及 checkpoint
均不纳入生产提交。
本轮前端/Trainer/优化器对照是只读 worktree `tabicl-torch28-baseline` 的
`d0cdd57`，因此下面的全模型加速**没有重复计入此前 ICL 优化收益**。
基线和优化代码均在同一 Torch 2.8 环境测量，不是旧 Torch 与新 Torch 的混合比较。

## 实施内容

- 标签仅编码一次再沿列广播，避免 B×F 次重复编码和对应反向归约。
- 列模块跳过随后被 row CLS 参数覆盖的占位列。仅在 paired 模型、dropout=0、
  非 ragged d 的训练路径使用；公开 column 接口默认保留旧输出，推理不变。
- `SkippableLinear` 用 `where` 代替 `.any()` CPU 同步和布尔写入。
- 回归路径删除无意义的标签最大值 CPU 同步。
- 修复 mixed-radix 多 digit 重用/修改 buffer 导致的原有 backward version 错误。
- 列/行纯张量方法可选动态编译，保持 module、Parameter identity、state_dict keys。
  非连续路由、自定义 ICL kernels 仍按其已有边界执行，并非整个模型 fullgraph。
  实际 NCCL 启动发现 Torch 2.8 DDP bucket 图切分器处理 RoPE 浮点输出时
  `float has no attribute meta`，单进程 raw module 测试无法发现。
  最终仅在 component 调用期间通过配置上下文关闭该 compiler pass，
  不关闭 DDP 通信，不常驻改全局配置；新增真实 DDP backward/config 恢复测试。
- H2D 使用 non_blocking；每 micro 的 detached loss/accuracy 在 GPU 累加，
  step 末一次转回 CPU；保留 fail_fast 和梯度诊断。
- 混合优化器内部 AdamW 按 device/dtype/每参数自己的 step 分组 foreach，
  保持缺失梯度不推进时钟、不衰减；cautious decay 保留原语义路径。
- Muon Newton–Schulz 可选编译。静态 shape 初版在多种矩阵和缺失梯度时
  耗尽 Dynamo recompile_limit，测试发现后改为动态 graph；不改全局缓存上限。
  优化器 Python 状态管理和 checkpoint 格式不参与编译。
- 动态真实训练额外发现 SSMax 的 `torch.tensor(math.log(SymInt))` 会在
  sentinel 分支的编译断点后把 support 长度特化，耗尽8次默认重编译额度并
  局部回退 eager。这在第一次固定列数的真实训练里也存在，不能只查看
  exit0/无NaN就声称动态编译稳定。最终改用图内 scalar_tensor + log；
  保留 double log 到最终 dtype 的舍入语义，eager 路径不变。不提高缓存上限，
  新增12种support长度的实际编译反向测试，设置超上限即报错以防静默回退。

两个编译选项均默认关闭：`--compile_frontend none|row|all`、
`--compile_optimizer True|False`。配方对应 `COMPILE_FRONTEND`、`COMPILE_OPTIMIZER`。
实测 frontend 收益显著；optimizer 编译在本组完整 step 中只节省约 0–1.4ms，
宽表 FP32 个别测点有约 1ms 波动，不应把它描述为主要加速来源。

## 包含更新的完整 step 测量

本节24组矩阵为 SSMax 动态长度修复前的固定 support 数据；该修复仅影响
编译路径，最终配置另做六组完整step复测，原始记录全部保留，不覆盖旧证据。

真实 Trainer、E128、3 column + 3 row + 12 ICL blocks、128 inducing、4 CLS、
global_sparse/ELU、QK norm、QASSMax column、无 recompute；输入 pinned CPU，
batch8/micro4、60% support；TF32 允许，训练 SDPA 不使用 cuDNN。
每种配置独立进程，5 warmup + 15 measurement，串行独占 GPU。
包含 H2D、两次 forward/backward、梯度累计、裁剪、混合 Muon/AdamW、
scheduler 和指标回传；**不含 prior 生成、DDP、保存日志/checkpoint**。

| dtype | 行×列 | 基线 ms | 新 eager ms | 编译前端 ms | 前端+优化器编译 ms | 基线/最终 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 1024×100 | 1364.76 | 1337.01 | 1088.74 | 1089.92 | 1.25× |
| BF16 | 1024×100 | 627.82 | 608.18 | 373.00 | 372.41 | 1.69× |
| FP32 | 128×100 | 264.86 | 252.89 | 204.07 | 203.16 | 1.30× |
| BF16 | 128×100 | 164.84 | 154.74 | 105.54 | 104.20 | 1.58× |
| FP32 | 128×10 | 87.26 | 76.34 | 71.40 | 70.01 | 1.25× |
| BF16 | 128×10 | 82.36 | 73.26 | 67.62 | 66.72 | 1.23× |

宽表峰值 allocated（此脚本包含 warmup 与优化器 state）：FP32 18.319→17.360GiB，
BF16 11.725→9.214GiB。短表收益不消失，但 batch 很小的 BF16 不一定比 FP32 快；
启动、路由、小 kernel 等固定开销仍在，不应只按理论 FLOPS 推断。

原始报告：`../perf_results/torch28_step_{base,eager,front,both}_*.json`。
每个文件保存完整 samples/metrics、shape、环境、首轮时间。首次编译遇到的
graph break 包括分类标签 `max().item()`（mixed-radix 分支需要），不是 NaN 或
编译失败；不要为去警告删除分类边界检查。

## 正确性与训练验证

测试覆盖：分类/回归、same/valid/no feature grouping、affine、target-aware
开关、sentinel、dropout/ragged 回退、冻结列、mixed-radix backward、
FP32/BF16 动态 batch/行/列、recompute、model strict load、optimizer moments、
缺失梯度、TF32 开关、实际梯度累计与保存/续训。

编译纯数值对齐使用 dense ICL 隔离不可微路由；global_sparse 另做
实际 Trainer checkpoint/resume 与已有 checkpoint 的质量回归，不能用放宽
top-k 输出容差来假装完全等价。新编译配置不写入模型 state keys。

最终针对本轮范围的 14 个测试模块：**607 passed / 9 skipped / 5 warnings，86.23s**。
日志：`../perf_results/torch28_final_dynamiclog_tests.log`。四条警告来自既有 prior
fork 测试的 Python 3.12 多线程 fork 警告，一条是测试未启用 TF32 的性能提示。
这里没有声称运行全部 sklearn、forecast、SHAP 及外部官方预训练权重下载测试。
额外尝试 datetime 测试时，因缺少可选依赖 statsmodels 在收集阶段停止，
该项不计入通过数，日志为 `../perf_results/torch28_datetime_tests.log`。

DDP 与 SSMax 动态长度修复后，最终六组完整 step 复测（5 warmup / 15 repeat）：

| dtype | 行×列 | 最终 step ms | 同口径基线/最终 | peak allocated GiB |
|---|---:|---:|---:|---:|
| FP32 | 1024×100 | 1092.75 | 1.25× | 17.361 |
| BF16 | 1024×100 | 372.75 | 1.68× | 9.214 |
| FP32 | 128×100 | 203.34 | 1.30× | 4.051 |
| BF16 | 128×100 | 104.22 | 1.58× | 2.344 |
| FP32 | 128×10 | 69.82 | 1.25× | 0.920 |
| BF16 | 128×10 | 65.90 | 1.25× | 0.702 |

原始记录：`../perf_results/torch28_final_step_*.json`。包括局部 DDP compiler
配置上下文的运行开销；没有把它漏出计时。

## 真实供数、NCCL、保存与独立重启

实际 `torch.distributed.run --nproc_per_node=1`，graph_scm、4 workers、
prefetch2、in_order=False、过滤不可预测数据开；batch8/micro4、真实 3+3+12，
随机 support 30–90%，每配置 16 steps，保存后重启进程续训到 18 steps。
检查 JSONL step 完整连续、loss/梯度有限、0 OOM；模型、混合优化器和 scheduler
真实保存/读取。以下为 step7–16 中位数，不含启动、保存、退出等固定成本。

| dtype | 行×列 | 基线 train ms | 优化 train ms | train 加速 | 基线 prior+train ms | 优化 prior+train ms |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 1024×100 | 1350.82 | 1074.23 | 1.26× | 1350.88 | 1074.31 |
| BF16 | 1024×100 | 585.20 | 365.75 | 1.60× | 654.57 | 373.92 |
| FP32 | 128×100 | 262.09 | 203.28 | 1.29× | 296.43 | 278.54 |
| BF16 | 128×100 | 165.89 | 113.66 | 1.46× | 465.44 | 125.77 |

另 BF16 128×10 quantile regression，train **77.00ms**，重启到 step18 成功，
此项没有回归任务同口径基线，不声称其加速比。
另两组 FP32/BF16 每表随机 1–100 列、随机 support，训练48步后独立重启到50步；
稳定段 train 中位数分别 **602.11ms / 252.25ms**，prior+train 分别
**602.20ms / 337.70ms**，包含启动保存续训的 wall 为 **107.91s / 106.04s**。
随机宽度没有同任务基线，不报告加速比。最终 **11 种配置全部成功**，所有
train/resume 日志扫描均未发现 recompile_limit 回退、Traceback 或运行错误。
最终保存数据：`../perf_results/torch28_cli_dynamiclog/*/{summary.json,train.log,resume.log}`。
修复前结果：`../perf_results/torch28_cli/`（此前 BF16 wide458ms、short144ms，
FP32 wide1142ms、short233ms，包含局部重编译额度耗尽回退，**不是最终推荐配置**）。

真实 DDP 的加速比随 dtype/shape 变化，不能用固定 support 的 1.68× 直接外推生产。
support 动态、DDP 调度、CPU 供数/争用等均包含在实际执行中，但本轮未逐项
消融归因。基线/优化的 prior 等待差异也**不是 prior 算法加速证据**：编译期
让 worker 先填满队列，且 in_order=False 会改变实际数据顺序；只有十个稳定
测量 step，不足以判断供数长期吞吐。优化前后这轮没有修改 prior 生成算法。

完整启动+训练+保存+退出+重启+续训 wall：BF16 1024×100 **72.0→87.7s**，
FP32 **83.4→108.1s**；128×100 BF16 **65.0→79.5s**，FP32 **63.7→81.2s**。
因此极短训练在本轮编译配置下总墙钟**更慢**，需要长训练摊销编译；不能把
steady-state 加速当成冷启动加速。单次前端 cold-use 约 18–39s 随缓存/形状
改变，既有完整 step 样本中的缓存已被其他测点预热，不能声称是清缓存的
纯编译成本。本轮没有删除用户编译缓存。

这里将 max_steps16 改18做续训，是流程检查，不是同 horizon 确定性重放；
相同 horizon 的下一次参数更新由单独 optimizer resume 测试覆盖。
真实多 GPU/multi-node NCCL 性能和长期训练质量未验证。

## 已有 50K checkpoint 的本轮增量回归

同一只读 checkpoint、48 张保存表、support1024/4096/16384 各16张、
5/20/50列、每张128 query，每种 dtype 共6144预测；严格加载及整模型反向均通过。
对照是 **d0cdd57 → 本轮**，不是旧 ICL 原版 → 本轮。使用 training path +
no_grad 对比前端优化所影响的路径，不等同 sklearn eval/cache API 的全量回归。

| 前端编译 | dtype | 基线准确率 | 本轮准确率 | 基线 CE | 本轮 CE | 预测翻转率 |
|---|---|---:|---:|---:|---:|---:|
| none | FP32 | 79.2969% | 79.3783% | 0.505123 | 0.503479 | 1.4648% |
| none | BF16 | 79.5247% | 79.4759% | 0.506226 | 0.506617 | 1.4648% |
| all | FP32 | 79.2969% | 79.0690% | 0.505123 | 0.503961 | 2.2298% |
| all | BF16 | 79.5247% | 79.2806% | 0.506226 | 0.504266 | 3.3529% |

上表 all 为最终 SSMax 修复后的重跑；none 为未受编译分支影响的 eager 记录。
有限集上 FP32/BF16 all 的准确率变化分别为 **−0.2279 / −0.2441 个百分点**，CE 下降。
按表配对 bootstrap 20000 次、seed42，准确率增量 95% 区间（百分点）：
none FP32 [−0.4069,+0.5697]、BF16 [−0.4232,+0.3092]；
all FP32 [−0.5371,+0.0814]、BF16 [−0.8301,+0.2445]，均跨0。
这不是非劣性证明，也不足以证明系统降质。none FP32 的 CE 增量区间
[−0.003580,−0.000229] 不跨0，其余 CE 区间跨0，不能概括为“所有指标无变化”。
all FP32/BF16 的 CE 增量95%区间分别为 [−0.002885,+0.000473] /
[−0.004694,+0.000493]。
all BF16 在 16K support 的翻转率达6.20%，单表最大32.81%；none BF16
单表最大31.25%，绝不能用总体准确率接近掩盖个别任务的明显变化。

原因边界：权重格式兼容不保证算术顺序兼容；标签广播改变反向归约，跳过
CLS 占位改变 GEMM batch shape，编译融合改变前端舍入，FP32 也不豁免。
稀疏 leaf/排序/top-k 在边界上不连续，会逐层放大小扰动。此前固定旧 ICL
路由实验已验证该机制；本轮独立前端/固定路由消融如下。
不自动改回旧 kernels，不直接用更宽 allclose 容差代替 checkpoint 质量审计。

原始 logits/记录：`../perf_results/torch28_checkpoint_none/` 与
`../perf_results/torch28_checkpoint_final_all/`。此前修复前的 all 数据保留在
`../perf_results/torch28_checkpoint_all/`，不可当作最终值。

### 前端变化与固定旧路由消融

只读加载同一权重，新旧 ICL 实现均为 d0cdd57 的算术路径；区别仅为本轮前端。
选5列、case3/6/12/15、support1024/2048/4096/8192/16384，共20表/精度。
这是有意关注窄表和已知漂移表的诊断子集，不能拿它估计总体质量。
原始模型重复执行的 logits、概率与预测完全相同，排除本次对照的重复执行噪声。
TF32 在此与性能测量中均开启，因此 FP32 指脚本 float32 模式，并非强制 IEEE GEMM。

| dtype | 前端 | 默认路由翻转率 | 固定旧路由翻转率 | 默认概率RMSE均值 | 固定后概率RMSE均值 |
|---|---|---:|---:|---:|---:|
| FP32 | eager | 3.7891% | 0.0391% | 0.009889 | 0.000209 |
| FP32 | compiled | 3.5156% | 0.0781% | 0.009571 | 0.000197 |
| BF16 | eager | 4.1016% | 0.9766% | 0.012584 | 0.003951 |
| BF16 | compiled | 4.1016% | 1.3281% | 0.014137 | 0.006084 |

前端表征相对L2误差均值，FP32 eager/compiled 为0.0832%/0.0854%，
BF16为2.3324%/3.0529%，BF16 compiled 单表最高7.7887%，不能笼统称为
所有表征都只有机器epsilon量级差别。固定路由大幅降低翻转，支持离散路由
放大算术变化的判断；BF16固定后仍有约1%翻转，说明前端本身的低精度差异
也有实质影响，不能宣称“全部由路由造成”或“只要固定路由就完全等价”。

最早检索层（block2）到最后检索层（block11）的 token所属leaf变化均值：
FP32 compiled约4.19%→55.52%，BF16约40.40%→70.12%；抽样候选token保留率
分别92.26%→59.15%、71.10%→53.66%。leaf变化不等于最终候选集合变化，
因此同时报告候选token重合，不能仅比较block编号。该固定路由实验是因果定位，
不是可用于实际新输入的部署方案。原始记录为
`../perf_results/torch28_frontend_route_ablation.json`。

## 长表完整训练步

E128、3+3+12、全部参数训练、batch=micro=1、recompute=True、
compile_frontend=all、compile_optimizer=True，2 warmup + 3 repeat；
包括 H2D、FWD/BWD、clip、optimizer、metrics，不含 prior/DDP。

| dtype | 行×列 | step 中位数 ms | peak allocated GiB |
|---|---:|---:|---:|
| FP32 | 4096×100 | 905.10 | 4.702 |
| BF16 | 4096×100 | 384.95 | 3.292 |
| FP32 | 60000×10 | 3107.90 | 14.085 |
| BF16 | 60000×10 | 1456.78 | 11.762 |

全部 step 指标有限、无 OOM。这里是长表可运行/稳定性测试，没有长表整网
基线对照，不能用它报告加速比，也不代表60K×100能在相同显存下运行。
最终脚本输出：`../perf_results/torch28_final_long_step_*.json`。

## 下一步边界与优先级

1. Stage1 应先启用前端编译并实测长时间供数，不只优化 ICL；Muon compile
   收益很小，可先关闭。推荐先在任务验证集比较 checkpoint，再启动长训练。
2. 保持真实 prior 的过滤/分布，不为了速度改变训练数据；根据 prior_time 和
   CPU/RAM 调整 worker/prefetch，不能从单rank4workers推广到16卡128workers。
3. variable columns/support 会触发额外 graph guards；用随机宽度真实训练补测，
   仍不能保证未来所有配置零重编译。分类 mixed-radix 的 max().item() 是已知
   graph break，不为了消除警告删掉功能。
4. kernel 加速未消除 centroid 路由二次项。DSA 是训练/架构新实验，不能当作
   checkpoint 等价的纯执行优化混入本轮提交。
