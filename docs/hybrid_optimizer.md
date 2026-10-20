# Hybrid Muon + AdamW 训练与恢复

`--muon True` 现在表示 hybrid，原 all-parameter Muon 行为已经替换。
`--muon False` 仍表示全 AdamW，学习率由 `--lr` 控制。
本次不改变 ICL 架构，先独立验证优化器是否改善稳定性。

## 参数分组

| 组 | 参数 | peak LR | weight decay |
| --- | --- | --- | --- |
| `muon` | 所有 `MultiheadAttentionBlock` 内的 QKV、attention out projection、FFN linear1/linear2 weight | `--lr` | `--weight_decay` |
| `adamw_decay` | 其余二维及以上参数，包括输入/标签映射、输出 head、slots、gate、SSMax 网络 | `--adamw_lr` | `--weight_decay` |
| `adamw_no_decay` | 所有其余一维/标量参数，包括 norm、bias、QK scale 等 | `--adamw_lr` | 0 |

根据模块归属选择 Muon 参数，而不是仅检查张量维数。冻结的参数不进入优化器，
每个可训练参数恰好出现一次。AdamW LR 未显式指定时默认为 Muon LR 的 1/4；
分类 Stage 1 脚本和 screen 脚本显式默认为 AdamW LR `1e-4`。
所有参数组按同一个 scheduler factor 缩放，因此两种 LR 的比例保持不变。
`cosine_lr_end` 对应基准 `--lr` 的终值，AdamW 的终值按比例缩放。

日志目录中的 `optimizer_groups.json` 保存每组完整参数名、形状、初始 LR、
weight decay 和参数量；每步 `metrics.jsonl` 包含
`lr/muon`、`lr/adamw_decay`、`lr/adamw_no_decay`。
新 checkpoint 保存 optimizer routing spec，完整恢复时校验分组身份及顺序。

## 训练服务器：从 35k checkpoint 启动

在训练服务器的 `tabicl` 仓库目录执行：

```bash
conda activate stdplm
DRY_RUN=1 bash scripts/train_v2_clf_stage1_hybrid_from35k.sh
bash scripts/train_v2_clf_stage1_hybrid_from35k.sh
```

该脚本继承 `train_v2_clf_stage1.sh` 的机器配置：默认 16 GPU、全局 batch 64、
microbatch 4，dtype、ICL 架构、数据和日志配置也沿用原脚本。初始权重为
`/ossfs/workspace/tabicl/checkpoints/tabiclv2-clf/stage1/step-35000.ckpt`，
输出到独立的 `stage1_hybrid_from35k` 目录。Muon LR 为 `4e-4`，AdamW LR
为 `1e-4`，重新 warmup 500 步，再训练 20k 步；新目录的 `step-20000.ckpt`
对应旧 35k 权重累计再训练至约 55k。中断后重复同一启动命令，完整恢复新目录
的 hybrid 状态。首次正式启动会检查初始 checkpoint 是否存在。

可通过环境变量覆盖，例如 `MAX_STEPS=40000 bash scripts/train_v2_clf_stage1_hybrid_from35k.sh`；
同一实验后续恢复时应保持首次启动的训练配置。

## 从旧 checkpoint 开始稳定性实验

优先选择失稳前的 40k 权重；当前本地目录只提供 20k、60k、100k，
因此下面本地命令使用 20k。不要用已明显失稳的 100k 作为主实验起点。

```bash
conda activate stdplm
cd /home/zj/table_foundation/sparse/tabicl

export NUM_GPUS=1
export INIT_CKPT="$PWD/checkpoints/tabiclv2-clf/stage1/step-20000.ckpt"
export CKPT_DIR="$PWD/checkpoints/tabiclv2-clf/stage1_hybrid_from20k"
export LOG_DIR="$CKPT_DIR/logs"
export LR=4e-4
export ADAMW_LR=1e-4
export MAX_STEPS=20000
export WARMUP_PROPORTION=-1
export WARMUP_STEPS=500
export BATCH_SIZE=64
export MICRO_BATCH_SIZE=4
export TRAIN_DTYPE=bfloat16
export RECOMPUTE=True
export WANDB_LOG=False

# 只输出并检查展开后的命令；不会启动训练。
DRY_RUN=1 bash scripts/train_v2_clf_stage1.sh

# 确认目录与超参数后启动。
bash scripts/train_v2_clf_stage1.sh
```

上面的 `MAX_STEPS=20000` 表示新阶段运行 20k 步。新日志 local step 从 0 开始；
例如新实验 `step-5000.ckpt` 对应旧 20k 权重再训练 5k。
模型权重继承，optimizer moments 和 scheduler 不继承。这不是原优化器状态的精确续跑，
也不保证重放原 prior 数据流。网络所有未冻结参数继续训练。

16 GPU 服务器上的 40k 恢复示例（在服务器的 tabicl 仓库目录执行）：

```bash
NUM_GPUS=16 \
INIT_CKPT=/ossfs/workspace/tabicl/checkpoints/tabiclv2-clf/stage1/step-40000.ckpt \
CKPT_DIR=/ossfs/workspace/tabicl/checkpoints/tabiclv2-clf/stage1_hybrid_from40k \
LOG_DIR=/ossfs/workspace/tabicl/checkpoints/tabiclv2-clf/stage1_hybrid_from40k/logs \
LR=4e-4 ADAMW_LR=1e-4 \
MAX_STEPS=20000 WARMUP_PROPORTION=-1 WARMUP_STEPS=500 \
BATCH_SIZE=64 MICRO_BATCH_SIZE=4 \
TRAIN_DTYPE=bfloat16 RECOMPUTE=True WANDB_LOG=False \
bash scripts/train_v2_clf_stage1.sh
```

全局 batch 64：16 GPU 时每卡 batch 4；1 GPU 时通过多个 microbatch 累积到 64。
`NUM_GPUS` 应为实际可见 GPU 数。BF16 不要求把模型参数转换为 BF16：
保留 FP32 参数与优化器状态，通过 autocast 使用 BF16 计算。
输入 checkpoint 显式指定但不存在时会报错，不会静默从头训练。

## 中断后恢复新 hybrid 实验

重复完全相同的启动命令即可。Stage 1 脚本优先检查目标 `CKPT_DIR`：

- 目标没有 checkpoint 且 `INIT_CKPT` 非空：只加载旧模型权重。
- 目标已有 checkpoint：忽略 `INIT_CKPT`，加载目标最新 checkpoint 的模型、
  optimizer、scheduler 和 local step，继续跑到该实验的 `MAX_STEPS`。
- 目标没有 checkpoint 且未设置 `INIT_CKPT`：从随机初始化训练。

完整恢复应保持 LR、warmup、MAX_STEPS、架构和冻结配置等 recipe 相同。
不要将旧 all-Muon 目录直接作为新 `CKPT_DIR`；不兼容的 optimizer state 会被拒绝。
独立重复实验也必须使用另一个空目录。

直接调用 Python trainer 时，对应语义是：

```bash
# 在原有完整模型/prior参数基础上添加这些参数：
# --muon True --lr 4e-4 --adamw_lr 1e-4
# --checkpoint_path /path/to/old/step-40000.ckpt --only_load_model True
# --checkpoint_dir /path/to/new-hybrid --log_dir /path/to/new-hybrid/logs
# --max_steps 20000 --warmup_proportion -1 --warmup_steps 500
```

## 从头正式训练 Stage 1

稳定性实验通过后，在空目录重新训练。下面沿用较保守的 Muon LR `4e-4`，
该值是实验起点，不代表已经证明优于原 `8e-4`；要比较两者请分开使用目录。

```bash
NUM_GPUS=16 INIT_CKPT= \
CKPT_DIR=/ossfs/workspace/tabicl/checkpoints/tabiclv2-clf/stage1_hybrid_fresh \
LOG_DIR=/ossfs/workspace/tabicl/checkpoints/tabiclv2-clf/stage1_hybrid_fresh/logs \
LR=4e-4 ADAMW_LR=1e-4 \
MAX_STEPS=500000 WARMUP_PROPORTION=-1 WARMUP_STEPS=5000 \
BATCH_SIZE=64 MICRO_BATCH_SIZE=4 \
TRAIN_DTYPE=bfloat16 RECOMPUTE=True WANDB_LOG=False \
bash scripts/train_v2_clf_stage1.sh
```

其他 v2 launcher 仍传入 `--muon True`，所以也会采用 hybrid；未指定
`--adamw_lr` 时自动使用该 stage 的 `--lr / 4`。

## 架构筛选命令

screen 也使用新 hybrid 分组。新增两个显式学习率选项，不改变架构集合或初始化规则。
必须使用新输出目录，避免自动恢复旧 all-Muon 试验。

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/screen_icl_attention.py plan \
  --output checkpoints/screen_hybrid_routing \
  --preset routing --steps 20000 --num-gpus 1 \
  --batch-size 16 --micro-batch-size 1 \
  --lr 4e-4 --adamw-lr 1e-4 \
  --variants dense global_elu global_sparse_elu global_sparse_alternating \
  --seeds 42 43 44 --memory-limit-gib 0
```

将 `plan` 改成 `run` 才会生成固定验证集、公共初始化并启动训练。
该 screen 从公共随机初始化训练；它不是从 Stage 1 checkpoint 进行恢复。
`sparse_all` 尚未新增，本次只实现 optimizer 改动。

## 验证

```bash
OMP_NUM_THREADS=4 PYTHONPATH=src python -m pytest -q \
  tests/test_hybrid_optimizer.py \
  tests/test_icl_training_diagnostics.py \
  tests/test_icl_screening.py
```

测试覆盖真实模型分组（dense / ELU / induced）、冻结参数排除、
AdamW 小梯度与缺失梯度更新对齐、旧优化器状态拒绝与权重加载、
CPU/CUDA-BF16 前向反向更新及 hybrid checkpoint 恢复、launcher 首次加载与重复恢复。
这些测试不证明长时间训练一定不再失稳，需要上述稳定性实验验证。
