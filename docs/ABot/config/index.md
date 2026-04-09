# ABot/config — 训练配置

管理 DeepSpeed 分布式策略与训练超参，按阶段（预训练 / 微调）组织。

## 目录结构

```
ABot/config/
└── deepseeds/
    ├── deepspeed_zero2.yaml   # Zero-2 简化配置
    ├── deepspeed_zero3.yaml   # Zero-3 简化配置
    ├── ds_config.yaml         # 基础 DeepSpeed 配置
    ├── zero2.yaml             # Zero-2 完整配置（含 fp16/bf16）
    └── zero3.yaml             # Zero-3 完整配置（含参数分片）
```

## 功能说明

- **Zero-2**：梯度与优化器状态分片，适合显存充足场景（A100 80G × 8）
- **Zero-3**：参数也分片，适合超大模型或显存受限场景
- 训练脚本通过 `Accelerate + DeepSpeedPlugin` 自动加载对应配置

## 使用方式

配置文件由训练脚本的 shell 环境变量或 `accelerate` 配置指定：

```bash
# 在 run_pretrain.sh 或 run_libero_train.sh 中通常有如下设置
accelerate launch \
  --config_file ABot/config/deepseeds/zero2.yaml \
  ABot/training/train.py \
  --config_yaml examples/Pretrain/ABot_pretrain.yaml
```

## 扩展指引

- 调整 Zero 等级：将 `zero_optimization.stage` 从 2 改为 3
- 调整混合精度：修改 `fp16` / `bf16` 字段
- 各 Benchmark 的训练脚本位于 `examples/<Benchmark>/train_files/`
