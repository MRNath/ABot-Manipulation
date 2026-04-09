# ABot/training — 训练流程

提供分布式训练入口、`VLATrainer` 训练器与辅助工具集，基于 Accelerate + DeepSpeed。

## 子模块索引

| 子模块 | 职责 | 说明 |
|--------|------|------|
| [trainer_utils](trainer_utils/index.md) | Trainer 工具集：配置追踪、参数分组、日志 | → trainer_utils/index.md |
| `train.py` | 训练主入口，包含 `VLATrainer` 类与 `main()` 函数 | 直接运行此文件 |

## 启动训练

```bash
# 预训练（示例）
bash examples/Pretrain/run_pretrain.sh

# 等价的手动命令
accelerate launch \
  --config_file ABot/config/deepseeds/zero2.yaml \
  ABot/training/train.py \
  --config_yaml examples/Pretrain/ABot_pretrain.yaml \
  trainer.learning_rate.base=1e-4 trainer.max_train_steps=100000
```

## VLATrainer 生命周期

```
prepare_training()
  ├── _init_checkpointing()   # 加载预训练权重或 resume
  ├── freeze_backbones()      # 按 YAML 冻结模块
  └── setup_distributed_training()  # Accelerate wrap
train()
  ├── _train_step()           # forward → backward → step
  ├── eval_action_model()     # 每 eval_interval 步评估 MSE
  ├── _log_metrics()          # W&B 日志
  └── _save_checkpoint()      # 每 save_interval 步保存
_finalize_training()          # 保存 final_model
```

## 关键训练配置

```yaml
trainer:
  max_train_steps: 100000
  learning_rate:
    base: 1e-4
  gradient_accumulation_steps: 4
  gradient_clipping: 1.0
  save_interval: 2000
  eval_interval: 500
  logging_frequency: 10
  freeze_modules: [qwen_vl_interface]   # 冻结 VLM 骨干
  pretrained_checkpoint: /path/to/ckpt  # 加载预训练权重
  is_resume: false                       # true = 从最新 checkpoint 续训
  save_format: pt                        # pt 或 safetensors
```

## 扩展指引

- **多任务损失**：在 `_train_step` 中累加额外损失项
- **自定义评估**：重写 `eval_action_model`，替换 MSE 为任务专属指标
- **新调度器**：修改 `setup_optimizer_and_scheduler` 中的 `lr_scheduler_type`

## 论文实现对齐 (Paper Consistency)

- **两阶段训练机制 (Two-Stage Pretrain/SFT)**：通过 `VLATrainer` 系统配合灵活的配置文件 `yaml`，支持加载 `pretrained_checkpoint` 并在不同层级实现冻结（`freeze_modules`）。完美支持论文里第一阶段混合知识预训练、第二阶段小数据集空间感知对齐（SFT）的两阶段解耦需求。
- **连续性建模优势反馈**：不同于在 `train.py` 头中残存的早期 `Fast Token` (可能作为备选 baseline 或旧分支遗留) 定义，真正的动作推理解算使用完全端到端的 MSE 流匹配方案（体现在前/后向传播中均是对实数向量执行损失）。

## 模块对外接口 (Public Interfaces)

`training` 作为项目执行管道中枢，向用户与系统级调用提供以下控制点：

- **自动化脚本调用入口**:
  - `bash examples/Pretrain/run_pretrain.sh` 或者 `accelerate launch ABot/training/train.py`:
  面向研究人员最主要的调度屏障，所有的 DeepSpeed 与多卡初始化配置通过包裹在该顶层命令下透明执行，研究者无需手动在代码中管理多机器或参数分区设置。
- **编程与拓展级封装接口**:
  - 核心执行器 `VLATrainer`:
    - 组装：`trainer = VLATrainer(cfg, model, dataloader, optimizer, lr_scheduler, accelerator)`
    - 生命周期运转：调用 `trainer.prepare_training()` 准备环境及多卡分片，随之调用 `trainer.train()` 执行含带 checkpoint 保存、评估验证在内的完整大循环。该类提供了高内聚的扩展重写点（如 `_train_step`，`eval_action_model`）。
