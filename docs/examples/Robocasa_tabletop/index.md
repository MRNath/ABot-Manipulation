# examples/Robocasa_tabletop — RoboCasa GR1 Tabletop Benchmark

在 RoboCasa-GR1-Tabletop 数据集上训练并评测 ABot-M0，取得 **58.3%** 成功率。

## 关键文件

| 文件 | 说明 |
|------|------|
| `examples/Robocasa_tabletop/train_files/` | 训练配置与启动脚本 |
| `examples/Robocasa_tabletop/eval_files/` | 仿真评测脚本 |

## 模型权重

```
HuggingFace: https://huggingface.co/acvlab/ABot-M0-Robocasa
```

## 训练

```bash
bash examples/Robocasa_tabletop/train_files/run_robocasa_train.sh
# 确认脚本中的数据路径和权重路径已正确设置
```

## 评测（两终端）

```bash
# Terminal 1 — ABot 环境（推理服务）
bash examples/Robocasa_tabletop/eval_files/run_policy_server.sh

# Terminal 2 — RoboCasa 仿真环境
bash examples/Robocasa_tabletop/eval_files/eval_robocasa.sh
```

## 环境依赖

- 仿真环境：RoboCasa + GR1 机器人仿真配置
- 已验证 GPU：NVIDIA A100

## 扩展指引

- 评测其他 RoboCasa 场景：修改评测脚本的场景配置参数
- 与 Pretrain 权重对比：`pretrained_checkpoint` 改为预训练权重路径，对比迁移效果
