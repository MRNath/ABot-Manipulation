# examples — Benchmark 训练与评测

每个子目录对应一个 Benchmark，包含独立的训练配置、启动脚本和评测流程。

## Benchmark 索引

| Benchmark | 模型权重 | 文档 |
|-----------|----------|------|
| [Pretrain](Pretrain/index.md) | [ModelScope ABot-M0-Pretrain](https://www.modelscope.cn/models/amap_cvlab/ABot-M0-Pretrain) | → Pretrain/index.md |
| [LIBERO](LIBERO/index.md) | [HuggingFace ABot-M0-LIBERO](https://huggingface.co/acvlab/ABot-M0-LIBERO) | → LIBERO/index.md |
| [LIBERO-plus](LIBERO-plus/index.md) | 同 LIBERO 权重（零样本泛化） | → LIBERO-plus/index.md |
| [Alicia](Alicia/index.md) | 使用 Alicia LeRobot v3 数据进行微调 | → Alicia/index.md |
| [Robotwin](Robotwin/index.md) | [HuggingFace ABot-M0-RoboTwin2](https://huggingface.co/acvlab/ABot-M0-RoboTwin2) | → Robotwin/index.md |
| [Robocasa_tabletop](Robocasa_tabletop/index.md) | [HuggingFace ABot-M0-Robocasa](https://huggingface.co/acvlab/ABot-M0-Robocasa) | → Robocasa_tabletop/index.md |

## 通用工作流

所有 Benchmark 均遵循以下两阶段评测流程：

```
Terminal 1（ABot 环境）      Terminal 2（仿真环境）
bash run_policy_server.sh  →  bash eval_*.sh
     模型推理服务                   仿真执行 + 指标统计
```

## 性能结果

| Benchmark | 指标 | ABot-M0 |
|-----------|------|---------|
| LIBERO | 成功率 | **98.6%** |
| LIBERO-Plus | 成功率（零样本） | **80.5%** |
| RoboCasa-GR1-Tabletop | 成功率 | **58.3%** |
| RoboTwin 2.0 | 成功率 | **86.1%** |

## 详细 Benchmark 文档

→ [benchmarks.md](benchmarks.md)（跨 Benchmark 对比与通用注意事项）
