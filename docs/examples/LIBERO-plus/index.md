# examples/LIBERO-plus — LIBERO-Plus 零样本泛化评测

使用 **LIBERO 权重**直接在 LIBERO-Plus 场景上进行零样本推理，取得 **80.5%** 成功率，验证 ABot-M0 的跨任务泛化能力。

## 关键说明

> **无需重新训练**：LIBERO-Plus 评测复用 LIBERO 微调权重（`ABot-M0-LIBERO`），不额外训练。

## 模型权重

```
HuggingFace: https://huggingface.co/acvlab/ABot-M0-LIBERO
```

## 评测流程

```bash
# Terminal 1 — ABot 环境（与 LIBERO 评测共用同一服务）
bash examples/LIBERO/eval_files/run_policy_server.sh

# Terminal 2 — LIBERO-Plus 仿真环境
bash examples/LIBERO-plus/eval_files/eval_libero_plus.sh
```

## 与 LIBERO 的区别

| 对比项 | LIBERO | LIBERO-Plus |
|--------|--------|-------------|
| 训练数据 | LIBERO 四子集 | 不训练 |
| 场景 | 标准 LIBERO 任务 | 泛化场景（更复杂） |
| 权重 | ABot-M0-LIBERO | ABot-M0-LIBERO（复用） |
| 指标 | 98.6% | 80.5% |

## 环境依赖

- 仿真环境：同 LIBERO（`examples/LIBERO/README.md`）
- 已验证 GPU：NVIDIA A100、RTX 4090

## 扩展指引

- 提升泛化性：在 LIBERO + LIBERO-Plus 混合数据上联合微调
- 自定义泛化场景：参考 LIBERO 数据格式创建新场景并注册到评测脚本
