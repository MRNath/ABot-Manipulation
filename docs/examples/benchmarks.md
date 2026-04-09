# Benchmark 对比与通用注意事项

跨 Benchmark 的共性设计说明、环境要求和常见问题汇总。

## 评测架构（通用）

全部 Benchmark 采用 **Client-Server** 分离架构：

```
┌─────────────────────────┐     WebSocket     ┌──────────────────────────┐
│   ABot 环境（Terminal 1）│ ←─── 图像+指令 ───→│  仿真环境（Terminal 2）   │
│  server_policy.py 推理  │ ──── 归一化动作 ──→ │  仿真器执行 + 指标统计    │
└─────────────────────────┘                    └──────────────────────────┘
```

## GPU 要求

| Benchmark | 已验证 GPU | 显存需求（推理） |
|-----------|-----------|----------------|
| LIBERO | A100 / RTX 4090 | ~24 GB |
| RoboTwin 2.0 | RTX 4090 | ~24 GB |
| RoboCasa | A100 | ~24 GB |
| Pretrain | A100 × 8 | ~80 GB × 8 |

## 动作反归一化

推理时仿真环境需使用 `stats_delta_state.json` 反归一化：

```bash
# 若未提供，可自行计算
cd examples/Pretrain
bash run_compute_delta_state_stats.sh
```

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| 端口冲突 | `server_policy.py --port 10094` 指定其他端口 |
| 权重路径错误 | 在各评测 shell 脚本中确认 `ckpt_path` 与 `base_vlm` |
| 视频解码失败 | 将 `video_backend` 从 `decord` 切换为 `torchvision_av` |
| CUDA OOM | 启用 `--use_bf16` 或设置 `use_vggt: false` |

## Checkpoint 路径配置

每个 Benchmark 的推理脚本中通常需要配置两处路径：

```yaml
# config.yaml 或 deploy_policy.yml 中
framework:
  qwenvl:
    base_vlm: /path/to/base-qwen3vl-weights   # VLM 基础权重
trainer:
  pretrained_checkpoint: /path/to/finetuned-ckpt  # 微调权重
```

```bash
# run_policy_server.sh 中
--ckpt_path /path/to/finetuned-ckpt
```
