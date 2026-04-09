# examples/Pretrain — 大规模预训练

使用 600 万+ 开源轨迹进行 ABot-M0 基础能力预训练，生成可迁移到各下游 Benchmark 的通用权重。

## 关键文件

| 文件 | 说明 |
|------|------|
| `examples/Pretrain/ABot_pretrain.yaml` | 预训练 YAML 配置（数据混合、模型、训练超参） |
| `examples/Pretrain/run_pretrain.sh` | 多卡预训练启动脚本（A100 × 8 推荐） |
| `examples/Pretrain/compute_delta_state_stats.py` | 计算各数据集的 delta state 归一化统计量 |
| `examples/Pretrain/run_compute_delta_state_stats.sh` | 批量计算统计量的 shell 封装 |

## 数据准备

支持的数据来源（按 `data_mix: pretrain` 配置）：

- OXE（Open X-Embodiment）：[IPEC-COMMUNITY](https://huggingface.co/collections/IPEC-COMMUNITY/openx-lerobot)
- RoboCoin：[ModelScope](https://modelscope.cn/organization/RoboCOIN?tab=dataset)
- RoboMind：[HuggingFace](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND)
- AgibotWorld-Beta：需通过 `data_process/any4lerobot/agibot2lerobot/` 转换

下载预处理好的 `/meta` 文件（含 `modality.json`、`stats_delta_state.json`）：
```bash
# 从 ModelScope 下载
# https://www.modelscope.cn/datasets/amap_cvlab/Abot-M0-MetaData
```

## 训练流程

```bash
# Step 1：验证 dataloader 可正常加载
python ABot/dataloader/lerobot_datasets.py \
  --config_yaml examples/Pretrain/ABot_pretrain.yaml

# Step 2：验证模型前向可运行
python ABot/model/framework/ABot_M0.py \
  --config_yaml examples/Pretrain/ABot_pretrain.yaml

# Step 3：启动预训练（推荐 A100 × 8）
bash examples/Pretrain/run_pretrain.sh
```

## 关键配置

```yaml
datasets:
  vla_data:
    data_mix: pretrain          # 使用预训练数据混合
    data_root_dir: /data/pretrain
trainer:
  max_train_steps: 100000
  repeated_diffusion_steps: 4   # 增大可提高 AML 稳定性
```

## 扩展指引

- 下游微调：将 `pretrained_checkpoint` 指向此处输出的权重，切换 `data_mix` 为对应 Benchmark
- 自定义数据：在 `mixtures.py` 添加新 mix，在 YAML 中使用对应键名
