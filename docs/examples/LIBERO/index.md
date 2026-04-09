# examples/LIBERO — LIBERO Benchmark

在 LIBERO（空间/物体/目标/LIBERO-10 四个子集）上训练并评测 ABot-M0，取得 **98.6%** 成功率。

## 关键文件

| 文件 | 说明 |
|------|------|
| `examples/LIBERO/train_files/run_libero_train.sh` | 训练启动脚本（总 batch 8×8） |
| `examples/LIBERO/eval_files/run_policy_server.sh` | 推理服务启动脚本（ABot 环境） |
| `examples/LIBERO/eval_files/eval_libero.sh` | 仿真评测脚本（LIBERO 环境） |

## 数据准备（训练）

```bash
# 下载四个子集（LeRobot 格式）
# LIBERO-spatial: https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
# LIBERO-object:  https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot
# LIBERO-goal:    https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot
# LIBERO-10:      https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot

# 将 modality.json 放置到各子集的 meta/ 目录
cp modality.json $LEROBOT_LIBERO_DATA/subset/meta/modality.json
```

## 训练

```bash
bash examples/LIBERO/train_files/run_libero_train.sh
# 确认脚本中的权重路径和数据路径已正确设置
```

## 评测（两终端）

```bash
# Terminal 1 — ABot 环境
bash examples/LIBERO/eval_files/run_policy_server.sh

# Terminal 2 — LIBERO 环境（需先安装 LIBERO 仿真器）
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
bash examples/LIBERO/eval_files/eval_libero.sh
```

## 环境依赖

- 仿真环境：[LIBERO 官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)
- 推理服务：ABot conda 环境（`requirements.txt`）
- 已验证 GPU：NVIDIA A100、RTX 4090

## 扩展指引

- **LIBERO-Plus 零样本评测**：无需重新训练，直接使用 LIBERO 权重运行 `examples/LIBERO-plus/`
- 调整 action chunk 长度：修改 YAML `framework.action_model.future_action_window_size`
