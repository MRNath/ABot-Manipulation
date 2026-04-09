# data_process/lerobot — LeRobot 官方库

`data_process/lerobot/` 是 [HuggingFace LeRobot](https://github.com/huggingface/lerobot) 的本地镜像（git submodule），提供 LeRobot 格式数据集的底层实现。

## 用途

- 提供 `LeRobotDataset` 基础类（`v2.x` / `v3.0`）
- 提供数据集可视化工具（`vis.sh`、`batch_extract.sh`）
- 为 `any4lerobot` 各转换脚本提供 LeRobot API 依赖

## 安装（独立环境）

```bash
cd data_process/lerobot
pip install -e ".[feetech]"
# 或按 README.md 中的安装说明
```

> ⚠️ **注意**：ABot 训练环境（`ABot` conda）已通过 `requirements.txt` 安装了所需 LeRobot 依赖，
> 数据转换操作建议在单独的 conda 环境中进行。

## 数据集可视化

```bash
# 在 lerobot 环境中运行
cd data_process/lerobot
bash vis.sh   # 可视化一个 LeRobot 数据集的 episode
```

## 关键目录

| 路径 | 说明 |
|------|------|
| `src/lerobot/` | LeRobot 核心源码 |
| `examples/` | 官方使用示例 |
| `docs/` | LeRobot 官方文档 |

## 扩展指引

- 更新 submodule：`git submodule update --remote data_process/lerobot`
- 与 `any4lerobot` 配合使用时，确保两者使用兼容的 LeRobot 版本（当前 v3.0）
