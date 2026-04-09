# examples/Robotwin — RoboTwin 2.0 Benchmark

在 RoboTwin 2.0（50+ 任务，Clean + Randomized 双模式）上训练并评测，取得 **86.1%** 成功率。

## 关键文件

| 文件 | 说明 |
|------|------|
| `examples/Robotwin/train_files/run_robotwin_train.sh` | 训练启动脚本（总 batch 48×4） |
| `examples/Robotwin/eval_files/run_policy_server.sh` | 推理服务启动脚本（ABot 环境） |
| `examples/Robotwin/eval_files/eval.sh` | 单任务仿真评测脚本（robotwin 环境） |
| `examples/Robotwin/eval_files/deploy_policy.yml` | 部署配置（含权重路径） |

## 数据准备（训练）

```bash
# 从 HuggingFace 下载 RoboTwin 2.0 数据集
# https://huggingface.co/datasets/StarVLA/RoboTwin-Randomized
# 下载后按 train_files/README 配置数据路径
```

## 训练

```bash
bash examples/Robotwin/train_files/run_robotwin_train.sh
```

## 评测（两终端）

```bash
# Terminal 1 — ABot 环境
# 先编辑 deploy_policy.yml 和 run_policy_server.sh 中的权重路径
bash examples/Robotwin/eval_files/run_policy_server.sh

# Terminal 2 — robotwin 环境
pip install -r examples/Robotwin/eval_files/requirements.txt
conda activate robotwin
cd examples/Robotwin/eval_files
bash eval.sh <task_name> <mode> <exp_name> 0 0
# 示例：
bash eval.sh pick_dual_bottles demo_clean my_test_v1 0 0
```

## 并行评测（推荐）

```bash
# 对全部 50+ 任务并行评测，缩短整体时间
bash examples/Robotwin/eval_files/parallel_eval/eval_notebook.sh
# 根据可用 GPU 数量修改并行度
```

## 环境依赖

- 仿真环境：[RoboTwin 官方安装指引](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)
- 已验证 GPU：NVIDIA RTX 4090

## 可用任务模式

| 模式 | 说明 |
|------|------|
| `demo_clean` | 标准干净场景 |
| `demo_randomized` | 随机化场景（更具挑战性） |
