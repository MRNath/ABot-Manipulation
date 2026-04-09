# ABot-Manipulation

**ABot-M0** — VLA Foundation Model for Robotic Manipulation with Action Manifold Learning.  
Paper: [arXiv:2602.11236](https://arxiv.org/abs/2602.11236) · Weights: [HuggingFace acvlab](https://huggingface.co/acvlab) · Data: [ModelScope](https://www.modelscope.cn/datasets/amap_cvlab/Abot-M0-MetaData)

---

## 项目简介

ABot-M0 是 AMAP CV Lab 开源的通用机器人操控基础模型，核心创新：

- **Action Manifold Learning (AML)**：Flow Matching 头直接预测干净动作，无需去噪迭代
- **模块化 3D 感知**：可插拔 VGGT-1B 空间感知模块，提升精细操控精度
- **海量统一数据**：整合 600 万+ 开源轨迹，覆盖 LIBERO / RoboTwin / RoboCasa 等主流 Benchmark

---

## 快速上手

```bash
# 安装环境
conda create -n ABot python=3.10 -y && conda activate ABot
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e path_to_vggt   # 需先克隆 facebook/vggt
pip install -e .

# 启动预训练
bash examples/Pretrain/run_pretrain.sh

# 启动推理服务（LIBERO 示例）
bash examples/LIBERO/eval_files/run_policy_server.sh
```

---

## 文档导航

| 模块 | 说明 | 文档 |
|------|------|------|
| **ABot** | 核心 Python 包（模型 / 数据 / 训练） | [docs/ABot/index.md](docs/ABot/index.md) |
| ├─ config | DeepSpeed / 训练超参配置 | [docs/ABot/config/index.md](docs/ABot/config/index.md) |
| ├─ model | 模型架构（框架 + 子模块） | [docs/ABot/model/index.md](docs/ABot/model/index.md) |
| ├─ dataloader | LeRobot 格式数据加载 | [docs/ABot/dataloader/index.md](docs/ABot/dataloader/index.md) |
| └─ training | 训练入口与 Trainer 工具 | [docs/ABot/training/index.md](docs/ABot/training/index.md) |
| **deployment** | WebSocket 推理服务器 | [docs/deployment/index.md](docs/deployment/index.md) |
| **data_process** | 数据格式转换工具链 | [docs/data_process/index.md](docs/data_process/index.md) |
| **examples** | 各 Benchmark 训练/评测脚本 | [docs/examples/index.md](docs/examples/index.md) |

---

## 项目结构概览

```
ABot-Manipulation/
├── ABot/                   # 核心包
│   ├── config/             # DeepSpeed 配置 YAML
│   ├── model/              # 模型框架与子模块
│   ├── dataloader/         # LeRobot 数据加载
│   └── training/           # 训练器
├── deployment/             # 推理服务端
├── data_process/           # 数据处理工具
├── examples/               # Benchmark 示例
│   ├── Pretrain/
│   ├── LIBERO/
│   ├── Robotwin/
│   ├── Robocasa_tabletop/
│   └── LIBERO-plus/
├── requirements.txt
└── pyproject.toml
```

---

## Model Zoo

| 模型 | 链接 | 说明 |
|------|------|------|
| ABot-Pretrain | [ModelScope](https://www.modelscope.cn/models/amap_cvlab/ABot-M0-Pretrain) | 预训练基础权重 |
| ABot-LIBERO | [HuggingFace](https://huggingface.co/acvlab/ABot-M0-LIBERO) | LIBERO 微调 |
| ABot-RoboCasa | [HuggingFace](https://huggingface.co/acvlab/ABot-M0-Robocasa) | RoboCasa-GR1 微调 |
| ABot-RoboTwin2 | [HuggingFace](https://huggingface.co/acvlab/ABot-M0-RoboTwin2) | RoboTwin2.0 微调 |

---

## 环境变量

| 变量 | 说明 |
|------|------|
| `CHECKPOINT_BASEDIR` | 模型权重根目录（可选，默认 None） |
| `WANDB_PROJECT` | WandB 项目名 |
| `DEBUG` | 设为 `1` 启用 debugpy 远程调试 |
