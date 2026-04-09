# ABot 核心包

`ABot/` 是项目的核心 Python 包，包含模型架构、数据加载、训练流程及配置四个子模块。

## 子模块索引

| 子模块 | 职责 | 文档 |
|--------|------|------|
| [config](config/index.md) | DeepSpeed Zero-2/3 及训练超参 YAML 配置 | → config/index.md |
| [model](model/index.md) | ABot_M0 框架、VLM 骨干、AML 动作头、VGGT 3D 感知 | → model/index.md |
| [dataloader](dataloader/index.md) | LeRobot 格式数据集加载、混合采样 | → dataloader/index.md |
| [training](training/index.md) | 训练入口 `train.py`、`VLATrainer`、工具函数 | → training/index.md |

## 包入口

```python
# 构建模型
from ABot.model.framework import build_framework
model = build_framework(cfg)

# 构建数据加载器
from ABot.dataloader import build_dataloader
dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
```

## 安装

```bash
pip install -e .  # 在项目根目录执行
```

> 详见根目录 [CLAUDE.md](../../CLAUDE.md) 完整安装步骤。
