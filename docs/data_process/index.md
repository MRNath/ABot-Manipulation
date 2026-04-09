# data_process — 数据处理工具链

提供将各来源数据集转换为 LeRobot 格式的完整工具链，以及 LeRobot 数据集的合并与版本转换。

## 子模块索引

| 子模块 | 职责 | 说明 |
|--------|------|------|
| [any4lerobot](any4lerobot/index.md) | 多源格式 → LeRobot 转换套件（8 种来源） | → any4lerobot/index.md |
| [lerobot](lerobot/index.md) | LeRobot 官方库（作为子模块引用） | → lerobot/index.md |

## 支持的转换方向

```
agibot2lerobot      AgiBOT 格式       → LeRobot
libero2lerobot      LIBERO 格式       → LeRobot
openx2lerobot       Open-X-Embodiment → LeRobot
rlds2lerobot        RLDS/TFDS 格式    → LeRobot
robomind2lerobot    RoboMind 格式     → LeRobot
lerobot2rlds        LeRobot           → RLDS
ds_version_convert  LeRobot v1 → v2 版本升级
dataset_merging     多 LeRobot 数据集合并
```

## 快速使用（以 any4lerobot 为例）

```bash
# 参考 any4lerobot README
cd data_process/any4lerobot
# 具体命令见各子目录 README.md
python libero2lerobot/convert.py \
  --input_dir /path/to/libero \
  --output_dir /path/to/output
```

## 扩展指引

- 新增数据源：参考 `libero2lerobot/` 目录结构，实现 `convert.py` 并在此 index 中注册
- 数据合并：使用 `dataset_merging/` 工具合并多个 LeRobot 数据集后直接配置到 `data_mix`
