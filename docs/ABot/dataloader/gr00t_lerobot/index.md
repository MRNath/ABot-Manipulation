# dataloader/gr00t_lerobot — LeRobot 数据集实现

`gr00t_lerobot` 是改自 Isaac-GR00T 的 LeRobot 格式数据集实现，提供单数据集与混合数据集两种加载模式。

## 关键文件

| 文件 | 说明 |
|------|------|
| `datasets.py` | `LeRobotSingleDataset`、`LeRobotMixtureDataset`、`ValidationLeRobotMixtureDataset` |
| `mixtures.py` | `DATASET_NAMED_MIXTURES` 字典，定义各 Benchmark 的数据组合与权重 |
| `data_config.py` | `ROBOT_TYPE_CONFIG_MAP`，每种机器人的模态配置与图像变换 |
| `embodiment_tags.py` | `EmbodimentTag` 枚举，标识机器人本体类型 |

## mixture 格式

```python
# mixtures.py 中的条目格式
DATASET_NAMED_MIXTURES["libero_mix"] = [
    ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_single_arm"),
    ("libero_object_no_noops_1.0.0_lerobot",  1.0, "libero_single_arm"),
    # (dataset_name, weight, robot_type)
    # 或
    # (dataset_name, weight, robot_type, per_dataset_config_dict)
]
```

## 数据集目录结构（LeRobot 格式）

```
data_root_dir/
└── dataset_name/
    ├── meta/
    │   ├── modality.json    # 模态定义（必须）
    │   └── info.json
    └── data/
        └── chunk-000/
            └── episode_*.parquet
```

## 运行验证

```bash
python ABot/dataloader/lerobot_datasets.py \
  --config_yaml examples/LIBERO/train_files/libero_config.yaml
# 将遍历前 100 个 batch 并打印进度
```

## 扩展指引

- **新数据集**：新增 `LeRobotSingleDataset` 对应目录并在 `mixtures.py` 注册
- **新机器人**：在 `data_config.py` 实现 `modality_config()` 和 `transform()` 并注册到 `ROBOT_TYPE_CONFIG_MAP`
- **平衡采样**：`get_vla_dataset(balance_dataset_weights=True)` 按数据集均匀采样，忽略 weight
