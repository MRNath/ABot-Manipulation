# data_process/any4lerobot — 多源格式转换工具集

`any4lerobot`（原 `openx2lerobot`）是将各来源机器人数据集转换为 LeRobot v2/v3 格式的通用工具箱，作为 git submodule 引入。

## 子目录索引

| 子目录 | 功能 |
|--------|------|
| `openx2lerobot/` | Open X-Embodiment → LeRobot |
| `agibot2lerobot/` | AgiBotWorld-Beta → LeRobot |
| `robomind2lerobot/` | RoboMIND → LeRobot |
| `libero2lerobot/` | LIBERO → LeRobot |
| `lerobot2rlds/` | LeRobot → RLDS（反向转换） |
| `ds_version_convert/` | LeRobot v1.6 / v2.0 / v2.1 / v3.0 版本互转 |
| `dataset_merging/` | 多 LeRobot 数据集合并 |

## 版本说明

- ABot-M0 训练主要使用 LeRobot **v2** 格式
- AgiBotWorld / RoboMind 原始数据可先转为 v3，再用 `ds_version_convert` 降至 v2.1

## 快速示例（LIBERO → LeRobot）

```bash
cd data_process/any4lerobot
# 参考各子目录 README.md 获取具体参数
python libero2lerobot/convert.py \
  --input_dir /path/to/libero_hdf5 \
  --output_dir /path/to/output_lerobot
```

## 计算 delta state 统计量

模型推理时需要 `stats_delta_state.json` 用于动作反归一化：

```bash
cd examples/Pretrain
bash run_compute_delta_state_stats.sh
# 或直接运行 compute_delta_state_stats.py 并指定配置文件
```

## 扩展指引

- 新增数据源：仿照 `libero2lerobot/` 创建 `<source>2lerobot/` 子目录，实现 `convert.py`
- 升级数据版本：使用 `ds_version_convert/v21_to_v30/` 脚本升级为 v3.0 格式
- 合并数据集：使用 `dataset_merging/` 工具合并后统一配置 `data_mix`
