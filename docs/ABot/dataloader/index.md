# ABot/dataloader — 数据加载

基于 LeRobot 格式封装多数据集混合加载，支持按权重采样与多机器人类型。

## 子模块索引

| 子模块 | 职责 | 说明 |
|--------|------|------|
| [gr00t_lerobot](gr00t_lerobot/index.md) | LeRobot 单/混合数据集实现、体态配置 | → gr00t_lerobot/index.md |
| `lerobot_datasets.py` | `get_vla_dataset`、`make_LeRobotSingleDataset` 工厂函数 | 直接读取即可 |

## 关键文件

| 文件 | 说明 |
|------|------|
| `ABot/dataloader/__init__.py` | 导出 `build_dataloader` 统一接口 |
| `ABot/dataloader/lerobot_datasets.py` | 数据集构建工厂，依赖 `DATASET_NAMED_MIXTURES` |

## 快速使用

```python
from ABot.dataloader import build_dataloader
from omegaconf import OmegaConf

cfg = OmegaConf.load("examples/LIBERO/train_files/libero_config.yaml")
dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

for batch in dataloader:
    # batch: List[dict]，每个 dict 含 image / lang / action / state
    break
```

## 关键配置字段

```yaml
datasets:
  vla_data:
    data_root_dir: /path/to/datasets    # 数据集根目录
    data_mix: libero_mix                # 对应 DATASET_NAMED_MIXTURES 中的键
    per_device_batch_size: 8
    image_size: 224                     # 训练时 resize 分辨率
    include_state: "True"               # 是否加载关节状态
    video_backend: decord               # decord 或 torchvision_av
    delete_pause_frame: false
```

## 扩展指引

- 添加新数据混合：在 `gr00t_lerobot/mixtures.py` 中注册新的 `DATASET_NAMED_MIXTURES` 键
- 支持新机器人类型：在 `gr00t_lerobot/data_config.py` 的 `ROBOT_TYPE_CONFIG_MAP` 中添加配置

## 论文实现对齐 (Paper Consistency)

- **Pad-to-Dual-Arm (双臂补齐范式)**：代码从底层预处理了模型双/单臂兼容设计，将单臂任务的动作缺失部分统一填零补齐（总计输出14维），并传递 `action_mask` 以保证该维度对梯度的屏蔽，完美对应论文的单双臂全参数共享设计（实现在 `padding.py`）。
- **Delta Actions & Rotation Vectors (增量与旋转向量空间)**：`state_action.py` 严格按照论文设计，将空间坐标的解算建立在相对末端执行器增量（EEF Delta）基础上，并将姿态角度平铺为无奇异的旋转向量（Rotation vectors $\theta k$），保障跨数据集策略泛化。
- **Task-Uniform Sampling (特定任务分布重采样)**：实现于 `LeRobotMixtureDataset` 与混合逻辑。借助配置中的 `balance_dataset_weights` 调节数据集比例，对频繁小任务抑制并增强小样本库与轨迹，对齐了抗灾难性遗忘的均匀采样策略。

## 模块对外接口 (Public Interfaces)

负责数据流水线的集约化对接。为其他模块尤其是 Trainer 提供开箱即用的 Iterable 批次。

- `build_dataloader(cfg: OmegaConf, dataset_py: str) -> DataLoader`:
  核心工厂函数。接受全局的配置文件及数据集脚本指向（通常为 LeRobot 支持的脚本），依据 YAML 配置内的 `data_mix`（指定多源数据集），`per_device_batch_size` 提取数据和组装 `collate_fn`。向外部返回原生的 PyTorch `DataLoader` 对象。迭代出的批次会被打包填充处理，供模型 `forward` 拆解。
