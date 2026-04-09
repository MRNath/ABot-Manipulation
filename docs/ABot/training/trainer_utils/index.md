# training/trainer_utils — Trainer 工具集

提供训练器所需的参数分组、配置追踪、日志初始化、权重加载等辅助功能。

## 关键文件

| 文件 | 说明 |
|------|------|
| `trainer_tools.py` | `TrainerUtils` mixin、`build_param_lr_groups`、`resize_images`、`normalize_dotlist_args` |
| `config_tracker.py` | `AccessTrackedConfig` / `wrap_config`：记录训练中实际访问的配置键 |
| `overwatch.py` | `initialize_overwatch(name)` → Python `logging.Logger` 封装 |
| `__init__.py` | 导出 `initialize_overwatch` 等常用符号 |

## 参数学习率分组

```python
# trainer_tools.py: build_param_lr_groups
# 按 YAML trainer.learning_rate 分组设置差异化学习率
# 典型配置：
trainer:
  learning_rate:
    base:            1e-4   # 默认学习率
    action_model:    2e-4   # AML 头专属学习率
```

## 配置访问追踪

训练结束时自动将实际使用过的配置键保存为 `output_dir/config.yaml`，方便复现：

```python
from ABot.training.trainer_utils.config_tracker import wrap_config
cfg = wrap_config(OmegaConf.load("config.yaml"))
# 训练后 cfg.save_accessed_config("output/config.yaml")
```

## 日志初始化

```python
from ABot.training.trainer_utils import initialize_overwatch
logger = initialize_overwatch(__name__)
logger.info("Training started")
```

## 快速检查工具

```bash
# 验证配置追踪功能
python -c "
from omegaconf import OmegaConf
from ABot.training.trainer_utils.config_tracker import wrap_config
cfg = wrap_config(OmegaConf.load('examples/Pretrain/ABot_pretrain.yaml'))
_ = cfg.trainer.max_train_steps
cfg.save_accessed_config('/tmp/accessed_config.yaml')
print(open('/tmp/accessed_config.yaml').read())
"
```

## 扩展指引

- **新参数组**：在 `build_param_lr_groups` 中按模块名匹配并添加新 group
- **自定义日志**：`initialize_overwatch` 支持传入 `logging.Level` 覆盖默认级别
