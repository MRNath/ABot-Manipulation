# model/framework — 顶层框架

定义 `ABot_M0` 完整模型类及其注册机制，是训练与推理的统一入口。

## 关键文件

| 文件 | 说明 |
|------|------|
| `ABot/model/framework/ABot_M0.py` | 主模型，整合 VLM / AML / VGGT，实现 `forward` 与 `predict_action` |
| `ABot/model/framework/base_framework.py` | `baseframework` 基类，提供 `from_pretrained` 加载接口 |
| `ABot/model/framework/share_tools.py` | 参数冻结、权重加载等共享工具函数 |
| `ABot/model/tools.py` | `FRAMEWORK_REGISTRY`（装饰器注册表）与 `build_framework` 工厂 |

## 核心类：ABot_M0

```python
# ABot/model/framework/ABot_M0.py

@FRAMEWORK_REGISTRY.register("ABot_M0")
class ABot_M0(baseframework):
    def __init__(self, config): ...        # 构造 VLM + AML + VGGT
    def forward(self, examples) -> dict:   # 返回 {"action_loss": tensor}
    def predict_action(self, examples) -> dict:  # 返回 {"normalized_actions": ndarray}
```

### `forward` 输入格式

```python
example = {
    "image":  [PIL.Image, ...],          # 多视角图像列表
    "lang":   "pick up the cup",         # 自然语言指令
    "action": np.ndarray,                # shape [chunk_len, action_dim]
    "state":  np.ndarray | None,         # shape [1, state_dim]（可选）
    "action_mask": np.ndarray | None,    # shape [action_dim]（可选）
}
```

## 构建与加载

```python
from ABot.model.framework import build_framework
from omegaconf import OmegaConf

cfg = OmegaConf.load("examples/LIBERO/train_files/libero_config.yaml")
model = build_framework(cfg)          # 从 YAML 构建

# 从本地 checkpoint 加载
from ABot.model.framework.base_framework import baseframework
model = baseframework.from_pretrained("/path/to/checkpoint")
```

## 扩展指引

- 新增模型变体：创建新类并用 `@FRAMEWORK_REGISTRY.register("MyVariant")` 注册
- 在 YAML `framework.name` 字段切换模型变体
- `use_vggt: false` 可禁用 3D 感知模块以节省显存
