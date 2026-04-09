# model/modules/vlm — 视觉语言模型骨干

封装 Qwen3-VL，提供图文多模态特征提取能力，输出隐状态供动作头使用。

## 关键文件

| 文件 | 说明 |
|------|------|
| `ABot/model/modules/vlm/QWen3.py` | Qwen3-VL 接口封装，含 `build_qwenvl_inputs` |
| `ABot/model/modules/vlm/__init__.py` | 导出 `get_vlm_model` 工厂函数 |

## 功能说明

- 接受多视角 RGB 图像（PIL.Image 列表）与自然语言指令
- 调用 `build_qwenvl_inputs` 构造 Qwen3-VL 格式的 token 化输入
- 以 `bfloat16` 精度推理，输出 `hidden_states[-1]`（形状 `[B, L, H]`）
- `H`（hidden_size）自动对齐到动作头的 `cross_attention_dim`

## 使用示例

```python
from ABot.model.modules.vlm import get_vlm_model
from omegaconf import OmegaConf

cfg = OmegaConf.load("examples/LIBERO/train_files/libero_config.yaml")
vlm = get_vlm_model(config=cfg)

qwen_inputs = vlm.build_qwenvl_inputs(
    images=[[pil_image]],       # [B, [views]]
    instructions=["pick up the cup"],
)
outputs = vlm(**qwen_inputs, output_hidden_states=True, return_dict=True)
last_hidden = outputs.hidden_states[-1]  # [B, L, H]
```

## 配置字段

在训练 YAML 中通过 `framework.qwenvl` 配置：

```yaml
framework:
  qwenvl:
    base_vlm: /path/to/Qwen3-VL-weights   # HuggingFace 或本地路径
```

## 扩展指引

- 替换骨干：在 `__init__.py` 新增工厂分支，实现相同的 `build_qwenvl_inputs` / `forward` 接口
- 冻结骨干：在 YAML `trainer.freeze_modules` 中列出 `qwen_vl_interface` 即可
