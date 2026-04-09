# model/modules/action_model — AML 动作头

实现 Action Manifold Learning (AML)：基于 Flow Matching 的动作预测头，直接预测干净动作而非预测噪声。

## 关键文件

| 文件 | 说明 |
|------|------|
| `ABot/model/modules/action_model/AML_ActionHeader.py` | `FlowmatchingActionHead` 主类及 `get_action_model` 工厂 |
| `ABot/model/modules/action_model/DiT_modules/` | DiT（Diffusion Transformer）主干实现 |
| `ABot/model/modules/action_model/flow_matching_head/` | Flow Matching 采样器与条件模块 |

## 核心设计

- **输入**：VLM 隐状态 `[B, L, H]` + 机器人关节状态 `[B, 1, state_dim]`（可选）
- **输出（训练）**：Flow Matching 损失标量
- **输出（推理）**：动作 chunk `[B, T, action_dim]`，`T = future_action_window_size + 1`
- `repeated_diffusion_steps`（默认 4）：单 batch 多次重复以增强训练稳定性

## 关键配置

```yaml
framework:
  action_model:
    future_action_window_size: 15   # 预测未来帧数
    past_action_window_size:   1    # 条件历史帧数
    diffusion_model_cfg:
      cross_attention_dim: auto     # 自动从 VLM hidden_size 填入
trainer:
  repeated_diffusion_steps: 4       # 训练时重复 diffusion 步骤数
```

## 调用接口

```python
from ABot.model.modules.action_model.AML_ActionHeader import get_action_model

action_head = get_action_model(config=cfg)

# 训练
loss = action_head(hidden, actions_target, state, action_mask=mask)

# 推理
pred_actions = action_head.predict_action(hidden, state)  # [B, T, action_dim]
```

## 扩展指引

- 更换采样策略：修改 `flow_matching_head/` 中的 ODE 求解器
- 调整 DiT 深度/宽度：在对应 YAML `diffusion_model_cfg` 中配置 `num_layers`、`num_heads`
