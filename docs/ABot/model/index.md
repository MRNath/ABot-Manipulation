# ABot/model — 模型架构

实现 ABot-M0 的完整前向推理与预测流程，分为框架层与子模块层。

## 子模块索引

| 子模块 | 职责 | 文档 |
|--------|------|------|
| [framework](framework/index.md) | 顶层框架注册、`ABot_M0` 类、`base_framework` | → framework/index.md |
| [modules/vlm](modules/vlm/index.md) | Qwen3-VL 骨干封装，图文特征提取 | → modules/vlm/index.md |
| [modules/action_model](modules/action_model/index.md) | AML 动作头（Flow Matching + DiT） | → modules/action_model/index.md |
| [modules/vggt_tools](modules/vggt_tools/index.md) | VGGT 3D 感知预处理与 CrossAttention 融合 | → modules/vggt_tools/index.md |

## 关键文件

| 文件 | 说明 |
|------|------|
| `ABot/model/framework/ABot_M0.py` | 主模型类，整合 VLM + AML + VGGT |
| `ABot/model/framework/base_framework.py` | `baseframework` 基类，含 `from_pretrained` |
| `ABot/model/tools.py` | `FRAMEWORK_REGISTRY`，模型注册与 `build_framework` |

## 快速验证

```bash
# 用假数据走通前向 + predict_action
python ABot/model/framework/ABot_M0.py \
  --config_yaml examples/LIBERO/train_files/libero_config.yaml
```

## 数据流概览

```text
图像 + 语言指令
     │
     ▼
Qwen3-VL (bfloat16)  →  last_hidden [B, L, H]
     │
     ├── [可选] VGGT → spatial_tokens → CrossAttention 融合
     │
     ▼
AML ActionHead (float32) → 预测动作 [B, T, action_dim]
```

## 论文实现对齐 (Paper Consistency)

- **Action Manifold Learning (AML) 动作流形学习**：代码完全贯彻了抛弃噪声/速度等高维目标预测，改为预测连续真实动作 (`a-pred`) 的理念。实现在 `AML_ActionHeader.py` 中，通过截取重算速度场求 MSE 损失，并在推理中用 Euler ODE 进行采样生成。
- **视觉语言特征融合 (VLM Interaction)**：代码证实无需提供多层外挂查询特征，而是直接提取 `Qwen3-VL` 骨干前向运算后的最后一层隐状态（`last_hidden = qwenvl_outputs.hidden_states[-1]`）参与动作推断，匹配论文消融实验的最优解。
- **3D 几何特征注入 (3D Information Injection)**：
  - **VGGT 单视点注入**：已实现。通过配置 `use_vggt` 触发单目视觉感知编码提取并运用单层的 Cross-Attention 合并至骨干特征流（实现在 `vggt_tools.py`）。
  - **多视点合成提取**：尚未作为主干代码默认实现整合（Qwen-Image-Edit 生成暂缺）。

## 模块对外接口 (Public Interfaces)

`ABot_M0` 主模型包装了 VLM 骨干和 DiT 动作模块两大组件。对外提供以下高层数据访问接口用于训练与交互：

- `forward(examples: List[dict], **kwargs) -> dict`: 
  核心训练入口，接收从 `DataLoader` 打包而出的标准包含 `image`, `lang`, `state`, `action` 共计多模态数据的 Python 字典列表。内部调度两套骨干引擎完成反向所需梯度图，统一对外散出包含各类回归损失映射字典（如 `{"action_loss": action_loss}`）。
- `predict_action(examples: List[dict], **kwargs) -> dict`: 
  核心前向推论点，用于模型 Eval 或物理端侧部署时的单帧/多帧动作预测。直接接收实时拼接的图像、状态与指令等，并经过 ODE 取样，返回包含连续控制采样的字典对象（返回 `{"normalized_actions": np.ndarray}`）。
