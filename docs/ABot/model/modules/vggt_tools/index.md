# model/modules/vggt_tools — 3D 空间感知

用 VGGT-1B 提取多视角 3D 空间 token，并通过 CrossAttention 融合至 VLM 隐状态。

## 关键文件

| 文件 | 说明 |
|------|------|
| `ABot/model/modules/vggt_tools.py` | `preprocess_images`、`CrossAttention` 实现 |

## 功能说明

- `preprocess_images(batch_images, size)`：将 PIL 图像列表归一化为 VGGT 输入格式
- `VGGT.aggregator(spatial_input)`：无梯度前向，输出多层聚合 token
- `self.spatial_projector`：线性层将 VGGT 的 2048 维投影至 VLM hidden_size
- `CrossAttention(d_model, d_hidden, kv_dim)`：以 VLM 特征为 Query、空间 token 为 KV 进行注意力融合

## 开关控制

在训练 YAML 中设置：

```yaml
framework:
  use_vggt: true    # false 则跳过所有 3D 感知步骤，节省约 6GB 显存
```

## 显存与性能

| 模式 | 额外显存 | 说明 |
|------|----------|------|
| `use_vggt: true` | ~6 GB | VGGT-1B 以 `torch.no_grad` 运行 |
| `use_vggt: false` | 0 | 跳过 VGGT 全部计算 |

## 扩展指引

- 替换 3D 骨干：在 `ABot_M0.__init__` 中将 `VGGT` 替换为其他视觉编码器，保持 `spatial_projector` 输入维度一致
- 多帧融合：修改 `aggregated_tokens_list[-1][:, 0, ps_idx:, :]` 的时间维度切片方式
