# deployment/model_server — WebSocket 推理服务

实现 WebSocket 服务端与客户端，使用 msgpack 序列化传递图像与动作数据。

## 关键文件

| 文件 | 说明 |
|------|------|
| `server_policy.py` | 服务启动入口，加载模型并启动 `WebsocketPolicyServer` |
| `tools/websocket_policy_server.py` | WebSocket 服务器实现，含 idle 超时机制 |
| `tools/websocket_policy_client.py` | 仿真端客户端，发送 observation 并接收 action |
| `tools/image_tools.py` | `to_pil_preserve`：将 ndarray/tensor 转 PIL.Image，保留色彩空间 |
| `tools/msgpack_numpy.py` | msgpack 扩展，支持 numpy array 序列化 |
| `tools/debug_server_policy.py` | 离线调试版 server，不需要真实仿真环境 |

## 启动命令

```bash
# 正式服务
python deployment/model_server/server_policy.py \
  --ckpt_path /path/to/checkpoint \
  --port 10093 \
  --idle_timeout 1800    # 空闲 30 分钟自动关闭，-1 为永不关闭

# 离线调试（无需仿真环境）
python deployment/model_server/tools/debug_server_policy.py \
  --ckpt_path /path/to/checkpoint
```

## 通信协议

| 字段 | 来源 | 说明 |
|------|------|------|
| `images` | 客户端 → 服务端 | list of ndarray，各相机视角 RGB |
| `lang` | 客户端 → 服务端 | 任务描述字符串 |
| `state` | 客户端 → 服务端 | 关节状态 ndarray（可选） |
| `normalized_actions` | 服务端 → 客户端 | ndarray `[1, T, action_dim]` |

## 扩展指引

- **并发支持**：`WebsocketPolicyServer` 当前串行处理请求，可改为 `asyncio.gather` 实现并发
- **TLS 加密**：在 `websockets.serve` 调用中传入 `ssl` 上下文
- **动作反归一化**：客户端收到 `normalized_actions` 后，使用 `compute_delta_state_stats.py` 生成的统计量做反归一化
