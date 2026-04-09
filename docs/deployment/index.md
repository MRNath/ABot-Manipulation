# deployment — 推理服务

提供基于 WebSocket 的模型推理服务，使机器人仿真环境（LIBERO / RoboTwin 等）可通过网络调用 ABot-M0。

## 子模块索引

| 子模块 | 职责 | 说明 |
|--------|------|------|
| [model_server](model_server/index.md) | WebSocket 服务器、客户端、图像工具 | → model_server/index.md |

## 架构概述

```
仿真环境（LIBERO condaenv）
        │   WebSocket（默认端口 10093）
        ▼
deployment/model_server/tools/websocket_policy_server.py
        │
        ▼
ABot_M0.predict_action()
        │
        ▼
normalized_actions → 反序列化 → 仿真执行
```

## 启动推理服务

```bash
# 启动服务（ABot conda 环境）
python deployment/model_server/server_policy.py \
  --ckpt_path /path/to/checkpoint \
  --port 10093 \
  --use_bf16            # 可选：bfloat16 推理节省显存
```

## 扩展指引

- 修改端口：`--port` 参数或修改 `server_policy.py` 默认值
- 调试模式：`export DEBUG=1` 启动前设置，开启 debugpy 远程调试（端口 10095）
- 更换传输协议：实现新的 Server 类并替换 `WebsocketPolicyServer`
