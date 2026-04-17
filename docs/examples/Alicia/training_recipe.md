# Alicia Training Recipe

本页固定记录当前 Alicia 微调推荐配置，以及这些参数如何从公开 benchmark 数据规模和仓库现有后训练脚本推导而来。

## 推荐配置

目标：

- `action_horizon = 30`
- 等效全局 batch size = `64`
- 硬件：`4 x 80 GB`

推荐取值：

```yaml
framework:
  action_model:
    future_action_window_size: 29
    action_horizon: 30
    repeated_diffusion_steps: 4

datasets:
  vla_data:
    per_device_batch_size: 4

trainer:
  max_train_steps: 18000
  num_warmup_steps: 2000
  save_interval: 2000
  eval_interval: 500
  eval_num_batches: 8
  gradient_accumulation_steps: 4
  repeated_diffusion_steps: 4
  learning_rate:
    base: 5.0e-06
    qwen_vl_interface: 5.0e-06
    action_model: 5.0e-05
```

配套启动脚本：

```bash
num_processes=4
```

等效全局 batch size：

- `4 GPUs x 4 per_device_batch_size x 4 gradient_accumulation_steps = 64`

## Dataset Comparison

### Alicia

本地 Alicia 示例数据集元信息：

- `1` 个任务
- `72` 条 episode
- `129,384` 帧

训练时默认留出最后 `8` 条 episode 做验证，因此训练集约为：

- `64` 条 episode
- 约 `115,008` 个 frame-step

来源：[info.json](/home/kongqingwei/data/recore_dataset_two_box_location_offical/meta/info.json:2)

### LIBERO

公开 LeRobot 数据卡信息：

- `130` 个任务
- `273k` train rows

本仓库后训练脚本：

- `8` 卡
- 单卡 batch `8`
- 全局 batch `64`
- `max_train_steps = 40000`

来源：

- https://huggingface.co/docs/lerobot/libero
- https://huggingface.co/datasets/physical-intelligence/libero
- [run_libero_train.sh](/home/kongqingwei/ABot-Manipulation/examples/LIBERO/train_files/run_libero_train.sh:34)

### RoboCasa

公开 LeRobot 数据卡示例：

- `30` 个任务
- `1651` 条 episode
- `577,597` 帧

本仓库后训练脚本：

- `8` 卡
- 单卡 batch `16`
- 全局 batch `128`
- `max_train_steps = 50000`

来源：

- https://huggingface.co/datasets/yananchen/robocasa_lerobot
- [run_robocasa.sh](/home/kongqingwei/ABot-Manipulation/examples/Robocasa_tabletop/train_files/run_robocasa.sh:23)

### RoboTwin 2.0

官方公开说明：

- `50` 个任务
- `5` 个机器人平台
- `over 100,000 trajectories`

本仓库后训练脚本：

- `8` 卡
- 单卡 batch `4`
- 全局 batch `32`
- `max_train_steps = 150000`

来源：

- https://github.com/RoboTwin-Platform/RoboTwin
- https://robotwin-platform.github.io/doc/tasks/
- [run_robotwin_train.sh](/home/kongqingwei/ABot-Manipulation/examples/Robotwin/train_files/run_robotwin_train.sh:33)

## Why These Alicia Values

### Why `action_horizon = 30`

- Alicia 原始配置是 `10`
- 这里提高到 `30`，是为了让策略一次生成更长动作块，提升真实机器人执行时的时序规划能力
- 但 Alicia 仍然是单任务、小数据，因此不建议直接拉到 Robotwin 那种更长时序

保持一致性要求：

- `future_action_window_size = action_horizon - 1`

### Why global batch `64`

- 它与 LIBERO 脚本的有效全局 batch 完全一致
- 比 Robotwin 的 `32` 更强一档
- 比 RoboCasa 的 `128` 更保守，适合 Alicia 小数据

这代表 Alicia 训练强度已经处在“正式后训练”档位，而不是调试档位。

### Why `max_train_steps = 18000`

按 Alicia 训练集约 `115,008` 个 step 计算：

- 每个 epoch 的 optimizer step 约为 `115008 / 64 ≈ 1797`
- `18000` step 约为 `10.0` 个 epoch

对比其它示例：

- LIBERO：`40000 / (273000 / 64) ≈ 9.4` epoch
- RoboCasa：`50000 / (577597 / 128) ≈ 11.1` epoch

因此 Alicia 设为 `18000` step，可以把训练轮数控制在和原始示例接近的 `9-11` epoch 档位。  
如果 Alicia 继续使用 `40000` step，在当前数据规模下会变成约 `22.3` epoch，明显偏高，更容易过拟合。

### Why learning rate `5e-6 / 5e-5`

仓库原始后训练脚本常见学习率是：

- `base = 1e-5`
- `qwen_vl_interface = 1e-5`
- `action_model = 1e-4`

但 Alicia 与这些 benchmark 相比更小：

- 任务数最少
- episode 数量最少
- horizon 从 `10` 提高到了 `30`

因此推荐把学习率整体降到原始示例的 `0.5x`：

- `base = 5e-6`
- `qwen_vl_interface = 5e-6`
- `action_model = 5e-5`

这样更符合小数据真实机器人 SFT 的稳定性需求。

如果前几千步 loss 下降偏慢，可先只提高 action head 学习率：

```yaml
trainer:
  learning_rate:
    base: 5.0e-06
    qwen_vl_interface: 5.0e-06
    action_model: 7.5e-05
```

### Why `repeated_diffusion_steps = 4`

- Robotwin / RoboCasa 等原始后训练示例大多落在 `4`
- Alicia 原先是 `8`
- 在 `action_horizon = 30` 的前提下，继续保留 `8` 会增加训练计算和显存压力
- 设为 `4` 更接近原始后训练配方，也更利于 4 卡 80G 的稳定训练

## Operational Notes

- 该配置默认全量微调，不冻结 `qwen_vl_interface`
- 如果显存仍有明显余量，可以尝试：

```yaml
datasets:
  vla_data:
    per_device_batch_size: 8

trainer:
  gradient_accumulation_steps: 2
```

这同样等效全局 batch `64`，但单步峰值显存会更高。
