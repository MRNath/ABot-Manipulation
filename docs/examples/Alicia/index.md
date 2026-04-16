# Alicia

使用 Alicia 机械臂以 LeRobot `v3.0` 录制的数据集进行 ABot 微调。

## 数据前提

- 目录结构遵循 LeRobot `v3.0`
- 默认适配单臂双相机数据：
  - `observation.images.top_camera`
  - `observation.images.wrist_camera`
  - `observation.state`
  - `action`
  - `task_index`

## 训练入口

```bash
bash examples/Alicia/train_files/run_alicia_train.sh
```

## 关键配置

配置文件：`examples/Alicia/train_files/ABot_alicia.yaml`

- `data_mix: single_dataset`
- `dataset_name: recore_dataset_two_box_location_offical`
- `dataset_robot_type: alicia_joint_v3`
- `lerobot_version: v3.0`
- `video_backend: torchvision_av`
- 默认留出最后 `8` 条 episode 做验证
