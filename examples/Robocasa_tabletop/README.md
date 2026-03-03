# 🚀 Robocasa-GR1-Tabletop-Tasks Evaluation

This document provides instructions for reproducing our **experimental results** with [robocasa-gr1-tabletop-tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks).  
The evaluation process consists of two main parts:  

1. Setting up the `robocasa` environment and dependencies.  
2. Running the evaluation by launching services in both `ABot` and `robocasa` environments.  

We have verified that this workflow runs successfully on **NVIDIA A100** GPUs.  


# Evaluation

![Eval Videos](https://github.com/user-attachments/assets/a5ff9bdd-b47d-4eb0-95ac-c09556fb4b48)


## ⬇️ 0. Download Checkpoints
Please download Checkpoint from [🤗 ABot-M0-Robocasa](). You should replace the `base_vlm` in the `config.yaml` file with your own path.

## 📦 1. Environment Setup

To set up the environment, please first follow the [official RoboCasa installation guide](https://github.com/robocasa/robocasa-gr1-tabletop-tasks?tab=readme-ov-file#getting-started) to install the base `robocasa-gr1-tabletop-tasks` environment.  

than pip soceket support

'''bash
pip install tyro
'''

---

## 🚀 2. Evaluation Workflow

### Step 1. Start the server (ABot environment)

In the first terminal, activate the `ABot` conda environment and run:  

```bash
python deployment/model_server/server_policy.py \
        --ckpt_path ${your_ckpt} \
        --port 5678 \
        --use_bf16
```

---

### Step 2. Start the simulation (robocasa environment)

In the second terminal, activate the `robocasa` conda environment and run:  

```bash
export PYTHONPATH=$(pwd):${PYTHONPATH}
your_ckpt=path_to_checkpoint

python examples/Robocasa_tabletop/eval_files/simulation_env.py\
   --args.env_name ${env_name} \
   --args.port 5678 \
   --args.n_episodes 50 \
   --args.n_envs 1 \
   --args.max_episode_steps 720 \
   --args.n_action_steps 12 \
   --args.video_out_path ${video_out_path} \
   --args.pretrained_path ${your_ckpt}
```


### Optional: Batch Evaluation

If you have more GPU, you can use the batch evaluation script:
```bash
bash examples/Robocasa_tabletop/batch_eval_args.sh
```
⚠️ **Note:** Please ensure that you specify the correct checkpoint path in `batch_eval_args.sh`  

---
