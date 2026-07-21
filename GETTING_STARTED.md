# ABot-M0.5 — World Action Model (WAM) Inference Stack

ABot-M0.5 is a World Action Model (WAM) for robot policy learning, built on top of the Wan2.2 video diffusion backbone. This repository contains the inference stack and evaluation scripts for the **LIBERO** and **RoboCasa365** benchmarks.

Licensed under the Apache License, Version 2.0 — see [LICENSE.txt](LICENSE.txt) for details.

---

## Table of Contents

- [Installation](#installation)
- [Download Checkpoints](#download-checkpoints)
- [RoboCasa365 Evaluation](#robocasa365-evaluation)
- [LIBERO Evaluation](#libero-evaluation)

---

## Installation

### Main environment

```bash
conda create -n abot_m05 python=3.10 -y
conda activate abot_m05
pip install -U pip setuptools wheel

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

For LIBERO evaluation, also install:

```bash
pip install lerobot==0.3.3 scipy wandb --no-deps
```

### RoboCasa365 client environment

RoboCasa evaluation requires a separate conda environment with the simulator installed.

```bash
conda activate robocasa365          # your RoboCasa365 env
pip install websockets msgpack
```

---

## Download Checkpoints

Download both the post-training checkpoint and the base checkpoint from Hugging Face:

```bash
# Post-training checkpoint (fine-tuned for the target benchmark)
hf download acvlab/abot-m0.5 \
  --repo-type model \
  --include "checkpoint_step/**" \
  --local-dir /path/to

# Base checkpoint (Wan2.2 backbone + VAE + text encoder + tokenizer)
hf download acvlab/abot-m0.5 \
  --repo-type model \
  --include "base_checkpoint/**" \
  --local-dir /path/to
```

This gives you:

| Path | Contains |
|------|----------|
| `/path/to/checkpoint_step/` | Fine-tuned transformer weights + config |
| `/path/to/base_checkpoint/` | VAE, text encoder, tokenizer, base transformer |

---

## RoboCasa365 Evaluation

### 1. Fill local paths

Edit the variables at the top of the target script:

- `script/eval/local/eval_atomic_seen.sh`
- `script/eval/local/eval_composite_seen.sh`
- `script/eval/local/eval_composite_unseen.sh`

```bash
CKPT_PATH="/path/to/checkpoint_step"
WAN22_PRETRAINED_MODEL_NAME_OR_PATH="/path/to/base_checkpoint"
WRAPPER_PYTHON="/path/to/abot_m05/bin/python"   # ABot-M0.5 env python
SERVER_PYTHON="/path/to/abot_m05/bin/python"     # ABot-M0.5 env python
ROBOCASA_ENV_ROOT="/path/to/robocasa365"          # RoboCasa365 simulator root
GPU_IDS="0,1,2,3"
```

| Variable | Description |
|----------|-------------|
| `CKPT_PATH` | Post-training checkpoint directory. |
| `WAN22_PRETRAINED_MODEL_NAME_OR_PATH` | Base checkpoint directory (VAE, text encoder, etc.). |
| `WRAPPER_PYTHON` | Python from the ABot-M0.5 environment. |
| `SERVER_PYTHON` | Python from the ABot-M0.5 environment. |
| `ROBOCASA_ENV_ROOT` | RoboCasa365 simulator environment root. The client python is auto-detected as `${ROBOCASA_ENV_ROOT}/bin/python`. |
| `GPU_IDS` | Comma-separated GPU ids for parallel evaluation. |

### 2. Run evaluation

```bash
# Atomic seen tasks
GPU_IDS=0,1,2,3 BACKGROUND=1 bash script/eval/local/eval_atomic_seen.sh

# Composite seen tasks
GPU_IDS=0,1,2,3 BACKGROUND=1 bash script/eval/local/eval_composite_seen.sh

# Composite unseen tasks
GPU_IDS=0,1,2,3 BACKGROUND=1 bash script/eval/local/eval_composite_unseen.sh
```

### Optional parameters

```bash
SPLIT="pretrain"          # data split
NUM_EPISODES=50           # episodes per task
MAX_STEPS=500             # max steps per episode
ATTN_MODE="torch"         # attention implementation: torch | flashattn
HORIZON_MULTIPLIER=1      # action horizon multiplier
AUTO_AGGREGATE=1          # auto-aggregate results when all workers finish
BACKGROUND=1              # run workers in background
```

---

## LIBERO Evaluation

### 1. Fill local paths

Edit the variables at the top of the target script (`script/eval/local/eval_libero_*.sh`):

```bash
POSTTRAIN_CKPT="/path/to/libero/checkpoint_step"
WAN22_PRETRAINED_PATH="/path/to/base_checkpoint"
PY_BIN_SERVER="/path/to/abot_m05/bin/python"   # ABot-M0.5 env python
PY_BIN_CLIENT="/path/to/abot_m05/bin/python"   # LIBERO client python
CUDA_VISIBLE_DEVICES="0,1"                      # GPUs to use
```

| Variable | Description |
|----------|-------------|
| `POSTTRAIN_CKPT` | Post-training checkpoint directory (LIBERO fine-tuned). |
| `WAN22_PRETRAINED_PATH` | Base checkpoint directory. |
| `PY_BIN_SERVER` | Python from the ABot-M0.5 environment (runs the model server). |
| `PY_BIN_CLIENT` | Python for the LIBERO client (must have LIBERO + lerobot installed). |
| `CUDA_VISIBLE_DEVICES` | GPU ids for evaluation. |

### 2. Run evaluation

Each script targets a different LIBERO benchmark subset:

```bash
bash script/eval/local/eval_libero_goal.sh       # LIBERO-Goal
bash script/eval/local/eval_libero_long.sh       # LIBERO-Long (libero_10)
bash script/eval/local/eval_libero_object.sh     # LIBERO-Object
bash script/eval/local/eval_libero_spatial.sh    # LIBERO-Spatial
```


### Available LIBERO configs

| Config name | Benchmark |
|-------------|-----------|
| `libero_10` | LIBERO-Long |
| `libero_goal` | LIBERO-Goal |
| `libero_object` | LIBERO-Object |
| `libero_spatial` | LIBERO-Spatial |

### Optional parameters

```bash
NUM_TASKS=10              # number of tasks
TEST_NUM=50               # episodes per task
MAX_STEPS=800            # max steps per episode
ATTN_MODE="torch"        # attention implementation: torch | flashattn
SEED=6666                # random seed
```

