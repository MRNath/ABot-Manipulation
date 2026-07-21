#!/bin/bash
# =============================================================================
# Local multi-GPU LIBERO-Goal eval script
#
# Usage:
#   bash script/eval/local/eval_libero_goal.sh
#   POSTTRAIN_CKPT=/path/to/ckpt bash script/eval/local/eval_libero_goal.sh
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ======================== Checkpoint ========================
POSTTRAIN_CKPT=${POSTTRAIN_CKPT:?POSTTRAIN_CKPT is required, e.g. /path/to/checkpoint_step}

# ======================== Python environment ========================
export PY_BIN_SERVER=${PY_BIN_SERVER:-python}
export PY_BIN_CLIENT=${PY_BIN_CLIENT:-python}

# ======================== Base model paths ========================
export WAN22_PRETRAINED_PATH=${WAN22_PRETRAINED_PATH:?WAN22_PRETRAINED_PATH is required, e.g. /path/to/base_checkpoint}
export WAN22_BASE_CKPT=${WAN22_BASE_CKPT:-${WAN22_PRETRAINED_PATH}}

# ======================== Eval target ========================
BENCHMARK=${BENCHMARK:-libero_goal}
SEED=${SEED:-6666}

# ======================== Model config ========================
CONFIG_NAME=${CONFIG_NAME:-libero_all}
export ATTN_MODE=${ATTN_MODE:-torch}

# ======================== GPU selection ========================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5}

NUM_TASKS=${NUM_TASKS:-10}
TEST_NUM=${TEST_NUM:-50}
MAX_STEPS=${MAX_STEPS:-800}

# ======================== Run ========================
echo "============================================"
echo "[eval_libero_goal] local eval config:"
echo "  BENCHMARK            = $BENCHMARK"
echo "  SEED                 = $SEED"
echo "  CONFIG_NAME          = $CONFIG_NAME"
echo "  CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "  POSTTRAIN_CKPT       = $POSTTRAIN_CKPT"
echo "  NUM_TASKS            = $NUM_TASKS"
echo "  TEST_NUM             = $TEST_NUM"
echo "============================================"

POSTTRAIN_CKPT=${POSTTRAIN_CKPT} \
BENCHMARK=${BENCHMARK} \
NUM_TASKS=${NUM_TASKS} \
TEST_NUM=${TEST_NUM} \
MAX_STEPS=${MAX_STEPS} \
CONFIG_NAME=${CONFIG_NAME} \
SEED=${SEED} \
PY_BIN_SERVER=${PY_BIN_SERVER} \
PY_BIN_CLIENT=${PY_BIN_CLIENT} \
WAN22_PRETRAINED_PATH=${WAN22_PRETRAINED_PATH} \
WAN22_BASE_CKPT=${WAN22_BASE_CKPT} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    bash evaluation/libero/launch_eval_my_ckpt.sh
