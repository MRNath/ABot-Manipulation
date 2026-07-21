#!/usr/bin/env bash
# Launch a SINGLE wam server bound to one GPU.
#
# Usage:
#   bash launch_server_single.sh <GPU_ID> <SERVER_PORT> <MASTER_PORT> <SAVE_ROOT> [CONFIG_NAME]
#
# Example:
#   bash launch_server_single.sh 0 29056 29061 visualization/gpu0 libero
#
# Env overrides (optional):
#   ATTN_MODE           → attention impl. Default: torch (ckpt config.json may
#                         store 'flex' which is for training only)
#   EXTRA_SERVER_ARGS   → arbitrary extra args appended verbatim
#
# Note: MoT is determined by the checkpoint's config.json (use_mot field)
# or the config object's use_mot field (default True). No CLI flag needed.
set -euo pipefail

GPU_ID=${1:-0}
export START_PORT=${2:-29056}
export MASTER_PORT=${3:-29061}
export SAVE_ROOT=${4:-"visualization/gpu${GPU_ID}"}
export CONFIG_NAME=${5:-libero_original}
echo "[launch_server_single.sh] CONFIG_NAME=$CONFIG_NAME (from arg 5)" >&2

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHON_BIN=${PY_BIN_SERVER:-python}
# Default to 'torch' for inference: training ckpts save attn_mode=flex in
# config.json which is incompatible with inference (FlexAttn needs ctx kwargs).
export ATTN_MODE=${ATTN_MODE:-torch}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib/env.sh"
exec bash "${SCRIPT_DIR}/../_lib/serve.sh"
