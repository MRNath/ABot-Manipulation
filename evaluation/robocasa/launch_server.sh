#!/usr/bin/env bash
# RoboCasa inference server launcher. MASTER_PORT defaults to 29068 to avoid
# EADDRINUSE with other benches' defaults (29061/29069) or stale runs.
set -euo pipefail

export START_PORT=${START_PORT:-29056}
export MASTER_PORT=${MASTER_PORT:-29068}
export CONFIG_NAME=${CONFIG_NAME:-robocasa_train_test}
export SAVE_ROOT=${SAVE_ROOT:-visualization/robocasa}
export SAVE_PRED_VIDEO=${SAVE_PRED_VIDEO:-1}
export PRED_VIDEO_FPS=${PRED_VIDEO_FPS:-10}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHON_BIN=${PYTHON_BIN:-python}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib/env.sh"
exec bash "${SCRIPT_DIR}/../_lib/serve.sh"
