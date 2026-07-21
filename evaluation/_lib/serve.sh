#!/usr/bin/env bash
# Parameterized wam inference-server launcher. Source env.sh first (or run
# from the repo root with PYTHONPATH set), then invoke via per-benchmark
# wrappers that export the variables below.
#
# Required:
#   CONFIG_NAME        → wam config registry name
# Optional:
#   START_PORT         → websocket port for policy clients (default 29056)
#   MASTER_PORT        → distributed rendezvous port (default 29069)
#   NPROC_PER_NODE     → torch.distributed.run processes (default 1)
#   SAVE_ROOT          → output dir (default visualization/)
#   SAVE_PRED_VIDEO    → if set, passed as --save_pred_video
#   PRED_VIDEO_FPS     → if set, passed as --pred_video_fps
#   ATTN_MODE          → if set, passed as --attn-mode (e.g. torch)
#   PYTHON_BIN         → python executable (default python)
#   EXTRA_SERVER_ARGS  → extra args appended verbatim (word-split on purpose)
set -euo pipefail

CONFIG_NAME=${CONFIG_NAME:?CONFIG_NAME is required}
START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29069}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
SAVE_ROOT=${SAVE_ROOT:-visualization}
PYTHON_BIN=${PYTHON_BIN:-python}

mkdir -p "${SAVE_ROOT}"

cmd=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_port "${MASTER_PORT}"
    wam/server.py
    --config-name "${CONFIG_NAME}"
    --port "${START_PORT}"
    --save_root "${SAVE_ROOT}"
)
if [[ -n "${SAVE_PRED_VIDEO:-}" ]]; then
    cmd+=(--save_pred_video "${SAVE_PRED_VIDEO}")
fi
if [[ -n "${PRED_VIDEO_FPS:-}" ]]; then
    cmd+=(--pred_video_fps "${PRED_VIDEO_FPS}")
fi
if [[ -n "${ATTN_MODE:-}" ]]; then
    cmd+=(--attn-mode "${ATTN_MODE}")
fi
if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
    cmd+=(${EXTRA_SERVER_ARGS})
fi

echo "[serve.sh] ${cmd[*]}" >&2
exec "${cmd[@]}"
