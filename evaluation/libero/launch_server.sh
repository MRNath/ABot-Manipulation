#!/usr/bin/env bash
# LIBERO inference server launcher. Thin wrapper around evaluation/_lib:
# sets LIBERO defaults, resolves a CUDA-capable python, then delegates.
set -euo pipefail

export START_PORT=${START_PORT:-29056}
export MASTER_PORT=${MASTER_PORT:-29069}
export CONFIG_NAME=${CONFIG_NAME:-libero_all}
export SAVE_ROOT=${SAVE_ROOT:-visualization/libero}
export SAVE_PRED_VIDEO=${SAVE_PRED_VIDEO:-1}
export PRED_VIDEO_FPS=${PRED_VIDEO_FPS:-10}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
LIBERO_NAS_PYTHON_BIN=${LIBERO_NAS_PYTHON_BIN:-}
LIBERO_CUDA_PYTHON_BIN=${LIBERO_CUDA_PYTHON_BIN:-}
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && [[ -x "${LIBERO_CUDA_PYTHON_BIN}" ]]; then
        PYTHON_BIN="${LIBERO_CUDA_PYTHON_BIN}"
    elif [[ -x "${LIBERO_NAS_PYTHON_BIN}" ]]; then
        PYTHON_BIN="${LIBERO_NAS_PYTHON_BIN}"
    else
        PYTHON_BIN=python
    fi
fi
export PYTHON_BIN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
if [[ "${PYTHON_BIN}" == */* ]]; then
    PYTHON_DIR=$(cd "$(dirname "${PYTHON_BIN}")" && pwd)
    PYTHON_PREFIX=$(cd "${PYTHON_DIR}/.." && pwd)
    export PATH="${PYTHON_DIR}:${PATH}"
    export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib/env.sh"
exec bash "${SCRIPT_DIR}/../_lib/serve.sh"
