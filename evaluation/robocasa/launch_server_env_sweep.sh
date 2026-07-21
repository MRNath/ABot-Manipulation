#!/usr/bin/env bash
set -euo pipefail

# Single env evaluation with server lifecycle:
# 1) launch server; 2) wait until ready; 3) run client eval; 4) stop server.

CKPT_PATH=${CKPT_PATH:-}
START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29100}
CONFIG_NAME=${CONFIG_NAME:-robocasa_train}
GPU_ID=${GPU_ID:-0}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-1200}

HOST=${HOST:-127.0.0.1}
SERVER_PYTHON=${SERVER_PYTHON:-python}
CLIENT_PYTHON=${CLIENT_PYTHON:-python}
ENV_NAME=${ENV_NAME:-PickPlaceCounterToCabinet}
SPLIT=${SPLIT:-target}
NUM_EPISODES=${NUM_EPISODES:-50}
MAX_STEPS=${MAX_STEPS:-500}
SEED=${SEED:-0}
DEBUG_ACTIONS=${DEBUG_ACTIONS:-0}
ATTN_MODE=${ATTN_MODE:-}
HORIZON_MULTIPLIER=${HORIZON_MULTIPLIER:-1.0}

RUN_ROOT=${RUN_ROOT:-}
SAVE_ROOT=${SAVE_ROOT:-${RUN_ROOT:+${RUN_ROOT}/save_root}}
SAVE_ROOT=${SAVE_ROOT:-visualization/robocasa_env_sweep}
RESULT_DIR=${RESULT_DIR:-${RUN_ROOT:+${RUN_ROOT}/result}}
RESULT_DIR=${RESULT_DIR:-evaluation/robocasa/results/env_sweep}
SAVE_VIDEO_DIR=${SAVE_VIDEO_DIR:-${RESULT_DIR}/videos}
SAVE_JSON=${SAVE_JSON:-${RESULT_DIR}/robocasa_eval.json}
SERVER_LOG=${SERVER_LOG:-${RESULT_DIR}/server.log}

mkdir -p "${SAVE_ROOT}" "${RESULT_DIR}" "${SAVE_VIDEO_DIR}"

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

unset PYTHONHOME
unset PYTHONPATH

ROBOCASA_CLIENT_PYTHONPATH=${ROBOCASA_CLIENT_PYTHONPATH:-}
# Client MuJoCo EGL: robosuite treats CUDA_VISIBLE_DEVICES as EGL index when set, which
# breaks non-zero physical GPU ids. Unset CUDA for the client and use EGL device 0.
client_python_env=(env -u PYTHONHOME -u PYTHONPATH -u CUDA_VISIBLE_DEVICES -u MUJOCO_EGL_DEVICE_ID)
if [[ -n "${ROBOCASA_CLIENT_PYTHONPATH}" ]]; then
  client_python_env=(env -u PYTHONHOME -u CUDA_VISIBLE_DEVICES -u MUJOCO_EGL_DEVICE_ID "PYTHONPATH=${ROBOCASA_CLIENT_PYTHONPATH}")
fi

wait_for_server() {
  local host="$1"
  local port="$2"
  local timeout="$3"
  local pid="$4"
  "${client_python_env[@]}" "${CLIENT_PYTHON}" - "$host" "$port" "$timeout" "$pid" <<'PY'
import os
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout = int(sys.argv[3])
pid = int(sys.argv[4])
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        print(f"[ERROR] Server process {pid} exited before listening on {host}:{port}", file=sys.stderr)
        sys.exit(2)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect((host, port))
        print(f"[INFO] Server is ready on {host}:{port}")
        sys.exit(0)
    except OSError:
        time.sleep(1)
    finally:
        try:
            s.close()
        except Exception:
            pass
print(f"[ERROR] Timeout ({timeout}s) waiting for server on {host}:{port}", file=sys.stderr)
sys.exit(1)
PY
}

cleanup_server() {
  if [[ -n "${SERVER_PGID:-}" ]]; then
    if kill -0 "${SERVER_PGID}" >/dev/null 2>&1; then
      kill -TERM -- "-${SERVER_PGID}" >/dev/null 2>&1 || true
      for _ in {1..10}; do
        if ! kill -0 "${SERVER_PGID}" >/dev/null 2>&1; then
          break
        fi
        sleep 1
      done
      if kill -0 "${SERVER_PGID}" >/dev/null 2>&1; then
        kill -KILL -- "-${SERVER_PGID}" >/dev/null 2>&1 || true
      fi
    fi
    if [[ -n "${SERVER_PID:-}" ]]; then
      wait "${SERVER_PID}" 2>/dev/null || true
    fi
  fi
}

trap cleanup_server EXIT

debug_flag=()
if [[ "${DEBUG_ACTIONS}" == "1" || "${DEBUG_ACTIONS}" == "true" ]]; then
  debug_flag=(--debug-actions)
fi

echo "[INFO] ===== Evaluate ENV ${ENV_NAME} ====="
echo "[INFO] CKPT_PATH=${CKPT_PATH:-<default in launch_server.sh>}"
echo "[INFO] master_port=${MASTER_PORT}, ws_port=${START_PORT}, gpu=${GPU_ID}"
echo "[INFO] SERVER_PYTHON=${SERVER_PYTHON}"
echo "[INFO] CLIENT_PYTHON=${CLIENT_PYTHON}"
echo "[INFO] ROBOCASA_CLIENT_PYTHONPATH=${ROBOCASA_CLIENT_PYTHONPATH:-<unset>}"
echo "[INFO] NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-<unset>}"
echo "[INFO] ATTN_MODE=${ATTN_MODE:-<inherit-from-ckpt>}"

launch_envs=(
  "START_PORT=${START_PORT}"
  "MASTER_PORT=${MASTER_PORT}"
  "CONFIG_NAME=${CONFIG_NAME}"
  "SAVE_ROOT=${SAVE_ROOT}"
  "SAVE_PRED_VIDEO=0"
  "PRED_VIDEO_FPS=10"
  "PYTHON_BIN=${SERVER_PYTHON}"
)
if [[ -n "${CKPT_PATH}" ]]; then
  launch_envs+=("ROBOCASA_POSTTRAIN_MODEL_PATH_TEST=${CKPT_PATH}")
fi

if [[ -n "${ATTN_MODE}" ]]; then
  echo "[INFO] ATTN_MODE override: running wam server with --attn-mode=${ATTN_MODE}"
  inline_envs=(
    "PYTHON_BIN=${SERVER_PYTHON}"
    "CUDA_VISIBLE_DEVICES=${GPU_ID}"
  )
  if [[ -n "${CKPT_PATH}" ]]; then
    inline_envs+=("ROBOCASA_POSTTRAIN_MODEL_PATH_TEST=${CKPT_PATH}")
  fi
  setsid env -u PYTHONHOME -u PYTHONPATH "${inline_envs[@]}" \
      "${SERVER_PYTHON}" -m torch.distributed.run \
          --nproc_per_node 1 \
          --master_port "${MASTER_PORT}" \
          wam/server.py \
          --config-name "${CONFIG_NAME}" \
          --port "${START_PORT}" \
          --save_root "${SAVE_ROOT}" \
          --save_pred_video "0" \
          --pred_video_fps "10" \
          --attn-mode "${ATTN_MODE}" \
      >"${SERVER_LOG}" 2>&1 &
else
  setsid env -u PYTHONHOME -u PYTHONPATH "${launch_envs[@]}" bash evaluation/robocasa/launch_server.sh >"${SERVER_LOG}" 2>&1 &
fi
SERVER_PID=$!
SERVER_PGID=${SERVER_PID}

if ! wait_for_server "${HOST}" "${START_PORT}" "${WAIT_TIMEOUT}" "${SERVER_PID}"; then
  echo "[ERROR] Server failed to become ready. Last server log tail:"
  "${client_python_env[@]}" "${CLIENT_PYTHON}" - "${SERVER_LOG}" <<'PY'
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
if not path.exists():
    print("[ERROR] server log not found")
    sys.exit(0)
lines = path.read_text(errors="ignore").splitlines()
for line in lines[-80:]:
    print(line)
PY
  exit 1
fi

"${client_python_env[@]}" MUJOCO_EGL_DEVICE_ID=0 "${CLIENT_PYTHON}" evaluation/robocasa/eval_policy_client.py \
  --env-name "${ENV_NAME}" \
  --split "${SPLIT}" \
  --num-episodes "${NUM_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --seed "${SEED}" \
  --host "${HOST}" \
  --port "${START_PORT}" \
  --save-video-dir "${SAVE_VIDEO_DIR}" \
  --server-save-root "${SAVE_ROOT}" \
  --save-json "${SAVE_JSON}" \
  --horizon-multiplier "${HORIZON_MULTIPLIER}" \
  "${debug_flag[@]}"

cleanup_server
SERVER_PID=""
SERVER_PGID=""

echo "[INFO] Done ENV ${ENV_NAME}: ${SAVE_JSON}"
