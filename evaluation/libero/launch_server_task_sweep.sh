#!/usr/bin/env bash
set -euo pipefail

# Single task evaluation with server lifecycle (mirrors robocasa's
# launch_server_env_sweep.sh, adapted for libero's task-based API):
# 1) launch server; 2) wait until ready; 3) run client eval; 4) stop server.
#
# NOTE: This script starts/stops a server per invocation (model is reloaded
# each time). It is intended for single-task debugging / CI, NOT the main
# eval flow. The main flow uses launch_parallel.sh with long-lived servers.
#
# Env-var driven (align with robocasa style). Example:
#   CKPT_PATH=... GPU_ID=0 BENCHMARK=libero_10 TASK_ID=3 NUM_EPISODES=2 \
#   START_PORT=29056 MASTER_PORT=29061 CONFIG_NAME=libero_original \
#   bash evaluation/libero/launch_server_task_sweep.sh

CKPT_PATH=${CKPT_PATH:-}
START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
CONFIG_NAME=${CONFIG_NAME:-libero_original}
GPU_ID=${GPU_ID:-0}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-1200}

HOST=${HOST:-127.0.0.1}
SERVER_PYTHON=${SERVER_PYTHON:-python}
CLIENT_PYTHON=${CLIENT_PYTHON:-python}

# libero eval parameters
BENCHMARK=${BENCHMARK:-libero_10}
TASK_ID=${TASK_ID:-0}
NUM_EPISODES=${NUM_EPISODES:-50}
MAX_STEPS=${MAX_STEPS:-800}
SEED=${SEED:-0}
XLAB_DATA=${XLAB_DATA:-0}
DEBUG=${DEBUG:-0}
CAMERA_KEYS=${CAMERA_KEYS:-}
ATTN_MODE=${ATTN_MODE:-torch}

# Output paths (RUN_ROOT cascades into SAVE_ROOT / RESULT_DIR when unset)
RUN_ROOT=${RUN_ROOT:-}
SAVE_ROOT=${SAVE_ROOT:-${RUN_ROOT:+${RUN_ROOT}/save_root}}
SAVE_ROOT=${SAVE_ROOT:-visualization/libero_task_sweep}
RESULT_DIR=${RESULT_DIR:-${RUN_ROOT:+${RUN_ROOT}/result}}
RESULT_DIR=${RESULT_DIR:-evaluation/libero/results/task_sweep}
OUT_DIR=${OUT_DIR:-${RESULT_DIR}/out}
SERVER_LOG=${SERVER_LOG:-${RESULT_DIR}/server.log}

mkdir -p "${SAVE_ROOT}" "${RESULT_DIR}" "${OUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ---- libero client env setup (mirrors launch_client_single.sh) ----
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/libero_mplconfig_${USER:-user}}
mkdir -p "${MPLCONFIGDIR}"

# libero config isolation: avoid input() interactive prompt + dir conflicts
if [[ -z "${LIBERO_CONFIG_PATH:-}" ]]; then
    _libero_cfg="${HOME}/.libero"
    if [[ -e "$_libero_cfg" && ! -d "$_libero_cfg" ]]; then
        rm -f "$_libero_cfg"
    fi
    mkdir -p "$_libero_cfg"
    if [[ ! -f "$_libero_cfg/config.yaml" ]]; then
        _libero_pkg=$("${CLIENT_PYTHON}" -c "
import importlib.util, os
spec = importlib.util.find_spec('libero.libero')
if spec is None:
    spec = importlib.util.find_spec('libero')
if spec is None:
    exit(1)
root = os.path.dirname(os.path.abspath(spec.origin))
subpkg = os.path.join(root, 'libero')
if os.path.isdir(subpkg) and os.path.isfile(os.path.join(subpkg, '__init__.py')):
    root = subpkg
print(root)
" 2>/dev/null || echo "")
        if [[ -n "$_libero_pkg" && -d "$_libero_pkg" ]]; then
            cat > "$_libero_cfg/config.yaml" <<EOF
assets: ${_libero_pkg}/assets
bddl_files: ${_libero_pkg}/bddl_files
benchmark_root: ${_libero_pkg}
datasets: ${HOME}/libero_datasets
init_states: ${_libero_pkg}/init_files
EOF
            echo "[task_sweep] auto-generated libero config at $_libero_cfg/config.yaml"
        else
            echo "[task_sweep][WARN] could not locate libero package, config.yaml not created"
        fi
    fi
    export LIBERO_CONFIG_PATH="$_libero_cfg"
fi

# robosuite macros_private fix: avoid PermissionError on /tmp/robosuite.log
_robosuite_pkg=$("${CLIENT_PYTHON}" -c "
import importlib.util, os
spec = importlib.util.find_spec('robosuite')
if spec: print(os.path.dirname(spec.origin))
" 2>/dev/null || echo "")
if [[ -n "$_robosuite_pkg" && -d "$_robosuite_pkg" ]]; then
    _macros_private="${_robosuite_pkg}/macros_private.py"
    if [[ ! -f "$_macros_private" ]]; then
        cp "${_robosuite_pkg}/macros.py" "$_macros_private"
        echo "[task_sweep] created robosuite macros_private.py at $_macros_private"
    fi
fi

unset PYTHONHOME
unset PYTHONPATH
export PYTHONPATH="${REPO_ROOT}"

# Client runs without CUDA (libero env rendering uses osmesa, no GPU needed).
client_python_env=(env -u PYTHONHOME -u CUDA_VISIBLE_DEVICES)

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
if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" ]]; then
  debug_flag=(--debug)
fi

xlab_flag=()
if [[ "${XLAB_DATA}" == "1" || "${XLAB_DATA}" == "true" ]]; then
  xlab_flag=(--xlab-data)
fi

camera_flag=()
if [[ -n "${CAMERA_KEYS}" ]]; then
  camera_flag=(--camera-keys "${CAMERA_KEYS}")
fi

echo "[INFO] ===== Evaluate TASK ${TASK_ID} (benchmark=${BENCHMARK}) ====="
echo "[INFO] CKPT_PATH=${CKPT_PATH:-<from config>}"
echo "[INFO] CONFIG_NAME=${CONFIG_NAME}"
echo "[INFO] master_port=${MASTER_PORT}, ws_port=${START_PORT}, gpu=${GPU_ID}"
echo "[INFO] SERVER_PYTHON=${SERVER_PYTHON}"
echo "[INFO] CLIENT_PYTHON=${CLIENT_PYTHON}"
echo "[INFO] ATTN_MODE=${ATTN_MODE}"
echo "[INFO] NUM_EPISODES=${NUM_EPISODES}, MAX_STEPS=${MAX_STEPS}, SEED=${SEED}"
echo "[INFO] XLAB_DATA=${XLAB_DATA}, OUT_DIR=${OUT_DIR}"

# ---- Launch server (via libero launch_server.sh → _lib/serve.sh) ----
launch_envs=(
  "START_PORT=${START_PORT}"
  "MASTER_PORT=${MASTER_PORT}"
  "CONFIG_NAME=${CONFIG_NAME}"
  "SAVE_ROOT=${SAVE_ROOT}"
  "SAVE_PRED_VIDEO=1"
  "PRED_VIDEO_FPS=10"
  "PYTHON_BIN=${SERVER_PYTHON}"
  "CUDA_VISIBLE_DEVICES=${GPU_ID}"
  "ATTN_MODE=${ATTN_MODE}"
)
if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
  launch_envs+=("EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS}")
fi

setsid env -u PYTHONHOME -u PYTHONPATH "${launch_envs[@]}" \
    bash evaluation/libero/launch_server.sh >"${SERVER_LOG}" 2>&1 &
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

# ---- Run client (single-task mode via --task-id) ----
"${client_python_env[@]}" "${CLIENT_PYTHON}" evaluation/libero/eval_policy_client.py \
  --libero-benchmark "${BENCHMARK}" \
  --task-id "${TASK_ID}" \
  --test-num "${NUM_EPISODES}" \
  --seed "${SEED}" \
  --host "${HOST}" \
  --port "${START_PORT}" \
  --out-dir "${OUT_DIR}" \
  --server-save-root "${SAVE_ROOT}" \
  "${camera_flag[@]}" \
  "${xlab_flag[@]}" \
  "${debug_flag[@]}"

cleanup_server
SERVER_PID=""
SERVER_PGID=""

echo "[INFO] Done TASK ${TASK_ID} (benchmark=${BENCHMARK}): results under ${OUT_DIR}"
