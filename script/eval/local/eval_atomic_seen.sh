#!/usr/bin/env bash
set -euo pipefail

# Local RoboCasa dynamic eval:
# - a shared task pool of (env_name, episode_idx) units;
# - local worker processes consume contiguous same-env batches;
# - each batch runs evaluation/robocasa/launch_server_env_sweep.sh, which
#   starts one server, waits for readiness, runs one client batch, then stops
#   the server.

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SRC_REPO="${_SCRIPT_DIR}"
while [ "${_SRC_REPO}" != "/" ] && [ ! -d "${_SRC_REPO}/wam" ]; do
    _SRC_REPO="$(dirname "${_SRC_REPO}")"
done
if [ ! -d "${_SRC_REPO}/wam" ]; then
    echo "Error: cannot locate repo root (wam/ not found) from ${_SCRIPT_DIR}" >&2
    exit 1
fi
cd "${_SRC_REPO}"

# ========== local worker / GPU layout ==========
GPU_IDS=${GPU_IDS:-0}
IFS=',' read -r -a _GPU_ID_LIST <<< "${GPU_IDS}"
if [ "${#_GPU_ID_LIST[@]}" -eq 0 ]; then
    echo "ERROR: GPU_IDS is empty" >&2
    exit 1
fi
WORKER_COUNT=${WORKER_COUNT:-${#_GPU_ID_LIST[@]}}
DRY_RUN=${DRY_RUN:-0}
SKIP_PORT_CHECK=${SKIP_PORT_CHECK:-${DRY_RUN}}
AUTO_AGGREGATE=${AUTO_AGGREGATE:-1}
BACKGROUND=${BACKGROUND:-0}

# ========== evaluation parameters ==========
HORIZON_JSON=${HORIZON_JSON:-${_SCRIPT_DIR}/target_atomic_seen_task_horizons.json}
SEED=${SEED:-0}
CKPT_PATH=${CKPT_PATH:?CKPT_PATH is required, e.g. /path/to/checkpoint_step}

START_PORT=${START_PORT:-29056}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29100}
PORT_OFFSET_MOD=${PORT_OFFSET_MOD:-200}
CONFIG_NAME=${CONFIG_NAME:-robocasa_train_test_atomic_target}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-3600}
HOST=${HOST:-127.0.0.1}
WRAPPER_PYTHON=${WRAPPER_PYTHON:-python}
SERVER_PYTHON=${SERVER_PYTHON:-python}
ROBOCASA_ENV_ROOT=${ROBOCASA_ENV_ROOT:-}
if [ -n "${ROBOCASA_ENV_ROOT}" ]; then
    CLIENT_PYTHON=${CLIENT_PYTHON:-${ROBOCASA_ENV_ROOT}/bin/python}
else
    CLIENT_PYTHON=${CLIENT_PYTHON:-python}
fi
ROBOCASA_CLIENT_PYTHONPATH=${ROBOCASA_CLIENT_PYTHONPATH:-"${_SRC_REPO}"}
export ROBOCASA_CLIENT_PYTHONPATH
NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/robocasa_numba_cache_${USER:-user}}
export NUMBA_CACHE_DIR
SPLIT=${SPLIT:-pretrain}
NUM_EPISODES=${NUM_EPISODES:-20}
MAX_STEPS=${MAX_STEPS:-500}
DEBUG_ACTIONS=${DEBUG_ACTIONS:-0}
HORIZON_MULTIPLIER=${HORIZON_MULTIPLIER:-1}
BATCH_MAX=${BATCH_MAX:-10}
INIT_WAIT_SECONDS=${INIT_WAIT_SECONDS:-600}
MAX_RETRIES=${MAX_RETRIES:-2}
ATTN_MODE=${ATTN_MODE:-torch}
RESUME_RUN_ROOT=${RESUME_RUN_ROOT:-}
SKIP_MESA_INSTALL=${SKIP_MESA_INSTALL:-1}
WAN22_PRETRAINED_MODEL_NAME_OR_PATH=${WAN22_PRETRAINED_MODEL_NAME_OR_PATH:-}
wan22_pretrained_env=()
if [ -n "${WAN22_PRETRAINED_MODEL_NAME_OR_PATH}" ]; then
    wan22_pretrained_env=("WAN22_PRETRAINED_PATH=${WAN22_PRETRAINED_MODEL_NAME_OR_PATH}")
fi

RUN_ROOT_BASE=${RUN_ROOT_BASE:-outputs/robocasa_eval_env_sweep_local_dynamic_mot_pretrain_atomic_seen}
if [ -n "${CKPT_PATH}" ]; then
    CKPT_PATH_NORM="${CKPT_PATH%/}"
    CKPT_STEP_NAME="$(basename "${CKPT_PATH_NORM}")"
    CKPT_EXP_NAME="$(basename "$(dirname "$(dirname "${CKPT_PATH_NORM}")")")"
else
    CKPT_STEP_NAME="default_step"
    CKPT_EXP_NAME="default_ckpt"
fi
CKPT_EXP_TAG="$(echo "${CKPT_EXP_NAME}" | sed 's/[^a-zA-Z0-9._-]/_/g' | cut -c1-24)"
CKPT_STEP_TAG="$(echo "${CKPT_STEP_NAME}" | sed 's/[^a-zA-Z0-9._-]/_/g' | cut -c1-18)"
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-"${RUN_ROOT_BASE}/${CKPT_EXP_TAG}_${CKPT_STEP_TAG}/${RUN_TAG}"}
SHARED_POOL_ROOT=${SHARED_POOL_ROOT:-"${RUN_ROOT}/shared_pool"}
LOCAL_LAUNCH_LOG_DIR=${LOCAL_LAUNCH_LOG_DIR:-"${RUN_ROOT}/local_launcher_logs"}
BACKGROUND_LOG=${BACKGROUND_LOG:-"${LOCAL_LAUNCH_LOG_DIR}/launcher.stdout.log"}

if [ "${BACKGROUND}" = "1" ] && [ "${_LOCAL_EVAL_BACKGROUND_CHILD:-0}" != "1" ]; then
    mkdir -p "${RUN_ROOT}" "${SHARED_POOL_ROOT}" "${LOCAL_LAUNCH_LOG_DIR}"
    self_path="${_SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
    echo "[launcher] BACKGROUND=1, launching detached job"
    echo "[launcher] run_root        : ${RUN_ROOT}"
    echo "[launcher] launcher_log    : ${BACKGROUND_LOG}"
    nohup setsid env \
        "${wan22_pretrained_env[@]}" \
        _LOCAL_EVAL_BACKGROUND_CHILD=1 \
        BACKGROUND=0 \
        RUN_TAG="${RUN_TAG}" \
        RUN_ROOT="${RUN_ROOT}" \
        SHARED_POOL_ROOT="${SHARED_POOL_ROOT}" \
        LOCAL_LAUNCH_LOG_DIR="${LOCAL_LAUNCH_LOG_DIR}" \
        BACKGROUND_LOG="${BACKGROUND_LOG}" \
        bash "${self_path}" "$@" >"${BACKGROUND_LOG}" 2>&1 &
    bg_pid=$!
    echo "${bg_pid}" >"${LOCAL_LAUNCH_LOG_DIR}/launcher.pid"
    echo "[launcher] pid             : ${bg_pid}"
    echo "[launcher] pid_file        : ${LOCAL_LAUNCH_LOG_DIR}/launcher.pid"
    exit 0
fi

RUN_TAG_NUMERIC="$(printf '%s' "${RUN_TAG}" | tr -cd '0-9')"
RUN_TAG_SUFFIX="${RUN_TAG_NUMERIC: -6}"
if [ -z "${RUN_TAG_SUFFIX}" ]; then
    RUN_TAG_SUFFIX=0
fi
PORT_OFFSET=$((10#${RUN_TAG_SUFFIX} % PORT_OFFSET_MOD))
START_PORT_EFFECTIVE=$((START_PORT + PORT_OFFSET))
MASTER_PORT_BASE_EFFECTIVE=$((MASTER_PORT_BASE + PORT_OFFSET))

WRAPPER_SCRIPT="script/robocasa_eval_wrapper.py"

if [ ! -f "${WRAPPER_SCRIPT}" ]; then
    echo "ERROR: wrapper not found: ${WRAPPER_SCRIPT}" >&2
    exit 1
fi
if [ ! -f "evaluation/robocasa/launch_server_env_sweep.sh" ]; then
    echo "ERROR: launch_server_env_sweep.sh not found" >&2
    exit 1
fi
if [ -n "${HORIZON_JSON}" ] && [ ! -f "${HORIZON_JSON}" ]; then
    echo "ERROR: horizon json not found: ${HORIZON_JSON}" >&2
    exit 1
fi
if [ -n "${CKPT_PATH}" ] && [ ! -d "${CKPT_PATH%/}" ]; then
    echo "ERROR: checkpoint directory not found: ${CKPT_PATH}" >&2
    exit 1
fi

check_port_available() {
    local port=$1
    python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("0.0.0.0", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

if [ "${SKIP_PORT_CHECK}" = "1" ]; then
    echo "Skipping local port availability check (SKIP_PORT_CHECK=1)."
else
    echo "Checking local websocket/master ports..."
    for ((rank = 0; rank < WORKER_COUNT; rank++)); do
        ws_port=$((START_PORT_EFFECTIVE + rank))
        master_port=$((MASTER_PORT_BASE_EFFECTIVE + rank))
        if ! check_port_available "${ws_port}"; then
            echo "ERROR: websocket port ${ws_port} is already in use" >&2
            exit 1
        fi
        if ! check_port_available "${master_port}"; then
            echo "ERROR: master port ${master_port} is already in use" >&2
            exit 1
        fi
    done
fi

wrapper_args=(
    --worker-count "${WORKER_COUNT}"
    --seed "${SEED}"
    --ckpt-path "${CKPT_PATH}"
    --start-port "${START_PORT_EFFECTIVE}"
    --master-port-base "${MASTER_PORT_BASE_EFFECTIVE}"
    --config-name "${CONFIG_NAME}"
    --gpu-id "0"
    --wait-timeout "${WAIT_TIMEOUT}"
    --host "${HOST}"
    --server-python "${SERVER_PYTHON}"
    --client-python "${CLIENT_PYTHON}"
    --split "${SPLIT}"
    --num-episodes "${NUM_EPISODES}"
    --max-steps "${MAX_STEPS}"
    --debug-actions "${DEBUG_ACTIONS}"
    --horizon-json "${HORIZON_JSON}"
    --run-root "${RUN_ROOT}"
    --shared-pool-root "${SHARED_POOL_ROOT}"
    --batch-max "${BATCH_MAX}"
    --init-wait-seconds "${INIT_WAIT_SECONDS}"
    --max-retries "${MAX_RETRIES}"
    --horizon-multiplier "${HORIZON_MULTIPLIER}"
    --resume-from-self 1
)
if [ -n "${RESUME_RUN_ROOT}" ]; then
    if [ ! -d "${RESUME_RUN_ROOT}" ]; then
        echo "WARNING: RESUME_RUN_ROOT does not exist, ignoring: ${RESUME_RUN_ROOT}" >&2
    else
        wrapper_args+=(--resume-run-root "${RESUME_RUN_ROOT}")
    fi
fi

echo "=========================================="
echo "Local RoboCasa Dynamic Eval (ATOMIC SEEN, MoT, PRETRAIN split)"
echo "=========================================="
echo "worker_count       : ${WORKER_COUNT}"
echo "gpu_ids            : ${GPU_IDS}"
echo "horizon_json       : ${HORIZON_JSON:-<none>}"
echo "horizon_multiplier : ${HORIZON_MULTIPLIER}"
echo "split              : ${SPLIT}"
echo "num_episodes       : ${NUM_EPISODES}"
echo "batch_max          : ${BATCH_MAX}"
echo "init_wait_seconds  : ${INIT_WAIT_SECONDS}"
echo "max_retries        : ${MAX_RETRIES}"
echo "attn_mode          : ${ATTN_MODE}"
echo "ckpt_exp_name      : ${CKPT_EXP_NAME}"
echo "ckpt_step_name     : ${CKPT_STEP_NAME}"
echo "run_tag            : ${RUN_TAG}"
echo "port_offset        : ${PORT_OFFSET} (mod=${PORT_OFFSET_MOD})"
echo "start_port         : ${START_PORT} -> ${START_PORT_EFFECTIVE}"
echo "master_port_base   : ${MASTER_PORT_BASE} -> ${MASTER_PORT_BASE_EFFECTIVE}"
echo "run_root           : ${RUN_ROOT}"
echo "shared_pool_root   : ${SHARED_POOL_ROOT}"
echo "launcher_logs      : ${LOCAL_LAUNCH_LOG_DIR}"
echo "wrapper_python     : ${WRAPPER_PYTHON}"
echo "server_python      : ${SERVER_PYTHON}"
echo "client_python      : ${CLIENT_PYTHON}"
echo "client_pythonpath  : ${ROBOCASA_CLIENT_PYTHONPATH}"
echo "numba_cache_dir    : ${NUMBA_CACHE_DIR}"
echo "wan22_pretrained   : ${WAN22_PRETRAINED_MODEL_NAME_OR_PATH:-<config default>}"
echo "skip_mesa_install  : ${SKIP_MESA_INSTALL}"
echo "skip_port_check    : ${SKIP_PORT_CHECK}"
echo "background         : ${BACKGROUND}"
echo "dry_run            : ${DRY_RUN}"
echo "=========================================="

printf 'Wrapper command template:\n  %q %q' "${WRAPPER_PYTHON}" "${WRAPPER_SCRIPT}"
printf ' %q' "${wrapper_args[@]}"
printf '\n\n'

if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN=1, not launching local workers."
    exit 0
fi

mkdir -p "${RUN_ROOT}" "${SHARED_POOL_ROOT}" "${LOCAL_LAUNCH_LOG_DIR}"

worker_pids=()
cleanup_done=0
cleanup_workers() {
    if [ "${cleanup_done}" = "1" ]; then
        return
    fi
    cleanup_done=1
    if [ "${#worker_pids[@]}" -eq 0 ]; then
        return
    fi
    echo "Stopping local worker process groups: ${worker_pids[*]}" >&2
    for pid in "${worker_pids[@]}"; do
        kill -TERM -- "-${pid}" >/dev/null 2>&1 || kill -TERM "${pid}" >/dev/null 2>&1 || true
    done
    for _ in {1..10}; do
        any_alive=0
        for pid in "${worker_pids[@]}"; do
            if kill -0 "${pid}" >/dev/null 2>&1; then
                any_alive=1
                break
            fi
        done
        if [ "${any_alive}" = "0" ]; then
            break
        fi
        sleep 1
    done
    for pid in "${worker_pids[@]}"; do
        kill -KILL -- "-${pid}" >/dev/null 2>&1 || kill -KILL "${pid}" >/dev/null 2>&1 || true
        wait "${pid}" 2>/dev/null || true
    done
}
on_interrupt() {
    trap - INT TERM EXIT
    cleanup_workers
    exit 130
}
on_exit() {
    rc=$?
    if [ "${rc}" -ne 0 ]; then
        cleanup_workers
    fi
}
trap on_interrupt INT TERM
trap on_exit EXIT

for ((rank = 0; rank < WORKER_COUNT; rank++)); do
    gpu_index=$((rank % ${#_GPU_ID_LIST[@]}))
    gpu_id="${_GPU_ID_LIST[$gpu_index]}"
    worker_log="${LOCAL_LAUNCH_LOG_DIR}/worker_${rank}.stdout.log"
    echo "[launcher] start worker rank=${rank} gpu=${gpu_id} log=${worker_log}"
    setsid env \
        "${wan22_pretrained_env[@]}" \
        RANK="${rank}" \
        WORKER_RANK="${rank}" \
        LOCAL_RANK="${gpu_id}" \
        LOCAL_PROCESS_RANK="${gpu_id}" \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        TOKENIZERS_PARALLELISM=false \
        ATTN_MODE="${ATTN_MODE}" \
        SKIP_MESA_INSTALL="${SKIP_MESA_INSTALL}" \
        "${WRAPPER_PYTHON}" "${WRAPPER_SCRIPT}" "${wrapper_args[@]}" \
        >"${worker_log}" 2>&1 &
    worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
trap - INT TERM EXIT

if [ "${AUTO_AGGREGATE}" = "1" ]; then
    echo "[launcher] aggregate per-batch JSONs"
    python evaluation/robocasa/aggregate_results.py \
        --run-root "${RUN_ROOT}" \
        --summary-json "${RUN_ROOT}/summary.json" || status=1
fi

echo "=========================================="
echo "Local dynamic eval finished"
echo "status          : ${status}"
echo "run_root        : ${RUN_ROOT}"
echo "summary_json    : ${RUN_ROOT}/summary.json"
echo "launcher_logs   : ${LOCAL_LAUNCH_LOG_DIR}"
echo "progress_tsv    : ${SHARED_POOL_ROOT}/progress.tsv"
echo "=========================================="
exit "${status}"
