#!/usr/bin/env bash
# One-click multi-GPU parallel LIBERO evaluation for Abot-m0.5.
#
# Pipeline:
#   - Spawns NUM_GPUS servers, each pinned to one GPU on its own port.
#   - Waits until every server's websocket is reachable.
#   - Spawns NUM_GPUS clients, each evaluating an even slice of tasks.
#   - Streams all logs into LOG_DIR, with one file per worker.
#
# Usage:
#   bash evaluation/libero/launch_parallel.sh                  # libero_10, 50 episodes / task
#   BENCHMARK=libero_goal bash evaluation/libero/launch_parallel.sh
#   NUM_GPUS=4 TEST_NUM=10 bash evaluation/libero/launch_parallel.sh
#
# Env overrides (with defaults):
#   NUM_GPUS    = 8
#   GPUS        = 0,1,2,...,N-1    # comma-separated GPU IDs (overrides NUM_GPUS count)
#   BENCHMARK   = libero_10
#   TEST_NUM    = 50
#   NUM_TASKS   = 10                # libero_{10,goal,spatial,object} all have 10 tasks; libero_90 has 90
#   BASE_SERVER_PORT = 29056        # ports used: 29056 .. 29056+NUM_GPUS-1
#   BASE_MASTER_PORT = 29161        # ports used: 29161 .. 29161+NUM_GPUS-1 (torch.distributed master_port)
#   OUT_ROOT    = outputs/libero
#   LOG_DIR     = $OUT_ROOT/logs
#   SAVE_ROOT_BASE = visualization  # per-gpu subdir under this
#   CONFIG_NAME = libero_original
#   SERVER_READY_TIMEOUT = 1800     # seconds to wait for each server to be reachable
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
BENCHMARK=${BENCHMARK:-libero_10}
TEST_NUM=${TEST_NUM:-50}
NUM_TASKS=${NUM_TASKS:-10}
BASE_SERVER_PORT=${BASE_SERVER_PORT:-29056}
BASE_MASTER_PORT=${BASE_MASTER_PORT:-29161}
OUT_ROOT=${OUT_ROOT:-outputs/libero}
LOG_DIR=${LOG_DIR:-${OUT_ROOT}/logs}
SAVE_ROOT_BASE=${SAVE_ROOT_BASE:-visualization}
CONFIG_NAME=${CONFIG_NAME:-libero_original}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-1800}

# ---- Parse GPU selection ----
# Priority: GPUS > CUDA_VISIBLE_DEVICES > default 0..NUM_GPUS-1
# All three produce GPU_IDS[] (physical GPU IDs passed to launch_server_single.sh)
if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -ra GPU_IDS <<< "$GPUS"
    NUM_GPUS=${#GPU_IDS[@]}
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#GPU_IDS[@]}
    echo "[parallel] derived GPU_IDS from CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
else
    GPU_IDS=()
    for ((g=0; g<NUM_GPUS; g++)); do
        GPU_IDS+=($g)
    done
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$OUT_ROOT" "$LOG_DIR"
PID_FILE="$LOG_DIR/pids.txt"
: > "$PID_FILE"

echo "[parallel] NUM_GPUS=$NUM_GPUS BENCHMARK=$BENCHMARK TEST_NUM=$TEST_NUM NUM_TASKS=$NUM_TASKS"
echo "[parallel] BASE_SERVER_PORT=$BASE_SERVER_PORT BASE_MASTER_PORT=$BASE_MASTER_PORT"
echo "[parallel] OUT_ROOT=$OUT_ROOT LOG_DIR=$LOG_DIR"

# ---------- 1) Generate episode-level balanced plan (one json per GPU) ----------
PLAN_DIR="${LOG_DIR}/plan"
mkdir -p "$PLAN_DIR"
echo "[parallel] planning jobs with episode-level balancing..."
${PY_BIN_SERVER:-python} "$SCRIPT_DIR/plan_jobs.py" \
    --benchmark "$BENCHMARK" \
    --num-tasks "$NUM_TASKS" \
    --test-num "$TEST_NUM" \
    --num-gpus "$NUM_GPUS" \
    --out-root "$OUT_ROOT" \
    --plan-dir "$PLAN_DIR" \
    --skip-completed

TOTAL_PENDING=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['total_pending_episodes'])" "$PLAN_DIR/plan_summary.json")
echo "[parallel] total pending episodes: ${TOTAL_PENDING}"
if [[ "$TOTAL_PENDING" -eq 0 ]]; then
    echo "[parallel] benchmark $BENCHMARK already fully completed under $OUT_ROOT, nothing to do."
    echo "[parallel] summarizing existing results..."
    python "$SCRIPT_DIR/aggregate_results.py" --run-root "$OUT_ROOT" --benchmark "$BENCHMARK" || true
    exit 0
fi

# ---------- 2) Launch servers ----------
declare -a SERVER_PIDS=()
declare -a SERVER_PORTS=()
echo "[parallel] launching ${NUM_GPUS} servers (GPUs: ${GPU_IDS[*]})..."
for ((i=0; i<NUM_GPUS; i++)); do
    gpu_id=${GPU_IDS[$i]}
    server_port=$(( BASE_SERVER_PORT + i ))
    master_port=$(( BASE_MASTER_PORT + i ))
    save_root="${SAVE_ROOT_BASE}/gpu${i}"
    log_file="${LOG_DIR}/server_gpu${i}.log"
    SERVER_PORTS[$i]=${server_port}

    echo "  -> slot=${i} gpu=${gpu_id} server_port=${server_port} master_port=${master_port} log=${log_file}"
    (
        nohup bash "$SCRIPT_DIR/launch_server_single.sh" \
            "${gpu_id}" "${server_port}" "${master_port}" "${save_root}" "${CONFIG_NAME}" \
            > "${log_file}" 2>&1 &
        echo $! > "${LOG_DIR}/server_gpu${i}.pid"
    )
    pid=$(cat "${LOG_DIR}/server_gpu${i}.pid")
    SERVER_PIDS[$i]=${pid}
    echo "server gpu${i}(id=${gpu_id}) pid=${pid} port=${server_port}" >> "$PID_FILE"
done

# ---------- 3) Wait until all server ports are reachable ----------
check_port() {
    # Returns 0 if a TCP listener is up on 127.0.0.1:$1
    local p=$1
    if command -v ss >/dev/null 2>&1; then
        ss -ltn "( sport = :${p} )" 2>/dev/null | grep -q ":${p}"
    elif command -v netstat >/dev/null 2>&1; then
        netstat -ltn 2>/dev/null | grep -q ":${p} "
    else
        python -c "import socket,sys; s=socket.socket(); s.settimeout(1)
try:
    s.connect(('127.0.0.1', ${p}))
    print('ok')
except Exception:
    sys.exit(1)
" >/dev/null 2>&1
    fi
}

echo "[parallel] waiting up to ${SERVER_READY_TIMEOUT}s for all servers to be ready..."
deadline=$(( $(date +%s) + SERVER_READY_TIMEOUT ))
for ((i=0; i<NUM_GPUS; i++)); do
    p=${SERVER_PORTS[$i]}
    while true; do
        if check_port "$p"; then
            echo "  -> gpu${i} server port ${p} is up"
            break
        fi
        # Detect early crash
        if ! kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null; then
            echo "[parallel][ERROR] server gpu${i} (pid=${SERVER_PIDS[$i]}) died, see ${LOG_DIR}/server_gpu${i}.log"
            exit 1
        fi
        if (( $(date +%s) > deadline )); then
            echo "[parallel][ERROR] timeout waiting for server gpu${i} port ${p}"
            exit 1
        fi
        sleep 5
    done
done
echo "[parallel] all servers up."

# ---------- 4) Launch clients (plan-file mode, episode-balanced) ----------
declare -a CLIENT_PIDS=()
echo "[parallel] launching up to ${NUM_GPUS} clients (plan-file mode)..."
for ((i=0; i<NUM_GPUS; i++)); do
    server_port=${SERVER_PORTS[$i]}
    plan_file="${PLAN_DIR}/gpu${i}.json"
    if [[ ! -f "$plan_file" ]]; then
        echo "  -> gpu${i} no plan file, skip"
        continue
    fi
    ep_count=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('total_episodes', 0))" "$plan_file")
    if [[ "$ep_count" -eq 0 ]]; then
        echo "  -> gpu${i} has 0 pending episodes, skip client"
        continue
    fi
    out_dir="${OUT_ROOT}/gpu${i}"
    log_file="${LOG_DIR}/client_gpu${i}.log"
    mkdir -p "${out_dir}"

    echo "  -> gpu=${i} server_port=${server_port} pending_eps=${ep_count} plan=${plan_file} log=${log_file}"
    # Background directly in the current shell so $! is a real child we can wait on.
    bash "$SCRIPT_DIR/launch_client_single.sh" \
        --plan-file "${plan_file}" \
        --port "${server_port}" \
        --benchmark "${BENCHMARK}" \
        --out-dir "${out_dir}" \
        --skip-completed \
        > "${log_file}" 2>&1 &
    pid=$!
    echo "${pid}" > "${LOG_DIR}/client_gpu${i}.pid"
    CLIENT_PIDS+=("${pid}")
    echo "client gpu${i} pid=${pid} port=${server_port} pending_eps=${ep_count}" >> "$PID_FILE"
done

echo "[parallel] all clients launched. pids:"
cat "$PID_FILE"
echo
echo "[parallel] Tail logs example:"
echo "  tail -f ${LOG_DIR}/client_gpu0.log"
echo "  tail -f ${LOG_DIR}/server_gpu0.log"
echo

# ---------- 5) Wait for all clients to finish ----------
echo "[parallel] waiting for clients to finish..."
exit_code=0
for pid in "${CLIENT_PIDS[@]}"; do
    if ! wait "${pid}"; then
        echo "[parallel][WARN] client pid=${pid} exited non-zero"
        exit_code=1
    fi
done

echo "[parallel] all clients finished (exit_code=${exit_code})."
echo "[parallel] stopping servers..."
bash "$SCRIPT_DIR/stop_parallel.sh" || true

echo "[parallel] summarizing results..."
${PY_BIN_SERVER:-python} "$SCRIPT_DIR/aggregate_results.py" \
    --run-root "${OUT_ROOT}" \
    --benchmark "${BENCHMARK}" \
    --summary-json "${OUT_ROOT}/summary.json" || true

exit ${exit_code}
