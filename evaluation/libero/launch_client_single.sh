#!/usr/bin/env bash
# Launch a SINGLE libero client connecting to one server.
#
# Two modes:
#
#   1) Plan-file mode (recommended, used by launch_parallel.sh):
#        bash launch_client_single.sh --plan-file <PLAN_JSON> --port <SERVER_PORT> \
#                                     [--benchmark libero_10] [--skip-completed]
#
#   2) Legacy positional mode (kept for backward compat):
#        bash launch_client_single.sh <SERVER_PORT> <TASK_START> <TASK_END> \
#                                     <OUT_DIR> [BENCHMARK] [TEST_NUM]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/libero_mplconfig_${USER:-user}}
mkdir -p "$MPLCONFIGDIR"

# ---- libero config isolation: avoid input() interactive prompt + directory conflicts ----
# libero/__init__.py prompts input() when config.yaml is missing; containers have no stdin -> EOFError.
# Fix: ensure config.yaml exists ahead of time to skip interactive config.
if [[ -z "${LIBERO_CONFIG_PATH:-}" ]]; then
    _libero_cfg="${HOME}/.libero"
    if [[ -e "$_libero_cfg" && ! -d "$_libero_cfg" ]]; then
        rm -f "$_libero_cfg"
    fi
    mkdir -p "$_libero_cfg"
    # If config.yaml does not exist, pre-write a default config (pointing to the libero in site-packages)
    if [[ ! -f "$_libero_cfg/config.yaml" ]]; then
        # Use importlib.util.find_spec to locate libero (does not run __init__.py, avoiding input() deadlock)
        _libero_pkg=$(${PY_BIN:-python} -c "
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
            echo "[launch_client_single] auto-generated libero config at $_libero_cfg/config.yaml"
        else
            echo "[launch_client_single][WARN] could not locate libero package, config.yaml not created"
        fi
    fi
    export LIBERO_CONFIG_PATH="$_libero_cfg"
fi

# ---- robosuite macros_private fix ----
# robosuite/__init__.py tries to write /tmp/robosuite.log when macros_private.py is missing;
# if that file is locked by another user (root), it raises PermissionError.
# Creating macros_private.py skips the problematic except branch.
_robosuite_pkg=$(${PY_BIN_CLIENT:-python} -c "
import importlib.util, os
spec = importlib.util.find_spec('robosuite')
if spec: print(os.path.dirname(spec.origin))
" 2>/dev/null || echo "")
if [[ -n "$_robosuite_pkg" && -d "$_robosuite_pkg" ]]; then
    _macros_private="${_robosuite_pkg}/macros_private.py"
    if [[ ! -f "$_macros_private" ]]; then
        cp "${_robosuite_pkg}/macros.py" "$_macros_private"
        echo "[launch_client_single] created robosuite macros_private.py at $_macros_private"
    fi
fi

# ---- Detect plan-file mode ----
PLAN_FILE=""
SERVER_PORT=""
BENCHMARK=""
SKIP_COMPLETED=""
OUT_DIR=""
# Save raw args because we also need positional fallback
RAW_ARGS=("$@")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan-file) PLAN_FILE="$2"; shift 2 ;;
        --port) SERVER_PORT="$2"; shift 2 ;;
        --benchmark) BENCHMARK="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --skip-completed) SKIP_COMPLETED="--skip-completed"; shift ;;
        *) break ;;
    esac
done

if [[ -n "$PLAN_FILE" ]]; then
    : "${SERVER_PORT:?--port required in plan-file mode}"
    : "${BENCHMARK:=libero_10}"
    # In plan-file mode the client reads out_dir from plan json itself unless
    # an explicit --out-dir override is given; eval_policy_client.py only uses CLI out-dir,
    # so we read it from plan if not provided.
    if [[ -z "$OUT_DIR" ]]; then
        OUT_DIR=$(${PY_BIN_CLIENT:-python} -c "import json,sys; print(json.load(open(sys.argv[1]))['out_dir'])" "$PLAN_FILE")
    fi
    mkdir -p "$OUT_DIR"
    exec ${PY_BIN_CLIENT:-python} evaluation/libero/eval_policy_client.py \
        --libero-benchmark "${BENCHMARK}" \
        --port "${SERVER_PORT}" \
        --plan-file "${PLAN_FILE}" \
        --out-dir "${OUT_DIR}" \
        ${SKIP_COMPLETED}
fi

# ---- Legacy positional mode ----
set -- "${RAW_ARGS[@]}"
SERVER_PORT=${1:-29056}
TASK_START=${2:-0}
TASK_END=${3:-10}
OUT_DIR=${4:-outputs/libero}
BENCHMARK=${5:-libero_10}
TEST_NUM=${6:-50}

${PY_BIN_CLIENT:-python} evaluation/libero/eval_policy_client.py \
    --libero-benchmark "${BENCHMARK}" \
    --port "${SERVER_PORT}" \
    --test-num "${TEST_NUM}" \
    --task-range "${TASK_START} ${TASK_END}" \
    --out-dir "${OUT_DIR}"
