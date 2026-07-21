#!/usr/bin/env bash
# Stop all libero parallel eval workers (servers + clients) recorded by launch_parallel.sh.
#
# Usage:
#   bash evaluation/libero/stop_parallel.sh                 # default LOG_DIR=outputs/libero/logs
#   LOG_DIR=/path/to/logs bash evaluation/libero/stop_parallel.sh
set -u

LOG_DIR=${LOG_DIR:-outputs/libero/logs}
echo "[stop] LOG_DIR=${LOG_DIR}"

if [[ ! -d "$LOG_DIR" ]]; then
    echo "[stop] log dir not found, nothing to do."
    exit 0
fi

killed=0
shopt -s nullglob
for pid_file in "$LOG_DIR"/*.pid; do
    pid=$(cat "$pid_file" 2>/dev/null || true)
    [[ -z "$pid" ]] && continue
    # kill the whole process group (torch.distributed.run spawns children)
    if kill -0 "$pid" 2>/dev/null; then
        pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
        if [[ -n "$pgid" ]]; then
            kill -TERM -- "-$pgid" 2>/dev/null || true
        else
            kill -TERM "$pid" 2>/dev/null || true
        fi
        echo "[stop] sent TERM to pid=$pid (file=$pid_file)"
        killed=$((killed+1))
    fi
done
sleep 3
# Force kill any survivors
for pid_file in "$LOG_DIR"/*.pid; do
    pid=$(cat "$pid_file" 2>/dev/null || true)
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
        pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
        if [[ -n "$pgid" ]]; then
            kill -KILL -- "-$pgid" 2>/dev/null || true
        else
            kill -KILL "$pid" 2>/dev/null || true
        fi
        echo "[stop] sent KILL to pid=$pid"
    fi
done
echo "[stop] done. killed_initial=${killed}"
