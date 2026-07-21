#!/usr/bin/env bash
# One-click 8-GPU parallel LIBERO evaluation using YOUR fine-tuned checkpoint.
#
# Pipeline:
#   1) Assemble an inference-ready ckpt dir (your transformer weights + base ckpt's vae/text_encoder/tokenizer)
#      -> if POSTTRAIN_CKPT is set and the directory exists, skip this step (use the pre-assembled ckpt)
#   2) Export POSTTRAIN_CKPT so libero_original_cfg picks up the new transformer path
#   3) Delegate to evaluation/libero/launch_parallel.sh for the actual 8-GPU eval
#
# Usage:
#   # Option 1: directly specify a pre-assembled ckpt (recommended, simplest)
#   POSTTRAIN_CKPT=/path/to/assembled_ckpt bash evaluation/libero/launch_eval_my_ckpt.sh
#
#   # Option 2: auto-assemble from training ckpt (requires TRAIN_CKPT_DIR + REF_CKPT_DIR + EVAL_CKPT_ROOT)
#   TRAIN_CKPT_DIR=/path/to/train_out/checkpoints/checkpoint_step_XXX \
#       bash evaluation/libero/launch_eval_my_ckpt.sh
#
# Env overrides (defaults shown):
#   CKPT_STEP        = 900
#   POSTTRAIN_CKPT   = (optional) path to a pre-assembled inference-ready ckpt; if set, skips the prepare step
#   TRAIN_CKPT_DIR   = <repo>/train_out/checkpoints/checkpoint_step_${CKPT_STEP}
#   REF_CKPT_DIR     = (required if POSTTRAIN_CKPT not set) path to reference ckpt with vae/text_encoder/tokenizer
#   EVAL_CKPT_ROOT   = (required if POSTTRAIN_CKPT not set) directory for assembled eval checkpoints
#   BENCHMARK        = libero_10
#   NUM_GPUS         = 8
#   TEST_NUM         = 50      # episodes per task
#   NUM_TASKS        = 10
#   OUT_ROOT         = outputs/libero_my_ckpt_step${CKPT_STEP}/${BENCHMARK}
#   LOG_DIR          = ${OUT_ROOT}/logs
#   BASE_SERVER_PORT = 29056
#   BASE_MASTER_PORT = 29161
#   USE_MOT          = <unset>   # 0/1: load ckpt as legacy / MoT. Default: follow
#                                 # config.use_mot (which defaults to True). MUST
#                                 # match the architecture used during training.
#   ATTN_MODE        = <unset>   # 'torch' / 'flashattn' / 'flex' /
#                                 # 'decomposed_varlen'. Default: torch (from
#                                 # libero_original_cfg via shared_config).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CKPT_STEP=${CKPT_STEP:-900}

BENCHMARK=${BENCHMARK:-libero_10}
NUM_GPUS=${NUM_GPUS:-8}
TEST_NUM=${TEST_NUM:-50}
NUM_TASKS=${NUM_TASKS:-10}
BASE_SERVER_PORT=${BASE_SERVER_PORT:-29056}
BASE_MASTER_PORT=${BASE_MASTER_PORT:-29161}

# ---- Checkpoint selection ----
# Priority: POSTTRAIN_CKPT (direct path) > TRAIN_CKPT_DIR (assemble from parts)
if [[ -n "${POSTTRAIN_CKPT:-}" ]]; then
    # POSTTRAIN_CKPT already set: skip prepare_eval_ckpt.sh
    [[ -d "$POSTTRAIN_CKPT" ]] \
        || { echo "[ERROR] POSTTRAIN_CKPT not found: $POSTTRAIN_CKPT"; exit 1; }
    echo "[launch_eval_my_ckpt] using pre-assembled POSTTRAIN_CKPT: $POSTTRAIN_CKPT"
    echo "[launch_eval_my_ckpt] skipping prepare_eval_ckpt.sh"
else
    # Need to assemble from TRAIN_CKPT_DIR + REF_CKPT_DIR
    TRAIN_CKPT_DIR=${TRAIN_CKPT_DIR:-"$REPO_ROOT/train_out/checkpoints/checkpoint_step_${CKPT_STEP}"}
    # REF_CKPT_DIR: reference checkpoint containing vae/text_encoder/tokenizer
    if [[ -z "${REF_CKPT_DIR:-}" ]]; then
        echo "[ERROR] REF_CKPT_DIR is not set. Please set it to the reference checkpoint directory"
        echo "        containing vae/text_encoder/tokenizer, e.g.:"
        echo "        export REF_CKPT_DIR=/path/to/abot-m05-posttrain-libero-long"
        echo "        OR set POSTTRAIN_CKPT to a pre-assembled checkpoint path."
        exit 1
    fi
    # EVAL_CKPT_ROOT: directory for assembled eval checkpoints
    if [[ -z "${EVAL_CKPT_ROOT:-}" ]]; then
        echo "[ERROR] EVAL_CKPT_ROOT is not set. Please set it to a directory for assembled"
        echo "        eval checkpoints, e.g.:"
        echo "        export EVAL_CKPT_ROOT=/path/to/huggingface_ckpt/eval"
        echo "        OR set POSTTRAIN_CKPT to a pre-assembled checkpoint path."
        exit 1
    fi

    # Sanity checks for TRAIN_CKPT_DIR
    [[ -d "$TRAIN_CKPT_DIR" ]] \
        || { echo "[ERROR] TRAIN_CKPT_DIR not found: $TRAIN_CKPT_DIR"; exit 1; }
    [[ -f "$TRAIN_CKPT_DIR/transformer/diffusion_pytorch_model.safetensors" ]] \
        || { echo "[ERROR] no transformer weights at $TRAIN_CKPT_DIR/transformer/diffusion_pytorch_model.safetensors"; exit 1; }

    EVAL_CKPT_DIR="${EVAL_CKPT_ROOT}/checkpoint_step_${CKPT_STEP}"

    echo "==================================================="
    echo "[launch_eval_my_ckpt] settings (assemble mode):"
    echo "  CKPT_STEP        = $CKPT_STEP"
    echo "  TRAIN_CKPT_DIR   = $TRAIN_CKPT_DIR"
    echo "  REF_CKPT_DIR     = $REF_CKPT_DIR"
    echo "  EVAL_CKPT_DIR    = $EVAL_CKPT_DIR  (will be assembled here)"
    echo "  BENCHMARK        = $BENCHMARK   NUM_TASKS=$NUM_TASKS   TEST_NUM=$TEST_NUM"
    echo "  NUM_GPUS         = $NUM_GPUS"
    echo "==================================================="

    # ---- Step 1: assemble the inference-ready ckpt dir ----
    echo "[launch_eval_my_ckpt] step 1: assembling inference ckpt dir..."
    bash "$SCRIPT_DIR/prepare_eval_ckpt.sh" \
        "$TRAIN_CKPT_DIR" \
        "$REF_CKPT_DIR" \
        "$EVAL_CKPT_DIR"

    export POSTTRAIN_CKPT="$EVAL_CKPT_DIR"
fi

OUT_ROOT=${OUT_ROOT:-"outputs/libero_my_ckpt_step${CKPT_STEP}/${BENCHMARK}"}
LOG_DIR=${LOG_DIR:-"${OUT_ROOT}/logs"}
SAVE_ROOT_BASE=${SAVE_ROOT_BASE:-"${OUT_ROOT}/visualization"}

mkdir -p "$OUT_ROOT" "$LOG_DIR"

# ---- Step 3: hand off to launch_parallel.sh ----
echo
echo "[launch_eval_my_ckpt] POSTTRAIN_CKPT = $POSTTRAIN_CKPT"
echo "[launch_eval_my_ckpt] launching parallel eval..."
export BENCHMARK NUM_GPUS TEST_NUM NUM_TASKS
export BASE_SERVER_PORT BASE_MASTER_PORT
export OUT_ROOT LOG_DIR SAVE_ROOT_BASE
export CONFIG_NAME=${CONFIG_NAME:-libero_original}

# Stop any stale workers first (best-effort)
LOG_DIR="$LOG_DIR" bash "$SCRIPT_DIR/stop_parallel.sh" >/dev/null 2>&1 || true

exec bash "$SCRIPT_DIR/launch_parallel.sh"
