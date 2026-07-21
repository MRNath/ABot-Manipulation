#!/usr/bin/env bash
# Prepare an inference-ready ckpt directory IN-PLACE on the training ckpt dir:
#   - Training transformer is already at TRAIN_CKPT_DIR/transformer/ (your weights + config.json)
#   - Just add symlinks under TRAIN_CKPT_DIR: vae / text_encoder / tokenizer / configuration.json
#     These are eval-required modules that training does not save (borrowed from base ckpt)
#   - Also ensure transformer/config.json has diffusers-required metadata (_class_name etc.)
#
# Default OUT_CKPT_DIR = TRAIN_CKPT_DIR (in-place, no extra copy to huggingface_ckpt/whn_eval/)
#
# Usage:
#   bash evaluation/libero/prepare_eval_ckpt.sh \
#       <TRAIN_CKPT_DIR> [<REF_CKPT_DIR>] [<OUT_CKPT_DIR>]
#
# Example (in-place):
#   bash evaluation/libero/prepare_eval_ckpt.sh \
#       /path/to/train_out/.../checkpoint_step_10000
#
# Example (copy to separate dir, use only when TRAIN_CKPT_DIR is not writable):
#   bash evaluation/libero/prepare_eval_ckpt.sh \
#       /path/to/train_out/.../checkpoint_step_10000 \
#       "" \
#       /path/to/some/other/eval_ckpt_dir
#
# Defaults:
#   REF_CKPT_DIR = (required, must be passed as 2nd arg or set via env var)
#   OUT_CKPT_DIR = TRAIN_CKPT_DIR  <- works in-place
set -euo pipefail

TRAIN_CKPT_DIR=${1:?"first arg: path to train_out/checkpoints/checkpoint_step_XXX"}
# REF_CKPT_DIR: must be provided as 2nd arg or env var; no hardcoded default
REF_CKPT_DIR=${2:-${REF_CKPT_DIR:-}}
# Empty string is treated as not provided (lets users skip arg 2 via "" while specifying arg 3)
if [[ -z "${REF_CKPT_DIR}" ]]; then
    echo "[ERROR] REF_CKPT_DIR not provided. Pass as 2nd arg or set env var, e.g.:"
    echo "  bash evaluation/libero/prepare_eval_ckpt.sh <TRAIN_CKPT_DIR> /path/to/ref_ckpt"
    echo "  # or: export REF_CKPT_DIR=/path/to/ref_ckpt"
    exit 1
fi
OUT_CKPT_DIR=${3:-${TRAIN_CKPT_DIR}}    # <- default: in-place

echo "[prepare_eval_ckpt] TRAIN_CKPT_DIR = $TRAIN_CKPT_DIR"
echo "[prepare_eval_ckpt] REF_CKPT_DIR   = $REF_CKPT_DIR"
echo "[prepare_eval_ckpt] OUT_CKPT_DIR   = $OUT_CKPT_DIR"
if [[ "$OUT_CKPT_DIR" == "$TRAIN_CKPT_DIR" ]]; then
    echo "[prepare_eval_ckpt] (working in-place on training ckpt dir)"
fi

# ---- sanity checks ----
[[ -d "$TRAIN_CKPT_DIR/transformer" ]] || { echo "[ERROR] no transformer/ under $TRAIN_CKPT_DIR"; exit 1; }
[[ -f "$TRAIN_CKPT_DIR/transformer/diffusion_pytorch_model.safetensors" ]] \
    || { echo "[ERROR] missing diffusion_pytorch_model.safetensors in $TRAIN_CKPT_DIR/transformer"; exit 1; }
[[ -f "$TRAIN_CKPT_DIR/transformer/config.json" ]] \
    || { echo "[ERROR] missing transformer/config.json in $TRAIN_CKPT_DIR (training should save it)"; exit 1; }

for sub in vae text_encoder tokenizer; do
    [[ -d "$REF_CKPT_DIR/$sub" ]] || { echo "[ERROR] reference ckpt missing $sub/"; exit 1; }
done

mkdir -p "$OUT_CKPT_DIR/transformer"

# ---- 1) transformer/config.json: add diffusers-required metadata ----
# The training config.json already has all architecture fields (use_mot/action_dim/mot_*),
# only missing diffusers metadata like _class_name / _diffusers_version.
# Patch the original config in-place (no full merge, to avoid overwriting training fields).
#
# Historical BUG (fixed):
#   The old version symlinked the base ckpt config.json directly, losing training-time
#   use_mot/action_dim and other key architecture fields. load_transformer then built
#   the wrong model + partial-loaded weights -> MoT action head was random -> actions
#   were garbage, success rate near zero.
#   Now the training config is authoritative; only diffusers metadata is patched in.
#   Now the training config is authoritative; only diffusers metadata is patched in.|
TRAIN_CFG_JSON="$TRAIN_CKPT_DIR/transformer/config.json"
OUT_CFG_JSON="$OUT_CKPT_DIR/transformer/config.json"
REF_CFG_JSON="$REF_CKPT_DIR/transformer/config.json"
# Temp file to avoid read-write race on the same file (in-place: OUT==TRAIN, same file)
# Temp file to avoid read-write race on the same file (in-place: OUT==TRAIN, same file)
# Temp file to avoid read-write race on the same file (in-place: OUT==TRAIN, same file)
TMP_CFG=$(mktemp)
${PY_BIN_SERVER:-python3} - "$TRAIN_CFG_JSON" "$REF_CFG_JSON" "$TMP_CFG" <<'PYEOF'
import json, sys
train_cfg_path, ref_cfg_path, out_cfg_path = sys.argv[1], sys.argv[2], sys.argv[3]
# Training config is authoritative (has action_dim / use_mot / mot_* and all arch fields)
# Training config is authoritative (has action_dim / use_mot / mot_* and all arch fields)
# Only patch in diffusers-required fields from ref when missing in training config
# Only patch in diffusers-required fields from ref when missing in training config
# Only patch in diffusers-required fields from ref when missing in training config
for k in ("_class_name", "_diffusers_version", "added_kv_proj_dim", "image_dim"):
    if k not in merged and k in ref_cfg:
        merged[k] = ref_cfg[k]
if not merged.get("use_mot"):
    print(f"[prepare_eval_ckpt][WARN] train config has no 'use_mot'. "
          f"If your training used MoT, this will silently break the model.")
with open(out_cfg_path, "w") as f:
    json.dump(merged, f, indent=2)
print(f"[prepare_eval_ckpt] patched config: "
      f"use_mot={merged.get('use_mot')}, "
      f"action_dim={merged.get('action_dim')}, "
      f"mot_action_hidden_dim={merged.get('mot_action_hidden_dim')}, "
      f"qk_norm={merged.get('qk_norm')}, "
# Atomic write to target (rm symlink -> mv temp file)
# Atomic write to target (rm symlink -> mv temp file)
# Atomic write to target (rm symlink -> mv temp file)
rm -f "$OUT_CFG_JSON"
# ---- 2) weights: in-place already has the file, skip; non-in-place creates symlink ----
# ---- 2) weights: in-place already has the file, skip; non-in-place creates symlink ----
# ---- 2) weights: in-place already has the file, skip; non-in-place creates symlink ----
if [[ "$OUT_CKPT_DIR" != "$TRAIN_CKPT_DIR" ]]; then
    ln -sfn "$TRAIN_CKPT_DIR/transformer/diffusion_pytorch_model.safetensors" \
# Defensive cleanup: remove old sharded leftovers (only delete symlinks, never real files)
# Defensive cleanup: remove old sharded leftovers (only delete symlinks, never real files)
# Defensive cleanup: remove old sharded leftovers (only delete symlinks, never real files)
find "$OUT_CKPT_DIR/transformer" -maxdepth 1 -type l -name "diffusion_pytorch_model-*-of-*.safetensors" -delete
# ---- 3) vae / text_encoder / tokenizer  (symlink to base ckpt) ----
# ---- 3) vae / text_encoder / tokenizer  (symlink to base ckpt) ----
# ---- 3) vae / text_encoder / tokenizer  (symlink to base ckpt) ----
for sub in vae text_encoder tokenizer; do
    ln -sfn "$REF_CKPT_DIR/$sub" "$OUT_CKPT_DIR/$sub"
done

# ---- 4) configuration.json (optional) ----
if [[ -f "$REF_CKPT_DIR/configuration.json" ]]; then
    ln -sfn "$REF_CKPT_DIR/configuration.json" "$OUT_CKPT_DIR/configuration.json"
fi

ls -la "$OUT_CKPT_DIR" | grep -v "training_state.pt" || true   # do not print 44GB optimizer state
echo "[prepare_eval_ckpt] ===== OUT_CKPT_DIR contents ====="
ls -la "$OUT_CKPT_DIR" | grep -v "training_state.pt" || true   # do not print 44GB optimizer state
echo
echo "[prepare_eval_ckpt] ===== transformer/ contents ====="
ls -la "$OUT_CKPT_DIR/transformer"
echo
echo "[prepare_eval_ckpt] Done. Use this path as posttrain_model_name_or_path:"
echo "    $OUT_CKPT_DIR"
