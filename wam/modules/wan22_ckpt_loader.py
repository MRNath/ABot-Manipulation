# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Load official Wan2.2 (WanModel) checkpoints into WanTransformer3DModel."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

_logger = logging.getLogger(__name__)

WAN22_CLASS_NAME = "WanModel"
WAM_CLASS_NAME = "WanTransformer3DModel"
WAM_CLASS_NAMES = {
    WAM_CLASS_NAME,
    "FSDPWanTransformer3DModel",
}
WAM_CONFIG_KEYS = {
    "num_attention_heads",
    "attention_head_dim",
    "in_channels",
    "out_channels",
    "num_layers",
}

# Default Wan2.2-TI2V-5B -> WAM transformer config when checkpoint has no config.json.
WAN22_TI2V_5B_WAM_CONFIG = dict(
    patch_size=[1, 2, 2],
    num_attention_heads=24,
    attention_head_dim=128,
    in_channels=48,
    out_channels=48,
    action_dim=30,
    text_dim=4096,
    freq_dim=256,
    ffn_dim=14336,
    num_layers=30,
    cross_attn_norm=True,
    eps=1e-6,
    rope_max_seq_len=1024,
    pos_embed_seq_len=None,
)


def resolve_transformer_checkpoint_dir(path: str | Path) -> Path:
    """Resolve an Abot/Wan-style checkpoint path to the actual transformer dir.

    Accepts either:
      - the checkpoint root containing ``transformer/config.json``; or
      - the transformer directory itself containing ``config.json``.

    Also supports Wan2.2-style layouts where ``config.json``/weights live at
    the provided root directly.
    """
    root = Path(path).expanduser()
    child = root / "transformer"

    if (child / "config.json").is_file():
        return child
    if (root / "config.json").is_file():
        return root
    if child.is_dir():
        return child
    return root


def is_wan22_checkpoint(path: str | Path) -> bool:
    """Return True if *path* looks like an official Wan2.2 WanModel checkpoint."""
    root = resolve_transformer_checkpoint_dir(path)
    if not root.is_dir():
        return False
    cfg = root / "config.json"
    if cfg.is_file():
        try:
            with open(cfg, encoding="utf-8") as f:
                meta = json.load(f)
            class_name = meta.get("_class_name")
            if class_name == WAN22_CLASS_NAME:
                return True
            if class_name in WAM_CLASS_NAMES:
                return False
            if WAM_CONFIG_KEYS.issubset(meta):
                return False
        except (json.JSONDecodeError, OSError):
            pass
    if (root / "transformer" / "config.json").is_file():
        try:
            with open(root / "transformer" / "config.json", encoding="utf-8") as f:
                meta = json.load(f)
            class_name = meta.get("_class_name")
            if class_name == WAN22_CLASS_NAME:
                return True
            if class_name in WAM_CLASS_NAMES:
                return False
            if WAM_CONFIG_KEYS.issubset(meta):
                return False
        except (json.JSONDecodeError, OSError):
            pass
    # Shards at root without VA layout: treat as Wan2.2 if no transformer/config.json.
    has_shards = bool(list(root.glob("diffusion_pytorch_model*.safetensors")))
    if has_shards and not (root / "transformer" / "config.json").is_file():
        return True
    return False


def resolve_transformer_load_path(
    resume_from: str | None,
    wan22_pretrained: str | None,
) -> tuple[str, str]:
    """Resolve (load_path, checkpoint_format) where format is 'abot_m05' or 'wan22'."""
    candidates: list[tuple[str, str]] = []
    if resume_from:
        root = Path(resume_from)
        tr = root / "transformer"
        if tr.is_dir() and (tr / "config.json").is_file():
            candidates.append((str(tr), "abot_m05"))
        elif is_wan22_checkpoint(root):
            candidates.append((str(root), "wan22"))
        elif tr.is_dir():
            candidates.append((str(tr), "abot_m05"))
        else:
            candidates.append((str(root), "abot_m05"))
    if wan22_pretrained:
        root = Path(wan22_pretrained)
        tr = root / "transformer"
        if tr.is_dir() and (tr / "config.json").is_file() and not is_wan22_checkpoint(root):
            candidates.append((str(tr), "abot_m05"))
        elif is_wan22_checkpoint(root):
            candidates.append((str(root), "wan22"))
    if not candidates:
        raise ValueError("No checkpoint path: set resume_from or wan22_pretrained_model_name_or_path")
    path, fmt = candidates[0]
    if fmt == "abot_m05" and not Path(path).joinpath("config.json").is_file():
        if is_wan22_checkpoint(path):
            fmt = "wan22"
    return path, fmt


def load_wan22_raw_state_dict(checkpoint_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load all diffusion_pytorch_model*.safetensors shards under *checkpoint_dir*."""
    root = resolve_transformer_checkpoint_dir(checkpoint_dir)
    index_path = root / "diffusion_pytorch_model.safetensors.index.json"
    state_dict: dict[str, torch.Tensor] = {}
    if index_path.is_file():
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shard_files = {root / fn for fn in set(weight_map.values())}
        for shard_path in sorted(shard_files):
            state_dict.update(load_file(shard_path, device="cpu"))
        return state_dict
    shard_paths = sorted(root.glob("diffusion_pytorch_model*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No Wan2.2 weight shards under {root}")
    for shard_path in shard_paths:
        state_dict.update(load_file(shard_path, device="cpu"))
    return state_dict


def convert_wan22_key_to_va(key: str) -> str | None:
    """Map one WanModel state-dict key to WanTransformer3DModel key, or None to skip."""
    if key.startswith("patch_embedding"):
        return None

    m = re.match(r"blocks\.(\d+)\.self_attn\.(q|k|v)\.(weight|bias)$", key)
    if m:
        i, proj, suffix = m.groups()
        return f"blocks.{i}.attn1.to_{proj}.{suffix}"

    m = re.match(r"blocks\.(\d+)\.self_attn\.o\.(weight|bias)$", key)
    if m:
        i, suffix = m.groups()
        return f"blocks.{i}.attn1.to_out.0.{suffix}"

    m = re.match(r"blocks\.(\d+)\.self_attn\.(norm_q|norm_k)\.weight$", key)
    if m:
        i, norm = m.groups()
        return f"blocks.{i}.attn1.{norm}.weight"

    m = re.match(r"blocks\.(\d+)\.cross_attn\.(q|k|v)\.(weight|bias)$", key)
    if m:
        i, proj, suffix = m.groups()
        return f"blocks.{i}.attn2.to_{proj}.{suffix}"

    m = re.match(r"blocks\.(\d+)\.cross_attn\.o\.(weight|bias)$", key)
    if m:
        i, suffix = m.groups()
        return f"blocks.{i}.attn2.to_out.0.{suffix}"

    m = re.match(r"blocks\.(\d+)\.cross_attn\.(norm_q|norm_k)\.weight$", key)
    if m:
        i, norm = m.groups()
        return f"blocks.{i}.attn2.{norm}.weight"

    m = re.match(r"blocks\.(\d+)\.ffn\.0\.(weight|bias)$", key)
    if m:
        i, suffix = m.groups()
        return f"blocks.{i}.ffn.net.0.proj.{suffix}"

    m = re.match(r"blocks\.(\d+)\.ffn\.2\.(weight|bias)$", key)
    if m:
        i, suffix = m.groups()
        return f"blocks.{i}.ffn.net.2.{suffix}"

    m = re.match(r"blocks\.(\d+)\.modulation$", key)
    if m:
        return f"blocks.{m.group(1)}.scale_shift_table"

    m = re.match(r"blocks\.(\d+)\.norm3\.(weight|bias)$", key)
    if m:
        i, suffix = m.groups()
        return f"blocks.{i}.norm2.{suffix}"

    static = {
        "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
        "text_embedding.0.bias": "condition_embedder.text_embedder.linear_1.bias",
        "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight",
        "text_embedding.2.bias": "condition_embedder.text_embedder.linear_2.bias",
        "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
        "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
        "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
        "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
        "time_projection.1.weight": "condition_embedder.time_proj.weight",
        "time_projection.1.bias": "condition_embedder.time_proj.bias",
        "head.head.weight": "proj_out.weight",
        "head.head.bias": "proj_out.bias",
        "head.modulation": "scale_shift_table",
    }
    return static.get(key)


def convert_wan22_state_dict_to_va(
    wan_sd: dict[str, torch.Tensor],
    *,
    duplicate_action_condition: bool = True,
    use_mot: bool = False,
) -> tuple[dict[str, torch.Tensor], list[str], list[str]]:
    """Convert WanModel keys to WanTransformer3DModel keys.

    When ``use_mot=True``, the converted single-tower keys are subsequently
    mirrored to MoT layout (each block tensor cloned into ``.lat`` /
    ``.act`` twins, top-level ``scale_shift_table`` cloned to
    ``scale_shift_table_{lat,act}``). For symmetric same-dim MoT the
    mirrored shapes already match the model; for ``act_dim != video_dim``
    the act-side clones will fail the loader's shape filter and get
    filled in later by ``init_act_stream_from_lat``.
    """
    va_sd: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    unmapped: list[str] = []
    for key, tensor in wan_sd.items():
        new_key = convert_wan22_key_to_va(key)
        if new_key is None:
            if key.startswith("patch_embedding"):
                skipped.append(key)
            else:
                unmapped.append(key)
            continue
        va_sd[new_key] = tensor
    if duplicate_action_condition and not use_mot:
        # Non-MoT path: action stream shares video_dim; deepcopy the
        # condition embedder weights so condition_embedder_action loads
        # identical values. (For MoT we instead let mirror+init handle it.)
        for key, tensor in list(va_sd.items()):
            if key.startswith("condition_embedder."):
                va_sd[key.replace(
                    "condition_embedder.",
                    "condition_embedder_action.", 1)] = tensor.clone()
    if use_mot:
        from .utils import mirror_legacy_state_dict
        va_sd = mirror_legacy_state_dict(va_sd)
    return va_sd, skipped, unmapped


def _init_va_action_heads_from_condition(model) -> None:
    """Non-MoT path: mirror condition_embedder into condition_embedder_action
    so the action stream gets sensible initialization (Wan2.2 has no
    action-head weights). Action MLP / proj stays at random init.

    For MoT, this is handled by ``init_act_stream_from_lat`` after load.
    """
    if getattr(model, "use_mot", False):
        return
    if not hasattr(model, "condition_embedder_action") or model.condition_embedder_action is None:
        return
    model.condition_embedder_action.load_state_dict(model.condition_embedder.state_dict())
    _logger.info("Initialized condition_embedder_action from condition_embedder (Wan2.2 has no action heads).")


def build_va_model_from_wan22_config(
    checkpoint_dir: str | Path,
    *,
    attn_mode: str = "torch",
    use_mot: bool = False,
    mot_action_hidden_dim: int = 768,
    mot_action_ffn_multiplier: int = 4,
    **config_overrides,
):
    from .model import WanTransformer3DModel

    checkpoint_dir = resolve_transformer_checkpoint_dir(checkpoint_dir)
    cfg = dict(WAN22_TI2V_5B_WAM_CONFIG)
    cfg_path = Path(checkpoint_dir) / "config.json"
    if cfg_path.is_file():
        with open(cfg_path, encoding="utf-8") as f:
            wan_cfg = json.load(f)
        if wan_cfg.get("num_heads"):
            cfg["num_attention_heads"] = int(wan_cfg["num_heads"])
        if wan_cfg.get("dim"):
            cfg["attention_head_dim"] = int(wan_cfg["dim"]) // cfg["num_attention_heads"]
        if wan_cfg.get("num_layers"):
            cfg["num_layers"] = int(wan_cfg["num_layers"])
        if wan_cfg.get("ffn_dim"):
            cfg["ffn_dim"] = int(wan_cfg["ffn_dim"])
        if wan_cfg.get("in_dim"):
            cfg["in_channels"] = int(wan_cfg["in_dim"])
        if wan_cfg.get("out_dim"):
            cfg["out_channels"] = int(wan_cfg["out_dim"])
        if wan_cfg.get("freq_dim"):
            cfg["freq_dim"] = int(wan_cfg["freq_dim"])
    cfg.update(config_overrides)
    cfg["attn_mode"] = attn_mode
    cfg["use_mot"] = use_mot
    cfg["mot_action_hidden_dim"] = mot_action_hidden_dim
    cfg["mot_action_ffn_multiplier"] = mot_action_ffn_multiplier
    return WanTransformer3DModel(**cfg)


def load_transformer_from_wan22(
    checkpoint_dir: str | Path,
    torch_dtype,
    torch_device,
    attn_mode: str | None = None,
    use_mot: bool = False,
    mot_action_hidden_dim: int = 768,
    mot_action_ffn_multiplier: int = 4,
    strict: bool = False,
):
    """Load official Wan2.2 WanModel weights into a new WanTransformer3DModel.

    For ``use_mot=True``, after mapping Wan2.2 → VA single-tower keys we
    mirror them to the MoT layout (clones into ``.lat`` / ``.act`` twins).
    For symmetric same-dim MoT (``mot_action_hidden_dim == video_dim``)
    the act tower is initialized identically to lat by the mirror; for
    asymmetric MoT the act-side mismatched-shape keys are filled in
    afterwards by ``init_act_stream_from_lat``.
    """
    from .utils import init_act_stream_from_lat

    checkpoint_dir = resolve_transformer_checkpoint_dir(checkpoint_dir)
    resolved_attn = attn_mode or "torch"
    model = build_va_model_from_wan22_config(
        checkpoint_dir,
        attn_mode=resolved_attn,
        use_mot=use_mot,
        mot_action_hidden_dim=mot_action_hidden_dim,
        mot_action_ffn_multiplier=mot_action_ffn_multiplier,
    )
    wan_sd = load_wan22_raw_state_dict(checkpoint_dir)
    va_sd, skipped, unmapped = convert_wan22_state_dict_to_va(
        wan_sd, use_mot=use_mot)
    model_sd = model.state_dict()
    filtered = {
        k: v for k, v in va_sd.items()
        if k in model_sd and v.shape == model_sd[k].shape
    }
    shape_mismatch = [
        k for k, v in va_sd.items()
        if k in model_sd and v.shape != model_sd[k].shape
    ]
    missing_keys = [k for k in model_sd if k not in filtered]
    unexpected = [k for k in va_sd if k not in model_sd] + unmapped

    model.load_state_dict(filtered, strict=False)
    if use_mot and mot_action_hidden_dim != model.inner_dim:
        # Asymmetric MoT: lat is loaded from Wan2.2 video weights; fill in
        # the shape-mismatched act tower by interpolation from lat.
        init_act_stream_from_lat(model)
    elif not use_mot:
        _init_va_action_heads_from_condition(model)

    _logger.info(
        "Loaded Wan2.2 checkpoint from %s: mapped=%d skipped_patch=%d "
        "shape_mismatch=%d missing=%d (action/MoT heads expected)",
        checkpoint_dir,
        len(filtered),
        len(skipped),
        len(shape_mismatch),
        len(missing_keys),
    )
    if skipped:
        _logger.warning(
            "Skipped patch_embedding Conv3d weights (%d tensors); "
            "patch_embedding_mlp is randomly initialized (latent training uses MLP path).",
            len(skipped),
        )
    if shape_mismatch:
        _logger.warning("Shape mismatch keys (first 5): %s", shape_mismatch[:5])
    if unmapped:
        _logger.warning("Unmapped Wan2.2 keys (first 5): %s", unmapped[:5])
    if strict and (missing_keys or shape_mismatch or unmapped):
        raise RuntimeError(
            f"strict=True but load incomplete: missing={len(missing_keys)} "
            f"shape_mismatch={len(shape_mismatch)} unmapped={len(unmapped)}"
        )

    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    return model.to(torch_device)


if __name__ == "__main__":
    import sys

    ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else "Wan-AI/Wan2.2-TI2V-5B"
    wan_sd = load_wan22_raw_state_dict(ckpt_dir)
    va_sd, skipped, unmapped = convert_wan22_state_dict_to_va(wan_sd)
    print(f"checkpoint={ckpt_dir}")
    print(f"wan_tensors={len(wan_sd)} va_tensors={len(va_sd)} skipped={skipped} unmapped={len(unmapped)}")
    if unmapped:
        raise SystemExit(f"unmapped keys: {unmapped[:10]}")
