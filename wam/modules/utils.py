# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Loading helpers + slim warm-start for symmetric MoT.

The two main entry points exported from this module are:

* :func:`load_transformer` — build / load a ``WanTransformer3DModel`` from
  disk, with support for native MoT checkpoints, single-tower (pre-MoT)
  checkpoints (auto-mirrored to MoT layout), and Wan2.2 official
  checkpoints. Optionally swaps the attention implementation post-load via
  ``attn_mode``.
* :func:`init_act_stream_from_lat` — slim (~50-line) interpolation-based
  warm-start of the act-side parameters from the lat-side counterparts.
  Only needed when ``mot_action_hidden_dim != num_attention_heads *
  attention_head_dim`` (i.e. the act tower is smaller than lat); a no-op
  for symmetric same-dim MoT.

The 270-line legacy ``initialize_mot_action_stream`` is gone — its
functionality is subsumed by ``mirror_legacy_state_dict`` (for the easy
"clone lat → act" case) plus ``init_act_stream_from_lat`` (for the
asymmetric "interpolate lat → act" case).
"""

import json
import logging
import math
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .attention_ops import SUPPORTED_ATTN_MODES, build_attn_op
from .model import (
    CrossAttention,
    WanAttention,
    WanTransformer3DModel,
    _mirror_legacy_state_dict_inplace,
)


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VAE / text encoder / tokenizer loaders
# ---------------------------------------------------------------------------

def load_vae(vae_path, torch_dtype, torch_device):
    vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=torch_dtype)
    return vae.to(torch_device)


def load_text_encoder(text_encoder_path, torch_dtype, torch_device):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path, torch_dtype=torch_dtype)
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path):
    return T5TokenizerFast.from_pretrained(tokenizer_path)


# ---------------------------------------------------------------------------
# Raw state-dict loading (sharded / single safetensors / pytorch_model.bin)
# ---------------------------------------------------------------------------

def _load_raw_state_dict(model_path):
    """Load a HuggingFace-style model directory's weights into a plain
    ``state_dict`` (sharded safetensors, single safetensors, or
    pytorch_model.bin)."""
    sharded_idx = os.path.join(
        model_path, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(sharded_idx):
        from safetensors.torch import load_file
        with open(sharded_idx) as f:
            idx = json.load(f)
        sd = {}
        for shard in sorted(set(idx["weight_map"].values())):
            sd.update(load_file(os.path.join(model_path, shard)))
        return sd

    single_st = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
    if os.path.exists(single_st):
        from safetensors.torch import load_file
        return load_file(single_st)

    pt_path = os.path.join(model_path, "diffusion_pytorch_model.bin")
    if os.path.exists(pt_path):
        return torch.load(pt_path, map_location="cpu")

    raise FileNotFoundError(
        f"No diffusion_pytorch_model weights found under {model_path}")


# ---------------------------------------------------------------------------
# Legacy (pre-MoT, single-tower) state-dict detection / mirroring
# ---------------------------------------------------------------------------

_LEGACY_BLOCK_RE = re.compile(r"^blocks\.\d+\.")


def _state_dict_is_mot(state_dict):
    """A MoT state dict has either ``scale_shift_table_lat`` at top level or
    a key ending in ``.lat`` / ``.act`` inside a block."""
    for k in state_dict:
        if k in ("scale_shift_table_lat", "scale_shift_table_act"):
            return True
        if ".lat." in k or ".act." in k:
            return True
        if k.endswith(".lat") or k.endswith(".act"):
            return True
    return False


def mirror_legacy_state_dict(state_dict):
    """Return a new dict with single-tower (pre-MoT) keys rewritten to MoT
    layout (each lat block tensor cloned into ``.lat.`` / ``.act.`` twins,
    top-level ``scale_shift_table`` cloned to ``scale_shift_table_{lat,act}``).
    Already-MoT inputs pass through unchanged.

    Thin wrapper around the in-place mirror function defined in
    ``model.py``; used by ``wan22_ckpt_loader`` and the legacy load path
    below.
    """
    if _state_dict_is_mot(state_dict):
        return dict(state_dict)
    out = dict(state_dict)
    _mirror_legacy_state_dict_inplace(out, prefix="")
    return out


# ---------------------------------------------------------------------------
# Per-axis interpolation + slim act-stream warm-start from lat
# ---------------------------------------------------------------------------

@torch.no_grad()
def _interp_to_shape(src, target_shape, alpha=1.0):
    """Linear-interpolate ``src`` per axis to ``target_shape``, then
    multiply by ``alpha``. Use ``alpha = sqrt(D_in_old / D_in_new)`` on
    weights whose Linear input dim shrinks, to preserve activation
    variance. ``alpha=1`` everywhere else.
    """
    target_shape = tuple(target_shape)
    src = src.detach()
    if tuple(src.shape) == target_shape:
        return (src.clone().float() * alpha) if alpha != 1.0 else src.clone()
    out = src.float()
    for axis, (s, t) in enumerate(zip(out.shape, target_shape)):
        if s == t:
            continue
        perm = list(range(out.ndim))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        x = out.permute(*perm).contiguous()
        head = x.shape[:-1]
        x = F.interpolate(x.reshape(-1, 1, s),
                          size=t, mode="linear", align_corners=True)
        out = x.reshape(*head, t).permute(*perm).contiguous()
    return out * alpha


def _copy_interp(dst, src, apply_alpha=False, alpha=1.0):
    """Interp-copy ``src`` (Parameter or Tensor) into ``dst`` (Parameter),
    matching ``dst.shape``/dtype/device. ``apply_alpha=True`` multiplies
    by ``alpha`` (use on weights whose input dim shrinks)."""
    src_data = src.data if isinstance(src, nn.Parameter) else src
    a = alpha if apply_alpha else 1.0
    new_val = _interp_to_shape(src_data, dst.shape, alpha=a).to(
        device=dst.device, dtype=dst.dtype)
    dst.data.copy_(new_val)


def _copy_condition_embedder(dst, src, alpha):
    """Copy a ``WanTimeTextImageEmbedding`` from a wider ``src`` (video_dim)
    into a narrower ``dst`` (act_dim). Inner Linears with input dim
    shrinking get ``alpha``; output-only shrinks don't."""
    # time_embedder.linear_1: input freq_dim unchanged, output shrinks.
    _copy_interp(dst.time_embedder.linear_1.weight,
                 src.time_embedder.linear_1.weight)
    if dst.time_embedder.linear_1.bias is not None:
        _copy_interp(dst.time_embedder.linear_1.bias,
                     src.time_embedder.linear_1.bias)
    # time_embedder.linear_2: input dim shrinks (video_dim -> act_dim).
    _copy_interp(dst.time_embedder.linear_2.weight,
                 src.time_embedder.linear_2.weight,
                 apply_alpha=True, alpha=alpha)
    if dst.time_embedder.linear_2.bias is not None:
        _copy_interp(dst.time_embedder.linear_2.bias,
                     src.time_embedder.linear_2.bias)
    # time_proj: input dim shrinks (video_dim -> act_dim).
    _copy_interp(dst.time_proj.weight, src.time_proj.weight,
                 apply_alpha=True, alpha=alpha)
    if dst.time_proj.bias is not None:
        _copy_interp(dst.time_proj.bias, src.time_proj.bias)
    # text_embedder.linear_1: input text_dim unchanged, output shrinks.
    _copy_interp(dst.text_embedder.linear_1.weight,
                 src.text_embedder.linear_1.weight)
    if dst.text_embedder.linear_1.bias is not None:
        _copy_interp(dst.text_embedder.linear_1.bias,
                     src.text_embedder.linear_1.bias)
    # text_embedder.linear_2: input dim shrinks (video_dim -> act_dim).
    _copy_interp(dst.text_embedder.linear_2.weight,
                 src.text_embedder.linear_2.weight,
                 apply_alpha=True, alpha=alpha)
    if dst.text_embedder.linear_2.bias is not None:
        _copy_interp(dst.text_embedder.linear_2.bias,
                     src.text_embedder.linear_2.bias)


@torch.no_grad()
def init_act_stream_from_lat(model):
    """Warm-start the symmetric MoT act tower from the already-loaded lat
    tower (and from any stashed pre-MoT top-level action weights on
    ``model._pretrained_*``).

    No-op unless ``model.use_mot=True AND model.act_dim != model.inner_dim``.
    For same-dim MoT, the legacy mirror clone is already correct.
    """
    if not getattr(model, "use_mot", False):
        return
    video_dim = model.inner_dim
    act_dim = model.act_dim
    if act_dim == video_dim:
        return  # mirror already cloned everything at the correct shape.
    alpha = math.sqrt(video_dim / act_dim)

    cp = _copy_interp

    for block in model.blocks:
        # ---- self-attention (joint) ----
        a1l, a1a = block.attn1["lat"], block.attn1["act"]
        for n in ("to_q", "to_k", "to_v"):
            cp(getattr(a1a, n).weight, getattr(a1l, n).weight,
               apply_alpha=True, alpha=alpha)  # input dim shrinks
            cp(getattr(a1a, n).bias, getattr(a1l, n).bias)
        cp(a1a.to_out[0].weight, a1l.to_out[0].weight)  # input dim unchanged
        cp(a1a.to_out[0].bias,   a1l.to_out[0].bias)
        cp(a1a.norm_q.weight, a1l.norm_q.weight)  # shape unchanged (inner_dim)
        cp(a1a.norm_k.weight, a1l.norm_k.weight)

        # ---- text cross-attention ----
        a2l, a2a = block.attn2["lat"], block.attn2["act"]
        for n in ("to_q", "to_k", "to_v"):
            cp(getattr(a2a, n).weight, getattr(a2l, n).weight,
               apply_alpha=True, alpha=alpha)
            cp(getattr(a2a, n).bias, getattr(a2l, n).bias)
        cp(a2a.to_out[0].weight, a2l.to_out[0].weight,
           apply_alpha=True, alpha=alpha)
        cp(a2a.to_out[0].bias,   a2l.to_out[0].bias)
        cp(a2a.norm_q.weight, a2l.norm_q.weight)
        cp(a2a.norm_k.weight, a2l.norm_k.weight)

        # ---- feed-forward ----
        ffl, ffa = block.ffn["lat"], block.ffn["act"]
        cp(ffa.net[0].proj.weight, ffl.net[0].proj.weight,
           apply_alpha=True, alpha=alpha)
        if ffa.net[0].proj.bias is not None:
            cp(ffa.net[0].proj.bias, ffl.net[0].proj.bias)
        cp(ffa.net[2].weight, ffl.net[2].weight,
           apply_alpha=True, alpha=alpha)
        if ffa.net[2].bias is not None:
            cp(ffa.net[2].bias, ffl.net[2].bias)

        # ---- norms (only norm2 has affine params when cross_attn_norm=True) ----
        n2l, n2a = block.norm2["lat"], block.norm2["act"]
        if getattr(n2a, "weight", None) is not None:
            cp(n2a.weight, n2l.weight)
            if getattr(n2a, "bias", None) is not None:
                cp(n2a.bias, n2l.bias)

        # ---- per-block scale_shift_table ----
        cp(block.scale_shift_table["act"],
           block.scale_shift_table["lat"], apply_alpha=True, alpha=alpha)

    # ---- top-level ----
    cp(model.scale_shift_table_act, model.scale_shift_table_lat,
       apply_alpha=True, alpha=alpha)
    _copy_condition_embedder(model.condition_embedder_action,
                             model.condition_embedder, alpha)

    # action_embedder / action_proj_out: no lat twin in MoT model. Use
    # pre-MoT weights stashed on the model by the loader, when available.
    if hasattr(model, "_pretrained_action_embedder_weight"):
        cp(model.action_embedder.weight,
           model._pretrained_action_embedder_weight)  # output shrinks, no alpha
        if (model.action_embedder.bias is not None
                and hasattr(model, "_pretrained_action_embedder_bias")):
            cp(model.action_embedder.bias,
               model._pretrained_action_embedder_bias)
    if hasattr(model, "_pretrained_action_proj_out_weight"):
        cp(model.action_proj_out.weight,
           model._pretrained_action_proj_out_weight,
           apply_alpha=True, alpha=alpha)  # input shrinks
        if (model.action_proj_out.bias is not None
                and hasattr(model, "_pretrained_action_proj_out_bias")):
            model.action_proj_out.bias.data.copy_(
                model._pretrained_action_proj_out_bias.to(
                    device=model.action_proj_out.bias.device,
                    dtype=model.action_proj_out.bias.dtype))

    _logger.info(
        "[init_act_stream_from_lat] act tower initialized from lat "
        "(video_dim=%d, act_dim=%d, alpha=%.4f)", video_dim, act_dim, alpha)


# ---------------------------------------------------------------------------
# Attention-mode swap
# ---------------------------------------------------------------------------

def _maybe_swap_attn_op(model, attn_mode):
    """Walk every attention sub-module and rebuild its ``attn_op`` for the
    requested mode. ``nn.Module.__setattr__`` refuses to replace a child
    Module with a non-Module callable, so we explicitly drop ``attn_op``
    from ``_modules`` first."""
    if attn_mode is None:
        return
    if attn_mode not in SUPPORTED_ATTN_MODES:
        raise ValueError(
            f"Unsupported attn_mode: {attn_mode}, "
            f"supported: {SUPPORTED_ATTN_MODES}")
    for module in model.modules():
        if not isinstance(module, (WanAttention, CrossAttention)):
            continue
        is_cross = isinstance(module, CrossAttention) or (
            getattr(module, "cross_attention_dim_head", None) is not None)
        module.attn_mode = attn_mode
        if "attn_op" in module._modules:
            module._modules.pop("attn_op")
        if "attn_op" in module.__dict__:
            object.__delattr__(module, "attn_op")
        module.attn_op = build_attn_op(attn_mode, is_cross=is_cross)


# ---------------------------------------------------------------------------
# load_transformer
# ---------------------------------------------------------------------------

# Top-level pre-MoT action-head keys that have no lat twin in the MoT model
# and whose shapes shrink (output dim for action_embedder, input dim for
# action_proj_out) when act_dim < video_dim. The loader stashes these on
# the model so init_act_stream_from_lat can recover them.
_PRETRAINED_TOP_LEVEL_KEYS = [
    ("action_embedder.weight",  "_pretrained_action_embedder_weight"),
    ("action_embedder.bias",    "_pretrained_action_embedder_bias"),
    ("action_proj_out.weight",  "_pretrained_action_proj_out_weight"),
    ("action_proj_out.bias",    "_pretrained_action_proj_out_bias"),
]


def _read_ckpt_config(transformer_path):
    """Read the config.json of a HF-style model directory."""
    cfg_path = os.path.join(transformer_path, "config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path) as f:
        return json.load(f)


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    attn_mode=None,
    use_mot=True,
    mot_action_hidden_dim=768,
    mot_action_ffn_multiplier=4,
    enable_m3=False,
    m3_hidden_dim=768,
    m3_ffn_multiplier=4,
    m3_input_dim=16,
    m3_output_dim=16,
    m3_action_indices=None,
    m3_rope_h_offset=1,
    legacy_mirror=False,
    checkpoint_format="auto",
):
    """Build / load a ``WanTransformer3DModel``.

    Parameters
    ----------
    attn_mode : str | None
        Optional attention implementation override applied after load:
        ``'torch'`` / ``'flashattn'``.
        ``None`` keeps whatever the constructor (i.e. ``config.attn_mode``)
        chose. Param-loading is unaffected.
    use_mot : bool
        Must be ``True`` in this inference-only build. Only dual-stream
        MoT checkpoints (lat + act towers) are supported.
    mot_action_hidden_dim, mot_action_ffn_multiplier : int
        Act-tower hidden / FFN dims. Set ``mot_action_hidden_dim =
        num_attention_heads * attention_head_dim`` for symmetric same-dim
        MoT (act is a perfect twin of lat).
    legacy_mirror : bool
        Force the manual load path: read raw state dict → mirror to MoT
        layout → build empty model → shape-filter load → optional slim
        interp init. Required when loading a pre-MoT checkpoint into a
        MoT model with ``mot_action_hidden_dim != video_dim`` (the act
        keys mismatch shape and ``from_pretrained`` would error).
    checkpoint_format : str
        ``'auto'`` (default) | ``'abot_m05'`` | ``'wan22'``.
    """
    if not use_mot:
        raise ValueError(
            "This inference-only build supports only dual-stream MoT "
            "(use_mot=True) checkpoints."
        )
    if enable_m3:
        raise ValueError(
            "This inference-only build supports only dual-stream MoT "
            "(lat+act). Checkpoints with enable_m3=True are not supported."
        )
    from .wan22_ckpt_loader import (
        is_wan22_checkpoint,
        load_transformer_from_wan22,
        resolve_transformer_checkpoint_dir,
    )

    transformer_path = resolve_transformer_checkpoint_dir(transformer_path)

    fmt = checkpoint_format
    if fmt == "auto":
        fmt = "wan22" if is_wan22_checkpoint(transformer_path) else "abot_m05"

    if fmt == "wan22":
        _logger.info("Loading transformer from Wan2.2 checkpoint: %s",
                     transformer_path)
        model = load_transformer_from_wan22(
            transformer_path,
            torch_dtype=torch_dtype,
            torch_device="cpu",
            attn_mode=attn_mode,
            use_mot=use_mot,
            mot_action_hidden_dim=mot_action_hidden_dim,
            mot_action_ffn_multiplier=mot_action_ffn_multiplier,
        )
        _maybe_swap_attn_op(model, attn_mode)
        return model.to(torch_device)

    # Always route MoT loads through the manual mirror+filter+init path.
    # Two reasons from_pretrained is unsafe here:
    #   1) Pre-MoT ckpt → MoT model: diffusers' low_cpu_mem_usage path uses
    #      meta tensors and bypasses PyTorch's load_state_dict pre-hook, so
    #      the legacy-key mirror silently no-ops and .lat/.act params stay
    #      on meta (crashes at model.to(device)).
    #   2) MoT ckpt with mismatched act_dim → MoT model: from_pretrained
    #      strict-loads the act-side keys and crashes on shape mismatch.
    # The legacy_mirror path handles both: mirror is a no-op on already-MoT
    # state dicts, and the shape filter drops any mismatched act-side keys
    # so init_act_stream_from_lat can re-init them by interp.
    if not legacy_mirror and use_mot:
        legacy_mirror = True

    if legacy_mirror:
        # ---- Manual load path: raw → mirror → filter → load → init ----
        config = WanTransformer3DModel.load_config(transformer_path)
        config_dict = dict(config)
        if bool(config_dict.get("enable_m3", False)):
            raise ValueError(
                "This inference-only build cannot load an M3 checkpoint "
                f"(enable_m3=True in {transformer_path}/config.json)."
            )
        config_dict["use_mot"] = use_mot
        config_dict["mot_action_hidden_dim"] = mot_action_hidden_dim
        config_dict["mot_action_ffn_multiplier"] = mot_action_ffn_multiplier
        if attn_mode is not None:
            config_dict["attn_mode"] = attn_mode
        # Strip diffusers-internal config keys
        for k in list(config_dict.keys()):
            if k.startswith("_"):
                config_dict.pop(k)
        model = WanTransformer3DModel.from_config(config_dict)

        raw_sd = _load_raw_state_dict(transformer_path)

        # ----------------------------------------------------------------
        # Determine whether the ckpt is "native MoT" (act tower trained) vs
        # "pre-MoT" (only lat tower).
        #
        # Native MoT ckpt signature: contains keys like `blocks.X.attn1.act.*`.
        # Pre-MoT ckpt signature: only `blocks.X.attn1.to_q.*`, no .lat/.act split.
        #
        # This distinction determines the source of act-tower weights:
        #   - Native MoT: act tower loads directly from ckpt; must NOT be
        #     overwritten by lat interpolation.
        #   - Pre-MoT: ckpt has no act tower; init_act_stream_from_lat
        #     warm-starts it via lat interpolation.
        # ----------------------------------------------------------------
        # ===== Detailed key-distribution stats for debugging =====
        all_keys = list(raw_sd.keys())
        act_keys = [k for k in all_keys if ".act." in k or k.endswith(".act") or k == "scale_shift_table_act"]
        lat_keys = [k for k in all_keys if ".lat." in k or k.endswith(".lat") or k == "scale_shift_table_lat"]
        legacy_act_top = [k for k in all_keys if k in {
            "action_embedder.weight", "action_proj_out.weight",
            "scale_shift_table_act",
        } or k.startswith("condition_embedder_action.")]
        ckpt_has_native_mot_act = len(act_keys) > 0

        _logger.info("=" * 80)
        _logger.info("[load_transformer] ENTER legacy_mirror path")
        _logger.info("[load_transformer]   transformer_path = %s", transformer_path)
        _logger.info("[load_transformer]   use_mot = %s", use_mot)
        _logger.info("[load_transformer]   mot_action_hidden_dim = %d  inner_dim = %d  (asymmetric=%s)",
                     mot_action_hidden_dim, model.inner_dim,
                     mot_action_hidden_dim != model.inner_dim)
        _logger.info("[load_transformer]   ckpt total keys     = %d", len(all_keys))
        _logger.info("[load_transformer]   ckpt .act.* keys    = %d   (native MoT act tower? %s)",
                     len(act_keys), ckpt_has_native_mot_act)
        _logger.info("[load_transformer]   ckpt .lat.* keys    = %d", len(lat_keys))
        _logger.info("[load_transformer]   ckpt legacy-top-act = %d   (e.g. action_embedder)", len(legacy_act_top))
        if act_keys:
            _logger.info("[load_transformer]   sample .act keys   : %s", act_keys[:5])
        if lat_keys:
            _logger.info("[load_transformer]   sample .lat keys   : %s", lat_keys[:5])
        _logger.info("=" * 80)

        # Stash pre-MoT top-level action-head weights for interp init.
        # Only needed for pre-MoT ckpt: action_embedder in the ckpt is (video_dim, action_dim)
        # shaped, different from the MoT model's (act_dim, action_dim); stash it for manual interp recovery.
        # Native MoT ckpt already has action_embedder as (act_dim, action_dim) and loads directly.
        need_warmstart = (
            use_mot
            and mot_action_hidden_dim != model.inner_dim
            and not ckpt_has_native_mot_act
        )
        if need_warmstart:
            for k_src, attr in _PRETRAINED_TOP_LEVEL_KEYS:
                if k_src in raw_sd:
                    setattr(model, attr, raw_sd[k_src].clone())

        sd = mirror_legacy_state_dict(raw_sd) if use_mot else dict(raw_sd)

        model_sd = model.state_dict()
        filtered = {
            k: v for k, v in sd.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        skipped_keys = [
            k for k in sd
            if k in model_sd and tuple(sd[k].shape) != tuple(model_sd[k].shape)
        ]
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        n_show = 5
        _logger.info(
            "[load_transformer/legacy_mirror] loaded=%d skipped_shape=%d "
            "missing=%d unexpected=%d",
            len(filtered), len(skipped_keys),
            len(missing), len(unexpected))
        if missing:
            _logger.info("  missing[:%d]: %s", n_show, missing[:n_show])
        if unexpected:
            _logger.info("  unexpected[:%d]: %s", n_show, unexpected[:n_show])
        if skipped_keys:
            _logger.info("  skipped_shape[:%d]: %s",
                         n_show, skipped_keys[:n_show])

        # ============================================================
        # Decision point: whether to warm-start the act tower (interp from lat)
        # Only needed for pre-MoT ckpt. Native MoT: act tower is already loaded
        # during load_state_dict; calling init_act_stream_from_lat again would
        # overwrite the trained weights (historical bug).
        # ============================================================
        _logger.info("=" * 80)
        _logger.info("[load_transformer] DECISION POINT: warm-start act tower?")
        _logger.info("[load_transformer]   need_warmstart = %s", need_warmstart)
        _logger.info("[load_transformer]   reasons:")
        _logger.info("[load_transformer]     use_mot                              = %s  (need True)", use_mot)
        _logger.info("[load_transformer]     mot_action_hidden_dim != inner_dim   = %s  (need True; %d vs %d)",
                     mot_action_hidden_dim != model.inner_dim,
                     mot_action_hidden_dim, model.inner_dim)
        _logger.info("[load_transformer]     not ckpt_has_native_mot_act          = %s  (need True)",
                     not ckpt_has_native_mot_act)
        _logger.info("=" * 80)

        if need_warmstart:
            _logger.warning(
                "[load_transformer] pre-MoT ckpt detected -> CALLING "
                "init_act_stream_from_lat (this OVERWRITES any random-init act tower "
                "with lat-interp init; this is correct for pre-MoT, wrong for native MoT)"
            )
            init_act_stream_from_lat(model)
            for _, attr in _PRETRAINED_TOP_LEVEL_KEYS:
                if hasattr(model, attr):
                    delattr(model, attr)
            _logger.info("[load_transformer] warm-start finished")
        elif use_mot and mot_action_hidden_dim != model.inner_dim:
            # Native MoT ckpt + act_dim != video_dim: check act-tower key completeness
            sample_act_keys = [
                "blocks.0.attn1.act.to_q.weight",
                "blocks.0.ffn.act.net.0.proj.weight",
                "scale_shift_table_act",
                "action_embedder.weight",
                "action_proj_out.weight",
                "condition_embedder_action.time_embedder.linear_1.weight",
            ]
            present_act = [k for k in sample_act_keys if k in raw_sd]
            missing_act = [k for k in sample_act_keys if k not in raw_sd]
            _logger.info("[load_transformer] NATIVE MoT ckpt -> SKIPPING init_act_stream_from_lat")
            _logger.info("[load_transformer]   sample act keys present (%d/%d):",
                         len(present_act), len(sample_act_keys))
            for k in present_act:
                _logger.info("[load_transformer]     OK %s  shape=%s",
                             k, tuple(raw_sd[k].shape))
            if missing_act:
                _logger.warning(
                    "[load_transformer]   WARN sample act keys MISSING (will stay at random init!): %s",
                    missing_act,
                )

            # ===== Verify: sample act weights from model and compare with ckpt =====
            try:
                model_sd_check = model.state_dict()
                verify_keys = [k for k in present_act if k in model_sd_check]
                for vk in verify_keys[:3]:
                    a = raw_sd[vk].float().flatten()[:8]
                    b = model_sd_check[vk].float().flatten()[:8]
                    match = torch.allclose(a, b, atol=1e-5)
                    _logger.info(
                        "[load_transformer]   verify %s : match=%s  ckpt[:8]=%s  model[:8]=%s",
                        vk, match, a.tolist(), b.tolist(),
                    )
            except Exception as _e:
                _logger.warning("[load_transformer]   verify failed: %s", _e)
        else:
            _logger.info("[load_transformer] (symmetric MoT or non-MoT, no warm-start needed)")

        if torch_dtype is not None:
            model = model.to(torch_dtype)
    else:
        # Direct from_pretrained. The model's state-dict pre-hook auto-
        # mirrors pre-MoT keys to MoT layout when use_mot=True and shapes
        # match (i.e. same-dim symmetric MoT). For mismatched dims, the
        # caller must set legacy_mirror=True (or it's auto-promoted above).
        model = WanTransformer3DModel.from_pretrained(
            transformer_path,
            torch_dtype=torch_dtype,
            use_mot=use_mot,
            mot_action_hidden_dim=mot_action_hidden_dim,
            mot_action_ffn_multiplier=mot_action_ffn_multiplier,
        )

    _maybe_swap_attn_op(model, attn_mode)
    return model.to(torch_device)


# ---------------------------------------------------------------------------
# VAE streaming wrapper
# ---------------------------------------------------------------------------

def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if (hasattr(self.vae.config, "patch_size")
                and self.vae.config.patch_size is not None):
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk,
                           feat_cache=self.feat_cache,
                           feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
