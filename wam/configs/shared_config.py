# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Base dataclass configs for WAM inference.

``WamConfig`` is the root: shared infra, runtime fields (mutated by
``server.py``), and common benchmark defaults. ``LiberoConfig`` adds the
LIBERO-specific norm-stat loader so the six libero subset configs collapse
to one-liners.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class WamConfig:
    """Root config: shared infra + runtime + common benchmark defaults.

    Runtime fields (``rank``, ``local_rank``, ``world_size``, ``attn_mode``)
    are declared here so ``server.py`` can assign them on the instance in
    place — the dataclass is intentionally non-frozen for this reason.
    """

    # ---- Serving / runtime (mutated by server.py) ----
    host: str = '0.0.0.0'
    port: int = 29536
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    attn_mode: Optional[str] = None

    # ---- Inference mode ----
    infer_mode: str = field(
        default_factory=lambda: os.getenv("VA_INFER_MODE", "server"))

    # ---- Model loading ----
    wan22_pretrained_model_name_or_path: str = "acvlab/abot-m0.5"
    posttrain_model_name_or_path: str = ""

    # ---- Video / VAE ----
    param_dtype: torch.dtype = torch.bfloat16
    save_root: str = './train_out'
    save_pred_video: bool = True
    pred_video_fps: int = 10
    patch_size: tuple = (1, 2, 2)
    enable_offload: bool = False

    # ---- History / sampling (training-side, kept for parity) ----
    min_history_latent_frames: int = 8
    length_weighted_sampling: bool = True
    length_sample_power: float = 0.5
    length_sample_min_span: int = 0
    length_bucketed: bool = True
    length_bucket_count: int = 8
    task_sample_weights_txt: Optional[str] = None
    prefetch_factor: int = 2

    # ---- Common benchmark defaults (libero values; overridden by demo/robocasa) ----
    attn_window: int = 30
    frame_chunk_size: int = 4
    env_type: str = 'none'
    height: int = 128
    width: int = 128
    action_dim: int = 30
    action_per_frame: int = 4
    obs_cam_keys: List[str] = field(
        default_factory=lambda: ['observation.images.agentview_rgb',
                                 'observation.images.eye_in_hand_rgb'])
    guidance_scale: float = 5
    action_guidance_scale: float = 1
    num_inference_steps: int = 20
    video_exec_step: int = -1
    action_num_inference_steps: int = 50
    snr_shift: float = 5.0
    action_snr_shift: float = 0.05
    used_action_channel_ids: List[int] = field(
        default_factory=lambda: list(range(0, 7)))
    inverse_used_action_channel_ids: List[int] = field(default_factory=list)
    action_norm_method: str = 'quantiles'
    norm_stat: Optional[Dict] = None

    def __post_init__(self):
        if self.used_action_channel_ids and not self.inverse_used_action_channel_ids:
            inv = [len(self.used_action_channel_ids)] * self.action_dim
            for i, j in enumerate(self.used_action_channel_ids):
                inv[j] = i
            self.inverse_used_action_channel_ids = inv


@dataclass
class LiberoConfig(WamConfig):
    """LIBERO benchmark config.

    Pass ``subset`` ('libero_10' | 'libero_goal' | 'libero_object' |
    'libero_spatial' | 'libero_all') to auto-load ``norm_stat`` from
    ``libero_norm_stats.json``. Leave ``subset=''`` and pass ``norm_stat``
    directly for custom/hardcoded stats (used by ``libero_original_cfg``).
    """
    subset: str = ''
    wan22_pretrained_model_name_or_path: str = field(
        default_factory=lambda: os.getenv("WAN22_PRETRAINED_PATH")
                               or os.getenv("WAN22_BASE_CKPT")
                               or "acvlab/abot-m0.5")
    posttrain_model_name_or_path: str = field(
        default_factory=lambda: os.getenv("LIBERO_POSTTRAIN_MODEL_PATH")
                               or os.getenv("POSTTRAIN_CKPT")
                               or "")

    def __post_init__(self):
        super().__post_init__()
        if self.subset and self.norm_stat is None:
            from ._norm_stat_loader import load_libero_norm_stat
            self.norm_stat = load_libero_norm_stat(
                self.subset, action_dim=self.action_dim, real_dim=7)
