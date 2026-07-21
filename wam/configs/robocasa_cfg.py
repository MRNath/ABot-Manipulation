# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .shared_config import WamConfig

# Align with offline ServeSteak / LeRobot: robot0_* image keys and dataset quantile norm.
@dataclass
class RobocasaConfig(WamConfig):
    attn_window: int = 72
    frame_chunk_size: int = 2
    height: int = 256
    width: int = 256
    action_per_frame: int = 16
    obs_cam_keys: List[str] = field(
        default_factory=lambda: [
            'observation.images.robot0_agentview_left',
            'observation.images.robot0_agentview_right',
            'observation.images.robot0_eye_in_hand'])
    num_inference_steps: int = 25
    action_snr_shift: float = 1.0
    used_action_channel_ids: List[int] = field(
        default_factory=lambda: list(range(7, 11)) + [29] + list(range(0, 6)) + [28])
    wan22_pretrained_model_name_or_path: str = field(
        default_factory=lambda: os.getenv("WAN22_PRETRAINED_PATH", "acvlab/abot-m0.5"))
    posttrain_model_name_or_path: str = field(
        default_factory=lambda: os.getenv("ROBOCASA_POSTTRAIN_MODEL_PATH", ""))
    norm_stat: Optional[Dict] = field(default_factory=lambda: {
        "q01": [
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.,
            -1.0, -1.0, -1.0, -1.0,
        ] + [0.] * 17 + [-1.0, -1.0],
        "q99": [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
        ] + [0.] * 17 + [1.0, 1.0],
    })


robocasa_cfg = RobocasaConfig()
