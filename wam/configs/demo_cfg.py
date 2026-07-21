# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .shared_config import WamConfig


@dataclass
class DemoConfig(WamConfig):
    height: int = 256
    width: int = 256
    action_per_frame: int = 8
    obs_cam_keys: List[str] = field(
        default_factory=lambda: ['observation.images.top', 'observation.images.wrist'])
    num_inference_steps: int = 5
    action_num_inference_steps: int = 10
    action_snr_shift: float = 1.0
    used_action_channel_ids: List[int] = field(
        default_factory=lambda: list(range(0, 5)) + list(range(28, 29)))
    wan22_pretrained_model_name_or_path: str = field(
        default_factory=lambda: os.getenv("WAN22_PRETRAINED_PATH", "acvlab/abot-m0.5"))
    posttrain_model_name_or_path: str = field(
        default_factory=lambda: os.getenv("DEMO_POSTTRAIN_MODEL_PATH", ""))
    norm_stat: Optional[Dict] = field(default_factory=lambda: {
        "q01": [
            -90.60303497314453,
            -98.73043060302734,
            -79.9008560180664,
            48.95470428466797,
            -32.794578552246094,
        ] + [0.] * 23 + [0.8250824809074402, 0],
        "q99": [
            71.735107421875,
            65.89081573486328,
            92.87967681884766,
            100.0,
            22.784151077270508,
        ] + [0.] * 23 + [100.0, 0],
    })


demo_cfg = DemoConfig()
