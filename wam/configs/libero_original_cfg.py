# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""LIBERO original cfg: old-dataset (acvlab/abot-m0.5) with hardcoded norm_stat.

The 4-subset + libero_all configs (libero_{10,goal,object,spatial,all}_cfg.py)
load norm_stat from libero_norm_stats.json via LiberoConfig(subset=...).
This cfg predates that JSON and hardcodes the stats directly.
"""
from .shared_config import LiberoConfig

libero_original_cfg = LiberoConfig(norm_stat={
    "q01": [
        -0.6589285731315613,
        -0.84375,
        -0.9375,
        -0.12107142806053162,
        -0.15964286029338837,
        -0.26571428775787354,
        -1.0
    ] + [0.] * 23,
    "q99": [
        0.8999999761581421,
        0.8544642925262451,
        0.9375,
        0.17142857611179352,
        0.1842857152223587,
        0.34392857551574707,
        1.0
    ] + [0.] * 23,
})
