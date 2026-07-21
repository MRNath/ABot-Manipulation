# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
from .demo_cfg import demo_cfg
from .robocasa_cfg import robocasa_cfg
from .robocasa_train_cfg import robocasa_train_cfg
from .robocasa_train_test_cfg import robocasa_train_test_cfg
from .libero_original_cfg import libero_original_cfg
from .libero_10_cfg import libero_10_cfg
from .libero_goal_cfg import libero_goal_cfg
from .libero_object_cfg import libero_object_cfg
from .libero_spatial_cfg import libero_spatial_cfg
from .libero_all_cfg import libero_all_cfg

WAM_CONFIGS = {
    "demo": demo_cfg,
    "robocasa": robocasa_cfg,
    "robocasa_train": robocasa_train_cfg,
    "robocasa_train_test": robocasa_train_test_cfg,
    "robocasa_train_test_atomic_target": robocasa_train_test_cfg,
    "libero": libero_original_cfg,
    "libero_original": libero_original_cfg,
    "libero_10": libero_10_cfg,
    "libero_goal": libero_goal_cfg,
    "libero_object": libero_object_cfg,
    "libero_spatial": libero_spatial_cfg,
    "libero_all": libero_all_cfg,
}
