# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
import os
from dataclasses import dataclass, field

from .robocasa_train_cfg import RobocasaTrainConfig


@dataclass
class RobocasaTrainTestConfig(RobocasaTrainConfig):
    posttrain_model_name_or_path: str = field(
        default_factory=lambda: os.getenv("ROBOCASA_POSTTRAIN_MODEL_PATH_TEST", ""))


robocasa_train_test_cfg = RobocasaTrainTestConfig()
