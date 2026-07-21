# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
import os
from dataclasses import dataclass, field
from typing import List

from .robocasa_cfg import RobocasaConfig


@dataclass
class RobocasaTrainConfig(RobocasaConfig):
    # Dataset / dataloader
    dataset_path: str = field(
        default_factory=lambda: os.getenv("ROBOCASA_DATASET_PATH", "data/robocasa365/pretrain"))
    empty_emb_path: str = ''  # computed in __post_init__
    load_worker: int = field(
        default_factory=lambda: int(os.getenv("ROBOCASA_LOAD_WORKER", "8")))
    save_interval: int = field(
        default_factory=lambda: int(os.getenv("ROBOCASA_SAVE_INTERVAL", "200")))
    gc_interval: int = 50
    cfg_prob: float = 0.1

    # Random offset sampling for multi-offset latent cache. Default [0] keeps the
    # legacy single-offset filename layout. To enable time-phase augmentation,
    # override in the specific training cfg together with `latent_dir_name`, e.g.:
    #   cfg.random_offsets   = [0, 3, 6, 9, 12, 15]
    #   cfg.latent_dir_name  = "latent_no_resize_multi_offset"
    random_offsets: List[int] = field(default_factory=lambda: [0])

    # Training parameters
    learning_rate: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_steps: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_steps: int = 160000

    def __post_init__(self):
        super().__post_init__()
        # empty_emb_path depends on dataset_path (resolved above)
        self.empty_emb_path = os.getenv(
            "ROBOCASA_EMPTY_EMB_PATH",
            os.path.join(self.dataset_path, 'empty_emb.pt'),
        )
        # Conditional env-var overrides for sampling/prefetch.
        # ROBOCASA_LENGTH_WEIGHTED=0 falls back to uniform sampling.
        _pf_env = os.getenv("ROBOCASA_PREFETCH_FACTOR")
        if _pf_env is not None:
            self.prefetch_factor = int(_pf_env)
        _lw_env = os.getenv("ROBOCASA_LENGTH_WEIGHTED")
        if _lw_env is not None:
            self.length_weighted_sampling = _lw_env not in ("0", "false", "False")
        _lp_env = os.getenv("ROBOCASA_LENGTH_SAMPLE_POWER")
        if _lp_env is not None:
            self.length_sample_power = float(_lp_env)
        _ms_env = os.getenv("ROBOCASA_LENGTH_SAMPLE_MIN_SPAN")
        if _ms_env is not None:
            self.length_sample_min_span = int(_ms_env)
        _lb_env = os.getenv("ROBOCASA_LENGTH_BUCKETED")
        if _lb_env is not None:
            self.length_bucketed = _lb_env not in ("0", "false", "False")
        _tw_env = os.getenv("ROBOCASA_TASK_WEIGHTS_TXT")
        if _tw_env:
            self.task_sample_weights_txt = _tw_env


robocasa_train_cfg = RobocasaTrainConfig()
