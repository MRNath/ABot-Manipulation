import json
import os
import importlib
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from omegaconf import OmegaConf

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")

def _clone_cfg_node(cfg_node):
    if hasattr(cfg_node, "_cfg"):
        cfg_node = cfg_node._cfg
    return OmegaConf.create(OmegaConf.to_container(cfg_node, resolve=True))


def _resolve_train_test_split(data_cfg, mode: str):
    split_cfg = data_cfg.get("train_test_split")
    if split_cfg in (None, False):
        return None

    if hasattr(split_cfg, "_cfg"):
        split_cfg = split_cfg._cfg
    split_container = OmegaConf.to_container(split_cfg, resolve=True)
    if not isinstance(split_container, dict):
        raise ValueError("`datasets.vla_data.train_test_split` must be a mapping")

    if mode in split_container and isinstance(split_container[mode], dict):
        selected_split = dict(split_container[mode])
    else:
        selected_split = dict(split_container)

    if mode != "train":
        has_nested_mode = any(isinstance(v, dict) for v in split_container.values())
        if has_nested_mode and mode not in split_container:
            return None

    selected_split["mode"] = mode
    return OmegaConf.create(selected_split)


def build_dataloader(cfg, dataset_py="lerobot_datasets", mode: str = "train"):
    from ABot.dataloader.lerobot_datasets import get_vla_dataset, get_vla_dataset_test, collate_fn

    vla_dataset_cfg = _clone_cfg_node(cfg.datasets.vla_data)
    split_cfg = _resolve_train_test_split(vla_dataset_cfg, mode=mode)

    if split_cfg is not None:
        vla_dataset_cfg.train_test_split = split_cfg
    elif mode != "train":
        return None

    if mode == "train":
        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg, mode=mode)
    else:
        vla_dataset = get_vla_dataset_test(data_cfg=vla_dataset_cfg, mode=mode)

    vla_train_dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.datasets.vla_data.per_device_batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.datasets.vla_data.num_workers,
        # shuffle=True,
    )
    return vla_train_dataloader
