# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Helpers for FSDP v2 (fully_shard) hybrid / HSDP meshes: replicate across nodes, shard within node."""
from __future__ import annotations

import os

import torch.distributed as dist


def resolve_gpus_per_node(config_value: int | None, world_size: int) -> int:
    if world_size <= 1:
        return 1
    if config_value is not None and int(config_value) > 0:
        return int(config_value)
    env = os.environ.get("LOCAL_WORLD_SIZE")
    if env is not None and int(env) > 0:
        return int(env)
    raise ValueError(
        "fsdp_mode=hybrid_node requires --fsdp-gpus-per-node or env LOCAL_WORLD_SIZE when WORLD_SIZE>1 "
        f"(world_size={world_size})"
    )


def build_hybrid_fsdp_mesh(world_size: int, gpus_per_node: int):
    """
    2D CUDA DeviceMesh (HSDP): dim0=replicate across nodes, dim1=shard within node.
    Requires global rank layout: rank = node_id * gpus_per_node + local_gpu (torchrun default).
    """
    if not dist.is_initialized():
        raise RuntimeError("build_hybrid_fsdp_mesh requires an initialized process group")
    ws = dist.get_world_size()
    if ws != world_size:
        world_size = ws
    if gpus_per_node <= 0 or world_size % gpus_per_node != 0:
        raise ValueError(f"Invalid hybrid FSDP mesh: world_size={world_size} gpus_per_node={gpus_per_node}")
    nnodes = world_size // gpus_per_node
    try:
        from torch.distributed.device_mesh import init_device_mesh
    except ImportError:
        from torch.distributed._tensor.device_mesh import init_device_mesh
    shape = (nnodes, gpus_per_node)
    try:
        return init_device_mesh("cuda", shape, mesh_dim_names=("replicate", "shard"))
    except TypeError:
        return init_device_mesh("cuda", shape)
