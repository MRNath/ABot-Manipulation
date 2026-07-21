# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Distributed-process helpers for multi-GPU inference."""
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


def _cast_floating_tensors_for_fsdp(model, param_dtype):
    """Cast every floating parameter/buffer to ``param_dtype`` in place."""
    with torch.no_grad():
        for param in model.parameters():
            if param.is_floating_point() and param.dtype != param_dtype:
                param.data = param.data.to(dtype=param_dtype)
        for name, buf in model.named_buffers():
            if not buf.is_floating_point() or buf.dtype == param_dtype:
                continue
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], buf.to(dtype=param_dtype))


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True):
    """Prepare ``model`` for inference and shard it when running multi-rank."""
    if eval_mode:
        model.eval().requires_grad_(False)
    # Only FSDP-shard when there are actually multiple ranks. Single-process
    # (world_size=1, e.g. server.py launched with torchrun
    # --nproc_per_node 1) gains no memory from sharding but still gets DTensor
    # weights, which then break forwards that bypass module.__call__ — e.g.
    # WanTransformerBlock.forward calls self.attn1.project_qkv_self(...) directly,
    # so attn1's FSDP pre-forward all-gather hook never fires and to_q.weight
    # stays as a sharded DTensor while the input is a plain Tensor.
    if dist.is_initialized() and dist.get_world_size() > 1:
        # Keep training behavior unchanged; only normalize dtype in eval/server mode.
        if eval_mode:
            # FSDP requires original parameter dtypes to be uniform before wrapping.
            _cast_floating_tensors_for_fsdp(model, param_dtype=param_dtype)
        dist.barrier()
        model = shard_fn(model)
    else:
        model.to(param_dtype)
        model.to(device)

    return model


def init_distributed(world_size, local_rank, rank, timeout: timedelta | None = None):
    """Pin the CUDA device and bring up the NCCL process group via env://."""
    torch.cuda.set_device(local_rank)
    kwargs: dict[str, Any] = dict(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    if timeout is not None:
        kwargs["timeout"] = timeout
    dist.init_process_group(**kwargs)

def dist_mean(local_tensor):
    """All-reduce average across ranks (no-op when not distributed)."""
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.AVG)
    return local_tensor

def dist_max(local_tensor):
    """All-reduce max across ranks (no-op when not distributed)."""
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)
    return local_tensor
