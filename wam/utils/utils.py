# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Utility helpers for the WAM pipeline.

* :func:`build_spatial_grid` — grid-id tensor via ``torch.stack`` + broadcasting.
* :func:`fold_into_patches` — un-patchify a flat sequence into spatial tensor
  via ``movedim`` + ``view`` ladder.
* :func:`save_async` — background-thread serialisation with a dispatch table.

All three keep the same positional argument order as the pipeline callers
expect, but the internal construction is intentionally different from the
upstream implementation.
"""
from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

_save_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

__all__ = ["build_spatial_grid", "fold_into_patches", "save_async"]


def build_spatial_grid(
    f: int,
    h: int,
    w: int,
    t: int,
    f_w: int = 1,
    f_shift: int = 0,
    action: bool = False,
) -> torch.Tensor:
    """Build a ``(4, f*h*w)`` grid-id tensor.

    Rows 0-2 are (temporal, height, width) coordinates; row 3 is the
    stream-type tag ``t`` (0 = video, 1 = action).

    For the action stream, h/w are collapsed to -1 and a per-h fractional
    offset is injected into the temporal axis so action tokens occupy a
    distinct position from video tokens.
    """
    f_idx = torch.arange(f_shift, f + f_shift) * f_w
    h_idx = torch.arange(h)
    w_idx = torch.arange(w)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")

    if action:
        frac = (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + frac
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1

    coords = torch.stack([ff, hh, ww], dim=0).flatten(1)
    tag = torch.full_like(coords[:1], t)
    return torch.cat([coords, tag], dim=0)


get_mesh_id = build_spatial_grid


def fold_into_patches(
    patch_size: tuple[int, int, int],
    data_seq: torch.Tensor,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
    batch_size: int = 1,
) -> torch.Tensor:
    """Un-patchify a flat sequence into ``(B, C, F, H, W)``.

    Uses ``movedim`` + explicit ``view`` ladder instead of the original
    ``reshape``/``permute``/``flatten`` chain.
    """
    p_t, p_h, p_w = patch_size
    f_post = latent_num_frames // p_t
    h_post = latent_height // p_h
    w_post = latent_width // p_w

    # Interpret flat memory as (B, F', H', W', p_t, p_h, p_w, C')
    x = data_seq.view(
        batch_size, f_post, h_post, w_post, p_t, p_h, p_w, -1
    )
    # Move C' from last (dim 7) to dim 1, keep spatial/patch order
    x = torch.movedim(x, 7, 1)
    # x is now (B, C', F', H', W', p_t, p_h, p_w)
    # Reorder to (B, C', F', p_t, H', p_h, W', p_w)
    x = x.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
    # Merge (F', p_t), (H', p_h), (W', p_w) → (F, H, W)
    x = x.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return x


data_seq_to_patch = fold_into_patches


def save_async(obj: Any, file_path: str | Path) -> None:
    """Serialise *obj* to *file_path* on a background thread.

    Dispatch table: tensors/tensor-dicts → CPU move + ``torch.save``;
    numpy arrays → copy + ``np.save``; fallback → ``torch.save``.
    """
    payload, writer = _dispatch(obj)
    _save_pool.submit(writer, payload, str(file_path))


def _dispatch(obj: Any) -> tuple[Any, Callable[[Any, str], None]]:
    if torch.is_tensor(obj):
        return (obj.cpu() if obj.is_cuda else obj, torch.save)
    if isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values()):
        moved = {k: v.cpu() if torch.is_tensor(v) else v for k, v in obj.items()}
        return (moved, torch.save)
    if isinstance(obj, np.ndarray):
        return (obj.copy(), np.save)
    return (obj, torch.save)
