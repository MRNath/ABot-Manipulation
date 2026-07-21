# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Unified attention implementation module.

This module provides two self-attention implementations that can be

seamlessly switched:

    - 'torch':     PyTorch built-in scaled_dot_product_attention

    - 'flashattn': Dao-AILab Flash Attention (dense)



Switch via ``WanAttention(attn_mode=...)``.

"""
from __future__ import annotations

from typing import Callable, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from flash_attn_interface import flash_attn_func  # type: ignore[reportMissingImports]
except Exception:
    try:
        from flash_attn import flash_attn_func  # type: ignore[reportMissingImports]
    except Exception:
        flash_attn_func = None



__all__ = [
    'build_attn_op',
    'custom_sdpa',
    'flash_attn_func',
    'SUPPORTED_ATTN_MODES',
]


SUPPORTED_ATTN_MODES = ('torch', 'flashattn')


def custom_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2)




# ---------------------------------------------------------------------------
# Factory: create attention op from attn_mode
# ---------------------------------------------------------------------------

def build_attn_op(attn_mode: str, is_cross: bool) -> Callable[..., torch.Tensor]:
    """Create an attention op from ``attn_mode``.

    - 'torch':     custom_sdpa (any scenario)
    - 'flashattn': flash_attn_func (any scenario, dense)
    """
    if attn_mode == 'torch':
        return custom_sdpa
    if attn_mode == 'flashattn':
        if flash_attn_func is None:
            raise RuntimeError("flash_attn_func is not available; install flash-attn first")
        return flash_attn_func
    raise ValueError(
        f"Unsupported attention mode: {attn_mode}, supported: {SUPPORTED_ATTN_MODES}"
    )
