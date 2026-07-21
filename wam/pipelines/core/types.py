# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Typed containers for dual-stream transformer input.

Replaces the nested-dict protocol (``{'latent_res_lst': {...}, 'action_res_lst': {...}}``)
with named dataclass fields so that call sites read ``input.video`` / ``input.action``
instead of string-keyed lookups.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StreamInput:
    """Input bundle for a single transformer stream (video or action)."""

    noisy_latents: torch.Tensor
    timesteps: torch.Tensor
    grid_id: torch.Tensor
    text_emb: torch.Tensor

    def to_dict(self) -> dict:
        """Convert to the dict layout expected by the transformer forward pass."""
        return {
            "noisy_latents": self.noisy_latents,
            "timesteps": self.timesteps,
            "grid_id": self.grid_id,
            "text_emb": self.text_emb,
        }


@dataclass
class TransformerInput:
    """Combined input carrying optional video and/or action streams."""

    video: StreamInput | None = None
    action: StreamInput | None = None
