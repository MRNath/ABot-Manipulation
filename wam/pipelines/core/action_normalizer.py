# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Standalone quantile-based action normalizer.

Encapsulates the conversion between raw action arrays (numpy, channel-reordered)
and normalized model-space tensors.  Implemented as an ``nn.Module`` so that
the quantile buffers participate in ``state_dict`` and device migration.

Uses ``torch.lerp`` for the interpolation and computes in ``float64`` for
numerical stability before casting back to the working precision.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-6


class ActionNormalizer(nn.Module):
    """Quantile-based action normalization between model space and action space.

    *to_model_space* mirrors the original ``prepare_action_condition`` path:
    numpy → tensor → pad → channel-reorder → quantile-normalize → add batch/width dims.

    *to_action_space* mirrors the original ``postprocess_action`` path:
    squeeze → quantile-denormalize → numpy → channel-select.
    """

    def __init__(
        self,
        q01: list[float],
        q99: list[float],
        used_channel_ids: list[int],
        inverse_channel_ids: list[int],
        action_dim: int,
        norm_method: str = "quantiles",
    ) -> None:
        super().__init__()
        self.register_buffer(
            "_q01", torch.tensor(q01, dtype=torch.float64).reshape(-1, 1, 1)
        )
        self.register_buffer(
            "_q99", torch.tensor(q99, dtype=torch.float64).reshape(-1, 1, 1)
        )
        self.used_ids = used_channel_ids
        self.inverse_ids = inverse_channel_ids
        self.dim = action_dim
        self.norm_method = norm_method

    def to_model_space(self, raw_action: np.ndarray) -> torch.Tensor:
        """Convert a raw action array to a normalized model input tensor.

        Returns shape ``(B=1, C, F, H, W=1)``.
        """
        action = torch.from_numpy(raw_action)
        action = F.pad(action, [0, 0, 0, 0, 0, 1], mode="constant", value=0)
        action = action[self.inverse_ids]

        if self.norm_method == "quantiles":
            action = self._normalize_to_model(action)
        else:
            raise NotImplementedError

        return action.unsqueeze(0).unsqueeze(-1)

    def to_action_space(self, model_output: torch.Tensor) -> np.ndarray:
        """Convert a model output tensor back to a raw action array.

        Input shape ``(B, C, F, H, W)`` → output shape ``(C_used, F, H)``.
        """
        action = model_output.cpu()
        action = action[0, ..., 0]  # C, F, H

        if self.norm_method == "quantiles":
            action = self._denormalize_to_action(action)
        else:
            raise NotImplementedError

        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.used_ids]

    def _normalize_to_model(self, action: torch.Tensor) -> torch.Tensor:
        q01 = self._q01.to(action.device, dtype=torch.float64)
        q99 = self._q99.to(action.device, dtype=torch.float64)
        span = (q99 - q01).clamp(min=_EPS)
        t = (action.double() - q01) / span
        normalized = torch.lerp(
            torch.full_like(t, -1.0), torch.full_like(t, 1.0), t
        )
        return normalized.to(torch.float32)

    def _denormalize_to_action(self, action: torch.Tensor) -> torch.Tensor:
        q01 = self._q01.to(action.device, dtype=torch.float64)
        q99 = self._q99.to(action.device, dtype=torch.float64)
        span = (q99 - q01).clamp(min=_EPS)
        t = (action.double() + 1.0) / 2.0
        return torch.lerp(q01, q99, t).to(torch.float32)
