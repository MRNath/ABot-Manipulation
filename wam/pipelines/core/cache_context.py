# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""KV cache lifecycle manager for the dual-stream transformer.

Encapsulates cache name and all cache operations (clear, clear-predictions,
allocate) so the pipeline never touches raw string cache names or calls
transformer cache methods directly.
"""
from __future__ import annotations

import torch


class CacheContext:
    """Manages KV cache lifecycle for the dual-stream MoT transformer.

    The cache name is an arbitrary string used as a dict key inside the
    transformer.  Encapsulating it here prevents scattering the literal
    across pipeline methods.
    """

    def __init__(self, transformer, name: str = "ctx"):
        self._transformer = transformer
        self.name = name

    def clear(self):
        """Drop all cached KV pairs (observed + predicted)."""
        self._transformer.clear_cache(self.name)

    def clear_predictions(self):
        """Drop only predicted-step KV pairs, keeping observed-step caches."""
        self._transformer.clear_pred_cache(self.name)

    def allocate(
        self,
        attn_window: int,
        latent_tokens: int,
        action_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
    ):
        """Pre-allocate empty cache slots for the upcoming episode."""
        self._transformer.create_empty_cache(
            self.name,
            attn_window,
            latent_tokens,
            action_tokens,
            dtype=dtype,
            device=device,
            batch_size=batch_size,
        )
