# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Typed outputs returned by WAM pipelines."""
from dataclasses import dataclass


@dataclass
class WAMPipelineOutput:
    """Result of a single ``WAMPipeline`` call.

    Attributes:
        actions: Predicted action chunk, an array of shape
            ``(batch, num_frames, action_per_frame)`` per action group.
        latents: Denoised video latents produced this call, if requested.
        frames: Decoded video frames, populated only when video output is on.
    """

    actions: object
    latents: object | None = None
    frames: object | None = None
