# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Websocket-facing policy adapter for remote evaluation.

Wire protocol (msgpack-encoded dicts, see the ``wam_client`` package):

- ``{"reset": True, "prompt": str?, "episode_tag": str?, "episode_name": str?}``
  resets episode state and starts a new rollout.
- ``{"compute_kv_cache": True, ...observation}`` prefills the KV cache from
  the initial observation without producing actions.
- ``{"flush_pred_video": True}`` decodes and writes any buffered predicted
  video of the finished episode.
- any other dict is treated as one observation and answered with
  ``{"action": np.ndarray}``, the next predicted action chunk.
"""
import os
import time

import torch

from ...utils import logger
from .pipeline_wam import WAMPipeline


class WAMServerPolicy(WAMPipeline):
    """Dispatch websocket requests to the episode APIs of ``WAMPipeline``."""

    @torch.no_grad()
    def infer(self, obs):
        if obs.get("reset", False):
            return self._handle_reset(obs)
        if obs.get("flush_pred_video", False):
            return self._handle_flush_pred_video()
        if obs.get("compute_kv_cache", False):
            return self._handle_kv_cache_prefill(obs)
        return self._handle_infer_chunk(obs)

    def _handle_reset(self, obs):
        logger.info("Reset server")
        self.reset_episode(
            prompt=obs.get("prompt"),
            episode_tag=obs.get("episode_tag"),
            episode_name=obs.get("episode_name"),
        )
        return {}

    def _handle_flush_pred_video(self):
        logger.info("Flush predicted video")
        root = getattr(self, "exp_save_root", None)
        if self.save_pred_video and root and os.path.isdir(root):
            if self.enable_offload and self.infer_mode == "server":
                self.decode_saved_latents_to_video(root)
            else:
                self.flush_pred_video_buffer()
        return {}

    def _handle_kv_cache_prefill(self, obs):
        logger.info("Compute KV cache")
        self.prepare_kv_cache(obs)
        return {}

    def _handle_infer_chunk(self, obs):
        logger.info("Infer one chunk")
        t_start = time.time()
        action = self(obs, frame_st_id=self.frame_st_id).actions
        self._log_chunk_perf(time.time() - t_start, action)
        return {"action": action}

    def _log_chunk_perf(self, elapsed, action):
        logger.info(f"[Shape] action numpy shape: {getattr(action, 'shape', None)}")
        try:
            _, num_frames, action_per_frame = action.shape
            total_action_steps = num_frames * action_per_frame
            logical_time = num_frames / 10.0
        except Exception:
            num_frames = None
            action_per_frame = None
            total_action_steps = None
            logical_time = None

        hz_wall = total_action_steps / elapsed if elapsed > 0 else float("inf")
        hz_logical = (
            total_action_steps / logical_time
            if logical_time is not None and logical_time > 0 else float("inf")
        )
        logger.info(
            f"[Perf] Chunk starting at frame {self.frame_st_id}: "
            f"elapsed={elapsed:.3f}s, "
            f"frames_in_chunk={num_frames}, "
            f"action_per_frame={action_per_frame}, "
            f"actions_in_chunk={total_action_steps}, "
            f"actions_per_second_wall={hz_wall:.2f}, "
            f"actions_per_second_logical={hz_logical:.2f}"
        )
