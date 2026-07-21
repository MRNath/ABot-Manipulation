# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
import dataclasses
import json
import logging
import os
import re
import time
from PIL import Image
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.utils.torch_utils import randn_tensor

from ...distributed.fsdp import shard_model
from ...distributed.util import _configure_model
from ...modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from ...utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    logger,
    save_async,
)
from .action_normalizer import ActionNormalizer
from .cache_context import CacheContext
from .pipeline_output import WAMPipelineOutput
from .types import StreamInput, TransformerInput


def _safe_name(text: str | None) -> str:
    if text is None:
        return "default"
    value = str(text).strip()
    if not value:
        return "default"
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^0-9A-Za-z._-]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "default"


class WAMPipeline:
    """Diffusers-style inference pipeline for Abot-m0.5 video-action rollout.

    This class owns model components and the sampling path. The websocket
    server adapter lives in ``server_policy.py`` and preserves the historical
    ``infer(obs_dict)`` protocol used by evaluation clients.
    """

    @classmethod
    def from_config(cls, job_config):
        """Build a pipeline from a ``WAM_CONFIGS`` entry (dataclass)."""
        return cls(job_config)

    def __init__(self, job_config):
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.save_pred_video = bool(getattr(job_config, "save_pred_video", True))
        self.pred_video_fps = int(getattr(job_config, "pred_video_fps", 10))
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.infer_mode = getattr(job_config, "infer_mode", None)
        self.enable_offload = getattr(job_config, 'enable_offload', True)
        self._guidance_scale = float(getattr(job_config, "guidance_scale", 1.0))
        self._action_guidance_scale = float(getattr(job_config, "action_guidance_scale", 1.0))
        self._num_timesteps = None
        self._current_timestep = None
        self._interrupt = False
        self.video_processor = VideoProcessor(vae_scale_factor=1) if self.save_pred_video else None

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        transformer_path, load_kwargs = self._resolve_transformer_load(job_config)
        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=self.dtype,
            torch_device=self.device,
            **load_kwargs,
        )
        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            vae_half = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

        self.cache = CacheContext(self.transformer)

    def _resolve_transformer_load(self, job_config):
        """Pick the transformer directory and MoT parameters for loading.

        Returns ``(transformer_path, load_kwargs)`` where ``load_kwargs``
        carries the attn-mode override and dual-stream MoT settings, read
        from the checkpoint's ``config.json`` with ``job_config`` as
        fallback.
        """
        infer_mode = getattr(job_config, "infer_mode", None)
        is_inference_mode = infer_mode in {"server", "i2va"}
        if is_inference_mode:
            if not hasattr(job_config, "posttrain_model_name_or_path") or not job_config.posttrain_model_name_or_path:
                raise ValueError(
                    f"[ConfigError] infer_mode='{infer_mode}' requires 'posttrain_model_name_or_path' "
                    f"to be set in config, but it is missing or empty. "
                    f"Available config keys: {[f.name for f in dataclasses.fields(job_config)]}"
                )
            transformer_root = job_config.posttrain_model_name_or_path
        else:
            transformer_root = job_config.wan22_pretrained_model_name_or_path

        transformer_path = os.path.join(transformer_root, 'transformer')
        if not os.path.isdir(transformer_path):
            raise FileNotFoundError(
                f"[ConfigError] Transformer directory not found: {transformer_path}. "
                f"transformer_root={transformer_root}"
            )
        attn_mode_override = getattr(job_config, "attn_mode", None)
        if isinstance(attn_mode_override, str):
            attn_mode_override = attn_mode_override.strip() or None
        logger.info(
            f"[ModelLoad] mode={'inference' if is_inference_mode else 'train'} "
            f"(infer_mode={infer_mode}), transformer_root={transformer_root}, "
            f"attn_mode_override={attn_mode_override}"
        )
        ckpt_cfg_path = os.path.join(transformer_path, "config.json")
        ckpt_cfg = {}
        if os.path.isfile(ckpt_cfg_path):
            try:
                with open(ckpt_cfg_path, "r", encoding="utf-8") as f:
                    ckpt_cfg = json.load(f)
            except Exception as exc:
                logger.warning(
                    "[ModelLoad] failed to read transformer config %s: %s",
                    ckpt_cfg_path, exc)
        if bool(ckpt_cfg.get("enable_m3", False)):
            raise ValueError(
                "This inference-only build supports only dual-stream MoT "
                f"(lat+act), but {ckpt_cfg_path} has enable_m3=True."
            )
        resolved_use_mot = (
            bool(ckpt_cfg["use_mot"]) if "use_mot" in ckpt_cfg
            else bool(getattr(job_config, "use_mot", True))
        )
        if not resolved_use_mot:
            raise ValueError(
                "This inference-only build supports only dual-stream MoT "
                "(use_mot=True) checkpoints."
            )
        resolved_mot_dim = (
            int(ckpt_cfg["mot_action_hidden_dim"])
            if ckpt_cfg.get("mot_action_hidden_dim") is not None
            else int(getattr(job_config, "mot_action_hidden_dim", 768))
        )
        resolved_mot_mul = (
            int(ckpt_cfg["mot_action_ffn_multiplier"])
            if ckpt_cfg.get("mot_action_ffn_multiplier") is not None
            else int(getattr(job_config, "mot_action_ffn_multiplier", 4))
        )
        logger.info(
            "[ModelLoad] resolved dual-stream MoT params: "
            "use_mot=%s, mot_action_hidden_dim=%s, "
            "mot_action_ffn_multiplier=%s (source=%s)",
            resolved_use_mot,
            resolved_mot_dim,
            resolved_mot_mul,
            "ckpt_config.json" if ckpt_cfg else "job_config",
        )
        load_kwargs = dict(
            attn_mode=attn_mode_override,
            use_mot=resolved_use_mot,
            mot_action_hidden_dim=resolved_mot_dim,
            mot_action_ffn_multiplier=resolved_mot_mul,
        )
        return transformer_path, load_kwargs

    # ------------------------------------------------------------------
    # Diffusers-style properties
    # ------------------------------------------------------------------

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def action_guidance_scale(self):
        return self._action_guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self.guidance_scale > 1.0 or self.action_guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        """Encode the task prompt (and negative prompt for CFG) with T5.

        Returns ``(prompt_embeds, negative_prompt_embeds)``; the negative
        embeds are ``None`` when classifier-free guidance is disabled.
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    # ------------------------------------------------------------------
    # Input validation and latent preparation
    # ------------------------------------------------------------------

    def check_inputs(
        self,
        obs,
        prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        frame_st_id=0,
    ):
        """Validate call arguments before any denoising work happens."""
        if frame_st_id < 0:
            raise ValueError(f"`frame_st_id` must be non-negative, got {frame_st_id}.")
        height = getattr(self, "height", self.job_config.height)
        width = getattr(self, "width", self.job_config.width)
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 16 but are {height} and {width}."
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}.")
        if prompt_embeds is None and negative_prompt_embeds is not None:
            raise ValueError("`negative_prompt_embeds` cannot be passed without `prompt_embeds`.")
        if frame_st_id == 0:
            if obs is None or "obs" not in obs:
                raise ValueError("`obs` must contain an `obs` entry when `frame_st_id == 0`.")

    def prepare_latents(
        self,
        batch_size=1,
        num_channels_latents=48,
        num_frames=None,
        height=None,
        width=None,
        dtype=None,
        device=None,
        generator=None,
        latents=None,
    ):
        """Return initial video latents: ``latents`` if given, else fresh noise."""
        dtype = dtype or self.dtype
        device = device or self.device
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_frames = num_frames or self.job_config.frame_chunk_size
        height = height or self.latent_height
        width = width or self.latent_width
        shape = (batch_size, num_channels_latents, num_frames, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, "
                f"but requested an effective batch size of {batch_size}."
            )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def prepare_action_latents(
        self,
        batch_size=1,
        num_frames=None,
        dtype=None,
        device=None,
        generator=None,
        actions=None,
    ):
        """Return initial action latents: ``actions`` if given, else fresh noise."""
        dtype = dtype or self.dtype
        device = device or self.device
        if actions is not None:
            return actions.to(device=device, dtype=dtype)

        num_frames = num_frames or self.job_config.frame_chunk_size
        shape = (
            batch_size,
            self.job_config.action_dim,
            num_frames,
            self.action_per_frame,
            1,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, "
                f"but requested an effective batch size of {batch_size}."
            )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def _normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    # ------------------------------------------------------------------
    # Action normalization (delegated to ActionNormalizer)
    # ------------------------------------------------------------------

    def prepare_action_condition(self, action):
        """Map raw robot actions into the model's normalized action space."""
        return self.action_normalizer.to_model_space(action)

    def postprocess_actions(self, action):
        """Map model outputs back to raw robot action units/channels."""
        out = self.action_normalizer.to_action_space(action)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[postprocess_actions] output shape=%s, used_channel_ids n=%s",
                getattr(out, "shape", None),
                len(self.job_config.used_action_channel_ids),
            )
        return out

    # ------------------------------------------------------------------
    # Stream input building (replaces dict-based prepare_transformer_input)
    # ------------------------------------------------------------------

    def _build_stream_input(
        self,
        is_video: bool,
        model_input: torch.Tensor,
        t: float,
        frame_st_id: int,
        cond: torch.Tensor | None = None,
    ) -> StreamInput:
        """Construct a typed StreamInput for one transformer stream."""
        patch_size = self.job_config.patch_size
        if is_video:
            grid_id = get_mesh_id(
                model_input.shape[-3] // patch_size[0],
                model_input.shape[-2] // patch_size[1],
                model_input.shape[-1] // patch_size[2],
                0, 1, frame_st_id).to(self.device)
        else:
            grid_id = get_mesh_id(
                model_input.shape[-3],
                model_input.shape[-2],
                model_input.shape[-1],
                1, 1, frame_st_id,
                action=True).to(self.device)

        stream = StreamInput(
            noisy_latents=model_input,
            timesteps=torch.ones(
                [model_input.shape[2]],
                dtype=torch.float32,
                device=self.device) * t,
            grid_id=grid_id,
            text_emb=self.prompt_embeds.to(self.dtype).clone(),
        )

        if cond is not None:
            stream.noisy_latents[:, :, 0:1] = cond[:, :, 0:1]
            stream.timesteps[0:1] *= 0

        if not is_video:
            stream.noisy_latents[:, ~self.action_mask] *= 0

        return stream

    def _build_transformer_input(
        self,
        latent_input=None,
        action_input=None,
        latent_t=0,
        action_t=0,
        latent_cond=None,
        action_cond=None,
        frame_st_id=0,
    ) -> TransformerInput:
        """Build a TransformerInput carrying optional video and/or action streams."""
        logger.info(f"FRAME START ID: {frame_st_id}")
        video = action = None
        if latent_input is not None:
            video = self._build_stream_input(
                True, latent_input, latent_t, frame_st_id, latent_cond)
        if action_input is not None:
            action = self._build_stream_input(
                False, action_input, action_t, frame_st_id, action_cond)
        return TransformerInput(video=video, action=action)

    # ------------------------------------------------------------------
    # Classifier-free guidance
    # ------------------------------------------------------------------

    def _apply_cfg_to_stream(self, stream: StreamInput) -> dict:
        """Batch-duplicate a stream for CFG and return a transformer-ready dict."""
        if self.use_cfg:
            stream.noisy_latents = stream.noisy_latents.repeat(2, 1, 1, 1, 1)
            stream.text_emb = torch.cat([
                self.prompt_embeds.to(self.dtype).clone(),
                self.negative_prompt_embeds.to(self.dtype).clone(),
            ], dim=0)
            stream.grid_id = stream.grid_id[None].repeat(2, 1, 1)
            stream.timesteps = stream.timesteps[None].repeat(2, 1)
        else:
            stream.grid_id = stream.grid_id[None]
            stream.timesteps = stream.timesteps[None]
        return stream.to_dict()

    def _combine_cfg(
        self,
        prediction: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Combine conditional and unconditional predictions via guidance scale."""
        if scale > 1:
            return prediction[1:] + scale * (prediction[:1] - prediction[1:])
        return prediction[:1]

    # ------------------------------------------------------------------
    # Observation encoding (split into focused sub-methods)
    # ------------------------------------------------------------------

    def _collect_camera_frames(self, obs) -> list[torch.Tensor] | None:
        """Resize and stack observation frames per camera key."""
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None

        frames = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:
                    h_i, w_i = self.height, self.width
                else:
                    h_i, w_i = self.height // 2, self.width // 2
            else:
                h_i, w_i = self.height, self.width

            video_k = torch.from_numpy(
                np.stack([each[k] for each in images])
            ).float().permute(3, 0, 1, 2)
            video_k = F.interpolate(
                video_k, size=(h_i, w_i),
                mode='bilinear', align_corners=False).unsqueeze(0)
            frames.append(video_k)
        return frames

    def _encode_through_vae(self, frames: list[torch.Tensor]) -> torch.Tensor:
        """Encode camera frames through the streaming VAE."""
        if self.env_type == 'robotwin_tshape':
            videos_high = frames[0] / 255.0 * 2.0 - 1.0
            videos_lr = torch.cat(frames[1:], dim=0) / 255.0 * 2.0 - 1.0
            vae_dev = next(self.streaming_vae.vae.parameters()).device
            enc_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_dev).to(self.dtype))
            enc_lr = self.streaming_vae_half.encode_chunk(
                videos_lr.to(vae_dev).to(self.dtype))
            return torch.cat([
                torch.cat(enc_lr.split(1, dim=0), dim=-1),
                enc_high,
            ], dim=-2)
        else:
            videos = torch.cat(frames, dim=0) / 255.0 * 2.0 - 1.0
            vae_dev = next(self.streaming_vae.vae.parameters()).device
            return self.streaming_vae.encode_chunk(
                videos.to(vae_dev).to(self.dtype))

    def _normalize_vae_output(self, enc_out: torch.Tensor) -> torch.Tensor:
        """Apply VAE mean/std normalization and concatenate camera streams."""
        mu, _ = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self._normalize_latents(mu, latents_mean, 1.0 / latents_std)
        return torch.cat(mu_norm.split(1, dim=0), dim=-1).to(self.device)

    def prepare_observation_latents(self, obs):
        """Encode observation images into latent space via streaming VAE."""
        frames = self._collect_camera_frames(obs)
        if frames is None:
            return None
        enc_out = self._encode_through_vae(frames)
        return self._normalize_vae_output(enc_out)

    # ------------------------------------------------------------------
    # Unified denoising loop
    # ------------------------------------------------------------------

    def _run_denoising_phase(
        self,
        *,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: FlowMatchScheduler,
        guidance_scale: float,
        is_video: bool,
        frame_st_id: int,
        cond: torch.Tensor | None,
        frame_chunk_size: int,
        video_exec_step: int = -1,
    ) -> torch.Tensor:
        """Run a single denoising phase (video or action) to completion."""
        for i, t in enumerate(tqdm(timesteps, desc="video" if is_video else "action")):
            self._current_timestep = t
            is_final = i == len(timesteps) - 1

            stream = self._build_stream_input(
                is_video=is_video,
                model_input=latents,
                t=t,
                frame_st_id=frame_st_id,
                cond=cond,
            )

            noise_pred = self.transformer(
                self._apply_cfg_to_stream(stream),
                update_cache=1 if is_final else 0,
                cache_name=self.cache.name,
                action_mode=not is_video,
            )

            if is_video:
                should_step = (not is_final) or (video_exec_step != -1)
            else:
                should_step = not is_final

            if should_step:
                if is_video:
                    noise_pred = data_seq_to_patch(
                        self.job_config.patch_size, noise_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width,
                        batch_size=2 if self.use_cfg else 1)
                else:
                    noise_pred = rearrange(
                        noise_pred, 'b (f n) c -> b c f n 1',
                        f=frame_chunk_size)
                noise_pred = self._combine_cfg(noise_pred, guidance_scale)
                latents = scheduler.step(
                    noise_pred, t, latents, return_dict=False)

            if cond is not None:
                latents[:, :, 0:1] = cond

        return latents

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset_episode(self, prompt=None, episode_tag=None, episode_name=None):
        """Reset all per-episode state and (optionally) re-encode the prompt.

        Flushes the previous episode's pending video, clears caches,
        re-allocates the KV-cache pools for the current config, rebuilds the
        action normalizer, and prepares the output directory for the new
        episode.
        """
        logger.info('Reset.')
        prev_exp_save_root = getattr(self, 'exp_save_root', None)
        self.flush_pred_video_buffer(exp_save_root_for_decode=prev_exp_save_root)
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        self._guidance_scale = float(getattr(self.job_config, "guidance_scale", 1.0))
        self._action_guidance_scale = float(getattr(self.job_config, "action_guidance_scale", 1.0))
        self._current_timestep = None
        self._interrupt = False
        self.frame_st_id = 0
        self.init_latent = None

        self.cache.clear()
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        self.cache.allocate(
            attn_window=self.job_config.attn_window,
            latent_tokens=latent_token_per_chunk,
            action_tokens=action_token_per_chunk,
            dtype=self.dtype,
            device=self.device,
            batch_size=2 if self.use_cfg else 1,
        )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.action_normalizer = ActionNormalizer(
            q01=self.job_config.norm_stat['q01'],
            q99=self.job_config.norm_stat['q99'],
            used_channel_ids=self.job_config.used_action_channel_ids,
            inverse_channel_ids=self.job_config.inverse_used_action_channel_ids,
            action_dim=self.job_config.action_dim,
            norm_method=self.job_config.action_norm_method,
        )

        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.use_cfg,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        tag_part = _safe_name(episode_tag)
        name_source = episode_name if episode_name is not None else prompt
        name_part = _safe_name(name_source)
        self.exp_name = f"{tag_part}_{name_part}"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        self.pred_video_frames = []
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Predicted video saving
    # ------------------------------------------------------------------

    def save_pred_video_chunk(self, latents, frame_st_id):
        if not self.save_pred_video:
            return
        if self.enable_offload and self.infer_mode == 'server':
            return
        if self.video_processor is None:
            self.video_processor = VideoProcessor(vae_scale_factor=1)
        with torch.no_grad():
            decoded_video = self.decode_latents(latents, 'np')[0]
        if decoded_video is None or len(decoded_video) == 0:
            return
        if frame_st_id == 0 and len(decoded_video) > 1:
            decoded_video = decoded_video[1:]
        if len(decoded_video) == 0:
            return
        pred_video_dir = os.path.join(self.exp_save_root, "pred_video_chunks")
        os.makedirs(pred_video_dir, exist_ok=True)
        chunk_video_path = os.path.join(pred_video_dir, f"chunk_{frame_st_id:06d}.mp4")
        export_to_video(decoded_video, chunk_video_path, fps=self.pred_video_fps)
        self.pred_video_frames.extend([np.asarray(frame) for frame in decoded_video])

    def flush_pred_video_buffer(self, exp_save_root_for_decode=None):
        if not self.save_pred_video:
            return
        if self.enable_offload and self.infer_mode == 'server':
            if exp_save_root_for_decode is None or not os.path.isdir(exp_save_root_for_decode):
                return
            self.decode_saved_latents_to_video(exp_save_root_for_decode)
            return
        if not hasattr(self, "pred_video_frames") or len(self.pred_video_frames) == 0:
            return
        merged_video_path = os.path.join(self.exp_save_root, "pred_video.mp4")
        export_to_video(self.pred_video_frames, merged_video_path, fps=self.pred_video_fps)
        logger.info(f"Saved predicted video to: {merged_video_path}")
        self.pred_video_frames = []

    def decode_saved_latents_to_video(self, exp_save_root):
        if not exp_save_root or not os.path.isdir(exp_save_root):
            logger.info("No valid exp_save_root for deferred decode, skip.")
            return
        pred_video_dir = os.path.join(exp_save_root, "pred_video_chunks")
        os.makedirs(pred_video_dir, exist_ok=True)
        latent_files = []
        for name in os.listdir(exp_save_root):
            if not (name.startswith("latents_") and name.endswith(".pt")):
                continue
            matched = re.match(r"latents_(\d+)\.pt$", name)
            if matched is None:
                continue
            latent_files.append((int(matched.group(1)), os.path.join(exp_save_root, name)))
        latent_files.sort(key=lambda item: item[0])
        if len(latent_files) == 0:
            logger.info("No latent chunk files found, skip deferred decode.")
            return

        if self.video_processor is None:
            self.video_processor = VideoProcessor(vae_scale_factor=1)

        vae_was_offloaded = self.enable_offload and next(self.vae.parameters()).device.type != "cuda"
        if vae_was_offloaded:
            logger.info("Move VAE to GPU for deferred decode.")
            self.vae = self.vae.to(self.device).to(self.dtype)

        decoded_frames = []
        for frame_st_id, latent_path in latent_files:
            latents = torch.load(latent_path, map_location=self.device)
            with torch.no_grad():
                decoded_video = self.decode_latents(latents, 'np')[0]
            if decoded_video is None or len(decoded_video) == 0:
                continue
            if frame_st_id == 0 and len(decoded_video) > 1:
                decoded_video = decoded_video[1:]
            if len(decoded_video) == 0:
                continue
            chunk_video_path = os.path.join(pred_video_dir, f"chunk_{frame_st_id:06d}.mp4")
            export_to_video(decoded_video, chunk_video_path, fps=self.pred_video_fps)
            decoded_frames.extend([np.asarray(frame) for frame in decoded_video])

        if len(decoded_frames) > 0:
            merged_video_path = os.path.join(exp_save_root, "pred_video.mp4")
            export_to_video(decoded_frames, merged_video_path, fps=self.pred_video_fps)
            logger.info(f"Saved predicted video to: {merged_video_path}")
        else:
            logger.info("No decoded frames generated from latent chunks.")

        if vae_was_offloaded:
            self.vae = self.vae.to("cpu")
            torch.cuda.empty_cache()
            logger.info("Move VAE back to CPU after deferred decode.")

    # ------------------------------------------------------------------
    # Main inference entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        obs,
        frame_st_id=0,
        prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        generator=None,
        latents=None,
        actions=None,
        return_dict=True,
    ):
        """Run one video+action denoising chunk for the current observation.

        At ``frame_st_id == 0`` the observation images are encoded into the
        initial (conditioning) latent; subsequent calls reuse the KV cache.
        Returns a ``WAMPipelineOutput`` with postprocessed actions and the
        denoised video latents.
        """
        self.check_inputs(
            obs=obs,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            frame_st_id=frame_st_id,
        )
        if prompt is not None or prompt_embeds is not None:
            if prompt_embeds is None:
                prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                    prompt=prompt,
                    negative_prompt=None,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    num_videos_per_prompt=1,
                    prompt_embeds=None,
                    negative_prompt_embeds=negative_prompt_embeds,
                    max_sequence_length=512,
                    device=self.device,
                    dtype=self.dtype,
                )
            self.prompt_embeds = prompt_embeds
            self.negative_prompt_embeds = negative_prompt_embeds

        frame_chunk_size = self.job_config.frame_chunk_size
        init_latent = None
        if frame_st_id == 0:
            init_latent = self.prepare_observation_latents(obs)
            self.init_latent = init_latent

        latents = self.prepare_latents(
            batch_size=1,
            num_channels_latents=48,
            num_frames=frame_chunk_size,
            height=self.latent_height,
            width=self.latent_width,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
            latents=latents,
        )
        actions = self.prepare_action_latents(
            batch_size=1,
            num_frames=frame_chunk_size,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
            actions=actions,
        )

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)
        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0, 1),
            mode='constant',
            value=0)
        self._num_timesteps = len(timesteps) + len(action_timesteps)

        video_cond = init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 else None
        action_cond = (
            torch.zeros(
                [1, self.job_config.action_dim, 1, self.action_per_frame, 1],
                device=self.device, dtype=self.dtype)
            if frame_st_id == 0 else None
        )

        with torch.no_grad():
            latents = self._run_denoising_phase(
                latents=latents,
                timesteps=timesteps,
                scheduler=self.scheduler,
                guidance_scale=self.guidance_scale,
                is_video=True,
                frame_st_id=frame_st_id,
                cond=video_cond,
                frame_chunk_size=frame_chunk_size,
                video_exec_step=video_step,
            )
            actions = self._run_denoising_phase(
                latents=actions,
                timesteps=action_timesteps,
                scheduler=self.action_scheduler,
                guidance_scale=self.action_guidance_scale,
                is_video=False,
                frame_st_id=frame_st_id,
                cond=action_cond,
                frame_chunk_size=frame_chunk_size,
            )

        self._current_timestep = None
        actions[:, ~self.action_mask] *= 0

        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))
        self.save_pred_video_chunk(latents, frame_st_id)

        actions = self.postprocess_actions(actions)
        torch.cuda.empty_cache()
        if not return_dict:
            return actions, latents
        return WAMPipelineOutput(actions=actions, latents=latents)

    @torch.no_grad()
    def infer_chunk(self, obs, frame_st_id=0):
        """Thin alias for ``__call__`` kept for older client integrations."""
        return self(obs, frame_st_id=frame_st_id)

    # ------------------------------------------------------------------
    # KV cache pre-computation
    # ------------------------------------------------------------------

    def prepare_kv_cache(self, obs):
        """Prefill the KV cache from the current observation window.

        Encodes the observation frames (concatenated with the episode's
        initial latent at ``frame_st_id == 0``) and the robot state, then
        runs both transformer streams with ``update_cache=2`` so later
        ``__call__`` chunks attend to this context. Advances
        ``self.frame_st_id`` by the number of latent frames consumed.
        """
        self.cache.clear_predictions()
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self.prepare_observation_latents(obs)
        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        action_model_input = self.prepare_action_condition(obs['state'])
        action_model_input = action_model_input.to(latent_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )

        tx_input = self._build_transformer_input(
            latent_input=latent_model_input,
            action_input=action_model_input,
            frame_st_id=self.frame_st_id,
        )

        with torch.no_grad():
            self.transformer(
                self._apply_cfg_to_stream(tx_input.video),
                update_cache=2,
                cache_name=self.cache.name,
                action_mode=False,
            )
            self.transformer(
                self._apply_cfg_to_stream(tx_input.action),
                update_cache=2,
                cache_name=self.cache.name,
                action_mode=True,
            )
        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]

    # ------------------------------------------------------------------
    # Latent decoding
    # ------------------------------------------------------------------

    def decode_latents(self, latents, output_type):
        """Decode video latents back to frames (undoing VAE normalization)."""
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video

    def load_initial_observation(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs

    # ------------------------------------------------------------------
    # i2va generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self):
        """Run a full image-to-video+action (i2va) rollout and export demo.mp4."""
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self.reset_episode(self.job_config.prompt)
        init_obs = self.load_initial_observation()
        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            frame_st_id = chunk_id * self.job_config.frame_chunk_size
            logger.info(f"################# I2VA Infer One Chunk #################")
            t_start = time.time()
            output = self(init_obs, frame_st_id=frame_st_id)
            actions, latents = output.actions, output.latents
            t_end = time.time()

            elapsed = t_end - t_start
            logger.info(f"[Shape] (i2va) action numpy shape: {getattr(actions, 'shape', None)}")
            try:
                _, num_frames, action_per_frame = actions.shape
                total_action_steps = num_frames * action_per_frame
                logical_time = num_frames / 10.0
            except Exception:
                num_frames = None
                action_per_frame = None
                total_action_steps = None
                logical_time = None

            hz_wall = total_action_steps / elapsed if elapsed and total_action_steps is not None else float('inf')
            hz_logical = (total_action_steps / logical_time
                          if logical_time and logical_time > 0 else float('inf'))

            logger.info(
                f"[Perf] (i2va) Chunk starting at frame {frame_st_id}: "
                f"elapsed={elapsed:.3f}s, "
                f"frames_in_chunk={num_frames}, "
                f"action_per_frame={action_per_frame}, "
                f"actions_in_chunk={total_action_steps}, "
                f"actions_per_second_wall={hz_wall:.2f}, "
                f"actions_per_second_logical={hz_logical:.2f}"
            )

            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.cache.clear()
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()

        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)

        decoded_video = self.decode_latents(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=self.pred_video_fps)
