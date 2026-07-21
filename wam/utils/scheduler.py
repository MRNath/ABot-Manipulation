# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Flow-matching noise schedule for the WAM diffusion pipeline.

The schedule is split into two concerns:

* :class:`SigmaSchedule` — pure sigma-timeline computation (linspace → shift →
  terminal rescale → optional flip).  No state beyond config.
* :class:`FlowMatchScheduler` — owns the schedule and implements the
  Euler-style update / add-noise / training-weight helpers used by the
  pipeline.

This split replaces the original monolithic scheduler with a composable
design so that alternative sigma parameterisations can be plugged in
without touching the step logic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class SigmaSchedule:
    """Declarative description of a flow-matching sigma timeline.

    All fields are configuration only; the actual tensors are materialised
    by :meth:`build`.
    """

    sigma_max: float = 1.0
    sigma_min: float = 0.003 / 1.002
    shift: float = 3.0
    inverse_timesteps: bool = False
    extra_one_step: bool = False
    reverse_sigmas: bool = False
    exponential_shift: bool = False
    exponential_shift_mu: float | None = None
    shift_terminal: float | None = None

    # ---- sigma construction ------------------------------------------

    def _linspace_sigmas(self, num_steps: int, denoising_strength: float) -> torch.Tensor:
        start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        count = num_steps + 1 if self.extra_one_step else num_steps
        sigmas = torch.linspace(start, self.sigma_min, count)
        return sigmas[:-1] if self.extra_one_step else sigmas

    def _apply_shift(self, sigmas: torch.Tensor, dynamic_len: int | None) -> torch.Tensor:
        if self.exponential_shift:
            mu = self._compute_shift_mu(dynamic_len) if dynamic_len is not None else self.exponential_shift_mu
            return torch.exp(torch.tensor(mu)) / (torch.exp(torch.tensor(mu)) + (1 / sigmas - 1))
        return self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

    def _apply_terminal_rescale(self, sigmas: torch.Tensor) -> torch.Tensor:
        if self.shift_terminal is None:
            return sigmas
        one_minus_z = 1 - sigmas
        scale = one_minus_z[-1] / (1 - self.shift_terminal)
        return 1 - (one_minus_z / scale)

    def build(self, num_steps: int, denoising_strength: float = 1.0, dynamic_len: int | None = None) -> torch.Tensor:
        sigmas = self._linspace_sigmas(num_steps, denoising_strength)
        if self.inverse_timesteps:
            sigmas = torch.flip(sigmas, dims=[0])
        sigmas = self._apply_shift(sigmas, dynamic_len)
        sigmas = self._apply_terminal_rescale(sigmas)
        if self.reverse_sigmas:
            sigmas = 1 - sigmas
        return sigmas

    # ---- helpers -----------------------------------------------------

    @staticmethod
    def _compute_shift_mu(
        seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ) -> float:
        slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        intercept = base_shift - slope * base_seq_len
        return seq_len * slope + intercept


class FlowMatchScheduler:
    """Euler flow-matching scheduler built on top of a :class:`SigmaSchedule`.

    Public API (used by the pipeline):
        * ``set_timesteps(num, training=, shift=, dynamic_shift_len=)``
        * ``step(model_output, timestep, sample, to_final=)``
        * ``add_noise(original, noise, timestep, t_dim=)``
        * ``return_to_timestep(timestep, sample, sample_stabilized)``
        * ``training_target(sample, noise, timestep)``
        * ``training_weight(timestep)``
        * ``calculate_shift(image_seq_len, ...)``
    """

    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
        exponential_shift: bool = False,
        exponential_shift_mu: float | None = None,
        shift_terminal: float | None = None,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self._schedule = SigmaSchedule(
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            shift=shift,
            inverse_timesteps=inverse_timesteps,
            extra_one_step=extra_one_step,
            reverse_sigmas=reverse_sigmas,
            exponential_shift=exponential_shift,
            exponential_shift_mu=exponential_shift_mu,
            shift_terminal=shift_terminal,
        )
        self._sigmas: torch.Tensor = torch.empty(0)
        self._timesteps: torch.Tensor = torch.empty(0)
        self._train_weights: torch.Tensor | None = None
        self.is_training: bool = False
        self.set_timesteps(num_inference_steps)

    # ---- properties (read-only views for the pipeline) ---------------

    @property
    def sigmas(self) -> torch.Tensor:
        return self._sigmas

    @property
    def timesteps(self) -> torch.Tensor:
        return self._timesteps

    @property
    def shift(self) -> float:
        return self._schedule.shift

    @shift.setter
    def shift(self, value: float) -> None:
        self._schedule.shift = value

    @property
    def sigma_max(self) -> float:
        return self._schedule.sigma_max

    @property
    def sigma_min(self) -> float:
        return self._schedule.sigma_min

    @property
    def inverse_timesteps(self) -> bool:
        return self._schedule.inverse_timesteps

    @property
    def extra_one_step(self) -> bool:
        return self._schedule.extra_one_step

    @property
    def reverse_sigmas(self) -> bool:
        return self._schedule.reverse_sigmas

    @property
    def exponential_shift(self) -> bool:
        return self._schedule.exponential_shift

    @property
    def exponential_shift_mu(self) -> float | None:
        return self._schedule.exponential_shift_mu

    @property
    def shift_terminal(self) -> float | None:
        return self._schedule.shift_terminal

    # ---- backward-compat aliases -------------------------------------

    @property
    def linear_timesteps_weights(self) -> torch.Tensor | None:
        return self._train_weights

    # ---- timestep setup ----------------------------------------------

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        training: bool = False,
        shift: float | None = None,
        dynamic_shift_len: int | None = None,
    ) -> None:
        if shift is not None:
            self._schedule.shift = shift

        self._sigmas = self._schedule.build(num_inference_steps, denoising_strength, dynamic_shift_len)
        self._timesteps = self._sigmas * self.num_train_timesteps
        self.is_training = training

        if training:
            self._train_weights = self._bsmntw_weights(num_inference_steps)
        else:
            self._train_weights = None

    @staticmethod
    def _bsmntw_weights(num_steps: int) -> torch.Tensor:
        """Bell-shaped minimum-noise-timestep weighting (BSMNTW)."""
        x = torch.arange(num_steps, dtype=torch.float32)
        y = torch.exp(-2 * ((x - num_steps / 2) / num_steps) ** 2)
        y = y - y.min()
        return y * (num_steps / y.sum())

    # ---- inference helpers -------------------------------------------

    def _resolve_step_index(self, timestep: torch.Tensor | float | int) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        return torch.argmin((self._timesteps - timestep).abs())

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        to_final: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        idx = self._resolve_step_index(timestep)
        sigma = self._sigmas[idx]
        if to_final or idx + 1 >= len(self._timesteps):
            next_sigma = torch.tensor(1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0)
        else:
            next_sigma = self._sigmas[idx + 1]
        return sample + model_output * (next_sigma - sigma)

    def return_to_timestep(
        self,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        sample_stabilized: torch.Tensor,
    ) -> torch.Tensor:
        idx = self._resolve_step_index(timestep)
        sigma = self._sigmas[idx]
        return (sample - sample_stabilized) / sigma

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor | float,
        t_dim: int = 2,
    ) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep = timestep[None]
        idx = torch.argmin((self._timesteps[:, None] - timestep).abs(), dim=0)
        shape = [1] * noise.ndim
        shape[t_dim] = idx.shape[0]
        sigma = self._sigmas[idx].to(original_samples).view(shape)
        return (1 - sigma) * original_samples + sigma * noise

    # ---- training helpers --------------------------------------------

    def training_target(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return noise - sample

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        idx = torch.argmin(
            (self._timesteps[:, None].to(timestep.device) - timestep[None]).abs(),
            dim=0,
        )
        if self._train_weights is None:
            return torch.ones_like(timestep, dtype=torch.float32)
        return self._train_weights.to(timestep.device)[idx].to(timestep.device)

    def calculate_shift(
        self,
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ) -> float:
        return SigmaSchedule._compute_shift_mu(
            image_seq_len, base_seq_len, max_seq_len, base_shift, max_shift
        )
