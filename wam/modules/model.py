# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""WanTransformer3DModel: dual-stream MoT video-action diffusion transformer.

This open-source inference build supports only ``use_mot=True`` checkpoints:
each block holds per-modality (``lat``, ``act``) copies of attn1/attn2/ffn/
norm1-3/scale_shift_table inside ``nn.ModuleDict``s. At inference time one
modality's Q attends to the concatenated valid KV cache from both modalities;
cross-attention to text is fully independent per modality. The act stream may
use a smaller hidden dim than the lat stream (``mot_action_hidden_dim`` config,
e.g. 768 vs. 3072) because each branch's projections map into the shared head
space ``num_heads * head_dim``.

The ``enable_m3=false`` fields left by older checkpoint configs are accepted
for compatibility, but M3 and non-MoT checkpoints are rejected.
"""
import math
import re
from copy import deepcopy

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange

from .attention_ops import (
    SUPPORTED_ATTN_MODES,
    build_attn_op,
)

__all__ = ['WanTransformer3DModel']


_MODALITIES = ('lat', 'act')


def _apply_rotary_emb(x, freqs):
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(3)
    return x_out.to(x.dtype)


class WanTimeTextImageEmbedding(nn.Module):

    def __init__(
        self,
        dim,
        time_freq_dim,
        time_proj_dim,
        text_embed_dim,
        pos_embed_seq_len,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim,
                                        flip_sin_to_cos=True,
                                        downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim,
                                               time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim,
                                                       dim,
                                                       act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        dtype=None,
    ):
        B, L = timestep.shape
        timestep = timestep.reshape(-1)
        timestep = self.timesteps_proj(timestep)
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(B, L, -1), timestep_proj.reshape(B, L, -1)


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.f_freqs_base = f_freqs_base
        self.h_freqs_base = h_freqs_base
        self.w_freqs_base = w_freqs_base

    def _precompute_freqs_base(self):
        f_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.f_dim, 2)[:(self.f_dim // 2)].double() / self.f_dim))
        h_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.h_dim, 2)[:(self.h_dim // 2)].double() / self.h_dim))
        w_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.w_dim, 2)[:(self.w_dim // 2)].double() / self.w_dim))
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        with torch.no_grad():
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(grid_ids.device)
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(grid_ids.device)
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(grid_ids.device)
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


class WanAttention(nn.Module):
    """One branch of self-attention. Holds the projection parameters and (for
    self-attention) a per-branch KV cache pool. The attention op itself is
    delegated to ``self.attn_op``, which is built via ``build_attn_op``
    from ``attention_ops`` and may be one of:

    * ``custom_sdpa`` (torch)
    * ``flash_attn_func`` (flashattn)
    * ``FlexAttnFunc`` (flex)
    * ``DecomposedVarlenAttnFunc`` (decomposed_varlen)

    The last two accept an additional ``attn_ctx_kwargs`` dict, dispatched
    here via :meth:`_run_attn`.

    MoT instantiates one ``WanAttention`` per modality (inside
    ``nn.ModuleDict({'lat': ..., 'act': ...})``). Both modalities share the
    same head space (``num_heads * head_dim``), so the per-branch ``dim``
    (hidden size of the residual stream) is only seen by the in/out
    Linears — and may differ across modalities.
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        eps=1e-5,
        dropout=0.0,
        cross_attention_dim_head=None,
        attn_mode='torch',
    ):
        super().__init__()
        if attn_mode not in SUPPORTED_ATTN_MODES:
            raise ValueError(
                f"Unsupported attention mode: {attn_mode}, supported: {SUPPORTED_ATTN_MODES}"
            )
        is_cross = cross_attention_dim_head is not None
        self.attn_mode = attn_mode
        self.is_cross = is_cross
        self.attn_op = build_attn_op(attn_mode, is_cross=is_cross)

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, dim, bias=True),
            nn.Dropout(dropout),
        ])
        self.norm_q = nn.RMSNorm(dim_head * heads,
                                 eps=eps,
                                 elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim_head * heads,
                                 eps=eps,
                                 elementwise_affine=True)
        # Cross-attention is stateless w.r.t. cache.
        self.attn_caches = {} if not is_cross else None

    # ---------- attention dispatch ----------

    def _run_attn(self, q, k, v, attn_ctx_kwargs=None):
        # 'flex' / 'decomposed_varlen' need batch shape info via
        # attn_ctx_kwargs to fetch / build masks lazily through
        # AttentionContext; 'torch' / 'flashattn' are plain callables.
        if self.attn_mode in ('flex', 'decomposed_varlen'):
            return self.attn_op(q, k, v, attn_ctx_kwargs=attn_ctx_kwargs)
        return self.attn_op(q, k, v)

    # ---------- cache management (self-attention only) ----------

    def clear_pred_cache(self, cache_name):
        if self.attn_caches is None:
            return
        cache = self.attn_caches.get(cache_name)
        if cache is None:
            return
        is_pred = cache['is_pred']
        cache['mask'][is_pred] = False

    def clear_cache(self, cache_name):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = None

    def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim,
                      device, dtype, batch_size):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = {
            'k':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'v':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'id':
            torch.full((total_tolen, ), -1, device=device),
            'mask':
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
            'is_pred':
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
        }

    def allocate_slots(self, cache_name, key_size):
        cache = self.attn_caches[cache_name]
        mask = cache['mask']
        ids = cache['id']
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)
            used_ids = ids[used]
            order = torch.argsort(used_ids)
            need = key_size - free.numel()
            to_free = used[order[:need]]
            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        assert free.numel() >= key_size
        return free[:key_size]

    def _next_cache_id(self, cache_name):
        ids = self.attn_caches[cache_name]['id']
        mask = self.attn_caches[cache_name]['mask']

        if mask.any():
            return ids[mask].max() + 1
        else:
            return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def write_self_cache(self, cache_name, key, value, is_pred):
        cache = self.attn_caches[cache_name]
        slots = self.allocate_slots(cache_name, key.shape[1])
        new_id = self._next_cache_id(cache_name)
        cache['k'][:, slots] = key
        cache['v'][:, slots] = value
        cache['mask'][slots] = True
        cache['id'][slots] = new_id
        cache['is_pred'][slots] = is_pred
        return slots

    def read_self_cache(self, cache_name):
        cache = self.attn_caches[cache_name]
        valid = cache['mask'].nonzero(as_tuple=False).squeeze(-1)
        return cache['k'][:, valid], cache['v'][:, valid]

    def restore_cache(self, cache_name, slots):
        self.attn_caches[cache_name]['mask'][slots] = False

    def has_active_cache(self, cache_name):
        if self.attn_caches is None:
            return False
        cache = self.attn_caches.get(cache_name)
        return cache is not None and cache.get('k') is not None

    # ---------- projection helpers ----------

    def project_qkv_self(self, hidden_states, rotary_emb=None):
        """Self-attention QKV projection (with optional rope) for one
        branch's ``hidden_states``. Returns ``(q, k, v)`` in
        ``[B, S, num_heads, head_dim]`` layout."""
        query = self.norm_q(self.to_q(hidden_states)).unflatten(2, (self.heads, -1))
        key = self.norm_k(self.to_k(hidden_states)).unflatten(2, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(2, (self.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, rotary_emb)
            key = _apply_rotary_emb(key, rotary_emb)
        return query, key, value

    def project_out(self, attn_out):
        h = attn_out.flatten(2, 3)
        h = self.to_out[0](h)
        h = self.to_out[1](h)
        return h


class CrossAttention(nn.Module):
    """Plain text cross-attention for one modality. Stateless — no KV
    cache. Used inside ``nn.ModuleDict({'lat': ..., 'act': ...})`` so that
    each modality runs its own attention call against its own K/V projection
    of the text encoder hidden states (which themselves come from a
    per-modality ``condition_embedder*.text_embedder``)."""

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        eps=1e-5,
        dropout=0.0,
        attn_mode='torch',
    ):
        super().__init__()
        if attn_mode not in SUPPORTED_ATTN_MODES:
            raise ValueError(
                f"Unsupported attention mode: {attn_mode}, supported: {SUPPORTED_ATTN_MODES}"
            )
        self.attn_mode = attn_mode
        self.attn_op = build_attn_op(attn_mode, is_cross=True)
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head

        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=True)
        self.norm_q = nn.RMSNorm(self.inner_dim, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(self.inner_dim, eps=eps, elementwise_affine=True)
        self.to_out = nn.ModuleList([
            nn.Linear(self.inner_dim, dim, bias=True),
            nn.Dropout(dropout),
        ])

    def _run_attn(self, q, k, v, attn_ctx_kwargs=None):
        if self.attn_mode in ('flex', 'decomposed_varlen'):
            return self.attn_op(q, k, v, attn_ctx_kwargs=attn_ctx_kwargs)
        return self.attn_op(q, k, v)

    def forward(self, hidden_states, encoder_hidden_states,
                attn_ctx_kwargs=None):
        q = self.norm_q(self.to_q(hidden_states)).unflatten(2, (self.heads, -1))
        k = self.norm_k(self.to_k(encoder_hidden_states)).unflatten(2, (self.heads, -1))
        v = self.to_v(encoder_hidden_states).unflatten(2, (self.heads, -1))
        out = self._run_attn(q, k, v, attn_ctx_kwargs=attn_ctx_kwargs).type_as(q)
        out = out.flatten(2, 3)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out


class WanTransformerBlock(nn.Module):
    """Legacy single-tower block kept only so old key-mirroring code can
    remain shape-aware. Public loading rejects ``use_mot=False``."""

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode='flashattn',
    ):
        super().__init__()
        self.attn_mode = attn_mode
        head_dim = dim // num_heads

        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=head_dim,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )
        self.attn2 = CrossAttention(
            dim=dim,
            heads=num_heads,
            dim_head=head_dim,
            eps=eps,
            attn_mode=attn_mode,
        )

        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.norm2 = (FP32LayerNorm(dim, eps, elementwise_affine=True)
                      if cross_attn_norm else nn.Identity())
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.ffn = FeedForward(dim, inner_dim=ffn_dim,
                               activation_fn="gelu-approximate")
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, dim) / dim**0.5)

    def _ada_ln_chunks(self, temb):
        table = self.scale_shift_table[None] + temb.float()
        chunks = rearrange(table, 'b l n c -> b n l c').chunk(6, dim=1)
        return tuple(c.squeeze(1) for c in chunks)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
        attn_ctx_kwargs=None,
    ):
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
            self._ada_ln_chunks(temb)

        # 1. Self-attention
        norm_h = (self.norm1(hidden_states.float()) *
                  (1. + scale_msa) + shift_msa).type_as(hidden_states)
        query, key, value = self.attn1.project_qkv_self(norm_h, rotary_emb)

        slots = None
        if self.attn1.has_active_cache(cache_name):
            slots = self.attn1.write_self_cache(
                cache_name, key, value, is_pred=(update_cache == 1))
            key, value = self.attn1.read_self_cache(cache_name)

        attn_out = self.attn1._run_attn(query, key, value, attn_ctx_kwargs)

        if update_cache == 0 and slots is not None:
            self.attn1.restore_cache(cache_name, slots)

        attn_out = self.attn1.project_out(attn_out.type_as(query))
        hidden_states = (hidden_states.float() +
                         attn_out * gate_msa).type_as(hidden_states)

        # 2. Cross-attention to text
        norm_h = self.norm2(hidden_states.float()).type_as(hidden_states)
        cross_out = self.attn2(norm_h, encoder_hidden_states,
                               attn_ctx_kwargs=attn_ctx_kwargs)
        hidden_states = hidden_states + cross_out

        # 3. FFN
        norm_h = (self.norm3(hidden_states.float()) *
                  (1. + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_out = self.ffn(norm_h)
        hidden_states = (hidden_states.float() +
                         ff_out.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


class MoTWanTransformerBlock(nn.Module):
    """Symmetric MoT block: latent and action tokens have independent
    self/cross attention parameters, FFN, norms and AdaLN tables, all kept
    in per-modality ``nn.ModuleDict({'lat': ..., 'act': ...})``s.

    Self-attention is computed jointly across both modalities:
      * Training: a single op over ``cat([Q_lat, Q_act])``; the mask /
        plan (FlexAttention BlockMask or varlen plan) is provided by
        ``AttentionContext`` and reflects the cross-modality visibility
        rules already encoded for this repo.
      * Inference: one modality's Q is attended against its own freshly
        written KV plus the *other* modality's cached KV (read-only).

    Cross-attention is fully independent per modality: each modality has
    its own ``CrossAttention`` module that projects its own K/V from a
    per-modality text encoding and runs its own attention call.

    Per-branch ``dim`` may differ (``dim_lat`` vs ``dim_act``). The joint
    self-attention still works because each branch's
    ``to_q/to_k/to_v`` projects from its own dim to the shared head space
    ``num_heads * head_dim`` (same across both branches)."""

    def __init__(
        self,
        dim_lat,
        dim_act,
        ffn_dim_lat,
        ffn_dim_act,
        num_heads,
        head_dim,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode='flashattn',
    ):
        super().__init__()
        self.attn_mode = attn_mode
        self.dims = {'lat': dim_lat, 'act': dim_act}
        self.ffn_dims = {'lat': ffn_dim_lat, 'act': ffn_dim_act}

        # 1. Self-attention (per modality, with KV cache). Both modalities
        # use the same num_heads * head_dim head space → joint cat works.
        self.attn1 = nn.ModuleDict({
            m: WanAttention(
                dim=self.dims[m],
                heads=num_heads,
                dim_head=head_dim,
                eps=eps,
                cross_attention_dim_head=None,
                attn_mode=attn_mode,
            ) for m in _MODALITIES
        })

        # 2. Cross-attention to text (per modality, independent). The act
        # branch shrinks the head count so that inner_dim == dim_act
        # (avoids wasting an expand→shrink round-trip in the smaller
        # branch).
        # Cross-attention is always dense (B=1 → trivial separation; no
        # noise/clean/window structure), so it uses plain SDPA regardless
        # of the self-attention mode. This avoids the joint-context
        # cross_mask shape mismatch when a per-modality query is used.
        cross_heads = {
            'lat': num_heads,
            'act': max(1, dim_act // head_dim),
        }
        cross_mode = 'torch' if attn_mode in ('flex', 'decomposed_varlen') else attn_mode
        self.attn2 = nn.ModuleDict({
            m: CrossAttention(
                dim=self.dims[m],
                heads=cross_heads[m],
                dim_head=head_dim,
                eps=eps,
                attn_mode=cross_mode,
            ) for m in _MODALITIES
        })
        # Mark these cross modules so _maybe_swap_attn_op won't promote
        # them back to a joint-context mask mode (flex/decomposed_varlen).
        for m in _MODALITIES:
            self.attn2[m]._mot_force_dense = True

        # 3. Norms (per modality)
        self.norm1 = nn.ModuleDict({
            m: FP32LayerNorm(self.dims[m], eps, elementwise_affine=False)
            for m in _MODALITIES
        })
        self.norm2 = nn.ModuleDict({
            m: (FP32LayerNorm(self.dims[m], eps, elementwise_affine=True)
                if cross_attn_norm else nn.Identity())
            for m in _MODALITIES
        })
        self.norm3 = nn.ModuleDict({
            m: FP32LayerNorm(self.dims[m], eps, elementwise_affine=False)
            for m in _MODALITIES
        })

        # 4. FFN (per modality)
        self.ffn = nn.ModuleDict({
            m: FeedForward(self.dims[m], inner_dim=self.ffn_dims[m],
                           activation_fn="gelu-approximate")
            for m in _MODALITIES
        })

        # 5. AdaLN scale-shift table (per modality)
        self.scale_shift_table = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, 6, self.dims[m]) /
                            self.dims[m]**0.5)
            for m in _MODALITIES
        })

    def _ada_ln_chunks(self, modality, temb):
        table = self.scale_shift_table[modality][None] + temb.float()
        chunks = rearrange(table, 'b l n c -> b n l c').chunk(6, dim=1)
        return tuple(c.squeeze(1) for c in chunks)

    # ---------- inference: single modality at a time ----------

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        modality='lat',
        update_cache=0,
        cache_name='pos',
        attn_ctx_kwargs=None,
    ):
        """Inference path: process a single modality. Self-attention
        combines the freshly projected K/V with the *other* modality's
        cached KV (read-only, taken from whatever was last written —
        typically the observation/GT clean tokens, or the previous
        denoising step's predictions)."""
        other = 'act' if modality == 'lat' else 'lat'
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
            self._ada_ln_chunks(modality, temb)

        attn_self = self.attn1[modality]
        attn_other = self.attn1[other]

        # 1. Self-attention
        norm_h = (self.norm1[modality](hidden_states.float()) *
                  (1. + scale_msa) + shift_msa).type_as(hidden_states)
        query, key, value = attn_self.project_qkv_self(norm_h, rotary_emb)

        slots = None
        if attn_self.has_active_cache(cache_name):
            slots = attn_self.write_self_cache(
                cache_name, key, value, is_pred=(update_cache == 1))
            key, value = attn_self.read_self_cache(cache_name)

        # Pull in the other modality's KV cache, if any.
        if attn_other.has_active_cache(cache_name):
            k_other, v_other = attn_other.read_self_cache(cache_name)
            if k_other.shape[1] > 0:
                key = torch.cat([key, k_other], dim=1)
                value = torch.cat([value, v_other], dim=1)

        attn_out = attn_self._run_attn(query, key, value, attn_ctx_kwargs)

        if update_cache == 0 and slots is not None:
            attn_self.restore_cache(cache_name, slots)

        attn_out = attn_self.project_out(attn_out.type_as(query))
        hidden_states = (hidden_states.float() +
                         attn_out * gate_msa).type_as(hidden_states)

        # 2. Cross-attention to text (this modality's text encoding only)
        norm_h = self.norm2[modality](hidden_states.float()).type_as(hidden_states)
        cross_out = self.attn2[modality](norm_h, encoder_hidden_states,
                                         attn_ctx_kwargs=attn_ctx_kwargs)
        hidden_states = hidden_states + cross_out

        # 3. FFN (per modality)
        norm_h = (self.norm3[modality](hidden_states.float()) *
                  (1. + c_scale_msa) + c_shift_msa).type_as(hidden_states)
        ff_out = self.ffn[modality](norm_h)
        hidden_states = (hidden_states.float() +
                         ff_out.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states

    # ---------- training: both modalities, joint self-attention ----------

    def forward_dual(
        self,
        hidden_states_lat,
        hidden_states_act,
        encoder_hidden_states_lat,
        encoder_hidden_states_act,
        temb_lat,
        temb_act,
        rotary_emb_lat,
        rotary_emb_act,
        attn_ctx_kwargs=None,
    ):
        """Training path: both modalities present, no KV cache.
        Self-attention runs once over ``cat([Q_lat, Q_act])``; the joint
        mask (FlexAttention BlockMask / decomposed-varlen plan from
        ``AttentionContext``) takes care of cross-modality visibility.
        Cross-attention runs once per modality against its own text
        encoding (independent text K/V projections)."""
        s_lat = self._ada_ln_chunks('lat', temb_lat)
        s_act = self._ada_ln_chunks('act', temb_act)
        shift_l, scale_l, gate_l, c_shift_l, c_scale_l, c_gate_l = s_lat
        shift_a, scale_a, gate_a, c_shift_a, c_scale_a, c_gate_a = s_act

        attn_lat = self.attn1['lat']
        attn_act = self.attn1['act']

        # 1. Joint self-attention
        n_lat = (self.norm1['lat'](hidden_states_lat.float()) *
                 (1. + scale_l) + shift_l).type_as(hidden_states_lat)
        n_act = (self.norm1['act'](hidden_states_act.float()) *
                 (1. + scale_a) + shift_a).type_as(hidden_states_act)

        q_l, k_l, v_l = attn_lat.project_qkv_self(n_lat, rotary_emb_lat)
        q_a, k_a, v_a = attn_act.project_qkv_self(n_act, rotary_emb_act)

        Q = torch.cat([q_l, q_a], dim=1)
        K = torch.cat([k_l, k_a], dim=1)
        V = torch.cat([v_l, v_a], dim=1)

        # Both branches share the same head space and (for flex) the same
        # class-level BlockMask, so either branch's attn_op is fine —
        # use lat's.
        joint = attn_lat._run_attn(Q, K, V, attn_ctx_kwargs).type_as(Q)
        L_lat = q_l.shape[1]
        out_lat = attn_lat.project_out(joint[:, :L_lat])
        out_act = attn_act.project_out(joint[:, L_lat:])
        hidden_states_lat = (hidden_states_lat.float() +
                             out_lat * gate_l).type_as(hidden_states_lat)
        hidden_states_act = (hidden_states_act.float() +
                             out_act * gate_a).type_as(hidden_states_act)

        # 2. Cross-attention to text (per modality, independent)
        n_lat_c = self.norm2['lat'](hidden_states_lat.float()).type_as(hidden_states_lat)
        n_act_c = self.norm2['act'](hidden_states_act.float()).type_as(hidden_states_act)
        c_lat = self.attn2['lat'](n_lat_c, encoder_hidden_states_lat,
                                  attn_ctx_kwargs=attn_ctx_kwargs)
        c_act = self.attn2['act'](n_act_c, encoder_hidden_states_act,
                                  attn_ctx_kwargs=attn_ctx_kwargs)
        hidden_states_lat = hidden_states_lat + c_lat
        hidden_states_act = hidden_states_act + c_act

        # 3. FFN (per modality)
        n_lat_f = (self.norm3['lat'](hidden_states_lat.float()) *
                   (1. + c_scale_l) + c_shift_l).type_as(hidden_states_lat)
        n_act_f = (self.norm3['act'](hidden_states_act.float()) *
                   (1. + c_scale_a) + c_shift_a).type_as(hidden_states_act)
        ff_lat = self.ffn['lat'](n_lat_f)
        ff_act = self.ffn['act'](n_act_f)
        hidden_states_lat = (hidden_states_lat.float() +
                             ff_lat.float() * c_gate_l).type_as(hidden_states_lat)
        hidden_states_act = (hidden_states_act.float() +
                             ff_act.float() * c_gate_a).type_as(hidden_states_act)
        return hidden_states_lat, hidden_states_act


# ---------------------------------------------------------------------------
# Legacy (single-tower → MoT) state-dict mirroring
# ---------------------------------------------------------------------------

# Pre-MoT block keys that need to be cloned into ``.lat`` / ``.act`` twins.
# Cross-attention (attn2) is fully per-modality now, so it's included.
_LEGACY_BLOCK_KEY_PAT = re.compile(
    r'^(blocks\.\d+\.)'
    r'(attn1|attn2|norm1|norm2|norm3|ffn|scale_shift_table)'
    r'(\.[^.]+)?'
    r'(\..*)?$'
)


def _mirror_legacy_state_dict_inplace(state_dict, prefix=''):
    """Rewrite a single-tower (pre-MoT) state dict to MoT layout in place.

    Pre-MoT keys (single tower, weights shared across modalities) become
    two sibling keys — one under ``.lat.`` and one under ``.act.`` —
    copying the same tensor. Already-MoT keys pass through untouched, so
    this function is idempotent.

    Mappings (per block ``i``):

    * ``blocks.{i}.attn1.<rest>``    → ``blocks.{i}.attn1.{lat,act}.<rest>``
    * ``blocks.{i}.attn2.<rest>``    → ``blocks.{i}.attn2.{lat,act}.<rest>``
    * ``blocks.{i}.norm{1,2,3}.<rest>`` → ``blocks.{i}.norm{1,2,3}.{lat,act}.<rest>``
    * ``blocks.{i}.ffn.<rest>``      → ``blocks.{i}.ffn.{lat,act}.<rest>``
    * ``blocks.{i}.scale_shift_table`` (Parameter) → ``...scale_shift_table.{lat,act}`` (ParameterDict)
    * top-level ``scale_shift_table`` → ``scale_shift_table_lat`` / ``_act``

    All other keys (top-level Linears, condition embedders, action heads,
    proj_out, norm_out, rope, …) pass through unchanged. If the act
    branch's per-modality hidden dim differs from lat's, the cloned act
    tensors will fail the model's shape check and end up in
    ``unexpected_keys`` — they're then filled in by
    ``init_act_stream_from_lat`` in ``utils.py``.
    """
    p = prefix

    top_key = p + 'scale_shift_table'
    if top_key in state_dict:
        v = state_dict.pop(top_key)
        state_dict[p + 'scale_shift_table_lat'] = v.clone()
        state_dict[p + 'scale_shift_table_act'] = v.clone()

    for k in list(state_dict.keys()):
        if not k.startswith(p):
            continue
        suf = k[len(p):]
        m = _LEGACY_BLOCK_KEY_PAT.match(suf)
        if not m:
            continue
        blk = m.group(1)
        sub = m.group(2)
        first = m.group(3) or ''
        rest = m.group(4) or ''

        if first in ('.lat', '.act'):
            continue  # already MoT

        if sub == 'scale_shift_table':
            v = state_dict.pop(k)
            for mod in _MODALITIES:
                state_dict[p + f'{blk}scale_shift_table.{mod}'] = v.clone()
            continue

        v = state_dict.pop(k)
        for mod in _MODALITIES:
            state_dict[p + f'{blk}{sub}.{mod}{first}{rest}'] = v.clone()

    # Force act-side text_embedder to be a clone of the lat-side
    # text_embedder. Reason: the pre-MoT inference path used
    # ``condition_embedder.text_embedder`` for the text branch even when
    # ``action_mode=True`` (only the time branch went through
    # ``condition_embedder_action``). The act-side text_embedder's saved
    # weights — though trained on the act stream — were never observed
    # at inference, so they reflect a dead training-only path. The
    # symmetric MoT model uses the act side consistently for both text
    # and time in action_mode; cloning lat-side weights here makes MoT
    # action_mode inference match legacy non-MoT action_mode inference
    # bit-for-bit (modulo bf16 noise) on the same checkpoint.
    #
    # Only fire when src/dst already share a shape (the
    # ``act_dim == video_dim`` case). When dims differ, the dst-side
    # would end up with a wrong-shape tensor (would crash any strict
    # load) and the manual load path drops it via shape filter anyway —
    # ``init_act_stream_from_lat`` then interpolates lat → act for
    # ``condition_embedder_action.text_embedder`` (see
    # ``_copy_condition_embedder``).
    for suf in (
        'text_embedder.linear_1.weight', 'text_embedder.linear_1.bias',
        'text_embedder.linear_2.weight', 'text_embedder.linear_2.bias',
    ):
        src_key = p + 'condition_embedder.' + suf
        dst_key = p + 'condition_embedder_action.' + suf
        if src_key not in state_dict:
            continue
        src = state_dict[src_key]
        dst = state_dict.get(dst_key)
        # Two contexts call this:
        # (a) Standalone mirror on the legacy raw state dict — dst exists
        #     (deepcopy in pre-MoT __init__) with the same shape as src;
        #     clone fires to overwrite act-side with lat-side weights.
        # (b) Re-fired via the model's load_state_dict pre-hook on a
        #     filtered dict (act_dim != video_dim path) — dst was just
        #     dropped by the outer shape filter. Re-adding src here would
        #     resurrect the wrong-shape tensor and crash the strict-shape
        #     check inside load_state_dict. Leave dst absent;
        #     init_act_stream_from_lat will interpolate lat → act.
        if dst is not None and tuple(dst.shape) == tuple(src.shape):
            state_dict[dst_key] = src.clone()


def _make_legacy_state_dict_pre_hook(use_mot: bool):
    """Build a state-dict pre-hook. When ``use_mot=True`` it mirrors
    legacy single-tower keys to the MoT layout in place."""
    def hook(state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs):
        if use_mot:
            _mirror_legacy_state_dict_inplace(state_dict, prefix=prefix)
    return hook


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class WanTransformer3DModel(ModelMixin, ConfigMixin):
    r"""Video-action diffusion transformer. See module docstring for the
    two architectures (single-tower vs. symmetric MoT) selected via
    ``use_mot``."""
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "patch_embedding_mlp",
        "condition_embedder",
        "condition_embedder_action",
        "norm",
    ]
    _no_split_modules = ["WanTransformerBlock", "MoTWanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "scale_shift_table_lat",
        "scale_shift_table_act",
        "norm1",
        "norm2",
        "norm3",
        "action_norm_out",
        "norm_out",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock", "MoTWanTransformerBlock"]

    @register_to_config
    def __init__(self,
                 patch_size=[1, 2, 2],
                 num_attention_heads=24,
                 attention_head_dim=128,
                 in_channels=48,
                 out_channels=48,
                 action_dim=30,
                 text_dim=4096,
                 freq_dim=256,
                 ffn_dim=14336,
                 num_layers=30,
                 cross_attn_norm=True,
                 eps=1e-06,
                 rope_max_seq_len=1024,
                 pos_embed_seq_len=None,
                 attn_mode="torch",
                 use_mot=True,
                 mot_action_hidden_dim=768,
                 mot_action_ffn_multiplier=4,
                 enable_m3=False,
                 m3_hidden_dim=768,
                 m3_ffn_multiplier=4,
                 m3_input_dim=16,
                 m3_output_dim=16,
                 m3_action_indices=None,
                 m3_rope_h_offset=1):
        r"""
        Args:
            use_mot: If True (default), build a symmetric Mixture-of-
                Transformers model with per-modality (lat/act) blocks.
                If False, build the original single-tower model — used
                for ablation and reproducing pre-MoT experiments.
            mot_action_hidden_dim: Hidden dim of the act stream when
                ``use_mot=True``. Set equal to ``num_attention_heads *
                attention_head_dim`` to recover symmetric same-dim
                behavior (act is then a perfect twin of lat).
            mot_action_ffn_multiplier: FFN expansion for the act stream's
                FeedForward, only used when ``use_mot=True``. The act
                FFN inner dim becomes
                ``mot_action_hidden_dim * mot_action_ffn_multiplier``.
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.use_mot = use_mot
        if not use_mot:
            raise ValueError(
                "This inference-only build supports only dual-stream MoT "
                "(use_mot=True) checkpoints."
            )
        if enable_m3:
            raise ValueError(
                "This inference-only build supports only dual-stream MoT "
                "(lat+act). Checkpoints with enable_m3=True are not supported."
            )
        if m3_action_indices not in (None, [], ()):
            raise ValueError(
                "This inference-only build does not support m3_action_indices."
            )

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        if use_mot:
            self.mot_action_hidden_dim = mot_action_hidden_dim
            self.mot_action_ffn_multiplier = mot_action_ffn_multiplier
            act_dim = mot_action_hidden_dim
            # When act_dim equals video_dim, force act_ffn_dim = ffn_dim so
            # the act tower is a true twin of lat (multiplier may not divide
            # ffn_dim exactly, e.g. Wan2.2 has ffn_dim=14336, dim=3072 →
            # ratio≈4.67). Otherwise use the configured multiplier.
            if act_dim == inner_dim:
                act_ffn_dim = ffn_dim
            else:
                act_ffn_dim = mot_action_hidden_dim * mot_action_ffn_multiplier
            self.act_dim = act_dim
            self.act_ffn_dim = act_ffn_dim

        # ---------- shared embedders / rope ----------
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size,
                                      rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2],
            inner_dim)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # ---------- action-side embedders / output heads ----------
        if use_mot:
            # Act stream lives in mot_action_hidden_dim throughout.
            self.action_embedder = nn.Linear(action_dim, act_dim)
            self.condition_embedder_action = WanTimeTextImageEmbedding(
                dim=act_dim,
                time_freq_dim=freq_dim,
                time_proj_dim=act_dim * 6,
                text_embed_dim=text_dim,
                pos_embed_seq_len=pos_embed_seq_len,
            )
            self.action_norm_out = FP32LayerNorm(act_dim, eps,
                                                 elementwise_affine=False)
            self.action_proj_out = nn.Linear(act_dim, action_dim)
            self.scale_shift_table_lat = nn.Parameter(
                torch.randn(1, 2, inner_dim) / inner_dim**0.5)
            self.scale_shift_table_act = nn.Parameter(
                torch.randn(1, 2, act_dim) / act_dim**0.5)
        else:
            # Pre-MoT (single-tower): act stream shares video dim.
            self.action_embedder = nn.Linear(action_dim, inner_dim)
            self.condition_embedder_action = deepcopy(self.condition_embedder)
            self.action_proj_out = nn.Linear(inner_dim, action_dim)
            self.scale_shift_table = nn.Parameter(
                torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        # ---------- transformer blocks ----------
        if use_mot:
            self.blocks = nn.ModuleList([
                MoTWanTransformerBlock(
                    dim_lat=inner_dim,
                    dim_act=act_dim,
                    ffn_dim_lat=ffn_dim,
                    ffn_dim_act=act_ffn_dim,
                    num_heads=num_attention_heads,
                    head_dim=attention_head_dim,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                    attn_mode=attn_mode,
                ) for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads,
                    cross_attn_norm, eps, attn_mode=attn_mode,
                ) for _ in range(num_layers)
            ])

        # ---------- shared output projection (lat / single-tower) ----------
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim,
                                  out_channels * math.prod(patch_size))

        # Auto-mirror single-tower (pre-MoT) state dicts to MoT layout when
        # loading. The pre-hook only fires when this model itself is MoT —
        # a non-MoT model leaves keys untouched. Idempotent for already-MoT
        # checkpoints.
        self._register_load_state_dict_pre_hook(
            _make_legacy_state_dict_pre_hook(use_mot))

    # ---------- cache plumbing ----------

    def _attn1_branches(self, block):
        """Iterate over the self-attention branches of ``block``,
        regardless of whether ``block`` is single-tower or MoT."""
        if self.use_mot:
            return [block.attn1[m] for m in _MODALITIES]
        return [block.attn1]

    def clear_cache(self, cache_name):
        for block in self.blocks:
            for attn in self._attn1_branches(block):
                attn.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name):
        for block in self.blocks:
            for attn in self._attn1_branches(block):
                attn.clear_pred_cache(cache_name)

    def create_empty_cache(self, cache_name, attn_window,
                           latent_token_per_chunk, action_token_per_chunk,
                           device, dtype, batch_size):
        """Allocate per-attn1 KV cache pools. In MoT mode each modality
        gets its own pool sized for ``(attn_window // 2) *
        tokens_per_chunk``; in single-tower mode a single pool covers
        the concatenated joint sequence (sum of both per-modality
        sizes), matching the pre-MoT behavior."""
        lat_total = (attn_window // 2) * latent_token_per_chunk
        act_total = (attn_window // 2) * action_token_per_chunk
        if self.use_mot:
            totals = {'lat': lat_total, 'act': act_total}
            for block in self.blocks:
                for m in _MODALITIES:
                    block.attn1[m].init_kv_cache(
                        cache_name, totals[m],
                        self.num_attention_heads,
                        self.attention_head_dim,
                        device, dtype, batch_size)
        else:
            total = lat_total + act_total
            for block in self.blocks:
                block.attn1.init_kv_cache(
                    cache_name, total,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    device, dtype, batch_size)

    # ---------- input embedding ----------

    def _input_embed(self, latents, input_type='latent'):
        if input_type == 'latent':
            hidden_states = rearrange(
                latents,
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            hidden_states = self.patch_embedding_mlp(hidden_states)
        elif input_type == 'action':
            hidden_states = rearrange(latents, 'b c f h w -> b (f h w) c')
            hidden_states = self.action_embedder(hidden_states)
        elif input_type == 'text':
            hidden_states = self.condition_embedder.text_embedder(latents)
        elif input_type == 'text_action':
            hidden_states = self.condition_embedder_action.text_embedder(latents)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        return hidden_states

    def _time_embed(self, timesteps, H, W, dtype, action_mode=False):
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps,
            (H // pach_scale_h) * (W // pach_scale_w), dim=1)
        current_condition_embedder = (self.condition_embedder_action
                                      if action_mode
                                      else self.condition_embedder)
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C
        return temb, timestep_proj

    # ---------- inference forward ----------

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name='pos',
        action_mode=False,
        train_mode=False,
        video_only=False,
    ):
        r"""Inference forward.

        Args:
            train_mode: unsupported in this inference-only build.
            video_only: unsupported in this inference-only build.
            action_mode: inference-time flag. ``False`` runs the lat
                stream, ``True`` runs the act stream. The KV cache from
                the previously-run modality is read transparently inside
                the MoT block.
        """
        if train_mode or video_only:
            raise NotImplementedError(
                "This inference-only build does not include training forwards."
            )

        # --- Build single-modality input (inference) ---
        if action_mode:
            latent_hidden_states = self._input_embed(
                input_dict['noisy_latents'], input_type='action')
            text_hidden_states = (self._input_embed(
                input_dict['text_emb'], input_type='text_action')
                if self.use_mot
                else self._input_embed(
                    input_dict['text_emb'], input_type='text'))
        else:
            latent_hidden_states = self._input_embed(
                input_dict['noisy_latents'], input_type='latent')
            text_hidden_states = self._input_embed(
                input_dict['text_emb'], input_type='text')

        rotary_emb = self.rope(input_dict['grid_id'])[:, :, None]

        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            input_dict['timesteps'],
            (input_dict['noisy_latents'].shape[-2] // pach_scale_h) *
            (input_dict['noisy_latents'].shape[-1] // pach_scale_w), dim=1)
        current_condition_embedder = (self.condition_embedder_action
                                      if action_mode
                                      else self.condition_embedder)
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=latent_hidden_states.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        if self.use_mot:
            modality = 'act' if action_mode else 'lat'
            for block in self.blocks:
                latent_hidden_states = block(
                    latent_hidden_states,
                    text_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    modality=modality,
                    update_cache=update_cache,
                    cache_name=cache_name,
                )
            out_table = (self.scale_shift_table_act if action_mode
                         else self.scale_shift_table_lat)
            out_norm = (self.action_norm_out if action_mode
                        else self.norm_out)
        else:
            for block in self.blocks:
                latent_hidden_states = block(
                    latent_hidden_states,
                    text_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    update_cache=update_cache,
                    cache_name=cache_name,
                )
            out_table = self.scale_shift_table
            out_norm = self.norm_out

        temb_scale_shift_table = out_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table,
                                 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(latent_hidden_states.device).squeeze(1)
        scale = scale.to(latent_hidden_states.device).squeeze(1)
        latent_hidden_states = (out_norm(latent_hidden_states.float()) *
                                (1. + scale) + shift).type_as(latent_hidden_states)

        if action_mode:
            latent_hidden_states = self.action_proj_out(latent_hidden_states)
        else:
            latent_hidden_states = self.proj_out(latent_hidden_states)
            latent_hidden_states = rearrange(latent_hidden_states,
                                             'b l (n c) -> b (l n) c',
                                             n=math.prod(self.patch_size))
        return latent_hidden_states


if __name__ == '__main__':
    model = WanTransformer3DModel(patch_size=[1, 2, 2],
                                  num_attention_heads=24,
                                  attention_head_dim=128,
                                  in_channels=48,
                                  out_channels=48,
                                  action_dim=30,
                                  text_dim=4096,
                                  freq_dim=256,
                                  ffn_dim=14336,
                                  num_layers=30,
                                  cross_attn_norm=True,
                                  eps=1e-6,
                                  rope_max_seq_len=1024,
                                  pos_embed_seq_len=None,
                                  attn_mode="torch",
                                  use_mot=True,
                                  mot_action_hidden_dim=768)
