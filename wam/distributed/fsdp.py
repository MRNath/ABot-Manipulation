# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""FSDP sharding strategies for multi-GPU inference."""
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


def shard_model(model,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                mesh=None,
                wrap_granularity="nested"):
    """Shard ``model`` with fully_shard according to ``wrap_granularity``.

    - nested: fully_shard each block's attn1/attn2/ffn, then block, then root
      — lower peak memory, more collectives per forward/backward.
    - flat: single fully_shard on the root model — fewer communications,
      higher peak memory.
    - auto_transformer: fully_shard each block, then the root model.

    Orthogonal to HSDP (mesh) vs global 1D FSDP.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": False}
    if mesh is not None:
        fsdp_config["mesh"] = mesh

    if wrap_granularity == "flat":
        fully_shard(model, **fsdp_config)
        return model
    if wrap_granularity == "auto_transformer":
        if not hasattr(model, "blocks") or len(model.blocks) == 0:
            raise ValueError("auto_transformer requires model.blocks with at least one block")
        for block in model.blocks:
            fully_shard(block, **fsdp_config)
        fully_shard(model, **fsdp_config)
        return model
    if wrap_granularity != "nested":
        raise ValueError(f"wrap_granularity must be 'nested', 'flat' or 'auto_transformer', got {wrap_granularity!r}")

    for block in model.blocks:
        # Symmetric MoT blocks expose attn1/attn2/ffn as ModuleDict({'lat','act'}).
        # Two reasons we can't sub-shard these:
        #   1) ModuleDict has no forward(), so fully_shard rejects the container.
        #   2) MoTWanTransformerBlock.forward calls attn_self.project_qkv_self
        #      (and similar) directly — bypassing forward(), so FSDP's all-gather
        #      hook never fires and the sharded DTensor weights aren't unsharded.
        # Block-level fully_shard below still covers the per-modality params.
        for name in ("attn1", "attn2", "ffn"):
            mod = getattr(block, name)
            if not isinstance(mod, torch.nn.ModuleDict):
                fully_shard(mod, **fsdp_config)
        fully_shard(block, **fsdp_config)

    fully_shard(model, **fsdp_config)
    return model
