# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Multi-rank websocket serving.

In async server mode rank 0 owns the websocket endpoint while every other
rank runs a broadcast loop: rank 0 forwards each incoming observation to the
workers via ``dist.broadcast_object_list`` so all ranks stay in lockstep
through the FSDP collectives inside the model forward.
"""
import torch
import torch.distributed as dist

from .logging import logger
from .policy_server import WebsocketPolicyServer


class DistributedModelWrapper:
    """Expose ``distributed_infer`` through the single-method policy API
    expected by ``WebsocketPolicyServer``."""

    def __init__(self, model, local_rank):
        self.model = model
        self.local_rank = local_rank

    def infer(self, obs):
        return distributed_infer(self.model, obs, self.local_rank)


def distributed_infer(model, obs, local_rank):
    """Run ``model.infer`` on rank 0 and keep the other ranks in sync.

    Rank 0 first broadcasts a command token so workers enter their infer
    branch, then broadcasts the observation itself before every rank calls
    ``model.infer`` (required for FSDP's collective all-gathers).
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return model.infer(obs)

    rank = dist.get_rank()
    assert rank == local_rank, "distributed_infer can only run at (rank 0)"

    cmd = torch.tensor(1,
                       dtype=torch.int64,
                       device='cuda' if torch.cuda.is_available() else 'cpu')
    dist.broadcast(cmd, src=0)

    obj_list = [obs]
    dist.broadcast_object_list(obj_list, src=0)

    result = model.infer(obs)

    return result


def worker_loop(model, local_rank):
    """Non-rank-0 loop: wait for command tokens and join each infer call."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rank = dist.get_rank()

    while True:
        cmd = torch.zeros(1, dtype=torch.int64, device=device)
        dist.broadcast(cmd, src=0)
        cmd_val = cmd.item()

        if cmd_val == -1:
            break
        elif cmd_val == 1:
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=0)
            obs = obj_list[0]
            _ = model.infer(obs)
        else:
            pass

    logger.info(f"[worker_loop] Rank {rank} exiting.")


def run_async_server_mode(model, local_rank, host, port):
    """Serve the policy over websocket on rank 0; workers run ``worker_loop``."""
    logger.info("Running in ASYNC SERVER mode")
    if local_rank == 0:
        dist_model = DistributedModelWrapper(model, local_rank=local_rank)
        model_server = WebsocketPolicyServer(dist_model, host=host, port=port)
        model_server.serve_forever()

        if dist.is_initialized() and dist.get_world_size() > 1:
            cmd = torch.tensor(
                -1,
                dtype=torch.int64,
                device='cuda' if torch.cuda.is_available() else 'cpu')
            dist.broadcast(cmd, src=0)
    else:
        try:
            worker_loop(model, local_rank)
        except KeyboardInterrupt:
            logger.info(f"Rank {local_rank}: Shutting down")
