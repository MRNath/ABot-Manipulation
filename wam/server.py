# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Launch a WAM inference server.

Typical usage (see ``evaluation/_lib/serve.sh``)::

    python -m torch.distributed.run --nproc_per_node 1 wam/server.py \
        --config-name libero_all --port 29056 --save_root visualization/libero
"""
import os
import sys
from typing import Optional

import torch
import typer

_WAM_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_WAM_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wam.configs import WAM_CONFIGS
from wam.distributed.util import init_distributed
from wam.pipelines.core import WAMServerPolicy
from wam.utils import init_logger, logger, run_async_server_mode

app = typer.Typer(pretty_exceptions_show_locals=False)


def _apply_cli_overrides(config, port, save_root, save_pred_video, pred_video_fps, attn_mode):
    """Mutate ``config`` in place with values coming from the CLI."""
    config.save_pred_video = bool(save_pred_video)
    config.pred_video_fps = pred_video_fps
    if save_root is not None:
        config.save_root = save_root
    if attn_mode and attn_mode.strip():
        config.attn_mode = attn_mode.strip()
        logger.info(f"Override attn_mode from CLI: {config.attn_mode}")
    if port is not None:
        config.port = port
    return config


def _configure_distributed(config):
    """Set up the process group (multi-rank) or pin the CUDA device."""
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        init_distributed(world_size, local_rank, rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    return local_rank


@app.command()
def main(
    config_name: str = typer.Option(
        "libero_all", "--config-name",
        help=f"Config registry name, one of: {sorted(WAM_CONFIGS)}.",
    ),
    port: Optional[int] = typer.Option(
        None, "--port",
        help="Websocket port; falls back to the config's `port` when unset.",
    ),
    save_root: Optional[str] = typer.Option(
        None, "--save_root",
        help="Output directory; falls back to the config's `save_root`.",
    ),
    save_pred_video: int = typer.Option(
        1, "--save_pred_video",
        help="Save model-predicted videos in server mode (0/1).",
    ),
    pred_video_fps: int = typer.Option(
        10, "--pred_video_fps",
        help="FPS of saved predicted videos.",
    ),
    attn_mode: Optional[str] = typer.Option(
        None, "--attn-mode",
        help=(
            "Attention implementation override: 'torch' / 'flashattn'. Unset keeps the config value."
        ),
    ),
) -> None:
    config = _apply_cli_overrides(
        WAM_CONFIGS[config_name], port, save_root, save_pred_video,
        pred_video_fps, attn_mode,
    )
    local_rank = _configure_distributed(config)
    model = WAMServerPolicy.from_config(config)

    if config.infer_mode == "i2va":
        logger.info("Inference mode: i2av")
        model.generate()
    elif config.infer_mode == "server":
        logger.info("Inference mode: server")
        run_async_server_mode(model, local_rank, config.host, config.port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

    logger.info("All processes finished")


if __name__ == "__main__":
    init_logger()
    app()
