# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Torch-free client helpers for talking to WAM inference servers.

This package must stay importable without torch and must never import the
`wam` package: simulator / eval environments typically only have
`websockets` and `msgpack` installed.
"""

from .websocket_client_policy import WebsocketClientPolicy

__all__ = ["WebsocketClientPolicy"]
