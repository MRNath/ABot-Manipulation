# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Logging setup for the WAM inference server.

Configures a shared root logger with a concise timestamped format and
silences the verbose Kineto profiler output.  Uses ``dictConfig`` for a
declarative configuration instead of imperative handler attachment.
"""
from __future__ import annotations

import logging
import logging.config
import os

_LOGGING_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}

logger = logging.getLogger()


def init_logger() -> None:
    logging.config.dictConfig(_LOGGING_CONFIG)
    os.environ["KINETO_LOG_LEVEL"] = "5"
