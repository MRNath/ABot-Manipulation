# Copyright 2024-2025 The Abot Team Authors. All rights reserved.
"""Shared helper: load q01/q99 for a subset from libero_norm_stats.json and pad to action_dim.

The JSON corresponds to the LIBERO-lerobot 4 subsets
(libero_10 / libero_goal / libero_object / libero_spatial) + libero_all (aggregate of all 4 subsets)
action quantile statistics.
"""

import json
from pathlib import Path
from typing import Dict

# JSON path relative to this file
_NORM_STATS_JSON = Path(__file__).parent / "libero_norm_stats.json"

# subset name whitelist
VALID_SUBSETS = ("libero_10", "libero_goal", "libero_object", "libero_spatial", "libero_all")

# cache: avoid re-reading JSON on every import
_cache: Dict[str, dict] = {}


def _load_blob() -> dict:
    if "blob" not in _cache:
        if not _NORM_STATS_JSON.is_file():
            raise FileNotFoundError(
                f"libero_norm_stats.json not found at {_NORM_STATS_JSON}. "
                f"Please run the quantile computation script first."
            )
        _cache["blob"] = json.loads(_NORM_STATS_JSON.read_text(encoding="utf-8"))
    return _cache["blob"]


def load_libero_norm_stat(subset: str, action_dim: int = 30, real_dim: int = 7) -> dict:
    """Load a subset's norm_stat, padded to action_dim.

    Args:
        subset: 'libero_10' | 'libero_goal' | 'libero_object' | 'libero_spatial' | 'libero_all'
        action_dim: total dimension of the final norm_stat (default 30)
        real_dim: number of action channels actually used (default 7)

    Returns:
        {"q01": [...action_dim...], "q99": [...action_dim...]}
    """
    if subset not in VALID_SUBSETS:
        raise ValueError(f"unknown subset '{subset}'. Valid: {VALID_SUBSETS}")
    if real_dim > action_dim:
        raise ValueError(f"real_dim ({real_dim}) > action_dim ({action_dim})")

    blob = _load_blob()
    if subset not in blob["stats"]:
        raise KeyError(
            f"subset '{subset}' not in JSON stats (have: {list(blob['stats'].keys())}). "
        )
    raw = blob["stats"][subset]

    pad_n = action_dim - real_dim
    return {
        "q01": list(raw["q01"][:real_dim]) + [0.0] * pad_n,
        "q99": list(raw["q99"][:real_dim]) + [0.0] * pad_n,
    }


def list_libero_subsets() -> list:
    """Return the subset names actually present in the JSON."""
    blob = _load_blob()
    return list(blob["stats"].keys())
