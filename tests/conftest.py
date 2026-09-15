"""Shared fixtures: raw default config, override helper, deterministic zero-vol settings."""

from __future__ import annotations

import copy
from typing import Any

import pytest
import yaml

from fosim.config import DEFAULT_CONFIG_PATH, SimConfig


@pytest.fixture
def raw() -> dict[str, Any]:
    with open(DEFAULT_CONFIG_PATH) as f:
        d: dict[str, Any] = yaml.safe_load(f)
    return d


def apply_overrides(d: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    d = copy.deepcopy(d)
    for k, v in over.items():
        node = d
        parts = k.split(".")
        for p in parts[:-1]:
            if p.isdigit():
                node = node[int(p)]
            else:
                node = node[p]
        last = int(parts[-1]) if parts[-1].isdigit() else parts[-1]
        node[last] = v
    return d


def make_cfg(raw: dict[str, Any], **over: Any) -> SimConfig:
    return SimConfig.model_validate(apply_overrides(raw, over))


def zero_vol_overrides(**extra: Any) -> dict[str, Any]:
    """Deterministic market: zero realised vol, no jumps, constant IV, flat rates, no tracking error, no costs."""
    base: dict[str, Any] = {
        "equity_model.type": "gbm",
        "equity_model.jumps_enabled": False,
        "equity_indices.0.realized_vol": 0.0,
        "equity_indices.1.realized_vol": 0.0,
        "held_equity_portfolio.tracking_error_vol": 0.0,
        "implied_vol.short.eta": 0.0,
        "implied_vol.short.jump_add": 0.0,
        "illiquids.0.vol": 0.0,
        "illiquids.1.vol": 0.0,
        "illiquids.2.vol": 0.0,
        "illiquids.0.jump_beta": 0.0,
        "illiquids.1.jump_beta": 0.0,
        "illiquids.2.jump_beta": 0.0,
        "rates.model": "flat",
        "costs.equity_bps": 0,
        "costs.equity_bps_stressed": 0,
        "pricing.bid_ask_vol_pts_new": 0.0,
        "pricing.bid_ask_vol_pts_unwind": 0.0,
        "pricing.commission_bps_notional": 0,
        "run.n_paths": 4,
    }
    base.update(extra)
    return base
