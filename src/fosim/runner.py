"""Pipeline helpers: Config → MarketPaths → StrategyEngine → results, with dotted-path overrides."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fosim.config.schema import SimConfig
from fosim.engine.simulator import Simulator
from fosim.engine.state import StrategyResult
from fosim.market.generator import generate_market_paths
from fosim.market.paths import MarketPaths


def with_overrides(cfg: SimConfig, overrides: dict[str, Any]) -> SimConfig:
    """Return a new validated config with dotted-path overrides applied (e.g. ``"rates.usd.r0": 0.03``).

    List elements are addressed by integer segments: ``"equity_indices.0.realized_vol"``.
    """
    raw = copy.deepcopy(cfg.model_dump(mode="python"))
    for key, value in overrides.items():
        node: Any = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[int(p)] if p.isdigit() else node[p]
        last = parts[-1]
        if last.isdigit():
            node[int(last)] = value
        else:
            node[last] = value
    return SimConfig.model_validate(raw)


@dataclass
class RunOutput:
    cfg: SimConfig
    paths: MarketPaths
    results: dict[str, StrategyResult]
    timings: dict[str, float] = field(default_factory=dict)


def run_config(
    cfg: SimConfig,
    n_paths: int | None = None,
    ledger_paths: list[int] | None = None,
    base_dir: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
    paths: MarketPaths | None = None,
) -> RunOutput:
    """Generate paths (unless supplied) and run all configured strategies on them."""
    import time

    t0 = time.time()
    mp = paths if paths is not None else generate_market_paths(cfg, n_paths=n_paths, base_dir=base_dir)
    t1 = time.time()
    sim = Simulator(cfg, mp, ledger_paths=ledger_paths, base_dir=base_dir)
    results = sim.run(progress=progress)
    t2 = time.time()
    return RunOutput(cfg, mp, results, {"paths_seconds": t1 - t0, "engine_seconds": t2 - t1})
