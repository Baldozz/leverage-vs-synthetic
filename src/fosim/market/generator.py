"""Market scenario generator (SPEC §4): Config → immutable ``MarketPaths``.

Random numbers: ``SeedSequence(seed).spawn(len(STREAMS))`` gives every risk factor its own PCG64
stream in a fixed order. **All streams are always drawn** (even for disabled modules), so toggling
jumps, the rate model or the IV model never changes the draws of any other factor (test 14).
Correlated shocks are Z_indep · Lᵀ with the driver order [indices…, illiquids…, IV, RATES], so a
factor's correlated draw depends only on factors earlier in the order.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from numpy.random import PCG64, Generator, SeedSequence
from numpy.typing import NDArray

from fosim.config.schema import SimConfig
from fosim.engine.conventions import TimeGrid
from fosim.market.correlation import validate_correlation
from fosim.market.equity import (
    index_params,
    running_drawdown,
    simulate_held_portfolio,
    simulate_indices,
)
from fosim.market.illiquids import simulate_illiquids, simulate_true_returns
from fosim.market.implied_vol import simulate_iv_short
from fosim.market.paths import MarketPaths
from fosim.market.rates import build_rate_model

F64 = NDArray[np.float64]

STREAMS: tuple[str, ...] = ("equity", "held", "illiquids", "iv", "rates", "jump_clock", "jump_size", "iv_jump", "bootstrap", "counterparty")


def spawn_streams(seed: int) -> dict[str, Generator]:
    children = SeedSequence(seed).spawn(len(STREAMS))
    return {name: Generator(PCG64(child)) for name, child in zip(STREAMS, children, strict=True)}


def build_correlation(cfg: SimConfig) -> tuple[list[str], F64]:
    names = [*cfg.index_names, *cfg.illiquid_names, "IV", "RATES"]
    if cfg.correlation is not None:
        order = [cfg.correlation.names.index(n) for n in names]
        m = np.asarray(cfg.correlation.matrix, dtype=np.float64)[np.ix_(order, order)]
        return names, m
    # build from rho_equity fields: equities mutually 0 (user should supply a full matrix for >1 index)
    n = len(names)
    m = np.eye(n)
    n_idx = len(cfg.index_names)
    if n_idx > 1:
        warnings.warn("No correlation block: multiple indices assumed uncorrelated (supply correlation.matrix)", stacklevel=2)
    for j, ill in enumerate(cfg.illiquids):
        for i in range(n_idx):
            m[n_idx + j, i] = m[i, n_idx + j] = ill.rho_equity
    iv_row = n_idx + len(cfg.illiquids)
    for i in range(n_idx):
        m[iv_row, i] = m[i, iv_row] = cfg.implied_vol.short.rho_equity
        m[iv_row + 1, i] = m[i, iv_row + 1] = cfg.rates.usd.rho_equity
    return names, m


def inception_illiquid_nav(cfg: SimConfig) -> F64:
    """Inception illiquid NAV per asset (identical in A, B, C): pro_rata → gross × w_ill; equity_only → NAV × w_ill."""
    if cfg.custom_inception.enabled:
        return np.array([cfg.custom_inception.illiquids * i.weight for i in cfg.illiquids], dtype=np.float64)
    nav0 = cfg.portfolio.nav0
    loan = cfg.leverage.ratio_of_nav * nav0
    if cfg.leverage.allocation_of_borrowed_funds == "pro_rata":
        total = (nav0 + loan) * cfg.portfolio.weights.illiquids
    else:
        total = nav0 * cfg.portfolio.weights.illiquids
    return np.array([total * i.weight for i in cfg.illiquids], dtype=np.float64)


def generate_market_paths(cfg: SimConfig, n_paths: int | None = None, base_dir: Path | None = None) -> MarketPaths:
    if cfg.equity_model.type == "bootstrap":
        from fosim.market.bootstrap import generate_bootstrap_paths

        return generate_bootstrap_paths(cfg, n_paths=n_paths, base_dir=base_dir)
    if cfg.equity_model.type == "historical_replay":
        from fosim.market.stress import build_historical_replay

        h = cfg.historical
        assert h.file is not None and h.start is not None
        return build_historical_replay(cfg, h.file, h.start, dict(h.columns), base_dir=base_dir, held_column=h.held_column).paths
    P = n_paths if n_paths is not None else cfg.run.n_paths
    grid = TimeGrid.build(cfg.run.horizon_years, cfg.run.dt)
    N, dt = grid.n_steps, grid.dt
    rng = spawn_streams(cfg.run.seed)
    names, corr = build_correlation(cfg)
    vc = validate_correlation(names, corr)
    n_idx, n_ill = len(cfg.index_names), len(cfg.illiquids)

    # --- independent draws, one stream per factor (always drawn) ---
    z_eq = rng["equity"].standard_normal((P, N, n_idx))
    z_held = rng["held"].standard_normal((P, N))
    z_ill = rng["illiquids"].standard_normal((P, N, n_ill)) if n_ill else np.zeros((P, N, 0))
    z_iv = rng["iv"].standard_normal((P, N))
    z_r = rng["rates"].standard_normal((P, N))
    lam = cfg.equity_model.jump_lambda if cfg.equity_model.jumps_enabled else 0.0
    jump_counts = rng["jump_clock"].poisson(lam * dt, size=(P, N)).astype(np.float64)
    jump_z = rng["jump_size"].standard_normal((P, N, n_idx))
    _ = rng["iv_jump"].standard_normal((P, N))  # reserved (deterministic IV jump add in v1)

    z_ind = np.concatenate([z_eq, z_ill, z_iv[:, :, None], z_r[:, :, None]], axis=2)
    z_cor = z_ind @ vc.cholesky.T
    zc_eq = z_cor[:, :, :n_idx]
    zc_ill = z_cor[:, :, n_idx : n_idx + n_ill]
    zc_iv = z_cor[:, :, n_idx + n_ill]
    zc_r = z_cor[:, :, n_idx + n_ill + 1]

    # --- rates ---
    rate_model, r0 = build_rate_model(cfg.rates, base_dir)
    r_short = rate_model.short_rate_path(r0, P, N, dt, zc_r if rate_model.stochastic else None)

    # --- implied vol ---
    use_jumps = cfg.equity_model.type in ("merton", "coupled_stochvol") and cfg.equity_model.jumps_enabled and lam > 0
    iv_short = simulate_iv_short(cfg.implied_vol.short, dt, zc_iv, jump_counts if use_jumps else None)

    # --- equities ---
    params = [index_params(c) for c in cfg.equity_indices]
    S, log_jumps = simulate_indices(
        params, cfg.equity_model, dt, zc_eq, jump_counts if use_jumps else None, jump_z if use_jumps else None, iv_short
    )
    held = simulate_held_portfolio(S, cfg.held_equity_portfolio, cfg.index_names, dt, z_held)

    # --- illiquids ---
    bw = np.array([c.option_book_weight for c in cfg.equity_indices], dtype=np.float64)
    book_log_jump = log_jumps @ bw if use_jumps else None
    jm = float(np.dot(bw, [p.jump_mean for p in params]))
    jv = float(np.sqrt(np.dot(bw**2, [p.jump_vol**2 for p in params])))
    G = simulate_true_returns(cfg.illiquids, dt, zc_ill, book_log_jump, lam if use_jumps else 0.0, jm, jv)
    book_level = np.exp(np.sum(np.log(S / S[:, :1, :]) * bw, axis=2))
    dd = running_drawdown(book_level)
    ill = simulate_illiquids(cfg.illiquids, inception_illiquid_nav(cfg), G, grid.steps_per_year, grid.is_month_end, grid.month_index, dd)

    return MarketPaths(
        grid=grid, index_names=list(cfg.index_names), index_params=params, S=S, held=held, iv_short=iv_short,
        r_short=r_short, jump_counts=jump_counts, log_jumps=log_jumps, illiquids=ill, rate_model=rate_model,
        book_weights=bw, seed=cfg.run.seed, correlation_repaired=vc.repaired,
        meta={"correlation_names": names, "frobenius_distance": vc.frobenius_distance, "model": cfg.equity_model.type},
    )
