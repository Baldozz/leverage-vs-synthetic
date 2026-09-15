"""Stationary block bootstrap (Politis–Romano 1994) of jointly sampled historical rows (SPEC §4.1.4).

Input: a CSV with a ``date`` column and one column per series, at the configured frequency
(``monthly`` or ``daily``). Column roles are mapped in ``bootstrap.columns``:
    {SPX: "spx", IV: "vix", RATES: "sofr"}   (IV and RATES optional → fall back to the models)
Index columns may be price levels (``bootstrap.series_type: price``) or simple/log returns.
Blocks have geometric lengths with mean ``expected_block_length``; rows are resampled jointly so
cross-dependence (index vs. implied vol vs. rates) is preserved. Optional re-centring shifts each
index's mean log return to the user's expected return (P-measure) while keeping the shape.

Only rows and columns supplied by the user are used; no market data is hard-coded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fosim.config.schema import SimConfig
from fosim.engine.conventions import TimeGrid
from fosim.market.equity import index_params, running_drawdown, simulate_held_portfolio
from fosim.market.generator import inception_illiquid_nav, spawn_streams
from fosim.market.illiquids import simulate_illiquids, simulate_true_returns
from fosim.market.implied_vol import simulate_iv_short
from fosim.market.paths import MarketPaths
from fosim.market.rates import build_rate_model

F64 = NDArray[np.float64]


def stationary_block_indices(rng: np.random.Generator, n_obs: int, n_paths: int, n_steps: int, mean_block: float) -> NDArray[np.intp]:
    """Row indices (n_paths, n_steps) from the stationary bootstrap with geometric block lengths."""
    p = 1.0 / mean_block
    idx = np.empty((n_paths, n_steps), dtype=np.intp)
    start = rng.integers(0, n_obs, size=n_paths)
    idx[:, 0] = start
    for k in range(1, n_steps):
        new_block = rng.random(n_paths) < p
        nxt = np.where(new_block, rng.integers(0, n_obs, size=n_paths), (idx[:, k - 1] + 1) % n_obs)
        idx[:, k] = nxt
    return idx


def load_bootstrap_table(cfg: SimConfig, base_dir: Path | None) -> pd.DataFrame:
    assert cfg.bootstrap.file is not None
    p = Path(cfg.bootstrap.file)
    if base_dir is not None and not p.is_absolute():
        p = base_dir / p
    if not p.exists() and not p.is_absolute():
        from fosim.config import PROJECT_ROOT

        p = PROJECT_ROOT / p
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    if "date" in cols:
        df = df.rename(columns={cols["date"]: "date"}).sort_values("date").reset_index(drop=True)
    return df


def generate_bootstrap_paths(cfg: SimConfig, n_paths: int | None = None, base_dir: Path | None = None) -> MarketPaths:
    bs = cfg.bootstrap
    if bs.frequency != cfg.run.dt:
        raise ValueError(f"bootstrap.frequency={bs.frequency} must equal run.dt={cfg.run.dt}")
    P = n_paths if n_paths is not None else cfg.run.n_paths
    grid = TimeGrid.build(cfg.run.horizon_years, cfg.run.dt)
    N, dt = grid.n_steps, grid.dt
    rng = spawn_streams(cfg.run.seed)
    df = load_bootstrap_table(cfg, base_dir)
    params = [index_params(c) for c in cfg.equity_indices]
    n_idx = len(params)

    # --- index log returns from the table ---
    logret = np.empty((len(df) - 1 if bs.series_type == "price" else len(df), n_idx))
    for i, c in enumerate(cfg.equity_indices):
        col = bs.columns.get(c.name)
        if col is None or col not in df.columns:
            raise ValueError(f"bootstrap.columns must map index {c.name!r} to a column of {bs.file}")
        x = df[col].to_numpy(dtype=np.float64)
        if bs.series_type == "price":
            logret[:, i] = np.diff(np.log(x))
        elif bs.series_type == "simple_return":
            logret[:, i] = np.log1p(x)
        else:
            logret[:, i] = x
    n_obs = logret.shape[0]
    if bs.recentre_means:
        for i, p in enumerate(params):
            target = (p.m - p.q_c_px - 0.5 * p.sigma**2) * dt
            logret[:, i] += target - logret[:, i].mean()
    idx = stationary_block_indices(rng["bootstrap"], n_obs, P, N, bs.expected_block_length)
    incr = logret[idx]  # (P, N, n_idx)
    S = np.empty((P, N + 1, n_idx))
    for i, p in enumerate(params):
        S[:, 0, i] = p.spot
        S[:, 1:, i] = p.spot * np.exp(np.cumsum(incr[:, :, i], axis=1))

    # --- IV and rates: from the table if mapped, else from the parametric models (own streams) ---
    z_iv = rng["iv"].standard_normal((P, N))
    z_r = rng["rates"].standard_normal((P, N))
    rate_model, r0 = build_rate_model(cfg.rates, base_dir)
    iv_col = bs.columns.get("IV")
    if iv_col is not None and iv_col in df.columns:
        iv_series = df[iv_col].to_numpy(dtype=np.float64)
        if bs.series_type == "price":
            iv_series = iv_series[1:]
        iv_short = np.empty((P, N + 1))
        iv_short[:, 0] = cfg.implied_vol.short.iv0 or cfg.implied_vol.short.theta
        iv_short[:, 1:] = np.clip(iv_series[idx], cfg.implied_vol.short.floor, cfg.implied_vol.short.cap)
    else:
        iv_short = simulate_iv_short(cfg.implied_vol.short, dt, z_iv, None)
    r_col = bs.columns.get("RATES")
    if r_col is not None and r_col in df.columns:
        r_series = df[r_col].to_numpy(dtype=np.float64)
        if bs.series_type == "price":
            r_series = r_series[1:]
        r_short = np.empty((P, N + 1))
        r_short[:, 0] = r0
        r_short[:, 1:] = r_series[idx]
    else:
        r_short = rate_model.short_rate_path(r0, P, N, dt, z_r if rate_model.stochastic else None)

    z_held = rng["held"].standard_normal((P, N))
    held = simulate_held_portfolio(S, cfg.held_equity_portfolio, cfg.index_names, dt, z_held)
    n_ill = len(cfg.illiquids)
    z_ill = rng["illiquids"].standard_normal((P, N, n_ill)) if n_ill else np.zeros((P, N, 0))
    # illiquid correlation to equities: regress on the book log-return shock (standardised)
    bw = np.array([c.option_book_weight for c in cfg.equity_indices])
    book = incr @ bw
    zb = (book - book.mean()) / max(book.std(), 1e-12)
    zc_ill = np.empty_like(z_ill)
    for j, ill_cfg in enumerate(cfg.illiquids):
        zc_ill[:, :, j] = ill_cfg.rho_equity * zb + np.sqrt(max(1 - ill_cfg.rho_equity**2, 0.0)) * z_ill[:, :, j]
    G = simulate_true_returns(cfg.illiquids, dt, zc_ill, None, 0.0, 0.0, 0.0)
    dd = running_drawdown(np.exp(np.sum(np.log(S / S[:, :1, :]) * bw, axis=2)))
    illiq = simulate_illiquids(cfg.illiquids, inception_illiquid_nav(cfg), G, grid.steps_per_year, grid.is_month_end, grid.month_index, dd)
    return MarketPaths(
        grid=grid, index_names=list(cfg.index_names), index_params=params, S=S, held=held, iv_short=iv_short,
        r_short=r_short, jump_counts=np.zeros((P, N)), log_jumps=np.zeros((P, N, n_idx)), illiquids=illiq,
        rate_model=rate_model, book_weights=bw, seed=cfg.run.seed, meta={"model": "bootstrap", "file": str(bs.file), "n_obs": n_obs},
    )
