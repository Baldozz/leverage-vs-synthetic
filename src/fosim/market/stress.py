"""Deterministic stress scenarios (SPEC §4.7): stylised builder and historical replay.

**Stylised scenarios are NOT historical data.** They are parametric shapes (drawdown depth, months
to trough, V/U/L recovery, IV peak and decay, rate shock, illiquid markdown with reporting lag,
capital-call acceleration). Templates live in ``config/stress_templates.yaml`` and every parameter
is editable. Historical replay uses only user-supplied files (``data/*.csv``).

Both produce a single-path ``MarketPaths`` (n_paths = 1) plus a dict of config overrides
(LTV haircuts, bid/ask stress) that the caller applies before running the strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from fosim.config import STRESS_TEMPLATES_PATH
from fosim.config.schema import SimConfig
from fosim.engine.conventions import TimeGrid
from fosim.market.equity import index_params, running_drawdown
from fosim.market.generator import inception_illiquid_nav
from fosim.market.illiquids import simulate_illiquids
from fosim.market.paths import MarketPaths
from fosim.market.rates import build_rate_model

F64 = NDArray[np.float64]


class StylisedScenario(BaseModel):
    """All fields editable; labelled 'stylised, not historical data' in the UI."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    drawdown: float  # peak-to-trough, negative (e.g. -0.50)
    months_to_trough: int
    recovery_shape: Literal["V", "U", "L"]
    months_to_recover: int  # months from trough back to the prior peak (L: never fully; uses recovery_fraction)
    recovery_fraction: float = 1.0  # L-shape: fraction of the drawdown recovered by the end of months_to_recover
    iv_peak: float  # short-dated IV at the trough
    iv_decay_months: int  # half-life of IV decay after the trough
    rate_shock_bp: float = 0.0  # applied linearly over months_to_trough, held afterwards
    ltv_multiplier: float = 1.0  # bank cuts advance rates at the trough
    illiquid_markdown: float = 0.0  # true markdown of illiquids at the trough (fraction, negative)
    illiquid_reporting_lag_months: int = 0
    capital_call_multiplier: float = 1.0
    dividend_cut: float = 0.0
    post_scenario_drift: bool = True  # after recovery, continue at the user's expected return with zero vol


def load_templates(path: Path = STRESS_TEMPLATES_PATH) -> dict[str, StylisedScenario]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {k: StylisedScenario.model_validate({"name": k, **v}) for k, v in raw["templates"].items()}


def _shape_path(n_months: int, sc: StylisedScenario) -> F64:
    """Book index level path (monthly, normalised to 1 at t0) for the stylised scenario."""
    lvl = np.ones(n_months + 1)
    trough = 1.0 + sc.drawdown
    m_t, m_r = sc.months_to_trough, sc.months_to_recover
    for k in range(1, n_months + 1):
        if k <= m_t:
            lvl[k] = np.exp(np.log(trough) * k / m_t)  # geometric decline
        elif k <= m_t + m_r:
            u = (k - m_t) / m_r
            if sc.recovery_shape == "V":
                frac = u
            elif sc.recovery_shape == "U":
                frac = u * u * (3 - 2 * u)  # smoothstep: slow start, faster later
            else:  # L
                frac = sc.recovery_fraction * (1 - (1 - u) ** 2)
            lvl[k] = trough + (1.0 - trough) * frac
        else:
            lvl[k] = lvl[k - 1]
    return lvl


@dataclass(frozen=True)
class StressResult:
    paths: MarketPaths
    overrides: dict[str, Any]
    label: str


def build_stylised_paths(cfg: SimConfig, sc: StylisedScenario, base_dir: Path | None = None) -> StressResult:
    grid = TimeGrid.build(cfg.run.horizon_years, cfg.run.dt)
    N, dt = grid.n_steps, grid.dt
    n_months = round(cfg.run.horizon_years * 12)
    monthly = _shape_path(n_months, sc)
    # map to the step grid by interpolating log level in time (monthly nodes at t = m/12)
    tm = np.arange(n_months + 1) / 12.0
    lvl = np.exp(np.interp(grid.t, tm, np.log(monthly)))
    params = [index_params(c) for c in cfg.equity_indices]
    n_idx = len(params)
    S = np.empty((1, N + 1, n_idx))
    end_month = sc.months_to_trough + sc.months_to_recover
    for i, p in enumerate(params):
        drift = np.zeros(N + 1)
        if sc.post_scenario_drift:
            after = np.maximum(grid.t - end_month / 12.0, 0.0)
            drift = (p.m - p.q_c_px) * after
        S[0, :, i] = p.spot * lvl * np.exp(drift)
    # IV: rises linearly to the peak at the trough, decays exponentially with the given half-life
    theta = cfg.implied_vol.short.theta
    iv = np.empty(N + 1)
    t_trough = sc.months_to_trough / 12.0
    for k, t in enumerate(grid.t):
        if t <= t_trough:
            iv[k] = theta + (sc.iv_peak - theta) * (t / t_trough if t_trough > 0 else 1.0)
        else:
            hl = max(sc.iv_decay_months, 1) / 12.0
            iv[k] = theta + (sc.iv_peak - theta) * 0.5 ** ((t - t_trough) / hl)
    iv_short = np.clip(iv, cfg.implied_vol.short.floor, cfg.implied_vol.short.cap)[None, :]
    # rates: linear shock to the trough, then held
    rate_model, r0 = build_rate_model(cfg.rates, base_dir)
    shock = sc.rate_shock_bp * 1e-4 * np.clip(grid.t / max(t_trough, 1e-9), 0.0, 1.0)
    r_short = (r0 + shock)[None, :]
    # held portfolio follows the book index with beta (no idiosyncratic noise)
    bw = np.array([c.option_book_weight for c in cfg.equity_indices])
    beta = np.array([cfg.held_equity_portfolio.betas[n] for n in cfg.index_names])
    dln = np.diff(np.log(S[0]), axis=0) @ beta
    held = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(dln)]))[None, :]
    # illiquids: deterministic expected drift plus a markdown at the trough (true), reported with lag
    n_ill = len(cfg.illiquids)
    G = np.zeros((1, N, n_ill))
    for j, ill in enumerate(cfg.illiquids):
        from fosim.engine.conventions import continuous_drift

        m = continuous_drift(ill.expected_return, ill.return_type, ill.vol)
        G[0, :, j] = np.expm1((m - 0.5 * ill.vol**2) * dt)
        if sc.illiquid_markdown != 0.0 and sc.months_to_trough > 0:
            k_t = round(t_trough * grid.steps_per_year)
            steps = max(k_t, 1)
            per_step = (1.0 + sc.illiquid_markdown) ** (1.0 / steps) - 1.0
            G[0, :steps, j] = (1.0 + G[0, :steps, j]) * (1.0 + per_step) - 1.0
    dd = running_drawdown(np.exp(np.sum(np.log(S / S[:, :1, :]) * bw, axis=2)))
    illiq = simulate_illiquids(cfg.illiquids, inception_illiquid_nav(cfg), G, grid.steps_per_year, grid.is_month_end, grid.month_index, dd)
    mp = MarketPaths(
        grid=grid, index_names=list(cfg.index_names), index_params=params, S=S, held=held, iv_short=iv_short,
        r_short=r_short, jump_counts=np.zeros((1, N)), log_jumps=np.zeros((1, N, n_idx)), illiquids=illiq,
        rate_model=rate_model, book_weights=bw, seed=cfg.run.seed, meta={"model": "stylised_stress", "scenario": sc.name},
    )
    overrides: dict[str, Any] = {}
    if sc.ltv_multiplier < 1.0:
        overrides["leverage.ltv_stress_schedule"] = [{"index_drawdown_below": sc.drawdown * 0.5, "multiplier": sc.ltv_multiplier}]
    if sc.capital_call_multiplier != 1.0:
        overrides["stress.capital_call_multiplier"] = sc.capital_call_multiplier
    if sc.dividend_cut != 0.0:
        overrides["stress.dividend_cut"] = sc.dividend_cut
    return StressResult(mp, overrides, f"Stylised: {sc.name} (not historical data)")


def build_historical_replay(
    cfg: SimConfig,
    file: str | Path,
    start: str,
    columns: dict[str, str],
    base_dir: Path | None = None,
    held_column: str | None = None,
) -> StressResult:
    """Replay a user-supplied history window (index levels; optional IV and RATES columns) on the run grid."""
    p = Path(file)
    if base_dir is not None and not p.is_absolute():
        p = base_dir / p
    if not p.exists() and not p.is_absolute():
        from fosim.config import PROJECT_ROOT

        p = PROJECT_ROOT / p
    df = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
    df = df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    grid = TimeGrid.build(cfg.run.horizon_years, cfg.run.dt)
    N = grid.n_steps
    if len(df) < N + 1:
        raise ValueError(f"history window from {start} has {len(df)} rows; the run needs {N + 1} at dt={cfg.run.dt}")
    df = df.iloc[: N + 1]
    params = [index_params(c) for c in cfg.equity_indices]
    n_idx = len(params)
    S = np.empty((1, N + 1, n_idx))
    for i, c in enumerate(cfg.equity_indices):
        col = columns.get(c.name)
        if col is None:
            raise ValueError(f"columns must map index {c.name!r} to a column of {p}")
        x = df[col].to_numpy(dtype=np.float64)
        S[0, :, i] = c.spot * x / x[0]
    rate_model, r0 = build_rate_model(cfg.rates, base_dir)
    def role_col(role: str) -> F64 | None:
        """A role's column as decimals; values quoted in percent / vol points (|max| > 1.5) are divided by 100."""
        name = columns.get(role)
        if name is None or name not in df.columns:
            return None
        v = df[name].to_numpy(dtype=np.float64)
        if np.isnan(v).any():
            raise ValueError(f"column {name!r} ({role}) has gaps in the replay window starting {start}; forward-fill or shorten the window")
        return v / 100.0 if np.nanmax(np.abs(v)) > 1.5 else v

    r_col = role_col("RATES")
    r_short = r_col[None, :] if r_col is not None else np.full((1, N + 1), r0)
    iv_col = role_col("IV")
    iv_short = np.clip(iv_col * cfg.historical.iv_scale, cfg.implied_vol.short.floor, cfg.implied_vol.short.cap)[None, :] if iv_col is not None else np.full((1, N + 1), cfg.implied_vol.short.theta)
    series: dict[str, F64] = {}
    for role, key in (("LOAN_BASE", "loan_base"), ("CASH_YIELD", "cash_yield"), ("OPTION_RATE", "option_rate")):
        v = role_col(role)
        if v is not None:
            series[key] = v[None, :]
    dv = role_col("DIV_YIELD")
    if dv is not None:
        series["div_yield"] = np.log1p(dv)[None, :]  # continuous yield from the annual yield
    bw = np.array([c.option_book_weight for c in cfg.equity_indices])
    if held_column is not None:
        hcol = df[held_column].to_numpy(dtype=np.float64)
        held = (100.0 * hcol / hcol[0])[None, :]
    else:
        beta = np.array([cfg.held_equity_portfolio.betas[n] for n in cfg.index_names])
        dln = np.diff(np.log(S[0]), axis=0) @ beta
        held = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(dln)]))[None, :]
    n_ill = len(cfg.illiquids)
    G = np.zeros((1, N, n_ill))
    from fosim.engine.conventions import continuous_drift

    for j, ill in enumerate(cfg.illiquids):
        m = continuous_drift(ill.expected_return, ill.return_type, ill.vol)
        G[0, :, j] = np.expm1((m - 0.5 * ill.vol**2) * grid.dt)
    dd = running_drawdown(np.exp(np.sum(np.log(S / S[:, :1, :]) * bw, axis=2)))
    illiq = simulate_illiquids(cfg.illiquids, inception_illiquid_nav(cfg), G, grid.steps_per_year, grid.is_month_end, grid.month_index, dd)
    mp = MarketPaths(
        grid=grid, index_names=list(cfg.index_names), index_params=params, S=S, held=held, iv_short=iv_short,
        r_short=r_short, jump_counts=np.zeros((1, N)), log_jumps=np.zeros((1, N, n_idx)), illiquids=illiq,
        rate_model=rate_model, book_weights=bw, seed=cfg.run.seed,
        meta={"model": "historical_replay", "file": str(p), "start": start, "end": str(df["date"].iloc[-1].date()), "series": sorted(series)},
        series=series,
    )
    return StressResult(mp, {}, f"Historical replay from {start} ({p.name})")
