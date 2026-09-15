"""Exposure and convexity analytics (SPEC §6.5)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fosim.analytics.inception import (
    book_index,
    held_index,
    inception_market,
    sizing_table,
    strike_for_mode,
)
from fosim.config.schema import SimConfig
from fosim.engine.state import StrategyResult
from fosim.market.paths import MarketPaths
from fosim.pricing.black_scholes import bsm_greeks
from fosim.strategies.base import held_beta_to_book, inception_balance_cfg

F64 = NDArray[np.float64]
FAN = (5, 25, 50, 75, 95)


def fan(series: F64, q: tuple[int, ...] = FAN) -> pd.DataFrame:
    return pd.DataFrame({f"p{p}": np.percentile(series, p, axis=0) for p in q})


def exposure_fans(results: dict[str, StrategyResult], paths: MarketPaths, cfg: SimConfig) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    beta = held_beta_to_book(cfg.held_equity_portfolio.betas, cfg.index_names, paths.book_weights, cfg.held_equity_portfolio.beta_to_option_book)
    a_exposure = results["A"].series("equity_exposure") if "A" in results else None
    for name, res in results.items():
        out[f"{name}_equity_exposure"] = fan(res.series("equity_exposure"))
        out[f"{name}_equity_exposure_pct_nav"] = fan(res.series("equity_exposure") / np.maximum(res.nav, 1e-300))
        if name == "B":
            out["B_dollar_delta"] = fan(res.series("dollar_delta"))
            if a_exposure is not None:
                out["B_pct_of_A_exposure"] = fan(res.series("equity_exposure") / np.maximum(a_exposure, 1e-300))
            for g in ("vega_1pt", "theta_year", "rho_100bp", "dq_100bp", "dollar_gamma"):
                out[f"B_{g}"] = fan(res.series(g))
        if res.meta.get("uses_margin"):
            out[f"{name}_effective_leverage"] = fan(res.series("effective_leverage"))
            out[f"{name}_utilisation"] = fan(res.series("utilisation"))
    # beta-adjusted total portfolio exposure incl. illiquids' equity beta: β_ill = ρ σ_ill / σ_eq
    sig_eq = cfg.equity_indices[book_index(cfg)].realized_vol
    for name, res in results.items():
        ill_beta = np.zeros_like(res.nav)
        il = paths.illiquids
        for j, ic in enumerate(cfg.illiquids):
            ill_beta_j = ic.rho_equity * ic.vol / sig_eq if sig_eq > 0 else 0.0
            ill_beta = ill_beta + ill_beta_j * il.nav_true[:, :, j]
        out[f"{name}_beta_adjusted_total_exposure_pct_nav"] = fan((res.series("equity_exposure") + ill_beta) / np.maximum(res.nav, 1e-300))
    out["_beta_held"] = pd.DataFrame({"beta_held": [float(beta)]})
    return out


def delta_value_grid(
    tranches: pd.DataFrame,
    S_now: float,
    r_now: float,
    q_now: float,
    index_shocks: F64,
    vol_shocks_pts: F64,
    rate_shocks_bp: F64,
) -> dict[str, F64]:
    """Book value, P&L and dollar delta for instantaneous shocks (index × vol × rate) — one path/date."""
    n_s, n_v, n_r = len(index_shocks), len(vol_shocks_pts), len(rate_shocks_bp)
    value = np.zeros((n_s, n_v, n_r))
    delta = np.zeros((n_s, n_v, n_r))
    base = 0.0
    for _, tr in tranches.iterrows():
        units, K, T, sig = float(tr["units"]), float(tr["strike"]), float(tr["residual_years"]), float(tr["vol_mid"])
        base += units * float(bsm_greeks(S_now, K, r_now, q_now, sig, T).price)
        for i, xs in enumerate(index_shocks):
            S = S_now * (1 + xs)
            for j, vs in enumerate(vol_shocks_pts):
                s = max(sig + vs / 100.0, 0.01)
                for k, rs in enumerate(rate_shocks_bp):
                    g = bsm_greeks(S, K, r_now + rs * 1e-4, q_now, s, T)
                    value[i, j, k] += units * float(g.price)
                    delta[i, j, k] += units * float(g.delta) * S
    return {"value": value, "pnl": value - base, "dollar_delta": delta, "base_value": np.array(base)}


def participation_profile(cfg: SimConfig, horizons: tuple[float, ...] = (1.0, 3.0, 5.0), levels: F64 | None = None) -> pd.DataFrame:
    """A vs B vs C portfolio value (ex illiquids) against the index level at horizons, static inception book.

    B's book: the configured sizing/strike mode at inception (all tranches struck at t0), marked at the
    horizon with the inception vol and rate; cash accrues at the cash rate; A's loan at the loan rate;
    dividends reinvested for A and C. Capture ratios: up = (V(1.3) − V(1))/(V_A(1.3) − V_A(1)), down likewise at 0.7.
    """
    mk = inception_market(cfg)
    bal = inception_balance_cfg(cfg)
    tab = sizing_table(cfg)
    row = tab[(tab.strike_mode == cfg.options.strike_mode) & (tab.sizing_mode == cfg.options.sizing_mode)].iloc[0]
    K = strike_for_mode(mk, cfg.options.strike_mode, cfg.options.strike_param, cfg.options.delta_definition)
    x = np.linspace(0.4, 1.8, 57) if levels is None else levels
    rows = []
    q_c = math.log1p(cfg.equity_indices[held_index(cfg)].dividend_yield)
    for h in horizons:
        T_rem = max(mk.T - h, 0.0)
        S = mk.S0 * x
        opt = float(row.notional) / mk.S0 * np.asarray(bsm_greeks(S, K, mk.r, mk.q, row.vol, T_rem).price)
        v_b = opt + float(row.cash) * math.exp(mk.cash_rate * h)
        v_a = bal["equity_A"] * x * math.exp(q_c * h) - bal["loan"] * math.exp(mk.loan_rate * h)
        v_c = bal["equity_C"] * x * math.exp(q_c * h)
        for xi, a, b, c in zip(x, v_a, v_b, v_c, strict=True):
            rows.append({"horizon": h, "index_level": xi, "V_A": a, "V_B": b, "V_C": c})
    df = pd.DataFrame(rows)
    caps = {}
    for h in horizons:
        d = df[df.horizon == h].set_index("index_level")
        lv = d.index.to_numpy(dtype=np.float64)
        i1 = float(lv[int(np.argmin(np.abs(lv - 1.0)))])
        iu = float(lv[int(np.argmin(np.abs(lv - 1.3)))])
        idn = float(lv[int(np.argmin(np.abs(lv - 0.7)))])
        caps[h] = {
            "up_capture_B_vs_A": (d.V_B[iu] - d.V_B[i1]) / (d.V_A[iu] - d.V_A[i1]),
            "down_capture_B_vs_A": (d.V_B[idn] - d.V_B[i1]) / (d.V_A[idn] - d.V_A[i1]),
            "up_capture_B_vs_C": (d.V_B[iu] - d.V_B[i1]) / (d.V_C[iu] - d.V_C[i1]),
            "down_capture_B_vs_C": (d.V_B[idn] - d.V_B[i1]) / (d.V_C[idn] - d.V_C[i1]),
        }
    df.attrs["capture"] = caps
    return df


def realised_capture(res: StrategyResult, paths: MarketPaths) -> dict[str, float]:
    """Regression of B's step returns on index returns in up and down months (up-beta, down-beta)."""
    level = paths.book_index_level()
    rx = level[:, 1:] / level[:, :-1] - 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        ry = np.where(res.nav[:, :-1] > 0, res.nav[:, 1:] / res.nav[:, :-1] - 1.0, 0.0)
    out = {}
    for label, mask in (("up", rx > 0), ("down", rx < 0), ("all", np.ones_like(rx, dtype=bool))):
        xs, ys = rx[mask], ry[mask]
        if xs.size > 2:
            beta = float(np.cov(xs, ys)[0, 1] / np.var(xs, ddof=1))
            alpha = float(ys.mean() - beta * xs.mean())
        else:
            beta, alpha = float("nan"), float("nan")
        out[f"{label}_beta"] = beta
        out[f"{label}_alpha_per_step"] = alpha
        out[f"{label}_n"] = int(xs.size)
    return out
