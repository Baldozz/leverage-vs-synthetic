"""Drawdown episodes, strategy snapshots at troughs, rolling vol and return distributions (historical view).

* ``drawdown_episodes``: peak → trough → recovery of a level series for every drawdown deeper than a
  threshold (episodes are separated by a full recovery to the prior peak; the last one may be open).
* ``strategy_snapshot``: what each strategy had at a given step — NAV, drawdown from its own peak, cash
  ("dry powder"), equity exposure (A/C: equities; B: option dollar-delta + deployed spot) and its share
  of NAV, A's margin utilisation, B's option book value and tranche count.
* ``risk_table``: annualised vol, Sharpe = CAGR / annualised vol and Sortino = CAGR / downside deviation
  (no cash-rate deduction, per the user's definition), worst step, VaR/CVaR 95/99 of step and 1-year
  returns, max drawdown — per strategy, from the recorded NAV path.
* ``rolling_vol``: annualised rolling standard deviation of log returns.
* ``return_distribution``: horizon returns of a level series (daily or rolling h-day) with VaR 95/99
  (loss quantiles) and CVaR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fosim.config.schema import SimConfig
from fosim.engine.state import StrategyResult
from fosim.market.paths import MarketPaths

F64 = NDArray[np.float64]


def drawdown_episodes(level: F64, threshold: float = -0.15) -> pd.DataFrame:
    lvl = np.asarray(level, dtype=np.float64)
    peak = np.maximum.accumulate(lvl)
    dd = lvl / peak - 1.0
    rows = []
    n = len(lvl)
    k = 0
    while k < n:
        if dd[k] < 0:
            start = k
            while k < n and dd[k] < 0:
                k += 1
            end = k  # first index back at (or above) the peak, or n
            seg = dd[start:end]
            depth = float(seg.min())
            if depth <= threshold:
                trough = start + int(np.argmin(seg))
                rows.append({"peak_idx": start - 1, "trough_idx": trough, "recovery_idx": end if end < n else None, "depth": depth, "recovered": end < n})
        else:
            k += 1
    return pd.DataFrame(rows, columns=["peak_idx", "trough_idx", "recovery_idx", "depth", "recovered"])


def strategy_snapshot(results: dict[str, StrategyResult], paths: MarketPaths, cfg: SimConfig, k: int, path: int = 0) -> pd.DataFrame:
    rows = {}
    for name, r in results.items():
        nav = r.nav[path]
        peak = nav[: k + 1].max()
        cash = float(r.series("cash")[path, k])
        expo = float(r.series("equity_exposure")[path, k])
        rows[name] = {
            "nav": float(nav[k]), "drawdown_from_own_peak": float(nav[k] / peak - 1.0), "cash_dry_powder": cash, "cash_pct_nav": cash / nav[k] if nav[k] else np.nan,
            "equity_exposure": expo, "exposure_pct_nav": expo / nav[k] if nav[k] else np.nan, "held_equities": float(r.series("held_mv")[path, k]),
            "deployed_spot": float(r.series("spot_mv")[path, k]), "option_book_value": float(r.series("option_mv")[path, k]), "n_tranches": int(r.series("n_tranches")[path, k]),
            "loan": float(r.series("loan")[path, k]), "utilisation": float(r.series("utilisation")[path, k]) if r.meta.get("uses_margin") else np.nan,
            "equity_fall_to_margin_call": float(margin_distance(r, cfg, paths, path)[k]) if r.meta.get("uses_margin") else np.nan,
        }
    return pd.DataFrame(rows).T


def episode_table(results: dict[str, StrategyResult], paths: MarketPaths, cfg: SimConfig, dates: pd.Series, threshold: float = -0.15, path: int = 0) -> pd.DataFrame:
    """One block of rows per episode: the underlying's peak/trough/depth and each strategy's snapshot at the trough."""
    hi = cfg.index_names.index("SPX") if "SPX" in cfg.index_names else 0
    eps = drawdown_episodes(paths.S[path, :, hi], threshold)
    rows = []
    for _, e in eps.iterrows():
        kt = int(e.trough_idx)
        snap = strategy_snapshot(results, paths, cfg, kt, path)
        for name, s in snap.iterrows():
            rows.append({
                "episode": f"{dates.iloc[int(e.peak_idx)].date()} → {dates.iloc[kt].date()}", "index_drawdown": e.depth,
                "recovered": dates.iloc[int(e.recovery_idx)].date() if e.recovered else "not yet", "strategy": name, **s.to_dict(),
            })
    return pd.DataFrame(rows)


def rolling_vol(nav: F64, window: int, steps_per_year: int) -> F64:
    r = np.diff(np.log(np.maximum(nav, 1e-300)))
    s = pd.Series(r).rolling(window).std(ddof=1).to_numpy() * np.sqrt(steps_per_year)
    return np.concatenate([[np.nan], s])


def risk_table(results: dict[str, StrategyResult], paths: MarketPaths, path: int = 0) -> pd.DataFrame:
    spy = paths.grid.steps_per_year
    rows = {}
    for name, r in results.items():
        nav = r.nav[path]
        ret = nav[1:] / nav[:-1] - 1.0
        cagr = float((nav[-1] / nav[0]) ** (1.0 / paths.grid.horizon_years) - 1.0)
        vol = ret.std(ddof=1) * np.sqrt(spy)
        down = np.sqrt((np.minimum(ret, 0.0) ** 2).mean() * spy)  # downside deviation below zero, annualised
        n1 = spy
        r1y = nav[n1:] / nav[:-n1] - 1.0 if len(nav) > n1 else np.array([np.nan])
        dd = nav / np.maximum.accumulate(nav) - 1.0
        rows[name] = {
            # Sharpe = annualised return (CAGR) / annualised vol of step returns — no cash-rate deduction (user's definition);
            # Sortino = CAGR / annualised downside deviation of step returns below zero
            "ann_vol": vol, "sharpe": cagr / vol if vol > 0 else np.nan, "sortino": cagr / down if down > 0 else np.nan,
            "worst_step": float(ret.min()), "best_step": float(ret.max()), "var95_step": float(-np.percentile(ret, 5)), "var99_step": float(-np.percentile(ret, 1)),
            "cvar99_step": float(-ret[ret <= np.percentile(ret, 1)].mean()), "var95_1y": float(-np.nanpercentile(r1y, 5)), "var99_1y": float(-np.nanpercentile(r1y, 1)),
            "worst_1y": float(np.nanmin(r1y)), "max_drawdown": float(dd.min()), "time_under_water": float((dd < -1e-12).mean()),
            "cagr": cagr,
        }
    return pd.DataFrame(rows).T


def return_distribution(level: F64, horizon_steps: int) -> dict[str, object]:
    """Returns over ``horizon_steps`` (1 = step-to-step; h = rolling h-step total returns) with VaR/CVaR 95/99 (losses)."""
    lvl = np.asarray(level, dtype=np.float64)
    ret = lvl[horizon_steps:] / lvl[:-horizon_steps] - 1.0
    q5, q1 = np.percentile(ret, 5), np.percentile(ret, 1)
    return {
        "returns": ret, "n": int(ret.size), "mean": float(ret.mean()), "median": float(np.median(ret)), "std": float(ret.std(ddof=1)),
        "var95": float(-q5), "var99": float(-q1), "cvar95": float(-ret[ret <= q5].mean()), "cvar99": float(-ret[ret <= q1].mean()),
        "worst": float(ret.min()), "best": float(ret.max()), "p_negative": float((ret < 0).mean()),
        "skew": float(np.asarray(pd.Series(ret).skew(), dtype=np.float64)), "kurtosis": float(np.asarray(pd.Series(ret).kurt(), dtype=np.float64)),
    }


def margin_distance(res: StrategyResult, cfg: SimConfig, paths: MarketPaths, path: int = 0) -> F64:
    """Further fall in equity value (fraction of current equities) that would trigger a margin call, per step.

    A call happens when loan = h·(ℓ_E·E + ℓ_I·I + ℓ_cash·cash), i.e. after an equity decline
        d = 1 − (loan/h − ℓ_I·I − ℓ_cash·cash) / (ℓ_E·E)
    using the current loan, cash, illiquid marks and the stress multiplier h in force. d ≥ 1 means the loan
    is covered by cash/illiquids alone; NaN when there is no loan.
    """
    from fosim.instruments.lombard_loan import ltv_multiplier

    lev = cfg.leverage
    loan = res.series("loan")[path]
    eq = res.series("held_mv")[path] + res.series("spot_mv")[path]
    cash = np.maximum(res.series("cash")[path], 0.0)
    ill = res.series("illiquid_mv")[path]
    held = paths.held[path]
    dd_held = np.minimum(held / np.maximum.accumulate(held) - 1.0, 0.0)
    h = ltv_multiplier(lev, paths.iv_short[path], dd_held)
    with np.errstate(divide="ignore", invalid="ignore"):
        need = loan / h - lev.ltv_base.illiquid * ill - lev.ltv_base.cash * cash
        d = np.where(loan > 1e-6, 1.0 - need / np.maximum(lev.ltv_base.equity_index * eq, 1e-300), np.nan)
    return np.asarray(d, dtype=np.float64)
