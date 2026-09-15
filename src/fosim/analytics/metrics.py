"""Risk and return metrics with Monte Carlo standard errors (SPEC §6.1).

Conventions: terminal NAV in USD; per-path CAGR = (NAV_T/NAV_0)^{1/T} − 1 (median and mean of the
per-path CAGRs are reported separately from the CAGR of the mean NAV); annualised vol, Sharpe and
Sortino from step returns in excess of the cash rate (smoothed illiquid marks bias these downwards);
VaR/CVaR are losses expressed as returns (positive numbers = losses). Standard errors: mean → σ/√n;
quantiles → bootstrap (fixed seed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fosim.engine.state import StrategyResult
from fosim.market.paths import MarketPaths

F64 = NDArray[np.float64]


def _bootstrap_se(x: F64, fn: str, q: float | None = None, n_boot: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        s = x[rng.integers(0, n, n)]
        vals[b] = np.median(s) if fn == "median" else np.percentile(s, q)  # type: ignore[arg-type]
    return float(vals.std(ddof=1))


@dataclass(frozen=True)
class TerminalStats:
    mean: float
    se_mean: float
    median: float
    se_median: float
    percentiles: dict[float, float]
    cagr_median: float
    cagr_mean: float
    cagr_of_mean_nav: float
    n: int


def terminal_stats(nav: F64, T: float, percentiles: list[float]) -> TerminalStats:
    nav_T, nav_0 = nav[:, -1], nav[:, 0]
    n = len(nav_T)
    with np.errstate(invalid="ignore", divide="ignore"):
        cagr = np.where(nav_T > 0, (np.maximum(nav_T, 1e-300) / nav_0) ** (1.0 / T) - 1.0, -1.0)
    return TerminalStats(
        mean=float(nav_T.mean()), se_mean=float(nav_T.std(ddof=1) / np.sqrt(n)), median=float(np.median(nav_T)),
        se_median=_bootstrap_se(nav_T, "median"), percentiles={p: float(np.percentile(nav_T, p)) for p in percentiles},
        cagr_median=float(np.median(cagr)), cagr_mean=float(cagr.mean()), cagr_of_mean_nav=float((nav_T.mean() / nav_0.mean()) ** (1.0 / T) - 1.0), n=n,
    )


def step_returns(nav: F64) -> F64:
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(nav[:, :-1] > 0, nav[:, 1:] / nav[:, :-1] - 1.0, 0.0)
    return np.asarray(r, dtype=np.float64)


def return_stats(nav: F64, cash_rate: F64, dt: float) -> dict[str, float]:
    """Annualised vol, Sharpe, Sortino from step returns in excess of the cash rate (per path, then averaged)."""
    r = step_returns(nav)
    ex = r - cash_rate[:, :-1] * dt
    vol = r.std(axis=1, ddof=1) * np.sqrt(1.0 / dt)
    mean_ex = ex.mean(axis=1) / dt
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe = np.where(vol > 0, mean_ex / vol, np.nan)
        down = np.sqrt((np.minimum(ex, 0.0) ** 2).mean(axis=1) / dt)
        sortino = np.where(down > 0, mean_ex / down, np.nan)
    n = nav.shape[0]
    return {
        "ann_vol_mean": float(np.nanmean(vol)), "ann_vol_se": float(np.nanstd(vol, ddof=1) / np.sqrt(n)),
        "sharpe_mean": float(np.nanmean(sharpe)), "sharpe_se": float(np.nanstd(sharpe, ddof=1) / np.sqrt(n)),
        "sortino_mean": float(np.nanmean(sortino)), "sortino_se": float(np.nanstd(sortino, ddof=1) / np.sqrt(n)),
        "pooled_ann_vol": float(r.std(ddof=1) * np.sqrt(1.0 / dt)),
    }


def drawdown_stats(nav: F64, dt: float) -> dict[str, object]:
    peak = np.maximum.accumulate(nav, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.where(peak > 0, nav / peak - 1.0, -1.0)
    mdd = dd.min(axis=1)
    under = (dd < -1e-12).mean(axis=1)
    # time to recovery: from the trough of the max drawdown to the first later step at a new peak
    P = nav.shape[0]
    trough = dd.argmin(axis=1)
    ttr = np.full(P, np.nan)
    for p in range(P):
        k = trough[p]
        after = np.flatnonzero(dd[p, k:] >= -1e-12)
        if after.size and mdd[p] < -1e-12:
            ttr[p] = after[0] * dt
    n = P
    return {
        "max_dd_mean": float(mdd.mean()), "max_dd_median": float(np.median(mdd)), "max_dd_p5": float(np.percentile(mdd, 5)),
        "max_dd_p1": float(np.percentile(mdd, 1)), "max_dd_se": float(mdd.std(ddof=1) / np.sqrt(n)),
        "time_under_water_mean": float(under.mean()), "time_to_recovery_median_years": float(np.nanmedian(ttr)) if np.isfinite(ttr).any() else float("nan"),
        "share_never_recovered": float(np.isnan(ttr[mdd < -1e-12]).mean()) if (mdd < -1e-12).any() else 0.0,
        "max_dd_per_path": mdd,
    }


def var_cvar(ret: F64, level: float) -> tuple[float, float]:
    """VaR and CVaR at ``level`` as losses (positive numbers) from a sample of returns."""
    loss = -ret
    var = float(np.percentile(loss, level * 100))
    tail = loss[loss >= var]
    return var, float(tail.mean()) if tail.size else var


def var_cvar_stats(nav: F64, steps_per_year: int, levels: list[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    n1 = min(steps_per_year, nav.shape[1] - 1)
    r1 = nav[:, n1] / nav[:, 0] - 1.0
    rH = nav[:, -1] / nav[:, 0] - 1.0
    for lv in levels:
        v, c = var_cvar(r1, lv)
        out[f"var_{int(lv * 100)}_1y"], out[f"cvar_{int(lv * 100)}_1y"] = v, c
        v, c = var_cvar(rH, lv)
        out[f"var_{int(lv * 100)}_horizon"], out[f"cvar_{int(lv * 100)}_horizon"] = v, c
    return out


def leverage_risk(res: StrategyResult) -> dict[str, float]:
    ev = res.events
    n = len(ev.margin_calls)
    p_call = float((ev.margin_calls > 0).mean())
    p_liq = float((ev.forced_liquidations > 0).mean())
    p_ruin = float(ev.ruin.mean())
    return {
        "p_margin_call": p_call, "p_margin_call_se": float(np.sqrt(p_call * (1 - p_call) / n)),
        "p_forced_liquidation": p_liq, "p_forced_liquidation_se": float(np.sqrt(p_liq * (1 - p_liq) / n)),
        "p_ruin": p_ruin, "p_ruin_se": float(np.sqrt(p_ruin * (1 - p_ruin) / n)),
        "p_margin_warning": float((ev.margin_warnings > 0).mean()),
        "forced_sale_volume_mean": float(ev.forced_sale_volume.mean()), "forced_sale_volume_p95": float(np.percentile(ev.forced_sale_volume, 95)),
        "cure_sale_volume_mean": float(ev.cure_sales_volume.mean()),
        "slippage_loss_mean": float(ev.slippage_loss.mean()), "slippage_loss_p95": float(np.percentile(ev.slippage_loss, 95)),
        "min_headroom_median": float(np.median(ev.min_headroom[np.isfinite(ev.min_headroom)])) if np.isfinite(ev.min_headroom).any() else float("nan"),
        "min_headroom_p5": float(np.percentile(ev.min_headroom[np.isfinite(ev.min_headroom)], 5)) if np.isfinite(ev.min_headroom).any() else float("nan"),
        "p_liquidity_shortfall": float((ev.liquidity_shortfalls > 0).mean()),
        "futures_margin_calls_mean": float(ev.futures_margin_calls.mean()),
    }


def dry_powder_stats(res: StrategyResult, paths: MarketPaths, cfg_spending_pct: float, capital_call_months: int) -> dict[str, float]:
    """Cash at the book-index trough, deployment statistics and liquidity coverage."""
    dd = paths.book_drawdown()
    trough = dd.argmin(axis=1)
    idx = np.arange(paths.n_paths)
    cash = res.series("cash")
    nav = res.nav
    cash_at_trough = cash[idx, trough]
    nav_at_trough = nav[idx, trough]
    ev = res.events
    spot = res.series("spot_mv")
    basis = res.series("spot_cost_basis") if "spot_cost_basis" in res.recorder.series else np.zeros_like(spot)
    with np.errstate(invalid="ignore", divide="ignore"):
        roi = np.where(basis[:, -1] > 0, spot[:, -1] / basis[:, -1] - 1.0, np.nan)
    # liquidity coverage ratio = cash / (unfunded due in 12m + 12m spending), per step, reported at min over time
    il = paths.illiquids
    n_steps = round(capital_call_months / 12 * paths.grid.steps_per_year)
    due = np.zeros_like(cash)
    from fosim.engine.conventions import per_step_rate

    for j in range(il.unfunded.shape[2]):
        rc = 0.25  # placeholder pacing when the fund config is unavailable here; unfunded fully due gives a conservative floor
        due += il.unfunded[:, :, j] * (1 - (1 - per_step_rate(rc, paths.grid.dt)) ** n_steps)
    denom = due + cfg_spending_pct * nav * capital_call_months / 12
    with np.errstate(invalid="ignore", divide="ignore"):
        lcr = np.where(denom > 0, cash / denom, np.inf)
    return {
        "cash_at_trough_mean": float(cash_at_trough.mean()), "cash_at_trough_pct_nav_mean": float(np.nanmean(cash_at_trough / nav_at_trough)),
        "cash_at_trough_p5": float(np.percentile(cash_at_trough, 5)),
        "p_deployed": float((ev.dry_powder_deployments > 0).mean()), "deployed_usd_mean": float(ev.dry_powder_deployed_usd.mean()),
        "deployed_usd_mean_given_deployed": float(ev.dry_powder_deployed_usd[ev.dry_powder_deployments > 0].mean()) if (ev.dry_powder_deployments > 0).any() else 0.0,
        "roi_on_deployed_median": float(np.nanmedian(roi)) if np.isfinite(roi).any() else float("nan"),
        "lcr_min_median": float(np.median(np.min(lcr, axis=1))), "lcr_inception": float(np.median(lcr[:, 0])),
        "p_cash_constrained_purchase": float((ev.cash_constrained_purchases > 0).mean()),
    }


def compare_strategies(res_b: StrategyResult, res_a: StrategyResult, percentiles: list[float]) -> dict[str, float]:
    d = res_b.nav[:, -1] - res_a.nav[:, -1]
    n = len(d)
    p = float((d > 0).mean())
    out = {"p_B_beats_A": p, "p_B_beats_A_se": float(np.sqrt(p * (1 - p) / n)), "diff_mean": float(d.mean()), "diff_se": float(d.std(ddof=1) / np.sqrt(n)), "diff_median": float(np.median(d))}
    for q in percentiles:
        out[f"diff_p{int(q)}"] = float(np.percentile(d, q))
    return out


def crra_certainty_equivalent(W: F64, gamma: float) -> dict[str, float]:
    """CE = (E[W^{1−γ}])^{1/(1−γ)}; γ = 1 → exp(E[ln W]). Paths with W ≤ 0 are reported and excluded."""
    bad = W <= 0
    Wp = W[~bad]
    if Wp.size == 0:
        return {"ce": float("nan"), "n_nonpositive": int(bad.sum()), "gamma": gamma}
    if abs(gamma - 1.0) < 1e-12:
        ce = float(np.exp(np.log(Wp).mean()))
        u = np.log(Wp)
    else:
        u = Wp ** (1 - gamma) / (1 - gamma)
        ce = float(((1 - gamma) * u.mean()) ** (1 / (1 - gamma)))
    # delta-method SE on CE
    se_u = float(u.std(ddof=1) / np.sqrt(Wp.size))
    dce = ce if abs(gamma - 1.0) < 1e-12 else ce ** gamma
    return {"ce": ce, "ce_se": float(abs(dce) * se_u), "n_nonpositive": int(bad.sum()), "gamma": gamma, "ce_pct_nav0": float("nan")}


def summary_table(results: dict[str, StrategyResult], paths: MarketPaths, percentiles: list[float], var_levels: list[float], gamma: float, spending_pct: float, capital_call_months: int) -> pd.DataFrame:
    T = paths.grid.horizon_years
    rows: dict[str, dict[str, float]] = {}
    for name, res in results.items():
        ts = terminal_stats(res.nav, T, percentiles)
        rs = return_stats(res.nav, res.series("cash_rate"), paths.grid.dt)
        ds = drawdown_stats(res.nav, paths.grid.dt)
        vc = var_cvar_stats(res.nav, paths.grid.steps_per_year, var_levels)
        lr = leverage_risk(res)
        ce = crra_certainty_equivalent(res.nav[:, -1], gamma)
        row: dict[str, float] = {
            "terminal_nav_mean": ts.mean, "terminal_nav_mean_se": ts.se_mean, "terminal_nav_median": ts.median, "terminal_nav_median_se": ts.se_median,
            "cagr_median": ts.cagr_median, "cagr_mean": ts.cagr_mean, "cagr_of_mean_nav": ts.cagr_of_mean_nav,
            **{f"terminal_nav_p{int(p)}": v for p, v in ts.percentiles.items()},
            "ann_vol": rs["ann_vol_mean"], "sharpe": rs["sharpe_mean"], "sortino": rs["sortino_mean"],
            "max_dd_mean": ds["max_dd_mean"], "max_dd_median": ds["max_dd_median"], "max_dd_p5": ds["max_dd_p5"], "max_dd_p1": ds["max_dd_p1"],  # type: ignore[dict-item]
            "time_under_water": ds["time_under_water_mean"],  # type: ignore[dict-item]
            **vc, **{k: v for k, v in lr.items()}, "crra_ce": ce["ce"], "crra_ce_se": ce.get("ce_se", float("nan")), "n_nonpositive_terminal": ce["n_nonpositive"],
            "transition_cost": res.transition_cost,
        }
        if name == "B" or res.meta.get("has_dry_powder"):
            row.update(dry_powder_stats(res, paths, spending_pct, capital_call_months))
        rows[name] = row
    df = pd.DataFrame(rows)
    if "A" in results and "B" in results:
        cmp_ = compare_strategies(results["B"], results["A"], percentiles)
        for k, v in cmp_.items():
            df.loc[k, "B"] = v
    return df
