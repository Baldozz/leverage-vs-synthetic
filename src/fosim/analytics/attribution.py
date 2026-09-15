"""Performance attribution (SPEC §6.3): full-revaluation P&L components and a Greek-based explain.

Full revaluation: the engine books every P&L source into ``COMPONENTS`` at each step; sums over
steps and paths are exact (they reconcile to NAV changes by the accounting identity).

Greek explain for the option book (per step, using the Greeks marked at the start of the step):
    explained = Δ$·x + ½·(Γ$/0.01)·x² + vega₁pt·Δσ(pts) + θ_year·dt + ρ₁₀₀bp·(Δr/0.01)
with x = ΔS/S of the option-book index, Δσ = change in the 5y ATM implied vol (the book's dominant
vol input) and Δr = change in the 5y zero rate. The unexplained residual (actual option MTM change −
explained) is shown explicitly and includes higher-order terms, skew/vol-surface effects, dividend
sensitivity and the vol difference across tranches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fosim.config.schema import SimConfig
from fosim.engine.state import COMP_IDX, COMPONENTS, StrategyResult
from fosim.market.paths import MarketPaths
from fosim.pricing.vol_surface import ParametricSurface


def component_totals(results: dict[str, StrategyResult]) -> pd.DataFrame:
    """Mean over paths of the horizon totals (steps 1..N) of each P&L component, by strategy (USD).

    Step-0 entries (transition costs, inception bid/ask, arrangement fees) are already inside NAV₀ and
    are reported separately as ``inception_costs_in_nav0``; ``total`` reconciles to NAV_T − NAV₀.
    """
    rows = {}
    for name, res in results.items():
        tot = res.recorder.components[:, 1:, :].sum(axis=1)
        rows[name] = {c: float(tot[:, i].mean()) for i, c in enumerate(COMPONENTS)}
        rows[name]["total"] = float(tot.sum(axis=1).mean())
        rows[name]["nav_change_check"] = float((res.nav[:, -1] - res.nav[:, 0]).mean())
        rows[name]["inception_costs_in_nav0"] = float(res.recorder.components[:, 0, :].sum(axis=1).mean())
    return pd.DataFrame(rows)


def component_percentiles(res: StrategyResult, q: tuple[float, ...] = (5, 25, 50, 75, 95)) -> pd.DataFrame:
    tot = res.recorder.components[:, 1:, :].sum(axis=1)
    return pd.DataFrame({c: {f"p{int(p)}": float(np.percentile(tot[:, i], p)) for p in q} for i, c in enumerate(COMPONENTS)}).T


def cumulative_mean_by_step(res: StrategyResult) -> pd.DataFrame:
    return pd.DataFrame(res.recorder.components.mean(axis=0).cumsum(axis=0), columns=list(COMPONENTS))


def greek_explain(res: StrategyResult, paths: MarketPaths, cfg: SimConfig) -> pd.DataFrame:
    """Mean over paths of the per-step Greek explain of the option book and the residual."""
    ps = ParametricSurface.from_config(cfg.implied_vol)
    T0 = cfg.options.tenor_years
    level = paths.book_index_level()
    x = level[:, 1:] / level[:, :-1] - 1.0
    dt = paths.grid.dt
    dd = res.series("dollar_delta")[:, :-1]
    dg = res.series("dollar_gamma")[:, :-1]
    vg = res.series("vega_1pt")[:, :-1]
    th = res.series("theta_year")[:, :-1]
    rh = res.series("rho_100bp")[:, :-1]
    atm = ps.atm_vol(T0, paths.iv_short)
    d_sig_pts = (atm[:, 1:] - atm[:, :-1]) * 100.0
    r5 = res.series("option_rate_5y")
    d_r = (r5[:, 1:] - r5[:, :-1]) / 0.01
    delta_pnl = dd * x
    gamma_pnl = 0.5 * (dg / 0.01) * x * x
    vega_pnl = vg * d_sig_pts
    theta_pnl = th * dt
    rho_pnl = rh * d_r
    explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl
    actual = res.recorder.components[:, 1:, COMP_IDX["option_mtm"]]
    resid = actual - explained
    df = pd.DataFrame(
        {
            "delta": delta_pnl.mean(axis=0), "gamma": gamma_pnl.mean(axis=0), "vega": vega_pnl.mean(axis=0), "theta": theta_pnl.mean(axis=0),
            "rho": rho_pnl.mean(axis=0), "explained": explained.mean(axis=0), "actual_option_mtm": actual.mean(axis=0), "unexplained_residual": resid.mean(axis=0),
        }
    )
    df.attrs["totals"] = {c: float(df[c].sum()) for c in df.columns}
    df.attrs["residual_share_of_abs_actual"] = float(np.abs(resid).sum() / max(np.abs(actual).sum(), 1e-300))
    return df


def realised_vs_implied_diagnostic(res: StrategyResult, paths: MarketPaths, cfg: SimConfig) -> dict[str, float]:
    """½·Γ·S²·(σ_realised² − σ_implied²)·dt summed over the horizon (mean over paths) — gamma/theta trade-off."""
    ps = ParametricSurface.from_config(cfg.implied_vol)
    T0 = cfg.options.tenor_years
    level = paths.book_index_level()
    x = np.log(level[:, 1:] / level[:, :-1])
    dt = paths.grid.dt
    dg = res.series("dollar_gamma")[:, :-1] / 0.01  # Γ n S²
    sig_impl = ps.atm_vol(T0, paths.iv_short)[:, :-1]
    real_var = x * x  # realised variance in the step (x² ≈ σ²dt)
    term = 0.5 * dg * (real_var - sig_impl**2 * dt)
    return {"gamma_vs_theta_pnl_mean": float(term.sum(axis=1).mean()), "realised_vol_mean": float(np.sqrt((x**2).mean() / dt)), "implied_5y_vol_mean": float(sig_impl.mean())}
