"""Fair comparison lenses (SPEC §5.7.9): equal capital (default), equal inception dollar-delta,
equal volatility and equal expected return (the last two solved by scaling B's delta fraction on CRN)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from fosim.analytics.metrics import return_stats, terminal_stats
from fosim.config.schema import SimConfig
from fosim.runner import RunOutput, run_config, with_overrides


def _cell(df: pd.DataFrame, row: str, col: str) -> float:
    return float(np.asarray(df.loc[row, col], dtype=np.float64))


def _stats(out: RunOutput) -> pd.DataFrame:
    rows = {}
    for name, res in out.results.items():
        ts = terminal_stats(res.nav, out.paths.grid.horizon_years, [5, 50, 95])
        rs = return_stats(res.nav, res.series("cash_rate"), out.paths.grid.dt)
        rows[name] = {"median_terminal_nav": ts.median, "mean_terminal_nav": ts.mean, "cagr_median": ts.cagr_median, "cagr_mean": ts.cagr_mean, "ann_vol": rs["ann_vol_mean"], "sharpe": rs["sharpe_mean"], "p5": ts.percentiles[5], "p95": ts.percentiles[95], "dollar_delta0": res.inception.get("dollar_delta0", np.nan)}
    return pd.DataFrame(rows)


def comparison_lenses(cfg: SimConfig, n_paths: int, lenses: list[str] | None = None, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    lenses = lenses or list(cfg.exposure.comparison_lenses)
    out: dict[str, Any] = {}
    base = run_config(cfg, n_paths=n_paths)
    out["equal_capital"] = {"table": _stats(base), "note": "B sized per the configured sizing mode; both start from the same NAV"}
    if progress:
        progress("equal_capital")
    if "equal_delta" in lenses:
        c = with_overrides(cfg, {"options.sizing_mode": "delta_match"})
        o = run_config(c, n_paths=n_paths)
        out["equal_delta"] = {"table": _stats(o), "note": "B notional = A exposure / Δ so that inception dollar-delta matches A"}
        if progress:
            progress("equal_delta")
    tab0: pd.DataFrame = out["equal_capital"]["table"]
    a_vol = _cell(tab0, "ann_vol", "A")
    a_cagr = _cell(tab0, "cagr_mean", "A")

    def run_frac(x: float) -> RunOutput:
        return run_config(with_overrides(cfg, {"options.sizing_mode": "delta_fraction_of_A", "options.sizing_param": float(x)}), n_paths=n_paths)

    for lens, target, key in (("equal_vol", a_vol, "ann_vol"), ("equal_return", a_cagr, "cagr_mean")):
        if lens not in lenses:
            continue
        try:

            def g(x: float, key: str = key, target: float = target) -> float:
                return _cell(_stats(run_frac(x)), key, "B") - target

            lo, hi = 0.25, 3.0
            if g(lo) * g(hi) > 0:
                out[lens] = {"table": None, "note": f"no delta fraction in [{lo}, {hi}] matches A's {key} ({target:.4f}); B cannot be scaled to A on this lens"}
            else:
                x = brentq(g, lo, hi, xtol=1e-3, maxiter=30)
                o = run_frac(x)
                out[lens] = {"table": _stats(o), "note": f"B delta fraction of A scaled to {x:.3f} so that B's {key} equals A's ({target:.4f})", "fraction": x}
        except Exception as e:
            out[lens] = {"table": None, "note": f"lens failed: {e}"}
        if progress:
            progress(lens)
    return out
