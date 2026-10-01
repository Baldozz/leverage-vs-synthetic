"""The call-share sweep of scenario 3 (docs/METHODOLOGY.md §10 "Call-share sweep", Assumptions 41): a scenario run alone equals its column
in the full run; a 0 % share is the long portfolio exactly; the sweep's rows are the engine's summary rows at the shares asked; the month-end
grid of starts and its cut; the surface and the bands across starts on a tiny grid."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fosim.analytics.call_share_sweep import (
    default_shares,
    grid_bands,
    month_end_starts,
    share_grid,
    surface_table,
    sweep_call_share,
    sweep_grid,
)
from fosim.analytics.five_scenarios import SCENARIOS, SUMMARY_ROWS, simulate_scenarios, summary

M = 1e6
C = 650 * M
NUMERIC = ("NAV today", "total return", "annualised return", "lowest NAV", "max drawdown", "returns counted", "volatility", "VaR 95 %", "CVaR 99 %",
           "Sharpe ratio", "premiums paid", "payoffs received", "tax paid", "SPX today", "calls today", "exposure today")


def _file(tmp_path: Path, spx: np.ndarray, base: float = 3.0, tbill: float = 2.0, div: float = 0.0, iv: float = 20.0, rate: float = 3.0, start: str = "2010-01-04") -> Path:
    dates = pd.bdate_range(start, periods=len(spx))
    df = pd.DataFrame({"date": dates, "spxfp": spx, "spx_px_last": spx, "spx_div_yld": div, "loan_base": base, "tbill_3m": tbill, "ust_5y": rate, "iv_5y": iv})
    p = tmp_path / "daily.csv"
    df.to_csv(p, index=False)
    return p


def _same(a: pd.Series, b: pd.Series) -> None:
    """Two summary columns agree on every numeric fact (NaN against NaN counts as equal) and on the economic dates (not on the build's
    calendar: scenario 3 at 0 % still reports when its empty build ended, scenario 1 has no build)."""
    for row in NUMERIC:
        x, y = float(a[row]), float(b[row])
        assert (np.isnan(x) and np.isnan(y)) or x == y, row
    for row in ("lowest NAV on", "max drawdown on", "wiped out on"):
        assert (pd.isna(a[row]) and pd.isna(b[row])) or a[row] == b[row], row


def test_scenarios_subset(tmp_path: Path) -> None:
    """``scenarios=("3",)`` runs scenario 3 alone, bit-identical to its column in the full run; the default still runs five; a wrong id or an empty
    tuple is refused."""
    spx = 100.0 * np.exp(np.linspace(0.0, 0.3, 600))
    f = _file(tmp_path, spx)
    full, full_i = simulate_scenarios("2010-01-04", "2012-04-20", tenor=1.0, build_steps=4, file=f)
    only, only_i = simulate_scenarios("2010-01-04", "2012-04-20", tenor=1.0, build_steps=4, file=f, scenarios=("3",))
    assert list(full) == list(SCENARIOS) and list(only) == ["3"] and list(only_i) == ["3"]
    pd.testing.assert_frame_equal(only["3"], full["3"], check_exact=True)
    assert only_i["3"].expiries == full_i["3"].expiries and only_i["3"].premium_build == full_i["3"].premium_build and only_i["3"].exposure_build == full_i["3"].exposure_build
    for bad in (("9",), (), ("3", "x")):
        with pytest.raises(ValueError):
            simulate_scenarios("2010-01-04", "2012-04-20", tenor=1.0, file=f, scenarios=bad)


def test_zero_share_is_scenario_1_and_the_shares_are_what_was_asked(tmp_path: Path) -> None:
    """The sweep's rows are the engine's summary of scenario 3 at each share, in the order asked; at 0 % no call is bought and the row is the long
    portfolio's (scenario 1) to the bit, under premium and under exposure sizing; the premium paid in the build is the share of the capital."""
    spx = 100.0 * np.exp(np.linspace(0.0, 0.3, 600))
    f = _file(tmp_path, spx)
    seen: list[tuple[int, int]] = []
    df = sweep_call_share("2010-01-04", "2012-04-20", (0.0, 0.1, 0.25), file=f, tenor=1.0, build_steps=1, progress=lambda i, n: seen.append((i, n)))
    assert list(df.index) == [0.0, 0.1, 0.25] and df.index.name == "share" and list(df.columns) == list(SUMMARY_ROWS) and seen == [(1, 3), (2, 3), (3, 3)]
    sm = summary(*simulate_scenarios("2010-01-04", "2012-04-20", tenor=1.0, build_steps=1, call_frac=0.25, file=f))
    _same(df.loc[0.0], sm["1"])
    _same(df.loc[0.25], sm["3"])
    assert df.loc[0.25, "premium paid in the build"] == pytest.approx(0.25 * C, rel=1e-12) and df.loc[0.0, "premium paid in the build"] == 0.0 and df.loc[0.0, "calls expired"] == 0
    sm10 = summary(*simulate_scenarios("2010-01-04", "2012-04-20", tenor=1.0, build_steps=1, call_frac=0.10, file=f, scenarios=("3",)))
    _same(df.loc[0.1], sm10["3"])
    # exposure sizing: the swept share is the exposure, 0 % still the long portfolio
    g = _file(tmp_path, np.full(300, 100.0), rate=4.0)
    dx = sweep_call_share("2010-01-04", "2011-02-25", (0.0, 0.5), sizing="exposure", file=g, tenor=1.0, build_steps=1)
    smx = summary(*simulate_scenarios("2010-01-04", "2011-02-25", tenor=1.0, build_steps=1, sizing="exposure", exposure_frac=0.5, file=g))
    _same(dx.loc[0.0], smx["1"])
    _same(dx.loc[0.5], smx["3"])
    assert dx.loc[0.0, "volatility"] == 0.0 and np.isnan(dx.loc[0.0, "Sharpe ratio"]) and dx.loc[0.5, "volatility"] > 0.0 and np.isfinite(dx.loc[0.5, "Sharpe ratio"])   # a flat SPX: no risk without calls
    # the grid of shares: the anchor always first, hi always included, no float dust
    assert share_grid(0.05, 0.50, 0.05) == (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5) == default_shares("premium")
    assert share_grid(0.10, 1.00, 0.10) == tuple(np.round(np.arange(0.0, 1.05, 0.1), 10)) == default_shares("exposure") and len(default_shares("exposure")) == 11
    assert share_grid(0.0, 0.2, 0.1) == (0.0, 0.1, 0.2) and share_grid(0.3, 0.3, 0.1) == (0.0, 0.3)
    for lo, hi, step in ((0.1, 0.5, 0.0), (0.5, 0.1, 0.1), (-0.1, 0.5, 0.1)):
        with pytest.raises(ValueError):
            share_grid(lo, hi, step)
    for kw in ({"shares": (0.0, 1.0)}, {"shares": (0.2, 0.1)}, {"shares": (0.1, 0.1)}, {"shares": ()}, {"shares": (0.0, 0.1), "call_frac": 0.3},
               {"shares": (0.0, 0.1), "sizing": "exposure", "rebalance": "proceeds"}, {"shares": (0.0, 0.1), "sizing": "none"}):
        with pytest.raises(ValueError):
            sweep_call_share("2010-01-04", "2012-04-20", file=f, tenor=1.0, build_steps=1, **kw)   # type: ignore[arg-type]
    with pytest.raises(ValueError):
        default_shares("none")


def test_grid_on_three_month_ends(tmp_path: Path) -> None:
    """The month-end grid: the last trading day of each calendar month up to end − tenor − the build; the long table equals the per-start sweeps;
    the surface is shares × starts; the bands are the empirical percentiles across the starts and the share of starts ahead of the long portfolio."""
    rng = np.random.default_rng(3)
    spx = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 330)))
    f = _file(tmp_path, spx)
    d = pd.read_csv(f, parse_dates=["date"])
    end = d["date"].iloc[-1]
    starts = month_end_starts(end, 1.0, 1, "week", file=f)
    expected = d["date"].groupby(d["date"].dt.to_period("M")).max()
    expected = expected[expected <= end - pd.DateOffset(months=12)]
    assert list(starts) == list(expected) and len(starts) == 3 and [s.strftime("%Y-%m") for s in starts] == ["2010-01", "2010-02", "2010-03"]
    assert len(month_end_starts(end, 1.0, 5, "week", file=f)) == 2 and len(month_end_starts(end, 1.0, 2, "month", file=f)) == 2   # the build shortens the window
    with pytest.raises(ValueError):
        month_end_starts(end, 1.0, 0, "week", file=f)
    seen: list[tuple[int, int]] = []
    long = sweep_grid(starts, end, (0.0, 0.25), file=f, tenor=1.0, build_steps=1, progress=lambda i, n: seen.append((i, n)))
    assert list(long.columns) == ["start", "share", *SUMMARY_ROWS] and len(long) == 6 and seen[-1] == (6, 6) and len(seen) == 6
    for s0 in starts:
        one = sweep_call_share(s0, end, (0.0, 0.25), file=f, tenor=1.0, build_steps=1)
        for x in (0.0, 0.25):
            _same(long[(long["start"] == s0) & (long["share"] == x)].iloc[0], one.loc[x])
    z = surface_table(long, "annualised return")
    assert z.shape == (2, 3) and list(z.index) == [0.0, 0.25] and list(z.columns) == list(starts) and np.isfinite(z.to_numpy()).all()
    b = grid_bands(long, "annualised return")
    v = z.loc[0.25].to_numpy()
    assert list(b.columns) == ["p5", "median", "p95", "n", "ahead of 0 %"] and list(b.index) == [0.0, 0.25] and list(b["n"]) == [3, 3]
    assert b.loc[0.25, "median"] == np.percentile(v, 50) and b.loc[0.25, "p5"] == np.percentile(v, 5) and b.loc[0.25, "p95"] == np.percentile(v, 95)
    assert b.loc[0.25, "ahead of 0 %"] == float(np.mean(v > z.loc[0.0].to_numpy())) and b.loc[0.0, "ahead of 0 %"] == 0.0
    with pytest.raises(ValueError):
        sweep_grid([], end, (0.0, 0.25), file=f)
