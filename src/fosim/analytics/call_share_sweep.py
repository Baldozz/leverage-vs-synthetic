"""Call-share sweep — scenario 3 of the five scenarios (SPX + a sleeve of rolling calls) with the sleeve's size swept (page 4 of
app/historical_app.py; docs/METHODOLOGY.md §10 "Call-share sweep", Assumptions 41).

The swept variable is the sleeve's size as the sidebar sets it: κ, the share of the capital spent on premium (``sizing="premium"``), or X,
the share of the capital carried in SPX-equivalent exposure (``sizing="exposure"``). Every other argument of ``simulate_scenarios`` is fixed.
A share of 0 buys no call at all, so scenario 3 at 0 % is the long portfolio (scenario 1) exactly; the grid of shares always starts there.
For each share the scenario's ``summary`` row (return, the risk measures, the flows, the holdings) is kept; a sweep over many start dates — the
last trading day of every calendar month up to the last start whose whole build has reached its first expiry on the end day — gives the
surface of a measure over (start, share) and its empirical bands across the starts. A driver over the page-3 engine: no accounting or pricing
of its own.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fosim.analytics.call_vs_cash import DAILY_FILE
from fosim.analytics.five_scenarios import SIZING, SUMMARY_ROWS, simulate_scenarios, summary
from fosim.analytics.leverage_stress import _load

Progress = Callable[[int, int], None] | None
SWEPT = {"premium": "call_frac", "exposure": "exposure_frac"}   # the engine argument the sweep drives under each sizing
BAND_Q = (0.05, 0.5, 0.95)


def share_grid(lo: float, hi: float, step: float) -> tuple[float, ...]:
    """The shares lo, lo + step, …, hi (fractions, rounded to 1e-10 so that 0.15 is 0.15), with 0 prepended when absent: the long portfolio
    is always the first point of a sweep."""
    if step <= 0.0 or lo < 0.0 or hi < lo:
        raise ValueError("share_grid needs 0 <= lo <= hi and step > 0")
    out = [float(v) for v in np.round(np.arange(lo, hi + step / 2.0, step), 10)]
    if not out or out[0] > 0.0:
        out.insert(0, 0.0)
    return tuple(out)


def default_shares(sizing: str) -> tuple[float, ...]:
    """The page's default grid: 5 → 50 % of the capital in premium in steps of 5 %, or 10 → 100 % in exposure in steps of 10 %."""
    if sizing not in SIZING:
        raise ValueError(f"sizing must be one of {SIZING}")
    return share_grid(0.05, 0.50, 0.05) if sizing == "premium" else share_grid(0.10, 1.00, 0.10)


def sweep_call_share(start: str | pd.Timestamp, end: str | pd.Timestamp, shares: Sequence[float], sizing: str = "premium",
                     file: Path | str | None = None, progress: Progress = None, **kw: Any) -> pd.DataFrame:
    """Scenario 3 from ``start`` to ``end`` once per share: one row per share (the index, fractions), the columns ``SUMMARY_ROWS`` of
    ``five_scenarios.summary``. ``kw`` goes to ``simulate_scenarios`` and must not carry the swept argument. ``progress(i, n)`` after each run."""
    if sizing not in SIZING:
        raise ValueError(f"sizing must be one of {SIZING}")
    if any(k in kw for k in ("call_frac", "exposure_frac", "sizing", "scenarios")):
        raise ValueError("the swept share is set by `shares` and `sizing`: do not pass call_frac, exposure_frac, sizing or scenarios")
    sh = [float(x) for x in shares]
    if not sh or sh[0] < 0.0 or any(b <= a for a, b in pairwise(sh)):
        raise ValueError("shares must be non-negative and strictly increasing")
    if sizing == "premium" and sh[-1] >= 1.0:
        raise ValueError("a premium share must be below 100 % of the capital")
    rows: dict[float, pd.Series] = {}
    for i, x in enumerate(sh):
        run_kw: dict[str, Any] = {SWEPT[sizing]: x, **kw}
        paths, infos = simulate_scenarios(start, end, scenarios=("3",), sizing=sizing, file=file, **run_kw)
        rows[x] = summary(paths, infos)["3"]
        if progress is not None:
            progress(i + 1, len(sh))
    out = pd.DataFrame(rows).T.reindex(columns=list(SUMMARY_ROWS))
    out.index = pd.Index(sh, name="share")
    return out


def month_end_starts(end: str | pd.Timestamp, tenor: float, build_steps: int = 52, build_unit: str = "week", file: Path | str | None = None) -> pd.DatetimeIndex:
    """The last trading day of every calendar month of the daily file up to end − tenor − the build's length (12·tenor months, then
    7·(steps − 1) days or steps − 1 months): every slot of a start's build has reached its first expiry on the end day."""
    if build_unit not in ("week", "month") or int(build_steps) < 1:
        raise ValueError("build_unit must be 'week' or 'month', build_steps at least 1")
    cut = pd.Timestamp(end) - pd.DateOffset(months=round(12 * tenor))
    cut = cut - (pd.DateOffset(days=7 * (int(build_steps) - 1)) if build_unit == "week" else pd.DateOffset(months=int(build_steps) - 1))
    dates = pd.Series(pd.DatetimeIndex(_load(str(file or DAILY_FILE)).date))
    last = dates.groupby(dates.dt.to_period("M")).max()
    return pd.DatetimeIndex(last[last <= cut].to_numpy())


def sweep_grid(starts: Sequence[pd.Timestamp] | pd.DatetimeIndex, end: str | pd.Timestamp, shares: Sequence[float], sizing: str = "premium",
               file: Path | str | None = None, progress: Progress = None, **kw: Any) -> pd.DataFrame:
    """``sweep_call_share`` from every start: one long table, columns ``start``, ``share`` and ``SUMMARY_ROWS``; ``progress`` counts the
    (start, share) runs. A start that cannot run raises."""
    starts_l = [pd.Timestamp(s) for s in starts]
    if not starts_l:
        raise ValueError("no start dates")
    n_sh = len(list(shares))
    total = len(starts_l) * n_sh
    frames: list[pd.DataFrame] = []
    for j, s0 in enumerate(starts_l):
        def _p(i: int, _n: int, j: int = j) -> None:
            if progress is not None:
                progress(j * n_sh + i, total)

        f = sweep_call_share(s0, end, shares, sizing, file, _p, **kw).reset_index()
        f.insert(0, "start", s0)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def surface_table(long: pd.DataFrame, measure: str) -> pd.DataFrame:
    """``measure`` of a ``sweep_grid`` table as a surface: rows = shares (ascending), columns = starts (ascending), floats, NaN where a
    (start, share) is missing."""
    z = long.pivot(index="share", columns="start", values=measure).astype(float)
    return z.sort_index().sort_index(axis=1)


def grid_bands(long: pd.DataFrame, measure: str, q: tuple[float, float, float] = BAND_Q) -> pd.DataFrame:
    """Across the starts of a ``sweep_grid`` table, per share: the empirical quantiles ``q`` of ``measure`` (numpy's linear interpolation, as
    the risk measures of §10), the number of starts with a value, and ``ahead of 0 %`` — the share of the starts whose measure at this share
    exceeds their own value at share 0 (the long portfolio; NaN when the table has no 0 % row)."""
    z = surface_table(long, measure)
    base = z.loc[0.0].to_numpy(dtype=float) if 0.0 in z.index else None
    rows: dict[float, dict[str, float]] = {}
    nan = float("nan")
    for share, v in zip(z.index.to_numpy(dtype=float), z.to_numpy(dtype=float), strict=True):
        ok = np.isfinite(v)
        p_lo, p_mid, p_hi = (float(np.percentile(v[ok], 100.0 * x)) for x in q) if ok.any() else (nan, nan, nan)
        if base is None:
            ahead = nan
        else:
            pair = ok & np.isfinite(base)
            ahead = float(np.mean(v[pair] > base[pair])) if pair.any() else nan
        rows[float(share)] = {"p5": p_lo, "median": p_mid, "p95": p_hi, "n": float(ok.sum()), "ahead of 0 %": ahead}
    out = pd.DataFrame(rows).T.reindex(columns=["p5", "median", "p95", "n", "ahead of 0 %"])
    out.index.name = "share"
    out["n"] = out["n"].astype(int)
    return out
