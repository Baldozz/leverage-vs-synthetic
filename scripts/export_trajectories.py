"""Export the trajectories of both portfolios from every start date to a long CSV (one row per start and sampling date).

    .venv/bin/python scripts/export_trajectories.py                      # weekly starts, month-end samples → reports/trajectories_weekly_monthly.csv
    .venv/bin/python scripts/export_trajectories.py --grid daily         # every trading-day start (≈ 5 min, ≈ 2 M rows)
    .venv/bin/python scripts/export_trajectories.py --sample W           # week-end samples instead of month-ends

Setup options mirror the app's sidebar (defaults = the app's defaults). Columns: start, date, then NAV, dry powder and the balance sheet of
both portfolios in USD m, and SPX with dividends rebased to 1 at the start.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fosim.analytics.leverage_stress import (  # noqa: E402
    DAILY_FILE,
    PATH_COLUMNS,
    _load,
    paths_table,
    rolling_paths,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", choices=["weekly", "daily"], default="weekly", help="start dates: every Friday or every trading day")
    ap.add_argument("--sample", default="M", help="sampling period of the trajectories (pandas period alias: M month, W week)")
    ap.add_argument("--first", default="1997-09-09")
    ap.add_argument("--equity", type=float, default=1000.0, help="USD m")
    ap.add_argument("--loan", type=float, default=250.0, help="USD m")
    ap.add_argument("--spread", type=float, default=75.0, help="bp over the base rate")
    ap.add_argument("--lv", type=float, default=75.0, help="lending value of the SPX, %%")
    ap.add_argument("--lv-calls", type=float, default=0.0, help="lending value of the calls, %%")
    ap.add_argument("--tenor", type=float, default=5.0, help="years")
    ap.add_argument("--weeks", type=int, default=52, help="weekly steps of the rotation (1 = one shot)")
    ap.add_argument("--wht", type=float, default=15.0, help="dividend withholding tax, %%")
    ap.add_argument("--surplus", choices=["cash", "equity", "calls"], default="cash")
    ap.add_argument("--replace-worthless", action="store_true", help="replace a call that expires worthless (default: it lapses)")
    ap.add_argument("--out", default=None, help="output CSV (default reports/trajectories_<grid>_<sample>.csv)")
    a = ap.parse_args()

    d = _load(str(DAILY_FILE))
    last = d.date.iloc[-1]
    last_start = last - pd.DateOffset(months=round(a.tenor * 12)) - pd.Timedelta(7 * (a.weeks - 1), unit="D")
    if a.grid == "daily":
        starts = pd.DatetimeIndex(d.date[(d.date >= pd.Timestamp(a.first)) & (d.date <= last_start)])
    else:
        starts = pd.date_range(a.first, last_start, freq="W-FRI")
    out = Path(a.out) if a.out else ROOT / "reports" / f"trajectories_{a.grid}_{ {'M': 'monthly', 'W': 'weekly'}.get(a.sample, a.sample) }.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    print(f"{len(starts):,} start dates {starts[0].date()} → {starts[-1].date()}, held to {last.date()}, sampled every {a.sample} …", flush=True)
    _rows, paths = rolling_paths(starts, last, columns=PATH_COLUMNS, sample=a.sample, progress=lambda i, n: print(f"  {i:,}/{n:,}", end="\r", flush=True),
                                equity0=a.equity * 1e6, loan0=a.loan * 1e6, spread=a.spread / 1e4, ltv_equity=a.lv / 100, ltv_call=a.lv_calls / 100, tenor=a.tenor, wht=a.wht / 100,
                                surplus=a.surplus, build_tranches=a.weeks, replace_worthless=a.replace_worthless)
    tbl = paths_table(paths)
    tbl.to_csv(out, index=False)
    print(f"\n{len(tbl):,} rows × {len(tbl.columns)} columns, {out.stat().st_size / 1e6:.0f} MB, {time.perf_counter() - t:.0f}s → {out}")


if __name__ == "__main__":
    main()
