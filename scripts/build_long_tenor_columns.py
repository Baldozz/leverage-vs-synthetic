"""Add 7- and 10-year Treasury yields and extrapolated 7y/10y ATM implied vols to data/market_data_{daily,weekly,monthly}.csv.

Yields: Bloomberg ``USGG10YR Index`` / ``USGG7YR Index`` (``PX_LAST``, daily) exported as one workbook per ticker
(``10yr for calls analysis.xlsx`` and ``7Y UST.xlsx``; see ``read_bbg_series`` for the two layouts recognised).
Before the first date of the 7-year series (Bloomberg's ``USGG7YR`` history starts on 2009-02-26) the 7-year yield is
interpolated linearly in maturity between the 5- and 10-year yields and flagged in ``ust_7y_interpolated``
(Assumptions 19n). Vols: same construction as ``iv_5y`` (Assumptions 19l) —
``iv_T = theta_T * (iv_24m_filled / theta_24m) ** (beta_T / beta_24m)`` with beta_T from the log-linear fit of the
observed tenor elasticities to VIX (3m … 24m, full history) and theta_T = theta_5y (flat mean level beyond 5 years,
PLACEHOLDER until a dealer / OVDV anchor is available). Fit parameters are written to data/vol_mapping_fit.json.

Run:  .venv/bin/python scripts/build_long_tenor_columns.py [--ust10 FILE] [--ust7 FILE]   (defaults: the two workbooks in the project root)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIT = DATA / "vol_mapping_fit.json"
FREQS = ("daily", "weekly", "monthly")
VOL_SERIES = {"vix3m": 0.25, "vix6m": 0.5, "vix1y": 1.0, "iv_12m": 1.0, "iv_18m": 1.5, "iv_24m": 2.0}
JOIN_TOLERANCE = pd.Timedelta(31, unit="D")  # daily gaps are ≤ 5 days; the last monthly row is a partial month carried from the last daily value


def read_bbg_series(path: Path, ticker: str, field: str = "PX_LAST") -> pd.Series:
    """One Bloomberg export → daily Series indexed by date, ascending. Two layouts are recognised: the BDH sheet ``Values``
    (row 0 = "Security", row 5 = header, data from row 6, Excel serial dates) and the two-column sheet ``Sheet1``
    (A1 = field, B1 = ticker, then date | value rows; rows without a value — the requested range bounds — are dropped)."""
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    if "Values" in sheets:
        raw = sheets["Values"]
        if str(raw.iat[0, 1]).strip() != ticker:
            raise ValueError(f"{path.name}: expected {ticker} in B1, found {raw.iat[0, 1]!r}")
        hdr = raw.iloc[5].tolist()
        if hdr[0] != "Date" or field not in hdr:
            raise ValueError(f"{path.name}: unexpected header row {hdr}")
        body = raw.iloc[6:, [0, hdr.index(field)]].dropna()
        dates = pd.to_datetime(pd.to_numeric(body.iloc[:, 0]), unit="D", origin="1899-12-30")
    elif "Sheet1" in sheets:
        raw = sheets["Sheet1"]
        if str(raw.iat[0, 0]).strip() != field or str(raw.iat[0, 1]).strip() != ticker:
            raise ValueError(f"{path.name}: expected {field} | {ticker} in row 1, found {raw.iloc[0].tolist()!r}")
        body = raw.iloc[1:, [0, 1]].dropna()
        dates = pd.to_datetime(body.iloc[:, 0])
    else:
        raise ValueError(f"{path.name}: no sheet 'Values' or 'Sheet1' (found {list(sheets)})")
    s = pd.Series(pd.to_numeric(body.iloc[:, 1]).to_numpy(), index=dates, name=ticker).sort_index()
    return s[~s.index.duplicated(keep="last")]


def asof_join(target_dates: pd.Series, s: pd.Series) -> np.ndarray:
    """Last value of ``s`` on or before each target date (within JOIN_TOLERANCE), else NaN."""
    left = pd.DataFrame({"date": pd.to_datetime(target_dates).to_numpy()})
    right = pd.DataFrame({"date": s.index.to_numpy(), "v": s.to_numpy()})
    out = pd.merge_asof(left, right, on="date", direction="backward", tolerance=JOIN_TOLERANCE)
    return out["v"].to_numpy(dtype=np.float64)


def fit_elasticities(daily: pd.DataFrame) -> dict[str, float]:
    """beta_T = slope of ln(iv_T) on ln(VIX) per tenor; beta(T) = a + b ln T fitted across tenors (Assumptions 19l)."""
    pts = []
    for col, tenor in VOL_SERIES.items():
        m = daily.dropna(subset=[col, "vix"])
        pts.append((tenor, float(np.polyfit(np.log(m["vix"].to_numpy()), np.log(m[col].to_numpy()), 1)[0])))
    lt = np.log([p[0] for p in pts])
    b, a = np.polyfit(lt, [p[1] for p in pts], 1)
    return {"beta_a": float(a), "beta_b": float(b), **{f"beta_{t}y_fit": float(a + b * np.log(t)) for t in (2, 5, 7, 10)}}


def main(ust10: Path, ust7: Path | None) -> None:
    fit = json.loads(FIT.read_text())
    daily = pd.read_csv(DATA / "market_data_daily.csv", parse_dates=["date"])
    el = fit_elasticities(daily)
    if abs(el["beta_5y_fit"] - fit["beta_5y"]) > 1e-3:
        raise RuntimeError(f"elasticity refit ({el['beta_5y_fit']:.4f}) does not reproduce the stored beta_5y ({fit['beta_5y']:.4f})")
    for t in (7, 10):
        fit[f"beta_{t}y"] = el[f"beta_{t}y_fit"]
        fit[f"theta_{t}y_placeholder"] = fit["theta_5y_placeholder"]  # flat mean level beyond 5y (PLACEHOLDER, Assumptions 19n)
    fit["beta_curve"] = {"a": el["beta_a"], "b": el["beta_b"], "form": "beta(T) = a + b ln T"}

    s10 = read_bbg_series(ust10, "USGG10YR Index")
    s7 = read_bbg_series(ust7, "USGG7YR Index") if ust7 is not None else None
    frames: dict[str, pd.DataFrame] = {f: pd.read_csv(DATA / f"market_data_{f}.csv", parse_dates=["date"]) for f in FREQS}
    for f, df in frames.items():
        df["ust_10y"] = asof_join(df["date"], s10)
        u7 = asof_join(df["date"], s7) if s7 is not None else np.full(len(df), np.nan)
        interp = df["ust_5y"].to_numpy() + (df["ust_10y"].to_numpy() - df["ust_5y"].to_numpy()) * (7.0 - 5.0) / (10.0 - 5.0)
        df["ust_7y_interpolated"] = np.isnan(u7)
        df["ust_7y"] = np.where(np.isnan(u7), interp, u7)
        for t in (7, 10):
            df[f"iv_{t}y"] = fit[f"theta_{t}y_placeholder"] * (df["iv_24m_filled"].to_numpy() / fit["theta_24m"]) ** (fit[f"beta_{t}y"] / fit["beta_24m"])
        for c in ("ust_10y", "ust_7y", "iv_7y", "iv_10y"):
            if df[c].isna().any():
                raise RuntimeError(f"{f}: {int(df[c].isna().sum())} missing values in {c}")
        df.to_csv(DATA / f"market_data_{f}.csv", index=False)
        print(f"{f:8s} {len(df):5d} rows  ust_10y {df.ust_10y.iloc[-1]:.3f}  ust_7y {df.ust_7y.iloc[-1]:.3f} ({int(df.ust_7y_interpolated.sum())} interpolated)  "
              f"iv_7y {df.iv_7y.iloc[-1]:.2f}  iv_10y {df.iv_10y.iloc[-1]:.2f}  (last {df.date.iloc[-1].date()})")
    FIT.write_text(json.dumps(fit, indent=1))
    print("fit:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in fit.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ust10", type=Path, default=ROOT / "10yr for calls analysis.xlsx")
    ap.add_argument("--ust7", type=Path, default=ROOT / "7Y UST.xlsx", help="USGG7YR export (daily from 2009-02-26); before its first date the 7y is interpolated between 5y and 10y")
    a = ap.parse_args()
    main(a.ust10, a.ust7)
