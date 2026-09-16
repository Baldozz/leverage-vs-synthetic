"""Call-vs-cash backtest (historical app, tab "Call vs cash backtest").

For every trading day t₀ in a strike-date range an ATM call on SPXFP maturing at t₀ + tenor is bought for a fixed
USD premium P (notional N = P / c(t₀), c = premium as a fraction of notional), and the same P is alternatively
invested in the index. Both are read at maturity T (the trading day nearest to the calendar date t₀ + tenor):

    call P&L = N · max(SPXFP_T / SPXFP_{t₀} − 1, 0) − P
    cash P&L = P · (I_T / I_{t₀} − 1),   I = SPXFP or SPX with dividends reinvested net of withholding

c(t₀) is the Black–Scholes ATM price at the implied vol of the day for the tenor (column ``iv_{n}y``) discounted at the
Treasury of the same tenor (``ust_{n}y``), with q = r because SPXFP is an excess-return index; when the tenor-specific
column is missing the 5-year column is used and reported. A fixed c can be given instead. See docs/METHODOLOGY.md §8.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fosim.config import PROJECT_ROOT
from fosim.pricing.black_scholes import bsm_price

DAILY_FILE = PROJECT_ROOT / "data" / "market_data_daily.csv"
REQUIRED = ("date", "spxfp", "spx_px_last", "spx_div_yld", "ust_5y", "iv_5y")


@dataclass(frozen=True)
class BacktestInfo:
    tenor: float
    rate_col: str
    vol_col: str
    n_strikes: int
    first_date: pd.Timestamp
    last_date: pd.Timestamp

    @property
    def fallback(self) -> bool:
        """True when the tenor has no series of its own and the 5-year columns stood in for it."""
        return abs(self.tenor - 5.0) > 1e-9 and (self.rate_col == "ust_5y" or self.vol_col == "iv_5y")


def tenor_columns(df: pd.DataFrame, tenor: float) -> tuple[str, str]:
    """(rate column, vol column) for the tenor: ``ust_{n}y`` / ``iv_{n}y`` when present, else the 5-year ones."""
    n = round(tenor)
    rate = f"ust_{n}y" if abs(tenor - n) < 1e-9 and f"ust_{n}y" in df.columns else "ust_5y"
    vol = f"iv_{n}y" if abs(tenor - n) < 1e-9 and f"iv_{n}y" in df.columns else "iv_5y"
    return rate, vol


def total_return_index(price: np.ndarray, div_yield_pct: np.ndarray, dates: np.ndarray, wht: float) -> np.ndarray:
    """Price index with the trailing dividend yield (percent, annual) reinvested continuously net of withholding."""
    dt = np.diff(dates.astype("datetime64[D]")).astype(float) / 365.0
    q_c = np.log1p(div_yield_pct / 100.0) * (1.0 - wht)
    out: np.ndarray = price * np.exp(np.concatenate([[0.0], np.cumsum(q_c[:-1] * dt)]))
    return out


def strike_date_range(tenor: float, file: Path | str | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last trading day usable as a strike date at this tenor: the maturity must still fall inside the data."""
    if tenor <= 0:
        raise ValueError("tenor must be positive")
    d = pd.read_csv(file or DAILY_FILE, parse_dates=["date"])
    rate_col, vol_col = tenor_columns(d, tenor)
    dates = d.dropna(subset=["spxfp", "spx_px_last", "spx_div_yld", rate_col, vol_col]).date.sort_values().reset_index(drop=True)
    usable = dates[dates + pd.DateOffset(months=round(tenor * 12)) <= dates.iloc[-1]]
    if usable.empty:
        raise ValueError(f"no strike date reaches a {tenor:g}-year maturity inside the data (ends {dates.iloc[-1].date()})")
    return dates.iloc[0], usable.iloc[-1]


def backtest(
    tenor: float, premium_usd: float, start: str, end: str, prem_fixed: float | None = None,
    cash_leg: str = "SPXFP", wht: float = 0.15, file: Path | str | None = None,
) -> tuple[pd.DataFrame, BacktestInfo]:
    """Rows indexed by maturity date; columns strike_date, prem, notional, spxfp, spxfp_T, call_pnl, cash_pnl (USD)."""
    if tenor <= 0 or premium_usd <= 0:
        raise ValueError("tenor and premium must be positive")
    if cash_leg not in ("SPXFP", "SPX_TR"):
        raise ValueError(f"cash_leg must be 'SPXFP' or 'SPX_TR', got {cash_leg!r}")
    d = pd.read_csv(file or DAILY_FILE, parse_dates=["date"])
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        raise ValueError(f"market data file lacks columns {missing}")
    rate_col, vol_col = tenor_columns(d, tenor)
    d = d.dropna(subset=["spxfp", "spx_px_last", "spx_div_yld", rate_col, vol_col]).sort_values("date").reset_index(drop=True)
    r = d[rate_col].to_numpy(dtype=np.float64) / 100.0
    if prem_fixed is not None:
        prem = np.full(len(d), float(prem_fixed))
    else:
        one = np.full(len(d), 100.0)
        prem = np.asarray(bsm_price(one, one, r, r, d[vol_col].to_numpy(dtype=np.float64) / 100.0, tenor), dtype=np.float64) / 100.0
    tr = total_return_index(d.spx_px_last.to_numpy(dtype=np.float64), d.spx_div_yld.to_numpy(dtype=np.float64), d.date.to_numpy(), wht)
    d = d.assign(prem=prem, cash_idx=d.spxfp.to_numpy() if cash_leg == "SPXFP" else tr)
    strikes = d[(d.date >= pd.Timestamp(start)) & (d.date <= pd.Timestamp(end))].copy()
    strikes["maturity_target"] = strikes.date + pd.DateOffset(months=round(tenor * 12))
    strikes = strikes[strikes.maturity_target <= d.date.iloc[-1]]
    mat = pd.merge_asof(
        strikes.sort_values("maturity_target"),
        d[["date", "spxfp", "cash_idx"]].rename(columns={"date": "maturity", "spxfp": "spxfp_T", "cash_idx": "cash_T"}),
        left_on="maturity_target", right_on="maturity", direction="nearest",
    )
    mat["notional"] = premium_usd / mat.prem
    mat["call_pnl"] = mat.notional * np.maximum(mat.spxfp_T / mat.spxfp - 1.0, 0.0) - premium_usd
    mat["cash_pnl"] = premium_usd * (mat.cash_T / mat.cash_idx - 1.0)
    out = mat.rename(columns={"date": "strike_date"})[["strike_date", "maturity", "prem", "notional", "spxfp", "spxfp_T", "call_pnl", "cash_pnl"]].set_index("maturity")
    info = BacktestInfo(tenor=float(tenor), rate_col=rate_col, vol_col=vol_col, n_strikes=len(out), first_date=d.date.iloc[0], last_date=d.date.iloc[-1])
    return out, info
