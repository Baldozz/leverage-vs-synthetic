"""Call-vs-cash backtest (docs/METHODOLOGY.md §8): payoff, notional, maturity matching, cash legs, tenor columns."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fosim.analytics.call_vs_cash import backtest, tenor_columns, total_return_index
from fosim.pricing.black_scholes import bsm_price


def _synthetic(tmp_path: Path, extra: dict[str, float] | None = None) -> Path:
    dates = pd.bdate_range("2010-01-04", "2015-12-31")
    n = len(dates)
    t = np.arange(n) / 252.0
    df = pd.DataFrame({
        "date": dates,
        "spxfp": 100.0 * np.exp(0.02 * t + 0.25 * np.sin(2 * np.pi * t / 4.0)),  # 4-year cycle: some 3-year windows end below the start
        "spx_px_last": 1000.0 * np.exp(0.07 * t),
        "spx_div_yld": 2.0,
        "ust_5y": 2.0 + np.sin(t),
        "iv_5y": 20.0 + 5 * np.cos(t),
        **(extra or {}),
    })
    p = tmp_path / "mkt.csv"
    df.to_csv(p, index=False)
    return p


def test_payoff_notional_and_maturity(tmp_path: Path) -> None:
    f = _synthetic(tmp_path)
    bt, info = backtest(3.0, 10e6, "2010-01-04", "2012-12-31", None, "SPXFP", 0.15, file=f)
    src = pd.read_csv(f, parse_dates=["date"]).set_index("date")
    assert info.rate_col == "ust_5y" and info.vol_col == "iv_5y" and info.fallback
    assert len(bt) == ((src.index >= "2010-01-04") & (src.index <= "2012-12-31")).sum()
    row = bt.iloc[100]
    t0 = row.strike_date
    r = src.loc[t0, "ust_5y"] / 100
    c = float(bsm_price(100.0, 100.0, r, r, src.loc[t0, "iv_5y"] / 100, 3.0)) / 100
    assert row.prem == pytest.approx(c, rel=1e-12)
    assert row.notional == pytest.approx(10e6 / c, rel=1e-12)
    # maturity = trading day nearest to t0 + 36 months
    target = t0 + pd.DateOffset(months=36)
    assert row.name == src.index[np.argmin(np.abs((src.index - target).days))]
    assert row.call_pnl == pytest.approx(row.notional * max(src.loc[row.name, "spxfp"] / src.loc[t0, "spxfp"] - 1, 0) - 10e6, rel=1e-12)
    assert row.cash_pnl == pytest.approx(10e6 * (src.loc[row.name, "spxfp"] / src.loc[t0, "spxfp"] - 1), rel=1e-12)
    # the call never loses more than the premium; some strike dates end worthless on this path
    assert (bt.call_pnl >= -10e6 - 1e-6).all()
    assert np.isclose(bt.call_pnl, -10e6).any() and (bt.call_pnl > 0).any()
    assert bt.index.is_monotonic_increasing


def test_fixed_premium_and_spx_total_return_leg(tmp_path: Path) -> None:
    f = _synthetic(tmp_path)
    bt, _ = backtest(2.0, 5e6, "2010-06-01", "2011-06-01", 0.12, "SPX_TR", 0.0, file=f)
    assert np.allclose(bt.prem, 0.12) and np.allclose(bt.notional, 5e6 / 0.12)
    src = pd.read_csv(f, parse_dates=["date"]).set_index("date")
    row = bt.iloc[0]
    # SPX with a 2 % yield reinvested gross: TR ratio = price ratio × exp(ln(1.02) × years)
    yrs = (row.name - row.strike_date).days / 365.0
    expected = 5e6 * (src.loc[row.name, "spx_px_last"] / src.loc[row.strike_date, "spx_px_last"] * np.exp(np.log1p(0.02) * yrs) - 1)
    assert row.cash_pnl == pytest.approx(expected, rel=1e-9)
    # withholding reduces the dividend contribution
    bt_w, _ = backtest(2.0, 5e6, "2010-06-01", "2011-06-01", 0.12, "SPX_TR", 0.5, file=f)
    assert (bt_w.cash_pnl < bt.cash_pnl).all()


def test_tenor_specific_columns_are_used_when_present(tmp_path: Path) -> None:
    f = _synthetic(tmp_path, {"ust_2y": 1.0, "iv_2y": 30.0})
    df = pd.read_csv(f)
    assert tenor_columns(df, 2.0) == ("ust_2y", "iv_2y")
    assert tenor_columns(df, 2.5) == ("ust_5y", "iv_5y")
    assert tenor_columns(df, 7.0) == ("ust_5y", "iv_5y")
    bt, info = backtest(2.0, 1e6, "2010-06-01", "2010-06-30", None, "SPXFP", 0.15, file=f)
    assert (info.rate_col, info.vol_col) == ("ust_2y", "iv_2y") and not info.fallback
    assert np.allclose(bt.prem, float(bsm_price(100.0, 100.0, 0.01, 0.01, 0.30, 2.0)) / 100)


def test_total_return_index_zero_yield_is_price() -> None:
    dates = pd.bdate_range("2020-01-01", periods=50).to_numpy()
    px = np.linspace(100, 120, 50)
    assert np.allclose(total_return_index(px, np.zeros(50), dates, 0.15), px)
    assert (total_return_index(px, np.full(50, 3.0), dates, 0.0)[1:] > px[1:]).all()


def test_real_data_smoke() -> None:
    bt, info = backtest(5.0, 10e6, "2006-05-01", "2021-08-31")
    assert len(bt) > 3500 and info.rate_col == "ust_5y" and not info.fallback  # 5y columns are the tenor's own
    _, info7 = backtest(7.0, 10e6, "2010-01-01", "2010-03-31")
    assert info7.fallback
    assert bt.call_pnl.min() == pytest.approx(-10e6)  # 2007 vintages expire worthless
    assert 0.10 < bt.prem.mean() < 0.20
    with pytest.raises(ValueError):
        backtest(5.0, 10e6, "2006-05-01", "2021-08-31", cash_leg="GOLD")
