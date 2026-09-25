"""Keep the loan vs. rotate into calls (docs/METHODOLOGY.md §9): loan accrual, rotation ladder, margin-call trigger, rolls, accounting identity, rolling starts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fosim.analytics.leverage_stress import (
    DAILY_FILE,
    PATH_COLUMNS,
    PATH_NAMES,
    AccountingIdentityError,
    _load,
    corrections,
    paths_table,
    rolling_paths,
    rolling_starts,
    simulate,
    starts_table,
    trades_table,
)
from fosim.pricing.black_scholes import bsm_greeks, bsm_price

M = 1e6
TOL_USD = 1e-6


def _file(tmp_path: Path, spx: np.ndarray, spxfp: np.ndarray | None = None, base: float = 3.0, tbill: float = 2.0, div: float = 0.0, iv: float = 20.0, rate: float = 3.0, start: str = "2010-01-04", name: str = "daily.csv") -> Path:
    dates = pd.bdate_range(start, periods=len(spx))
    df = pd.DataFrame({"date": dates, "spxfp": spx if spxfp is None else spxfp, "spx_px_last": spx, "spx_div_yld": div, "loan_base": base, "tbill_3m": tbill, "ust_5y": rate, "iv_5y": iv})
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_loan_accrues_act360_and_capitalises(tmp_path: Path) -> None:
    n = 400
    f = _file(tmp_path, np.full(n, 1000.0), base=3.0)
    p, _ = simulate("2010-01-04", "2011-07-15", loan0=250 * M, spread=0.0075, build_tranches=1, file=f)
    dt = np.diff(p.index.to_numpy().astype("datetime64[D]")).astype(float) / 360.0
    expected = 250 * M * np.cumprod(1.0 + 0.0375 * dt)
    assert np.allclose(p["loan"].to_numpy()[1:], expected)
    assert np.allclose(p["interest_A"].to_numpy()[1:], np.diff(p["loan"].to_numpy()))
    assert np.allclose(p["lending_value_A"], 0.75 * p["E_A"]) and np.allclose(p["ltv_A"], p["loan"] / (0.75 * p["E_A"]))
    assert p["ltv_A"].iloc[0] == pytest.approx(1 / 3) and p["ltv_A"].iloc[-1] > 1 / 3 and p["E_A"].iloc[0] == 1000 * M


def test_rotation_repays_the_loan_and_starts_unlevered(tmp_path: Path) -> None:
    n = 300
    f = _file(tmp_path, np.full(n, 1000.0), base=0.0, tbill=0.0, rate=0.0, iv=20.0)
    p, info = simulate("2010-01-04", "2011-02-25", equity0=1000 * M, loan0=250 * M, spread=0.0, build_tranches=1, file=f)
    c0 = float(bsm_price(100.0, 100.0, 0.0, 0.0, 0.20, 5.0)) / 100.0
    dm = float(np.asarray(bsm_greeks(100.0, 100.0, 0.0, 0.0, 0.20, 5.0).delta))   # the ATM call's own delta sizes the sleeve
    assert info.premium0 == pytest.approx(c0) and info.delta0 == pytest.approx(dm) and info.rotation == pytest.approx(250 * M / (1 - c0 / dm))
    assert info.notional0 == pytest.approx(info.rotation / dm) and p["call_notional"].iloc[0] == pytest.approx(info.notional0)
    assert p["E_B"].iloc[0] == pytest.approx(1000 * M - info.rotation) and p["cash_B"].iloc[0] == 0.0 and p["call_val"].iloc[0] == pytest.approx(c0 * info.notional0)
    assert p["nav_B"].iloc[0] == pytest.approx(p["nav_A"].iloc[0]) == pytest.approx(750 * M)   # the rotation is NAV-neutral on day one
    assert p["exposure_replaced_B"].iloc[0] == pytest.approx(1000 * M)   # delta exposure: equity kept + delta × notional = the exposure before
    tau0 = (pd.Timestamp("2015-01-04") - pd.Timestamp("2010-01-04")).days / 365   # the live delta runs on the calendar days to the expiry day (1,826 / 365 years), the marks' convention
    d_live = float(np.asarray(bsm_greeks(1000.0, 1000.0, 0.0, 0.0, 0.20, tau0).delta))
    assert p["exposure_B"].iloc[0] == pytest.approx(1000 * M - info.rotation + d_live * info.notional0) and abs(p["exposure_B"].iloc[0] / (1000 * M) - 1) < 5e-5
    assert np.allclose(p["exposure_A"], p["E_A"])
    p5, i5 = simulate("2010-01-04", "2011-02-25", equity0=1000 * M, loan0=250 * M, spread=0.0, delta=0.5, build_tranches=1, file=f)   # override for scripted use
    assert i5.delta0 == 0.5 and i5.rotation == pytest.approx(250 * M / (1 - 2 * c0)) and p5["exposure_replaced_B"].iloc[0] == pytest.approx(1000 * M)
    assert p5["exposure_B"].iloc[0] == pytest.approx(1000 * M - i5.rotation + d_live * i5.notional0)   # the live exposure uses the model delta, whatever sized the sleeve
    with pytest.raises(ValueError):
        simulate("2010-01-04", "2011-02-25", delta=1.5, build_tranches=1, file=f)
    # flat market, zero rates: A's NAV is flat, B's only loses the call's time value
    assert np.allclose(p["nav_A"], 750 * M) and (p["nav_B"].diff().dropna() <= 1e-6).all() and p["nav_B"].iloc[-1] < p["nav_B"].iloc[0]
    assert np.allclose(p["cap_B"], 0.75 * p["E_B"])


def test_margin_call_date_matches_closed_form(tmp_path: Path) -> None:
    n = 500
    spx = 1000.0 * (1.0 - 0.8 * np.arange(n) / (n - 1))   # linear fall to −80 %
    f = _file(tmp_path, spx, base=0.0, tbill=0.0)
    for m_call, thr in ((0.9, 1 - 250 / (0.9 * 750)), (1.0, 2 / 3)):   # the bank calls at 90 % of the lending value (default) or at the whole of it
        p, _ = simulate("2010-01-04", "2011-12-01", spread=0.0, ltv_equity=0.75, margin_call=m_call, build_tranches=1, file=f)
        decline = 1.0 - p["E_A"] / p["E_A"].iloc[0]
        first_call = p.index[(p["headroom_A"] < 0).to_numpy().argmax()]
        # closed form: the loan (frozen at 250 with zero rates) meets m · 0.75 · E when E has fallen by 1 − 250/(m · 750)
        assert decline.loc[first_call] > thr and decline.shift(1).loc[first_call] <= thr
        assert p.loc[first_call, "ltv_A"] > m_call and p["ltv_A"].shift(1).loc[first_call] <= m_call   # LTV = loan / lending value crosses the call level
        assert np.allclose(p["headroom_A"], m_call * 0.75 * p["E_A"] - p["loan"]) and np.allclose(p["ltv_A"], p["loan"] / (0.75 * p["E_A"]))


def test_roll_cash_flows_rising_and_falling(tmp_path: Path) -> None:
    n = 6 * 262
    up = 1000.0 * np.exp(0.08 * np.arange(n) / 262)
    f = _file(tmp_path, up, base=0.0, tbill=0.0, rate=0.0, iv=20.0)
    p, info = simulate("2010-01-04", "2015-12-31", build_tranches=1, roll="units", file=f)   # the units rule
    assert len(info.rolls) == 1
    r = info.rolls[0]
    k = p.index.get_loc(r.date)
    S0, ST = p["spxfp"].iloc[0], p["spxfp"].iloc[k]
    assert r.payoff == pytest.approx(info.units0 * (ST - S0)) and r.equity_sold == 0.0
    assert r.premium_paid == pytest.approx(r.premium_frac * info.units0 * ST)      # constant units: notional = units × S_T
    assert info.units0 == pytest.approx(info.notional0 / S0)
    assert p["call_val"].iloc[k] == pytest.approx(r.premium_paid) and p["cash_B"].iloc[k] == pytest.approx(r.payoff - r.premium_paid)
    assert p["call_notional"].iloc[k] == pytest.approx(info.units0 * ST)
    down = 1000.0 * np.exp(-0.05 * np.arange(n) / 262)
    fd = _file(tmp_path, down, base=0.0, tbill=0.0, rate=0.0, name="down.csv")
    p2, info2 = simulate("2010-01-04", "2015-12-31", build_tranches=1, replace_worthless=False, roll="delta", file=fd)   # option: a worthless call lapses, nothing bought, no SPX sold (delta rule: no band trades after it)
    r2 = info2.rolls[0]
    k2 = p2.index.get_loc(r2.date)
    assert r2.payoff == 0.0 and r2.premium_paid == 0.0 and r2.equity_sold == 0.0 and p2["cash_B"].iloc[k2] == 0.0
    assert (p2["n_calls"].iloc[:k2] == 1).all() and (p2["n_calls"].iloc[k2:] == 0).all() and (p["n_calls"] == 1).all()   # lapsed: no call from the expiry day; rolled: one call throughout
    assert (p2["call_notional"].iloc[k2:] == 0.0).all() and (p2["call_val"].iloc[k2:] == 0.0).all() and p2["E_B"].iloc[k2] == pytest.approx(p2["E_B"].iloc[k2 - 1] * down[k2] / down[k2 - 1])
    assert np.allclose(p2["exposure_B"].iloc[k2:], p2["E_B"].iloc[k2:])   # only the SPX held is exposed after the lapse
    p3, info3 = simulate("2010-01-04", "2015-12-31", build_tranches=1, roll="delta", file=fd)   # delta rule: a worthless call is replaced on the same index units, paid by selling SPX delta-for-delta
    r3 = info3.rolls[0]
    dm = float(np.asarray(bsm_greeks(100.0, 100.0, 0.0, 0.0, 0.20, 5.0).delta))   # zero rates, 20 % vol: the ATM delta on every day of the synthetic file
    n3 = info3.units0 * down[k2]
    assert r3.payoff == 0.0 and r3.premium_paid == pytest.approx(r3.premium_frac * n3) and p3["call_notional"].iloc[k2] == pytest.approx(n3)
    assert r3.equity_sold == pytest.approx(dm * n3) and p3["cash_B"].iloc[k2] == pytest.approx((dm - r3.premium_frac) * n3)   # SPX sold for the calls' delta, the excess to T-bills
    assert p3["E_B"].iloc[k2] == pytest.approx(p2["E_B"].iloc[k2] - dm * n3) and p3["nav_B"].iloc[k2] == pytest.approx(p2["nav_B"].iloc[k2])   # NAV-neutral: stock swapped for calls + T-bills
    # the delta rule on the rising path: the in-the-money call (intrinsic, delta 1) is replaced by units × S_T / δ of ATM notional, the same dollar delta
    pdl, idl = simulate("2010-01-04", "2015-12-31", build_tranches=1, roll="delta", file=f)
    rd = idl.rolls[0]
    nd = idl.units0 * ST / dm
    assert rd.payoff == pytest.approx(r.payoff) and pdl["call_notional"].iloc[k] == pytest.approx(nd) and rd.premium_paid == pytest.approx(rd.premium_frac * nd)
    assert rd.equity_sold == 0.0 and pdl["cash_B"].iloc[k] == pytest.approx(rd.payoff - rd.premium_paid) and rd.payoff > rd.premium_paid   # +8 %/yr for 6 years: the payoff covers the doubled premium
    assert pdl["exposure_replaced_B"].iloc[k] == pytest.approx(pdl["E_B"].iloc[k] + idl.units0 * ST) and pdl["nav_B"].iloc[k] == pytest.approx(p["nav_B"].iloc[k])   # exposure carried on; NAV-neutral
    assert pdl["exposure_B"].iloc[k] == pytest.approx(pdl["E_B"].iloc[k] + idl.units0 * ST, rel=1e-4)   # the live delta of the new call (calendar days to its expiry) is a hair off the tenor's
    # a smaller rise: the payoff does not cover the premium, the shortfall is funded by selling SPX delta-for-delta on the share the SPX pays for
    up2 = 1000.0 * np.exp(0.02 * np.arange(n) / 262)
    f2 = _file(tmp_path, up2, base=0.0, tbill=0.0, rate=0.0, iv=20.0, name="up2.csv")
    ps, i_s = simulate("2010-01-04", "2015-12-31", build_tranches=1, roll="delta", file=f2)
    rsh = i_s.rolls[0]
    ks = ps.index.get_loc(rsh.date)
    ns = i_s.units0 * ps["spxfp"].iloc[ks] / dm
    short = rsh.premium_paid - rsh.payoff
    assert short > 0 and rsh.premium_paid == pytest.approx(rsh.premium_frac * ns) and rsh.equity_sold == pytest.approx(short / rsh.premium_paid * dm * ns)
    assert ps["cash_B"].iloc[ks] == pytest.approx(rsh.equity_sold - short) and ps["E_B"].iloc[ks] == pytest.approx(ps["E_B"].iloc[ks - 1] * up2[ks] / up2[ks - 1] - rsh.equity_sold)
    p4, _ = simulate("2010-01-04", "2015-12-31", surplus="equity", build_tranches=1, roll="delta", file=f)
    assert p4["cash_B"].iloc[k] == 0.0 and p4["E_B"].iloc[k] > p["E_B"].iloc[k]
    p5, info5 = simulate("2010-01-04", "2015-12-31", surplus="calls", build_tranches=1, roll="delta", file=f)
    r5 = info5.rolls[0]
    assert p5["cash_B"].iloc[k] == 0.0 and r5.premium_paid == pytest.approx(r5.payoff)   # the whole payoff goes into calls
    assert p5["call_notional"].iloc[k] == pytest.approx(r5.payoff / r5.premium_frac) and p5["call_val"].iloc[k] == pytest.approx(r5.payoff)
    assert p5["nav_B"].iloc[k] == pytest.approx(p["nav_B"].iloc[k])   # reinvestment is NAV-neutral on the roll day


def test_cash_buffer_is_kept_after_the_rotation(tmp_path: Path) -> None:
    f = _file(tmp_path, np.full(300, 1000.0), base=0.0, tbill=0.0, rate=0.0)
    p, info = simulate("2010-01-04", "2011-02-25", loan0=250 * M, cash_buffer=100 * M, build_tranches=1, file=f)
    assert info.rotation == pytest.approx(350 * M / (1 - info.premium0 / info.delta0))
    assert p["cash_B"].iloc[0] == 100 * M and p["dry_powder_B"].iloc[0] == pytest.approx(0.9 * (0.75 * p["E_B"].iloc[0] + 0.9 * 100 * M))   # the T-bills count at their lending value (90 % default); the call comes at 90 % of the lending value
    p_full, _ = simulate("2010-01-04", "2011-02-25", loan0=250 * M, cash_buffer=100 * M, build_tranches=1, ltv_cash=1.0, margin_call=1.0, file=f)
    assert p_full["dry_powder_B"].iloc[0] == pytest.approx(0.75 * p["E_B"].iloc[0] + 100 * M) and p_full["nav_B"].iloc[0] == pytest.approx(p["nav_B"].iloc[0])   # 100 % = the cash itself, the call at the whole lending value; the NAV does not depend on it
    assert p["nav_B"].iloc[0] == pytest.approx(750 * M)   # still NAV-neutral on day one
    with pytest.raises(ValueError):
        simulate("2010-01-04", "2011-02-25", cash_buffer=-1.0, build_tranches=1, file=f)


def test_accounting_identity_columns(tmp_path: Path) -> None:
    n = 6 * 262
    rng = np.random.default_rng(3)
    spx = 1000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
    f = _file(tmp_path, spx, spxfp=spx * 0.98, div=2.0)
    p, _ = simulate("2010-01-04", "2015-12-31", build_tranches=1, file=f)
    gap_a = p["nav_A"].diff().to_numpy()[1:] - (p["eq_pnl_A"] - p["interest_A"]).to_numpy()[1:]
    gap_b = p["nav_B"].diff().to_numpy()[1:] - (p["eq_pnl_B"] + p["call_pnl_B"] + p["cash_int_B"] - p["unwind_cost_B"]).to_numpy()[1:]
    assert np.abs(gap_a).max() < 1e-5 and np.abs(gap_b).max() < 1e-5
    assert np.allclose(p["unwind_cost_cum_B"], p["unwind_cost_B"].cumsum())
    assert issubclass(AccountingIdentityError, AssertionError)


def test_rolling_start_row_equals_single_path(tmp_path: Path) -> None:
    n = 8 * 262
    rng = np.random.default_rng(7)
    spx = 1000.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, n)))
    f = _file(tmp_path, spx)
    starts = pd.DatetimeIndex(["2010-06-01", "2011-06-01", "2016-06-01"])
    rs = rolling_starts(starts, 5.0, build_tranches=1, file=f)
    assert len(rs) == 2   # the 2016 start does not fit a 5-year window
    p, info = simulate("2010-06-01", pd.Timestamp("2010-06-01") + pd.DateOffset(months=60), build_tranches=1, file=f)
    row = rs.iloc[0]
    assert row["premium0"] == pytest.approx(info.premium0) and row["A: max LTV"] == pytest.approx(p["ltv_A"].max()) and row["NAV B end"] == pytest.approx(p["nav_B"].iloc[-1])
    assert row["trough"] == p["drawdown"].idxmin() and row["B: capacity at trough"] == pytest.approx(p.loc[row["trough"], "cap_B"])
    assert row["A return"] == pytest.approx(p["nav_A"].iloc[-1] / p["nav_A"].iloc[0] - 1) and row["B return"] == pytest.approx(p["nav_B"].iloc[-1] / p["nav_B"].iloc[0] - 1) and row["B: max LTV"] == 0.0
    assert row["B: min borrowing capacity"] == pytest.approx((p["cap_B"] - p["loan_B"]).min()) and row["years"] == pytest.approx((p.index[-1] - p.index[0]).days / 365.25)
    assert row["premiums paid"] == pytest.approx(info.premium0 * info.notional0 + sum(r.premium_paid for r in info.rolls)) and row["payoffs received"] == pytest.approx(sum(r.payoff for r in info.rolls))
    assert row["interest paid A"] == pytest.approx(p["interest_A"].sum()) and row["interest paid B"] == pytest.approx(p["interest_B"].sum()) == 0.0   # one-shot rotation: the loan is gone on day 0
    assert row["end: equity B"] + row["end: calls B"] + row["end: cash B"] - row["end: loan B"] == pytest.approx(row["NAV B end"])
    seen: list[tuple[int, int]] = []
    rolling_starts(starts, 5.0, build_tranches=1, file=f, progress=lambda i, n: seen.append((i, n)))
    assert seen and seen[-1] == (3, 3)
    rs2 = rolling_starts(starts, 5.0, build_tranches=1, replace_worthless=False, roll="delta", file=f, until="2017-12-29")   # every start runs to the same end date; the lapse rule (delta rule: no band trades), to count lapses
    assert len(rs2) == 3 and (rs2["end"] == rs2["end"].iloc[0]).all() and rs2["rolls"].tolist() == [1, 1, 0]
    assert (rs2["build end"] == rs2.index).all()
    pu, iu = simulate("2010-06-01", "2017-12-29", build_tranches=1, replace_worthless=False, roll="delta", file=f)   # the first row against its own single path (its one call expired worthless on this random path)
    lapsed = [r for r in iu.rolls if r.payoff == 0.0 and r.premium_paid == 0.0]
    assert rs2["B: lapsed"].iloc[0] == len(lapsed) == 1 and rs2["B: calls lost on"].iloc[0] == lapsed[0].date and rs2["B: call notional end"].iloc[0] == 0.0 == pu["call_notional"].iloc[-1]
    assert rs2["B: lapsed"].iloc[2] == 0 and pd.isna(rs2["B: calls lost on"].iloc[2]) and rs2["B: call notional end"].iloc[2] > 0   # the 2016 start: no expiry before the end, calls still held
    fd = _file(tmp_path, 1000.0 * np.exp(-0.03 * np.arange(n) / 262), name="down.csv")   # falling path: the one call lapses → the option part is lost at its expiry
    rs3 = rolling_starts(starts[:1], 5.0, build_tranches=1, replace_worthless=False, roll="delta", file=fd, until="2017-12-29")
    pd_, i_ = simulate("2010-06-01", "2017-12-29", build_tranches=1, replace_worthless=False, roll="delta", file=fd)
    assert rs3["B: lapsed"].iloc[0] == 1 and rs3["B: calls lost on"].iloc[0] == i_.rolls[0].date and rs3["B: call notional end"].iloc[0] == 0.0 and (pd_["call_notional"].loc[i_.rolls[0].date:] == 0).all()
    assert rs2["B: min dry powder"].iloc[0] <= rs2["B: dry powder at trough"].iloc[0]


def test_rolling_paths_match_the_single_paths(tmp_path: Path) -> None:
    n = 8 * 262
    rng = np.random.default_rng(11)
    spx = 1000.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, n)))
    f = _file(tmp_path, spx)
    starts = pd.DatetimeIndex(["2010-06-01", "2012-03-15", "2017-12-26"])
    rows, paths = rolling_paths(starts, "2017-12-29", columns=("headroom_A", "dry_powder_B"), sample="M", build_tranches=1, file=f)
    rs = rolling_starts(starts, 5.0, build_tranches=1, file=f, until="2017-12-29")
    pd.testing.assert_frame_equal(rows, rs)   # the same summary rows
    assert set(paths) == {"headroom_A", "dry_powder_B"} and list(paths["headroom_A"].columns) == list(rows.index) == [starts[0], starts[1]]   # the 26 Dec 2017 start is less than a week before the end
    for c in ("headroom_A", "dry_powder_B"):
        w = paths[c]
        assert w.index.is_monotonic_increasing and w.index[-1] == pd.Timestamp("2017-12-29") and (w.index[:-1] == w.index[:-1] + pd.offsets.MonthEnd(0)).all()   # month-ends, then the last day
        for s in starts[:2]:
            p, _ = simulate(s, "2017-12-29", build_tranches=1, file=f)
            assert w[s].loc[: s - pd.Timedelta(1, unit="D")].isna().all() and w[s].iloc[-1] == p[c].iloc[-1]   # NaN before the start; the end value
            me = p[c].groupby(p.index.to_period("M")).last()   # the value on the last trading day of each month, keyed by the calendar month-end
            me.index = me.index.to_timestamp(how="end").normalize()
            assert np.allclose(w[s].loc[me.index[:-1]], me.iloc[:-1])
    assert rows.loc[starts[0], "A: headroom end"] == paths["headroom_A"][starts[0]].iloc[-1] and rows.loc[starts[0], "B: dry powder end"] == paths["dry_powder_B"][starts[0]].iloc[-1]
    e_rows, e_paths = rolling_paths(pd.DatetimeIndex(["2017-12-26"]), "2017-12-29", build_tranches=1, file=f)
    assert e_rows.empty and all(v.empty for v in e_paths.values())
    # a start whose 52-week rotation is still under way at the end day runs with the steps taken so far: 10 weeks → 10 tranches, loan partly repaid, identity intact
    p10, i10 = simulate("2017-10-20", "2017-12-29", build_tranches=52, file=f)
    assert len(i10.builds) == 10 and 0 < p10["loan_B"].iloc[-1] < 250 * M and p10["n_bought"].iloc[-1] == 10 and i10.build_end == pd.Timestamp("2017-12-22")
    assert p10["loan_B"].iloc[-1] == pytest.approx(p10["loan_B"].iloc[0] * 0.0 + 250 * M * (42 / 52) * (p10["loan_B"].iloc[-1] / (250 * M * 42 / 52)), rel=1e-12)   # 42 of 52 shares of the loan still outstanding (plus its interest)
    assert rolling_starts(pd.DatetimeIndex(["2017-10-20"]), 5.0, build_tranches=52, file=f, until="2017-12-29")["end: loan B"].iloc[0] == p10["loan_B"].iloc[-1]
    marks = (pd.Timestamp("2013-06-12"), pd.Timestamp("2009-01-05"))   # a trading day inside both paths, and one before every start (ignored)
    _, mp = rolling_paths(starts[:2], "2017-12-29", columns=("headroom_A",), mark_days=marks, build_tranches=1, file=f)
    w = mp["headroom_A"]
    assert marks[0] in w.index and marks[1] not in w.index and w.index.is_monotonic_increasing and not w.index.duplicated().any()
    for s in starts[:2]:
        p, _ = simulate(s, "2017-12-29", build_tranches=1, file=f)
        assert w.loc[marks[0], s] == p.loc[marks[0], "headroom_A"]   # the exact value on the marked day


def test_input_validation(tmp_path: Path) -> None:
    f = _file(tmp_path, np.full(50, 1000.0))
    for kw in ({"equity0": 0.0}, {"ltv_equity": 1.5}, {"ltv_cash": 1.5}, {"surplus": "gold"}, {"roll": "gold"}, {"tenor": 0.0}, {"build_tranches": 0},
               {"rebalance": "weekly"}, {"below": "gold"}, {"band": 1.0}, {"band": -0.1}, {"unwind_haircut": -0.01}, {"margin_call": 0.0}, {"margin_call": 1.5}):
        with pytest.raises(ValueError):
            simulate("2010-01-04", "2010-03-01", file=f, **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        simulate("2010-03-01", "2010-01-04", build_tranches=1, file=f)
    with pytest.raises(ValueError):   # loan too large to be repaid by rotating the equity
        simulate("2010-01-04", "2010-03-01", equity0=100 * M, loan0=95 * M, build_tranches=1, file=f)


def test_real_data_1997_path_and_rolling_starts() -> None:
    p, info = simulate("1997-09-09", "2026-09-14", build_tranches=1, replace_worthless=False, roll="delta")   # the lapse rule (delta rule: no band trades)
    assert 0.10 < info.premium0 < 0.25 and 0.40 < info.delta0 < 0.60 and 330 * M < info.rotation < 400 * M and info.notional0 == pytest.approx(info.rotation / info.delta0) and not info.fallback
    assert p["ltv_A"].iloc[0] == pytest.approx(0.25 / 0.75) and 0.60 < p["ltv_A"].max() < 0.75 and (p["headroom_A"] > 0).all()   # LTV = loan / (75 % × equity): never 100 %
    # the Sep-1997 call expires worthless in Sep-2002 and lapses (user's rule): one expiry, nothing bought, no calls afterwards
    assert len(info.rolls) == 1 and info.rolls[0].date.year == 2002 and info.rolls[0].payoff == 0.0 and info.rolls[0].premium_paid == 0.0 and (p["call_notional"].loc["2002-09-10":] == 0).all()
    _, info_r = simulate("1997-09-09", "2026-09-14", build_tranches=1, roll="delta")   # delta rule, always replace → five rolls, every premium paid; the 2002 one by selling SPX delta-for-delta
    assert len(info_r.rolls) == 5 and all(r.premium_paid > 0 for r in info_r.rolls) and info_r.rolls[0].payoff == 0.0 and info_r.rolls[0].equity_sold > info_r.rolls[0].premium_paid
    # the doubling ratchet meets its limit: the Sept-2012 replacement (worthless expiry after the 2007 doubling) cannot be funded even by selling every unit of SPX,
    # so it is cut to cash/c + SPX/δ, all the SPX goes, the excess over the premium to T-bills; the run continues with calls and T-bills only
    p_r, _ = simulate("1997-09-09", "2026-09-14", build_tranches=1, roll="delta")
    r12 = info_r.rolls[2]
    k12 = p_r.index.get_loc(r12.date)
    assert r12.date.year == 2012 and r12.cut and not any(r.cut for r in info_r.rolls[:2]) and p_r["E_B"].iloc[k12] == pytest.approx(0.0, abs=1.0)
    assert p_r["cash_B"].iloc[k12] == pytest.approx(p_r["cash_B"].iloc[k12 - 1] + p_r["cash_int_B"].iloc[k12] + r12.payoff - r12.premium_paid + r12.equity_sold, rel=1e-9)   # Δcash = bill interest + payoff − premium + SPX sold
    rs_r = rolling_starts(pd.DatetimeIndex(["1997-09-09"]), 0.0, build_tranches=1, roll="delta", until="2026-09-14")
    assert rs_r["B: cut rolls"].iloc[0] == sum(1 for r in info_r.rolls if r.cut) >= 1 and rs_r["B: lapsed"].iloc[0] == 0
    k = int(p["drawdown"].to_numpy().argmin())   # the deepest fall on the record (March 2009): B's capacity to borrow beats A's room
    assert p.index[k].year == 2009 and p["cap_B"].iloc[k] > p["headroom_A"].iloc[k] > 0
    rs = rolling_starts(pd.date_range("1997-10-01", "2021-09-01", freq="MS"), 5.0, build_tranches=1)
    assert len(rs) > 250 and not rs["A: margin call"].any() and rs["A: min headroom"].min() > 0 and rs["A: max LTV"].max() < 1.0
    _, info7 = simulate("2010-01-04", "2020-01-06", tenor=7.0, build_tranches=1)
    assert not info7.fallback and info7.rate_col == "ust_7y" and len(info7.rolls) == 1


def test_weekly_ladder_repays_the_loan_step_by_step(tmp_path: Path) -> None:
    """Four weekly tranches: each step sells (L/remaining)/(1 − c/δ) of equity, buys notional sold/δ, repays L/remaining; no debt after the last."""
    n = 6 * 262
    f = _file(tmp_path, np.full(n, 1000.0), base=0.0, tbill=0.0, rate=0.0, iv=20.0)
    p, info = simulate("2010-01-04", "2015-12-31", equity0=1000 * M, loan0=250 * M, spread=0.0, build_tranches=4, file=f)
    c = float(bsm_price(100.0, 100.0, 0.0, 0.0, 0.20, 5.0)) / 100.0
    dl = float(np.asarray(bsm_greeks(100.0, 100.0, 0.0, 0.0, 0.20, 5.0).delta))
    assert len(info.builds) == 4 and info.build_end == info.builds[-1].date and (info.builds[-1].date - info.start).days == 21
    step = (250 * M / 4) / (1 - c / dl)
    assert all(b.equity_sold == pytest.approx(step) and b.premium_paid == pytest.approx(c * step / dl) for b in info.builds)   # flat market, zero rates: identical steps
    assert info.rotation == pytest.approx(4 * step) and info.notional0 == pytest.approx(4 * step / dl)
    kb = [p.index.get_loc(b.date) for b in info.builds]
    assert np.allclose(p["loan_B"].iloc[kb], 250 * M * np.array([0.75, 0.5, 0.25, 0.0]))
    assert (p["loan_B"].iloc[kb[-1]:] == 0).all() and p["loan_B"].iloc[0] == pytest.approx(187.5 * M)
    assert p["nav_B"].iloc[0] == pytest.approx(750 * M) and abs(p["nav_B"].iloc[kb[-1]] - p["nav_A"].iloc[kb[-1]]) < 0.02 * 750 * M   # NAV-neutral steps; only time value lost
    assert p["exposure_replaced_B"].iloc[kb[-1]] == pytest.approx(1000 * M)   # delta exposure fully replaced after the build (at the purchase deltas)
    assert p["exposure_B"].iloc[kb[-1]] == pytest.approx(1000 * M, rel=2e-3) and p["exposure_B"].iloc[kb[-1]] != pytest.approx(1000 * M)   # live: the first tranches' delta has drifted with three weeks of time decay
    assert p["dry_powder_B"].iloc[0] == pytest.approx(0.9 * 0.75 * p["E_B"].iloc[0] - 187.5 * M)   # 90 % of the lending value − the loan left
    # each tranche rolls at its own expiry: four rolls a week apart five years later
    assert len(info.rolls) == 4 and (info.rolls[-1].date - info.rolls[0].date).days == 21
    gap_b = p["nav_B"].diff().to_numpy()[1:] - (p["eq_pnl_B"] + p["call_pnl_B"] + p["cash_int_B"] - p["interest_B"]).to_numpy()[1:]
    assert np.abs(gap_b).max() < 1e-5
    p1, i1 = simulate("2010-01-04", "2015-12-31", equity0=1000 * M, loan0=250 * M, spread=0.0, build_tranches=1, file=f)
    assert len(i1.builds) == 1 and i1.rotation == pytest.approx(250 * M / (1 - c / dl)) and (p1["loan_B"] == 0).all()


def test_weekly_ladder_real_data() -> None:
    p, info = simulate("1997-09-09", "2026-09-14", replace_worthless=False, roll="delta")   # 52 weekly tranches, the lapse rule (delta rule: no band trades)
    assert len(info.builds) == 52 and info.build_end.year == 1998 and (p["loan_B"].loc[info.build_end:] == 0).all()
    assert p["n_calls"].iloc[0] == 1 and p["n_calls"].loc[info.build_end] == 52 and (p["n_calls"].loc[info.build_end:"2002-09-01"] == 52).all() and (p["n_calls"].loc["2003-09-03":] == 0).all()   # 52 alive after the build, all lapsed by Sep-2003
    assert 0.0 < (p["loan_B"] / p["cap_B"]).max() < 0.5
    assert len(info.rolls) == 52 and all(r.payoff == 0.0 and r.premium_paid == 0.0 for r in info.rolls)   # all 52 tranches expire worthless in 2002–03 and lapse
    assert p["nav_B"].iloc[0] == pytest.approx(p["nav_A"].iloc[0])
    _, info_r = simulate("1997-09-09", "2026-09-14", roll="delta")
    assert len(info_r.rolls) >= 52 * 5   # delta rule, always replace: each tranche rolls every five years


def test_starts_table_is_the_rows_in_millions(tmp_path: Path) -> None:
    n = 8 * 262
    rng = np.random.default_rng(5)
    spx = 1000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    f = _file(tmp_path, spx)
    starts = pd.DatetimeIndex(["2010-06-01", "2011-06-01"])
    rs = rolling_starts(starts, 5.0, build_tranches=1, file=f, until="2017-12-29")
    tbl = starts_table(rs)
    assert len(tbl) == 2 and list(tbl["start"]) == [s.date() for s in starts] and (tbl["end"] == pd.Timestamp("2017-12-29").date()).all()
    r0, t0 = rs.iloc[0], tbl.iloc[0]
    assert t0["keep the loan: NAV today (m)"] == pytest.approx(r0["NAV A end"] / M, abs=0.05) and t0["rotate into calls: NAV today (m)"] == pytest.approx(r0["NAV B end"] / M, abs=0.05)
    assert t0["NAV at start (m)"] == pytest.approx(750.0, abs=0.05) and t0["keep the loan: annualised return"] == pytest.approx((1 + r0["A return"]) ** (1 / r0["years"]) - 1, abs=5e-5)
    assert t0["keep the loan: lowest dry powder (m)"] == pytest.approx(r0["A: min headroom"] / M, abs=0.05) and t0["rotate into calls: dry powder today (m)"] == pytest.approx(r0["B: dry powder end"] / M, abs=0.05)
    assert t0["calls expired"] == r0["rolls"] and t0["calls expired worthless"] == r0["B: lapsed"]
    assert t0["all calls gone on"] == ("" if pd.isna(r0["B: calls lost on"]) else r0["B: calls lost on"].date().isoformat())
    assert not tbl.isna().any().any()   # a clean CSV: no NaN cells
    csv = tbl.to_csv(index=False)
    back = pd.read_csv(pd.io.common.StringIO(csv))
    assert len(back) == 2 and list(back.columns) == list(tbl.columns)


def test_paths_table_is_the_long_form_of_the_paths(tmp_path: Path) -> None:
    n = 8 * 262
    rng = np.random.default_rng(9)
    spx = 1000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    f = _file(tmp_path, spx)
    starts = pd.DatetimeIndex(["2010-06-01", "2012-03-15"])
    _rows, paths = rolling_paths(starts, "2017-12-29", columns=PATH_COLUMNS, sample="M", build_tranches=1, file=f)
    tbl = paths_table(paths)
    assert list(tbl.columns) == ["start", "date"] + [PATH_NAMES[c] for c in PATH_COLUMNS]
    assert len(tbl) == int(paths["nav_A"].notna().sum().sum()) and not tbl.isna().any().any()   # one row per (start, date) the start was running, no NaN
    assert (tbl.groupby("start")["date"].min() >= pd.Series({s.date(): s.date() for s in starts})).all() and (tbl.groupby("start")["date"].max() == pd.Timestamp("2017-12-29").date()).all()
    s0 = starts[0]
    p, _ = simulate(s0, "2017-12-29", build_tranches=1, file=f)
    t0 = tbl[tbl["start"] == s0.date()].set_index("date")
    assert t0["keep the loan: NAV (m)"].iloc[-1] == pytest.approx(p["nav_A"].iloc[-1] / M, abs=0.005) and t0["SPX with dividends (start = 1)"].iloc[-1] == pytest.approx(p["spx_tr"].iloc[-1], abs=5e-6)
    me = p["dry_powder_B"].groupby(p.index.to_period("M")).last()
    me.index = me.index.to_timestamp(how="end").normalize().date
    assert np.allclose(t0["rotate into calls: dry powder (m)"].loc[me.index[:-1]], me.iloc[:-1] / M, atol=0.005)
    assert (tbl["rotate into calls: loan (m)"] == 0).all()   # one-shot rotation: no loan left on any sampled date
    assert paths_table({c: pd.DataFrame() for c in PATH_COLUMNS}).empty


def test_corrections_and_lowest_nav(tmp_path: Path) -> None:
    n = 900
    t = np.arange(n)
    spx = np.where(t < 300, 1000.0 + 0.5 * t, np.where(t < 500, 1150.0 - 1.5 * (t - 300), 850.0 + 2.0 * (t - 500)))   # peak 1150 at t=300, trough 850 (−26 %) at t=500, back above at t=650
    f = _file(tmp_path, spx, base=0.0, tbill=0.0, rate=0.0)
    dates = pd.bdate_range("2010-01-04", periods=n)
    c = corrections(0.20, file=f)
    label = f"{dates[300].year}–{dates[500].year}" if dates[300].year != dates[500].year else f"{dates[500].year}"
    assert len(c) == 1 and c.index[0] == label and c["peak"].iloc[0] == dates[300] and c["trough"].iloc[0] == dates[500]
    assert c["drawdown"].iloc[0] == pytest.approx(850 / 1150 - 1) and c["recovered"].iloc[0] == dates[650]
    assert corrections(0.30, file=f).empty
    with pytest.raises(ValueError):
        corrections(1.5, file=f)
    rs = rolling_starts(pd.DatetimeIndex([dates[0], dates[250]]), 1.0, build_tranches=1, file=f, until=dates[-1])
    p, _ = simulate(dates[250], dates[-1], build_tranches=1, file=f)
    assert rs["A: min NAV date"].iloc[1] == dates[500] and rs["A: min NAV"].iloc[1] == pytest.approx(p["nav_A"].min()) and rs["A: LTV at min NAV"].iloc[1] == pytest.approx(p["ltv_A"].loc[dates[500]])
    assert rs["B: min NAV"].iloc[1] == pytest.approx(p["nav_B"].min()) and rs["B: min NAV date"].iloc[1] == p["nav_B"].idxmin()
    real = corrections(0.20)
    assert list(real.index) == ["2000–2002", "2007–2009", "2020", "2022"] and [d.year for d in real["trough"]] == [2002, 2009, 2020, 2022] and (real["drawdown"] < -0.2).all()
    assert real["recovered"].notna().all() and real.loc["2007–2009", "recovered"].year in (2012, 2013)
    assert c["peak level"].iloc[0] == pytest.approx(1150.0) and c["trough level"].iloc[0] == pytest.approx(850.0)   # zero dividends: the total-return index is the price
    px = corrections(0.20, on="price")   # the SPX price index: the well-known peaks and bottoms
    assert [d.date().isoformat() for d in px["peak"]] == ["2000-03-24", "2007-10-09", "2020-02-19", "2022-01-03"]
    assert [d.date().isoformat() for d in px["trough"]] == ["2002-10-09", "2009-03-09", "2020-03-23", "2022-10-12"]
    assert px["trough level"].round(0).tolist() == [777.0, 677.0, 2237.0, 3577.0] and px["peak level"].round(0).tolist() == [1527.0, 1565.0, 3386.0, 4797.0]
    with pytest.raises(ValueError):
        corrections(0.2, on="futures")


def test_cumulative_cost_columns_match_the_summary() -> None:
    """interest_cum_A / interest_cum_B / premiums_cum_B / payoffs_cum_B are running totals that end on the sums reported per start (page 1, worst-trajectory block)."""
    p, info = simulate("2003-03-10", "2012-12-31", roll="delta")   # calls struck in the 2003 trough expire in the money in 2008: payoffs > 0 (delta rule: no purchases outside the rolls)
    assert p["interest_cum_A"].iloc[-1] == pytest.approx(p["interest_A"].sum()) and p["interest_cum_A"].iloc[0] == 0.0
    # the rotation's loan accrues while it is being repaid: the running total is flat once the build is over, and it is real money (the equity sold pays it)
    assert info.build_end is not None
    assert p["interest_cum_B"].iloc[-1] == pytest.approx(p["interest_B"].sum()) and p["interest_cum_B"].iloc[0] == 0.0 and 0.0 < p["interest_cum_B"].iloc[-1] < p["interest_cum_A"].iloc[-1]
    assert p["interest_cum_B"].iloc[-1] == pytest.approx(p.loc[: info.build_end, "interest_B"].sum()) and (p.loc[info.build_end :, "interest_B"].iloc[1:] == 0.0).all()
    sold = sum(b.equity_sold for b in info.builds)   # over the build: SPX sold = loan repaid (250 m + its interest) + premiums
    assert sold == pytest.approx(250e6 + p["interest_cum_B"].iloc[-1] + sum(b.premium_paid for b in info.builds))
    assert p["premiums_cum_B"].iloc[-1] == pytest.approx(sum(b.premium_paid for b in info.builds) + sum(r.premium_paid for r in info.rolls))
    assert p["premiums_cum_B"].iloc[0] == pytest.approx(info.builds[0].premium_paid)
    assert p["payoffs_cum_B"].iloc[-1] == pytest.approx(sum(r.payoff for r in info.rolls)) and p["payoffs_cum_B"].iloc[-1] > 0.0
    for c in ("interest_cum_A", "interest_cum_B", "premiums_cum_B", "payoffs_cum_B", "n_bought"):
        assert (np.diff(p[c].to_numpy()) >= -1e-9).all()
    # tranches bought to each day: one on the start day, the 52 build steps by the end of the build, then one per replacement at expiry
    assert p["n_bought"].iloc[0] == 1 and p.loc[info.build_end, "n_bought"] == 52 == len(info.builds) and p["n_bought"].iloc[-1] == 52 + sum(1 for r in info.rolls if r.premium_paid > 0.0)
    assert (p["n_calls"] <= p["n_bought"]).all() and p.loc[: info.build_end, "n_calls"].equals(p.loc[: info.build_end, "n_bought"])
    # the cash of the rotation is the payoffs of the calls that expired in the money, less the premiums of their replacements, plus T-bill interest;
    # it counts at its lending value in the dry powder (T-bills 90 % by default, SPX 75 %, the calls 0 %) and at 100 % in the NAV
    first_itm = next(r.date for r in info.rolls if r.payoff > 0.0)
    assert (p.loc[: first_itm, "cash_B"].iloc[:-1] == 0.0).all() and p.loc[first_itm, "cash_B"] > 0.0 and (p.loc[first_itm:, "cash_B"] > 0.0).all()
    assert p["cash_B"].iloc[-1] == pytest.approx(sum(r.payoff - r.premium_paid + r.equity_sold for r in info.rolls) + p["cash_int_B"].sum())
    assert np.allclose(p["dry_powder_B"], 0.9 * (0.75 * p["E_B"] + 0.0 * p["call_val"] + 0.9 * p["cash_B"]) - p["loan_B"]) and (p["dry_powder_B"] - 0.9 * 0.75 * p["E_B"]).iloc[-1] == pytest.approx(0.9 * 0.9 * p["cash_B"].iloc[-1])
    assert np.allclose(p["cap_B"], 0.75 * p["E_B"] + 0.9 * p["cash_B"])
    assert np.allclose(p["nav_B"], p["E_B"] + p["call_val"] + p["cash_B"] - p["loan_B"])
    # dividends: received on the SPX held, net of withholding, reinvested the same day — never paid out, never used for the interest (which is capitalised).
    # Each day's dividend = the SPX value less what the price move alone would have given; over the path they reconcile with the total return exactly.
    px = _load(str(DAILY_FILE)).set_index("date").loc[p.index, "spx_px_last"].to_numpy()
    assert np.allclose(p["div_A"].to_numpy()[1:], p["E_A"].to_numpy()[1:] - p["E_A"].to_numpy()[:-1] * px[1:] / px[:-1]) and p["div_A"].iloc[0] == 0.0
    assert p["div_cum_A"].iloc[-1] == pytest.approx(p["div_A"].sum()) and p["div_cum_B"].iloc[-1] == pytest.approx(p["div_B"].sum())
    assert (p["div_A"].iloc[1:] > 0.0).all() and (p["div_B"].iloc[1:] > 0.0).all() and 0.0 < p["div_cum_B"].iloc[-1] < p["div_cum_A"].iloc[-1]   # the rotation holds fewer SPX units
    assert np.allclose(p["div_B"], p["eq_pnl_B"] - (p["E_B"].shift(1).fillna(0.0) * (px / np.roll(px, 1) - 1.0)).where(p.index != p.index[0], 0.0))   # part of the equity P&L on the units held over the day
    yrs = (p.index[-1] - p.index[0]).days / 365.25
    yld = p["div_cum_A"].iloc[-1] / p["E_A"].mean() / yrs   # ≈ the average trailing yield net of 15 % withholding over 2003–2012 (≈ 2.0 % × 0.85)
    assert 0.012 < yld < 0.022


# ---- the exposure-matching roll (roll="target", default) and the band rebalance

GOLDEN = {   # (roll, start, build steps): NAV B end, cash end, call notional end, SPX end, rolls, cut rolls, premiums cumulated, payoffs cumulated, lowest NAV B — recorded before the target rule existed
    ("delta", "1997-09-09", 1): (20811331129.00, 3011353562.00, 31437198388.25, 0.00, 5, 1, 7777235750.00, 9130358442.69, 548005348.00),
    ("delta", "1997-09-09", 52): (29185102776.32, 2067086718.56, 38572422796.61, 4927438117.16, 260, 0, 8822647398.42, 9957456180.87, 491097476.96),
    ("delta", "2003-03-10", 52): (29559634208.23, 2613098279.87, 37805469873.45, 8126264017.59, 208, 0, 7938678733.07, 9822379928.42, 635141179.87),
    ("units", "1997-09-09", 1): (7509844085.72, 2062833029.46, 3235138527.18, 3615251352.40, 5, 0, 1427079558.97, 2566502461.07, 566570570.16),
    ("units", "1997-09-09", 52): (10171560943.33, 1545875801.88, 2921270523.58, 6916732634.58, 260, 0, 1281741384.30, 2301758771.90, 578006772.11),
    ("units", "2003-03-10", 52): (12676402720.89, 1880932433.47, 3242833522.16, 9214967399.35, 208, 0, 1199162969.72, 2606463851.66, 668983366.59),
}


def test_delta_and_units_rules_unchanged() -> None:
    """Golden values: the two earlier roll rules reproduce, to the cent, the results they gave before the target rule was added."""
    for (roll, start, bt), (nav, cash, notional, spx, n_rolls, n_cut, prem, pay, low) in GOLDEN.items():
        p, info = simulate(start, "2026-09-14", build_tranches=bt, roll=roll)
        got = (p["nav_B"].iloc[-1], p["cash_B"].iloc[-1], p["call_notional"].iloc[-1], p["E_B"].iloc[-1], len(info.rolls), sum(1 for r in info.rolls if r.cut),
               p["premiums_cum_B"].iloc[-1], p["payoffs_cum_B"].iloc[-1], p["nav_B"].min())
        for g, e in zip(got, (nav, cash, notional, spx, n_rolls, n_cut, prem, pay, low), strict=True):
            assert g == pytest.approx(e, rel=1e-9, abs=0.01), (roll, start, bt, got)
        assert not info.trades and (p["unwind_cost_B"] == 0.0).all()   # no band trades outside the target rule


def _zero_rates(tmp_path: Path, spx: np.ndarray, name: str = "daily.csv", iv: float = 20.0) -> Path:
    return _file(tmp_path, spx, base=0.0, tbill=0.0, rate=0.0, iv=iv, name=name)


def test_target_roll_closes_the_gap_to_keeps_exposure(tmp_path: Path) -> None:
    """At expiry the replacement is sized on the gap between Keep's SPX value and what the survivors were bought to carry (grown with the
    index), so with no survivor the live exposure equals Keep's: paid from the payoff and the T-bills, or by selling SPX on the closed
    form N = (gap − cash)/(δ − c), s = c·N − cash; cut to what all the SPX can fund; nothing bought when the gap ≤ 0."""
    n = 6 * 262
    # 1. a strong rally: the payoff covers the premium, no SPX sold
    up = 1000.0 * np.exp(0.08 * np.arange(n) / 262)
    p, info = simulate("2010-01-04", "2015-12-31", build_tranches=1, rebalance="none", file=_zero_rates(tmp_path, up))
    assert len(info.rolls) == 1 and len(info.trades) == 1 and info.trades[0].event == "roll"
    r, t = info.rolls[0], info.trades[0]
    k = p.index.get_loc(r.date)
    assert not r.skipped and not r.cut and r.equity_sold == 0.0 == t.spx_usd and r.payoff > r.premium_paid > 0.0
    gap = p["E_A"].iloc[k] - p["E_B"].iloc[k]                              # no surviving tranche: the gap is the whole of Keep's exposure over the SPX kept
    assert t.exposure_before == pytest.approx(p["E_B"].iloc[k]) and t.target == pytest.approx(p["E_A"].iloc[k]) == t.band_edge
    assert r.notional == pytest.approx(gap / t.delta) and r.premium_paid == pytest.approx(r.premium_frac * r.notional) == t.usd_traded and t.notional == pytest.approx(r.notional)
    assert p["exposure_B"].iloc[k] == pytest.approx(p["E_A"].iloc[k], rel=1e-9) and t.exposure_after == pytest.approx(p["E_A"].iloc[k], rel=1e-9)   # exposure = target
    assert p["cash_B"].iloc[k] == pytest.approx(r.payoff - r.premium_paid) and p["nav_B"].iloc[k] == pytest.approx(p["nav_B"].iloc[k - 1] + p["eq_pnl_B"].iloc[k] + p["call_pnl_B"].iloc[k])
    assert p["call_notional"].iloc[k] == pytest.approx(r.notional) and p["n_calls"].iloc[k] == 1 and p["n_bought"].iloc[k] == 2
    # 2. a small rise: the payoff is short, SPX sold on the closed form; cash fully used
    up2 = 1000.0 * np.exp(0.02 * np.arange(n) / 262)
    p2, i2 = simulate("2010-01-04", "2015-12-31", build_tranches=1, rebalance="none", file=_zero_rates(tmp_path, up2, "up2.csv"))
    r2, t2 = i2.rolls[0], i2.trades[0]
    k2 = p2.index.get_loc(r2.date)
    e_b = p2["E_B"].iloc[k2 - 1] * up2[k2] / up2[k2 - 1]                   # the SPX held before the sale
    gap2 = p2["E_A"].iloc[k2] - e_b
    n_closed = (gap2 - r2.payoff) / (t2.delta - r2.premium_frac)
    assert r2.notional == pytest.approx(n_closed) and r2.equity_sold == pytest.approx(r2.premium_frac * n_closed - r2.payoff) and r2.equity_sold > 0.0 and not r2.cut
    assert p2["E_B"].iloc[k2] == pytest.approx(e_b - r2.equity_sold) and p2["cash_B"].iloc[k2] == pytest.approx(0.0, abs=1e-6) and t2.spx_usd == pytest.approx(-r2.equity_sold)
    assert p2["exposure_B"].iloc[k2] == pytest.approx(p2["E_A"].iloc[k2], rel=1e-9)
    assert p2["nav_B"].iloc[k2] == pytest.approx(e_b + r2.payoff - r2.premium_paid + r2.premium_paid)   # NAV-neutral: SPX + payoff swapped for calls (booked at the premium) and nothing else
    # 3. a worthless expiry on a falling path: gap > 0 (Keep still holds more), paid entirely by selling SPX
    down = 1000.0 * np.exp(-0.05 * np.arange(n) / 262)
    p3, i3 = simulate("2010-01-04", "2015-12-31", build_tranches=1, rebalance="none", file=_zero_rates(tmp_path, down, "down.csv"))
    r3 = i3.rolls[0]
    k3 = p3.index.get_loc(r3.date)
    assert r3.payoff == 0.0 and r3.premium_paid > 0.0 and r3.equity_sold == pytest.approx(r3.premium_paid) and not r3.cut and not r3.skipped
    assert p3["exposure_B"].iloc[k3] == pytest.approx(p3["E_A"].iloc[k3], rel=1e-9) and p3["n_calls"].iloc[k3] == 1
    p3l, i3l = simulate("2010-01-04", "2015-12-31", build_tranches=1, rebalance="none", replace_worthless=False, file=_zero_rates(tmp_path, down, "down.csv"))
    assert len(i3l.rolls) == 1 and i3l.rolls[0].premium_paid == 0.0 and not i3l.trades and (p3l["n_calls"].iloc[k3:] == 0).all()   # the lapse rule still lapses
    # 4. cut: a big loan (550 m) rotated, then a fall — not even all the SPX pays for the gap: every unit sold, the tranche cut, the roll flagged
    p4, i4 = simulate("2010-01-04", "2015-12-31", equity0=1000 * M, loan0=550 * M, build_tranches=1, rebalance="none", file=_zero_rates(tmp_path, down, "down.csv"))
    r4, t4 = i4.rolls[0], i4.trades[0]
    k4 = p4.index.get_loc(r4.date)
    assert r4.cut and t4.cut and p4["E_B"].iloc[k4] == 0.0 and p4["cash_B"].iloc[k4] == pytest.approx(0.0, abs=1e-6)
    assert r4.notional == pytest.approx(r4.equity_sold / r4.premium_frac) and p4["exposure_B"].iloc[k4] < p4["E_A"].iloc[k4] and t4.exposure_after < t4.target
    # 5. gap ≤ 0: on a long fall the quarterly checks buy calls from a cash buffer (their slots count from their own purchase day); by the expiry the
    #    survivors' slots cover Keep's exposure: nothing bought, the expiry logged as skipped, the payoff (nil) stays in T-bills
    fall = 1000.0 * np.exp(-0.12 * np.arange(n) / 262)
    p5, i5 = simulate("2010-01-04", "2015-12-31", build_tranches=1, cash_buffer=150 * M, rebalance="quarterly", band=0.10, below="calls", file=_zero_rates(tmp_path, fall, "fall.csv"))
    assert len(i5.rolls) == 1 and i5.rolls[0].skipped and i5.rolls[0].premium_paid == 0.0 and i5.rolls[0].notional == 0.0 and i5.rolls[0].payoff == 0.0
    k5 = p5.index.get_loc(i5.rolls[0].date)
    t5 = next(t for t in i5.trades if t.event == "roll")
    assert t5.usd_traded == 0.0 and t5.exposure_after == t5.exposure_before and p5["exposure_replaced_B"].iloc[k5] >= p5["E_A"].iloc[k5] - TOL_USD
    assert p5["n_calls"].iloc[k5] == p5["n_calls"].iloc[k5 - 1] - 1 and p5["n_bought"].iloc[k5] == p5["n_bought"].iloc[k5 - 1]   # the expired call gone, nothing added
    rs = rolling_starts(pd.DatetimeIndex(["2010-01-04"]), 0.0, build_tranches=1, cash_buffer=150 * M, rebalance="quarterly", band=0.10, below="calls", until="2015-12-31", file=_zero_rates(tmp_path, fall, "fall.csv"))
    assert rs["B: rolls skipped"].iloc[0] == 1 and rs["B: lapsed"].iloc[0] == 0 and rs["rolls"].iloc[0] == 1 and rs["B: cut rolls"].iloc[0] == 0
    tbl = starts_table(rs)
    assert tbl["rolls skipped (exposure already at Keep's)"].iloc[0] == 1 and tbl["exposure ratio today"].iloc[0] == pytest.approx(p5["exposure_B"].iloc[-1] / p5["E_A"].iloc[-1], abs=1e-4)
    # 6. a ladder keeps its count: four tranches a week apart on the rally; each replacement buys its own slot x_w · TR_t / TR_w (the survivors count at
    #    theirs), so the replaced exposure equals Keep's after every roll while the live exposure does not (the survivors' delta has moved)
    p6, i6 = simulate("2010-01-04", "2015-12-31", build_tranches=4, rebalance="none", file=_zero_rates(tmp_path, up))
    assert len(i6.builds) == 4 and len(i6.rolls) == 4 and not any(r.skipped or r.cut for r in i6.rolls) and (p6["n_calls"].loc[i6.build_end:] == 4).all()
    for b, r6 in zip(i6.builds, i6.rolls, strict=True):
        k6 = p6.index.get_loc(r6.date)
        kb = p6.index.get_loc(b.date)
        x_slot = b.equity_sold * up[k6] / up[kb]   # the SPX this tranche replaced, grown with the index (no dividends on the synthetic file)
        t6 = next(t for t in i6.trades if t.date == r6.date)
        assert r6.notional * t6.delta == pytest.approx(x_slot, rel=1e-9) and p6["exposure_replaced_B"].iloc[k6] == pytest.approx(p6["E_A"].iloc[k6], rel=1e-9)
        assert p6["exposure_B"].iloc[k6] != pytest.approx(p6["E_A"].iloc[k6], rel=1e-6) and p6["n_calls"].iloc[k6] == 4


def test_band_rebalance_up_sells_the_most_in_the_money_calls_first(tmp_path: Path) -> None:
    n = 6 * 262
    up = 1000.0 * np.exp(0.15 * np.arange(n) / 262)   # a strong rally: the calls' delta rises, the live exposure runs above the band
    f = _zero_rates(tmp_path, up)
    base, _ib = simulate("2010-01-04", "2015-12-31", build_tranches=2, rebalance="none", file=f)
    p, info = simulate("2010-01-04", "2015-12-31", build_tranches=2, rebalance="quarterly", band=0.10, unwind_haircut=0.0, file=f)
    ups = [t for t in info.trades if t.event == "rebalance_up"]
    assert ups and not [t for t in info.trades if t.event == "rebalance_down"]
    t0 = ups[0]
    k = p.index.get_loc(t0.date)
    assert p.index[k].is_quarter_end or (p.index[k + 1].quarter != p.index[k].quarter)   # the check runs on the last trading day of the quarter
    assert t0.exposure_before == pytest.approx(base["exposure_B"].iloc[k]) and t0.exposure_before > 1.1 * p["E_A"].iloc[k]
    assert t0.band_edge == pytest.approx(1.1 * p["E_A"].iloc[k]) and t0.strike == pytest.approx(up[0])   # the first tranche, struck lowest, is the most in the money
    same_day = [t for t in ups if t.date == t0.date]
    assert sum(t.exposure_before - t.exposure_after for t in same_day) == pytest.approx(t0.exposure_before - 1.1 * p["E_A"].iloc[k], rel=1e-9)
    assert p["exposure_B"].iloc[k] == pytest.approx(1.1 * p["E_A"].iloc[k], rel=1e-9)   # lands on the band edge
    assert all(t.notional < 0 and t.usd_traded > 0 and t.spx_usd == 0.0 for t in same_day)
    if len(same_day) == 1:   # a partial tranche: still two calls alive, the first one lighter by the notional sold
        assert p["n_calls"].iloc[k] == 2 and p["call_notional"].iloc[k] == pytest.approx(base["call_notional"].iloc[k] + t0.notional)
    assert p["nav_B"].iloc[k] == pytest.approx(base["nav_B"].iloc[k], rel=1e-12) and p["unwind_cost_B"].iloc[k] == 0.0   # sold at the mark: no value lost
    assert p["cash_B"].iloc[k] == pytest.approx(base["cash_B"].iloc[k] + sum(t.usd_traded for t in same_day))
    assert (p["exposure_B"] / p["E_A"]).loc[t0.date:].max() < 1.1 + 0.15   # between checks it may drift out, never far
    # with a haircut the sale is below the mark: the difference is booked as unwind cost, the identity holds, the NAV is lower by it
    ph, ih = simulate("2010-01-04", "2015-12-31", build_tranches=2, rebalance="quarterly", band=0.10, unwind_haircut=0.02, file=f)
    th = [t for t in ih.trades if t.event == "rebalance_up" and t.date == t0.date]
    assert th and ph["unwind_cost_B"].iloc[k] > 0.0 and ph["nav_B"].iloc[k] == pytest.approx(base["nav_B"].iloc[k] - ph["unwind_cost_B"].iloc[k], rel=1e-12)
    assert th[0].price < t0.price and th[0].notional == pytest.approx(t0.notional) and ph["exposure_B"].iloc[k] == pytest.approx(1.1 * ph["E_A"].iloc[k], rel=1e-9)
    from fosim.pricing.black_scholes import bsm_price as _bp
    tau = (pd.Timestamp(th[0].expiry) - t0.date).days / 365
    assert th[0].price == pytest.approx(float(_bp(up[k], th[0].strike, 0.0, 0.0, 0.18, tau)) / th[0].strike) and t0.price == pytest.approx(float(_bp(up[k], t0.strike, 0.0, 0.0, 0.20, tau)) / t0.strike)
    assert ph["unwind_cost_B"].iloc[k] == pytest.approx(-th[0].notional * (t0.price - th[0].price))
    gap_b = ph["nav_B"].diff().to_numpy()[1:] - (ph["eq_pnl_B"] + ph["call_pnl_B"] + ph["cash_int_B"] - ph["interest_B"] - ph["unwind_cost_B"]).to_numpy()[1:]
    assert np.abs(gap_b).max() < 1e-5
    # the CSV counts equal the trade log's
    rs = rolling_starts(pd.DatetimeIndex(["2010-01-04"]), 0.0, build_tranches=2, rebalance="quarterly", band=0.10, unwind_haircut=0.02, until="2015-12-31", file=f)
    log = trades_table(rs)
    assert rs["B: rebalances up"].iloc[0] == log[log["event"] == "rebalance_up"]["date"].nunique() == len({t.date for t in ih.trades if t.event == "rebalance_up"}) >= 1
    assert rs["B: calls sold at rebalances"].iloc[0] == pytest.approx(sum(t.usd_traded for t in ih.trades if t.event == "rebalance_up")) and rs["B: unwind cost"].iloc[0] == pytest.approx(ph["unwind_cost_B"].sum())
    assert list(log.columns) == ["start", "date", "event", "exposure_before", "exposure_after", "target", "band_edge", "usd_traded", "notional", "spx_usd", "price", "delta", "strike", "expiry", "partial", "cut"]
    assert len(log) == len(ih.trades) and log["usd_traded"].sum() == pytest.approx(sum(t.usd_traded for t in ih.trades) / M, abs=1e-2)
    tbl = starts_table(rs)
    assert tbl["rebalances up"].iloc[0] == rs["B: rebalances up"].iloc[0] and tbl["unwind cost (m)"].iloc[0] == pytest.approx(ph["unwind_cost_B"].sum() / M, abs=0.01)


def test_band_rebalance_down_buys_calls_from_the_t_bills(tmp_path: Path) -> None:
    n = 6 * 262
    down = 1000.0 * np.exp(-0.12 * np.arange(n) / 262)   # a long fall: the calls' delta melts, the live exposure drops below the band
    f = _zero_rates(tmp_path, down)
    p, info = simulate("2010-01-04", "2015-12-31", build_tranches=1, cash_buffer=150 * M, rebalance="quarterly", band=0.10, below="calls", file=f)   # 150 m of T-bills to buy with, nothing sold
    downs = [t for t in info.trades if t.event == "rebalance_down"]
    assert downs and not [t for t in info.trades if t.event == "rebalance_up"]
    t0 = downs[0]
    k = p.index.get_loc(t0.date)
    assert t0.exposure_before < 0.9 * p["E_A"].iloc[k] and t0.band_edge == pytest.approx(0.9 * p["E_A"].iloc[k]) and t0.notional > 0 and t0.spx_usd == 0.0
    if not t0.partial:
        assert p["exposure_B"].iloc[k] == pytest.approx(0.9 * p["E_A"].iloc[k], rel=1e-9) == pytest.approx(t0.exposure_after, rel=1e-9)
        assert t0.notional == pytest.approx((0.9 * p["E_A"].iloc[k] - t0.exposure_before) / t0.delta) and t0.usd_traded == pytest.approx(t0.price * t0.notional)
    assert p["n_calls"].iloc[k] == 2 and p["n_bought"].iloc[k] == 2 and p["cash_B"].iloc[k] == pytest.approx(p["cash_B"].iloc[k - 1] - t0.usd_traded)   # a new ATM tranche of the tenor, paid from the T-bills
    assert pd.Timestamp(t0.expiry) == t0.date + pd.DateOffset(months=60) and t0.strike == pytest.approx(down[k])
    # once the T-bills are gone the shortfall is bought only as far as they go and the check is flagged partial
    partial = [t for t in downs if t.partial]
    assert partial and p["cash_B"].loc[partial[0].date] == pytest.approx(0.0, abs=1e-6) and partial[0].exposure_after < partial[0].band_edge
    assert (p["unwind_cost_B"] == 0.0).all()
    rs = rolling_starts(pd.DatetimeIndex(["2010-01-04"]), 0.0, build_tranches=1, cash_buffer=150 * M, rebalance="quarterly", band=0.10, below="calls", until="2015-12-31", file=f)
    assert rs["B: rebalances down"].iloc[0] == len({t.date for t in downs}) and rs["B: partial rebalances"].iloc[0] == len({t.date for t in partial})
    assert rs["B: calls bought at rebalances"].iloc[0] == pytest.approx(sum(t.usd_traded for t in downs)) and rs["B: SPX bought at rebalances"].iloc[0] == 0.0
    # the spec's variant: SPX bought instead of calls
    ps, i_s = simulate("2010-01-04", "2015-12-31", build_tranches=1, cash_buffer=150 * M, rebalance="quarterly", band=0.10, below="spx", file=f)
    s0 = next(t for t in i_s.trades if t.event == "rebalance_down")
    ks = ps.index.get_loc(s0.date)
    assert s0.notional == 0.0 and s0.spx_usd == pytest.approx(s0.usd_traded) and s0.usd_traded > 0 and s0.delta == 1.0 and np.isnan(s0.strike) and pd.isna(s0.expiry) and s0.price == pytest.approx(down[ks])
    assert ps["E_B"].iloc[ks] == pytest.approx(ps["E_B"].iloc[ks - 1] * down[ks] / down[ks - 1] + s0.usd_traded) and ps["n_bought"].iloc[ks] == 1
    if not s0.partial:
        assert ps["exposure_B"].iloc[ks] == pytest.approx(0.9 * ps["E_A"].iloc[ks], rel=1e-9)
    assert s0.usd_traded < t0.usd_traded * 5   # SPX gives one dollar of exposure per dollar; calls about three
    # the option that goes on when the T-bills run out: SPX sold for calls, s = rest / (δ/c − 1), so the exposure lands on the edge; NAV-neutral on the day
    px, ix = simulate("2010-01-04", "2015-12-31", build_tranches=1, rebalance="quarterly", band=0.10, below="calls_spx", file=f)   # no cash buffer: SPX only
    x0 = next(t for t in ix.trades if t.event == "rebalance_down")
    kx = px.index.get_loc(x0.date)
    e_before = px["E_B"].iloc[kx - 1] * down[kx] / down[kx - 1]
    rest = 0.9 * px["E_A"].iloc[kx] - x0.exposure_before
    s_x = rest / (x0.delta / x0.price - 1.0)
    assert x0.spx_usd == pytest.approx(-s_x) and x0.notional == pytest.approx(s_x / x0.price) and x0.usd_traded == pytest.approx(s_x) and not x0.partial
    assert px["E_B"].iloc[kx] == pytest.approx(e_before - s_x) and px["cash_B"].iloc[kx] == pytest.approx(0.0, abs=1e-6)
    assert px["exposure_B"].iloc[kx] == pytest.approx(0.9 * px["E_A"].iloc[kx], rel=1e-9) and px["nav_B"].iloc[kx] == pytest.approx(px["nav_B"].iloc[kx - 1] + px["eq_pnl_B"].iloc[kx] + px["call_pnl_B"].iloc[kx])
    pc_, ic_ = simulate("2010-01-04", "2015-12-31", build_tranches=1, cash_buffer=150 * M, rebalance="quarterly", band=0.10, below="calls_spx", file=f)   # with T-bills: they go first
    dc = [t for t in ic_.trades if t.event == "rebalance_down"]
    assert dc[0].spx_usd == 0.0 and dc[0].usd_traded == pytest.approx(t0.usd_traded) and dc[0].notional == pytest.approx(t0.notional)   # same first trade as the T-bills-only rule
    mixed = next(t for t in dc if t.spx_usd < 0.0)
    km = pc_.index.get_loc(mixed.date)
    assert pc_["cash_B"].iloc[km] == pytest.approx(0.0, abs=1e-6) and pc_["exposure_B"].iloc[km] == pytest.approx(0.9 * pc_["E_A"].iloc[km], rel=1e-9) and not mixed.partial
    assert all(not t.partial for t in dc) and pc_["E_B"].iloc[-1] < p["E_B"].iloc[-1]   # never short of funding while SPX remains; the stock erodes into calls
    rsx = rolling_starts(pd.DatetimeIndex(["2010-01-04"]), 0.0, build_tranches=1, rebalance="quarterly", band=0.10, below="calls_spx", until="2015-12-31", file=f)
    assert rsx["B: SPX sold at rebalances"].iloc[0] == pytest.approx(-sum(t.spx_usd for t in ix.trades if t.event == "rebalance_down")) and rsx["B: SPX sold at rebalances"].iloc[0] > 0 and rsx["B: SPX bought at rebalances"].iloc[0] == 0.0
    assert starts_table(rsx)["SPX sold at rebalances (m)"].iloc[0] == pytest.approx(rsx["B: SPX sold at rebalances"].iloc[0] / M, abs=0.05)
    # when even all the SPX cannot close the shortfall: every unit sold, flagged partial (a 550 m loan rotated leaves little SPX)
    pz, iz = simulate("2010-01-04", "2015-12-31", equity0=1000 * M, loan0=550 * M, build_tranches=1, rebalance="quarterly", band=0.10, below="calls_spx", file=f)
    z = next(t for t in iz.trades if t.event == "rebalance_down" and t.partial)
    assert pz["E_B"].loc[z.date] == 0.0 and z.exposure_after < z.band_edge and z.spx_usd < 0.0


def test_band_check_timing_and_no_trade_cases(tmp_path: Path) -> None:
    n = 6 * 262
    flat = np.full(n, 1000.0)
    f = _zero_rates(tmp_path, flat)
    p, info = simulate("2010-01-04", "2015-12-31", build_tranches=1, rebalance="quarterly", band=0.10, file=f)   # flat: the exposure drifts a little with time decay, never out of the band
    assert [t.event for t in info.trades] == ["roll"] and (abs(p["exposure_B"] / p["E_A"] - 1.0) < 0.10).all()
    pm, im = simulate("2010-01-04", "2015-12-31", build_tranches=1, cash_buffer=200 * M, rebalance="monthly", band=0.0, file=f)   # a zero band: every month-end trades (time decay pulls the delta down: calls bought from the buffer)
    checks = [t for t in im.trades if t.event != "roll"]
    assert len(checks) >= 60 and all(t.event == "rebalance_down" for t in checks) and not checks[0].partial
    funded = [t.date for t in checks if not t.partial]
    assert (abs(pm["exposure_B"] / pm["E_A"] - 1.0).loc[funded] < 1e-9).all() and len(funded) >= 12
    _pn, i_n = simulate("2010-01-04", "2015-12-31", build_tranches=1, rebalance="none", band=0.0, file=f)
    assert [t.event for t in i_n.trades] == ["roll"]
    # a roll and a check on the same day: the roll first (a cut roll on a month-end leaves the exposure below the band, the check follows on the same date)
    down = 1000.0 * np.exp(-0.05 * np.arange(n) / 262)
    fd = _zero_rates(tmp_path, down, "down.csv")
    _pc, ic = simulate("2010-04-30", "2015-12-31", equity0=1000 * M, loan0=550 * M, build_tranches=1, rebalance="monthly", band=0.10, file=fd)
    day = next(t for t in ic.trades if t.event == "roll").date
    assert day == pd.Timestamp("2015-04-30") and [t.event for t in ic.trades if t.date == day] == ["roll", "rebalance_down"]
    same = [t for t in ic.trades if t.date == day]
    assert same[0].cut and same[1].partial and same[1].usd_traded == 0.0 and same[1].exposure_before == pytest.approx(same[0].exposure_after)   # nothing left to buy with
    # the live exposure is the SPX held plus Σ units · δ · S of the calls, δ from the model on the day
    k = 400
    tr = p.iloc[k]
    tau = (pd.Timestamp(info.builds[0].date) + pd.DateOffset(months=60) - p.index[k]).days / 365
    dl = float(np.asarray(bsm_greeks(1000.0, 1000.0, 0.0, 0.0, 0.20, tau).delta))
    assert tr["exposure_B"] == pytest.approx(tr["E_B"] + info.units0 * dl * 1000.0) and tr["exposure_A"] == tr["E_A"]
    # the build steps are never checked (the loan is being repaid): first check after the build
    _pl, il = simulate("2010-01-04", "2015-12-31", build_tranches=52, rebalance="monthly", band=0.0, file=f)
    assert min(t.date for t in il.trades if t.event != "roll") > il.build_end


def test_target_rule_on_the_record() -> None:
    """Real data, the Sep-1997 start under the defaults (monthly check, calls bought below the band from the T-bills then from SPX sold for
    them): the replaced exposure equals Keep's after every funded roll, the live exposure sits inside the band on every month-end after the
    build, nothing is ever left unfunded, no roll is cut; the calls bought in the 2001–02 fall take over the slots of the calls that then
    expire (skipped rolls), so the ladder thins after the first bear market; the summary row carries the counts."""
    p, info = simulate("1997-09-09", "2026-09-14")
    assert info.build_end is not None and not any(r.cut for r in info.rolls) and len(info.rolls) >= 50
    skipped = sum(1 for r in info.rolls if r.skipped)
    assert len(info.rolls) // 3 <= skipped < len(info.rolls) and p["n_calls"].loc[info.build_end : "2001-12"].min() >= 45 and p["n_calls"].loc["2004":].max() <= 20 and p["n_calls"].loc[info.build_end :].min() >= 1
    band_days = {t.date for t in info.trades if t.event != "roll"}
    for t in info.trades:
        if t.event == "roll" and t.notional > 0.0 and t.date not in band_days:   # (a month-end check after the roll moves it again)
            assert p.loc[t.date, "exposure_replaced_B"] == pytest.approx(t.target, rel=1e-9)   # the roll restores the slot: what the holdings were bought to carry = Keep's
    m_end = p.index[[*(p.index.to_period("M")[1:] != p.index.to_period("M")[:-1]).tolist(), False]]
    checked = [d for d in m_end if d > info.build_end]
    ratio = p["exposure_B"] / p["exposure_A"]
    assert not any(t.partial for t in info.trades) and all(abs(ratio.loc[d] - 1.0) <= 0.10 + 1e-9 for d in checked) and len(checked) > 300   # SPX always there to fund the shortfall
    assert ratio.iloc[0] == pytest.approx(1.0, abs=1e-4) and 1.1 < ratio.max() < 1.3 and 0.7 < ratio.min() < 0.85   # between checks it drifts with the delta
    assert p["unwind_cost_B"].sum() > 0.0 and (p["unwind_cost_B"] >= 0.0).all()   # 1 vol point off the mark on every early sale
    ups, downs = [t for t in info.trades if t.event == "rebalance_up"], [t for t in info.trades if t.event == "rebalance_down"]
    assert ups and downs and any(t.spx_usd < 0.0 for t in downs) and p["n_bought"].iloc[-1] == 52 + sum(1 for r in info.rolls if r.notional > 0.0) + sum(1 for t in downs if t.notional > 0.0)
    rs = rolling_starts(pd.DatetimeIndex(["1997-09-09"]), 0.0, until="2026-09-14")
    row = rs.iloc[0]
    assert row["B: rebalances up"] == len({t.date for t in ups}) >= 1 and row["B: rebalances down"] == len({t.date for t in downs}) >= 1 and row["B: cut rolls"] == 0 and row["B: partial rebalances"] == 0
    assert row["end: exposure B"] == pytest.approx(p["exposure_B"].iloc[-1]) and row["B: rolls skipped"] == skipped and row["B: SPX sold at rebalances"] == pytest.approx(-sum(t.spx_usd for t in downs)) and row["B: SPX sold at rebalances"] > 0
    assert row["B: unwind cost"] == pytest.approx(p["unwind_cost_B"].sum()) and len(row["B: trades"]) == len(info.trades)
    assert 0.8 < row["NAV B end"] / row["NAV A end"] < 1.25   # risk-matched: the two portfolios end in the same region
