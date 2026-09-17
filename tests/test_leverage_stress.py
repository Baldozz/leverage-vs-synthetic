"""Keep the loan vs. rotate into calls (docs/METHODOLOGY.md §9): loan accrual, rotation ladder, margin-call trigger, rolls, accounting identity, rolling starts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fosim.analytics.leverage_stress import AccountingIdentityError, rolling_starts, simulate
from fosim.pricing.black_scholes import bsm_greeks, bsm_price

M = 1e6


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
    assert p["exposure_B"].iloc[0] == pytest.approx(1000 * M)   # delta exposure: equity kept + delta × notional = the exposure before
    p5, i5 = simulate("2010-01-04", "2011-02-25", equity0=1000 * M, loan0=250 * M, spread=0.0, delta=0.5, build_tranches=1, file=f)   # override for scripted use
    assert i5.delta0 == 0.5 and i5.rotation == pytest.approx(250 * M / (1 - 2 * c0)) and p5["exposure_B"].iloc[0] == pytest.approx(1000 * M)
    with pytest.raises(ValueError):
        simulate("2010-01-04", "2011-02-25", delta=1.5, build_tranches=1, file=f)
    # flat market, zero rates: A's NAV is flat, B's only loses the call's time value
    assert np.allclose(p["nav_A"], 750 * M) and (p["nav_B"].diff().dropna() <= 1e-6).all() and p["nav_B"].iloc[-1] < p["nav_B"].iloc[0]
    assert np.allclose(p["cap_B"], 0.75 * p["E_B"])


def test_margin_call_date_matches_closed_form(tmp_path: Path) -> None:
    n = 500
    spx = 1000.0 * (1.0 - 0.8 * np.arange(n) / (n - 1))   # linear fall to −80 %
    f = _file(tmp_path, spx, base=0.0, tbill=0.0)
    p, _ = simulate("2010-01-04", "2011-12-01", spread=0.0, ltv_equity=0.75, build_tranches=1, file=f)
    decline = 1.0 - p["E_A"] / p["E_A"].iloc[0]
    first_call = p.index[(p["headroom_A"] < 0).to_numpy().argmax()]
    # closed form: the loan (frozen at 250 with zero rates) meets the lending value 0.75 · E when E has fallen by 1 − 250/750 = 2/3
    assert decline.loc[first_call] > 2 / 3 and decline.shift(1).loc[first_call] <= 2 / 3
    assert p.loc[first_call, "ltv_A"] > 1.0 and p["ltv_A"].shift(1).loc[first_call] <= 1.0   # LTV = loan / lending value crosses 100 %


def test_roll_cash_flows_rising_and_falling(tmp_path: Path) -> None:
    n = 6 * 262
    up = 1000.0 * np.exp(0.08 * np.arange(n) / 262)
    f = _file(tmp_path, up, base=0.0, tbill=0.0, rate=0.0, iv=20.0)
    p, info = simulate("2010-01-04", "2015-12-31", build_tranches=1, file=f)
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
    p2, info2 = simulate("2010-01-04", "2015-12-31", build_tranches=1, file=fd)   # default: a worthless call lapses, nothing bought, no SPX sold
    r2 = info2.rolls[0]
    k2 = p2.index.get_loc(r2.date)
    assert r2.payoff == 0.0 and r2.premium_paid == 0.0 and r2.equity_sold == 0.0 and p2["cash_B"].iloc[k2] == 0.0
    assert (p2["call_notional"].iloc[k2:] == 0.0).all() and (p2["call_val"].iloc[k2:] == 0.0).all() and p2["E_B"].iloc[k2] == pytest.approx(p2["E_B"].iloc[k2 - 1] * down[k2] / down[k2 - 1])
    assert np.allclose(p2["exposure_B"].iloc[k2:], p2["E_B"].iloc[k2:])   # only the SPX held is exposed after the lapse
    p3, info3 = simulate("2010-01-04", "2015-12-31", build_tranches=1, replace_worthless=True, file=fd)   # option: replace it anyway, SPX sold to pay
    r3 = info3.rolls[0]
    assert r3.payoff == 0.0 and r3.premium_paid > 0 and r3.equity_sold == pytest.approx(r3.premium_paid) and p3["cash_B"].iloc[k2] == 0.0 and p3["call_notional"].iloc[k2] > 0
    p4, _ = simulate("2010-01-04", "2015-12-31", surplus="equity", build_tranches=1, file=f)
    assert p4["cash_B"].iloc[k] == 0.0 and p4["E_B"].iloc[k] > p["E_B"].iloc[k]
    p5, info5 = simulate("2010-01-04", "2015-12-31", surplus="calls", build_tranches=1, file=f)
    r5 = info5.rolls[0]
    assert p5["cash_B"].iloc[k] == 0.0 and r5.premium_paid == pytest.approx(r5.payoff)   # the whole payoff goes into calls
    assert p5["call_notional"].iloc[k] == pytest.approx(r5.payoff / r5.premium_frac) and p5["call_val"].iloc[k] == pytest.approx(r5.payoff)
    assert p5["nav_B"].iloc[k] == pytest.approx(p["nav_B"].iloc[k])   # reinvestment is NAV-neutral on the roll day


def test_cash_buffer_is_kept_after_the_rotation(tmp_path: Path) -> None:
    f = _file(tmp_path, np.full(300, 1000.0), base=0.0, tbill=0.0, rate=0.0)
    p, info = simulate("2010-01-04", "2011-02-25", loan0=250 * M, cash_buffer=100 * M, build_tranches=1, file=f)
    assert info.rotation == pytest.approx(350 * M / (1 - info.premium0 / info.delta0))
    assert p["cash_B"].iloc[0] == 100 * M and p["dry_powder_B"].iloc[0] == pytest.approx(0.75 * p["E_B"].iloc[0] + 100 * M)
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
    gap_b = p["nav_B"].diff().to_numpy()[1:] - (p["eq_pnl_B"] + p["call_pnl_B"] + p["cash_int_B"]).to_numpy()[1:]
    assert np.abs(gap_a).max() < 1e-5 and np.abs(gap_b).max() < 1e-5
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
    assert row["end: equity B"] + row["end: calls B"] + row["end: cash B"] - row["end: loan B"] == pytest.approx(row["NAV B end"])
    seen: list[tuple[int, int]] = []
    rolling_starts(starts, 5.0, build_tranches=1, file=f, progress=lambda i, n: seen.append((i, n)))
    assert seen and seen[-1] == (3, 3)
    rs2 = rolling_starts(starts, 5.0, build_tranches=1, file=f, until="2017-12-29")   # every start runs to the same end date
    assert len(rs2) == 3 and (rs2["end"] == rs2["end"].iloc[0]).all() and rs2["rolls"].tolist() == [1, 1, 0]
    assert (rs2["build end"] == rs2.index).all()
    pu, iu = simulate("2010-06-01", "2017-12-29", build_tranches=1, file=f)   # the first row against its own single path (its one call expired worthless on this random path)
    lapsed = [r for r in iu.rolls if r.payoff == 0.0 and r.premium_paid == 0.0]
    assert rs2["B: lapsed"].iloc[0] == len(lapsed) == 1 and rs2["B: calls lost on"].iloc[0] == lapsed[0].date and rs2["B: call notional end"].iloc[0] == 0.0 == pu["call_notional"].iloc[-1]
    assert rs2["B: lapsed"].iloc[2] == 0 and pd.isna(rs2["B: calls lost on"].iloc[2]) and rs2["B: call notional end"].iloc[2] > 0   # the 2016 start: no expiry before the end, calls still held
    fd = _file(tmp_path, 1000.0 * np.exp(-0.03 * np.arange(n) / 262), name="down.csv")   # falling path: the one call lapses → the option part is lost at its expiry
    rs3 = rolling_starts(starts[:1], 5.0, build_tranches=1, file=fd, until="2017-12-29")
    pd_, i_ = simulate("2010-06-01", "2017-12-29", build_tranches=1, file=fd)
    assert rs3["B: lapsed"].iloc[0] == 1 and rs3["B: calls lost on"].iloc[0] == i_.rolls[0].date and rs3["B: call notional end"].iloc[0] == 0.0 and (pd_["call_notional"].loc[i_.rolls[0].date:] == 0).all()
    assert rs2["B: min dry powder"].iloc[0] <= rs2["B: dry powder at trough"].iloc[0]


def test_input_validation(tmp_path: Path) -> None:
    f = _file(tmp_path, np.full(50, 1000.0))
    for kw in ({"equity0": 0.0}, {"ltv_equity": 1.5}, {"surplus": "gold"}, {"tenor": 0.0}, {"build_tranches": 0}):
        with pytest.raises(ValueError):
            simulate("2010-01-04", "2010-03-01", file=f, **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        simulate("2010-03-01", "2010-01-04", build_tranches=1, file=f)
    with pytest.raises(ValueError):   # loan too large to be repaid by rotating the equity
        simulate("2010-01-04", "2010-03-01", equity0=100 * M, loan0=95 * M, build_tranches=1, file=f)


def test_real_data_1997_path_and_rolling_starts() -> None:
    p, info = simulate("1997-09-09", "2026-09-14", build_tranches=1)
    assert 0.10 < info.premium0 < 0.25 and 0.40 < info.delta0 < 0.60 and 330 * M < info.rotation < 400 * M and info.notional0 == pytest.approx(info.rotation / info.delta0) and not info.fallback
    assert p["ltv_A"].iloc[0] == pytest.approx(0.25 / 0.75) and 0.60 < p["ltv_A"].max() < 0.75 and (p["headroom_A"] > 0).all()   # LTV = loan / (75 % × equity): never 100 %
    # the Sep-1997 call expires worthless in Sep-2002 and lapses (user's rule): one expiry, nothing bought, no calls afterwards
    assert len(info.rolls) == 1 and info.rolls[0].date.year == 2002 and info.rolls[0].payoff == 0.0 and info.rolls[0].premium_paid == 0.0 and (p["call_notional"].loc["2002-09-10":] == 0).all()
    _, info_r = simulate("1997-09-09", "2026-09-14", build_tranches=1, replace_worthless=True)   # option: always replace → five rolls, every premium paid
    assert len(info_r.rolls) == 5 and all(r.premium_paid > 0 for r in info_r.rolls)
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
    assert p["exposure_B"].iloc[kb[-1]] == pytest.approx(1000 * M)   # delta exposure fully replaced after the build
    assert p["dry_powder_B"].iloc[0] == pytest.approx(0.75 * p["E_B"].iloc[0] - 187.5 * M)   # lending value − loan left + cash
    # each tranche rolls at its own expiry: four rolls a week apart five years later
    assert len(info.rolls) == 4 and (info.rolls[-1].date - info.rolls[0].date).days == 21
    gap_b = p["nav_B"].diff().to_numpy()[1:] - (p["eq_pnl_B"] + p["call_pnl_B"] + p["cash_int_B"] - p["interest_B"]).to_numpy()[1:]
    assert np.abs(gap_b).max() < 1e-5
    p1, i1 = simulate("2010-01-04", "2015-12-31", equity0=1000 * M, loan0=250 * M, spread=0.0, build_tranches=1, file=f)
    assert len(i1.builds) == 1 and i1.rotation == pytest.approx(250 * M / (1 - c / dl)) and (p1["loan_B"] == 0).all()


def test_weekly_ladder_real_data() -> None:
    p, info = simulate("1997-09-09", "2026-09-14")   # default: 52 weekly tranches
    assert len(info.builds) == 52 and info.build_end.year == 1998 and (p["loan_B"].loc[info.build_end:] == 0).all()
    assert 0.0 < (p["loan_B"] / p["cap_B"]).max() < 0.5
    assert len(info.rolls) == 52 and all(r.payoff == 0.0 and r.premium_paid == 0.0 for r in info.rolls)   # all 52 tranches expire worthless in 2002–03 and lapse
    assert p["nav_B"].iloc[0] == pytest.approx(p["nav_A"].iloc[0])
    _, info_r = simulate("1997-09-09", "2026-09-14", replace_worthless=True)
    assert len(info_r.rolls) >= 52 * 5   # always replace: each tranche rolls every five years
