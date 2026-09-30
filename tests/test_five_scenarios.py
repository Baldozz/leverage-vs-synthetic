"""The investor's five scenarios (docs/METHODOLOGY.md §10, Assumptions 40): closed forms on synthetic paths, the expiry rules of each
scenario, the tax, the margin-call flag, the fixed-premium marks, the record."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fosim.analytics.five_scenarios import (
    LABELS,
    SCENARIOS,
    SUMMARY_ROWS,
    default_starts,
    paths_table,
    simulate_scenarios,
    summary,
)
from fosim.pricing.black_scholes import bsm_greeks, bsm_price

M = 1e6
C = 650 * M
OLD = {"build_steps": 1, "sizing": "premium", "tax_on": ("3", "4")}   # the 29-Sep rules: one call on the start day, κ of the capital in premium, 5 untaxed


def _file(tmp_path: Path, spx: np.ndarray, base: float = 3.0, tbill: float = 2.0, div: float = 0.0, iv: float = 20.0, rate: float = 3.0, start: str = "2010-01-04") -> Path:
    dates = pd.bdate_range(start, periods=len(spx))
    df = pd.DataFrame({"date": dates, "spxfp": spx, "spx_px_last": spx, "spx_div_yld": div, "loan_base": base, "tbill_3m": tbill, "ust_5y": rate, "iv_5y": iv})
    p = tmp_path / "daily.csv"
    df.to_csv(p, index=False)
    return p


def _identity(p: pd.DataFrame) -> None:
    gap = p["nav"].diff().iloc[1:] - (p["eq_pnl"] + p["call_pnl"] + p["cash_int"] - p["interest"] - p["tax"]).iloc[1:]
    assert np.abs(gap.to_numpy()).max() < 1e-6 * max(1.0, float(p["nav"].abs().max()) / 1e9)


def _accrued(p: pd.DataFrame, loan0: float, rate: float) -> np.ndarray:
    dt = np.concatenate([[0.0], np.diff(p.index.to_numpy().astype("datetime64[D]")).astype(float) / 360.0])
    return loan0 * np.cumprod(1.0 + rate * dt)


def _expiry_index(p: pd.DataFrame, months: int) -> int:
    target = p.index[0] + pd.DateOffset(months=months)
    return int(np.argmin(np.abs((p.index - target).days)))


def test_long_and_levered_long_closed_forms(tmp_path: Path) -> None:
    """1: NAV = C · TR. 2: the loan accrues simple ACT/360 at the flat rate, capitalised, and is repaid from SPX on the day nearest t₀ + tenor;
    with repay='never' it rolls up to the end."""
    spx = 100.0 * np.exp(np.linspace(0.0, 0.3, 600))
    f = _file(tmp_path, spx)
    paths, infos = simulate_scenarios("2010-01-04", "2012-04-20", tenor=1.0, file=f, **OLD)
    p1, p2 = paths["1"], paths["2"]
    tr = spx / spx[0]
    assert np.allclose(p1["nav"].to_numpy(), C * tr, rtol=1e-12) and (p1["loan"] == 0.0).all() and (p1["call_val"] == 0.0).all() and p1["ltv"].isna().all()
    loan = _accrued(p2, 0.25 * C, 0.055)
    k = _expiry_index(p2, 12)
    assert np.allclose(p2["loan"].to_numpy()[:k], loan[:k], rtol=1e-12) and (p2["loan"].to_numpy()[k:] == 0.0).all()
    assert np.allclose(p2["E"].to_numpy()[:k], 1.25 * C * tr[:k], rtol=1e-12)
    assert p2["E"].iloc[k] == pytest.approx(1.25 * C * tr[k] - loan[k], rel=1e-12) and p2["spx_traded"].iloc[k] == pytest.approx(-loan[k])   # SPX sold for the loan
    assert np.allclose(p2["nav"].to_numpy()[:k], 1.25 * C * tr[:k] - loan[:k], rtol=1e-12) and np.allclose(p2["nav"].to_numpy()[k:], (1.25 * C * tr[k] - loan[k]) * tr[k:] / tr[k], rtol=1e-12)
    assert infos["2"].loan_repaid_on == p2.index[k] and infos["2"].margin_call_first is None and infos["2"].max_ltv == pytest.approx(0.25 / (0.75 * 1.25))   # highest on the start day: the SPX outgrows the loan
    assert p2["interest_cum"].iloc[-1] == pytest.approx(loan[k] - 0.25 * C, rel=1e-9) and (p2["ltv"].to_numpy()[k:] == 0.0).all()
    for p in paths.values():
        _identity(p)
    paths_n, infos_n = simulate_scenarios("2010-01-04", "2012-04-20", tenor=1.0, repay="never", file=f, **OLD)
    pn = paths_n["2"]
    assert np.allclose(pn["loan"].to_numpy(), loan, rtol=1e-12) and infos_n["2"].loan_repaid_on is None and np.allclose(pn["nav"].to_numpy(), 1.25 * C * tr - loan, rtol=1e-12)


def test_margin_call_flagged_on_the_levered_scenarios_only(tmp_path: Path) -> None:
    """On a path falling to a fifth the levered long and the long-plus-calls are called when the loan exceeds m × ℓ_E × SPX (the calls carry no
    lending value): first day, days in call and LTV against the closed form; the unlevered scenarios report NaN."""
    spx = np.linspace(100.0, 20.0, 400)
    f = _file(tmp_path, spx)
    paths, infos = simulate_scenarios("2010-01-04", "2011-07-15", tenor=2.0, file=f, **OLD)
    tr = spx / spx[0]
    for sid, e0 in (("2", 1.25 * C), ("4", C)):
        p = paths[sid]
        loan = _accrued(p, 0.25 * C, 0.055)
        head = 0.9 * 0.75 * e0 * tr - loan
        first = int(np.argmax(head < 0.0))
        assert head[first] < 0.0 < head[first - 1] and infos[sid].margin_call_first == p.index[first] and infos[sid].days_in_margin_call == int((head < 0.0).sum())
        assert np.allclose(p["headroom"].to_numpy(), head, rtol=1e-9) and p["ltv"].iloc[first] == pytest.approx(loan[first] / (0.75 * e0 * tr[first])) and p["ltv"].iloc[first] > 0.9
        assert bool(p["margin_call"].iloc[first]) and not bool(p["margin_call"].iloc[first - 1]) and infos[sid].max_ltv == pytest.approx(loan[-1] / (0.75 * e0 * tr[-1]))
    for sid in ("1", "3", "5"):
        assert paths[sid]["ltv"].isna().all() and paths[sid]["headroom"].isna().all() and not paths[sid]["margin_call"].any() and infos[sid].margin_call_first is None
    assert len(infos["2"].expiries) == 0 and infos["3"].unexpired and infos["5"].unexpired   # a 2-year call on an 18-month path: not yet expired
    paths_hi, infos_hi = simulate_scenarios("2010-01-04", "2011-07-15", tenor=2.0, margin_call=1.0, file=f, **OLD)   # the call at 100 % of the lending value comes later
    assert infos_hi["2"].margin_call_first > infos["2"].margin_call_first and paths_hi["2"]["headroom"].iloc[0] == pytest.approx(0.75 * 1.25 * C - 0.25 * C)


def test_scenario_3_tax_and_rebalance_at_expiry(tmp_path: Path) -> None:
    """75/25: at an in-the-money expiry the tax is 24 % of payoff − premium, then the whole portfolio is back to 75 % SPX / 25 % new calls
    (SPX bought from the proceeds); under 'proceeds' only the after-tax proceeds are split and the SPX held is untouched."""
    spx = np.concatenate([np.linspace(100.0, 150.0, 262), np.full(400, 150.0)])
    f = _file(tmp_path, spx)
    paths, infos = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, file=f, **OLD)
    p, info = paths["3"], infos["3"]
    k = _expiry_index(p, 12)
    assert p["expiry"].iloc[k] and len(info.expiries) == 2 and info.expiries[0].date == p.index[k]
    units = (0.25 * C / 0.145) / 100.0
    payoff = units * (spx[k] - 100.0)
    tax = 0.24 * (payoff - 0.25 * C)
    e_before = 0.75 * C * spx[k] / 100.0
    total = e_before + payoff - tax
    e = info.expiries[0]
    assert e.payoff == pytest.approx(payoff) and e.tax == pytest.approx(tax) and e.premium_paid == pytest.approx(0.25 * C) and e.strike == 100.0 and not e.worthless
    assert e.spx_traded == pytest.approx(0.75 * total - e_before) and e.new_premium == pytest.approx(0.25 * total) and e.new_notional == pytest.approx(0.25 * total / 0.145)
    assert p["E"].iloc[k] == pytest.approx(0.75 * total) and p["call_val"].iloc[k] == pytest.approx(0.25 * total) and abs(p["cash"].iloc[k]) < 1e-6 and p["tax"].iloc[k] == pytest.approx(tax)
    assert p["call_notional"].iloc[k] == pytest.approx(0.25 * total / 0.145) and p["strike"].iloc[k] == 150.0 and p["nav"].iloc[k] == pytest.approx(total)
    assert p["call_pnl"].iloc[k] == pytest.approx(payoff - p["call_val"].iloc[k - 1]) and p["tax_cum"].iloc[-1] == pytest.approx(tax + info.expiries[1].tax)
    assert info.expiries[1].worthless and info.expiries[1].tax == 0.0   # the second call, struck at 150 on a flat path, expires worthless: no tax, the SPX sold for the 25 %
    k2 = _expiry_index(p, 24)
    assert info.expiries[1].spx_traded == pytest.approx(-0.25 * p["E"].iloc[k2 - 1]) and info.expiries[1].new_premium == pytest.approx(0.25 * p["E"].iloc[k2 - 1])
    _identity(p)
    q, jnfo = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, rebalance="proceeds", file=f, **OLD)
    pq, jn = q["3"], jnfo["3"]
    proceeds = payoff - tax
    assert jn.expiries[0].spx_traded == pytest.approx(0.75 * proceeds) and jn.expiries[0].new_premium == pytest.approx(0.25 * proceeds)
    assert pq["E"].iloc[k] == pytest.approx(e_before + 0.75 * proceeds) and pq["call_val"].iloc[k] == pytest.approx(0.25 * proceeds)
    assert jn.expiries[1].worthless and jn.expiries[1].new_premium == 0.0 and jn.expiries[1].spx_traded == 0.0 and pq["call_notional"].iloc[k2] == 0.0   # nothing to split
    _identity(pq)


def test_scenario_4_repays_the_loan_then_rolls_the_calls(tmp_path: Path) -> None:
    """Long + calls on a loan: an in-the-money expiry pays the tax, repays the rolled-up loan and puts the rest in new calls (no new loan);
    the next expiry, loan-free, rolls the after-tax proceeds. A worthless expiry sells SPX for the loan and buys no call."""
    spx = np.concatenate([np.linspace(100.0, 200.0, 262), np.linspace(200.0, 300.0, 400)])
    f = _file(tmp_path, spx)
    paths, infos = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, file=f, **OLD)
    p, info = paths["4"], infos["4"]
    k = _expiry_index(p, 12)
    loan = _accrued(p, 0.25 * C, 0.055)
    units = (0.25 * C / 0.145) / 100.0
    payoff = units * (spx[k] - 100.0)
    tax = 0.24 * (payoff - 0.25 * C)
    rest = payoff - tax - loan[k]
    e = info.expiries[0]
    assert rest > 0 and e.loan_repaid == pytest.approx(loan[k]) and e.spx_traded == 0.0 and e.new_premium == pytest.approx(rest) and e.tax == pytest.approx(tax)
    assert p["loan"].iloc[k] == 0.0 and p["loan"].iloc[k - 1] == pytest.approx(loan[k - 1]) and p["call_val"].iloc[k] == pytest.approx(rest) and p["E"].iloc[k] == pytest.approx(C * spx[k] / 100.0)
    assert info.loan_repaid_on == p.index[k] and p["interest_cum"].iloc[-1] == pytest.approx(loan[k] - 0.25 * C, rel=1e-9)
    k2 = _expiry_index(p, 24)
    e2 = info.expiries[1]
    payoff2 = (rest / 0.145 / spx[k]) * (spx[k2] - spx[k])
    assert e2.payoff == pytest.approx(payoff2) and e2.tax == pytest.approx(0.24 * (payoff2 - rest)) and e2.loan_repaid == 0.0 and e2.new_premium == pytest.approx(payoff2 - e2.tax)
    _identity(p)
    spx_down = np.concatenate([np.linspace(100.0, 80.0, 262), np.full(400, 80.0)])
    g = _file(tmp_path, spx_down)
    q, jn = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, file=g, **OLD)
    pq, j = q["4"], jn["4"]
    loan_q = _accrued(pq, 0.25 * C, 0.055)
    e = j.expiries[0]
    assert e.worthless and e.payoff == 0.0 and e.tax == 0.0 and e.spx_traded == pytest.approx(-loan_q[k]) and e.loan_repaid == pytest.approx(loan_q[k]) and e.new_premium == 0.0
    assert pq["loan"].iloc[k] == 0.0 and pq["E"].iloc[k] == pytest.approx(C * 0.8 - loan_q[k]) and (pq["call_notional"].to_numpy()[k:] == 0.0).all() and len(j.expiries) == 1
    assert np.allclose(pq["nav"].to_numpy()[k:], pq["nav"].iloc[k], rtol=1e-12)   # flat path, no loan, no call: nothing moves
    _identity(pq)


def test_scenario_5_rolls_untaxed_and_ends_on_a_worthless_expiry(tmp_path: Path) -> None:
    spx_up = np.concatenate([np.linspace(100.0, 200.0, 262), np.linspace(200.0, 300.0, 400)])
    f = _file(tmp_path, spx_up)
    paths, infos = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, file=f, **OLD)
    p, info = paths["5"], infos["5"]
    k = _expiry_index(p, 12)
    payoff = (C / 0.145 / 100.0) * (spx_up[k] - 100.0)
    e = info.expiries[0]
    assert p["E"].to_numpy().max() == 0.0 and e.payoff == pytest.approx(payoff) and e.tax == 0.0 and e.new_premium == pytest.approx(payoff) and e.new_notional == pytest.approx(payoff / 0.145)
    assert p["nav"].iloc[k] == pytest.approx(payoff) and info.wiped_out_on is None and p["tax_cum"].iloc[-1] == 0.0
    _, jn = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, file=f, **{**OLD, "tax_on": ("3", "4", "5")})
    assert jn["5"].expiries[0].tax == pytest.approx(0.24 * (payoff - C)) and jn["5"].expiries[0].new_premium == pytest.approx(payoff - 0.24 * (payoff - C))
    assert jn["3"].expiries[0].tax == infos["3"].expiries[0].tax   # the other scenarios unchanged
    spx_down = np.concatenate([np.linspace(100.0, 80.0, 262), np.full(400, 90.0)])
    g = _file(tmp_path, spx_down)
    r, rn = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, file=g, **OLD)
    pr, ri = r["5"], rn["5"]
    assert ri.expiries[0].worthless and ri.expiries[0].wiped_out and ri.wiped_out_on == pr.index[k] and (pr["nav"].to_numpy()[k:] == 0.0).all() and len(ri.expiries) == 1
    assert pr["nav"].iloc[k - 1] > 0.0 and summary(r, rn).loc["annualised return", "5"] == -1.0
    for sid in SCENARIOS:
        _identity(r[sid])


def test_fixed_premium_marks_and_market_mode(tmp_path: Path) -> None:
    """Fixed premium: the call costs 14.5 % of notional whatever the vol of the day and is marked at the vol that gives 14.5 % on the purchase day
    (so the purchase is NAV-neutral); market mode prices and marks at the vol of the day, the same fraction as the rotation engine's ATM call."""
    spx = np.full(300, 100.0)
    f = _file(tmp_path, spx, iv=30.0, rate=4.0)
    paths, infos = simulate_scenarios("2010-01-04", "2011-02-25", tenor=1.0, file=f, **OLD)
    i3 = infos["3"]
    assert i3.prem_frac0 == 0.145 and i3.notional0 == pytest.approx(0.25 * C / 0.145) and float(bsm_price(100.0, 100.0, 0.04, 0.04, i3.vol0, 1.0)) / 100.0 == pytest.approx(0.145, abs=1e-10)
    p = paths["3"]
    assert p["call_val"].iloc[0] == pytest.approx(0.25 * C) and p["nav"].iloc[0] == pytest.approx(C)
    tau = (p.index[0] + pd.DateOffset(months=12) - p.index[1]).days / 365.0
    assert p["call_val"].iloc[1] == pytest.approx(i3.notional0 / 100.0 * float(bsm_price(100.0, 100.0, 0.04, 0.04, i3.vol0, tau)))
    q, jn = simulate_scenarios("2010-01-04", "2011-02-25", tenor=1.0, premium=None, file=f, **OLD)
    c_mkt = float(bsm_price(100.0, 100.0, 0.04, 0.04, 0.30, 1.0)) / 100.0
    assert jn["3"].prem_frac0 == pytest.approx(c_mkt) and jn["3"].vol0 == 0.30 and jn["3"].notional0 == pytest.approx(0.25 * C / c_mkt) and c_mkt < 0.145   # a 1-year call at 30 % vol costs less than the fixed 14.5 %
    assert q["3"]["call_val"].iloc[1] == pytest.approx(jn["3"].notional0 / 100.0 * float(bsm_price(100.0, 100.0, 0.04, 0.04, 0.30, tau)))
    assert 0.0 < jn["3"].delta0 < 1.0 and jn["5"].notional0 == pytest.approx(C / c_mkt)


def test_validation_and_labels(tmp_path: Path) -> None:
    f = _file(tmp_path, np.full(50, 100.0))
    for kw in ({"capital": 0.0}, {"loan_frac": 1.0}, {"tax_rate": 1.0}, {"premium": 0.0}, {"loan_rate": -0.01}, {"margin_call": 0.0}, {"tax_on": ("6",)}, {"repay": "later"}, {"rebalance": "none"}):
        with pytest.raises(ValueError):
            simulate_scenarios("2010-01-04", "2010-03-01", file=f, **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        simulate_scenarios("2010-03-01", "2010-01-04", file=f)
    assert tuple(LABELS) == SCENARIOS and default_starts(f) == {"data start": pd.Timestamp("2010-01-04")}   # a flat path: no correction, no other start


def test_the_record_and_the_tables() -> None:
    """The five start dates of the page on the real data; the facts of the dot-com-peak start checked by hand (2026-09-29): the all-in-calls
    portfolio is wiped out at its first expiry in March 2005, the levered long is never called at the 90 % level and repays the loan then."""
    starts = default_starts()
    assert starts == {"data start": pd.Timestamp("1997-09-09"), "dot-com peak": pd.Timestamp("2000-03-24"), "dot-com bottom": pd.Timestamp("2002-10-09"),
                      "GFC peak": pd.Timestamp("2007-10-09"), "GFC bottom": pd.Timestamp("2009-03-09")}
    paths, infos = simulate_scenarios(starts["dot-com peak"], "2026-09-14", **OLD)
    assert infos["5"].wiped_out_on == pd.Timestamp("2005-03-24") and infos["5"].expiries[0].worthless and paths["5"]["nav"].iloc[-1] == 0.0
    assert infos["2"].margin_call_first is None and 0.5 < infos["2"].max_ltv < 0.7 and infos["2"].loan_repaid_on == pd.Timestamp("2005-03-24")
    assert infos["4"].loan_repaid_on == pd.Timestamp("2005-03-24") and infos["4"].expiries[0].worthless and infos["4"].expiries[0].spx_traded < 0 and len(infos["4"].expiries) == 1
    assert 45 * M < paths["2"]["interest_cum"].iloc[-1] < 60 * M   # 162.5 m at 5.5 % over five years, capitalised
    assert len(infos["3"].expiries) == 5 and sum(e.worthless for e in infos["3"].expiries) == 2 and infos["3"].unexpired and paths["3"]["tax_cum"].iloc[-1] > 0
    assert paths["3"]["nav"].iloc[-1] > paths["1"]["nav"].iloc[-1] > paths["2"]["nav"].iloc[-1] > paths["4"]["nav"].iloc[-1] > 0.0
    for sid in ("3", "4", "5"):
        assert infos[sid].prem_frac0 == 0.145 and infos[sid].notional0 == pytest.approx((C if sid == "5" else 0.25 * C) / 0.145)
    sm = summary(paths, infos)
    assert list(sm.index) == list(SUMMARY_ROWS) and list(sm.columns) == list(SCENARIOS)
    assert sm.loc["NAV today", "1"] == paths["1"]["nav"].iloc[-1] and sm.loc["wiped out on", "5"] == pd.Timestamp("2005-03-24") and pd.isna(sm.loc["wiped out on", "1"])
    years = (paths["1"].index[-1] - paths["1"].index[0]).days / 365.25
    assert sm.loc["annualised return", "1"] == pytest.approx((paths["1"]["nav"].iloc[-1] / C) ** (1 / years) - 1) and sm.loc["calls expired", "3"] == 5 and sm.loc["expired worthless", "3"] == 2
    assert sm.loc["lowest NAV on", "2"] == paths["2"]["nav"].idxmin() and sm.loc["max drawdown", "5"] == -1.0 and pd.isna(sm.loc["margin call first on", "2"]) and sm.loc["max LTV", "2"] == infos["2"].max_ltv
    long = paths_table({"dot-com peak": paths})
    assert len(long) == 5 * len(paths["1"]) and list(long.columns[:3]) == ["start", "scenario", "date"] and long["scenario"].iloc[0] == "1 Long the market"
    assert long.loc[long["scenario"] == "5 All in calls", "nav"].iloc[-1] == 0.0 and long["nav"].iloc[0] == pytest.approx(650.0)
    gfc, gi = simulate_scenarios(starts["GFC bottom"], "2026-09-14", **OLD)
    assert all(gi[s].margin_call_first is None for s in ("2", "4")) and gfc["5"]["nav"].iloc[-1] > gfc["4"]["nav"].iloc[-1] > gfc["3"]["nav"].iloc[-1] > gfc["1"]["nav"].iloc[-1]
    assert all(gi[s].unexpired for s in ("3", "4", "5")) and all(len(gi[s].expiries) == 3 for s in ("3", "4", "5"))


# ---------------------------------------------------------------- the ladder and the exposure sizing (user's decisions of 2026-09-30)


def _build_days(p: pd.DataFrame, steps: int) -> list[int]:
    """Indices of the trading days nearest t₀ + 7 w days, w < steps."""
    return [int(np.argmin(np.abs((p.index - (p.index[0] + pd.DateOffset(days=7 * w))).days.to_numpy()))) for w in range(steps)]


def test_ladder_build_rotates_spx_slot_by_slot(tmp_path: Path) -> None:
    """Four weekly slots on a flat path, premium sizing: each step sells one slot of premium out of the SPX (3), draws it as a loan (4) or sells an
    equal share of the SPX still held (5: all of it at the last step); every slot has its own expiry a tenor after its purchase; the marks, the
    notional and the dollar delta are sums over the live slots; a worthless slot of scenario 3 is refilled to κ·V/N with V the whole portfolio."""
    spx = np.full(700, 100.0)
    f = _file(tmp_path, spx)
    paths, infos = simulate_scenarios("2010-01-04", "2012-09-07", tenor=1.0, build_steps=4, build_unit="week", sizing="premium", file=f)
    p3, p4, p5 = paths["3"], paths["4"], paths["5"]
    days = _build_days(p3, 4)
    assert days == [0, 5, 10, 15] and infos["3"].n_tranches == 4 and infos["3"].build_end == p3.index[15] and infos["1"].n_tranches == 0
    slot = 0.25 * C / 4
    for k in days:
        assert p3["spx_traded"].iloc[k] == pytest.approx(-slot) and p4["spx_traded"].iloc[k] == 0.0 and p5["spx_traded"].iloc[k] == pytest.approx(-C / 4)   # flat path: C/4 of SPX at each step
    assert np.allclose(p3["E"].to_numpy()[days], C - slot * np.arange(1, 5)) and p3["n_calls"].tolist()[:16] == [1] * 5 + [2] * 5 + [3] * 5 + [4]
    assert p3["call_notional"].iloc[15] == pytest.approx(0.25 * C / 0.145) and infos["3"].premium_build == pytest.approx(0.25 * C) and infos["3"].notional_build == pytest.approx(0.25 * C / 0.145)
    assert (p4["E"].to_numpy()[:16] == C).all() and p4["loan"].iloc[0] == pytest.approx(slot) and infos["4"].loan0 == pytest.approx(0.25 * C)
    loan = sum(_accrued(p4.iloc[k:], slot, 0.055)[15 - k] for k in days)   # each slot's loan accrues from its own day
    assert p4["loan"].iloc[15] == pytest.approx(loan, rel=1e-12)
    assert p5["E"].iloc[15] == 0.0 and (p5["E"].to_numpy()[days[:3]] == pytest.approx(C - C / 4 * np.arange(1, 4))) and infos["5"].premium_build == pytest.approx(C)
    # the mark of the ladder is the sum of the four marks, the dollar delta the sum of the four deltas (each at its own time to expiry)
    k = 20
    marks = deltas = 0.0
    for kb in days:
        t_exp = p3.index[kb] + pd.DateOffset(months=12)
        tau = (t_exp - p3.index[k]).days / 365.0
        units = slot / 0.145 / 100.0
        marks += units * float(bsm_price(100.0, 100.0, 0.03, 0.03, infos["3"].vol0, tau))
        deltas += units * 100.0 * float(np.asarray(bsm_greeks(100.0, 100.0, 0.03, 0.03, infos["3"].vol0, tau).delta))
    assert p3["call_val"].iloc[k] == pytest.approx(marks) and p3["call_delta"].iloc[k] == pytest.approx(deltas) and p3["exposure"].iloc[k] == pytest.approx(p3["E"].iloc[k] + deltas)
    # each slot expires on its own day, worthless on the flat path; scenario 3 refills the slot to κ·V/4, V = SPX + the survivors' marks + cash
    ex = infos["3"].expiries
    assert [e.date for e in ex[:4]] == [p3.index[_expiry_index(p3.iloc[kb:], 12) + kb] for kb in days] and [e.slot for e in ex[:4]] == [0, 1, 2, 3] and all(e.worthless for e in ex[:4])
    ke = int(np.where(p3.index == ex[0].date)[0][0])
    survivors = p3["call_val"].iloc[ke] - ex[0].new_premium
    assert p3["n_calls"].iloc[ke] == 4 and ex[0].new_premium == pytest.approx(0.25 * (p3["E"].iloc[ke] - ex[0].spx_traded + survivors) / 4) and ex[0].spx_traded == pytest.approx(-ex[0].new_premium)
    ex5 = infos["5"].expiries
    assert all(e.worthless for e in ex5) and not ex5[2].wiped_out and ex5[3].wiped_out and infos["5"].wiped_out_on == ex5[3].date and p5["n_calls"].loc[ex5[2].date] == 1
    for sid in SCENARIOS:
        _identity(paths[sid])
    q, qi = simulate_scenarios("2010-01-04", "2012-09-07", tenor=1.0, build_steps=2, build_unit="month", sizing="premium", file=f)
    assert qi["3"].n_tranches == 2 and qi["3"].build_end == q["3"].index[int(np.argmin(np.abs((q["3"].index - (q["3"].index[0] + pd.DateOffset(months=1))).days)))]
    _, ri = simulate_scenarios("2010-01-04", "2010-02-10", tenor=1.0, build_steps=52, sizing="premium", file=f)   # a short path: the steps beyond the end are not taken
    assert ri["3"].n_tranches == 6 and ri["3"].build_end == pd.Timestamp("2010-02-08") and ri["3"].premium_build == pytest.approx(6 * 0.25 * C / 52)   # 4 Jan … 8 Feb, six Mondays


def test_exposure_sizing_closed_form(tmp_path: Path) -> None:
    """Exposure sizing: a slot carries X·C/N of SPX-equivalent, its premium is X·C/N × c ÷ δ (δ the model delta of the ATM call at the vol the fixed
    premium implies); at t₀ scenario 3 holds C − P of SPX + X·C of call delta, scenario 4 C + X·C on a loan of P, scenario 5 the whole capital
    in calls whatever X; the dollar-delta column follows bsm_greeks the next day; 'proceeds' is refused under exposure sizing."""
    spx = np.full(300, 100.0)
    f = _file(tmp_path, spx, rate=4.0)
    X = 0.86
    paths, infos = simulate_scenarios("2010-01-04", "2011-02-25", tenor=1.0, build_steps=1, sizing="exposure", exposure_frac=X, file=f)
    i3 = infos["3"]
    delta = float(np.asarray(bsm_greeks(100.0, 100.0, 0.04, 0.04, i3.vol0, 1.0).delta))
    P = X * C * 0.145 / delta
    assert i3.delta0 == pytest.approx(delta) and i3.premium0 == pytest.approx(P) and i3.premium_build == pytest.approx(P) and i3.notional0 == pytest.approx(P / 0.145)
    p3, p4, p5 = paths["3"], paths["4"], paths["5"]
    assert p3["E"].iloc[0] == pytest.approx(C - P) and p3["call_delta"].iloc[0] == pytest.approx(X * C) and p3["exposure"].iloc[0] == pytest.approx(C - P + X * C) and i3.exposure_build == pytest.approx(C - P + X * C)
    assert p4["E"].iloc[0] == C and p4["loan"].iloc[0] == pytest.approx(P) and p4["exposure"].iloc[0] == pytest.approx(C + X * C) and infos["4"].loan0 == pytest.approx(P)
    assert p5["call_val"].iloc[0] == pytest.approx(C) and p5["exposure"].iloc[0] == pytest.approx(C / 0.145 * delta)
    tau = (p3.index[0] + pd.DateOffset(months=12) - p3.index[1]).days / 365.0
    assert p3["call_delta"].iloc[1] == pytest.approx(i3.notional0 / 100.0 * 100.0 * float(np.asarray(bsm_greeks(100.0, 100.0, 0.04, 0.04, i3.vol0, tau).delta)))
    assert (paths["1"]["exposure"] == paths["1"]["E"]).all() and (paths["2"]["exposure"] == paths["2"]["E"]).all()
    q, qi = simulate_scenarios("2010-01-04", "2011-02-25", tenor=1.0, build_steps=4, sizing="exposure", exposure_frac=X, file=f)
    assert q["3"]["call_delta"].iloc[0] == pytest.approx(X * C / 4) and qi["3"].premium_build == pytest.approx(P) and abs(qi["3"].exposure_build - (C - P + X * C)) < 0.01 * C   # the earlier slots' deltas drift a little with time
    with pytest.raises(ValueError):
        simulate_scenarios("2010-01-04", "2011-02-25", tenor=1.0, sizing="exposure", rebalance="proceeds", file=f)
    for kw in ({"sizing": "delta"}, {"build_unit": "day"}, {"build_steps": 0}, {"exposure_frac": -0.1}):
        with pytest.raises(ValueError):
            simulate_scenarios("2010-01-04", "2011-02-25", tenor=1.0, file=f, **kw)  # type: ignore[arg-type]


def test_scenario_3_refills_a_paying_slot_and_scenario_4_repays_slot_by_slot(tmp_path: Path) -> None:
    """Two weekly slots on a rising path. 3: at the first slot's expiry the tax, then the slot refilled to κ·V/2 with V = SPX + the survivor's mark
    + the after-tax proceeds, the SPX bought with the rest. 4: the first expiry repays half the loan (its slot's share) from the proceeds and rolls
    the rest, the second repays what is left; the loan is gone that day."""
    spx = np.concatenate([np.linspace(100.0, 200.0, 262), np.linspace(200.0, 300.0, 400)])
    f = _file(tmp_path, spx)
    paths, infos = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, build_steps=2, sizing="premium", file=f)
    p3, i3 = paths["3"], infos["3"]
    e = i3.expiries[0]
    k = int(np.where(p3.index == e.date)[0][0])
    slot = 0.25 * C / 2
    units = slot / 0.145 / 100.0
    payoff = units * (spx[k] - 100.0)
    tax = 0.24 * (payoff - slot)
    survivor = p3["call_val"].iloc[k] - e.new_premium
    e_pre = p3["E"].iloc[k] - e.spx_traded
    assert e.payoff == pytest.approx(payoff) and e.tax == pytest.approx(tax) and e.slot == 0 and survivor > 0.0 and p3["n_calls"].iloc[k] == 2
    assert e.new_premium == pytest.approx(0.25 * (e_pre + survivor + payoff - tax) / 2) and e.spx_traded == pytest.approx(payoff - tax - e.new_premium) and abs(p3["cash"].iloc[k]) < 1e-6
    p4, i4 = paths["4"], infos["4"]
    e1, e2 = i4.expiries[:2]
    k1, k2 = (int(np.where(p4.index == x.date)[0][0]) for x in (e1, e2))
    assert e1.loan_repaid == pytest.approx(p4["loan"].iloc[k1]) and e1.spx_traded == 0.0 and e1.new_premium == pytest.approx(e1.payoff - e1.tax - e1.loan_repaid)   # half the loan: what is left equals what was repaid
    assert e2.loan_repaid == pytest.approx(p4["loan"].iloc[k2 - 1] * (1 + 0.055 * (p4.index[k2] - p4.index[k2 - 1]).days / 360)) and p4["loan"].iloc[k2] == 0.0 and i4.loan_repaid_on == e2.date
    assert i4.expiries[2].loan_repaid == 0.0 and i4.expiries[2].new_premium == pytest.approx(i4.expiries[2].payoff - i4.expiries[2].tax)   # the second cycle rolls the after-tax proceeds
    for sid in SCENARIOS:
        _identity(paths[sid])
    # the loan rolled up to the end: never repaid, every after-tax payoff into new calls, the LTV never zero, the SPX never sold
    q, qi = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, build_steps=2, sizing="premium", repay_calls="never", file=f)
    p4n, i4n = q["4"], qi["4"]
    loan_n = _accrued(p4n, 0.25 * C / 2, 0.055) + np.concatenate([np.zeros(5), _accrued(p4n.iloc[5:], 0.25 * C / 2, 0.055)])   # the two slots' loans, each from its own day
    assert np.allclose(p4n["loan"].to_numpy(), loan_n, rtol=1e-12) and i4n.loan_repaid_on is None and (p4n["ltv"] > 0.0).all() and (p4n["spx_traded"] == 0.0).all()
    assert all(e.loan_repaid == 0.0 and e.new_premium == pytest.approx(e.payoff - e.tax) for e in i4n.expiries) and len(i4n.expiries) == 4
    assert np.allclose(p4n["E"].to_numpy(), C * spx[: len(p4n)] / 100.0, rtol=1e-12) and summary(q, qi).loc["loan today", "4"] == pytest.approx(loan_n[-1])
    _identity(p4n)
    with pytest.raises(ValueError):
        simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, repay_calls="later", file=f)


def test_scenario_5_lapses_slot_by_slot(tmp_path: Path) -> None:
    """Two weekly slots: the index jumps between the two purchases, so the first slot pays at its expiry and rolls while the second lapses
    worthless a week later — not wiped out, one call alive; the rolled call lapses a year later with nothing left: wiped out that day."""
    spx = np.concatenate([np.full(3, 100.0), np.full(697, 120.0)])
    f = _file(tmp_path, spx)
    paths, infos = simulate_scenarios("2010-01-04", "2012-09-07", tenor=1.0, build_steps=2, sizing="premium", tax_on=("3", "4"), file=f)
    p, i = paths["5"], infos["5"]
    ex = i.expiries
    assert len(ex) == 3 and ex[0].strike == 100.0 and not ex[0].worthless and ex[0].new_premium == pytest.approx(ex[0].payoff) and ex[0].slot == 0
    assert ex[1].strike == 120.0 and ex[1].worthless and not ex[1].wiped_out and p["n_calls"].loc[ex[1].date] == 1 and p["nav"].loc[ex[1].date] > 0.0
    assert ex[2].strike == 120.0 and ex[2].worthless and ex[2].wiped_out and ex[2].slot == 0 and i.wiped_out_on == ex[2].date and (p["nav"].loc[ex[2].date:] == 0.0).all()
    _identity(p)


def test_the_record_on_the_defaults() -> None:
    """The defaults after the review of the investor's note (2026-09-30): premium sizing 25 % — 162.5 m of premium, 1,120.7 m of notional, the
    note's dollars to the cent — built in 52 weekly slots, tax on 3, 4 and 5, scenario 3 refilled from the whole portfolio. Facts checked by hand.
    Dot-com peak start: the build runs to 16 Mar 2001; scenario 3 rolls 52 slots five times (260 expiries, 102 worthless); scenario 4's 52 slots
    expire in 2005–06 all but one worthless, its loan repaid slot by slot and gone on 16 Mar 2006, never called at 90 % (LTV peaks near 71 %);
    the all-in-calls portfolio survives its first cycle on one slot and is wiped out on 16 Mar 2011. GFC-bottom start: no slot ever worthless,
    5 > 4 > 3 > 2 > 1."""
    starts = default_starts()
    paths, infos = simulate_scenarios(starts["dot-com peak"], "2026-09-14")
    i3, i4, i5 = infos["3"], infos["4"], infos["5"]
    assert i3.n_tranches == 52 and i3.build_end == pd.Timestamp("2001-03-16") and i3.premium_build == pytest.approx(0.25 * C, abs=1e-6) and i4.loan0 == pytest.approx(0.25 * C, abs=1e-6)
    assert i3.notional_build == pytest.approx(0.25 * C / 0.145, rel=1e-12) and i3.premium0 == pytest.approx(0.25 * C / 52) and 0.4 < i3.delta0 < 0.5
    assert paths["3"]["E"].iloc[0] == pytest.approx(C - i3.premium0) and paths["3"]["call_delta"].iloc[0] == pytest.approx(i3.premium0 / 0.145 * i3.delta0)
    assert len(i3.expiries) == 260 and sum(e.worthless for e in i3.expiries) == 102 and i3.unexpired and int(paths["3"]["n_calls"].iloc[-1]) == 52
    assert len(i4.expiries) == 52 and sum(e.worthless for e in i4.expiries) == 51 and i4.loan_repaid_on == pd.Timestamp("2006-03-16") and i4.margin_call_first is None and 0.6 < i4.max_ltv < 0.8
    assert i5.wiped_out_on == pd.Timestamp("2011-03-16") and len(i5.expiries) == 53 and paths["5"]["nav"].iloc[-1] == 0.0 and i5.premium_build < C   # the SPX fell during the build year
    assert infos["2"].margin_call_first is None and infos["2"].loan_repaid_on == pd.Timestamp("2005-03-24")
    assert paths["3"]["nav"].iloc[-1] > paths["1"]["nav"].iloc[-1] > paths["2"]["nav"].iloc[-1] > paths["4"]["nav"].iloc[-1] > 0.0
    sm = summary(paths, infos)
    assert list(sm.index) == list(SUMMARY_ROWS) and sm.loc["build completed on", "3"] == pd.Timestamp("2001-03-16") and pd.isna(sm.loc["build completed on", "1"]) and sm.loc["calls alive today", "3"] == 52
    assert sm.loc["exposure today", "1"] == paths["1"]["E"].iloc[-1] and sm.loc["exposure at the end of the build", "3"] == i3.exposure_build and sm.loc["tax paid", "5"] == 0.0   # every slot of 5 lapsed: nothing to tax
    long = paths_table({"dot-com peak": paths})
    assert {"call_delta", "exposure", "n_calls"} <= set(long.columns) and long["n_calls"].max() == 52
    gfc, gi = simulate_scenarios(starts["GFC bottom"], "2026-09-14")
    assert all(gi[s].margin_call_first is None for s in ("2", "4")) and gfc["5"]["nav"].iloc[-1] > gfc["4"]["nav"].iloc[-1] > gfc["3"]["nav"].iloc[-1] > gfc["2"]["nav"].iloc[-1] > gfc["1"]["nav"].iloc[-1]
    assert all(len(gi[s].expiries) == 156 and not any(e.worthless for e in gi[s].expiries) for s in ("3", "4", "5")) and gfc["5"]["tax_cum"].iloc[-1] > 0.0   # 5 taxed too
    assert gi["4"].loan_repaid_on == pd.Timestamp("2015-03-02") and gi["3"].premium_build == pytest.approx(0.25 * C, abs=1e-6)
