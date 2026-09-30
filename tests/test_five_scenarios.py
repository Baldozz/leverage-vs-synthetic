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
from fosim.pricing.black_scholes import bsm_price

M = 1e6
C = 650 * M


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
    paths, infos = simulate_scenarios("2010-01-04", "2012-04-20", tenor=1.0, file=f)
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
    paths_n, infos_n = simulate_scenarios("2010-01-04", "2012-04-20", tenor=1.0, repay="never", file=f)
    pn = paths_n["2"]
    assert np.allclose(pn["loan"].to_numpy(), loan, rtol=1e-12) and infos_n["2"].loan_repaid_on is None and np.allclose(pn["nav"].to_numpy(), 1.25 * C * tr - loan, rtol=1e-12)


def test_margin_call_flagged_on_the_levered_scenarios_only(tmp_path: Path) -> None:
    """On a path falling to a fifth the levered long and the long-plus-calls are called when the loan exceeds m × ℓ_E × SPX (the calls carry no
    lending value): first day, days in call and LTV against the closed form; the unlevered scenarios report NaN."""
    spx = np.linspace(100.0, 20.0, 400)
    f = _file(tmp_path, spx)
    paths, infos = simulate_scenarios("2010-01-04", "2011-07-15", tenor=2.0, file=f)
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
    paths_hi, infos_hi = simulate_scenarios("2010-01-04", "2011-07-15", tenor=2.0, margin_call=1.0, file=f)   # the call at 100 % of the lending value comes later
    assert infos_hi["2"].margin_call_first > infos["2"].margin_call_first and paths_hi["2"]["headroom"].iloc[0] == pytest.approx(0.75 * 1.25 * C - 0.25 * C)


def test_scenario_3_tax_and_rebalance_at_expiry(tmp_path: Path) -> None:
    """75/25: at an in-the-money expiry the tax is 24 % of payoff − premium, then the whole portfolio is back to 75 % SPX / 25 % new calls
    (SPX bought from the proceeds); under 'proceeds' only the after-tax proceeds are split and the SPX held is untouched."""
    spx = np.concatenate([np.linspace(100.0, 150.0, 262), np.full(400, 150.0)])
    f = _file(tmp_path, spx)
    paths, infos = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, file=f)
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
    q, jnfo = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, rebalance="proceeds", file=f)
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
    paths, infos = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, file=f)
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
    q, jn = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, file=g)
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
    paths, infos = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, file=f)
    p, info = paths["5"], infos["5"]
    k = _expiry_index(p, 12)
    payoff = (C / 0.145 / 100.0) * (spx_up[k] - 100.0)
    e = info.expiries[0]
    assert p["E"].to_numpy().max() == 0.0 and e.payoff == pytest.approx(payoff) and e.tax == 0.0 and e.new_premium == pytest.approx(payoff) and e.new_notional == pytest.approx(payoff / 0.145)
    assert p["nav"].iloc[k] == pytest.approx(payoff) and info.wiped_out_on is None and p["tax_cum"].iloc[-1] == 0.0
    _, jn = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, tax_on=("3", "4", "5"), file=f)
    assert jn["5"].expiries[0].tax == pytest.approx(0.24 * (payoff - C)) and jn["5"].expiries[0].new_premium == pytest.approx(payoff - 0.24 * (payoff - C))
    assert jn["3"].expiries[0].tax == infos["3"].expiries[0].tax   # the other scenarios unchanged
    spx_down = np.concatenate([np.linspace(100.0, 80.0, 262), np.full(400, 90.0)])
    g = _file(tmp_path, spx_down)
    r, rn = simulate_scenarios("2010-01-04", "2012-07-13", tenor=1.0, file=g)
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
    paths, infos = simulate_scenarios("2010-01-04", "2011-02-25", tenor=1.0, file=f)
    i3 = infos["3"]
    assert i3.prem_frac0 == 0.145 and i3.notional0 == pytest.approx(0.25 * C / 0.145) and float(bsm_price(100.0, 100.0, 0.04, 0.04, i3.vol0, 1.0)) / 100.0 == pytest.approx(0.145, abs=1e-10)
    p = paths["3"]
    assert p["call_val"].iloc[0] == pytest.approx(0.25 * C) and p["nav"].iloc[0] == pytest.approx(C)
    tau = (p.index[0] + pd.DateOffset(months=12) - p.index[1]).days / 365.0
    assert p["call_val"].iloc[1] == pytest.approx(i3.notional0 / 100.0 * float(bsm_price(100.0, 100.0, 0.04, 0.04, i3.vol0, tau)))
    q, jn = simulate_scenarios("2010-01-04", "2011-02-25", tenor=1.0, premium=None, file=f)
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
    """The three start dates of the page on the real data; the facts of the dot-com-peak start checked by hand (2026-09-29): the all-in-calls
    portfolio is wiped out at its first expiry in March 2005, the levered long is never called at the 90 % level and repays the loan then."""
    starts = default_starts()
    assert starts == {"data start": pd.Timestamp("1997-09-09"), "dot-com peak": pd.Timestamp("2000-03-24"), "GFC bottom": pd.Timestamp("2009-03-09")}
    paths, infos = simulate_scenarios(starts["dot-com peak"], "2026-09-14")
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
    gfc, gi = simulate_scenarios(starts["GFC bottom"], "2026-09-14")
    assert all(gi[s].margin_call_first is None for s in ("2", "4")) and gfc["5"]["nav"].iloc[-1] > gfc["4"]["nav"].iloc[-1] > gfc["3"]["nav"].iloc[-1] > gfc["1"]["nav"].iloc[-1]
    assert all(gi[s].unexpired for s in ("3", "4", "5")) and all(len(gi[s].expiries) == 3 for s in ("3", "4", "5"))
