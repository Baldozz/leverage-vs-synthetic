"""Spec test 29 (static payoff §6.6) and the §9 illustrative inception check via the closed-form analytics.

29. r_L = 5 %, cash at r = 4 %, dividends reinvested: at S_T = 100, V_A ≈ 622.14, V_B ≈ 533.71 (notional
    match), ≈ 405.92 (delta match). Breakevens: notional match → single lower breakeven ≈ 90.62, no upper;
    delta match → ≈ 77.07 and ≈ 151.64.
"""

from typing import Any

import pytest
from tests.conftest import make_cfg

from fosim.analytics.inception import (
    balance_sheets,
    carry_comparison,
    margin_trigger_distance,
    put_call_parity_decomposition,
    sizing_table,
)
from fosim.analytics.static_payoff import StaticPayoffInputs, breakevens, v_a, v_b
from fosim.pricing.black_scholes import bsm_greeks


def test_29_static_payoff() -> None:
    g = bsm_greeks(100, 100, 0.04, 0.015, 0.20, 5.0)
    c = float(g.price) / 100
    d = float(g.delta)
    base = dict(E0=875.0, L=250.0, S0=100.0, K=100.0, c=c, T=5.0, r=0.04, r_L=0.05, q=0.015)
    nm = StaticPayoffInputs(N=875.0, **base)  # type: ignore[arg-type]
    dm = StaticPayoffInputs(N=875.0 / d, **base)  # type: ignore[arg-type]
    assert float(v_a(nm, 100.0)) == pytest.approx(622.14, abs=0.01)
    assert float(v_b(nm, 100.0)) == pytest.approx(533.71, abs=0.01)
    assert float(v_b(dm, 100.0)) == pytest.approx(405.92, abs=0.01)
    r_nm = breakevens(nm)
    assert len(r_nm) == 1 and r_nm[0] == pytest.approx(90.62, abs=0.01)
    r_dm = breakevens(dm)
    assert len(r_dm) == 2
    assert r_dm[0] == pytest.approx(77.07, abs=0.01) and r_dm[1] == pytest.approx(151.64, abs=0.01)


def test_inception_illustrative_check(raw: dict[str, Any]) -> None:
    sigma = 0.20
    cfg = make_cfg(
        raw,
        **{
            "options.ladder_mode": "bullet", "implied_vol.short.theta": sigma, "implied_vol.short.iv0": sigma,
            "implied_vol.term_structure": {"tenors": [1, 2, 5], "theta": [sigma, sigma, sigma], "beta": [0.0, 0.0, 0.0]},
            "implied_vol.skew_1y": 0.0, "pricing.bid_ask_vol_pts_new": 0.0,
            "equity_indices.0.underlying_type": "price_return", "equity_indices.0.pricing_dividend_curve": {"tenors": [1, 5], "q": [0.015, 0.015]},
            "held_equity_portfolio.beta_to_option_book": 1.0,
        },
    )
    tab = sizing_table(cfg)
    nm = tab[(tab.strike_mode == "atm_spot") & (tab.sizing_mode == "notional_match")].iloc[0]
    assert nm.premium / 1e6 == pytest.approx(188.0, abs=0.05)
    assert nm.premium_pct_notional == pytest.approx(0.2149, abs=1e-4)
    assert nm.cash / 1e6 == pytest.approx(437.0, abs=0.05)
    assert nm.dollar_delta / 1e6 == pytest.approx(562.2, abs=0.1)
    assert nm.dollar_delta_pct_nav == pytest.approx(0.562, abs=1e-3)
    dm = tab[(tab.strike_mode == "atm_spot") & (tab.sizing_mode == "delta_match")].iloc[0]
    assert dm.notional / 1e6 == pytest.approx(1361.8, abs=0.1)
    assert dm.premium / 1e6 == pytest.approx(292.7, abs=0.05)
    assert dm.cash / 1e6 == pytest.approx(332.3, abs=0.05)
    d50 = tab[(tab.strike_mode == "delta_target") & (tab.sizing_mode == "delta_match")].iloc[0]
    assert d50.strike_pct_spot == pytest.approx(1.1987, abs=1e-3)
    assert d50.notional / 1e6 == pytest.approx(1750.0, abs=0.5)
    assert d50.premium / 1e6 == pytest.approx(250.88, abs=0.05)
    bs = balance_sheets(cfg)
    assert bs.loc["A", "equities"] == pytest.approx(875e6) and bs.loc["A", "loan"] == pytest.approx(250e6)
    assert bs.loc["A", "dollar_delta_pct_nav"] == pytest.approx(0.875)
    assert bs.loc["B", "cash"] / 1e6 == pytest.approx(437.0, abs=0.05)
    assert bs.loc["C", "equities"] == pytest.approx(625e6)
    pcp = put_call_parity_decomposition(cfg)
    assert abs(pcp["parity_check"]) < 1e-10
    assert pcp["call_pct"] == pytest.approx(0.2149, abs=1e-4) and pcp["put_pct"] == pytest.approx(0.1059, abs=1e-4)
    carry = carry_comparison(cfg)
    assert carry.loc["A", "loan_interest"] == pytest.approx(-250e6 * 0.05)
    assert carry.loc["B", "option_theta"] < 0
    d = margin_trigger_distance(cfg)
    assert d["d_star_base"] == pytest.approx(0.428571, abs=1e-6)
    assert d["d_star_stress_x0.8"] == pytest.approx(0.285714, abs=1e-6)
