"""Spec tests 18 (engine level), 22, 23 (engine level), 28 — deterministic-path checks.

18. Margin call triggers exactly at the step where the equity decline first exceeds d* = 42.86%
    (and 28.57% after an LTV cut to 0.40) on a deterministic drawdown path.
22. No look-ahead: perturbing market data after step k* leaves every recorded value up to k* bit-identical.
23. Rolling ladder in the engine: strikes = spot at purchase, 12 tranches after build-up with residual
    maturities in [4, 5] years, roll dates as in the hand-built schedule.
28. Inception sizing (E₀ = 875, L = 250, sleeve 625, σ = 20 %, r = 4 %, q = 1.5 %, no costs):
    notional match → premium 188.04, cash 436.96; delta match → notional 1,361.8, premium 292.66,
    cash 332.34; 50-delta strike with dollar-delta 875 → notional 1,750, premium 250.88, cash 374.12.
"""

import dataclasses
import math
from typing import Any

import numpy as np
import pytest
from tests.conftest import make_cfg, zero_vol_overrides

from fosim.engine.simulator import Simulator
from fosim.instruments.lombard_loan import margin_trigger_decline
from fosim.market.generator import generate_market_paths
from fosim.market.illiquids import IlliquidPaths
from fosim.market.stress import StylisedScenario, build_stylised_paths


def _flat_iv_overrides(sigma: float = 0.20) -> dict[str, Any]:
    """Constant flat implied vol surface at ``sigma`` for every tenor and strike."""
    return {
        "implied_vol.short.theta": sigma, "implied_vol.short.iv0": sigma, "implied_vol.short.floor": 0.01, "implied_vol.short.cap": 2.0,
        "implied_vol.term_structure": {"tenors": [1, 2, 5], "theta": [sigma, sigma, sigma], "beta": [0.0, 0.0, 0.0]},
        "implied_vol.skew_1y": 0.0,
    }


@pytest.mark.parametrize("ltv,months", [(0.50, 9), (0.40, 5)])
def test_18_margin_call_step_on_deterministic_drawdown(raw: dict[str, Any], ltv: float, months: int) -> None:
    cfg = make_cfg(
        raw,
        **zero_vol_overrides(**{
            # closed-form d* assumes the loan is backed by equity only: cash LTV 0 so fund distributions
            # (which accrue to A's cash) do not add lending value
            "run.n_paths": 1, "run.horizon_years": 2, "leverage.ltv_base": {"equity_index": ltv, "illiquid": 0.0, "cash": 0.0},
            "leverage.ltv_stress_schedule": [], "rates.usd.r0": 0.0, "rates.cash.spread": 0.0, "loan_terms.spread_tiers": [{"utilisation_below": 1.0, "spread": 0.0}],
            "equity_indices.0.dividend_yield": 0.0, "equity_indices.0.expected_return": 0.0,
            "illiquids.1.unfunded_commitments": 0, "illiquids.2.unfunded_commitments": 0,
        }),
    )
    sc = StylisedScenario(name="t18", drawdown=-0.50, months_to_trough=10, recovery_shape="V", months_to_recover=6, iv_peak=0.20, iv_decay_months=1, post_scenario_drift=False)
    mp = build_stylised_paths(cfg, sc).paths
    res = Simulator(cfg, mp).run()
    a = res["A"]
    d_star = margin_trigger_decline(250, 875, 375, ltv, 0.0)
    lvl = mp.held[0] / mp.held[0, 0]
    first = int(np.argmax(1 - lvl >= d_star))
    assert first == months
    u = a.series("utilisation")[0]
    assert u[first - 1] < 1.0
    assert u[first] >= 1.0 - 1e-12 or a.events.margin_calls[0] >= 1
    # the ledger shows the call at exactly that step and no earlier
    led = a.ledger.to_frame()
    calls = led[led["description"].str.contains("MARGIN CALL")]
    assert int(calls["step"].min()) == first


def test_22_no_look_ahead(raw: dict[str, Any]) -> None:
    cfg = make_cfg(raw, **{"run.n_paths": 40, "run.horizon_years": 3, "exposure.policy": "rebalance_bands", "dry_powder.enabled": True})
    mp = generate_market_paths(cfg)
    base = Simulator(cfg, mp).run()
    k_star = 17
    # perturb everything strictly after k_star
    S = mp.S.copy()
    S[:, k_star + 1 :, :] *= 0.6
    held = mp.held.copy()
    held[:, k_star + 1 :] *= 0.6
    iv = mp.iv_short.copy()
    iv[:, k_star + 1 :] *= 2.0
    r = mp.r_short.copy()
    r[:, k_star + 1 :] += 0.02
    il = mp.illiquids
    nav_t = il.nav_true.copy()
    nav_t[:, k_star + 1 :, :] *= 0.5
    nav_r = il.nav_reported.copy()
    nav_r[:, k_star + 1 :, :] *= 0.5
    contr = il.contributions.copy()
    contr[:, k_star + 1 :, :] *= 3.0
    il2 = IlliquidPaths(il.names, nav_t, nav_r, contr, il.distributions.copy(), il.unfunded.copy(), il.true_return.copy(), il.reported_return.copy())
    mp2 = dataclasses.replace(mp, S=S, held=held, iv_short=iv, r_short=r, illiquids=il2)
    pert = Simulator(cfg, mp2).run()
    for name in base:
        for s in ("nav", "cash", "option_mv", "dollar_delta", "utilisation", "n_tranches"):
            assert np.array_equal(base[name].series(s)[:, : k_star + 1], pert[name].series(s)[:, : k_star + 1]), (name, s)
        assert np.array_equal(base[name].recorder.components[:, : k_star + 1, :], pert[name].recorder.components[:, : k_star + 1, :])
        assert not np.array_equal(base[name].nav[:, k_star + 1 :], pert[name].nav[:, k_star + 1 :])


def test_23_rolling_ladder_in_engine(raw: dict[str, Any]) -> None:
    cfg = make_cfg(raw, **{"run.n_paths": 3, "run.horizon_years": 4, "dry_powder.enabled": False})
    mp = generate_market_paths(cfg)
    sim = Simulator(cfg, mp, ledger_paths=[0, 1, 2])
    res = sim.run()
    b = res["B"]
    tr = b.ledger.tranche_frame()
    n = b.series("n_tranches")
    # build-up: k tranches after k months (1 at month 0), 12 from month 11 onwards
    for k in range(0, 12):
        assert n[:, k].tolist() == [k + 1] * 3
    assert (n[:, 11:] == 12).all()
    # strikes equal spot at purchase for every tranche, residual maturities within [4, 5] years after build-up
    for p in range(3):
        t = tr[tr["path"] == p]
        for _, row in t.iterrows():
            k_p = int(row["purchase_step"])
            assert row["strike"] == pytest.approx(mp.S[p, k_p, 0], rel=1e-12)
            assert int(row["expiry_step"]) == k_p + 60
        late = t[t["step"] >= 12]
        assert late["residual_years"].min() >= 4.0 - 1e-9 and late["residual_years"].max() <= 5.0 + 1e-9
        # roll dates: slot j repurchased at months j + 12 n (purchase_step of the tranche in slot j)
        for j in range(12):
            ps = sorted(set(t[t["slot"] == j]["purchase_step"]))
            assert ps == list(range(j, 49, 12))
    # step 48 is itself a month-end: slot 0 rolls at 12, 24, 36, 48 (4 sales), slots 1–11 three times each
    assert b.events.option_sales[0] == 4 + 11 * 3 and b.events.option_purchases[0] == 12 + 4 + 11 * 3


@pytest.mark.parametrize(
    "sizing,strike_mode,strike_param,exp_notional,exp_premium,exp_cash",
    [
        ("notional_match", "atm_spot", None, 875.0, 188.04, 436.96),
        ("delta_match", "atm_spot", None, 1361.8, 292.66, 332.34),
        ("delta_match", "delta_target", 0.50, 1750.0, 250.88, 374.12),
    ],
)
def test_28_inception_sizing_in_engine(raw: dict[str, Any], sizing: str, strike_mode: str, strike_param: float | None, exp_notional: float, exp_premium: float, exp_cash: float) -> None:
    cfg = make_cfg(
        raw,
        **zero_vol_overrides(**{
            **_flat_iv_overrides(0.20), "run.n_paths": 2, "run.horizon_years": 1, "options.ladder_mode": "bullet",
            "options.sizing_mode": sizing, "options.strike_mode": strike_mode, "options.strike_param": strike_param,
            # the spec's reference inputs assume a price-return underlying with q = 1.5 %
            "rates.usd.r0": 0.04, "equity_indices.0.underlying_type": "price_return", "equity_indices.0.pricing_dividend_curve": {"tenors": [1, 5], "q": [0.015, 0.015]},
            "held_equity_portfolio.beta_to_option_book": 1.0,
            "dry_powder.enabled": False,
        }),
    )
    mp = generate_market_paths(cfg)
    res = Simulator(cfg, mp).run()
    b = res["B"]
    tr = b.ledger.tranche_frame()
    t0 = tr[tr["step"] == 0]
    assert len(t0) == 1
    assert float(t0["notional_usd"].iloc[0]) / 1e6 == pytest.approx(exp_notional, abs=0.1)
    assert b.inception["option_premium0"] / 1e6 == pytest.approx(exp_premium, abs=0.01)
    assert b.inception["cash0"] / 1e6 == pytest.approx(exp_cash, abs=0.01)
    assert b.inception["nav0"] / 1e6 == pytest.approx(1000.0, abs=1e-6)
    if sizing == "delta_match":
        assert b.inception["dollar_delta0"] / 1e6 == pytest.approx(875.0, abs=0.05)
    else:
        assert b.inception["dollar_delta0"] / 1e6 == pytest.approx(875 * 0.64252, abs=0.1)
    if strike_mode == "delta_target":
        assert float(t0["strike"].iloc[0]) == pytest.approx(119.87, abs=0.01)
    # the illustrative inception check of SPEC §9: A holds 875 / 375 / 250
    a = res["A"]
    assert a.inception["equity"] == pytest.approx(875e6) and a.inception["loan"] == pytest.approx(250e6)
    assert math.isclose(a.inception["illiquid"], 375e6)
