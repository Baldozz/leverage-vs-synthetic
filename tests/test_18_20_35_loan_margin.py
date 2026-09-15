"""Spec tests 18, 19, 20, 35 (instrument level) — Lombard loan and margin mechanics.

18. Margin trigger d* = 42.86% for L=250, E₀=875, ℓ_E=0.5, ℓ_I=0; 28.57% with ℓ_E=0.4.
19. Cure sale: L=250, E=481.25, ℓ=0.5, u*=0.8, s=1% → x ≈ 97.46, post-sale utilisation 0.80 to 1e-9.
20. Loan interest: L=250, 5%, ACT/360, 12 monthly steps of 365/12 days → paid ≈ 12.674, capitalised ≈ 12.972.
35. Base-rate floor, spread tier switching at the configured thresholds, commitment fee hand-check.
"""

from typing import Any

import numpy as np
import pytest

from fosim.config.schema import LeverageConfig, LoanTermsConfig, LtvStressRule
from fosim.engine.conventions import days_per_step
from fosim.instruments.lombard_loan import (
    LoanTerms,
    cure_sale_size,
    lending_value,
    ltv_multiplier,
    margin_trigger_decline,
    utilisation,
)


def _terms(**kw: object) -> LoanTerms:
    lt = LoanTermsConfig(spread_tiers=[{"utilisation_below": 1.0, "spread": 0.01}], **kw)  # type: ignore[arg-type]
    return LoanTerms(lt, LeverageConfig())


def test_18_margin_trigger_closed_form() -> None:
    assert margin_trigger_decline(250, 875, 375, 0.5, 0.0) == pytest.approx(0.428571428571, abs=1e-9)
    assert margin_trigger_decline(250, 875, 375, 0.4, 0.0) == pytest.approx(0.285714285714, abs=1e-9)
    # the engine formula agrees: utilisation reaches 1 exactly at the closed-form decline
    lev = LeverageConfig()
    d = margin_trigger_decline(250, 875, 375, 0.5, 0.0)
    lv = lending_value(lev, np.array([875 * (1 - d)]), np.array([375.0]), np.array([0.0]), np.array([1.0]))
    assert float(utilisation(np.array([250.0]), lv)[0]) == pytest.approx(1.0, abs=1e-12)
    # LTV stress multiplier: IV above 0.35 cuts to 0.8 -> trigger at 28.57%
    lev2 = LeverageConfig(ltv_stress_schedule=[LtvStressRule(iv_short_above=0.35, multiplier=0.8)])
    h = ltv_multiplier(lev2, np.array([0.30, 0.40]), np.array([0.0, 0.0]))
    assert h.tolist() == [1.0, 0.8]


def test_19_cure_sale_formula() -> None:
    L, E, ltv, u_t, s = 250.0, 875.0 * 0.55, 0.5, 0.8, 0.01
    lv = ltv * E
    x = cure_sale_size(np.array([L]), np.array([lv]), u_t, np.array([ltv]), s)
    assert float(x[0]) == pytest.approx(97.46, abs=0.01)
    L_after = L - float(x[0]) * (1 - s)
    lv_after = ltv * (E - float(x[0]))
    assert L_after / lv_after == pytest.approx(0.80, abs=1e-9)
    with pytest.raises(ValueError):
        cure_sale_size(np.array([L]), np.array([lv]), 2.5, np.array([ltv]), s)


def test_20_loan_interest_hand_check() -> None:
    t = _terms(base_floor=0.0)
    days = days_per_step("monthly")
    L = np.array([250.0])
    rate = np.array([0.05])
    paid = sum(float(t.interest(L, rate, days)[0]) for _ in range(12))
    assert paid == pytest.approx(12.674, abs=5e-4)
    Lc = np.array([250.0])
    for _ in range(12):
        Lc = Lc + t.interest(Lc, rate, days)
    assert float(Lc[0]) - 250.0 == pytest.approx(12.972, abs=5e-4)
    # ACT/365 variant
    t365 = LoanTerms(t.cfg, LeverageConfig(day_count="ACT/365"))
    assert float(t365.interest(L, rate, days)[0]) * 12 == pytest.approx(250 * 0.05, abs=1e-12)


def test_35_floor_tiers_commitment_fee() -> None:
    lt = LoanTermsConfig(
        base_floor=0.01,
        spread_tiers=[{"utilisation_below": 0.5, "spread": 0.0080}, {"utilisation_below": 1.0, "spread": 0.0120}],  # type: ignore[list-item]
        facility_limit=400e6,
        commitment_fee_bps=25,
    )
    t = LoanTerms(lt, LeverageConfig())
    loan = np.array([100e6, 199.999e6, 200e6, 350e6])
    # floor: SOFR 0.2% -> base 1.0%
    r = t.floating_rate(np.array([0.002, 0.002, 0.002, 0.002]), loan)
    assert r.tolist() == pytest.approx([0.018, 0.018, 0.022, 0.022])
    # no floor binding
    r2 = t.floating_rate(np.array([0.04, 0.04, 0.04, 0.04]), loan)
    assert r2.tolist() == pytest.approx([0.048, 0.048, 0.052, 0.052])
    # commitment fee on undrawn: (400 - 100)m * 25bp * days/360
    days = days_per_step("monthly")
    fee = t.commitment_fee(loan[:1], days)
    assert float(fee[0]) == pytest.approx(300e6 * 0.0025 * days / 360)
    assert t.arrangement_fee(250e6) == 0.0
    # loan-size basis
    lt2 = LoanTermsConfig.model_validate(
        {**lt.model_dump(), "tier_basis": "loan_size", "spread_tiers": [{"utilisation_below": 150e6, "spread": 0.005}, {"utilisation_below": 1e12, "spread": 0.02}]}
    )
    t2 = LoanTerms(lt2, LeverageConfig())
    assert t2.spread(np.array([100e6, 200e6])).tolist() == pytest.approx([0.005, 0.02])
    # swap MTM is zero at par when the fixed rate equals the flat curve (annual periods, continuous DF approximation)
    lt3 = LoanTermsConfig.model_validate({**lt.model_dump(), "rate_type": "floating_swapped", "fixed_years": 3, "fixed_rate": 0.04})
    t3 = LoanTerms(lt3, LeverageConfig())
    mtm = t3.swap_mtm(np.array([250e6]), lambda tau: np.array([0.04]), 0.0)
    # par fixed rate for annual periods with continuous 4% ≈ 4.08%; MTM small but not zero
    assert abs(float(mtm[0])) < 250e6 * 0.01
    mtm_up = t3.swap_mtm(np.array([250e6]), lambda tau: np.array([0.06]), 0.0)
    assert float(mtm_up[0]) > float(mtm[0])  # pay-fixed gains when rates rise


def test_20b_monthly_capitalisation_on_weekly_grid(raw: dict[str, Any]) -> None:
    """Loan rolled monthly on a weekly grid: interest accrues each week on the balance fixed at the last roll and is
    added to the principal at month-end → after one year the loan equals 250 · Π_m (1 + r · days_m/360) with
    days_m = (weeks in month m) · 365/52; no cash leaves the account for interest."""
    from tests.conftest import make_cfg, zero_vol_overrides

    from fosim.runner import run_config

    cfg = make_cfg(
        raw,
        **zero_vol_overrides(**{
            "run.n_paths": 1, "run.horizon_years": 1, "run.dt": "weekly", "leverage.interest": "capitalise", "leverage.capitalisation_frequency": "monthly",
            "rates.usd.r0": 0.04, "loan_terms.spread_tiers": [{"utilisation_below": 1.0, "spread": 0.01}], "loan_terms.base_floor": 0.0,
            "equity_indices.1.dividend_yield": 0.0, "illiquids.1.unfunded_commitments": 0, "illiquids.2.unfunded_commitments": 0, "dry_powder.enabled": False,
        }),
    )
    out = run_config(cfg, ledger_paths=[0])
    a = out.results["A"]
    g = out.paths.grid
    loan = 250e6
    weeks_in_month = 0
    for k in range(1, g.n_steps + 1):
        weeks_in_month += 1
        if g.is_month_end[k]:
            loan *= 1 + 0.05 * (weeks_in_month * g.days) / 360.0
            weeks_in_month = 0
    assert a.series("loan")[0, -1] == pytest.approx(loan, rel=1e-12)
    assert a.series("accrued_interest")[0, -1] == pytest.approx(0.0, abs=1e-6)  # step 52 is a month-end
    # intra-month the accrued liability is in NAV: loan + accrued grows every week
    debt = a.series("loan")[0] + a.series("accrued_interest")[0]
    assert (np.diff(debt) > 0).all()
    # nothing was paid from cash for interest
    led = a.ledger.to_frame()
    assert not led[(led.account == "cash") & led.description.str.contains("interest paid")].shape[0]
    assert a.series("loan")[0, -1] / 250e6 - 1 == pytest.approx(12.972 / 250, abs=2e-4)  # ≈ spec test 20 monthly figure
