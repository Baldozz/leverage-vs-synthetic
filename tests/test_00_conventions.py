"""Conventions (SPEC §3): time grid, day counts, return conversions, dividend yield, flow rates."""

import math

import numpy as np
import pytest

from fosim.engine import conventions as cv


def test_steps_per_year_and_dt() -> None:
    assert cv.steps_per_year("monthly") == 12
    assert cv.steps_per_year("weekly") == 52
    assert cv.steps_per_year("daily") == 252
    assert cv.dt_years("monthly") == pytest.approx(1 / 12)
    assert cv.days_per_step("monthly") == pytest.approx(365 / 12)
    assert cv.days_per_step("daily") == pytest.approx(365 / 252)


def test_accrual_factor_act360_act365() -> None:
    days = 365 / 12
    assert cv.accrual_factor(days, "ACT/360") == pytest.approx(days / 360)
    assert cv.accrual_factor(days, "ACT/365") == pytest.approx(days / 365)
    with pytest.raises(ValueError):
        cv.accrual_factor(days, "30/360")  # type: ignore[arg-type]


def test_time_grid_monthly() -> None:
    g = cv.TimeGrid.build(horizon_years=2, dt="monthly")
    assert g.n_steps == 24
    assert g.t[0] == 0.0
    assert g.t[-1] == pytest.approx(2.0)
    # every monthly step (k >= 1) is a month end; t = 0 is not a step
    assert not g.is_month_end[0]
    assert g.is_month_end[1:].all()
    assert g.month_index[-1] == 24


def test_time_grid_daily_month_ends() -> None:
    g = cv.TimeGrid.build(horizon_years=1, dt="daily")
    assert g.n_steps == 252
    # 21 trading days per month -> 12 month ends at steps 21, 42, ..., 252
    ends = np.flatnonzero(g.is_month_end)
    assert ends.tolist() == [21 * k for k in range(1, 13)]


def test_time_grid_weekly_month_ends() -> None:
    g = cv.TimeGrid.build(horizon_years=1, dt="weekly")
    ends = np.flatnonzero(g.is_month_end)
    # 12 month ends in a year, last one at step 52
    assert len(ends) == 12
    assert ends[-1] == 52
    # month index is non-decreasing and increments by at most 1 per step
    d = np.diff(g.month_index)
    assert (d >= 0).all() and (d <= 1).all()


def test_return_conversions() -> None:
    sigma = 0.2
    # arithmetic 7% -> m = ln(1.07)
    m = cv.continuous_drift(0.07, "arithmetic", sigma)
    assert m == pytest.approx(math.log(1.07))
    assert cv.log_mean_per_year(0.07, "arithmetic", sigma) == pytest.approx(math.log(1.07) - sigma**2 / 2)
    # geometric 7% -> log-mean = ln(1.07), m = ln(1.07) + sigma^2/2
    assert cv.log_mean_per_year(0.07, "geometric", sigma) == pytest.approx(math.log(1.07))
    assert cv.continuous_drift(0.07, "geometric", sigma) == pytest.approx(math.log(1.07) + sigma**2 / 2)


def test_dividend_yield_and_flow_rates() -> None:
    assert cv.continuous_dividend_yield(0.015) == pytest.approx(math.log(1.015))
    # per-step flow rate: 1 - (1 - r)^dt; check round trip over 12 monthly steps
    r_step = cv.per_step_rate(0.25, 1 / 12)
    assert 1 - (1 - r_step) ** 12 == pytest.approx(0.25)
    assert cv.per_step_rate(0.0, 1 / 12) == 0.0
    assert cv.per_step_rate(1.0, 1 / 12) == 1.0


def test_dividend_cash_formula() -> None:
    # D = u * S_{t+dt} * (exp(q_c dt) - 1) * (1 - WHT)
    u, s1, q_c, dt, wht = 10.0, 110.0, math.log(1.02), 1 / 12, 0.15
    d = cv.dividend_cash(u, s1, q_c, dt, wht)
    assert d == pytest.approx(10 * 110 * (math.exp(q_c / 12) - 1) * 0.85)
