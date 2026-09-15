"""Conventions enforced everywhere (SPEC §3).

Time is in years. Money-market accrual is simple interest on ACT/360 (default) or ACT/365 with
365/steps_per_year calendar days per step. Option pricing uses continuous compounding with ACT/365
year fractions (handled in ``fosim.pricing``). Return inputs are converted to continuous drift
``m`` and log-mean ``m − σ²/2``; dividend yields to continuous ``q_c = ln(1 + q)``.

All functions here are pure and documented in ``docs/METHODOLOGY.md`` §1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

StepFrequency = Literal["monthly", "weekly", "daily"]
DayCount = Literal["ACT/360", "ACT/365"]
ReturnType = Literal["arithmetic", "geometric"]

_STEPS_PER_YEAR: dict[str, int] = {"monthly": 12, "weekly": 52, "daily": 252}
CALENDAR_DAYS_PER_YEAR = 365.0


def steps_per_year(freq: StepFrequency) -> int:
    """Number of simulation steps per year for a step frequency."""
    return _STEPS_PER_YEAR[freq]


def dt_years(freq: StepFrequency) -> float:
    """Step length in years: 1/12, 1/52 or 1/252."""
    return 1.0 / steps_per_year(freq)


def days_per_step(freq: StepFrequency) -> float:
    """Calendar days per step used for money-market accrual = 365 / steps_per_year (approximation)."""
    return CALENDAR_DAYS_PER_YEAR / steps_per_year(freq)


def accrual_factor(days: float, day_count: DayCount) -> float:
    """Simple-interest accrual factor for ``days`` calendar days under the day count."""
    if day_count == "ACT/360":
        return days / 360.0
    if day_count == "ACT/365":
        return days / 365.0
    raise ValueError(f"Unsupported day count {day_count!r}; use ACT/360 or ACT/365")


def continuous_drift(expected_return: float, return_type: ReturnType, sigma: float) -> float:
    """Continuous drift ``m`` such that E[S_T/S_0] = e^{mT} for a lognormal asset.

    arithmetic μ_a → m = ln(1 + μ_a);  geometric g → m = ln(1 + g) + σ²/2.
    """
    if return_type == "arithmetic":
        return math.log1p(expected_return)
    if return_type == "geometric":
        return math.log1p(expected_return) + 0.5 * sigma * sigma
    raise ValueError(f"Unknown return_type {return_type!r}")


def log_mean_per_year(expected_return: float, return_type: ReturnType, sigma: float) -> float:
    """Mean of ln(S_{t+1}/S_t) per year = m − σ²/2."""
    return continuous_drift(expected_return, return_type, sigma) - 0.5 * sigma * sigma


def continuous_dividend_yield(q_annual: float) -> float:
    """q_c = ln(1 + q_annual)."""
    return math.log1p(q_annual)


def per_step_rate(rate_annual: float, dt: float) -> float:
    """Convert an annual flow rate (e.g. capital-call rate) to a per-step rate: 1 − (1 − r)^dt."""
    return float(1.0 - (1.0 - rate_annual) ** dt)


def dividend_cash(
    units: float | NDArray[np.float64],
    s_end: float | NDArray[np.float64],
    q_c: float | NDArray[np.float64],
    dt: float,
    wht: float,
) -> float | NDArray[np.float64]:
    """Per-step dividend cash: ``u · S_{t+dt} · (exp(q_c dt) − 1) · (1 − WHT)`` (SPEC §3)."""
    out = units * s_end * np.expm1(np.asarray(q_c, dtype=np.float64) * dt) * (1.0 - wht)
    if np.ndim(out) == 0:
        return float(out)
    return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class TimeGrid:
    """Simulation time grid. ``t`` has ``n_steps + 1`` points starting at 0.

    ``is_month_end[k]`` is True when step k (the transition from t[k-1] to t[k]) ends a calendar
    month; ``month_index[k]`` = number of completed months at t[k]. Month ends are defined by
    ``floor(t·12 + 1e-9)`` changing, which gives every step for monthly, every 21st step for
    daily and a 4/5-week alternation for weekly stepping. Index 0 is never a month end.
    """

    freq: StepFrequency
    horizon_years: float
    n_steps: int
    dt: float
    days: float
    t: NDArray[np.float64]
    is_month_end: NDArray[np.bool_]
    month_index: NDArray[np.int64]

    @staticmethod
    def build(horizon_years: float, dt: StepFrequency) -> TimeGrid:
        spy = steps_per_year(dt)
        n_steps = round(horizon_years * spy)
        if not math.isclose(n_steps / spy, horizon_years, rel_tol=0, abs_tol=1e-12):
            raise ValueError(
                f"horizon_years={horizon_years} is not a whole number of {dt} steps ({spy}/year)"
            )
        t = np.arange(n_steps + 1, dtype=np.float64) / spy
        month_index = np.floor(t * 12.0 + 1e-9).astype(np.int64)
        is_month_end = np.zeros(n_steps + 1, dtype=np.bool_)
        is_month_end[1:] = month_index[1:] > month_index[:-1]
        return TimeGrid(
            freq=dt,
            horizon_years=float(horizon_years),
            n_steps=n_steps,
            dt=1.0 / spy,
            days=days_per_step(dt),
            t=t,
            is_month_end=is_month_end,
            month_index=month_index,
        )

    @property
    def steps_per_year(self) -> int:
        return steps_per_year(self.freq)
