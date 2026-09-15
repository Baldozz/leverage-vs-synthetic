"""Static hold-to-expiry payoff analysis (SPEC §6.6) — closed form, continuous compounding.

Single bullet tranche held to expiry, constant rates, cash at the risk-free rate r, dividends
reinvested at yield q, no costs, WHT or margin events:
    V_A(S_T) = E₀ (S_T/S₀) e^{qT} − L e^{r_L T}
    V_B(S_T) = N max(S_T/S₀ − K/S₀, 0) + (E₀ − L − N c) e^{rT},   c = premium per unit notional
Breakevens solve V_A = V_B (0, 1 or 2 roots). Reading: notional-matched B beats A only below a lower
breakeven (insurance); delta-matched B beats A below a lower or above an upper breakeven (convexity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class StaticPayoffInputs:
    E0: float  # A's equity exposure
    L: float  # loan
    S0: float
    K: float
    N: float  # option notional (USD)
    c: float  # premium per unit notional
    T: float
    r: float  # cash / discount rate (continuous)
    r_L: float  # loan rate (continuous)
    q: float  # reinvested dividend yield (continuous)


def v_a(x: StaticPayoffInputs, S_T: F64 | float) -> F64:
    s = np.asarray(S_T, dtype=np.float64)
    out: F64 = x.E0 * (s / x.S0) * np.exp(x.q * x.T) - x.L * np.exp(x.r_L * x.T)
    return out


def v_b(x: StaticPayoffInputs, S_T: F64 | float) -> F64:
    s = np.asarray(S_T, dtype=np.float64)
    out: F64 = x.N * np.maximum(s / x.S0 - x.K / x.S0, 0.0) + (x.E0 - x.L - x.N * x.c) * np.exp(x.r * x.T)
    return out


def breakevens(x: StaticPayoffInputs, s_max_rel: float = 10.0, n_grid: int = 4001) -> list[float]:
    """Index levels where V_A = V_B, found by sign changes on a grid and refined with brentq."""
    grid = np.linspace(1e-6, s_max_rel * x.S0, n_grid)
    f = v_a(x, grid) - v_b(x, grid)
    roots: list[float] = []
    for i in range(len(grid) - 1):
        if f[i] == 0.0:
            roots.append(float(grid[i]))
        elif f[i] * f[i + 1] < 0:
            roots.append(float(brentq(lambda s: float(v_a(x, s) - v_b(x, s)), grid[i], grid[i + 1], xtol=1e-12)))
    # deduplicate near-identical roots
    out: list[float] = []
    for r_ in roots:
        if not out or abs(r_ - out[-1]) > 1e-6 * x.S0:
            out.append(r_)
    return out


def reading(x: StaticPayoffInputs, roots: list[float]) -> str:
    if not roots:
        better = "B" if float(v_b(x, x.S0) - v_a(x, x.S0)) > 0 else "A"
        return f"No breakeven: {better} dominates over the whole range at expiry."
    if len(roots) == 1:
        return (
            f"One breakeven at S_T ≈ {roots[0]:.2f}: B beats A only below it (insurance value of the call's "
            "limited downside outweighs A's leverage); above it A's full exposure wins."
        )
    return (
        f"Two breakevens at S_T ≈ {roots[0]:.2f} and {roots[1]:.2f}: B beats A below the lower one (insurance) and "
        "above the upper one (convexity of the larger notional); A wins in between (carry and premium drag)."
    )
