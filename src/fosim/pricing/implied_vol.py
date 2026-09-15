"""Implied-volatility inversion with no-arbitrage bound checks (SPEC §4.2b, test 31)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from fosim.pricing.black_scholes import OptionKind, bsm_price, no_arbitrage_bounds


class ArbitrageBoundError(ValueError):
    """Raised when a quoted premium lies outside the BSM no-arbitrage bounds."""


def implied_vol(
    price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    kind: OptionKind = "call",
    sigma_lo: float = 1e-8,
    sigma_hi: float = 10.0,
    tol: float = 1e-12,
) -> float:
    """Back out σ from a premium by ``brentq`` on the BSM price. Raises ``ArbitrageBoundError``.

    Bounds (call): max(S e^{−qT} − K e^{−rT}, 0) < C < S e^{−qT}. A price exactly at the lower
    bound corresponds to σ = 0 and is returned as 0.0; a price at or above the upper bound is
    rejected.
    """
    if T <= 0:
        raise ValueError("implied_vol requires T > 0")
    lo, hi = no_arbitrage_bounds(S, K, r, q, T, kind)
    lo_f, hi_f = float(lo), float(hi)
    if price < lo_f - 1e-12 or price >= hi_f:
        raise ArbitrageBoundError(
            f"{kind} premium {price:.6g} outside no-arbitrage bounds [{lo_f:.6g}, {hi_f:.6g}) "
            f"for S={S}, K={K}, r={r}, q={q}, T={T}"
        )
    if abs(price - lo_f) <= 1e-12:
        return 0.0

    def f(s: float) -> float:
        return float(bsm_price(S, K, r, q, s, T, kind)) - price

    f_lo, f_hi = f(sigma_lo), f(sigma_hi)
    if f_lo > 0:
        return 0.0
    if f_hi < 0:
        raise ArbitrageBoundError(f"premium {price} not attainable with sigma <= {sigma_hi}")
    root: float = brentq(f, sigma_lo, sigma_hi, xtol=tol, rtol=4 * np.finfo(float).eps, maxiter=200)
    return root
