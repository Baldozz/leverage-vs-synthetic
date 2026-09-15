"""Finite-difference Greeks (validation only) and smile-consistent delta (SPEC §5.2).

``fd_greeks`` is the independent reference used by the tests to validate the analytical Greeks in
``black_scholes.py``; it is never used by the engine.

Smile delta under sticky moneyness with σ(K) = σ_ATM + ψ·ln(K/F): ∂σ/∂S = −ψ/S, so
    Δ_smile = Δ_BSM + vega_raw · (−ψ/S).
Under sticky strike the surface does not move with S, so Δ_smile = Δ_BSM.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fosim.pricing.black_scholes import OptionKind, bsm_greeks, bsm_price

F64 = NDArray[np.float64]


def fd_greeks(
    S: float, K: float, r: float, q: float, sigma: float, T: float, kind: OptionKind = "call", h_rel: float = 1e-4
) -> dict[str, float]:
    """Central finite differences of the BSM price. Scalar inputs only (validation use)."""

    def p(**kw: float) -> float:
        args = {"S": S, "K": K, "r": r, "q": q, "sigma": sigma, "T": T}
        args.update(kw)
        return float(bsm_price(args["S"], args["K"], args["r"], args["q"], args["sigma"], args["T"], kind))

    hS = S * h_rel
    hs = max(sigma * h_rel, 1e-6)
    hr = 1e-5
    hT = max(T * h_rel, 1e-6)
    hK = K * h_rel
    return {
        "delta": (p(S=S + hS) - p(S=S - hS)) / (2 * hS),
        "gamma": (p(S=S + hS) - 2 * p() + p(S=S - hS)) / (hS * hS),
        "vega": (p(sigma=sigma + hs) - p(sigma=sigma - hs)) / (2 * hs),
        "theta": -(p(T=T + hT) - p(T=T - hT)) / (2 * hT),
        "rho": (p(r=r + hr) - p(r=r - hr)) / (2 * hr),
        "dq": (p(q=q + hr) - p(q=q - hr)) / (2 * hr),
        "dual_delta": (p(K=K + hK) - p(K=K - hK)) / (2 * hK),
    }


def smile_delta(
    S: ArrayLike,
    K: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    T: ArrayLike,
    psi_T: ArrayLike,
    sticky: str = "sticky_moneyness",
    kind: OptionKind = "call",
) -> F64:
    """Smile-consistent delta. ``psi_T`` is the skew slope ψ(T) in vol per unit log-moneyness."""
    g = bsm_greeks(S, K, r, q, sigma, T, kind)
    if sticky == "sticky_strike":
        out: F64 = np.asarray(g.delta, dtype=np.float64)
        return out
    S_ = np.asarray(S, dtype=np.float64)
    out2: F64 = g.delta + g.vega * (-np.asarray(psi_T, dtype=np.float64) / S_)
    return out2


def fd_smile_delta(
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    vol_of_spot: Callable[[float], float],
    kind: OptionKind = "call",
    h_rel: float = 1e-5,
) -> float:
    """Finite-difference delta with the surface vol re-evaluated at each bumped spot (validation)."""
    h = S * h_rel
    up = float(bsm_price(S + h, K, r, q, vol_of_spot(S + h), T, kind))
    dn = float(bsm_price(S - h, K, r, q, vol_of_spot(S - h), T, kind))
    return (up - dn) / (2 * h)
