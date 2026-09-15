"""Black–Scholes–Merton pricing (Q-measure) with continuous dividend yield (SPEC §5.2).

    C = S e^{−qT} N(d₁) − K e^{−rT} N(d₂),   P = K e^{−rT} N(−d₂) − S e^{−qT} N(−d₁)
    d₁ = [ln(S/K) + (r − q + σ²/2) T] / (σ√T),   d₂ = d₁ − σ√T

Conventions: continuous compounding, T in years (ACT/365), σ annualised, q continuous.
Vectorised over numpy arrays (broadcasting). Numerically stable limits:
  * T → 0: intrinsic value; delta = 1{S>K} (call) / −1{S<K} (put); other Greeks 0.
  * σ → 0 (T > 0): discounted-forward intrinsic max(S e^{−qT} − K e^{−rT}, 0); Greeks are the
    corresponding indicator limits.
  * deep ITM/OTM: ``scipy.special.ndtr`` (no cancellation in the normal CDF); prices clipped at 0
    only to remove −1e−17-type rounding residue.
Documented in docs/METHODOLOGY.md §3; tested in tests/test_01_06_pricing.py and
tests/test_24_27_greeks.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

OptionKind = Literal["call", "put"]
F64 = NDArray[np.float64]

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def _npdf(x: F64) -> F64:
    out: F64 = np.exp(-0.5 * x * x) / _SQRT_2PI
    return out


def _as(x: ArrayLike) -> F64:
    return np.asarray(x, dtype=np.float64)


def bsm_d1_d2(S: ArrayLike, K: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike, T: ArrayLike) -> tuple[F64, F64, F64]:
    """Return (d1, d2, sigma_sqrt_T). Where σ√T = 0, d1 = d2 = ±∞ by the sign of the discounted forward intrinsic."""
    S_, K_, r_, q_, s_, T_ = (_as(v) for v in (S, K, r, q, sigma, T))
    S_, K_, r_, q_, s_, T_ = np.broadcast_arrays(S_, K_, r_, q_, s_, T_)
    if np.any(S_ <= 0) or np.any(K_ <= 0):
        raise ValueError("S and K must be strictly positive")
    if np.any(T_ < 0) or np.any(s_ < 0):
        raise ValueError("T and sigma must be non-negative")
    ssT = s_ * np.sqrt(T_)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S_ / K_) + (r_ - q_ + 0.5 * s_ * s_) * T_) / ssT
    d2 = d1 - ssT
    degenerate = ssT <= 0.0
    if np.any(degenerate):
        # forward intrinsic sign: S e^{-qT} - K e^{-rT} (T=0 -> S - K)
        fwd = np.log(S_ / K_) + (r_ - q_) * T_
        inf = np.where(fwd > 0, np.inf, -np.inf)
        d1 = np.where(degenerate, inf, d1)
        d2 = np.where(degenerate, inf, d2)
    return d1, d2, ssT


def bsm_price(
    S: ArrayLike, K: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike, T: ArrayLike, kind: OptionKind = "call"
) -> F64:
    """European option price under BSM with continuous dividend yield."""
    S_, K_, r_, q_, T_ = np.broadcast_arrays(_as(S), _as(K), _as(r), _as(q), _as(T))
    d1, d2, _ = bsm_d1_d2(S, K, r, q, sigma, T)
    disc_s = S_ * np.exp(-q_ * T_)
    disc_k = K_ * np.exp(-r_ * T_)
    if kind == "call":
        p = disc_s * ndtr(d1) - disc_k * ndtr(d2)
    elif kind == "put":
        p = disc_k * ndtr(-d2) - disc_s * ndtr(-d1)
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    out: F64 = np.maximum(p, 0.0)
    return out


@dataclass(frozen=True)
class Greeks:
    """Raw analytical Greeks per unit of underlying (per option on one index unit).

    price: option value
    delta: ∂C/∂S
    gamma: ∂²C/∂S²
    vega:  ∂C/∂σ  (per unit of σ, i.e. per 100 vol points)
    theta: ∂C/∂t = −∂C/∂T  (per year; negative for long calls in normal conditions)
    rho:   ∂C/∂r  (per unit of r, i.e. per 10,000 bp)
    dq:    ∂C/∂q  (per unit of q)
    dual_delta: ∂C/∂K
    """

    price: F64
    delta: F64
    gamma: F64
    vega: F64
    theta: F64
    rho: F64
    dq: F64
    dual_delta: F64

    @property
    def elasticity(self) -> F64:
        """Option elasticity Ω = Δ·S/C — set by ``bsm_greeks`` via the ``_S`` field on request."""
        raise AttributeError("use elasticity(S) via bsm_elasticity()")


def bsm_greeks(
    S: ArrayLike, K: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike, T: ArrayLike, kind: OptionKind = "call"
) -> Greeks:
    """Analytical BSM Greeks (raw units, see ``Greeks``). Vectorised."""
    S_, K_, r_, q_, s_, T_ = np.broadcast_arrays(_as(S), _as(K), _as(r), _as(q), _as(sigma), _as(T))
    d1, d2, ssT = bsm_d1_d2(S_, K_, r_, q_, s_, T_)
    eq = np.exp(-q_ * T_)
    er = np.exp(-r_ * T_)
    sqT = np.sqrt(T_)
    pdf1 = np.where(np.isfinite(d1), _npdf(np.where(np.isfinite(d1), d1, 0.0)), 0.0)
    N1, N2 = ndtr(d1), ndtr(d2)
    Nm1, Nm2 = ndtr(-d1), ndtr(-d2)
    degenerate = ssT <= 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_raw = eq * pdf1 / (S_ * ssT)
        theta_common = -S_ * eq * pdf1 * s_ / (2.0 * sqT)
    gamma = np.where(degenerate, 0.0, gamma_raw)
    theta_common = np.where(degenerate, 0.0, theta_common)
    vega = np.where(degenerate, 0.0, S_ * eq * pdf1 * sqT)
    if kind == "call":
        price = np.maximum(S_ * eq * N1 - K_ * er * N2, 0.0)
        delta = eq * N1
        theta = theta_common + q_ * S_ * eq * N1 - r_ * K_ * er * N2
        rho = K_ * T_ * er * N2
        dq = -S_ * T_ * eq * N1
        dual = -er * N2
    elif kind == "put":
        price = np.maximum(K_ * er * Nm2 - S_ * eq * Nm1, 0.0)
        delta = -eq * Nm1
        theta = theta_common - q_ * S_ * eq * Nm1 + r_ * K_ * er * Nm2
        rho = -K_ * T_ * er * Nm2
        dq = S_ * T_ * eq * Nm1
        dual = er * Nm2
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return Greeks(
        price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho, dq=dq, dual_delta=dual
    )


def bsm_elasticity(g: Greeks, S: ArrayLike) -> F64:
    """Ω = Δ·S/C (premium leverage). Returns NaN where the price is zero."""
    S_ = _as(S)
    with np.errstate(divide="ignore", invalid="ignore"):
        out: F64 = np.where(g.price > 0, g.delta * S_ / g.price, np.nan)
    return out


@dataclass(frozen=True)
class ReportingGreeks:
    """Greeks in the reporting conventions of SPEC §5.2 for a position of ``n`` units.

    dollar_delta: Δ·n·S   (USD per 100 % index move)
    dollar_gamma: Γ·n·S²·0.01   (change in dollar delta for a 1 % move)
    delta_pp_per_1pct: Γ·S·0.01   (change in delta, in delta points, per 1 % move)
    vega_1pt: ∂C/∂σ·0.01·n   (USD per 1 vol point)
    theta_year: −∂C/∂T·n   (USD per year), theta_day = theta_year/365 (ACT/365)
    rho_1bp / rho_100bp: ∂C/∂r·1e−4·n / ∂C/∂r·0.01·n
    dq_100bp: ∂C/∂q·0.01·n
    elasticity: Δ·S/C
    """

    value: F64
    dollar_delta: F64
    dollar_gamma: F64
    delta_pp_per_1pct: F64
    vega_1pt: F64
    theta_year: F64
    theta_day: F64
    rho_1bp: F64
    rho_100bp: F64
    dq_100bp: F64
    elasticity: F64


def reporting_greeks(g: Greeks, S: ArrayLike, n: ArrayLike) -> ReportingGreeks:
    S_, n_ = _as(S), _as(n)
    return ReportingGreeks(
        value=g.price * n_,
        dollar_delta=g.delta * n_ * S_,
        dollar_gamma=g.gamma * n_ * S_ * S_ * 0.01,
        delta_pp_per_1pct=g.gamma * S_ * 0.01,
        vega_1pt=g.vega * 0.01 * n_,
        theta_year=g.theta * n_,
        theta_day=g.theta * n_ / 365.0,
        rho_1bp=g.rho * 1e-4 * n_,
        rho_100bp=g.rho * 0.01 * n_,
        dq_100bp=g.dq * 0.01 * n_,
        elasticity=bsm_elasticity(g, S_),
    )


def forward_price(S: ArrayLike, r: ArrayLike, q: ArrayLike, T: ArrayLike) -> F64:
    """F = S e^{(r − q) T}."""
    out: F64 = _as(S) * np.exp((_as(r) - _as(q)) * _as(T))
    return out


def no_arbitrage_bounds(S: ArrayLike, K: ArrayLike, r: ArrayLike, q: ArrayLike, T: ArrayLike, kind: OptionKind = "call") -> tuple[F64, F64]:
    """(lower, upper) no-arbitrage bounds: call ∈ [max(S e^{−qT} − K e^{−rT}, 0), S e^{−qT}]; put ∈ [max(K e^{−rT} − S e^{−qT}, 0), K e^{−rT}]."""
    S_, K_, r_, q_, T_ = np.broadcast_arrays(_as(S), _as(K), _as(r), _as(q), _as(T))
    ds, dk = S_ * np.exp(-q_ * T_), K_ * np.exp(-r_ * T_)
    if kind == "call":
        return np.maximum(ds - dk, 0.0), ds
    return np.maximum(dk - ds, 0.0), dk
