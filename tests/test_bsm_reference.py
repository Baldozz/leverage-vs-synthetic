"""The Black–Scholes engine against independent references: Hull's textbook values, a from-scratch scipy implementation on random inputs,
put–call parity, finite-difference Greeks with a shrinking step, the implied-vol round trip, the Black-76 equivalence used for SPXFP (q = r), and limits."""

from math import exp, log, sqrt

import numpy as np
import pytest
from scipy.stats import norm

from fosim.pricing.black_scholes import bsm_greeks, bsm_price
from fosim.pricing.implied_vol import implied_vol


def _ref(S: float, K: float, r: float, q: float, s: float, T: float) -> tuple[float, float, float]:
    d1 = (log(S / K) + (r - q + 0.5 * s * s) * T) / (s * sqrt(T))
    d2 = d1 - s * sqrt(T)
    call = S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    put = K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)
    return call, put, exp(-q * T) * norm.cdf(d1)


def test_hull_example_15_6() -> None:
    """Hull, Options, Futures and Other Derivatives: S = 42, K = 40, r = 10 %, σ = 20 %, T = 0.5 → call 4.76, put 0.81."""
    assert float(bsm_price(42, 40, 0.10, 0.0, 0.20, 0.5)) == pytest.approx(4.7594, abs=1e-4)
    assert float(bsm_price(42, 40, 0.10, 0.0, 0.20, 0.5, "put")) == pytest.approx(0.8086, abs=1e-4)


def test_random_inputs_against_independent_formula() -> None:
    rng = np.random.default_rng(0)
    for _ in range(2000):
        S, K = rng.uniform(50, 150), rng.uniform(50, 150)
        r, q, s, T = rng.uniform(-0.01, 0.08), rng.uniform(0, 0.06), rng.uniform(0.05, 0.8), rng.uniform(0.01, 12)
        call, put, delta = _ref(S, K, r, q, s, T)
        assert float(bsm_price(S, K, r, q, s, T)) == pytest.approx(call, abs=1e-9)
        assert float(bsm_price(S, K, r, q, s, T, "put")) == pytest.approx(put, abs=1e-9)
        assert float(bsm_greeks(S, K, r, q, s, T).delta) == pytest.approx(delta, abs=1e-12)


def test_put_call_parity() -> None:
    S, K, r, q, s, T = 100.0, 95.0, 0.04, 0.02, 0.25, 5.0
    assert float(bsm_price(S, K, r, q, s, T)) - float(bsm_price(S, K, r, q, s, T, "put")) == pytest.approx(S * exp(-q * T) - K * exp(-r * T), abs=1e-10)


def test_greeks_converge_to_finite_differences() -> None:
    S, K, r, q, s, T = 100.0, 95.0, 0.04, 0.02, 0.25, 5.0
    g = bsm_greeks(S, K, r, q, s, T)

    def px(**kw: float) -> float:
        a = {"S": S, "K": K, "r": r, "q": q, "s": s, "T": T, **kw}
        return float(bsm_price(a["S"], a["K"], a["r"], a["q"], a["s"], a["T"]))

    best: dict[str, float] = {}
    for h in (1e-3, 1e-4, 1e-5):   # central differences: truncation error ∝ h², so the analytic Greek must be approached as h shrinks (until round-off)
        errs = {   # relative to the size of the Greek (rho ≈ 190 here)
            "delta": abs(float(g.delta) - (px(S=S + h) - px(S=S - h)) / (2 * h)) / max(1.0, abs(float(g.delta))),
            "vega": abs(float(g.vega) - (px(s=s + h) - px(s=s - h)) / (2 * h)) / max(1.0, abs(float(g.vega))),
            "theta": abs(float(g.theta) + (px(T=T + h) - px(T=T - h)) / (2 * h)) / max(1.0, abs(float(g.theta))),
            "rho": abs(float(g.rho) - (px(r=r + h) - px(r=r - h)) / (2 * h)) / max(1.0, abs(float(g.rho))),
        }
        best = {k: min(errs[k], best.get(k, float("inf"))) for k in errs}
    assert max(best.values()) < 1e-8
    assert abs(float(g.rho) - (px(r=r + 1e-3) - px(r=r - 1e-3)) / 2e-3) > abs(float(g.rho) - (px(r=r + 1e-4) - px(r=r - 1e-4)) / 2e-4)   # rho: coarser step, larger error
    assert float(g.gamma) == pytest.approx((px(S=S + 1e-3) - 2 * px() + px(S=S - 1e-3)) / 1e-6, rel=1e-5)


def test_implied_vol_round_trip() -> None:
    for s0 in (0.08, 0.20, 0.60):
        price = float(bsm_price(100, 100, 0.03, 0.03, s0, 5.0))
        assert float(implied_vol(price, 100, 100, 0.03, 0.03, 5.0, "call")) == pytest.approx(s0, abs=1e-8)


def test_spxfp_convention_equals_black_76() -> None:
    """An excess-return index has forward = spot: BSM with q = r is Black-76 on the futures price; the spot delta is e^{−rT} N(d₁)."""
    r, s, T = 0.0482, 0.196, 5.0
    b76 = exp(-r * T) * (norm.cdf(0.5 * s * sqrt(T)) - norm.cdf(-0.5 * s * sqrt(T)))
    assert float(bsm_price(100, 100, r, r, s, T)) / 100 == pytest.approx(b76, abs=1e-12)
    assert float(bsm_greeks(100, 100, r, r, s, T).delta) == pytest.approx(exp(-r * T) * norm.cdf(0.5 * s * sqrt(T)), abs=1e-12)


def test_limits() -> None:
    assert float(bsm_price(110, 100, 0.05, 0.0, 0.2, 0.0)) == pytest.approx(10.0, abs=1e-12)                       # T = 0: intrinsic
    assert float(bsm_price(110, 100, 0.05, 0.0, 0.0, 1.0)) == pytest.approx(110 - 100 * exp(-0.05), abs=1e-9)     # σ = 0: discounted forward intrinsic
    assert float(bsm_price(100, 1000, 0.05, 0.0, 0.2, 1.0)) < 1e-20                                                # deep OTM
    assert float(bsm_price(1000, 100, 0.05, 0.0, 0.2, 1.0)) == pytest.approx(1000 - 100 * exp(-0.05), abs=1e-6)    # deep ITM
