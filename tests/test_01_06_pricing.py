"""Spec tests 1–6: BSM reference values, put–call parity, Greeks vs FD, limits, Q-measure MC.

1. Hull: S=42, K=40, r=10%, σ=20%, T=0.5, q=0 → call ≈ 4.76, put ≈ 0.81.
2. 5y ATM anchor: S=K=100, r=4%, q=1.5%, σ=20%, T=5 → call ≈ 21.49, put ≈ 10.59, Δ ≈ 0.6425,
   recomputed independently with scipy inside the test.
3. Put–call parity to 1e-10 on a random grid (hypothesis).
4. All Greeks vs central finite differences.
5. Limits: T→0 intrinsic; σ→0 discounted-forward intrinsic; monotonicity in S, σ, T (q=0), K.
6. Q-measure Monte Carlo converges to BSM within 3 SE.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm

from fosim.pricing.black_scholes import bsm_elasticity, bsm_greeks, bsm_price, reporting_greeks
from fosim.pricing.greeks import fd_greeks


def _ref(S: float, K: float, r: float, q: float, s: float, T: float) -> tuple[float, float, float]:
    """Independent scipy recomputation (not using fosim)."""
    d1 = (math.log(S / K) + (r - q + 0.5 * s * s) * T) / (s * math.sqrt(T))
    d2 = d1 - s * math.sqrt(T)
    c = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    p = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    return c, p, math.exp(-q * T) * norm.cdf(d1)


def test_01_hull_reference() -> None:
    c = float(bsm_price(42, 40, 0.10, 0.0, 0.20, 0.5, "call"))
    p = float(bsm_price(42, 40, 0.10, 0.0, 0.20, 0.5, "put"))
    assert c == pytest.approx(4.76, abs=0.01)
    assert p == pytest.approx(0.81, abs=0.01)
    rc, rp, _ = _ref(42, 40, 0.10, 0.0, 0.20, 0.5)
    assert c == pytest.approx(rc, abs=1e-12)
    assert p == pytest.approx(rp, abs=1e-12)


def test_02_5y_atm_anchor() -> None:
    c = float(bsm_price(100, 100, 0.04, 0.015, 0.20, 5.0, "call"))
    p = float(bsm_price(100, 100, 0.04, 0.015, 0.20, 5.0, "put"))
    g = bsm_greeks(100, 100, 0.04, 0.015, 0.20, 5.0, "call")
    assert c == pytest.approx(21.49, abs=0.01)
    assert p == pytest.approx(10.59, abs=0.01)
    assert float(g.delta) == pytest.approx(0.6425, abs=5e-4)
    rc, rp, rd = _ref(100, 100, 0.04, 0.015, 0.20, 5.0)
    assert c == pytest.approx(rc, abs=1e-12)
    assert p == pytest.approx(rp, abs=1e-12)
    assert float(g.delta) == pytest.approx(rd, abs=1e-12)


@settings(max_examples=300, deadline=None)
@given(
    S=st.floats(10, 500),
    K=st.floats(10, 500),
    r=st.floats(-0.02, 0.15),
    q=st.floats(0.0, 0.08),
    s=st.floats(0.01, 1.5),
    T=st.floats(0.01, 10),
)
def test_03_put_call_parity(S: float, K: float, r: float, q: float, s: float, T: float) -> None:
    c = float(bsm_price(S, K, r, q, s, T, "call"))
    p = float(bsm_price(S, K, r, q, s, T, "put"))
    lhs = c - p
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(lhs - rhs) <= 1e-10 * max(1.0, S, K)


@pytest.mark.parametrize(
    "S,K,r,q,s,T,kind",
    [
        (100, 100, 0.04, 0.015, 0.20, 5.0, "call"),
        (100, 100, 0.04, 0.015, 0.20, 5.0, "put"),
        (42, 40, 0.10, 0.0, 0.20, 0.5, "call"),
        (80, 120, 0.02, 0.03, 0.35, 2.0, "call"),
        (130, 100, 0.05, 0.01, 0.15, 0.25, "put"),
        (100, 60, 0.03, 0.02, 0.25, 4.0, "call"),
    ],
)
def test_04_greeks_vs_finite_differences(S: float, K: float, r: float, q: float, s: float, T: float, kind: str) -> None:
    g = bsm_greeks(S, K, r, q, s, T, kind)  # type: ignore[arg-type]
    fd = fd_greeks(S, K, r, q, s, T, kind)  # type: ignore[arg-type]
    assert float(g.delta) == pytest.approx(fd["delta"], rel=1e-6, abs=1e-8)
    assert float(g.gamma) == pytest.approx(fd["gamma"], rel=1e-4, abs=1e-8)
    assert float(g.vega) == pytest.approx(fd["vega"], rel=1e-6, abs=1e-8)
    assert float(g.theta) == pytest.approx(fd["theta"], rel=1e-5, abs=1e-7)
    assert float(g.rho) == pytest.approx(fd["rho"], rel=1e-6, abs=1e-8)
    assert float(g.dq) == pytest.approx(fd["dq"], rel=1e-6, abs=1e-8)
    assert float(g.dual_delta) == pytest.approx(fd["dual_delta"], rel=1e-6, abs=1e-8)


def test_05_limits_and_monotonicity() -> None:
    # T -> 0: intrinsic
    assert float(bsm_price(110, 100, 0.04, 0.01, 0.2, 0.0)) == pytest.approx(10.0)
    assert float(bsm_price(90, 100, 0.04, 0.01, 0.2, 0.0)) == 0.0
    assert float(bsm_price(90, 100, 0.04, 0.01, 0.2, 0.0, "put")) == pytest.approx(10.0)
    g0 = bsm_greeks(110, 100, 0.04, 0.01, 0.2, 0.0)
    assert float(g0.delta) == 1.0 and float(g0.gamma) == 0.0 and float(g0.vega) == 0.0
    # T -> 0 continuity: tiny T close to intrinsic
    assert float(bsm_price(110, 100, 0.04, 0.01, 0.2, 1e-9)) == pytest.approx(10.0, abs=1e-6)
    # sigma -> 0: discounted forward intrinsic
    S, K, r, q, T = 100.0, 100.0, 0.04, 0.015, 5.0
    fwd_intr = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    assert float(bsm_price(S, K, r, q, 0.0, T)) == pytest.approx(fwd_intr, abs=1e-12)
    assert float(bsm_price(S, K, r, q, 1e-9, T)) == pytest.approx(fwd_intr, abs=1e-6)
    gs = bsm_greeks(S, K, r, q, 0.0, T)
    assert float(gs.delta) == pytest.approx(math.exp(-q * T))
    assert float(gs.rho) == pytest.approx(K * T * math.exp(-r * T))
    # deep ITM / OTM stable and non-negative
    assert float(bsm_price(100, 1e-3, r, q, 0.2, T)) == pytest.approx(100 * math.exp(-q * T) - 1e-3 * math.exp(-r * T), rel=1e-12)
    assert float(bsm_price(100, 1e5, r, q, 0.2, T)) >= 0.0
    assert float(bsm_price(100, 1e5, r, q, 0.2, T)) < 1e-12
    # monotonicity
    Ss = np.linspace(50, 200, 151)
    assert np.all(np.diff(bsm_price(Ss, 100, r, q, 0.2, T)) > 0)
    assert np.all(np.diff(bsm_price(Ss, 100, r, q, 0.2, T, "put")) < 0)
    sig = np.linspace(0.01, 1.0, 100)
    assert np.all(np.diff(bsm_price(100, 100, r, q, sig, T)) > 0)
    Ts = np.linspace(0.05, 10, 200)
    assert np.all(np.diff(bsm_price(100, 100, 0.04, 0.0, 0.2, Ts)) > 0)  # q = 0
    Ks = np.linspace(50, 200, 151)
    assert np.all(np.diff(bsm_price(100, Ks, r, q, 0.2, T)) < 0)


def test_06_q_measure_monte_carlo() -> None:
    S, K, r, q, s, T = 100.0, 100.0, 0.04, 0.015, 0.20, 5.0
    rng = np.random.default_rng(12345)
    n = 400_000
    z = rng.standard_normal(n)
    z = np.concatenate([z, -z])  # antithetic
    ST = S * np.exp((r - q - 0.5 * s * s) * T + s * math.sqrt(T) * z)
    pay = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    pairs = 0.5 * (pay[:n] + pay[n:])
    est, se = pairs.mean(), pairs.std(ddof=1) / math.sqrt(n)
    bsm = float(bsm_price(S, K, r, q, s, T))
    assert abs(est - bsm) < 3 * se
    assert se < 0.1


def test_reporting_conventions() -> None:
    g = bsm_greeks(100, 100, 0.04, 0.015, 0.20, 5.0)
    rg = reporting_greeks(g, 100.0, 8.75)
    assert float(rg.dollar_delta) == pytest.approx(float(g.delta) * 875)
    assert float(rg.dollar_gamma) == pytest.approx(float(g.gamma) * 8.75 * 1e4 * 0.01)
    assert float(rg.vega_1pt) == pytest.approx(float(g.vega) * 0.01 * 8.75)
    assert float(rg.theta_day) == pytest.approx(float(g.theta) * 8.75 / 365)
    assert float(rg.rho_1bp) == pytest.approx(float(g.rho) * 1e-4 * 8.75)
    assert float(rg.elasticity) == pytest.approx(float(bsm_elasticity(g, 100.0)))
