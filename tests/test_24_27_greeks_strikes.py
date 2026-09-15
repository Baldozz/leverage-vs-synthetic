"""Spec tests 24–27 and 30–31: 5y ATM Greeks, instantaneous shocks, forward-ATM and delta-target
strikes, smile delta, implied vol inversion.

Reference inputs: S = K = 100, r = 4%, q = 1.5%, σ = 20%, T = 5, flat vol, continuous compounding.
"""

import math

import numpy as np
import pytest

from fosim.pricing.black_scholes import bsm_elasticity, bsm_greeks, bsm_price, forward_price
from fosim.pricing.greeks import fd_smile_delta, smile_delta
from fosim.pricing.implied_vol import ArbitrageBoundError, implied_vol
from fosim.pricing.vol_surface import solve_delta_target_strike

S, K, R, Q, SIG, T = 100.0, 100.0, 0.04, 0.015, 0.20, 5.0


def test_24_greeks_5y_atm() -> None:
    g = bsm_greeks(S, K, R, Q, SIG, T)
    assert float(g.delta) == pytest.approx(0.6425, abs=5e-4)
    assert float(g.gamma) == pytest.approx(0.00729, abs=5e-6)  # per index point
    assert float(g.vega) * 0.01 == pytest.approx(0.729, abs=1e-3)  # per vol point, per 100 notional
    assert float(g.theta) == pytest.approx(-2.205, abs=1e-3)  # per year
    assert float(g.rho) * 0.01 == pytest.approx(2.138, abs=1e-3)  # per 100 bp
    assert float(g.dq) * 0.01 == pytest.approx(-3.213, abs=1e-3)  # per 100 bp
    assert float(bsm_elasticity(g, S)) == pytest.approx(2.99, abs=0.01)


@pytest.mark.parametrize(
    "spot,sig,exp_delta,exp_value",
    [
        (60.0, 0.20, 0.242, None),
        (70.0, 0.20, 0.356, 6.19),
        (110.0, 0.20, 0.708, None),
        (130.0, 0.20, 0.800, None),
        (70.0, 0.30, 0.460, 11.91),
    ],
)
def test_25_instantaneous_shocks(spot: float, sig: float, exp_delta: float, exp_value: float | None) -> None:
    g = bsm_greeks(spot, K, R, Q, sig, T)
    assert float(g.delta) == pytest.approx(exp_delta, abs=1e-3)
    if exp_value is not None:
        assert float(g.price) == pytest.approx(exp_value, abs=0.01)


def test_26_forward_atm_strike() -> None:
    F = float(forward_price(S, R, Q, T))
    assert F == pytest.approx(113.31, abs=0.01)
    g = bsm_greeks(S, F, R, Q, SIG, T)
    assert float(g.price) == pytest.approx(16.42, abs=0.01)
    assert float(g.delta) == pytest.approx(0.546, abs=1e-3)


def test_27_delta_target_flat_vol() -> None:
    K50 = solve_delta_target_strike(S, R, Q, T, 0.50, lambda _K: SIG)
    assert K50 == pytest.approx(119.87, abs=0.01)
    assert float(bsm_price(S, K50, R, Q, SIG, T)) == pytest.approx(14.34, abs=0.01)
    assert float(bsm_greeks(S, K50, R, Q, SIG, T).delta) == pytest.approx(0.5, abs=1e-10)


def test_27_delta_target_with_skew_bsm_and_smile() -> None:
    psi = -0.10 / math.sqrt(T)
    F = float(forward_price(S, R, Q, T))

    def vol_of_K(k_: float) -> float:
        return max(SIG + psi * math.log(k_ / F), 0.05)

    K_bsm = solve_delta_target_strike(S, R, Q, T, 0.50, vol_of_K, "bsm")
    assert float(bsm_greeks(S, K_bsm, R, Q, vol_of_K(K_bsm), T).delta) == pytest.approx(0.5, abs=1e-8)
    K_sm = solve_delta_target_strike(S, R, Q, T, 0.50, vol_of_K, "smile", psi_T=psi)
    d_sm = float(smile_delta(S, K_sm, R, Q, vol_of_K(K_sm), T, psi))
    assert d_sm == pytest.approx(0.5, abs=1e-8)
    assert K_bsm != pytest.approx(K_sm, abs=1e-3)  # the two definitions differ with skew on


def test_30_smile_delta() -> None:
    # psi = 0 -> smile delta == BSM delta
    g = bsm_greeks(S, 110.0, R, Q, SIG, T)
    assert float(smile_delta(S, 110.0, R, Q, SIG, T, 0.0)) == pytest.approx(float(g.delta), abs=1e-14)
    # sticky strike -> smile delta == BSM delta even with psi != 0
    assert float(smile_delta(S, 110.0, R, Q, SIG, T, -0.05, sticky="sticky_strike")) == pytest.approx(float(g.delta))
    # sticky moneyness: matches FD delta with the surface vol re-evaluated at the bumped spot
    psi = -0.10 / math.sqrt(T)
    Kx = 110.0
    sigma_atm = 0.21

    def vol_of_spot(s_: float) -> float:
        F_ = s_ * math.exp((R - Q) * T)
        return sigma_atm + psi * math.log(Kx / F_)

    analytic = float(smile_delta(S, Kx, R, Q, vol_of_spot(S), T, psi))
    fd = fd_smile_delta(S, Kx, R, Q, T, vol_of_spot)
    assert analytic == pytest.approx(fd, abs=1e-7)
    assert analytic != pytest.approx(float(bsm_greeks(S, Kx, R, Q, vol_of_spot(S), T).delta), abs=1e-4)


def test_31_implied_vol_inversion_grid_and_bounds() -> None:
    n_vol_checked = 0
    for T_ in (0.25, 1.0, 5.0, 10.0):
        for m in (0.6, 0.8, 1.0, 1.2, 1.5):
            for sig in (0.08, 0.2, 0.45):
                K_ = m * S
                for kind in ("call", "put"):
                    px = float(bsm_price(S, K_, R, Q, sig, T_, kind))  # type: ignore[arg-type]
                    lo = max(S * math.exp(-Q * T_) - K_ * math.exp(-R * T_), 0.0)
                    if kind == "put":
                        lo = max(K_ * math.exp(-R * T_) - S * math.exp(-Q * T_), 0.0)
                    iv = implied_vol(px, S, K_, R, Q, T_, kind)  # type: ignore[arg-type]
                    if px - lo > 1e-10:
                        # time value is identifiable: sigma recovered to 1e-8
                        assert iv == pytest.approx(sig, abs=1e-8), (T_, m, sig, kind)
                        n_vol_checked += 1
                    else:
                        # time value below double precision (deep ITM, short T, low vol): sigma is not
                        # identifiable from the price; the inversion must still reprice exactly
                        assert float(bsm_price(S, K_, R, Q, iv, T_, kind)) == pytest.approx(px, abs=1e-12)  # type: ignore[arg-type]
    assert n_vol_checked >= 100
    upper = S * math.exp(-Q * T)
    lower = max(S * math.exp(-Q * T) - K * math.exp(-R * T), 0.0)
    with pytest.raises(ArbitrageBoundError):
        implied_vol(upper + 1e-6, S, K, R, Q, T)
    with pytest.raises(ArbitrageBoundError):
        implied_vol(lower - 1e-6, S, K, R, Q, T)
    with pytest.raises(ArbitrageBoundError):
        implied_vol(-1.0, S, K, R, Q, T)
    assert implied_vol(lower, S, K, R, Q, T) == 0.0


def test_vectorised_pricing_shapes() -> None:
    Ss = np.linspace(50, 150, 7).reshape(7, 1)
    Ks = np.array([90.0, 100.0, 110.0]).reshape(1, 3)
    p = bsm_price(Ss, Ks, R, Q, SIG, T)
    assert p.shape == (7, 3)
    g = bsm_greeks(Ss, Ks, R, Q, SIG, T)
    assert g.delta.shape == (7, 3)
