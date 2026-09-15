"""Spec tests 32 and 40 (surface part): surface loader arbitrage detection; dealer-quote precedence.

Also: parametric term structure / skew mechanics and grid interpolation in total variance.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fosim.config import DEFAULT_CONFIG_PATH, SimConfig
from fosim.pricing.black_scholes import bsm_price, forward_price
from fosim.pricing.vol_surface import ArbitrageViolation, GridSurface, ParametricSurface, VolSource


@pytest.fixture
def cfg() -> SimConfig:
    with open(DEFAULT_CONFIG_PATH) as f:
        return SimConfig.model_validate(yaml.safe_load(f))


def _write_surface(path: Path, rows: list[tuple[float, float, float]]) -> Path:
    pd.DataFrame(rows, columns=["tenor", "moneyness", "iv"]).to_csv(path, index=False)
    return path


def test_parametric_term_structure_and_skew(cfg: SimConfig) -> None:
    ps = ParametricSurface.from_config(cfg.implied_vol)
    # at IV_s = theta_s the ATM level equals theta_T
    assert float(ps.atm_vol(5.0, 0.18)) == pytest.approx(0.21)
    assert float(ps.atm_vol(1.0, 0.18)) == pytest.approx(0.19)
    assert float(ps.atm_vol(3.5, 0.18)) == pytest.approx(0.205)  # linear in T
    # elasticity: 5y beta = 0.35; IV_s doubles -> level * 2^0.35
    assert float(ps.atm_vol(5.0, 0.36)) == pytest.approx(0.21 * 2**0.35)
    # skew: psi(T) = psi_1y / sqrt(T); OTM call (k > 0) has lower vol
    assert float(ps.psi_T(4.0)) == pytest.approx(-0.10 / 2)
    theta_4y = 0.20 + (4.0 - 2.0) / (5.0 - 2.0) * 0.01
    assert float(ps.vol(0.1, 4.0, 0.18)) == pytest.approx(theta_4y + (-0.05) * 0.1)
    # floor
    assert float(ps.vol(5.0, 4.0, 0.18)) == pytest.approx(cfg.implied_vol.vol_floor)


def test_32_calendar_arbitrage_detected(tmp_path: Path, cfg: SimConfig) -> None:
    # total variance must be non-decreasing in T: 1y at 30% (w=0.09) vs 2y at 20% (w=0.08) -> violation
    rows = []
    for m in (0.8, 1.0, 1.2):
        rows += [(1.0, m, 0.30), (2.0, m, 0.20), (5.0, m, 0.21)]
    p = _write_surface(tmp_path / "cal.csv", rows)
    gs = GridSurface.from_csv(p)
    assert gs.calendar_violations, "calendar arbitrage not detected"
    assert not gs.butterfly_violations
    pc = cfg.pricing.model_copy(update={"source": "vol_surface_file", "vol_surface_file": str(p)})
    with pytest.raises(ArbitrageViolation, match="calendar"):
        VolSource.build(cfg.implied_vol, pc, "SPX", 100.0, 0.04, 0.015, 0.18)
    pw = pc.model_copy(update={"arbitrage_check": "warn"})
    with pytest.warns(UserWarning, match="arbitrage"):
        VolSource.build(cfg.implied_vol, pw, "SPX", 100.0, 0.04, 0.015, 0.18)


def test_32_butterfly_arbitrage_detected(tmp_path: Path) -> None:
    # a violent smile kink: vol spikes at one strike -> negative implied density
    rows = [(1.0, 0.8, 0.20), (1.0, 0.95, 0.20), (1.0, 1.0, 0.60), (1.0, 1.05, 0.20), (1.0, 1.2, 0.20)]
    gs = GridSurface.from_csv(_write_surface(tmp_path / "bfly.csv", rows))
    assert gs.butterfly_violations, "butterfly arbitrage not detected"
    # a clean smile has no violations
    rows_ok = [(t, m, 0.20 - 0.10 / math.sqrt(t) * math.log(m)) for t in (1.0, 2.0, 5.0) for m in (0.7, 0.85, 1.0, 1.15, 1.3)]
    gs_ok = GridSurface.from_csv(_write_surface(tmp_path / "ok.csv", rows_ok))
    assert not gs_ok.has_arbitrage


def test_grid_total_variance_interpolation(tmp_path: Path) -> None:
    rows = [(t, m, v) for t, v in ((1.0, 0.20), (3.0, 0.25)) for m in (0.8, 1.0, 1.2)]
    gs = GridSurface.from_csv(_write_surface(tmp_path / "g.csv", rows))
    # at T = 2: w = 0.04*1 + 0.5*(0.1875 - 0.04) -> sigma = sqrt(w/2)
    w = 0.04 + 0.5 * (0.25**2 * 3 - 0.04)
    assert float(gs.vol(0.0, 2.0)) == pytest.approx(math.sqrt(w / 2.0))
    assert float(gs.vol(0.0, 1.0)) == pytest.approx(0.20)
    assert float(gs.vol(0.0, 3.0)) == pytest.approx(0.25)
    # flat extrapolation in tenor keeps sigma at the boundary
    assert float(gs.vol(0.0, 10.0)) == pytest.approx(0.25)


def test_40_dealer_quote_overrides_surface(cfg: SimConfig) -> None:
    S0, r0, q0, T = 100.0, 0.04, 0.015, 5.0
    # dealer quotes a 5y ATM(spot) call at 24% mid vol (bid 23.5 / ask 24.5) as premium %
    K = S0
    bid = float(bsm_price(S0, K, r0, q0, 0.235, T)) / S0
    ask = float(bsm_price(S0, K, r0, q0, 0.245, T)) / S0
    from fosim.config.schema import PricingConfig

    pc = PricingConfig.model_validate(
        {
            **cfg.pricing.model_dump(),
            "source": "dealer_quotes",
            "dealer_quotes": [
                {"index": "SPX", "tenor": 5.0, "strike_pct_spot": 1.0, "bid_pct": bid, "ask_pct": ask}
            ],
        }
    )
    vs = VolSource.build(cfg.implied_vol, pc, "SPX", S0, r0, q0, 0.18)
    F = float(forward_price(S0, r0, q0, T))
    k = math.log(K / F)
    sig, src = vs.vol(k, T, 0.18, month=0)
    assert src == "dealer_quote"
    assert float(sig) == pytest.approx(0.24, abs=1e-6)
    # half spread from the quote
    hs = vs.half_spread(T, k, 0.18, "new")
    assert float(hs) == pytest.approx(0.005, abs=1e-6)
    # reconciliation table populated and the parametric level calibrated at the quoted tenor
    assert len(vs.reconciliation) == 1
    rec = vs.reconciliation[0]
    assert rec.quote_mid_iv == pytest.approx(0.24, abs=1e-6)
    assert rec.shift_applied == pytest.approx(0.24 - rec.model_iv_before, abs=1e-9)
    # a different strike at the same tenor uses the calibrated parametric surface
    sig2, src2 = vs.vol(k + 0.2, T, 0.18, month=0)
    assert src2 == "parametric_quote_calibrated"
    assert float(sig2) == pytest.approx(0.24 + vs.parametric.psi_T(T) * 0.2, abs=1e-6)
    # a vector of moneyness (paths) still works
    sig3, _ = vs.vol(np.array([k, k + 0.1]), T, np.array([0.18, 0.18]), month=3)
    assert sig3.shape == (2,)


def test_scenario_override_precedence(cfg: SimConfig) -> None:
    from fosim.config.schema import PricingConfig

    pc = PricingConfig.model_validate(
        {**cfg.pricing.model_dump(), "source": "scenario_overrides", "scenario_overrides": [{"index": "SPX", "month_from": 6, "month_to": 12, "tenor": 5.0, "iv": 0.35}]}
    )
    vs = VolSource.build(cfg.implied_vol, pc, "SPX", 100.0, 0.04, 0.015, 0.18)
    s_in, src_in = vs.vol(0.0, 5.0, 0.18, month=8)
    assert src_in == "scenario_override" and float(s_in) == pytest.approx(0.35)
    s_out, src_out = vs.vol(0.0, 5.0, 0.18, month=13)
    assert src_out == "parametric" and float(s_out) == pytest.approx(0.21)
    s_t, src_t = vs.vol(0.0, 2.0, 0.18, month=8)  # different tenor untouched
    assert src_t == "parametric" and float(s_t) == pytest.approx(0.20)


def test_bid_ask_stress_multiplier(cfg: SimConfig) -> None:
    vs = VolSource.build(cfg.implied_vol, cfg.pricing, "SPX", 100.0, 0.04, 0.015, 0.18)
    calm = vs.half_spread(5.0, None, np.array([0.18]), "new")
    stressed = vs.half_spread(5.0, None, np.array([0.40]), "new")
    assert float(calm[0]) == pytest.approx(0.0025)
    assert float(stressed[0]) == pytest.approx(0.0025 * 3.0)
    assert float(vs.half_spread(5.0, None, np.array([0.18]), "unwind")[0]) == pytest.approx(0.00375)
