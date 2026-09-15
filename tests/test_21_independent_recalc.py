"""Spec test 21: deterministic zero-vol scenario for A, B and C recomputed by an independent minimal
implementation (validation/independent_recalc.py, sharing no code with the engine); engine NAVs
must match to 1e-8 (relative) at every step.
"""

from typing import Any

import numpy as np
from tests.conftest import make_cfg, zero_vol_overrides
from validation.independent_recalc import default_scenario, run

from fosim.engine.simulator import Simulator
from fosim.market.generator import generate_market_paths


def test_21_engine_matches_independent_recalc(raw: dict[str, Any]) -> None:
    sigma = 0.20
    cfg = make_cfg(
        raw,
        **zero_vol_overrides(**{
            "run.n_paths": 2, "run.horizon_years": 3, "options.ladder_mode": "bullet", "dry_powder.enabled": False,
            "implied_vol.short.theta": sigma, "implied_vol.short.iv0": sigma, "implied_vol.short.floor": 0.01, "implied_vol.short.cap": 2.0,
            "implied_vol.term_structure": {"tenors": [1, 2, 5], "theta": [sigma, sigma, sigma], "beta": [0.0, 0.0, 0.0]},
            "implied_vol.skew_1y": 0.0,
            "rates.usd.r0": 0.04, "rates.cash.spread": 0.001, "loan_terms.base_floor": 0.0,
            "loan_terms.spread_tiers": [{"utilisation_below": 1.0, "spread": 0.01}],
            "equity_indices.0.expected_return": 0.03, "equity_indices.0.underlying_type": "excess_return",
            "equity_indices.1.expected_return": 0.07, "equity_indices.1.dividend_yield": 0.015, "equity_indices.1.dividend_wht": 0.15,
            "held_equity_portfolio.beta_to_option_book": 1.0,
            "illiquids.1.ta_model.crisis": None, "illiquids.2.ta_model.crisis": None,
        }),
    )
    mp = generate_market_paths(cfg)
    res = Simulator(cfg, mp).run()
    ref = run(default_scenario(36))
    # market path agrees
    assert np.allclose(mp.S[0, :, 1], ref["index"], rtol=1e-12)  # SPX (held)
    assert np.allclose(mp.S[0, :, 0], ref["index_opt"], rtol=1e-12)  # SPXFP (option underlying)
    assert np.allclose(mp.held[0] / mp.held[0, 0], np.asarray(ref["index"]) / 100.0, rtol=1e-12)
    assert np.allclose(mp.illiquids.nav_reported[0].sum(axis=1), ref["illiquid_official"], rtol=1e-10)
    for name in ("A", "B", "C"):
        eng = res[name].nav[0]
        ind = np.asarray(ref[name])
        rel = np.abs(eng - ind) / np.abs(ind)
        assert rel.max() < 1e-8, (name, int(rel.argmax()), eng[rel.argmax()], ind[rel.argmax()])
        # both paths identical (zero vol)
        assert np.array_equal(res[name].nav[0], res[name].nav[1])
    # sanity on the economics of the deterministic case: A carries positive leverage in a rising market
    assert ref["A"][-1] > ref["C"][-1]
