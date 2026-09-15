"""Spec tests 36–40 (engine level).

36. rebalance_bands: after execution the book delta is within the band (or cash-limit binding flagged);
    transaction costs appear in the ledger.
37. futures_overlay: variation margin flows equal futures P&L each step; B's margin statistics populated.
38. Held portfolio with β = 1, tracking error 0, α = 0 is identical to holding the option index.
39. total_return underlying: pricing uses q = 0 and A receives no dividends.
40. Price-source precedence: a dealer quote overrides the surface at its tenor/strike and the tranche table shows the source.
"""

import math
from typing import Any

import numpy as np
import pytest
from tests.conftest import make_cfg

from fosim.engine.simulator import Simulator
from fosim.market.generator import generate_market_paths
from fosim.pricing.black_scholes import bsm_price


def test_36_rebalance_bands(raw: dict[str, Any]) -> None:
    cfg = make_cfg(raw, **{"run.n_paths": 60, "run.horizon_years": 3, "exposure.policy": "rebalance_bands", "exposure.band_pp": 0.10, "dry_powder.enabled": False})
    mp = generate_market_paths(cfg)
    res = Simulator(cfg, mp, ledger_paths=[0]).run()
    b = res["B"]
    target = 875e6
    expo = b.series("equity_exposure")
    alive = ~b.events.ruin
    constrained = b.events.cash_constrained_purchases > 0
    for k in np.flatnonzero(mp.grid.is_month_end | (np.arange(mp.n_steps + 1) == 0)):
        dev = np.abs(expo[:, k] - target) / target
        ok = (dev <= 0.10 + 1e-9) | constrained | ~alive
        assert ok.all(), (k, dev[~ok])
    led = b.ledger.to_frame()
    assert led["description"].str.contains("rebalance_bands").any()
    assert led[led["account"] == "memo:bid_ask_cost"]["amount_usd"].lt(0).any()
    assert b.events.option_purchases[0] > 12


def test_37_futures_overlay(raw: dict[str, Any]) -> None:
    cfg = make_cfg(
        raw,
        **{"run.n_paths": 40, "run.horizon_years": 3, "exposure.policy": "futures_overlay", "exposure.futures_overlay": {"enabled": True, "initial_margin_pct": 0.10}, "dry_powder.enabled": False},
    )
    mp = generate_market_paths(cfg)
    res = Simulator(cfg, mp).run()
    b = res["B"]
    from fosim.engine.state import COMP_IDX

    fut = b.series("futures_notional")
    vm = b.recorder.components[:, :, COMP_IDX["futures_pnl"]]
    S = mp.S[:, :, 0]
    units = fut / S
    # VM_{k+1} = units_k (S_{k+1} − S_k)
    expected = units[:, :-1] * (S[:, 1:] - S[:, :-1])
    assert np.allclose(vm[:, 1:], expected, atol=1e-3)
    assert np.abs(fut).max() > 0
    assert b.events.futures_margin_calls.shape == (40,)
    assert np.all(b.events.futures_margin_calls >= 0)
    # after each month-end the total exposure equals the target
    expo = b.series("equity_exposure")
    for k in np.flatnonzero(mp.grid.is_month_end):
        assert np.allclose(expo[~b.events.ruin, k], 875e6, rtol=1e-9)


def test_38_held_portfolio_beta_one(raw: dict[str, Any]) -> None:
    cfg = make_cfg(raw, **{"run.n_paths": 100, "run.horizon_years": 3, "held_equity_portfolio.tracking_error_vol": 0.0, "held_equity_portfolio.expected_alpha": 0.0})
    mp = generate_market_paths(cfg)
    hi = cfg.index_names.index("SPX")  # the index the held book has beta 1 to
    assert np.allclose(mp.held / mp.held[:, :1], mp.S[:, :, hi] / mp.S[:, :1, hi], rtol=1e-12)
    res = Simulator(cfg, mp).run()
    a = res["A"]
    # A's equity leg tracks the index exactly: held_mv / held_mv0 == S / S0
    ratio = a.series("held_mv") / a.series("held_mv")[:, :1]
    idx = mp.S[:, :, hi] / mp.S[:, :1, hi]
    m = a.series("held_mv") > 0
    # only before any margin sale (units constant): compare on paths without margin events
    clean = a.events.margin_calls == 0
    assert np.allclose(ratio[clean], idx[clean], rtol=1e-12)
    # with alpha, the held portfolio outperforms by e^{alpha t} in expectation (deterministic check at zero TE)
    cfg2 = make_cfg(raw, **{"run.n_paths": 100, "run.horizon_years": 3, "held_equity_portfolio.tracking_error_vol": 0.0, "held_equity_portfolio.expected_alpha": 0.02})
    mp2 = generate_market_paths(cfg2)
    assert np.allclose(mp2.held[:, -1] / mp2.held[:, 0], idx[:, -1] * math.exp(0.02 * 3), rtol=1e-12)
    del m


def test_39_total_return_underlying(raw: dict[str, Any]) -> None:
    cfg = make_cfg(
        raw,
        **{
            "run.n_paths": 20, "run.horizon_years": 2,
            "equity_indices.0.underlying_type": "total_return", "equity_indices.0.pricing_dividend_curve": {"tenors": [1, 5], "q": [0.0, 0.0]},
            "equity_indices.1.underlying_type": "total_return", "equity_indices.1.pricing_dividend_curve": {"tenors": [1, 5], "q": [0.0, 0.0]},
        },
    )
    mp = generate_market_paths(cfg)
    sim = Simulator(cfg, mp)
    assert sim._q_fns[0](5.0) == 0.0
    res = sim.run()
    from fosim.engine.state import COMP_IDX

    assert np.all(res["A"].recorder.components[:, :, COMP_IDX["dividends"]] == 0.0)
    # price index drift includes the dividend for a TR index: q_px = 0
    assert mp.index_params[0].q_c_px == 0.0
    # with price_return the same seed pays dividends
    cfg_pr = make_cfg(raw, **{"run.n_paths": 20, "run.horizon_years": 2})
    res_pr = Simulator(cfg_pr, generate_market_paths(cfg_pr)).run()
    assert res_pr["A"].recorder.components[:, 1:, COMP_IDX["dividends"]].min() > 0


def test_40_dealer_quote_in_tranche_table(raw: dict[str, Any]) -> None:
    S0, r0, q0, T = 100.0, 0.04, 0.015, 5.0
    bid = float(bsm_price(S0, S0, r0, q0, 0.235, T)) / S0
    ask = float(bsm_price(S0, S0, r0, q0, 0.245, T)) / S0
    cfg = make_cfg(
        raw,
        **{
            "run.n_paths": 3, "run.horizon_years": 2, "options.ladder_mode": "bullet", "dry_powder.enabled": False,
            "pricing.source": "dealer_quotes",
            "pricing.dealer_quotes": [{"index": "SPXFP", "tenor": 5.0, "strike_pct_spot": 1.0, "bid_pct": bid, "ask_pct": ask}],
            "equity_indices.0.underlying_type": "price_return", "equity_indices.0.pricing_dividend_curve": {"tenors": [1, 5], "q": [0.015, 0.015]},
            "implied_vol.short.iv0": 0.18,
        },
    )
    mp = generate_market_paths(cfg)
    res = Simulator(cfg, mp, ledger_paths=[0]).run()
    tr = res["B"].ledger.tranche_frame()
    t0 = tr[(tr["step"] == 0) & (tr["path"] == 0)]
    assert t0["source"].iloc[0] == "dealer_quote"
    assert float(t0["vol_at_purchase_ask"].iloc[0]) == pytest.approx(0.245, abs=1e-6)
    # premium paid = quoted ask
    assert res["B"].inception["option_premium0"] == pytest.approx(875e6 * ask, rel=1e-9)
    # later marks come from the calibrated parametric surface (moneyness has moved)
    later = tr[(tr["step"] == 12) & (tr["path"] == 0)]
    assert later["source"].iloc[0] in ("dealer_quote", "parametric_quote_calibrated")
