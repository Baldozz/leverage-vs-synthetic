"""Spec tests 15, 16, 17 — engine accounting identity and strategy identities.

15. Accounting identity asserted at every step (a tampered component raises AccountingIdentityError).
16. Leverage = 0 → Strategy A identical to Strategy C.
17. Option notional = 0 and no deployment → Strategy B equals illiquids + cash.
"""

from typing import Any

import numpy as np
import pytest
from tests.conftest import make_cfg

from fosim.engine.conventions import accrual_factor
from fosim.engine.simulator import AccountingIdentityError, Simulator
from fosim.engine.state import COMP_IDX
from fosim.market.generator import generate_market_paths


def test_15_identity_holds_default_config_all_strategies(raw: dict[str, Any]) -> None:
    cfg = make_cfg(raw, **{"run.n_paths": 400, "run.horizon_years": 5, "leverage.strategy_d.enabled": True, "spending.pct_nav_pa": 0.01})
    mp = generate_market_paths(cfg)
    res = Simulator(cfg, mp).run()
    assert set(res) == {"A", "B", "C", "D"}
    for r in res.values():
        nav = r.nav
        comp = r.recorder.components.sum(axis=2)
        # explicit re-check of the identity from the recorded arrays
        assert np.allclose(np.diff(nav, axis=1), comp[:, 1:], atol=2e-6)
        assert np.isfinite(nav).all()


def test_15_identity_daily_stepping(raw: dict[str, Any]) -> None:
    # daily stepping needs a daily bootstrap file (a monthly file with dt=daily is rejected by design)
    cfg = make_cfg(raw, **{"run.n_paths": 30, "run.horizon_years": 1, "run.dt": "daily", "bootstrap.file": "data/SPX_SPXFP_aligned_daily.csv", "bootstrap.frequency": "daily"})
    mp = generate_market_paths(cfg)
    res = Simulator(cfg, mp).run()
    for r in res.values():
        comp = r.recorder.components.sum(axis=2)
        assert np.allclose(np.diff(r.nav, axis=1), comp[:, 1:], atol=2e-6)


def test_15_tampering_raises(raw: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """A component booked wrongly by 1 USD must be caught by the identity assertion."""
    cfg = make_cfg(raw, **{"run.n_paths": 5, "run.horizon_years": 1})
    mp = generate_market_paths(cfg)
    sim = Simulator(cfg, mp)
    real_step = Simulator._step

    def tampered_step(self: Simulator, ctx: Any, k: int) -> None:
        ctx.recorder.components[:, k + 1, COMP_IDX["dividends"]] -= 1.0  # pre-book a wrong amount
        real_step(self, ctx, k)

    monkeypatch.setattr(Simulator, "_step", tampered_step)
    with pytest.raises(AccountingIdentityError, match="residual"):
        sim.run()


def test_16_zero_leverage_A_equals_C(raw: dict[str, Any]) -> None:
    cfg = make_cfg(
        raw,
        **{
            "run.n_paths": 200, "run.horizon_years": 4, "leverage.ratio_of_nav": 0.0,
            "illiquids.1.unfunded_commitments": 0, "illiquids.2.unfunded_commitments": 0,
        },
    )
    mp = generate_market_paths(cfg)
    res = Simulator(cfg, mp).run()
    a, c = res["A"], res["C"]
    assert np.array_equal(a.nav, c.nav)
    assert np.array_equal(a.series("cash"), c.series("cash"))
    assert np.array_equal(a.series("held_mv"), c.series("held_mv"))
    assert a.series("loan").max() == 0.0
    assert a.events.margin_calls.sum() == 0


def test_17_zero_notional_B_is_cash_plus_illiquids(raw: dict[str, Any]) -> None:
    cfg = make_cfg(
        raw,
        **{
            "run.n_paths": 50, "run.horizon_years": 3, "options.sizing_mode": "premium_budget", "options.sizing_param": 0.0,
            "dry_powder.enabled": False, "costs.equity_bps": 0, "costs.equity_bps_stressed": 0, "costs.stamp_duty_bps": 0,
        },
    )
    mp = generate_market_paths(cfg)
    res = Simulator(cfg, mp).run()
    b = res["B"]
    assert b.series("n_tranches").max() == 0.0
    assert b.series("option_mv").max() == 0.0
    assert np.allclose(b.nav, b.series("cash") + b.series("illiquid_mv"), atol=1e-6)
    # hand recomputation of the cash account: 625m at inception, simple interest ACT/360 on the
    # start-of-step balance at (r − spread), minus capital calls plus distributions
    days = mp.grid.days
    cash = np.full(mp.n_paths, 625e6)
    il = mp.illiquids
    for k in range(mp.n_steps):
        r = mp.r_short[:, k] - cfg.rates.cash.spread
        cash = cash + cash * r * accrual_factor(days, "ACT/360")
        cash = cash - il.contributions[:, k + 1, :].sum(axis=1) + il.distributions[:, k + 1, :].sum(axis=1)
        assert np.allclose(cash, b.series("cash")[:, k + 1], rtol=1e-12, atol=1e-6)
    assert b.transition_cost == 0.0
