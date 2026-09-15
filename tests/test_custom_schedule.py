"""Custom inception balance sheets and the fixed-notional (weekly) option purchase schedule.

* custom_inception: each strategy starts with exactly the entered equities / loan / cash; illiquids common.
* fixed_notional_schedule, weekly: one purchase of `notional_per_purchase` at every weekly step until the
  active book reaches `target_total_notional`; nothing more until tranches expire; the accounting
  identity holds throughout; A's margin logic still sees no lending value from options.
"""

from typing import Any

import numpy as np
import pytest
from tests.conftest import make_cfg

from fosim.runner import run_config


def _custom(raw: dict[str, Any], **extra: Any) -> Any:
    return make_cfg(
        raw,
        **{
            "run.n_paths": 1, "run.horizon_years": 3, "run.dt": "weekly",
            "equity_model.type": "historical_replay",
            "historical.file": "data/SPX_SPXFP_aligned_weekly.csv", "historical.start": "2010-01-08",
            "historical.columns": {"SPXFP": "spxfp", "SPX": "spx_rebased"}, "historical.held_column": "spx_rebased",
            "custom_inception": {
                "enabled": True, "illiquids": 300e6,
                "A": {"equities": 900e6, "loan": 200e6, "cash": 0.0},
                "B": {"equities": 100e6, "loan": 0.0, "cash": 600e6},
                "C": {"equities": 700e6, "loan": 0.0, "cash": 0.0},
                "reference_equity_exposure": 900e6,
            },
            "options.ladder_mode": "fixed_notional_schedule", "options.purchase_frequency": "weekly",
            "options.notional_per_purchase": 20e6, "options.target_total_notional": 500e6,
            "dry_powder.enabled": False, "costs.equity_bps": 0, "pricing.bid_ask_vol_pts_new": 0.0, "pricing.bid_ask_vol_pts_unwind": 0.0,
            **extra,
        },
    )


def test_custom_inception_balances(raw: dict[str, Any]) -> None:
    cfg = _custom(raw)
    out = run_config(cfg, ledger_paths=[0])
    a, b, c = out.results["A"], out.results["B"], out.results["C"]
    assert a.inception["equity"] == pytest.approx(900e6) and a.inception["loan"] == pytest.approx(200e6)
    assert a.nav[0, 0] == pytest.approx(900e6 + 300e6 - 200e6)
    assert c.nav[0, 0] == pytest.approx(700e6 + 300e6)
    assert b.series("held_mv")[0, 0] == pytest.approx(100e6)
    # B's NAV at t0 = 100 equities + 600 cash + 300 illiquids (the first weekly tranche is bought at t0 at mid: NAV-neutral)
    assert b.nav[0, 0] == pytest.approx(1000e6, rel=1e-9)
    assert b.transition_cost == 0.0
    assert out.paths.illiquids.nav_true[0, 0, :].sum() == pytest.approx(300e6)


def test_weekly_fixed_notional_schedule(raw: dict[str, Any]) -> None:
    cfg = _custom(raw)
    out = run_config(cfg, ledger_paths=[0])
    b = out.results["B"]
    tr = b.ledger.tranche_frame()
    n = b.series("n_tranches")[0]
    # 500 / 20 = 25 purchases: one per weekly step from step 0, then the book is full
    assert n[:25].tolist() == list(range(1, 26))
    assert (n[25:] == 25).all()  # 5y tenor > 3y horizon: nothing expires, nothing more is bought
    purchases = tr.groupby("slot")["purchase_step"].min().sort_values().tolist()
    assert purchases == list(range(25))
    per = tr.groupby("slot")["notional_usd"].max()
    assert np.allclose(per.to_numpy(), 20e6)
    # strikes = SPXFP spot at purchase (ATM), expiry 5 years = 260 weekly steps later
    for _, row in tr.drop_duplicates("slot").iterrows():
        k = int(row.purchase_step)
        assert row.strike == pytest.approx(out.paths.S[0, k, 0], rel=1e-12)
        assert int(row.expiry_step) == k + 260
    # book notional at the target
    assert tr[tr.step == 30]["notional_usd"].sum() == pytest.approx(500e6)
    assert b.events.option_purchases[0] == 25
    # identity re-check on the recorded arrays
    comp = b.recorder.components[0].sum(axis=1)
    assert np.allclose(np.diff(b.nav[0]), comp[1:], atol=2e-6)


def test_monthly_frequency_and_validation(raw: dict[str, Any]) -> None:
    cfg = _custom(raw, **{"options.purchase_frequency": "monthly"})
    out = run_config(cfg, ledger_paths=[0])
    n = out.results["B"].series("n_tranches")[0]
    # purchases only at month-end steps (t0 counts): 25 tranches take ~24 months on a weekly grid
    assert n[-1] == 25 and n[4] <= 2
    with pytest.raises(Exception, match="weekly option purchases need"):
        make_cfg(raw, **{"options.ladder_mode": "fixed_notional_schedule", "options.purchase_frequency": "weekly", "options.notional_per_purchase": 1e6, "options.target_total_notional": 1e7})
    with pytest.raises(Exception, match="fixed_notional_schedule requires"):
        make_cfg(raw, **{"options.ladder_mode": "fixed_notional_schedule"})


def test_dry_powder_into_calls_sits_on_top_of_the_schedule(raw: dict[str, Any]) -> None:
    """With instrument=call_tranches the deployments buy extra ATM tranches (origin 2) that do not count toward the
    scheduled book's target, and the schedule keeps its own tranches at the target."""
    cfg = _custom(
        raw,
        **{
            "historical.start": "2007-10-31", "run.horizon_years": 3, "run.dt": "monthly", "historical.file": "data/SPX_SPXFP_aligned_monthly.csv",
            "options.purchase_frequency": "monthly", "options.notional_per_purchase": 50e6, "options.target_total_notional": 500e6,
            "dry_powder.enabled": True, "dry_powder.instrument": "call_tranches",
            "dry_powder.tiers": [{"index_drawdown": -0.20, "deploy_pct_of_deployable": 0.3}, {"index_drawdown": -0.40, "deploy_pct_of_deployable": 0.5}],
        },
    )
    out = run_config(cfg, ledger_paths=[0])
    b = out.results["B"]
    tr = b.ledger.tranche_frame()
    assert b.events.dry_powder_deployments[0] == 2 and b.events.dry_powder_deployed_usd[0] > 0
    last = tr[tr.step == out.paths.n_steps]
    sched = last[last.origin == 0]["notional_usd"].sum()
    extra = last[last.origin == 2]["notional_usd"].sum()
    assert sched == pytest.approx(500e6, rel=1e-9)  # the schedule still fills its target
    assert extra > 0  # plus the dry-powder tranches on top
    led = b.ledger.to_frame()
    assert led.description.str.contains("dry powder tier").any()
    # no spot was bought
    assert b.series("spot_mv").max() == 0.0
