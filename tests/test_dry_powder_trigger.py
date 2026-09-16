"""Dry-powder trigger index: tiers measured on SPX re-arm at SPX's October-2007 high and fire again in 2008;
measured on SPXFP (which never regained its 2000 peak before 2008) they do not."""

from typing import Any

import pandas as pd
import pytest
from tests.conftest import make_cfg

from fosim.runner import run_config

TABLE = {
    "enabled": True, "illiquids": 375e6,
    "A": {"equities": 875e6, "loan": 250e6, "cash": 0.0},
    "B": {"equities": 0.0, "loan": 0.0, "cash": 625e6},
    "C": {"equities": 625e6, "loan": 0.0, "cash": 0.0},
}


@pytest.mark.parametrize("trigger,expect_2008", [("SPX", True), ("SPXFP", False)])
def test_dry_powder_trigger_index(raw: dict[str, Any], trigger: str, expect_2008: bool) -> None:
    cfg = make_cfg(
        raw,
        **{
            "run.n_paths": 1, "run.horizon_years": 9, "run.dt": "monthly",
            "equity_model.type": "historical_replay", "historical.file": "data/SPX_SPXFP_aligned_monthly.csv", "historical.start": "2000-03-31",
            "historical.columns": {"SPXFP": "spxfp", "SPX": "spx_rebased"}, "historical.held_column": "spx_rebased",
            "custom_inception": TABLE,
            "options.ladder_mode": "fixed_notional_schedule", "options.purchase_frequency": "monthly",
            "options.notional_per_purchase": 50e6, "options.target_total_notional": 500e6,
            "dry_powder.enabled": True, "dry_powder.instrument": "call_tranches", "dry_powder.trigger_index": trigger,
            "dry_powder.tiers": [
                {"index_drawdown": -0.20, "deploy_pct_of_deployable": 0.33},
                {"index_drawdown": -0.30, "deploy_pct_of_deployable": 0.5},
                {"index_drawdown": -0.40, "deploy_pct_of_deployable": 1.0},
            ],
        },
    )
    out = run_config(cfg, ledger_paths=[0])
    dates = pd.read_csv("data/SPX_SPXFP_aligned_monthly.csv", parse_dates=["date"])
    dates = dates[dates.date >= "2000-03-31"]["date"].reset_index(drop=True)
    led = out.results["B"].ledger.to_frame()
    buys = led[(led.account == "cash") & led.description.str.contains("dry powder tier") & led.description.str.contains("premium")]
    after_2007 = [dates[int(k)] for k in buys.step if dates[int(k)] > pd.Timestamp("2007-10-31")]
    assert (len(after_2007) > 0) == expect_2008, after_2007
    assert len(buys) > 0  # the tiers fire in 2001-02 under both definitions
