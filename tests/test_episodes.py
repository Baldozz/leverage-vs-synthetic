"""Drawdown episodes, trough snapshots, risk table and return distributions on known inputs."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_cfg

from fosim.analytics.episodes import (
    drawdown_episodes,
    episode_table,
    return_distribution,
    risk_table,
    rolling_vol,
    strategy_snapshot,
)
from fosim.runner import run_config


def test_drawdown_episodes_synthetic() -> None:
    lvl = np.array([100, 110, 100, 90, 80, 95, 111, 112, 100, 112, 113, 60, 70, 80], dtype=float)
    eps = drawdown_episodes(lvl, threshold=-0.15)
    # episode 1: peak 110 (idx 1) -> trough 80 (idx 4), recovered at idx 6 (111); the 112->100 dip (-10.7 %) is below threshold;
    # episode 2: peak 113 (idx 10) -> trough 60 (idx 11), not recovered
    assert len(eps) == 2
    assert eps.iloc[0].peak_idx == 1 and eps.iloc[0].trough_idx == 4 and eps.iloc[0].recovery_idx == 6 and eps.iloc[0].depth == pytest.approx(80 / 110 - 1)
    assert eps.iloc[1].peak_idx == 10 and eps.iloc[1].trough_idx == 11 and pd.isna(eps.iloc[1].recovery_idx) and not eps.iloc[1].recovered


def test_return_distribution_and_rolling_vol() -> None:
    rng = np.random.default_rng(3)
    r = rng.normal(0.0003, 0.01, 5000)
    lvl = 100 * np.exp(np.cumsum(r))
    d = return_distribution(lvl, 1)
    assert d["n"] == 4999 and abs(d["var95"] - (-np.percentile(np.diff(lvl) / lvl[:-1], 5))) < 1e-12
    assert d["cvar99"] >= d["var99"] >= d["var95"] > 0
    d3 = return_distribution(lvl, 756)
    assert d3["n"] == 5000 - 756
    rv = rolling_vol(lvl, 252, 252)
    assert np.isnan(rv[:252]).all() and abs(np.nanmean(rv[300:]) - 0.01 * np.sqrt(252)) < 0.02


def test_snapshot_and_risk_table_on_replay(raw: dict[str, Any]) -> None:
    cfg = make_cfg(
        raw,
        **{
            "run.n_paths": 1, "run.horizon_years": 4, "equity_model.type": "historical_replay",
            "historical.file": "data/SPX_SPXFP_aligned_monthly.csv", "historical.start": "2007-10-31",
            "historical.columns": {"SPXFP": "spxfp", "SPX": "spx_rebased"}, "historical.held_column": "spx_rebased", "dry_powder.enabled": False,
        },
    )
    out = run_config(cfg, ledger_paths=[0])
    dates = pd.read_csv("data/SPX_SPXFP_aligned_monthly.csv", parse_dates=["date"])
    dates = dates[dates.date >= "2007-10-31"]["date"].reset_index(drop=True)
    tab = episode_table(out.results, out.paths, cfg, dates, threshold=-0.15)
    assert len(tab) >= 3 and set(tab.strategy) == {"A", "B", "C"}
    gfc = tab[tab.episode.str.startswith("2007")]
    assert (gfc.index_drawdown < -0.40).all()
    a = gfc[gfc.strategy == "A"].iloc[0]
    assert a.utilisation > 0 and a.cash_dry_powder >= 0 and a.exposure_pct_nav > 0
    b = gfc[gfc.strategy == "B"].iloc[0]
    assert b.cash_pct_nav > a.cash_pct_nav  # B holds the dry powder
    snap = strategy_snapshot(out.results, out.paths, cfg, 12)
    assert snap.loc["A", "nav"] == pytest.approx(out.results["A"].nav[0, 12])
    rt = risk_table(out.results, out.paths)
    assert rt.loc["A", "ann_vol"] > rt.loc["C", "ann_vol"] > 0
    assert rt.loc["A", "max_drawdown"] < rt.loc["C", "max_drawdown"] < 0
    assert rt.loc["B", "var99_step"] >= rt.loc["B", "var95_step"]


def test_margin_distance_matches_closed_form(raw: dict[str, Any]) -> None:
    """At inception with no cash: d = 1 − L/(ℓ_E E) = 42.86 % for the reference balance sheet."""
    from fosim.analytics.episodes import margin_distance

    cfg = make_cfg(raw, **{"run.n_paths": 1, "run.horizon_years": 1, "equity_model.type": "historical_replay", "historical.file": "data/SPX_SPXFP_aligned_monthly.csv", "historical.start": "2015-01-30", "historical.columns": {"SPXFP": "spxfp", "SPX": "spx_rebased"}, "historical.held_column": "spx_rebased", "leverage.ltv_stress_schedule": []})
    out = run_config(cfg, ledger_paths=[0])
    d = margin_distance(out.results["A"], cfg, out.paths)
    assert d[0] == pytest.approx(1 - 250 / (0.5 * 875), abs=1e-9)
    # consistency with utilisation: when the equity has fallen by d the utilisation would be exactly 1
    a = out.results["A"]
    k = 6
    lev = cfg.leverage
    eq, cash, loan = a.series("held_mv")[0, k], a.series("cash")[0, k], a.series("loan")[0, k]
    lv_after = lev.ltv_base.equity_index * eq * (1 - d[k]) + lev.ltv_base.cash * max(cash, 0.0)
    assert loan / lv_after == pytest.approx(1.0, abs=1e-9)
    assert np.isnan(margin_distance(out.results["C"], cfg, out.paths)).all()  # no loan
