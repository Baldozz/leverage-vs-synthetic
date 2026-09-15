"""Bootstrap (§4.1.4) and stylised / historical stress paths (§4.7): shape, re-centring, templates."""

import copy
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from fosim.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, SimConfig
from fosim.market.bootstrap import stationary_block_indices
from fosim.market.generator import generate_market_paths
from fosim.market.stress import build_historical_replay, build_stylised_paths, load_templates


@pytest.fixture
def raw() -> dict[str, Any]:
    with open(DEFAULT_CONFIG_PATH) as f:
        d: dict[str, Any] = yaml.safe_load(f)
    return d


def test_stationary_block_lengths() -> None:
    rng = np.random.default_rng(1)
    idx = stationary_block_indices(rng, 300, 2000, 400, mean_block=12.0)
    # consecutive rows within a block increase by 1 (mod n); block starts are independent draws
    cont = (np.diff(idx, axis=1) % 300 == 1).mean()
    # P(continue) = 1 - 1/12 minus the chance that an independent draw happens to be the successor
    assert abs(cont - (1 - 1 / 12)) < 0.02


def test_bootstrap_paths_from_spx(raw: dict[str, Any]) -> None:
    d = copy.deepcopy(raw)
    d["equity_model"]["type"] = "bootstrap"
    d["bootstrap"]["file"] = "data/SPX_SPXFP_aligned_monthly.csv"
    d["bootstrap"]["columns"] = {"SPXFP": "spxfp", "SPX": "spx_rebased"}
    d["run"]["n_paths"] = 3000
    d["run"]["horizon_years"] = 5
    cfg = SimConfig.model_validate(d)
    mp = generate_market_paths(cfg, base_dir=PROJECT_ROOT)
    assert mp.S.shape == (3000, 61, 2)
    assert (mp.S > 0).all()
    # re-centred: mean log return per month ≈ (m - q_px - σ²/2)/12 with the user's expected return
    p = mp.index_params[0]
    lr = np.diff(np.log(mp.S[:, :, 0]), axis=1)
    target = (p.m - p.q_c_px - 0.5 * p.sigma**2) / 12
    se = lr.std() / math.sqrt(lr.size) * 3  # generous: blocks are dependent
    assert abs(lr.mean() - target) < 5 * se + 5e-4
    # same seed -> identical
    mp2 = generate_market_paths(cfg, base_dir=PROJECT_ROOT)
    assert np.array_equal(mp.S, mp2.S)
    assert mp.meta["model"] == "bootstrap"


def test_stylised_templates_shapes(raw: dict[str, Any]) -> None:
    cfg = SimConfig.model_validate(raw)
    templates = load_templates()
    assert set(templates) >= {"GFC_like", "COVID_like_V", "DotCom_like_grind", "Rates_and_equity_2022_like", "Lost_decade_L"}
    for name, sc in templates.items():
        res = build_stylised_paths(cfg, sc)
        mp = res.paths
        assert mp.n_paths == 1
        lvl = mp.S[0, :, 0] / mp.S[0, 0, 0]
        k_trough = sc.months_to_trough  # monthly grid
        assert lvl[k_trough] == pytest.approx(1 + sc.drawdown, rel=1e-9), name
        assert lvl[:k_trough].min() >= lvl[k_trough] - 1e-12
        assert mp.iv_short[0, k_trough] == pytest.approx(min(sc.iv_peak, cfg.implied_vol.short.cap))
        assert "not historical" in res.label
        if sc.recovery_shape != "L":
            k_rec = sc.months_to_trough + sc.months_to_recover
            if k_rec <= mp.n_steps:
                assert lvl[k_rec] == pytest.approx(1.0, rel=1e-9)
        assert mp.r_short[0, k_trough] == pytest.approx(cfg.rates.usd.r0 + sc.rate_shock_bp * 1e-4)


def test_historical_replay(raw: dict[str, Any]) -> None:
    d = copy.deepcopy(raw)
    d["run"]["horizon_years"] = 5
    cfg = SimConfig.model_validate(d)
    cols = {"SPXFP": "spxfp", "SPX": "spx_rebased"}
    res = build_historical_replay(cfg, "data/SPX_SPXFP_aligned_monthly.csv", "2007-10-31", cols, base_dir=PROJECT_ROOT, held_column="spx_rebased")
    mp = res.paths
    assert mp.S.shape == (1, 61, 2)
    src = pd.read_csv(PROJECT_ROOT / "data/SPX_SPXFP_aligned_monthly.csv", parse_dates=["date"])
    src = src[src["date"] >= "2007-10-31"].iloc[:61]
    ratio = src["spx_rebased"].to_numpy() / src["spx_rebased"].to_numpy()[0]
    assert np.allclose(mp.S[0, :, 1] / mp.S[0, 0, 1], ratio)
    assert np.allclose(mp.held[0] / 100.0, ratio)
    assert mp.meta["model"] == "historical_replay"
    with pytest.raises(ValueError, match="rows"):
        build_historical_replay(cfg, "data/SPX_SPXFP_aligned_monthly.csv", "2024-01-31", cols, base_dir=PROJECT_ROOT)
