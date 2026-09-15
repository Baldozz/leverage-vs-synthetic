"""Analytics sanity tests (§6.1, §6.3, §6.4, §6.5, §5.7.9): metrics reconcile with the engine, the Greek
explain shows an explicit residual, stochastic dominance behaves on known distributions, sensitivity
runs on common random numbers, exposure grids and lenses execute on a small run.
"""

from typing import Any

import numpy as np
import pytest
from tests.conftest import make_cfg

from fosim.analytics.attribution import (
    component_totals,
    greek_explain,
    realised_vs_implied_diagnostic,
)
from fosim.analytics.dominance import stochastic_dominance
from fosim.analytics.exposure import (
    delta_value_grid,
    exposure_fans,
    participation_profile,
    realised_capture,
)
from fosim.analytics.lenses import comparison_lenses
from fosim.analytics.metrics import (
    crra_certainty_equivalent,
    drawdown_stats,
    summary_table,
    terminal_stats,
    var_cvar,
)
from fosim.analytics.sensitivity import (
    SensitivityInput,
    breakeven,
    evaluate,
    heatmap,
    metric_median_terminal_nav,
    tornado,
)
from fosim.runner import RunOutput, run_config, with_overrides


@pytest.fixture(scope="module")
def raw_module() -> dict[str, Any]:
    import yaml

    from fosim.config import DEFAULT_CONFIG_PATH

    with open(DEFAULT_CONFIG_PATH) as f:
        d: dict[str, Any] = yaml.safe_load(f)
    return d


@pytest.fixture(scope="module")
def small_run(raw_module: dict[str, Any]) -> RunOutput:
    cfg = make_cfg(raw_module, **{"run.n_paths": 300, "run.horizon_years": 4, "leverage.strategy_d.enabled": True})
    return run_config(cfg, ledger_paths=[0])


def test_summary_table_reconciles(small_run: RunOutput) -> None:
    out = small_run
    cfg = out.cfg
    df = summary_table(out.results, out.paths, cfg.analytics.percentiles, cfg.analytics.var_levels, cfg.analytics.crra_gamma, cfg.spending.pct_nav_pa, cfg.dry_powder.liquidity_reserve.capital_call_months)
    assert set(df.columns) == {"A", "B", "C", "D"}
    for s in df.columns:
        assert df.loc["terminal_nav_mean", s] == pytest.approx(out.results[s].nav[:, -1].mean())
        assert df.loc["terminal_nav_p50", s] == pytest.approx(np.median(out.results[s].nav[:, -1]))
        assert 0 <= df.loc["p_margin_call", s] <= 1
    assert df.loc["p_B_beats_A", "B"] == pytest.approx((out.results["B"].nav[:, -1] > out.results["A"].nav[:, -1]).mean())
    ts = terminal_stats(out.results["A"].nav, 4.0, [5, 50, 95])
    assert ts.se_mean > 0 and ts.se_median > 0
    # CAGR labels: mean of CAGR ≠ CAGR of mean in general, both finite
    assert np.isfinite(ts.cagr_mean) and np.isfinite(ts.cagr_of_mean_nav)
    dd = drawdown_stats(out.results["A"].nav, out.paths.grid.dt)
    assert -1 <= dd["max_dd_mean"] <= 0  # type: ignore[operator]
    v, c = var_cvar(np.array([-0.5, -0.2, 0.0, 0.1, 0.3]), 0.8)
    assert c >= v


def test_crra_ce() -> None:
    W = np.array([100.0, 100.0, 100.0])
    assert crra_certainty_equivalent(W, 3.0)["ce"] == pytest.approx(100.0)
    assert crra_certainty_equivalent(W, 1.0)["ce"] == pytest.approx(100.0)
    W2 = np.array([50.0, 150.0])
    ce3 = crra_certainty_equivalent(W2, 3.0)["ce"]
    ce1 = crra_certainty_equivalent(W2, 1.0)["ce"]
    assert ce3 < ce1 < 100.0  # more risk aversion -> lower CE, both below the mean
    bad = crra_certainty_equivalent(np.array([-1.0, 50.0]), 2.0)
    assert bad["n_nonpositive"] == 1


def test_attribution_totals_and_greek_explain(small_run: RunOutput) -> None:
    out = small_run
    tot = component_totals(out.results)
    for s in tot.columns:
        assert tot.loc["total", s] == pytest.approx(tot.loc["nav_change_check", s], abs=1e-3)
    assert tot.loc["inception_costs_in_nav0", "B"] < 0 < -tot.loc["inception_costs_in_nav0", "B"] + 1  # transition + t0 bid/ask
    ge = greek_explain(out.results["B"], out.paths, out.cfg)
    assert "unexplained_residual" in ge.columns
    # the explain captures most of the option MTM variation: residual small relative to actual
    assert ge.attrs["residual_share_of_abs_actual"] < 0.35
    diag = realised_vs_implied_diagnostic(out.results["B"], out.paths, out.cfg)
    assert np.isfinite(diag["gamma_vs_theta_pnl_mean"])


def test_stochastic_dominance_known_cases() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 1.0, 5000)
    y = x - 0.5  # shifted down: x FSD y
    r = stochastic_dominance(x, y)
    assert r["x_fsd_y"] and r["x_ssd_y"] and not r["y_fsd_x"]
    z = 1.0 + 3.0 * (x - 1.0)  # mean-preserving spread: x SSD z, no FSD
    r2 = stochastic_dominance(x, z)
    assert r2["x_ssd_y"] and not r2["x_fsd_y"] and not r2["y_fsd_x"]


def test_exposure_analytics(small_run: RunOutput) -> None:
    out = small_run
    fans = exposure_fans(out.results, out.paths, out.cfg)
    assert "B_pct_of_A_exposure" in fans and fans["B_pct_of_A_exposure"].shape[0] == out.paths.n_steps + 1
    assert "A_effective_leverage" in fans
    tr = out.results["B"].ledger.tranche_frame()
    t12 = tr[(tr.step == 12) & (tr.path == 0)]
    S_now = float(out.paths.S[0, 12, 0])
    grid = delta_value_grid(t12, S_now, 0.04, 0.015, np.array([-0.5, -0.2, 0.0, 0.2, 0.5]), np.array([-5.0, 0.0, 20.0]), np.array([-100.0, 0.0, 100.0]))
    assert grid["value"].shape == (5, 3, 3)
    assert grid["pnl"][2, 1, 1] == pytest.approx(0.0, abs=1e-6)  # no shock -> no P&L
    assert grid["dollar_delta"][4, 1, 1] > grid["dollar_delta"][0, 1, 1]  # delta rises with the index
    assert grid["value"][0, 2, 1] > grid["value"][0, 1, 1]  # vol spike cushions the down move
    prof = participation_profile(out.cfg)
    caps = prof.attrs["capture"]
    assert 0 < caps[5.0]["up_capture_B_vs_A"] < 1.5
    assert caps[5.0]["down_capture_B_vs_A"] < 1.0  # B loses less than A on the downside
    rc = realised_capture(out.results["B"], out.paths)
    assert rc["down_beta"] < rc["up_beta"] + 0.5  # long-gamma book: down-beta not above up-beta by much


def test_sensitivity_crn_and_tornado(raw_module: dict[str, Any]) -> None:
    cfg = make_cfg(raw_module, **{"run.n_paths": 150, "run.horizon_years": 3})
    base = evaluate(cfg, {}, metric_median_terminal_nav, 150)
    again = evaluate(cfg, {}, metric_median_terminal_nav, 150)
    assert base == again  # CRN: identical draws, identical result
    inputs = [
        SensitivityInput("equity_indices.1.expected_return", "ER SPX (held by A/C)", 0.04, 0.10),
        SensitivityInput("equity_indices.0.expected_return", "ER SPXFP (option underlying)", 0.0, 0.06),
        SensitivityInput("loan_terms.spread_tiers.0.spread", "spread", 0.005, 0.02),
    ]
    tor = tornado(cfg, inputs, metric_median_terminal_nav, 150)
    assert len(tor) == 3
    er = tor[tor.path == "equity_indices.1.expected_return"].iloc[0]
    assert er.A_high > er.A_low and er.C_high > er.C_low  # A and C hold SPX
    erb = tor[tor.path == "equity_indices.0.expected_return"].iloc[0]
    assert erb.B_high > erb.B_low and erb.A_high == pytest.approx(erb.A_low)  # only B's options depend on SPXFP
    sp = tor[tor.path == "loan_terms.spread_tiers.0.spread"].iloc[0]
    assert sp.A_high < sp.A_low and sp.B_high == pytest.approx(sp.B_low)  # B has no loan
    hm = heatmap(cfg, "equity_indices.0.expected_return", [0.05, 0.09], "rates.usd.r0", [0.02, 0.05], metric_median_terminal_nav, 150)
    assert hm["BminusA"].shape == (2, 2)


def test_breakeven_solver(raw_module: dict[str, Any]) -> None:
    cfg = make_cfg(raw_module, **{"run.n_paths": 200, "run.horizon_years": 3})
    # the borrowing spread at which A's and B's median terminal NAV are equal must exist between 0 and 19 % (schema cap 20 %) (B is spread-insensitive)
    res = breakeven(cfg, "loan_terms.spread_tiers.0.spread", 0.0, 0.19, metric_median_terminal_nav, 200)
    assert res.value is not None and 0.0 < res.value < 0.19
    assert res.ci_low is not None and res.ci_low <= res.value <= (res.ci_high or 1)


def test_with_overrides_roundtrip(raw_module: dict[str, Any]) -> None:
    cfg = make_cfg(raw_module)
    c2 = with_overrides(cfg, {"rates.usd.r0": 0.02, "illiquids.1.ta_model.rc": 0.4, "implied_vol.term_structure.theta": [0.2, 0.2, 0.2]})
    assert c2.rates.usd.r0 == 0.02 and c2.illiquids[1].ta_model is not None and c2.illiquids[1].ta_model.rc == 0.4
    assert c2.implied_vol.term_structure.theta == [0.2, 0.2, 0.2]
    assert cfg.rates.usd.r0 == 0.04  # original untouched


@pytest.mark.slow
def test_comparison_lenses(raw_module: dict[str, Any]) -> None:
    # bullet ladder: the whole book is bought at t0, so the inception dollar-delta is comparable to A's
    cfg = make_cfg(raw_module, **{"run.n_paths": 150, "run.horizon_years": 3, "options.ladder_mode": "bullet"})
    out = comparison_lenses(cfg, 150, ["equal_delta", "equal_vol"])
    assert "equal_capital" in out and "equal_delta" in out and "equal_vol" in out
    assert out["equal_delta"]["table"].loc["dollar_delta0", "B"] == pytest.approx(out["equal_delta"]["table"].loc["dollar_delta0", "A"], rel=0.02)
