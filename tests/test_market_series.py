"""Historical replay driven by observed series (rates, vol, dividends) from data/market_data_*.csv.

* loan rate at each monthly reset = observed 3m loan base (LIBOR→Term SOFR spliced) + spread;
* cash accrues at the observed T-bill yield (less the cash spread);
* options are priced and marked with r = observed 5y Treasury and σ = the 5y vol series of the day
  (term structure set to identity so IV_ATM(5y, t) = iv_5y(t));
* A's cash dividends follow the observed S&P dividend yield;
* the accounting identity holds; values quoted in percent are converted.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_cfg

from fosim.pricing.black_scholes import bsm_price
from fosim.runner import run_config

COLS = {"SPXFP": "spxfp", "SPX": "spx_rebased", "LOAN_BASE": "loan_base", "CASH_YIELD": "tbill_3m", "OPTION_RATE": "ust_5y", "DIV_YIELD": "spx_div_yld", "IV": "iv_5y"}
TABLE = {"enabled": True, "illiquids": 250e6, "A": {"equities": 1000e6, "loan": 250e6, "cash": 0.0}, "B": {"equities": 0.0, "loan": 0.0, "cash": 750e6}, "C": {"equities": 750e6, "loan": 0.0, "cash": 0.0}}


def _cfg(raw: dict[str, Any], start: str = "2006-01-06", years: float = 3, **extra: Any) -> Any:
    theta = 20.0
    return make_cfg(
        raw,
        **{
            "run.n_paths": 1, "run.horizon_years": years, "run.dt": "weekly", "equity_model.type": "historical_replay",
            "historical.file": "data/market_data_weekly.csv", "historical.start": start, "historical.columns": COLS, "historical.held_column": "spx_rebased",
            "custom_inception": TABLE, "dry_powder.enabled": False, "costs.equity_bps": 0,
            # continuous purchases (no cap) so every week has a purchase to compare across vol regimes
            "options.ladder_mode": "fixed_notional_schedule", "options.purchase_frequency": "weekly", "options.notional_per_purchase": 4e6, "options.target_total_notional": None,
            "pricing.bid_ask_vol_pts_new": 0.0, "pricing.bid_ask_vol_pts_unwind": 0.0, "pricing.scenario_overrides": [],
            # identity term structure: IV_ATM(T, t) = iv_5y(t) for every tenor
            "implied_vol.short.theta": theta / 100, "implied_vol.short.iv0": None, "implied_vol.short.floor": 0.03, "implied_vol.short.cap": 2.0,
            "implied_vol.term_structure": {"tenors": [1, 2, 5], "theta": [theta / 100] * 3, "beta": [1.0, 1.0, 1.0]}, "implied_vol.skew_1y": 0.0,
            "loan_terms.spread_tiers": [{"utilisation_below": 1.0, "spread": 0.005}], "loan_terms.base_floor": 0.0, "rates.cash.spread": 0.0,
            "leverage.interest": "capitalise", "leverage.capitalisation_frequency": "monthly",
            **extra,
        },
    )


def test_series_drive_rates_vol_and_dividends(raw: dict[str, Any]) -> None:
    cfg = _cfg(raw)
    out = run_config(cfg, ledger_paths=[0])
    mp = out.paths
    assert set(mp.series) == {"loan_base", "cash_yield", "option_rate", "div_yield"}
    src = pd.read_csv("data/market_data_weekly.csv", parse_dates=["date"])
    src = src[src.date >= "2006-01-06"].reset_index(drop=True).iloc[: mp.n_steps + 1]
    # percent → decimal conversion and alignment
    assert np.allclose(mp.series["loan_base"][0], src.loan_base.to_numpy() / 100)
    assert np.allclose(mp.series["cash_yield"][0], src.tbill_3m.to_numpy() / 100)
    assert np.allclose(mp.iv_short[0], src.iv_5y.to_numpy() / 100)
    assert np.allclose(mp.series["div_yield"][0], np.log1p(src.spx_div_yld.to_numpy() / 100))
    a, b = out.results["A"], out.results["B"]
    # loan rate at every month-end reset = observed base + 50 bp; cash rate = observed T-bill
    g = mp.grid
    for k in np.flatnonzero(g.is_month_end)[:20]:
        assert a.series("loan_rate")[0, k] == pytest.approx(src.loan_base[k] / 100 + 0.005, abs=1e-12)
    assert np.allclose(b.series("cash_rate")[0, 1:], src.tbill_3m.to_numpy()[:-1] / 100, atol=1e-12)
    # 2008: the observed T-bill collapses, so B's cash income must fall accordingly (weekly interest ≈ cash × y × 7.02/360)
    k08 = int(np.flatnonzero(src.date >= "2008-12-19")[0])
    ci = b.recorder.components[0, k08 + 1, 6]  # cash_interest component of that step
    assert ci == pytest.approx(b.series("cash")[0, k08] * (src.tbill_3m[k08] / 100) * g.days / 360, rel=1e-9)
    assert ci < b.recorder.components[0, 10, 6] * 0.2  # much lower than in 2006
    # option purchases are priced at the 5y Treasury and the vol of the day
    tr = b.ledger.tranche_frame().drop_duplicates(["slot", "purchase_step"])
    for _, t in tr.head(30).iterrows():
        k = int(t.purchase_step)
        r = src.ust_5y[k] / 100
        px = float(bsm_price(mp.S[0, k, 0], mp.S[0, k, 0], r, r, src.iv_5y[k] / 100, 5.0))  # excess-return underlying: q = r
        assert t.cost_usd == pytest.approx(t.units * px, rel=1e-9)
        assert t.vol_at_purchase_ask == pytest.approx(src.iv_5y[k] / 100, abs=1e-12)
    # premium varies with the vol regime: late-2008 purchases cost more than 2006 ones (as % of notional)
    tr["prem_pct"] = tr.cost_usd / tr.notional_usd
    assert tr[tr.purchase_step >= k08].prem_pct.mean() > 1.3 * tr[tr.purchase_step < 20].prem_pct.mean()
    # dividends follow the observed yield series (A holds SPX only)
    k = 30
    units = a.series("held_mv")[0, k] / mp.held[0, k]
    expected = units * mp.held[0, k + 1] * np.expm1(np.log1p(src.spx_div_yld[k] / 100) * g.dt) * (1 - 0.15)
    assert a.recorder.components[0, k + 1, 4] == pytest.approx(expected, rel=1e-9)
    # accounting identity on all strategies
    for r_ in out.results.values():
        assert np.allclose(np.diff(r_.nav[0]), r_.recorder.components[0, 1:, :].sum(axis=1), atol=2e-6)


def test_gap_in_series_is_rejected(raw: dict[str, Any]) -> None:
    # 24m implied vol starts in May 2005: a window from 2004 with that column must fail loudly
    cols = {**COLS, "IV": "iv_24m"}
    with pytest.raises(ValueError, match="gaps"):
        run_config(_cfg(raw, start="2004-01-09", years=2, **{"historical.columns": cols}), ledger_paths=[0])


def test_fixed_premium_is_exact_under_varying_rates(raw: dict[str, Any]) -> None:
    """Fixed-premium mode: every purchase costs exactly the override (15 %) although the discount rate moves."""
    cols = {k: v for k, v in COLS.items() if k != "IV"}
    cfg = _cfg(raw, start="2006-01-06", years=4, **{"historical.columns": cols, "pricing.scenario_overrides": [{"index": "SPXFP", "month_from": 0, "tenor": 5.0, "premium_pct": 0.15}]})
    out = run_config(cfg, ledger_paths=[0])
    tr = out.results["B"].ledger.tranche_frame().drop_duplicates(["slot", "purchase_step"])
    assert len(tr) > 150
    assert np.allclose(tr.cost_usd / tr.notional_usd, 0.15, atol=1e-9)
    assert (tr.source == "scenario_override").all()
    # the rate moved a lot in 2006-2009, so the backed-out vol must differ across purchases
    assert tr.vol_at_purchase_ask.max() - tr.vol_at_purchase_ask.min() > 0.01
