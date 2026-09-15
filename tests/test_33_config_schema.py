"""Config schema validation.

Spec test 33: USD-only guard — any non-USD currency field, FX series or quanto option type is
rejected with a clear message; no FX module is importable.
Spec test 39 (schema part): inconsistent underlying_type / dividend flags raise.
Plus: weights sum to one, PSD correlation, tenor grids, fail-fast on unknown keys.
"""

import copy
import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from fosim.config import DEFAULT_CONFIG_PATH, SimConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw() -> dict[str, Any]:
    with open(DEFAULT_CONFIG_PATH) as f:
        d: dict[str, Any] = yaml.safe_load(f)
    return d


def test_default_config_loads(raw: dict[str, Any]) -> None:
    cfg = SimConfig.model_validate(raw)
    assert cfg.run.currency == "USD"
    assert cfg.portfolio.nav0 == 1_000_000_000
    assert abs(sum(i.weight for i in cfg.illiquids) - 1.0) < 1e-12
    cfg2 = load_config(DEFAULT_CONFIG_PATH)
    assert cfg2 == cfg


def test_non_usd_currency_rejected(raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    bad["run"]["currency"] = "CHF"
    with pytest.raises(ValidationError, match="USD"):
        SimConfig.model_validate(bad)


def test_nested_currency_field_rejected(raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    bad["equity_indices"][0]["currency"] = "EUR"
    with pytest.raises(ValidationError, match="(?i)usd|currency"):
        SimConfig.model_validate(bad)


def test_fx_series_rejected(raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    bad["fx"] = {"pair": "USDCHF", "spot": 0.85}
    with pytest.raises(ValidationError, match="(?i)fx|usd"):
        SimConfig.model_validate(bad)


def test_quanto_option_rejected(raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    bad["options"]["option_type"] = "quanto"
    with pytest.raises(ValidationError, match="(?i)quanto|vanilla"):
        SimConfig.model_validate(bad)


def test_american_exercise_rejected(raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    bad["options"]["exercise"] = "american"
    with pytest.raises(ValidationError, match="(?i)european"):
        SimConfig.model_validate(bad)


def test_no_fx_module() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("fosim.market.fx")
    assert not (ROOT / "src" / "fosim" / "market" / "fx.py").exists()


def test_weights_must_sum_to_one(raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    bad["portfolio"]["weights"] = {"equities": 0.7, "illiquids": 0.4}
    with pytest.raises(ValidationError, match="(?i)sum"):
        SimConfig.model_validate(bad)
    bad = copy.deepcopy(raw)
    bad["illiquids"][0]["weight"] = 0.5
    with pytest.raises(ValidationError, match="(?i)sum"):
        SimConfig.model_validate(bad)


def test_unknown_key_rejected(raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    bad["leverage"]["ratio_of_navv"] = 0.3
    with pytest.raises(ValidationError):
        SimConfig.model_validate(bad)


def test_ranges(raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    bad["leverage"]["ltv_base"]["equity_index"] = 1.2
    with pytest.raises(ValidationError):
        SimConfig.model_validate(bad)
    bad = copy.deepcopy(raw)
    bad["leverage"]["thresholds"] = {"warn": 0.9, "call": 0.8, "closeout": 1.1, "target_after_cure": 0.8}
    with pytest.raises(ValidationError, match="(?i)warn|call|closeout"):
        SimConfig.model_validate(bad)


def test_underlying_type_dividend_consistency(raw: dict[str, Any]) -> None:
    # total_return underlying with a non-zero pricing dividend curve is inconsistent (test 39)
    i = next(k for k, e in enumerate(raw["equity_indices"]) if e["underlying_type"] == "price_return")  # SPX
    bad = copy.deepcopy(raw)
    bad["equity_indices"][i]["underlying_type"] = "total_return"
    with pytest.raises(ValidationError, match="(?i)total_return"):
        SimConfig.model_validate(bad)
    ok = copy.deepcopy(raw)
    ok["equity_indices"][i]["underlying_type"] = "total_return"
    ok["equity_indices"][i]["pricing_dividend_curve"] = {"tenors": [1, 5], "q": [0.0, 0.0]}
    ok["equity_indices"][i]["dividend_yield"] = 0.0
    cfg = SimConfig.model_validate(ok)
    assert cfg.equity_indices[i].underlying_type == "total_return"
    # an excess-return index cannot pay dividends
    bad = copy.deepcopy(raw)
    bad["equity_indices"][0]["dividend_yield"] = 0.01
    with pytest.raises(ValidationError, match="(?i)excess_return"):
        SimConfig.model_validate(bad)


def test_correlation_matrix_validation(raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    bad["correlation"] = {"names": ["SPX", "HedgeFunds"], "matrix": [[1.0, 0.5], [0.4, 1.0]]}
    with pytest.raises(ValidationError, match="(?i)symmetric"):
        SimConfig.model_validate(bad)
    bad["correlation"] = {"names": ["SPX", "HedgeFunds"], "matrix": [[1.0, 0.5], [0.5, 0.9]]}
    with pytest.raises(ValidationError, match="(?i)diagonal"):
        SimConfig.model_validate(bad)


def test_tenor_grid_consistency(raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    bad["implied_vol"]["term_structure"] = {"tenors": [1, 2, 5], "theta": [0.19, 0.2], "beta": [0.6, 0.45, 0.35]}
    with pytest.raises(ValidationError, match="(?i)length|tenor"):
        SimConfig.model_validate(bad)
    bad = copy.deepcopy(raw)
    bad["implied_vol"]["term_structure"]["tenors"] = [1, 5, 2]
    with pytest.raises(ValidationError, match="(?i)increasing"):
        SimConfig.model_validate(bad)


def test_pricing_source_requirements(raw: dict[str, Any]) -> None:
    bad = copy.deepcopy(raw)
    bad["pricing"]["source"] = "vol_surface_file"
    with pytest.raises(ValidationError, match="(?i)vol_surface_file"):
        SimConfig.model_validate(bad)
    bad = copy.deepcopy(raw)
    bad["pricing"]["source"] = "dealer_quotes"
    with pytest.raises(ValidationError, match="(?i)dealer_quotes"):
        SimConfig.model_validate(bad)


def test_strike_param_required_for_pct_and_delta_modes(raw: dict[str, Any]) -> None:
    for mode in ("pct_spot", "pct_forward", "delta_target"):
        bad = copy.deepcopy(raw)
        bad["options"]["strike_mode"] = mode
        bad["options"]["strike_param"] = None
        with pytest.raises(ValidationError, match="strike_param"):
            SimConfig.model_validate(bad)
    ok = copy.deepcopy(raw)
    ok["options"]["strike_mode"] = "delta_target"
    ok["options"]["strike_param"] = 0.5
    SimConfig.model_validate(ok)
