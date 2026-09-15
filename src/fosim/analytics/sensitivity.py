"""Sensitivity analysis (SPEC §6.4): tornado, 2-D heatmaps and the breakeven solver, all on common
random numbers (the seed is fixed, so every run draws the same shocks; only the model parameters change).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import brentq

from fosim.analytics.metrics import crra_certainty_equivalent, var_cvar
from fosim.config.schema import SimConfig
from fosim.engine.state import StrategyResult
from fosim.runner import RunOutput, run_config, with_overrides

MetricFn = Callable[[StrategyResult, RunOutput], float]


@dataclass(frozen=True)
class SensitivityInput:
    path: str
    label: str
    low: Any
    high: Any
    fmt: str = "{:.4g}"


def default_inputs(cfg: SimConfig) -> list[SensitivityInput]:
    """One-at-a-time inputs per SPEC §6.4 (ranges around the current config values)."""
    ins: list[SensitivityInput] = []
    for i, ic in enumerate(cfg.equity_indices):
        ins.append(SensitivityInput(f"equity_indices.{i}.expected_return", f"Expected return {ic.name}", ic.expected_return - 0.03, ic.expected_return + 0.03))
        ins.append(SensitivityInput(f"equity_indices.{i}.realized_vol", f"Realised vol {ic.name}", max(ic.realized_vol - 0.05, 0.02), ic.realized_vol + 0.08))
    ic = cfg.equity_indices[0]
    ins += [
        SensitivityInput("implied_vol.term_structure.theta", "Implied vol level (all tenors)", [t - 0.04 for t in cfg.implied_vol.term_structure.theta], [t + 0.08 for t in cfg.implied_vol.term_structure.theta]),
        SensitivityInput("equity_model.vrp_ratio", "Variance risk premium ratio", 0.0, 0.3),
        SensitivityInput("implied_vol.skew_1y", "Skew ψ(1y)", -0.2, 0.0),
        SensitivityInput("loan_terms.spread_tiers.0.spread", "Borrowing spread", 0.005, 0.02),
        SensitivityInput("rates.usd.r0", "Base rate", max(cfg.rates.usd.r0 - 0.03, -0.01), cfg.rates.usd.r0 + 0.03),
        SensitivityInput("rates.cash.spread", "Cash spread", 0.0, 0.005),
        SensitivityInput("leverage.ltv_base.equity_index", "Equity LTV", 0.4, 0.6),
        SensitivityInput("leverage.ltv_stress_schedule.0.multiplier", "LTV stress haircut", 0.6, 1.0),
        SensitivityInput("equity_model.jump_lambda", "Crash intensity λ", 0.0, 0.3),
        SensitivityInput("equity_indices.0.jump_mean", "Crash size μ_J", -0.3, -0.05),
        SensitivityInput("pricing.bid_ask_vol_pts_new", "Option bid/ask (vol pts)", 0.0, 0.02),
        SensitivityInput("options.hold_months_before_roll", "Ladder hold months", 6, 24),
        SensitivityInput("dry_powder.tiers.0.index_drawdown", "First dry-powder trigger", -0.30, -0.10),
        SensitivityInput("illiquids.1.ta_model.rc", "PE capital-call rate", 0.1, 0.5),
        SensitivityInput("equity_indices.0.pricing_dividend_curve.q", "Pricing dividend yield", [max(q - 0.01, 0.0) for q in ic.pricing_dividend_curve.q], [q + 0.01 for q in ic.pricing_dividend_curve.q]),
        SensitivityInput("equity_indices.0.dividend_yield", "Realised dividend yield", max(ic.dividend_yield - 0.01, 0.0), ic.dividend_yield + 0.01),
        SensitivityInput("loan_terms.base_floor", "Base-rate floor", 0.0, 0.02),
        SensitivityInput("pricing.stress_bid_ask_multiplier", "Bid/ask stress multiplier", 1.0, 5.0),
    ]
    return ins


# ---------------------------------------------------------------------------- metric functions


def metric_median_terminal_nav(res: StrategyResult, out: RunOutput) -> float:
    return float(np.median(res.nav[:, -1]))


def metric_mean_terminal_nav(res: StrategyResult, out: RunOutput) -> float:
    return float(res.nav[:, -1].mean())


def metric_cvar95_horizon(res: StrategyResult, out: RunOutput) -> float:
    return -var_cvar(res.nav[:, -1] / res.nav[:, 0] - 1.0, 0.95)[1]  # as a return (negative = loss)


def metric_p_margin_call(res: StrategyResult, out: RunOutput) -> float:
    return float((res.events.margin_calls > 0).mean())


def metric_crra_ce(res: StrategyResult, out: RunOutput) -> float:
    return float(crra_certainty_equivalent(res.nav[:, -1], out.cfg.analytics.crra_gamma)["ce"])


METRICS: dict[str, MetricFn] = {
    "median_terminal_nav": metric_median_terminal_nav,
    "mean_terminal_nav": metric_mean_terminal_nav,
    "cvar95_horizon_return": metric_cvar95_horizon,
    "p_margin_call": metric_p_margin_call,
    "crra_ce": metric_crra_ce,
}


# ---------------------------------------------------------------------------- runners


def evaluate(cfg: SimConfig, overrides: dict[str, Any], metric: MetricFn, n_paths: int, strategies: tuple[str, ...] = ("A", "B", "C")) -> dict[str, float]:
    c = with_overrides(cfg, overrides) if overrides else cfg
    out = run_config(c, n_paths=n_paths)
    return {s: metric(out.results[s], out) for s in strategies if s in out.results}


def tornado(cfg: SimConfig, inputs: list[SensitivityInput], metric: MetricFn, n_paths: int, progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    base = evaluate(cfg, {}, metric, n_paths)
    rows = []
    total = 2 * len(inputs)
    done = 0
    for inp in inputs:
        vals = {}
        for side, v in (("low", inp.low), ("high", inp.high)):
            try:
                vals[side] = evaluate(cfg, {inp.path: v}, metric, n_paths)
            except Exception as e:
                vals[side] = {"error": float("nan"), "_msg": str(e)}  # type: ignore[dict-item]
            done += 1
            if progress:
                progress(done, total)
        row: dict[str, Any] = {"input": inp.label, "path": inp.path, "low_value": inp.low, "high_value": inp.high}
        for s in ("A", "B", "C"):
            row[f"{s}_base"] = base.get(s, float("nan"))
            row[f"{s}_low"] = vals["low"].get(s, float("nan"))
            row[f"{s}_high"] = vals["high"].get(s, float("nan"))
        row["BminusA_base"] = base.get("B", np.nan) - base.get("A", np.nan)
        row["BminusA_low"] = vals["low"].get("B", np.nan) - vals["low"].get("A", np.nan)
        row["BminusA_high"] = vals["high"].get("B", np.nan) - vals["high"].get("A", np.nan)
        row["swing_BminusA"] = abs(row["BminusA_high"] - row["BminusA_low"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("swing_BminusA", ascending=False).reset_index(drop=True)


def heatmap(cfg: SimConfig, path_x: str, xs: list[Any], path_y: str, ys: list[Any], metric: MetricFn, n_paths: int, progress: Callable[[int, int], None] | None = None) -> dict[str, pd.DataFrame]:
    grids = {s: np.full((len(ys), len(xs)), np.nan) for s in ("A", "B", "C")}
    done, total = 0, len(xs) * len(ys)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            v = evaluate(cfg, {path_x: x, path_y: y}, metric, n_paths)
            for s in grids:
                grids[s][i, j] = v.get(s, np.nan)
            done += 1
            if progress:
                progress(done, total)
    out = {s: pd.DataFrame(g, index=[str(y) for y in ys], columns=[str(x) for x in xs]) for s, g in grids.items()}
    out["BminusA"] = out["B"] - out["A"]
    return out


@dataclass(frozen=True)
class BreakevenResult:
    input_path: str
    value: float | None
    ci_low: float | None
    ci_high: float | None
    f_low: float
    f_high: float
    se_at_solution: float
    reliable: bool
    message: str


def breakeven(cfg: SimConfig, input_path: str, lo: float, hi: float, metric: MetricFn, n_paths: int, per_path_metric: Callable[[StrategyResult], NDArray[np.float64]] | None = None) -> BreakevenResult:
    """Value of ``input_path`` at which metric(B) = metric(A), by brentq on common random numbers.

    The CI comes from the MC standard error of the per-path B−A difference at the solution divided by the
    slope of f; when the sign of f at the bracket ends is not significant at 2 SE, the result is flagged unreliable.
    """

    def f(x: float) -> float:
        v = evaluate(cfg, {input_path: x}, metric, n_paths, ("A", "B"))
        return v["B"] - v["A"]

    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return BreakevenResult(input_path, None, None, None, f_lo, f_hi, float("nan"), False, "no sign change on the bracket: A and B are not equal for any value in range")
    x_star: float = brentq(f, lo, hi, xtol=max(1e-6 * abs(hi - lo), 1e-9), maxiter=60)
    # SE of the difference at the solution
    c = with_overrides(cfg, {input_path: x_star})
    out = run_config(c, n_paths=n_paths)
    if per_path_metric is None:
        d = out.results["B"].nav[:, -1] - out.results["A"].nav[:, -1]
    else:
        d = per_path_metric(out.results["B"]) - per_path_metric(out.results["A"])
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    h = 0.05 * (hi - lo)
    slope = (f(min(x_star + h, hi)) - f(max(x_star - h, lo))) / (min(x_star + h, hi) - max(x_star - h, lo))
    ci = 1.96 * se / abs(slope) if slope != 0 else float("inf")
    reliable = abs(f_lo) > 2 * se and abs(f_hi) > 2 * se and np.isfinite(ci)
    msg = "" if reliable else "MC noise is large relative to the metric difference: increase n_paths or widen the bracket"
    return BreakevenResult(input_path, x_star, x_star - ci, x_star + ci, f_lo, f_hi, se, reliable, msg)
