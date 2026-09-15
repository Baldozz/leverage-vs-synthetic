"""Inception carry and structure analytics (SPEC §6.2, §5.7.1) — closed form, no simulation.

* Inception balance sheets of A, B, C (gross exposure, dollar delta, cash, loan, % of NAV).
* Sizing × strike mode table for B: notional, premium, cash, dollar delta, % of A's exposure,
  vega, theta, rho, dividend sensitivity, elasticity.
* Put–call parity decomposition: C = S e^{−qT} − K e^{−rT} + P, i.e. long call + cash ≡ long
  equity + long put − forgone dividends with financing embedded at the option-implied rate.
* Annualised expected carry of each strategy under the user's inputs; margin-trigger distance d*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fosim.config.schema import SimConfig
from fosim.engine.conventions import continuous_dividend_yield, continuous_drift
from fosim.instruments.lombard_loan import margin_trigger_decline
from fosim.pricing.black_scholes import bsm_elasticity, bsm_greeks, bsm_price, forward_price
from fosim.pricing.vol_surface import ParametricSurface, solve_delta_target_strike
from fosim.strategies.base import held_beta_to_book, inception_balance_cfg

SIZING_MODES = ("notional_match", "delta_match", "beta_adjusted_delta_match", "premium_budget", "cash_reserve_target", "delta_fraction_of_A")
STRIKE_MODES = ("atm_spot", "atm_forward", "pct_spot", "pct_forward", "delta_target")


@dataclass(frozen=True)
class InceptionMarket:
    S0: float
    r: float  # zero rate for the option tenor (+ funding spread)
    q: float  # pricing dividend yield at the tenor
    T: float
    sigma_atm: float  # ATM (spot) vol at the tenor
    psi_T: float  # skew slope
    half_spread: float  # vol points (ask − mid)
    loan_rate: float
    cash_rate: float


def book_index(cfg: SimConfig) -> int:
    """Index carrying the largest option-book weight (the option underlying)."""
    return int(np.argmax([ic.option_book_weight for ic in cfg.equity_indices]))


def held_index(cfg: SimConfig) -> int:
    """Index with the largest weight in A's equity sleeve (the plain-vanilla holding)."""
    return int(np.argmax([ic.weight_in_equity_sleeve for ic in cfg.equity_indices]))


def held_beta(cfg: SimConfig) -> float:
    w = [ic.option_book_weight for ic in cfg.equity_indices]
    return held_beta_to_book(cfg.held_equity_portfolio.betas, cfg.index_names, w, cfg.held_equity_portfolio.beta_to_option_book)


def inception_market(cfg: SimConfig, index: int | None = None) -> InceptionMarket:
    index = book_index(cfg) if index is None else index
    ic = cfg.equity_indices[index]
    T = cfg.options.tenor_years
    from fosim.market.rates import build_rate_model

    model, r0 = build_rate_model(cfg.rates)
    r = float(model.zero_rate(r0, T)) + cfg.pricing.option_funding_spread
    if ic.underlying_type == "excess_return":
        q = r  # flat forward (rolled futures / excess-return index): Black-76
    elif ic.underlying_type == "total_return":
        q = 0.0
    else:
        q = float(np.interp(T, ic.pricing_dividend_curve.tenors, ic.pricing_dividend_curve.q))
    ps = ParametricSurface.from_config(cfg.implied_vol)
    iv0 = cfg.implied_vol.short.iv0 or cfg.implied_vol.short.theta
    sigma = float(ps.atm_vol(T, iv0))
    loan_rate = max(r0, cfg.loan_terms.base_floor or -1e9) + cfg.loan_terms.spread_tiers[0].spread
    return InceptionMarket(
        S0=ic.spot, r=r, q=q, T=T, sigma_atm=sigma, psi_T=float(ps.psi_T(T)), half_spread=0.5 * cfg.pricing.bid_ask_vol_pts_new,
        loan_rate=loan_rate, cash_rate=r0 - cfg.rates.cash.spread,
    )


def strike_for_mode(mk: InceptionMarket, mode: str, param: float | None, delta_definition: str = "bsm") -> float:
    F = float(forward_price(mk.S0, mk.r, mk.q, mk.T))
    if mode == "atm_spot":
        return mk.S0
    if mode == "atm_forward":
        return F
    if mode == "pct_spot":
        return mk.S0 * (param or 1.0)
    if mode == "pct_forward":
        return F * (param or 1.0)
    if mode == "delta_target":
        target = param if param is not None else 0.5

        def vol_of_K(K: float) -> float:
            return max(mk.sigma_atm + mk.psi_T * math.log(K / F), 0.01)

        return solve_delta_target_strike(mk.S0, mk.r, mk.q, mk.T, target, vol_of_K, delta_definition, mk.psi_T)  # type: ignore[arg-type]
    raise ValueError(mode)


def tranche_terms(mk: InceptionMarket, K: float, at_ask: bool = False) -> dict[str, float]:
    F = float(forward_price(mk.S0, mk.r, mk.q, mk.T))
    sig = max(mk.sigma_atm + mk.psi_T * math.log(K / F), 0.01) + (mk.half_spread if at_ask else 0.0)
    g = bsm_greeks(mk.S0, K, mk.r, mk.q, sig, mk.T)
    c = float(g.price) / mk.S0
    return {
        "strike": K, "strike_pct_spot": K / mk.S0, "vol": sig, "premium_pct": c, "delta": float(g.delta),
        "delta_smile": float(g.delta + g.vega * (-mk.psi_T / mk.S0)), "gamma": float(g.gamma), "vega_1pt_pct": float(g.vega) * 0.01 / mk.S0,
        "theta_year_pct": float(g.theta) / mk.S0, "rho_100bp_pct": float(g.rho) * 0.01 / mk.S0, "dq_100bp_pct": float(g.dq) * 0.01 / mk.S0,
        "elasticity": float(bsm_elasticity(g, mk.S0)),
    }


def sizing_table(cfg: SimConfig, include_ask: bool = False) -> pd.DataFrame:
    """B's inception structure for every sizing × strike mode (SPEC §5.7.1)."""
    bal = inception_balance_cfg(cfg)
    mk = inception_market(cfg)
    E, sleeve, nav0 = bal["reference_equity_exposure"], bal["equity_sleeve_B"], bal["nav0"]
    beta_held = held_beta(cfg)
    rows = []
    for smode in STRIKE_MODES:
        param = cfg.options.strike_param if cfg.options.strike_mode == smode else {"pct_spot": 1.10, "pct_forward": 1.0, "delta_target": 0.50}.get(smode)
        K = strike_for_mode(mk, smode, param, cfg.options.delta_definition)
        tt = tranche_terms(mk, K, at_ask=include_ask)
        delta = tt["delta"] if cfg.options.delta_definition == "bsm" else tt["delta_smile"]
        c = tt["premium_pct"]
        for zmode in SIZING_MODES:
            p = cfg.options.sizing_param if cfg.options.sizing_mode == zmode else None
            if zmode == "notional_match":
                N = E
            elif zmode == "delta_match":
                N = E / delta
            elif zmode == "beta_adjusted_delta_match":
                N = E * beta_held / delta
            elif zmode == "delta_fraction_of_A":
                N = (p if p is not None else 0.5) * E / delta
            elif zmode == "premium_budget":
                N = (p if p is not None else 0.10) * nav0 / c
            else:
                N = max(sleeve - (p if p is not None else 0.35) * nav0, 0.0) / c
            prem = N * c
            rows.append(
                {
                    "strike_mode": smode, "sizing_mode": zmode, "strike": K, "strike_pct_spot": tt["strike_pct_spot"], "vol": tt["vol"],
                    "notional": N, "premium": prem, "premium_pct_notional": c, "cash": sleeve - prem, "cash_pct_nav": (sleeve - prem) / nav0,
                    "delta": delta, "dollar_delta": N * delta, "pct_of_A_exposure": N * delta / E, "dollar_delta_pct_nav": N * delta / nav0,
                    "vega_1pt": N * tt["vega_1pt_pct"], "theta_year": N * tt["theta_year_pct"], "rho_100bp": N * tt["rho_100bp_pct"],
                    "dq_100bp": N * tt["dq_100bp_pct"], "elasticity": tt["elasticity"], "feasible": sleeve - prem >= 0,
                }
            )
    return pd.DataFrame(rows)


def balance_sheets(cfg: SimConfig) -> pd.DataFrame:
    bal = inception_balance_cfg(cfg)
    mk = inception_market(cfg)
    K = strike_for_mode(mk, cfg.options.strike_mode, cfg.options.strike_param, cfg.options.delta_definition)
    tt = tranche_terms(mk, K)
    tab = sizing_table(cfg)
    row = tab[(tab.strike_mode == cfg.options.strike_mode) & (tab.sizing_mode == cfg.options.sizing_mode)].iloc[0]
    nav0 = bal["nav0"]
    beta_held = held_beta(cfg)
    if cfg.options.ladder_mode == "fixed_notional_schedule":
        per = cfg.options.notional_pct_nav * nav0 if cfg.options.notional_pct_nav is not None else float(cfg.options.notional_per_purchase or 0.0)
        n_target = cfg.options.target_total_notional if cfg.options.target_total_notional is not None else per * cfg.options.tenor_years * (52 if cfg.options.purchase_frequency == "weekly" else 12)
        prem_b = n_target * float(row.premium_pct_notional)
        dd_b = n_target * float(row.delta)
    else:
        n_target, prem_b, dd_b = float(row.notional), float(row.premium), float(row.dollar_delta)
    data = {
        "A": {"equities": bal["equity_A"], "options_premium": 0.0, "illiquids": bal["illiquid"], "cash": bal["cash_A"], "loan": bal["loan"], "gross_equity_exposure": bal["equity_A"], "dollar_delta": bal["equity_A"] * beta_held},
        "B": {"equities": bal["equity_B"], "options_premium": prem_b, "illiquids": bal["illiquid"], "cash": bal["cash_B"] - prem_b, "loan": bal["loan_B"], "gross_equity_exposure": n_target + bal["equity_B"], "dollar_delta": dd_b + bal["equity_B"] * beta_held},
        "C": {"equities": bal["equity_C"], "options_premium": 0.0, "illiquids": bal["illiquid"], "cash": bal["cash_C"], "loan": bal["loan_C"], "gross_equity_exposure": bal["equity_C"], "dollar_delta": bal["equity_C"] * beta_held},
    }
    if cfg.leverage.strategy_d.enabled:
        buf = cfg.leverage.strategy_d.cash_buffer_pct_nav * nav0
        data["D"] = {**data["A"], "cash": buf, "loan": bal["loan"] + buf}
    df = pd.DataFrame(data).T
    df["nav"] = df.equities + df.options_premium + df.illiquids + df.cash - df.loan
    for c in ("equities", "options_premium", "illiquids", "cash", "loan", "gross_equity_exposure", "dollar_delta"):
        df[c + "_pct_nav"] = df[c] / df["nav"]
    df.attrs["tranche"] = tt
    return df


def put_call_parity_decomposition(cfg: SimConfig) -> dict[str, float]:
    """C = S e^{−qT} − K e^{−rT} + P for the configured strike; per unit of index and per USD of notional."""
    mk = inception_market(cfg)
    K = strike_for_mode(mk, cfg.options.strike_mode, cfg.options.strike_param, cfg.options.delta_definition)
    F = float(forward_price(mk.S0, mk.r, mk.q, mk.T))
    sig = max(mk.sigma_atm + mk.psi_T * math.log(K / F), 0.01)
    call = float(bsm_price(mk.S0, K, mk.r, mk.q, sig, mk.T, "call"))
    put = float(bsm_price(mk.S0, K, mk.r, mk.q, sig, mk.T, "put"))
    pv_div = mk.S0 * (1.0 - math.exp(-mk.q * mk.T))  # PV of forgone dividends over T
    pv_strike = K * math.exp(-mk.r * mk.T)
    financing_embedded = mk.S0 * math.exp(-mk.q * mk.T) - pv_strike  # forward-minus-strike leg (implicit financing)
    implied_fin_rate = mk.r  # continuous rate embedded in the call's forward
    return {
        "S0": mk.S0, "K": K, "T": mk.T, "sigma": sig, "call": call, "put": put, "call_pct": call / mk.S0, "put_pct": put / mk.S0,
        "pv_forgone_dividends_pct": pv_div / mk.S0, "pv_strike_pct": pv_strike / mk.S0, "embedded_financing_leg_pct": financing_embedded / mk.S0,
        "implied_financing_rate": implied_fin_rate, "lombard_loan_rate_simple": mk.loan_rate, "cash_yield_on_unspent_premium": mk.cash_rate,
        "parity_check": call - put - (mk.S0 * math.exp(-mk.q * mk.T) - pv_strike),
    }


def carry_comparison(cfg: SimConfig) -> pd.DataFrame:
    """Annualised expected carry / drift of each strategy under the user's inputs (first year, no rebalancing)."""
    bal = inception_balance_cfg(cfg)
    mk = inception_market(cfg)
    ic = cfg.equity_indices[held_index(cfg)]
    m_eq = continuous_drift(ic.expected_return, ic.return_type, ic.realized_vol)
    q_c = continuous_dividend_yield(ic.dividend_yield)
    ill_ret = sum(i.weight * continuous_drift(i.expected_return, i.return_type, i.vol) for i in cfg.illiquids)
    tab = sizing_table(cfg)
    row = tab[(tab.strike_mode == cfg.options.strike_mode) & (tab.sizing_mode == cfg.options.sizing_mode)].iloc[0]
    nav0 = bal["nav0"]
    rows = {
        "A": {
            "equity_price_drift": bal["equity_A"] * (m_eq - q_c + cfg.held_equity_portfolio.expected_alpha),
            "dividends_net": bal["equity_A"] * q_c * (1 - ic.dividend_wht) * (ic.underlying_type == "price_return"),
            "illiquids": bal["illiquid"] * ill_ret, "cash_interest": 0.0, "loan_interest": -bal["loan"] * mk.loan_rate,
            "option_theta": 0.0, "option_delta_drift": 0.0,
        },
        "B": {
            "equity_price_drift": 0.0, "dividends_net": 0.0, "illiquids": bal["illiquid"] * ill_ret,
            "cash_interest": float(row.cash) * mk.cash_rate, "loan_interest": 0.0, "option_theta": float(row.theta_year),
            "option_delta_drift": float(row.dollar_delta) * (m_eq - q_c),
        },
        "C": {
            "equity_price_drift": bal["equity_sleeve_B"] * (m_eq - q_c + cfg.held_equity_portfolio.expected_alpha),
            "dividends_net": bal["equity_sleeve_B"] * q_c * (1 - ic.dividend_wht) * (ic.underlying_type == "price_return"),
            "illiquids": bal["illiquid"] * ill_ret, "cash_interest": 0.0, "loan_interest": 0.0, "option_theta": 0.0, "option_delta_drift": 0.0,
        },
    }
    df = pd.DataFrame(rows).T
    df["total"] = df.sum(axis=1)
    df["total_pct_nav"] = df["total"] / nav0
    return df


def margin_trigger_distance(cfg: SimConfig) -> dict[str, float]:
    bal = inception_balance_cfg(cfg)
    lb = cfg.leverage.ltv_base
    out = {"d_star_base": margin_trigger_decline(bal["loan"], bal["equity_A"], bal["illiquid"], lb.equity_index, lb.illiquid)}
    for rule in cfg.leverage.ltv_stress_schedule:
        out[f"d_star_stress_x{rule.multiplier}"] = margin_trigger_decline(bal["loan"], bal["equity_A"], bal["illiquid"], lb.equity_index * rule.multiplier, lb.illiquid * rule.multiplier)
    return out
