"""Independent minimal re-implementation of Strategies A, B (bullet) and C on a deterministic
zero-volatility scenario (SPEC §7 test 21).

This file deliberately shares **no code** with ``fosim``: it uses only the standard library and
re-derives every rule from docs/METHODOLOGY.md with scalar arithmetic. It is the second pair of
eyes on the engine; if the two disagree, one of them is wrong and the test fails.

Scenario (monthly steps, all amounts USD):
  * held index (SPX) grows deterministically: S_{k+1} = S_k · exp((m − q_c)/12), m = ln(1 + μ_a), q_c = ln(1 + q);
    the option underlying (SPXFP, excess-return index) grows at exp(m_opt/12) and is priced with q = r;
  * held portfolio = held index (β = 1, no tracking error, no alpha);
  * flat implied vol σ_iv, flat zero rate r, pricing dividend q_p; no bid/ask, no commissions, no
    trading costs, no jumps, no stochastic rates, no spending;
  * illiquids: deterministic growth exp(m_i/12) − 1, Takahashi–Alexander flows for closed-end funds,
    Geltner-smoothed quarterly reporting (unit-index formulation);
  * A: loan L₀, interest paid in cash monthly at max(r, floor) + spread on ACT/360 with 365/12 days;
    cash at r − spread_cash; capital calls funded from cash, then from the facility (headroom);
  * C: same illiquids, sleeve in equities, calls funded by selling equities;
  * B: transition at t0 (sell equities, repay loan), one bullet 5y ATM call tranche with notional
    = A's equity exposure bought at mid, marked monthly with Black–Scholes–Merton.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bsm_call(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    if T <= 0.0:
        return max(S - K, 0.0)
    if sigma <= 0.0:
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    sT = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sT
    d2 = d1 - sT
    return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


@dataclass
class Illiquid:
    name: str
    nav0: float
    expected_return: float
    phi: float
    reporting_months: int
    closed_end: bool
    unfunded: float = 0.0
    rc: float = 0.0
    b: float = 0.0
    life: float = 0.0
    age0: float = 0.0


@dataclass
class Scenario:
    nav0: float = 1_000_000_000.0
    w_eq: float = 0.70
    w_ill: float = 0.30
    leverage: float = 0.25
    allocation: str = "pro_rata"
    mu_a: float = 0.07  # arithmetic expected total return of the HELD index (SPX)
    mu_opt: float = 0.03  # arithmetic expected return of the OPTION underlying (SPXFP, excess-return index)
    opt_is_excess_return: bool = True  # option underlying priced with q = r (flat forward), no dividends
    q_realised: float = 0.015
    wht: float = 0.15
    r: float = 0.04
    loan_floor: float = 0.0
    loan_spread: float = 0.01
    cash_spread: float = 0.001
    sigma_iv: float = 0.20
    q_pricing: float = 0.015
    tenor: float = 5.0
    horizon_months: int = 36
    day_count_days: float = 365.0 / 12.0
    facility_limit: float = 400_000_000.0
    ltv_equity: float = 0.5
    ltv_cash: float = 0.9
    call_threshold: float = 1.0
    illiquids: list[Illiquid] | None = None


def per_step(rate_annual: float, dt: float) -> float:
    return float(1.0 - (1.0 - rate_annual) ** dt)


def run(sc: Scenario) -> dict[str, list[float]]:
    dt = 1.0 / 12.0
    N = sc.horizon_months
    m = math.log(1.0 + sc.mu_a)
    q_c = math.log(1.0 + sc.q_realised)
    # ---------------- market paths: held index (price index, dividends leave the price) and option underlying
    S = [100.0 * math.exp((m - q_c) * dt * k) for k in range(N + 1)]
    m_opt = math.log(1.0 + sc.mu_opt)
    S_opt = [100.0 * math.exp(m_opt * dt * k) for k in range(N + 1)]
    q_opt = sc.r if sc.opt_is_excess_return else sc.q_pricing
    # ---------------- inception balance sheet
    loan0 = sc.leverage * sc.nav0
    if sc.allocation == "pro_rata":
        gross = sc.nav0 + loan0
        eq_A, ill_total = gross * sc.w_eq, gross * sc.w_ill
    else:
        ill_total = sc.nav0 * sc.w_ill
        eq_A = sc.nav0 * sc.w_eq + loan0
    sleeve = sc.nav0 - ill_total
    ills = sc.illiquids or []
    # ---------------- illiquids (identical for all strategies): true NAV, reported NAV, flows
    ill_true = [[0.0] * (N + 1) for _ in ills]
    ill_rep = [[0.0] * (N + 1) for _ in ills]
    calls = [0.0] * (N + 1)
    dists = [0.0] * (N + 1)
    for j, il in enumerate(ills):
        nav = il.nav0
        ill_true[j][0] = nav
        ill_rep[j][0] = nav
        U = il.unfunded
        i_true, i_true_last, i_obs_last, r_obs_prev = 1.0, 1.0, 1.0, 0.0
        g = math.exp(math.log(1.0 + il.expected_return) * dt) - 1.0  # sigma = 0 -> m dt
        rc_step = per_step(il.rc, dt) if il.closed_end else 0.0
        for k in range(N):
            i_true *= 1.0 + g
            after = nav * (1.0 + g)
            C = D = 0.0
            if il.closed_end:
                age = il.age0 + (k + 1) * dt
                rd_annual = min((age / il.life) ** il.b, 1.0)
                rd_step = per_step(rd_annual, dt)
                C = rc_step * U
                U -= C
                D = rd_step * after
            nav = after + C - D
            ill_true[j][k + 1] = nav
            calls[k + 1] += C
            dists[k + 1] += D
            if (k + 1) % il.reporting_months == 0:
                r_true = i_true / i_true_last - 1.0
                r_obs = (1.0 - il.phi) * r_true + il.phi * r_obs_prev
                i_obs_last *= 1.0 + r_obs
                i_true_last = i_true
                r_obs_prev = r_obs
            ill_rep[j][k + 1] = nav * i_obs_last / i_true
    ill_official = [sum(ill_rep[j][k] for j in range(len(ills))) for k in range(N + 1)]
    days = sc.day_count_days
    loan_rate = max(sc.r, sc.loan_floor) + sc.loan_spread
    cash_rate = sc.r - sc.cash_spread

    def dividends(units: float, s_end: float) -> float:
        return units * s_end * (math.exp(q_c * dt) - 1.0) * (1.0 - sc.wht)

    # ---------------- Strategy A
    nav_A = [0.0] * (N + 1)
    units = eq_A / S[0]
    cash, loan = 0.0, loan0
    nav_A[0] = cash - loan + units * S[0] + ill_official[0]
    for k in range(N):
        cash -= loan * loan_rate * days / 360.0
        cash += max(cash, 0.0) * cash_rate * days / 360.0 + min(cash, 0.0) * loan_rate * days / 360.0
        cash += dividends(units, S[k + 1])
        cash += dists[k + 1] - calls[k + 1]
        if cash < 0.0:
            # liquidity waterfall: draw on the facility within headroom (limit and lending value at the call threshold)
            lv = sc.ltv_equity * units * S[k + 1] + sc.ltv_cash * max(cash, 0.0)
            head = max(min(sc.facility_limit - loan, sc.call_threshold * lv - loan), 0.0)
            draw = min(-cash, head)
            loan += draw
            cash += draw
            if cash < -1e-9:
                sell = -cash
                units -= sell / S[k + 1]
                cash += sell
        nav_A[k + 1] = cash - loan + units * S[k + 1] + ill_official[k + 1]
    # ---------------- Strategy C
    nav_C = [0.0] * (N + 1)
    units = sleeve / S[0]
    cash = 0.0
    nav_C[0] = cash + units * S[0] + ill_official[0]
    for k in range(N):
        cash += max(cash, 0.0) * cash_rate * days / 360.0 + min(cash, 0.0) * loan_rate * days / 360.0
        cash += dividends(units, S[k + 1])
        cash += dists[k + 1] - calls[k + 1]
        if cash < -1e-9:
            sell = -cash
            units -= sell / S[k + 1]
            cash += sell
        nav_C[k + 1] = cash + units * S[k + 1] + ill_official[k + 1]
    # ---------------- Strategy B (bullet, notional match, no costs)
    nav_B = [0.0] * (N + 1)
    cash = sleeve  # sold all equities, repaid the loan (no costs)
    notional = eq_A
    n_units = notional / S_opt[0]
    K = S_opt[0]
    premium = n_units * bsm_call(S_opt[0], K, sc.r, q_opt, sc.sigma_iv, sc.tenor)
    cash -= premium
    expiry = round(sc.tenor * 12)
    nav_B[0] = cash + premium + ill_official[0]
    for k in range(N):
        cash += max(cash, 0.0) * cash_rate * days / 360.0 + min(cash, 0.0) * loan_rate * days / 360.0
        cash += dists[k + 1] - calls[k + 1]
        T = (expiry - (k + 1)) / 12.0
        if k + 1 == expiry:
            cash += n_units * max(S_opt[k + 1] - K, 0.0)
            n_units = 0.0
        opt = n_units * bsm_call(S_opt[k + 1], K, sc.r, q_opt, sc.sigma_iv, T) if n_units > 0 else 0.0
        nav_B[k + 1] = cash + opt + ill_official[k + 1]
    return {"A": nav_A, "B": nav_B, "C": nav_C, "index": S, "index_opt": S_opt, "illiquid_official": ill_official}


def default_scenario(horizon_months: int = 36) -> Scenario:
    """The default.yaml illiquid book (PLACEHOLDER inputs) in zero-vol form."""
    gross = 1_250_000_000.0
    ill_total = gross * 0.30
    ills = [
        Illiquid("HedgeFunds", ill_total * 0.33, 0.06, 0.2, 3, False),
        Illiquid("PrivateEquity", ill_total * 0.34, 0.10, 0.5, 3, True, 100_000_000.0, 0.25, 2.5, 10.0, 4.0),
        Illiquid("Infrastructure", ill_total * 0.33, 0.08, 0.5, 3, True, 50_000_000.0, 0.25, 2.0, 12.0, 5.0),
    ]
    return Scenario(horizon_months=horizon_months, illiquids=ills)


if __name__ == "__main__":
    out = run(default_scenario())
    for name in ("A", "B", "C"):
        print(name, [round(v / 1e6, 4) for v in out[name][:4]], "...", round(out[name][-1] / 1e6, 4))
