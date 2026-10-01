"""Five ways to hold the capital — the investor's scenarios on one historical path (docs/METHODOLOGY.md §10, Assumptions 40).

Capital C (650 m), loan share ℓ (25 %), n-year ATM calls on SPXFP at a fixed premium c (14.5 % of notional). The call sleeve is sized by
**premium** (default: κ = 25 % of the capital in premium, the investor's note — 162.5 m buys 1,120 m of notional) or by **exposure** (option:
X of the capital in SPX-equivalent, delta × notional; the premium follows from the model delta of the day, P = X·C·c/δ), and **built over time**:
N steps a week or a month apart (52 weekly by default, 1 = one shot), the money waiting in SPX and rotated step by step; every tranche has
its own expiry and is rolled on its own slot.

    1  Long the market:          C in SPX
    2  Levered long:             (1 + ℓ)·C in SPX, a loan ℓ·C at a flat rate (5.5 %), interest capitalised, repaid from SPX after one tenor
    3  SPX + calls:              C in SPX, each step sells one slot of premium for calls; at a tranche's expiry tax on the profit, then the
                                 slot refilled to its share of the portfolio (X·V/N of exposure, or κ·V/N of premium), SPX absorbing the rest
    4  Long + calls on a loan:   C in SPX, each step draws one slot of premium as a loan; at a tranche's expiry tax, then the slot's share of the
                                 loan repaid from the proceeds (SPX sold for a shortfall) and only the surplus in new calls — or repaid and the slot
                                 refilled from SPX — or never repaid, the slot refilled on a new loan
    5  All in calls:             C in SPX rotated into calls step by step; each tranche rolls its own after-tax payoff; a
                                 worthless tranche lapses, the last one lapsing with nothing left ends it

SPX with dividends reinvested net of withholding (TR), the loan simple ACT/360 capitalised daily, calls on SPXFP marked with Black–Scholes
(q = r = the tenor's Treasury of the day, ACT/365) at the vol implied by the fixed premium on the purchase day — or, with ``premium=None``, at
the market's implied vol of the day; the dollar delta of the live calls (Σ units · δ(t) · S(t)) is reported every day and the SPX-equivalent
exposure is SPX + dollar delta. Tax = rate × max(payoff − premium paid, 0) at expiry, no loss carry-forward. Margin call (2 and 4): the
loan above ``margin_call`` × the lending value of the holdings; flagged, no forced sale by default — or, with ``on_call="liquidate"``, SPX sold
and the loan repaid that day until LTV = ``ltv_after_call`` (the calls sold at their mark when the SPX runs out, then the cash). NAV = SPX + calls + cash − loan; the accounting
identity ΔNAV = ΔSPX + Δcalls + cash interest − loan interest − tax is asserted every day for every scenario. ``summary`` adds each path's
risk measures on its own daily NAV returns: volatility, 1-day historical VaR/CVaR at 95 and 99 %, max drawdown and its day, and a Sharpe-like
ratio against the 3-month T-bill averaged over the holding period.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fosim.analytics.call_vs_cash import DAILY_FILE, tenor_columns, total_return_index
from fosim.analytics.leverage_stress import (
    TOL,
    AccountingIdentityError,
    _load,
    _nearest_index,
    corrections,
)
from fosim.analytics.metrics import var_cvar
from fosim.pricing.black_scholes import bsm_greeks, bsm_price
from fosim.pricing.implied_vol import implied_vol

F64 = NDArray[np.float64]
TRADING_DAYS = 252                     # one step of the daily file = one trading day; volatility annualised by √252

SCENARIOS = ("1", "2", "3", "4", "5")
LABELS = {"1": "Long the market", "2": "Levered long", "3": "SPX + calls", "4": "Long + calls on a loan", "5": "All in calls"}
LEVERED = ("2", "4")
REPAY = ("tenor", "never")             # scenario 2: the loan repaid from SPX after one tenor, or rolled up to the end
REPAY_CALLS = ("surplus", "refill_spx", "reborrow")   # scenario 4 at a slot's expiry: the loan share repaid and only the surplus in new calls; repaid and the slot refilled from SPX; never repaid, the slot refilled on a new loan
REBALANCE = ("portfolio", "proceeds")  # scenario 3 at expiry: the slot refilled to its share of the portfolio, or the after-tax proceeds split (1 − κ)/κ
SIZING = ("exposure", "premium")       # the call sleeve: X of the capital in SPX-equivalent exposure, or κ of the capital in premium
BUILD_UNITS = ("week", "month")        # the build steps 7 calendar days or one calendar month apart
ON_CALL = ("flag", "liquidate")        # a margin call: flagged only, or SPX (then calls, then cash) sold to bring the LTV back to ltv_after_call


def liquidation_sales(e: float, calls: float, cash: float, loan: float, lv_e: float, lv_c: float, lv_t: float, target: float) -> tuple[float, float, float]:
    """USD of SPX, then of calls, then of cash sold at the mark and paid into the loan until loan ÷ lending value ≤ ``target``: each stage
    solves (loan − s) = target · (lending value − ℓ·s) for its asset, capped at what is held; the next stage only when the asset runs out."""
    s = (loan - target * (lv_e * e + lv_c * calls + lv_t * cash)) / (1.0 - target * lv_e)
    if s <= e:
        return max(s, 0.0), 0.0, 0.0
    loan -= e
    x = (loan - target * (lv_c * calls + lv_t * cash)) / (1.0 - target * lv_c)
    if x <= calls:
        return e, max(x, 0.0), 0.0
    loan -= calls
    p = (loan - target * lv_t * cash) / (1.0 - target * lv_t)
    return e, calls, min(max(p, 0.0), cash)


@dataclass(frozen=True)
class Expiry:
    """One tranche's expiry: what it paid, what was taken off, what was done with the rest."""
    date: pd.Timestamp
    strike: float
    payoff: float
    premium_paid: float      # for the tranche that expired
    tax: float
    loan_repaid: float
    spx_traded: float        # SPX bought (+) or sold (−) on the day
    new_premium: float
    new_notional: float
    worthless: bool
    slot: int = 0            # the build step this tranche descends from (0-based)
    wiped_out: bool = False  # scenario 5: nothing left to reinvest anywhere
    loan_drawn: float = 0.0  # scenario 4 under "reborrow": the new call bought on a new loan


@dataclass(frozen=True)
class Liquidation:
    """One forced sale on a margin call: the LTV before, what was sold (at the mark, paid into the loan), the LTV after."""
    date: pd.Timestamp
    ltv_before: float
    spx_sold: float
    calls_sold: float
    cash_used: float
    ltv_after: float


@dataclass(frozen=True)
class ScenarioInfo:
    id: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    spx0: float              # SPX after the first day's trades
    loan0: float             # the loan drawn: at t₀ (2) or over the build (4)
    premium0: float          # premium of the first tranche, USD
    notional0: float
    prem_frac0: float        # premium as a fraction of notional at t₀
    delta0: float            # model delta of the ATM call at t₀
    vol0: float              # vol the first call is marked at
    expiries: tuple[Expiry, ...]
    margin_call_first: pd.Timestamp | None
    days_in_margin_call: int
    max_ltv: float
    loan_repaid_on: pd.Timestamp | None
    wiped_out_on: pd.Timestamp | None
    unexpired: bool          # a live call has not reached expiry: its value on the end day is a model mark
    rate_col: str
    vol_col: str
    premium_build: float     # premium paid over the build
    notional_build: float    # notional bought over the build
    exposure_build: float    # SPX + dollar delta on the last build day
    build_end: pd.Timestamp  # the last build day
    n_tranches: int          # build steps taken (1 = one shot)
    liquidations: tuple[Liquidation, ...] = ()   # the forced sales on margin calls (``on_call="liquidate"``)


def simulate_scenarios(
    start: str | pd.Timestamp, end: str | pd.Timestamp, capital: float = 650e6, loan_frac: float = 0.25, call_frac: float = 0.25,
    loan_rate: float | None = 0.055, spread: float = 0.0075, premium: float | None = 0.145, tenor: float = 5.0,
    tax_rate: float = 0.24, tax_on: tuple[str, ...] = ("3", "4", "5"), lv_equity: float = 0.75, lv_calls: float = 0.0, lv_cash: float = 0.90,
    margin_call: float = 0.90, wht: float = 0.15, repay: str = "tenor", repay_calls: str = "surplus", rebalance: str = "portfolio",
    sizing: str = "premium", exposure_frac: float = 0.86, build_steps: int = 52, build_unit: str = "week", file: Path | str | None = None,
    scenarios: tuple[str, ...] = SCENARIOS, on_call: str = "flag", ltv_after_call: float = 0.5,
) -> tuple[dict[str, pd.DataFrame], dict[str, ScenarioInfo]]:
    """Daily path of the five scenarios from ``start`` to ``end`` (nearest trading days). See the module docstring. ``scenarios``: the ids to
    run (all five by default; each scenario's path is independent of the others, so a subset is bit-identical to its columns in the full run).
    ``on_call``: a margin call (2 and 4) ``"flag"`` — flagged, nothing sold (default) — or ``"liquidate"``: that day SPX is sold and the loan repaid
    until loan ÷ lending value = ``ltv_after_call``, the calls sold at their mark (pro rata across the slots) when the SPX runs out, then the cash
    (``liquidation_sales``); the day counts as a margin call, the sale is recorded in ``ScenarioInfo.liquidations`` and the ``liquidated`` column.

    ``loan_rate``: flat annual rate of the loans (0.055 default); ``None`` uses the 3-month base rate of the day + ``spread``.
    ``premium``: the call premium as a fraction of notional (0.145 default), the calls marked at the vol it implies on the purchase day;
    ``None`` prices and marks at the market's implied vol of the day. ``tax_on``: the scenarios whose call profits are taxed at expiry (all three call scenarios by default).
    ``repay``: scenario 2's loan, ``"tenor"`` (sold SPX on the trading day nearest t₀ + tenor) or ``"never"``. ``repay_calls``: scenario 4 at a
    slot's expiry — ``"surplus"`` (each build slot's share of the loan repaid from its proceeds, SPX sold for a shortfall; only the cash left
    buys a new call, so a worthless call ends its slot), ``"refill_spx"`` (the same repayment, then the slot refilled to its share of the
    portfolio from the cash and SPX sold for the rest, no new loan) or ``"reborrow"`` (never repaid: the after-tax proceeds buy SPX and the
    slot's replacement is bought on a new loan, so the loan runs to the end). ``rebalance``: scenario 3
    at a tranche's expiry, ``"portfolio"`` (the slot refilled to its share of the whole portfolio, SPX sold or bought for the difference) or
    ``"proceeds"`` (premium sizing only: the after-tax proceeds split (1 − κ)/κ, the SPX untouched). ``sizing``: ``"premium"`` (default) — the sleeve
    is ``call_frac`` of the capital in premium — or ``"exposure"`` — ``exposure_frac`` of the capital in SPX-equivalent (delta × notional),
    premium = exposure × c ÷ δ of the day. ``build_steps`` / ``build_unit``: the calls bought in that many steps a week or a month apart
    (1 = all on the start day), the money waiting in SPX; steps falling on or after the end day are not taken. Returns one path per
    scenario id and one ``ScenarioInfo`` each.
    """
    if capital <= 0 or tenor <= 0:
        raise ValueError("capital and tenor must be positive")
    if not 0.0 <= loan_frac < 1.0 or not 0.0 <= call_frac < 1.0 or not 0.0 <= tax_rate < 1.0 or not 0.0 <= wht < 1.0:
        raise ValueError("loan_frac, call_frac, tax_rate and wht must be in [0, 1)")
    if premium is not None and not 0.0 < premium < 1.0:
        raise ValueError("premium must be in (0, 1) or None for the market's vol")
    if loan_rate is not None and loan_rate < 0.0:
        raise ValueError("loan_rate must be non-negative or None for the base rate + spread")
    if not 0.0 < lv_equity <= 1.0 or not 0.0 <= lv_calls <= 1.0 or not 0.0 <= lv_cash <= 1.0 or not 0.0 < margin_call <= 1.0:
        raise ValueError("lv_equity and margin_call in (0, 1], lv_calls and lv_cash in [0, 1]")
    if any(s not in SCENARIOS for s in tax_on):
        raise ValueError(f"tax_on must be a subset of {SCENARIOS}")
    if repay not in REPAY or repay_calls not in REPAY_CALLS or rebalance not in REBALANCE:
        raise ValueError(f"repay must be one of {REPAY}, repay_calls one of {REPAY_CALLS}, rebalance one of {REBALANCE}")
    if sizing not in SIZING or build_unit not in BUILD_UNITS:
        raise ValueError(f"sizing must be one of {SIZING}, build_unit one of {BUILD_UNITS}")
    if exposure_frac < 0.0 or int(build_steps) < 1:
        raise ValueError("exposure_frac must be non-negative and build_steps at least 1")
    if rebalance == "proceeds" and sizing == "exposure":
        raise ValueError("rebalance='proceeds' splits a premium budget: use it with sizing='premium'")
    if not scenarios or any(s not in SCENARIOS for s in scenarios):
        raise ValueError(f"scenarios must be a non-empty subset of {SCENARIOS}")
    if on_call not in ON_CALL or (on_call == "liquidate" and not 0.0 < ltv_after_call < margin_call):
        raise ValueError(f"on_call must be one of {ON_CALL}; ltv_after_call in (0, margin_call) when liquidating")
    full = _load(str(file or DAILY_FILE))
    rate_col, vol_col = tenor_columns(full, tenor)
    d = full.dropna(subset=["spxfp", "spx_px_last", "spx_div_yld", "loan_base", "tbill_3m", rate_col, vol_col]).reset_index(drop=True)
    dates_all = d.date.to_numpy().astype("datetime64[D]")
    k0 = _nearest_index(dates_all, np.datetime64(pd.Timestamp(start), "D"))
    k1 = _nearest_index(dates_all, np.datetime64(pd.Timestamp(end), "D"))
    if k1 <= k0:
        raise ValueError("end must be after start")
    d = d.iloc[k0 : k1 + 1].reset_index(drop=True)
    n = len(d)
    dates = d.date.to_numpy().astype("datetime64[D]")
    dt360 = np.concatenate([[0.0], np.diff(dates).astype(float) / 360.0])
    S = d.spxfp.to_numpy(dtype=np.float64)
    r = d[rate_col].to_numpy(dtype=np.float64) / 100.0
    iv = d[vol_col].to_numpy(dtype=np.float64) / 100.0
    tbill = d.tbill_3m.to_numpy(dtype=np.float64) / 100.0
    rate = np.full(n, float(loan_rate)) if loan_rate is not None else d.loan_base.to_numpy(dtype=np.float64) / 100.0 + spread
    px = d.spx_px_last.to_numpy(dtype=np.float64)
    tr = total_return_index(px, d.spx_div_yld.to_numpy(dtype=np.float64), d.date.to_numpy(), wht)
    tr = tr / tr[0]
    months = round(tenor * 12)
    n_tr = int(build_steps)
    # the build schedule: step w on the trading day nearest t₀ + w weeks (or months); steps on or after the end day are not taken
    if build_unit == "week":
        targets = [dates[0] + np.timedelta64(7 * w, "D") for w in range(1, n_tr)]
    else:
        targets = [np.datetime64(pd.Timestamp(dates[0]) + pd.DateOffset(months=w), "D") for w in range(1, n_tr)]
    build_days = [0] + [k for t in targets if (k := _nearest_index(dates, t)) < n - 1]
    if len(set(build_days)) != len(build_days):
        raise ValueError("the build schedule does not fit the path (need distinct trading days)")
    build_index = {k: w for w, k in enumerate(build_days)}

    def expiry_of(k: int) -> tuple[np.datetime64, int]:
        """Calendar expiry of a call bought on day k and the index of the nearest trading day (−1 when beyond the data)."""
        target = np.datetime64(pd.Timestamp(dates[k]) + pd.DateOffset(months=months), "D")
        return target, (_nearest_index(dates, target) if target <= dates[-1] else -1)

    def atm(k: int) -> tuple[float, float, float]:
        """(premium fraction, vol, model delta) of the ATM call of the tenor bought on day k (q = r)."""
        if premium is None:
            sig = float(iv[k])
            c = float(bsm_price(100.0, 100.0, r[k], r[k], sig, tenor)) / 100.0
        else:
            c = float(premium)
            sig = implied_vol(c * 100.0, 100.0, 100.0, float(r[k]), float(r[k]), tenor)
        dl = float(np.asarray(bsm_greeks(100.0, 100.0, r[k], r[k], sig, tenor).delta))
        return c, sig, dl

    def slot_premium(k: int, base: float) -> float:
        """Premium of one slot on day k when the whole sleeve is a share of ``base``: κ·base/N, or X·base/N of exposure at the delta of the day."""
        if sizing == "premium":
            return call_frac * base / n_tr
        c, _, dl = atm(k)
        return exposure_frac * base / n_tr * c / dl

    k_repay = -1
    if repay == "tenor":
        _, k_repay = expiry_of(0)

    class Tranche:
        __slots__ = ("exp_date", "first_cycle", "k_buy", "k_exp", "premium", "sigma", "slot", "strike", "units")

        def __init__(self, units: float, strike: float, exp_date: np.datetime64, k_exp: int, sigma: float, k_buy: int, prem: float, slot: int, first_cycle: bool) -> None:
            self.units, self.strike, self.exp_date, self.k_exp, self.sigma, self.k_buy, self.premium = units, strike, exp_date, k_exp, sigma, k_buy, prem
            self.slot, self.first_cycle = slot, first_cycle

    def run(sid: str) -> tuple[pd.DataFrame, ScenarioInfo]:
        taxed = sid in tax_on
        buys_calls = sid in ("3", "4", "5")
        E0 = {"1": capital, "2": (1.0 + loan_frac) * capital, "3": capital, "4": capital, "5": capital}[sid]
        eq_units = E0 / tr[0]
        loan, cash = (loan_frac * capital if sid == "2" else 0.0), 0.0
        marks = np.zeros(n)                       # model value of the live calls on each day (each tranche's purchase and expiry days excluded)
        deltas = np.zeros(n)                      # dollar delta of the live calls, Σ units · δ(t) · S(t) (the purchase day included at the sizing delta)
        tranches: list[Tranche] = []
        E, call_val, call_notional, strike_arr, cash_arr, loan_arr = (np.empty(n) for _ in range(6))
        n_calls = np.zeros(n, dtype=np.int64)
        eq_pnl, call_pnl, cash_int, interest, tax_day, prem_day, pay_day, spx_traded = (np.zeros(n) for _ in range(8))
        is_expiry = np.zeros(n, dtype=bool)
        expiries: list[Expiry] = []
        liquidations: list[Liquidation] = []
        called = np.zeros(n, dtype=bool)          # the days of a forced sale: in margin call whatever the LTV after it
        liq_day = np.zeros(n)                     # USD sold on a margin call that day (SPX + calls + cash paid into the loan)
        wiped_out_on: pd.Timestamp | None = None
        loan_repaid_on: pd.Timestamp | None = None
        premium_build = notional_build = loan_build = 0.0

        def mark_path(t: Tranche) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """(days, price per unit, S × delta per unit) of a tranche from the day after its purchase to the day before its expiry."""
            k_end = t.k_exp if t.k_exp > 0 else n
            idx = np.arange(t.k_buy + 1, k_end)
            if not idx.size:
                return idx, np.zeros(0), np.zeros(0)
            tau = (t.exp_date - dates[idx]).astype(float) / 365.0
            vol = np.full(idx.size, t.sigma) if premium is not None else iv[idx]
            strikes = np.full(idx.size, t.strike)
            price = np.asarray(bsm_price(S[idx], strikes, r[idx], r[idx], vol, tau), dtype=np.float64)
            dlt = S[idx] * np.asarray(bsm_greeks(S[idx], strikes, r[idx], r[idx], vol, tau).delta, dtype=np.float64)
            return idx, price, dlt

        def buy(k: int, prem: float, slot: int, first_cycle: bool) -> Tranche:
            """Buy ``prem`` USD of ATM calls of the tenor on day k; the mark from the next day to the day before expiry."""
            c, sig, dl = atm(k)
            notional = prem / c
            e_date, k_e = expiry_of(k)
            t = Tranche(notional / S[k], S[k], e_date, k_e, sig, k, prem, slot, first_cycle)
            idx, price, dlt = mark_path(t)
            marks[idx] += t.units * price
            deltas[idx] += t.units * dlt
            deltas[k] += t.units * S[k] * dl
            prem_day[k] += prem
            return t

        def sell_calls(k: int, frac: float) -> None:
            """Sell the fraction ``frac`` of every live tranche at the day's mark: the units, the premium carried and the marks from today on scale down."""
            _, _, dl = atm(k)
            for t in tranches:
                idx, price, dlt = mark_path(t)
                m = idx >= k
                marks[idx[m]] -= frac * t.units * price[m]
                deltas[idx[m]] -= frac * t.units * dlt[m]
                if t.k_buy == k:
                    deltas[k] -= frac * t.units * S[k] * dl
                t.units *= 1.0 - frac
                t.premium *= 1.0 - frac
            if frac >= 1.0 - 1e-12:
                tranches.clear()

        def build_step(k: int, w: int) -> float:
            """Build step w on day k: one slot of premium bought, from SPX sold (3, 5) or a loan drawn (4). Returns the premium."""
            nonlocal eq_units, loan, premium_build, notional_build, loan_build
            e_now = eq_units * tr[k]
            if sid == "5":
                prem = e_now / (n_tr - w)                       # an equal share of the SPX still held at each step: all of it at the last
            else:
                prem = slot_premium(k, capital)
            if prem <= TOL:
                return 0.0
            if sid == "4":
                loan += prem
                loan_build += prem
            else:
                prem = min(prem, e_now)
                eq_units -= prem / tr[k]
                spx_traded[k] -= prem
            t = buy(k, prem, w, True)
            tranches.append(t)
            premium_build += prem
            notional_build += t.units * t.strike
            return float(prem)

        c0, sig0, dl0 = atm(0)
        prem_today = build_step(0, 0) if buys_calls else 0.0
        first = tranches[0] if tranches else None
        premium0, notional0 = (first.premium, first.units * first.strike) if first else (0.0, 0.0)

        def book(k: int, prem_today: float) -> None:
            E[k], cash_arr[k], loan_arr[k] = eq_units * tr[k], cash, loan
            call_val[k] = marks[k] + prem_today
            notional = sum(t.units * t.strike for t in tranches)
            call_notional[k], n_calls[k] = notional, len(tranches)
            strike_arr[k] = sum(t.units * t.strike * t.strike for t in tranches) / notional if notional > 0.0 else np.nan

        book(0, prem_today)
        for k in range(1, n):
            e_now = eq_units * tr[k]
            eq_pnl[k] = eq_units * (tr[k] - tr[k - 1])
            cash_int[k] = cash * tbill[k - 1] * dt360[k]
            cash += cash_int[k]
            interest[k] = loan * rate[k - 1] * dt360[k]
            loan += interest[k]
            marks_pre = marks[k]
            prem_today = payoff_today = 0.0
            date = pd.Timestamp(dates[k])
            for t in [t for t in tranches if t.k_exp == k]:
                first_cycle_alive = sum(1 for u in tranches if u.first_cycle)   # this tranche included
                tranches.remove(t)
                payoff = t.units * max(S[k] - t.strike, 0.0)
                payoff_today += payoff
                is_expiry[k] = True
                tax = tax_rate * max(payoff - t.premium, 0.0) if taxed else 0.0
                cash += payoff - tax
                tax_day[k] += tax
                repaid = bought = new_prem = new_notional = 0.0
                drawn = 0.0
                if sid == "4" and loan > 0.0 and t.first_cycle and repay_calls != "reborrow":   # this slot's share of the loan, from the proceeds, then from SPX sold
                    share = loan / first_cycle_alive
                    repaid = min(share, cash)
                    cash -= repaid
                    loan -= repaid
                    short = share - repaid
                    if short > TOL:
                        sold = min(short, e_now)
                        eq_units -= sold / tr[k]
                        e_now = eq_units * tr[k]
                        loan -= sold
                        repaid += sold
                        bought = -sold
                    if loan <= TOL:
                        loan = 0.0
                        loan_repaid_on = date
                if k < n - 1:
                    if sid == "3":
                        if rebalance == "portfolio":         # the slot refilled to its share of the whole portfolio, SPX for the difference
                            total = e_now + marks[k] + prem_today + cash
                            new_prem = min(slot_premium(k, total), cash + e_now)
                            bought = cash - new_prem
                        elif cash > 0.0:                     # the after-tax proceeds split (1 − κ)/κ, the SPX untouched
                            bought = (1.0 - call_frac) * cash
                            new_prem = call_frac * cash
                        eq_units += bought / tr[k]
                        e_now = eq_units * tr[k]
                        cash -= bought + new_prem
                    elif sid == "4" and repay_calls == "refill_spx":   # the slot refilled to its share of the portfolio, from the cash and then SPX sold; no new loan
                        total = e_now + marks[k] + prem_today + cash - loan
                        new_prem = max(min(slot_premium(k, total), cash + e_now), 0.0)
                        bought = cash - new_prem
                        eq_units += bought / tr[k]
                        e_now = eq_units * tr[k]
                        cash -= bought + new_prem
                    elif sid == "4" and repay_calls == "reborrow":     # the after-tax proceeds into SPX, the slot refilled on a new loan
                        total = e_now + marks[k] + prem_today + cash - loan
                        bought = cash
                        eq_units += bought / tr[k]
                        e_now = eq_units * tr[k]
                        cash = 0.0
                        new_prem = max(slot_premium(k, total), 0.0)
                        drawn = new_prem
                        loan += drawn
                    elif sid in ("4", "5") and cash > 0.0:   # the rest in new calls (no new loan)
                        new_prem, cash = cash, 0.0
                    if new_prem > TOL:
                        nt = buy(k, new_prem, t.slot, False)
                        tranches.append(nt)
                        prem_today += new_prem
                        new_notional = nt.units * nt.strike
                    elif new_prem > 0.0:                     # dust: kept as cash rather than a call of a few cents (a dust loan is undone)
                        if drawn > 0.0:
                            loan -= drawn
                            drawn = 0.0
                        else:
                            cash += new_prem
                        new_prem = 0.0
                spx_traded[k] += bought
                expiries.append(Expiry(date, t.strike, payoff, t.premium, tax, repaid, bought, new_prem, new_notional, payoff <= 0.0, t.slot, loan_drawn=drawn))
            if buys_calls and k in build_index:
                prem_today += build_step(k, build_index[k])
                e_now = eq_units * tr[k]
            if sid == "5" and wiped_out_on is None and not tranches and cash <= TOL and k >= build_days[-1] and k < n - 1:
                wiped_out_on = date                          # the last tranche lapsed with nothing left anywhere
                if expiries and expiries[-1].date == date:
                    expiries[-1] = replace(expiries[-1], wiped_out=True)
            if sid == "2" and k == k_repay and loan > 0.0:  # the loan repaid from SPX after one tenor
                sold = min(loan, e_now)
                eq_units -= sold / tr[k]
                e_now = eq_units * tr[k]
                loan -= sold
                spx_traded[k] -= sold
                if loan <= TOL:
                    loan = 0.0
                    loan_repaid_on = date
            if on_call == "liquidate" and sid in LEVERED and loan > 0.0:   # the forced sale: SPX, then the calls, then the cash, until LTV = ltv_after_call
                calls_now = marks[k] + prem_today
                lv_now = lv_equity * e_now + lv_calls * calls_now + lv_cash * cash
                if loan > margin_call * lv_now + TOL:
                    ltv_before = loan / lv_now if lv_now > 0.0 else np.inf
                    s_, x_, p_ = liquidation_sales(e_now, calls_now, cash, loan, lv_equity, lv_calls, lv_cash, ltv_after_call)
                    if s_ > 0.0:
                        eq_units -= s_ / tr[k]
                        e_now = eq_units * tr[k]
                        spx_traded[k] -= s_
                        loan -= s_
                    if x_ > 0.0:
                        frac = min(x_ / calls_now, 1.0)
                        sell_calls(k, frac)
                        prem_today *= 1.0 - frac
                        loan -= x_
                    if p_ > 0.0:
                        cash -= p_
                        loan -= p_
                    if loan <= TOL:
                        loan = 0.0
                        if loan_repaid_on is None:
                            loan_repaid_on = date
                    lv_after = lv_equity * e_now + lv_calls * (marks[k] + prem_today) + lv_cash * cash
                    called[k] = True
                    liq_day[k] = s_ + x_ + p_
                    liquidations.append(Liquidation(date, float(ltv_before), float(s_), float(x_), float(p_), float(loan / lv_after) if lv_after > 0.0 else (np.inf if loan > 0.0 else 0.0)))
            pay_day[k] = payoff_today
            book(k, prem_today)
            call_pnl[k] = marks_pre + payoff_today - call_val[k - 1]
        nav = E + call_val + cash_arr - loan_arr
        gap = np.diff(nav) - (eq_pnl[1:] + call_pnl[1:] + cash_int[1:] - interest[1:] - tax_day[1:])
        tol = TOL * np.maximum(1.0, np.abs(nav[1:]) / 1e9)
        if (np.abs(gap) > tol).any():
            k = int(np.argmax(np.abs(gap) - tol)) + 1
            raise AccountingIdentityError(f"scenario {sid}, {pd.Timestamp(dates[k]).date()}: ΔNAV − P&L = {gap[k - 1]:.3e} USD")
        lending_value = lv_equity * E + lv_calls * call_val + lv_cash * cash_arr
        levered = sid in LEVERED
        with np.errstate(divide="ignore", invalid="ignore"):
            ltv = np.where(lending_value > 0.0, loan_arr / lending_value, np.where(loan_arr > 0.0, np.inf, 0.0)) if levered else np.full(n, np.nan)
        headroom = margin_call * lending_value - loan_arr if levered else np.full(n, np.nan)
        in_call = ((headroom < 0.0) | called) if levered else np.zeros(n, dtype=bool)
        exposure = E + deltas
        out = pd.DataFrame({
            "spx_tr": tr, "spxfp": S, "E": E, "call_val": call_val, "call_notional": call_notional, "call_delta": deltas, "exposure": exposure, "n_calls": n_calls,
            "strike": strike_arr, "cash": cash_arr, "loan": loan_arr, "nav": nav,
            "lending_value": lending_value, "ltv": ltv, "headroom": headroom, "margin_call": in_call,
            "interest": interest, "tax": tax_day, "eq_pnl": eq_pnl, "call_pnl": call_pnl, "cash_int": cash_int, "spx_traded": spx_traded, "expiry": is_expiry,
            "interest_cum": np.cumsum(interest), "tax_cum": np.cumsum(tax_day), "premiums_cum": np.cumsum(prem_day), "payoffs_cum": np.cumsum(pay_day),
            "tbill": tbill,   # the 3-month T-bill of the day (fraction): the cash leg's rate, and the Sharpe's risk-free rate in ``summary``
            "liquidated": liq_day,
        }, index=pd.DatetimeIndex(d.date, name="date"))
        first_call = pd.Timestamp(out.index[int(np.argmax(in_call))]) if in_call.any() else None
        k_build_end = build_days[-1] if buys_calls else 0
        info = ScenarioInfo(sid, LABELS[sid], pd.Timestamp(dates[0]), pd.Timestamp(dates[-1]), float(E[0]), (loan_build if sid == "4" else (loan_frac * capital if sid == "2" else 0.0)),
                            premium0, notional0, c0, dl0, sig0, tuple(expiries),
                            first_call, int(in_call.sum()), float(np.nanmax(ltv)) if levered else float("nan"), loan_repaid_on, wiped_out_on,
                            any(t.k_exp < 0 for t in tranches), rate_col, vol_col,
                            premium_build, notional_build, float(exposure[k_build_end]), pd.Timestamp(dates[k_build_end]), len(build_days) if buys_calls else 0,
                            liquidations=tuple(liquidations))
        return out, info

    paths: dict[str, pd.DataFrame] = {}
    infos: dict[str, ScenarioInfo] = {}
    for sid in scenarios:
        paths[sid], infos[sid] = run(sid)
    return paths, infos


def default_starts(file: Path | str | None = None) -> dict[str, pd.Timestamp]:
    """The five start dates of the page, in date order: the first data day, the dot-com peak and bottom (the first ≥ 20 % correction on the
    SPX price index) and the GFC peak and bottom (the second correction). Fewer corrections in the data give fewer starts."""
    d = _load(str(file or DAILY_FILE))
    out = {"data start": pd.Timestamp(d.date.iloc[0])}
    corr = corrections(0.20, file=file, on="price")
    if len(corr) >= 1:
        out["dot-com peak"] = pd.Timestamp(corr["peak"].iloc[0])
        out["dot-com bottom"] = pd.Timestamp(corr["trough"].iloc[0])
    if len(corr) >= 2:
        out["GFC peak"] = pd.Timestamp(corr["peak"].iloc[1])
        out["GFC bottom"] = pd.Timestamp(corr["trough"].iloc[1])
    return out


def daily_returns(nav: F64) -> F64:
    """Simple step returns r_k = NAV_k / NAV_{k−1} − 1 on the days with NAV_{k−1} > 0 (one step = one trading day of the daily file).
    A wiped-out path is passed cut at its wipe-out day: that day's return is −100 %, nothing follows."""
    nav = np.asarray(nav, dtype=np.float64)
    prev, nxt = nav[:-1], nav[1:]
    keep = prev > 0.0                  # masked before dividing: never a division by a zero NAV
    return np.asarray(nxt[keep] / prev[keep] - 1.0, dtype=np.float64)


@dataclass(frozen=True)
class RiskMeasures:
    """Risk measures of one path (docs/METHODOLOGY.md §10, "Risk measures"): fractions, losses positive; NaN with fewer than two returns."""
    n: int                 # daily returns in the sample
    volatility: float      # std(ddof = 1) × √252
    var95: float           # 1-day historical VaR at 95 %
    cvar95: float          # the mean loss on the days with a loss ≥ VaR 95
    var99: float
    cvar99: float
    sharpe: float          # (annualised return − T-bill average) / volatility; NaN when the volatility is 0


def risk_measures(nav: F64, ann_return: float, tbill_avg: float, steps_per_year: int = TRADING_DAYS) -> RiskMeasures:
    """Volatility, 1-day historical VaR/CVaR at 95 and 99 % (``metrics.var_cvar``, numpy's linear percentile) and the Sharpe-like ratio of a
    NAV path, on its own step returns (``daily_returns``)."""
    r = daily_returns(nav)
    n = int(r.size)
    nan = float("nan")
    if n < 2:                          # std(ddof = 1) and the percentile need two returns (a RuntimeWarning is an error in the suite)
        return RiskMeasures(n, nan, nan, nan, nan, nan, nan)
    vol = float(r.std(ddof=1)) * float(np.sqrt(steps_per_year))
    v95, c95 = var_cvar(r, 0.95)
    v99, c99 = var_cvar(r, 0.99)
    sharpe = (ann_return - tbill_avg) / vol if vol > 0.0 else nan
    return RiskMeasures(n, vol, v95 + 0.0, c95 + 0.0, v99 + 0.0, c99 + 0.0, sharpe)   # + 0.0: a flat path gives −0.0 from −percentile(0)


SUMMARY_ROWS = ("NAV today", "total return", "annualised return", "lowest NAV", "lowest NAV on", "max drawdown", "max drawdown on", "returns counted",
                "volatility", "VaR 95 %", "CVaR 95 %", "VaR 99 %", "CVaR 99 %", "T-bill average", "Sharpe ratio", "margin call first on", "LTV at the first margin call",
                "days in margin call", "liquidations", "sold on margin calls", "max LTV", "loan repaid on", "build completed on", "premium paid in the build", "exposure at the end of the build",
                "calls expired", "expired worthless", "wiped out on", "premiums paid", "payoffs received", "tax paid",
                "interest paid", "SPX bought at expiries", "SPX sold at expiries", "SPX today", "calls today", "calls alive today", "call notional today", "exposure today",
                "cash today", "loan today", "last call not yet expired")


def summary(paths: dict[str, pd.DataFrame], infos: dict[str, ScenarioInfo]) -> pd.DataFrame:
    """One column per scenario, one row per fact (``SUMMARY_ROWS``): USD, fractions, counts, dates (NaT / None where not applicable). The risk
    rows are read on the scenario's daily NAV returns, a wiped-out path cut at its wipe-out day (``risk_measures``)."""
    cols: dict[str, dict[str, object]] = {}
    for sid, p in paths.items():
        info = infos[sid]
        nav, nav0 = p["nav"], float(p["nav"].iloc[0])
        years = (p.index[-1] - p.index[0]).days / 365.25
        end = float(nav.iloc[-1])
        ann = (max(end, 0.0) / nav0) ** (1.0 / years) - 1.0
        dd = nav / nav.cummax() - 1.0
        stop = int((p.index <= info.wiped_out_on).sum()) if info.wiped_out_on is not None else len(p)   # the wipe-out day included, nothing after
        tbill_avg = float(p["tbill"].mean())
        rm = risk_measures(nav.to_numpy(dtype=np.float64)[:stop], ann, tbill_avg)
        first = info.margin_call_first
        cols[sid] = {
            "NAV today": end, "total return": end / nav0 - 1.0, "annualised return": ann,
            "lowest NAV": float(nav.min()), "lowest NAV on": nav.idxmin(), "max drawdown": float(dd.min()), "max drawdown on": dd.idxmin(),
            "returns counted": rm.n, "volatility": rm.volatility, "VaR 95 %": rm.var95, "CVaR 95 %": rm.cvar95, "VaR 99 %": rm.var99, "CVaR 99 %": rm.cvar99,
            "T-bill average": tbill_avg, "Sharpe ratio": rm.sharpe,
            "margin call first on": first if first is not None else pd.NaT, "LTV at the first margin call": float(p["ltv"].loc[first]) if first is not None else np.nan,
            "days in margin call": info.days_in_margin_call, "liquidations": len(info.liquidations), "sold on margin calls": float(p["liquidated"].sum()), "max LTV": info.max_ltv, "loan repaid on": info.loan_repaid_on if info.loan_repaid_on is not None else pd.NaT,
            "build completed on": info.build_end if info.n_tranches else pd.NaT, "premium paid in the build": info.premium_build, "exposure at the end of the build": info.exposure_build,
            "calls expired": len(info.expiries), "expired worthless": sum(1 for e in info.expiries if e.worthless), "wiped out on": info.wiped_out_on if info.wiped_out_on is not None else pd.NaT,
            "premiums paid": float(p["premiums_cum"].iloc[-1]), "payoffs received": float(p["payoffs_cum"].iloc[-1]), "tax paid": float(p["tax_cum"].iloc[-1]),
            "interest paid": float(p["interest_cum"].iloc[-1]), "SPX bought at expiries": float(sum(max(e.spx_traded, 0.0) for e in info.expiries)),
            "SPX sold at expiries": float(-sum(min(e.spx_traded, 0.0) for e in info.expiries)) + 0.0,   # + 0.0: no −0
            "SPX today": float(p["E"].iloc[-1]), "calls today": float(p["call_val"].iloc[-1]), "calls alive today": int(p["n_calls"].iloc[-1]),
            "call notional today": float(p["call_notional"].iloc[-1]), "exposure today": float(p["exposure"].iloc[-1]),
            "cash today": float(p["cash"].iloc[-1]), "loan today": float(p["loan"].iloc[-1]), "last call not yet expired": info.unexpired,
        }
    return pd.DataFrame(cols).reindex(list(SUMMARY_ROWS))


PATH_COLUMNS = ("nav", "E", "call_val", "call_notional", "call_delta", "exposure", "n_calls", "cash", "loan", "ltv", "headroom", "margin_call", "tax_cum", "interest_cum",
                "premiums_cum", "payoffs_cum", "spx_tr", "spxfp")
_UNSCALED = ("ltv", "margin_call", "n_calls", "spx_tr", "spxfp")


def paths_table(runs: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Every (start, scenario, day) as one long table for export, USD in millions (LTV a fraction, ``n_calls`` a count, ``spx_tr`` a ratio):
    ``runs`` maps a start label to the scenario paths of ``simulate_scenarios``."""
    frames = []
    for label, paths in runs.items():
        for sid, p in paths.items():
            f = p[list(PATH_COLUMNS)].copy()
            for c in PATH_COLUMNS:
                if c not in _UNSCALED:
                    f[c] = (f[c] / 1e6).round(3)
            f.insert(0, "scenario", f"{sid} {LABELS[sid]}")
            f.insert(0, "start", label)
            frames.append(f.reset_index())
    if not frames:
        return pd.DataFrame(columns=["start", "scenario", "date", *PATH_COLUMNS])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.DatetimeIndex(out["date"]).date
    return out[["start", "scenario", "date", *PATH_COLUMNS]]
