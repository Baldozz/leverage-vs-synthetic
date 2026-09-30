"""Five ways to hold the capital — the investor's scenarios on one historical path (docs/METHODOLOGY.md §10, Assumptions 40).

Capital C (650 m), loan share ℓ (25 %), call share κ (25 %), 5-year ATM calls on SPXFP at a fixed premium c (14.5 % of notional):

    1  Long the market:          C in SPX
    2  Levered long:             (1 + ℓ)·C in SPX, a loan ℓ·C at a flat rate (5.5 %), interest capitalised, repaid from SPX after one tenor
    3  75 % SPX + 25 % calls:    (1 − κ)·C in SPX, κ·C of premium in calls; at expiry tax on the profit, then the whole portfolio back to (1 − κ)/κ
    4  Long + calls on a loan:   C in SPX, κ·C of premium funded by a loan κ·C; at expiry tax, the loan repaid from the proceeds, the rest in new calls
    5  All in calls:             C of premium; at expiry the payoff buys new calls (untaxed unless asked); a worthless expiry ends it

SPX with dividends reinvested net of withholding (TR), the loan simple ACT/360 capitalised daily, calls on SPXFP marked with Black–Scholes
(q = r = the tenor's Treasury of the day, ACT/365) at the vol implied by the fixed premium on the purchase day — or, with ``premium=None``, at
the market's implied vol of the day. Tax = rate × max(payoff − premium paid, 0) at expiry, no loss carry-forward. Margin call (2 and 4): the
loan above ``margin_call`` × the lending value of the holdings; flagged, no forced sale. NAV = SPX + calls + cash − loan; the accounting
identity ΔNAV = ΔSPX + Δcalls + cash interest − loan interest − tax is asserted every day for every scenario.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fosim.analytics.call_vs_cash import DAILY_FILE, tenor_columns, total_return_index
from fosim.analytics.leverage_stress import (
    TOL,
    AccountingIdentityError,
    _load,
    _nearest_index,
    corrections,
)
from fosim.pricing.black_scholes import bsm_greeks, bsm_price
from fosim.pricing.implied_vol import implied_vol

SCENARIOS = ("1", "2", "3", "4", "5")
LABELS = {"1": "Long the market", "2": "Levered long", "3": "75 % SPX + 25 % calls", "4": "Long + calls on a loan", "5": "All in calls"}
LEVERED = ("2", "4")
REPAY = ("tenor", "never")             # scenario 2: the loan repaid from SPX after one tenor, or rolled up to the end
REBALANCE = ("portfolio", "proceeds")  # scenario 3 at expiry: the whole portfolio back to (1 − κ)/κ, or the after-tax proceeds split (1 − κ)/κ


@dataclass(frozen=True)
class Expiry:
    """One call expiry: what it paid, what was taken off, what was done with the rest."""
    date: pd.Timestamp
    strike: float
    payoff: float
    premium_paid: float      # for the call that expired
    tax: float
    loan_repaid: float
    spx_traded: float        # SPX bought (+) or sold (−) on the day
    new_premium: float
    new_notional: float
    worthless: bool
    wiped_out: bool = False  # scenario 5: nothing left to reinvest


@dataclass(frozen=True)
class ScenarioInfo:
    id: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    spx0: float
    loan0: float
    premium0: float          # premium paid at t₀, USD
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
    unexpired: bool          # the last call has not reached expiry: its value on the end day is a model mark
    rate_col: str
    vol_col: str


def simulate_scenarios(
    start: str | pd.Timestamp, end: str | pd.Timestamp, capital: float = 650e6, loan_frac: float = 0.25, call_frac: float = 0.25,
    loan_rate: float | None = 0.055, spread: float = 0.0075, premium: float | None = 0.145, tenor: float = 5.0,
    tax_rate: float = 0.24, tax_on: tuple[str, ...] = ("3", "4"), lv_equity: float = 0.75, lv_calls: float = 0.0, lv_cash: float = 0.90,
    margin_call: float = 0.90, wht: float = 0.15, repay: str = "tenor", rebalance: str = "portfolio", file: Path | str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, ScenarioInfo]]:
    """Daily path of the five scenarios from ``start`` to ``end`` (nearest trading days). See the module docstring.

    ``loan_rate``: flat annual rate of the loans (0.055 default); ``None`` uses the 3-month base rate of the day + ``spread``.
    ``premium``: the call premium as a fraction of notional (0.145 default), the calls marked at the vol it implies on the purchase day;
    ``None`` prices and marks at the market's implied vol of the day. ``tax_on``: the scenarios whose call profits are taxed at expiry.
    ``repay``: scenario 2's loan, ``"tenor"`` (sold SPX on the trading day nearest t₀ + tenor) or ``"never"``. ``rebalance``: scenario 3
    at expiry, ``"portfolio"`` (SPX sold or bought so the whole portfolio is back to (1 − κ) SPX / κ calls) or ``"proceeds"`` (the
    after-tax proceeds split (1 − κ)/κ, the SPX untouched). Returns one path per scenario id and one ``ScenarioInfo`` each.
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
    if repay not in REPAY or rebalance not in REBALANCE:
        raise ValueError(f"repay must be one of {REPAY}, rebalance one of {REBALANCE}")
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

    k_repay = -1
    if repay == "tenor":
        _, k_repay = expiry_of(0)

    class Tranche:
        __slots__ = ("exp_date", "k_buy", "k_exp", "premium", "sigma", "strike", "units")

        def __init__(self, units: float, strike: float, exp_date: np.datetime64, k_exp: int, sigma: float, k_buy: int, prem: float) -> None:
            self.units, self.strike, self.exp_date, self.k_exp, self.sigma, self.k_buy, self.premium = units, strike, exp_date, k_exp, sigma, k_buy, prem

    def run(sid: str) -> tuple[pd.DataFrame, ScenarioInfo]:
        taxed = sid in tax_on
        E0 = {"1": capital, "2": (1.0 + loan_frac) * capital, "3": (1.0 - call_frac) * capital, "4": capital, "5": 0.0}[sid]
        L0 = {"2": loan_frac * capital, "4": call_frac * capital}.get(sid, 0.0)
        P0 = {"3": call_frac * capital, "4": call_frac * capital, "5": capital}.get(sid, 0.0)
        eq_units = E0 / tr[0]
        loan, cash = L0, 0.0
        marks = np.zeros(n)                       # model value of the live call on each day (purchase and expiry days excluded)
        tranche: Tranche | None = None
        E, call_val, call_notional, strike_arr, cash_arr, loan_arr = (np.empty(n) for _ in range(6))
        eq_pnl, call_pnl, cash_int, interest, tax_day, prem_day, pay_day, spx_traded = (np.zeros(n) for _ in range(8))
        is_expiry = np.zeros(n, dtype=bool)
        expiries: list[Expiry] = []
        wiped_out_on: pd.Timestamp | None = None
        loan_repaid_on: pd.Timestamp | None = None

        def buy(k: int, prem: float) -> Tranche:
            """Buy ``prem`` USD of ATM calls of the tenor on day k; the mark from the next day to the day before expiry."""
            c, sig, _ = atm(k)
            notional = prem / c
            e_date, k_e = expiry_of(k)
            t = Tranche(notional / S[k], S[k], e_date, k_e, sig, k, prem)
            k_end = k_e if k_e > 0 else n
            idx = np.arange(k + 1, k_end)
            if idx.size:
                tau = (e_date - dates[idx]).astype(float) / 365.0
                vol = np.full(idx.size, sig) if premium is not None else iv[idx]
                marks[idx] += t.units * np.asarray(bsm_price(S[idx], np.full(idx.size, t.strike), r[idx], r[idx], vol, tau), dtype=np.float64)
            prem_day[k] += prem
            return t

        c0, sig0, dl0 = atm(0)
        if P0 > 0.0:
            tranche = buy(0, P0)
        E[0], call_val[0], call_notional[0], strike_arr[0], cash_arr[0], loan_arr[0] = eq_units * tr[0], P0, (tranche.units * tranche.strike if tranche else 0.0), (tranche.strike if tranche else np.nan), cash, loan
        for k in range(1, n):
            e_now = eq_units * tr[k]
            eq_pnl[k] = eq_units * (tr[k] - tr[k - 1])
            cash_int[k] = cash * tbill[k - 1] * dt360[k]
            cash += cash_int[k]
            interest[k] = loan * rate[k - 1] * dt360[k]
            loan += interest[k]
            marks_pre = marks[k]
            new_premium = payoff_today = 0.0
            if tranche is not None and tranche.k_exp == k:
                date = pd.Timestamp(dates[k])
                payoff = tranche.units * max(S[k] - tranche.strike, 0.0)
                payoff_today = payoff
                strike, prem_paid = tranche.strike, tranche.premium
                tranche = None
                is_expiry[k] = True
                tax = tax_rate * max(payoff - prem_paid, 0.0) if taxed else 0.0
                cash += payoff - tax
                tax_day[k] = tax
                repaid = bought = new_prem = new_notional = 0.0
                if sid == "4" and loan > 0.0:                 # the loan repaid from the proceeds, then from SPX sold
                    repaid = min(loan, cash)
                    cash -= repaid
                    loan -= repaid
                    if loan > TOL:
                        sold = min(loan, e_now)
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
                        if rebalance == "portfolio":         # the whole portfolio back to (1 − κ) SPX / κ calls
                            total = e_now + cash
                            bought = (1.0 - call_frac) * total - e_now
                            new_prem = call_frac * total
                        elif cash > 0.0:                     # the after-tax proceeds split (1 − κ)/κ, the SPX untouched
                            bought = (1.0 - call_frac) * cash
                            new_prem = call_frac * cash
                        eq_units += bought / tr[k]
                        e_now = eq_units * tr[k]
                        cash -= bought + new_prem
                    elif sid == "4" and loan == 0.0 and cash > 0.0:   # the rest in new calls, no new loan
                        new_prem, cash = cash, 0.0
                    elif sid == "5":
                        if cash > 0.0:
                            new_prem, cash = cash, 0.0
                        else:
                            wiped_out_on = date
                    if new_prem > TOL:
                        tranche = buy(k, new_prem)
                        new_premium = new_prem
                        new_notional = tranche.units * tranche.strike
                    elif new_prem > 0.0:                     # dust: kept as cash rather than a call of a few cents
                        cash += new_prem
                        new_prem = 0.0
                    if sid == "5" and new_prem == 0.0 and wiped_out_on is None:
                        wiped_out_on = date
                spx_traded[k] += bought
                expiries.append(Expiry(date, strike, payoff, prem_paid, tax, repaid, bought, new_prem, new_notional, payoff <= 0.0, sid == "5" and wiped_out_on == date))
            if sid == "2" and k == k_repay and loan > 0.0:  # the loan repaid from SPX after one tenor
                sold = min(loan, e_now)
                eq_units -= sold / tr[k]
                e_now = eq_units * tr[k]
                loan -= sold
                spx_traded[k] -= sold
                if loan <= TOL:
                    loan = 0.0
                    loan_repaid_on = pd.Timestamp(dates[k])
            pay_day[k] = payoff_today
            call_val[k] = marks[k] + new_premium
            call_pnl[k] = marks_pre + payoff_today - call_val[k - 1]
            E[k], cash_arr[k], loan_arr[k] = e_now, cash, loan
            call_notional[k] = tranche.units * tranche.strike if tranche is not None else 0.0
            strike_arr[k] = tranche.strike if tranche is not None else np.nan
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
        in_call = (headroom < 0.0) if levered else np.zeros(n, dtype=bool)
        out = pd.DataFrame({
            "spx_tr": tr, "spxfp": S, "E": E, "call_val": call_val, "call_notional": call_notional, "strike": strike_arr, "cash": cash_arr, "loan": loan_arr, "nav": nav,
            "lending_value": lending_value, "ltv": ltv, "headroom": headroom, "margin_call": in_call,
            "interest": interest, "tax": tax_day, "eq_pnl": eq_pnl, "call_pnl": call_pnl, "cash_int": cash_int, "spx_traded": spx_traded, "expiry": is_expiry,
            "interest_cum": np.cumsum(interest), "tax_cum": np.cumsum(tax_day), "premiums_cum": np.cumsum(prem_day), "payoffs_cum": np.cumsum(pay_day),
        }, index=pd.DatetimeIndex(d.date, name="date"))
        first_call = pd.Timestamp(out.index[int(np.argmax(in_call))]) if in_call.any() else None
        info = ScenarioInfo(sid, LABELS[sid], pd.Timestamp(dates[0]), pd.Timestamp(dates[-1]), E0, L0, P0, P0 / c0 if P0 > 0.0 else 0.0, c0, dl0, sig0, tuple(expiries),
                            first_call, int(in_call.sum()), float(np.nanmax(ltv)) if levered else float("nan"), loan_repaid_on, wiped_out_on,
                            tranche is not None and tranche.k_exp < 0, rate_col, vol_col)
        return out, info

    paths: dict[str, pd.DataFrame] = {}
    infos: dict[str, ScenarioInfo] = {}
    for sid in SCENARIOS:
        paths[sid], infos[sid] = run(sid)
    return paths, infos


def default_starts(file: Path | str | None = None) -> dict[str, pd.Timestamp]:
    """The three start dates of the page: the first data day, the dot-com peak (the first ≥ 20 % correction's peak on the SPX price index)
    and the GFC bottom (the second correction's trough). Fewer corrections in the data give fewer starts."""
    d = _load(str(file or DAILY_FILE))
    out = {"data start": pd.Timestamp(d.date.iloc[0])}
    corr = corrections(0.20, file=file, on="price")
    if len(corr) >= 1:
        out["dot-com peak"] = pd.Timestamp(corr["peak"].iloc[0])
    if len(corr) >= 2:
        out["GFC bottom"] = pd.Timestamp(corr["trough"].iloc[1])
    return out


SUMMARY_ROWS = ("NAV today", "total return", "annualised return", "lowest NAV", "lowest NAV on", "max drawdown", "margin call first on", "LTV at the first margin call",
                "days in margin call", "max LTV", "loan repaid on", "calls expired", "expired worthless", "wiped out on", "premiums paid", "payoffs received", "tax paid",
                "interest paid", "SPX bought at expiries", "SPX sold at expiries", "SPX today", "calls today", "call notional today", "cash today", "loan today", "last call not yet expired")


def summary(paths: dict[str, pd.DataFrame], infos: dict[str, ScenarioInfo]) -> pd.DataFrame:
    """One column per scenario, one row per fact (``SUMMARY_ROWS``): USD, fractions, dates (NaT / None where not applicable)."""
    cols: dict[str, dict[str, object]] = {}
    for sid, p in paths.items():
        info = infos[sid]
        nav, nav0 = p["nav"], float(p["nav"].iloc[0])
        years = (p.index[-1] - p.index[0]).days / 365.25
        end = float(nav.iloc[-1])
        first = info.margin_call_first
        cols[sid] = {
            "NAV today": end, "total return": end / nav0 - 1.0, "annualised return": (max(end, 0.0) / nav0) ** (1.0 / years) - 1.0,
            "lowest NAV": float(nav.min()), "lowest NAV on": nav.idxmin(), "max drawdown": float((nav / nav.cummax() - 1.0).min()),
            "margin call first on": first if first is not None else pd.NaT, "LTV at the first margin call": float(p["ltv"].loc[first]) if first is not None else np.nan,
            "days in margin call": info.days_in_margin_call, "max LTV": info.max_ltv, "loan repaid on": info.loan_repaid_on if info.loan_repaid_on is not None else pd.NaT,
            "calls expired": len(info.expiries), "expired worthless": sum(1 for e in info.expiries if e.worthless), "wiped out on": info.wiped_out_on if info.wiped_out_on is not None else pd.NaT,
            "premiums paid": float(p["premiums_cum"].iloc[-1]), "payoffs received": float(p["payoffs_cum"].iloc[-1]), "tax paid": float(p["tax_cum"].iloc[-1]),
            "interest paid": float(p["interest_cum"].iloc[-1]), "SPX bought at expiries": float(sum(max(e.spx_traded, 0.0) for e in info.expiries)),
            "SPX sold at expiries": float(-sum(min(e.spx_traded, 0.0) for e in info.expiries)),
            "SPX today": float(p["E"].iloc[-1]), "calls today": float(p["call_val"].iloc[-1]), "call notional today": float(p["call_notional"].iloc[-1]),
            "cash today": float(p["cash"].iloc[-1]), "loan today": float(p["loan"].iloc[-1]), "last call not yet expired": info.unexpired,
        }
    return pd.DataFrame(cols).reindex(list(SUMMARY_ROWS))


PATH_COLUMNS = ("nav", "E", "call_val", "call_notional", "cash", "loan", "ltv", "headroom", "margin_call", "tax_cum", "interest_cum", "premiums_cum", "payoffs_cum", "spx_tr", "spxfp")


def paths_table(runs: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Every (start, scenario, day) as one long table for export, USD in millions (LTV a fraction, ``spx_tr`` a ratio): ``runs`` maps a start
    label to the scenario paths of ``simulate_scenarios``."""
    frames = []
    for label, paths in runs.items():
        for sid, p in paths.items():
            f = p[list(PATH_COLUMNS)].copy()
            for c in PATH_COLUMNS:
                if c not in ("ltv", "margin_call", "spx_tr", "spxfp"):
                    f[c] = (f[c] / 1e6).round(3)
            f.insert(0, "scenario", f"{sid} {LABELS[sid]}")
            f.insert(0, "start", label)
            frames.append(f.reset_index())
    if not frames:
        return pd.DataFrame(columns=["start", "scenario", "date", *PATH_COLUMNS])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.DatetimeIndex(out["date"]).date
    return out[["start", "scenario", "date", *PATH_COLUMNS]]
