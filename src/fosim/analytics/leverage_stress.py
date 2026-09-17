"""Leverage vs. call rotation — historical stress test on one path (docs/METHODOLOGY.md §9, Assumptions 19p).

Setup at t₀: equity E₀ in SPX (dividends reinvested net of withholding), a Lombard loan L₀ at (base rate + spread)
with interest capitalised, other investments funded by the loan (0 % LTV, excluded from both NAVs).

    A (current):  E_A(t) = E₀ · TR(t)/TR(t₀);  L(t) accrues daily, simple ACT/360, capitalised
                  lending value LV_A = ℓ_E · E_A (the most the bank lends against the equity);  LTV_A = L / LV_A
                  headroom_A = LV_A − L   (spare borrowing capacity; < 0 is a margin call, LTV > 100 %, |·| the cure amount)
    B (rotation): in weekly steps over a year (default; or one shot), sell x_w of equity, buy an ATM call on SPXFP with
                  notional N_w = x_w / δ_w so that δ_w · N_w replaces the exposure sold (δ_w = the model delta of the ATM call
                  that day, ≈ 45–55 %), premium c_w · N_w, and the cash x_w − c_w · N_w repays an equal share of the loan
                  then outstanding  ⇒  x_w = (L_w / remaining steps) / (1 − c_w/δ_w); the loan is gone after the last step
                  E_B(t) = (E₀ − x) · TR(t)/TR(t₀);  calls marked daily (BSM, q = r, remaining tenor);
                  at expiry the payoff is cashed; a call that ends in the money is replaced by a new ATM call on the same
                  index units, paid from the payoff, then cash, then by selling equity (payoff left over: T-bills, equity
                  or more calls); a call that expires worthless lapses (nothing to reinvest) unless ``replace_worthless``
                  dry powder_B = capacity_B + cash (an optional cash buffer can be kept at t₀)
                  capacity_B = ℓ_E · E_B + ℓ_C · call value   (the lending value; no loan)

NAV_A = E_A − L, NAV_B = E_B + call value + cash. The accounting identity ΔNAV = market P&L − interest (+ cash
interest) is asserted every day for both. c₀ and the roll premiums come from the tenor's Treasury / implied-vol
columns (``ust_{n}y`` / ``iv_{n}y``, 5-year fallback reported).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from fosim.analytics.call_vs_cash import DAILY_FILE, tenor_columns, total_return_index
from fosim.pricing.black_scholes import bsm_greeks, bsm_price

REQUIRED = ("date", "spxfp", "spx_px_last", "spx_div_yld", "loan_base", "tbill_3m", "ust_5y", "iv_5y")
SURPLUS_POLICIES = ("cash", "equity", "calls")
TOL = 1e-6


class AccountingIdentityError(AssertionError):
    """ΔNAV does not equal the sum of the day's P&L components."""


@dataclass(frozen=True)
class Roll:
    date: pd.Timestamp
    premium_frac: float
    payoff: float
    premium_paid: float
    equity_sold: float


@dataclass(frozen=True)
class StressInfo:
    start: pd.Timestamp
    end: pd.Timestamp
    premium0: float
    rotation: float
    delta0: float
    notional0: float
    units0: float
    loan_base0: float
    rate_col: str
    vol_col: str
    tenor: float
    rolls: tuple[Roll, ...]
    builds: tuple[Roll, ...] = ()        # one entry per build step: premium fraction, premium paid, equity sold
    build_end: pd.Timestamp | None = None

    @property
    def fallback(self) -> bool:
        return abs(self.tenor - 5.0) > 1e-9 and (self.rate_col == "ust_5y" or self.vol_col == "iv_5y")


@lru_cache(maxsize=4)
def _load_cached(file: str, mtime_ns: int, size: int) -> pd.DataFrame:
    d = pd.read_csv(file, parse_dates=["date"])
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        raise ValueError(f"market data file lacks columns {missing}")
    return d.sort_values("date").reset_index(drop=True)


def _load(file: str) -> pd.DataFrame:
    """The daily market file, cached on (path, modification time, size) so a rewritten file is re-read."""
    st = Path(file).stat()
    return _load_cached(file, st.st_mtime_ns, st.st_size)


def _nearest_index(dates: np.ndarray, target: np.datetime64) -> int:
    """Position of the trading day nearest to ``target`` (ties → the earlier day)."""
    k = int(np.searchsorted(dates, target))
    if k == 0:
        return 0
    if k >= len(dates):
        return len(dates) - 1
    before, after = dates[k - 1], dates[k]
    return k - 1 if (target - before) <= (after - target) else k


def simulate(
    start: str | pd.Timestamp, end: str | pd.Timestamp, equity0: float = 1000e6, loan0: float = 250e6, spread: float = 0.0075,
    ltv_equity: float = 0.75, ltv_call: float = 0.0, tenor: float = 5.0, wht: float = 0.15,  # ltv_* are lending values (advance rates)
    surplus: str = "cash", cash_buffer: float = 0.0, delta: float | None = None, build_tranches: int = 52, replace_worthless: bool = False,
    file: Path | str | None = None,
) -> tuple[pd.DataFrame, StressInfo]:
    """Daily path of both setups from ``start`` to ``end`` (nearest trading days). See the module docstring.

    ``surplus``: what B does with payoff cash left after paying the roll premium — ``"cash"`` (T-bills), ``"equity"`` (buy SPX)
    or ``"calls"`` (buy more ATM calls at the same premium fraction, so the whole payoff stays exposed to the index).
    ``replace_worthless``: a call that expires worthless is not replaced (default: nothing to reinvest, no SPX sold); ``True`` buys
    the new ATM call anyway, paid from cash then by selling SPX. A call that expires in the money is always replaced.
    ``cash_buffer``: extra equity rotated at t₀ so that B keeps this much cash after repaying the loan (0 = fully invested).
    ``delta``: the call delta used to size the sleeve (notional = equity sold / delta). ``None`` (default) uses the model delta of
    the ATM call at t₀, ∂C/∂S of BSM with q = r — implicit in the option being at the money; a number overrides it (scripted use).
    ``build_tranches``: the rotation is done in this many equal steps a week apart (52 = weekly over a year, default; 1 = one shot);
    each step sells the equity that, after the tranche's premium, repays an equal share of the loan then outstanding.
    """
    if equity0 <= 0 or loan0 < 0 or tenor <= 0 or cash_buffer < 0:
        raise ValueError("equity0 and tenor must be positive, loan0 and cash_buffer non-negative")
    if not 0.0 < ltv_equity <= 1.0 or not 0.0 <= ltv_call <= 1.0 or not 0.0 <= wht < 1.0:
        raise ValueError("ltv_equity in (0, 1], ltv_call in [0, 1], wht in [0, 1)")
    if delta is not None and not 0.0 < delta <= 1.0:
        raise ValueError("delta must be in (0, 1] or None for the model delta")
    if int(build_tranches) < 1:
        raise ValueError("build_tranches must be ≥ 1")
    if surplus not in SURPLUS_POLICIES:
        raise ValueError(f"surplus must be one of {SURPLUS_POLICIES}")
    full = _load(str(file or DAILY_FILE))
    rate_col, vol_col = tenor_columns(full, tenor)
    d = full.dropna(subset=[c for c in REQUIRED if c != "date"] + [rate_col, vol_col]).reset_index(drop=True)
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
    base = d.loan_base.to_numpy(dtype=np.float64) / 100.0
    tbill = d.tbill_3m.to_numpy(dtype=np.float64) / 100.0
    tr = total_return_index(d.spx_px_last.to_numpy(dtype=np.float64), d.spx_div_yld.to_numpy(dtype=np.float64), d.date.to_numpy(), wht)
    tr = tr / tr[0]

    # ---- A: equity and the capitalised loan
    E_A = equity0 * tr
    loan = np.empty(n)
    loan[0] = loan0
    for k in range(1, n):
        loan[k] = loan[k - 1] * (1.0 + (base[k - 1] + spread) * dt360[k])
    interest = np.concatenate([[0.0], np.diff(loan)])

    # ---- B: the rotation, built in ``build_tranches`` equal steps a week apart (1 = one shot at t₀); the loan is repaid step by step
    months = round(tenor * 12)
    n_tr = int(build_tranches)
    build_days = [0] + [_nearest_index(dates, dates[0] + np.timedelta64(7 * w, "D")) for w in range(1, n_tr)]
    if len(set(build_days)) != n_tr or (n_tr > 1 and build_days[-1] >= n - 1):
        raise ValueError("the build schedule does not fit the path (need distinct trading days a week apart, all before the end)")

    def expiry_of(k: int) -> tuple[np.datetime64, int]:
        """Calendar expiry of a call bought on day k and the index of the nearest trading day (−1 when beyond the data)."""
        target = np.datetime64(pd.Timestamp(dates[k]) + pd.DateOffset(months=months), "D")
        return target, (_nearest_index(dates, target) if target <= dates[-1] else -1)

    def atm(k: int) -> tuple[float, float]:
        """Premium fraction and delta of the ATM call of the tenor on day k (q = r)."""
        c = float(bsm_price(100.0, 100.0, r[k], r[k], iv[k], tenor)) / 100.0
        dl = float(np.asarray(bsm_greeks(100.0, 100.0, r[k], r[k], iv[k], tenor).delta)) if delta is None else float(delta)
        return c, dl

    class Tranche:
        __slots__ = ("delta0", "exp_date", "k_exp", "strike", "units")

        def __init__(self, units: float, strike: float, exp_date: np.datetime64, k_exp: int, delta0: float) -> None:
            self.units, self.strike, self.exp_date, self.k_exp, self.delta0 = units, strike, exp_date, k_exp, delta0

    marks = np.zeros(n)   # model value of the live tranches on each day, accumulated segment by segment (purchase day and expiry day excluded)

    def mark_segment(t: Tranche, k_buy: int) -> None:
        k_end = t.k_exp if t.k_exp > 0 else n
        idx = np.arange(k_buy + 1, k_end)
        if idx.size:
            tau = (t.exp_date - dates[idx]).astype(float) / 365.0
            marks[idx] += t.units * np.asarray(bsm_price(S[idx], np.full(idx.size, t.strike), r[idx], r[idx], iv[idx], tau), dtype=np.float64)

    live: dict[int, list[Tranche]] = {}      # tranches by expiry index
    delta_notional = 0.0                     # Σ δ_i · N_i over live tranches (exposure replaced)
    eq_units = equity0 / tr[0]
    loan_B = loan0
    cash = 0.0
    E_B, call_val, call_notional, cash_B, loan_B_arr = (np.empty(n) for _ in range(5))
    eq_pnl_B, call_pnl_B, cash_int, interest_B, exposure_B = (np.zeros(n) for _ in range(5))
    is_roll = np.zeros(n, dtype=bool)
    rolls: list[Roll] = []
    builds: list[Roll] = []
    tot_sold = tot_notional = tot_premium = 0.0

    def buy(k: int, notional: float, c_k: float, d_k: float) -> Tranche:
        e_date, k_e = expiry_of(k)
        t = Tranche(notional / S[k], S[k], e_date, k_e, d_k)
        live.setdefault(k_e, []).append(t)
        mark_segment(t, k)
        return t

    def build_step(k: int, remaining: int) -> float:
        """Sell equity, buy one tranche, repay the loan's share: returns the premium booked as call value today."""
        nonlocal eq_units, loan_B, cash, delta_notional, tot_sold, tot_notional, tot_premium
        c_k, d_k = atm(k)
        if c_k / d_k >= 1.0:
            raise ValueError(f"premium {c_k:.1%} of notional exceeds the delta {d_k:.0%}: the calls cost more than the exposure they replace")
        repay = loan_B / remaining
        buf = cash_buffer / n_tr
        sold = (repay + buf) / (1.0 - c_k / d_k)
        if sold > eq_units * tr[k] + TOL:
            raise ValueError(f"build step on {pd.Timestamp(dates[k]).date()}: {sold / 1e6:,.0f} m of equity needed, only {eq_units * tr[k] / 1e6:,.0f} m held")
        notional = sold / d_k
        premium = c_k * notional
        eq_units -= sold / tr[k]
        loan_B -= repay
        cash += buf                             # = sold − premium − repay, exactly
        buy(k, notional, c_k, d_k)
        delta_notional += d_k * notional
        tot_sold += sold
        tot_notional += notional
        tot_premium += premium
        builds.append(Roll(pd.Timestamp(dates[k]), c_k, 0.0, premium, sold))
        return premium

    prem_today = build_step(0, n_tr)
    E_B[0], call_val[0], call_notional[0], cash_B[0], loan_B_arr[0] = eq_units * tr[0], prem_today, tot_notional, cash, loan_B
    exposure_B[0] = E_B[0] + delta_notional
    for k in range(1, n):
        eq_pnl_B[k] = eq_units * (tr[k] - tr[k - 1])
        cash_int[k] = cash * tbill[k - 1] * dt360[k]
        cash += cash_int[k]
        interest_B[k] = loan_B * (base[k - 1] + spread) * dt360[k]
        loan_B += interest_B[k]
        new_premium = 0.0
        payoff_today = 0.0
        if k in build_days:
            new_premium += build_step(k, n_tr - build_days.index(k))
        for t in live.pop(k, []):            # tranches expiring today: cash-settle, roll into a new ATM call on the same units
            payoff = t.units * max(S[k] - t.strike, 0.0)
            payoff_today += payoff
            cash += payoff
            delta_notional -= t.delta0 * t.units * t.strike
            if k < n - 1 and payoff <= 0.0 and not replace_worthless:   # expired worthless: nothing to reinvest, the call lapses
                rolls.append(Roll(pd.Timestamp(dates[k]), atm(k)[0], 0.0, 0.0, 0.0))
                is_roll[k] = True
            elif k < n - 1:
                c_k, d_k = atm(k)
                notional = t.units * S[k]            # same index units: the new ATM notional is units × today's index
                premium = c_k * notional
                cash -= premium
                sold = 0.0
                if cash < 0.0:                   # the payoff did not cover the new premium: sell equity for the shortfall
                    sold = -cash
                    if sold > eq_units * tr[k] + TOL:
                        raise ValueError(f"roll on {pd.Timestamp(dates[k]).date()}: premium {premium / 1e6:,.1f} m exceeds payoff, cash and equity")
                    eq_units -= sold / tr[k]
                    cash = 0.0
                if surplus == "equity" and cash > 0.0:
                    eq_units += cash / tr[k]
                    cash = 0.0
                elif surplus == "calls" and cash > 0.0:   # the payoff left buys more ATM calls at the same premium fraction
                    notional += cash / c_k
                    premium += cash
                    cash = 0.0
                nt = buy(k, notional, c_k, d_k)
                delta_notional += d_k * nt.units * nt.strike
                new_premium += premium
                rolls.append(Roll(pd.Timestamp(dates[k]), c_k, payoff, premium, sold))
                is_roll[k] = True
        call_val[k] = marks[k] + new_premium
        call_pnl_B[k] = marks[k] + payoff_today - call_val[k - 1]   # value of the surviving tranches + payoffs realised, vs yesterday's book
        E_B[k] = eq_units * tr[k]
        call_notional[k] = sum(t.units * t.strike for ts in live.values() for t in ts)
        cash_B[k] = cash
        loan_B_arr[k] = loan_B
        exposure_B[k] = E_B[k] + delta_notional

    nav_A = E_A - loan
    nav_B = E_B + call_val + cash_B - loan_B_arr
    eq_pnl_A = np.concatenate([[0.0], np.diff(E_A)])
    # ---- accounting identity, every day, both setups
    gap_A = np.diff(nav_A) - (eq_pnl_A[1:] - interest[1:])
    gap_B = np.diff(nav_B) - (eq_pnl_B[1:] + call_pnl_B[1:] + cash_int[1:] - interest_B[1:])
    for name, gap, nav in (("A", gap_A, nav_A), ("B", gap_B, nav_B)):
        tol = TOL * np.maximum(1.0, np.abs(nav[1:]) / 1e9)   # 1e-6 USD per 1 bn of NAV (Assumptions 35), the float64 limit
        if (np.abs(gap) > tol).any():
            k = int(np.argmax(np.abs(gap) - tol)) + 1
            raise AccountingIdentityError(f"setup {name}, {pd.Timestamp(dates[k]).date()}: ΔNAV − P&L = {gap[k - 1]:.3e} USD")

    drawdown = tr / np.maximum.accumulate(tr) - 1.0
    lv_B = ltv_equity * E_B + ltv_call * call_val
    out = pd.DataFrame({
        "spx_tr": tr, "spxfp": S, "drawdown": drawdown, "loan_rate": base + spread,
        "E_A": E_A, "loan": loan, "lending_value_A": ltv_equity * E_A, "ltv_A": loan / (ltv_equity * E_A), "headroom_A": ltv_equity * E_A - loan, "nav_A": nav_A, "interest_A": interest, "eq_pnl_A": eq_pnl_A,
        "E_B": E_B, "call_val": call_val, "call_notional": call_notional, "cash_B": cash_B, "loan_B": loan_B_arr, "cap_B": lv_B, "dry_powder_B": lv_B - loan_B_arr + cash_B, "nav_B": nav_B,
        "exposure_B": exposure_B, "eq_pnl_B": eq_pnl_B, "call_pnl_B": call_pnl_B, "cash_int_B": cash_int, "interest_B": interest_B, "roll": is_roll,
    }, index=pd.DatetimeIndex(d.date, name="date"))
    info = StressInfo(start=pd.Timestamp(dates[0]), end=pd.Timestamp(dates[-1]), premium0=tot_premium / tot_notional, rotation=tot_sold, delta0=tot_sold / tot_notional, notional0=tot_notional,
                      units0=sum(b.premium_paid / b.premium_frac for b in builds) / S[0], loan_base0=float(base[0]), rate_col=rate_col, vol_col=vol_col, tenor=float(tenor),
                      rolls=tuple(rolls), builds=tuple(builds), build_end=pd.Timestamp(dates[build_days[-1]]))
    return out, info


def rolling_starts(starts: list[pd.Timestamp] | pd.DatetimeIndex, horizon_years: float, file: Path | str | None = None, until: pd.Timestamp | str | None = None,
                   progress: Callable[[int, int], None] | None = None, **kw: object) -> pd.DataFrame:
    """The same setup put on at each start date and run for ``horizon_years`` (windows ending beyond the data are skipped), or, with
    ``until``, from each start to that date (starts less than 30 days before it are skipped); one row per start."""
    full = _load(str(file or DAILY_FILE))
    last = full.date.iloc[-1]
    n_build = kw.get("build_tranches", 52)
    build_days = 7 * (int(n_build if isinstance(n_build, int | float) else 52) - 1)   # the build must be complete ≥ 30 days before the end
    rows = []
    all_starts = pd.DatetimeIndex(starts)
    for i, s in enumerate(all_starts):
        if progress is not None and (i % 25 == 0 or i == len(all_starts) - 1):
            progress(i + 1, len(all_starts))
        if until is not None:
            e = min(pd.Timestamp(until), last)
            if s > e - pd.Timedelta(30 + build_days, unit="D"):
                continue
        else:
            e = s + pd.DateOffset(months=round(horizon_years * 12))
            if e > last:
                continue
        p, info = simulate(s, e, file=file, **kw)  # type: ignore[arg-type]
        t = int(p["drawdown"].to_numpy().argmin())
        m = int(p["headroom_A"].to_numpy().argmin())
        b = int(p["dry_powder_B"].to_numpy().argmin())
        rows.append({
            "start": info.start, "premium0": info.premium0, "rotation": info.rotation, "notional0": info.notional0,
            "A: max LTV": float(p["ltv_A"].max()), "A: min headroom": float(p["headroom_A"].iloc[m]), "A: min headroom date": p.index[m], "A: margin call": bool(p["headroom_A"].iloc[m] < 0.0),
            "B: min dry powder": float(p["dry_powder_B"].iloc[b]), "B: min dry powder date": p.index[b],
            "trough": p.index[t], "drawdown at trough": float(p["drawdown"].iloc[t]), "B: capacity at trough": float(p["cap_B"].iloc[t]), "B: dry powder at trough": float(p["dry_powder_B"].iloc[t]), "A: headroom at trough": float(p["headroom_A"].iloc[t]),
            "A: LTV at trough": float(p["ltv_A"].iloc[t]), "interest paid A": float(p["interest_A"].sum()),
            "B: min borrowing capacity": float((p["cap_B"] - p["loan_B"]).min()), "years": float((p.index[-1] - p.index[0]).days / 365.25),
            "NAV A end": float(p["nav_A"].iloc[-1]), "NAV B end": float(p["nav_B"].iloc[-1]), "end": info.end, "rolls": len(info.rolls),
            "A return": float(p["nav_A"].iloc[-1] / p["nav_A"].iloc[0] - 1.0), "B return": float(p["nav_B"].iloc[-1] / p["nav_B"].iloc[0] - 1.0),
            "B: max LTV": float((p["loan_B"] / p["cap_B"]).max()),
            "premiums paid": float(sum(b.premium_paid for b in info.builds) + sum(r.premium_paid for r in info.rolls)), "payoffs received": float(sum(r.payoff for r in info.rolls)),
            "equity sold at rolls": float(sum(r.equity_sold for r in info.rolls)),
            "end: equity A": float(p["E_A"].iloc[-1]), "end: loan A": float(p["loan"].iloc[-1]),
            "end: equity B": float(p["E_B"].iloc[-1]), "end: calls B": float(p["call_val"].iloc[-1]), "end: cash B": float(p["cash_B"].iloc[-1]), "end: loan B": float(p["loan_B"].iloc[-1]),
        })
    return pd.DataFrame(rows).set_index("start") if rows else pd.DataFrame()
