"""Shared by the pages: the sidebars (one per page, the tenor and the withholding tax common to both), cached engine calls, labels and colours."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import fosim.analytics.call_share_sweep as css  # noqa: E402
import fosim.analytics.call_vs_cash as cvc  # noqa: E402
import fosim.analytics.five_scenarios as fsc  # noqa: E402
import fosim.analytics.leverage_stress as lvs  # noqa: E402

M = 1e6
A, B = "Keep the loan", "Rotate into calls"
RED, BLUE, GREY, ORANGE, GREEN = "#c00000", "#0b2a6f", "#999999", "#e08a1e", "#2e8b57"
LAYOUT = {"template": "plotly_white", "margin": {"l": 40, "r": 20, "t": 30, "b": 40}, "legend": {"orientation": "h", "y": 1.1}}


@dataclass(frozen=True)
class Setup:
    equity: float          # USD
    loan: float            # USD
    spread: float          # over the 3-month base rate, fraction
    lv_equity: float       # lending value of the equity, fraction
    lv_calls: float        # lending value of the calls, fraction
    lv_cash: float         # lending value of the T-bills the rotation holds, fraction
    margin_call: float     # the share of the lending value at which the bank calls, fraction
    wht: float             # dividend withholding tax, fraction
    tenor: float           # years
    build: int             # weekly steps of the rotation (1 = one shot)
    surplus: str           # "cash" | "equity" | "calls"
    replace_worthless: bool = True    # a call that expires worthless: replaced (True) or lapses (False)
    roll: str = "target"              # what an expiring call is replaced on: "target" (the gap to Keep's exposure), "delta" (the same dollar delta), "units" (the same index units)
    rebalance: str = "monthly"        # target rule: when the exposure is checked against the band ("quarterly", "monthly", "none")
    band: float = 0.10                # target rule: the band around Keep's exposure, fraction
    haircut: float = 0.01             # target rule: vol points (fraction) taken off the mark on calls sold early
    below: str = "calls_spx"          # target rule: below the band, "calls" / "spx" from the T-bills, or "calls_spx": calls from the T-bills then from SPX sold for them


_SURPLUS = {"kept in T-bills": "cash", "reinvested in SPX": "equity", "reinvested in more calls": "calls"}
_ROLL = {"Keep's exposure": "target", "the same dollar delta": "delta", "the same index units": "units"}
_REBALANCE = {"every quarter": "quarterly", "every month": "monthly", "never": "none"}
_BELOW = {"ATM calls from the T-bills": "calls", "SPX from the T-bills": "spx", "ATM calls, SPX sold for them when the T-bills run out": "calls_spx"}


@dataclass(frozen=True)
class PremiumSetup:
    tenor: float                 # years (shared with the rotation)
    premium_usd: float           # USD committed per strike date: the call's premium, or the cash investment
    prem_fixed: float | None     # premium as a fraction of notional when fixed; None = the vol and Treasury of the day
    start: date                  # first strike date
    end: date                    # last strike date
    cash_leg: str                # "SPXFP" | "SPX_TR"
    wht: float                   # fraction (shared)


def _restore(key: str, default: object) -> None:
    """A widget drawn on one page only loses its value when the other page is shown: keep a copy under ``_key`` and put it back before drawing."""
    if key not in st.session_state:
        st.session_state[key] = st.session_state.get("_" + key, default)


def _remember(*keys: str) -> None:
    for k in keys:
        st.session_state["_" + k] = st.session_state[k]


def _tenor_widget() -> None:
    _restore("s_tenor", 5)
    st.selectbox("Call tenor (years)", [5, 7, 10], key="s_tenor", help="Tenors with their own Treasury and implied-vol series (Assumptions 19l/19n). Common to both pages.")


def _wht_widget(help_: str) -> None:
    _restore("s_wht", 15.0)
    st.number_input("Dividend withholding tax (%)", 0.0, 50.0, step=1.0, key="s_wht", help=help_)


def sidebar_setup() -> Setup:
    """Page 1's sidebar: the two portfolios. Drawn by the entry script when that page is shown; read back with ``setup``. The page itself
    runs only what its *Launch simulation* button last launched (the sidebar setup, the grid, the start year and the end day)."""
    for k, v in (("s_equity", 1000.0), ("s_loan", 250.0), ("s_spread", 75.0), ("s_lv", 75.0), ("s_margin", 90.0), ("s_weeks", 52), ("s_worthless", "replaced, SPX sold to pay it"), ("s_roll", "Keep's exposure"),
                 ("s_rebalance", "every month"), ("s_band", 10.0), ("s_haircut", 1.0), ("s_below", "ATM calls, SPX sold for them when the T-bills run out"),
                 ("s_surplus", "kept in T-bills"), ("s_lv_calls", 0.0), ("s_lv_cash", 90.0), ("s_grid", "every trading day")):
        _restore(k, v)
    with st.sidebar:
        st.markdown("**Today**")
        st.number_input("SPX exposure (USD m)", 10.0, 100000.0, step=50.0, key="s_equity")
        st.number_input("Lombard loan (USD m)", 0.0, 100000.0, step=10.0, key="s_loan", help="Borrowed against the SPX; interest capitalised.")
        st.number_input("Spread over SOFR (bp)", 0.0, 500.0, step=5.0, key="s_spread", help="Base rate: 3-month LIBOR to 2018, Term SOFR after.")
        st.number_input("Lending value of the SPX (%)", 1.0, 100.0, step=5.0, key="s_lv", help="The most the bank lends against the SPX. LTV = loan ÷ lending value.")
        st.number_input("Margin call at LTV (%)", 1.0, 100.0, step=5.0, key="s_margin", help="The bank calls when the loan exceeds this share of the lending value (PLACEHOLDER 90 %). Dry powder = this share × the lending value of what is held − loan, for both portfolios.")
        st.markdown("**The rotation**")
        _tenor_widget()
        st.number_input("Weeks to complete the rotation", 1, 156, step=1, key="s_weeks", help="Each week: sell SPX, buy one tranche of ATM calls sized by delta, repay an equal share of the loan. 1 = all on the start date.")
        with st.expander("Advanced"):
            _wht_widget("On the dividends of the SPX held. Common to both pages.")
            st.radio("An expiring call is replaced on", list(_ROLL), key="s_roll", help="Keep's exposure: the new call closes the gap between Keep's SPX value and the rotation's live exposure (its SPX plus the calls' dollar delta of the day), paid from the payoff and the T-bills, then by selling SPX; a gap of zero or less buys nothing. The same dollar delta: the expiring call is intrinsic (delta 1), the new ATM call is sized to carry the same exposure — notional = units × index ÷ its delta, about twice the units. The same index units: notional = units × index. Under the last two, paid from the payoff, then the T-bills, then by selling SPX delta-for-delta.")
            target = str(st.session_state["s_roll"]) == "Keep's exposure"
            st.radio("Exposure checked against the band", list(_REBALANCE), key="s_rebalance", disabled=not target, help="Keep's exposure only. On the last trading day of the period, after the build: above the band the excess is sold from the calls, the most in the money first, to the band edge; below it the shortfall is bought from the T-bills as far as they go.")
            st.number_input("Band around Keep's exposure (± %)", 0.0, 50.0, step=1.0, key="s_band", disabled=not target)
            st.number_input("Haircut on calls sold early (vol points)", 0.0, 10.0, step=0.5, key="s_haircut", disabled=not target, help="Taken off the implied vol when a call is sold before expiry at a band check: 0 = sold at the model mark. PLACEHOLDER 1 point.")
            st.radio("Below the band, buy", list(_BELOW), key="s_below", disabled=not target, help="ATM calls of the tenor (about three times the exposure per dollar, the convexity kept) or SPX, from the T-bills only — the shortfall left when they run out is tolerated until the next check or roll; or calls from the T-bills and then from SPX sold for them (each dollar switched adds delta ÷ premium − 1 of exposure), which keeps the exposure matched through a long fall at the cost of turning stock into options at the low.")
            st.radio("A call that expires worthless is", ["replaced, SPX sold to pay it", "not replaced"], key="s_worthless", help="Replaced: under Keep's exposure by the gap; under the other rules a new ATM call on the same index units, paid from the T-bills then by selling SPX delta-for-delta (the excess over the premium goes to T-bills). Not replaced: the call lapses, nothing is bought, no SPX sold.")
            st.radio("Payoff left after a roll", list(_SURPLUS), key="s_surplus", disabled=target, help="What happens to the part of a payoff not needed for the new call's premium (the same dollar delta and the same index units rules).")
            if target:
                st.caption("Not used under Keep's exposure: a leftover payoff stays in T-bills.")
            st.number_input("Lending value of the calls (%)", 0.0, 100.0, step=5.0, key="s_lv_calls")
            st.number_input("Lending value of the T-bills (%)", 0.0, 100.0, step=5.0, key="s_lv_cash", help="The most the bank lends against the T-bills the rotation holds (the payoffs kept in cash). PLACEHOLDER 90 %.")
            st.radio("Start dates", ["every trading day", "every week"], key="s_grid", help="Every trading day takes a few minutes the first time, then it is cached.")
        st.caption("Historical data only, September 1997 to today.")
    _remember("s_equity", "s_loan", "s_spread", "s_lv", "s_margin", "s_tenor", "s_weeks", "s_wht", "s_roll", "s_rebalance", "s_band", "s_haircut", "s_below", "s_worthless", "s_surplus", "s_lv_calls", "s_lv_cash", "s_grid")
    return setup()


def setup() -> Setup:
    """The Setup the sidebar widgets currently show (pages run after the entry script has drawn them)."""
    s = st.session_state
    return Setup(float(s["s_equity"]) * M, float(s["s_loan"]) * M, float(s["s_spread"]) / 1e4, float(s["s_lv"]) / 100.0, float(s["s_lv_calls"]) / 100.0, float(s["s_lv_cash"]) / 100.0,
                 float(s["s_margin"]) / 100.0, float(s["s_wht"]) / 100.0, float(s["s_tenor"]), int(s["s_weeks"]), _SURPLUS[str(s["s_surplus"])], str(s["s_worthless"]).startswith("replaced"), _ROLL[str(s["s_roll"])],
                 _REBALANCE[str(s["s_rebalance"])], float(s["s_band"]) / 100.0, float(s["s_haircut"]) / 100.0, _BELOW[str(s["s_below"])])


@st.cache_data
def strike_range(tenor: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last trading day usable as a strike date at this tenor (the maturity must fall inside the data)."""
    return cvc.strike_date_range(tenor)


def premium_sidebar() -> PremiumSetup:
    """Page 2's sidebar: the call-vs-cash backtest. Drawn by the entry script when that page is shown; read back with ``premium_setup``."""
    _restore("bt_prem_usd", 10.0)
    _restore("bt_fixed", 15.0)
    _restore("bt_leg", "SPXFP")
    with st.sidebar:
        st.markdown("**The call**")
        _tenor_widget()
        st.number_input("Premium / investment (USD m)", 1.0, 1000.0, step=1.0, key="bt_prem_usd", help="Paid for the call on each strike date, or invested in the index instead.")
        tenor = float(st.session_state["s_tenor"])
        # the options carry the tenor, so this radio has no key: the choice is kept in bt_mode_fixed across tenor changes
        mode = st.radio("Premium paid", [f"Market: {tenor:g}y vol and {tenor:g}y Treasury of the day", "Fixed % of notional"], index=1 if st.session_state.get("bt_mode_fixed") else 0)
        st.session_state["bt_mode_fixed"] = not mode.startswith("Market")
        st.number_input("Fixed premium (% of notional)", 1.0, 60.0, step=0.5, key="bt_fixed", disabled=mode.startswith("Market"))
        st.markdown("**Strike dates**")
        first_strike, last_strike = strike_range(tenor)
        lo, hi = first_strike.date(), last_strike.date()
        _restore("bt_start", lo)
        _restore("bt_end", hi)
        # the usable window shrinks as the tenor grows: follow it unless the user has moved the field themselves
        if st.session_state.get("bt_end_auto") in (None, st.session_state["bt_end"]):
            st.session_state["bt_end"] = st.session_state["bt_end_auto"] = hi
        else:
            st.session_state["bt_end"] = min(st.session_state["bt_end"], hi)
        st.session_state["bt_start"] = min(st.session_state["bt_start"], hi)
        st.date_input("First strike date", min_value=lo, max_value=hi, key="bt_start")
        st.date_input("Last strike date", min_value=lo, max_value=hi, key="bt_end", help=f"The data run to {last_strike + pd.DateOffset(months=round(tenor * 12)):%d %b %Y}: at {tenor:g} years the last strike that still reaches maturity is {hi:%d %b %Y}.")
        st.markdown("**The cash investment**")
        st.radio("Invested in", ["SPXFP", "SPX with dividends reinvested"], key="bt_leg", help="The J.P. Morgan chart invests the cash leg in SPXFP itself.")
        _wht_widget("On the dividends of the SPX leg. Common to both pages.")
        st.caption("Historical data only, September 1997 to today.")
    _remember("s_tenor", "bt_prem_usd", "bt_fixed", "bt_start", "bt_end", "bt_leg", "s_wht")
    return premium_setup()


def premium_setup() -> PremiumSetup:
    """The PremiumSetup from the sidebar widgets' session state."""
    s = st.session_state
    return PremiumSetup(float(s["s_tenor"]), float(s["bt_prem_usd"]) * M, float(s["bt_fixed"]) / 100.0 if s.get("bt_mode_fixed") else None, s["bt_start"], s["bt_end"],
                        "SPXFP" if s["bt_leg"] == "SPXFP" else "SPX_TR", float(s["s_wht"]) / 100.0)


@dataclass(frozen=True)
class ScenarioSetup:
    """Page 3's inputs: the investor's five scenarios."""
    capital: float               # USD
    call_frac: float             # share of the capital in call premium (scenarios 3, 4, 5 use it as written)
    loan_frac: float             # share of the capital borrowed (scenarios 2 and 4)
    loan_rate: float | None      # flat annual rate, fraction; None = 3-month base rate + spread of the day
    spread: float                # fraction, used when loan_rate is None
    premium: float | None        # premium as a fraction of notional; None = the market's vol of the day
    tax_rate: float              # fraction of the call's profit taken off at expiry
    tax_on: tuple[str, ...]      # the scenarios taxed
    tenor: float                 # years (shared)
    wht: float                   # fraction (shared)
    lv_equity: float             # lending value of the SPX, fraction
    lv_calls: float              # lending value of the calls, fraction
    margin_call: float           # the share of the lending value at which the bank calls
    repay: str                   # scenario 2's loan: "tenor" | "never"
    repay_calls: str             # scenario 4's loan: "expiry" | "never"
    rebalance: str               # scenario 3 at expiry: "portfolio" | "proceeds"
    sizing: str                  # the call sleeve: "exposure" (exposure_frac of the capital in SPX-equivalent) | "premium" (call_frac of the capital in premium)
    exposure_frac: float         # share of the capital in SPX-equivalent exposure (delta × notional) under "exposure"
    build_steps: int             # the calls bought in this many steps (1 = one shot)
    build_unit: str              # "week" | "month" between steps


_RATE_MODE = {"fixed rate": "fixed", "3-month base rate + spread of the day": "base"}
_PREM_MODE = {"fixed % of notional": "fixed", "market vol of the day": "market"}
_REPAY = {"repaid from SPX after one tenor": "tenor", "rolled up to the end": "never"}
_REPAY4 = {"repaid from the call proceeds, only the surplus in new calls": "surplus", "repaid from the call proceeds, the slot refilled from SPX": "refill_spx", "never repaid: the slot refilled on a new loan": "reborrow"}
_REBAL = {"the whole portfolio, SPX sold or bought": "portfolio", "the after-tax proceeds only, SPX untouched": "proceeds"}
_SIZING = {"premium: % of the capital": "premium", "exposure: % of the capital in SPX-equivalent (delta × notional)": "exposure"}
_BUILD_UNIT = {"weekly": "week", "monthly": "month"}


# the sleeve, the calls and scenario 3's rule are drawn by pages 3 and 4 alike, each under its own key prefix ("f_" page 3, "c_" page 4) so the two
# pages' settings are independent; the tenor and the withholding tax stay the shared s_* widgets
_SCEN_DEFAULTS: dict[str, object] = {"capital": 650.0, "sizing": next(iter(_SIZING)), "exp_pct": 86.0, "call_pct": 25.0, "build_unit": "weekly", "build_steps": 52,
                                     "prem_mode": "fixed % of notional", "prem": 14.5, "tax": 24.0, "rebalance": next(iter(_REBAL))}
_SWEEP_DEFAULTS: dict[str, object] = {"prem_from": 5.0, "prem_to": 50.0, "prem_step": 5.0, "exp_from": 10.0, "exp_to": 100.0, "exp_step": 10.0}   # page 4: the swept range, % of the capital
_SIZING_HELP = ("Premium (the investor's note): this share of the capital is spent on premium — 25 % = 162.5 m, 1,120 m of notional — whatever exposure that buys at the model's delta. "
                "Exposure (option): the sleeve carries this share of the capital in SPX-equivalent (the model delta × the notional) and the premium follows — exposure × premium % ÷ delta of the day; 86 % = the note's 560 m of delta. "
                "Scenario 3 holds the rest in SPX and refills each slot to its share of the portfolio at expiry; scenario 4 borrows the premium; scenario 5 puts everything in calls either way.")


def _capital_widget(p: str, help_: str) -> None:
    st.markdown("**The capital**")
    st.number_input("Capital (USD m)", 10.0, 100000.0, step=50.0, key=f"{p}_capital", help=help_)


def _range_row(p: str, name: str, label: str, hi: float, step: float, disabled: bool) -> None:
    """Three inputs on one row — from / to / step, % of the capital — for the share page 4 sweeps."""
    st.caption(label)
    c1, c2, c3 = st.columns(3)
    c1.number_input("from", 0.0, hi, step=step, key=f"{p}_{name}_from", disabled=disabled)
    c2.number_input("to", 0.0, hi, step=step, key=f"{p}_{name}_to", disabled=disabled)
    c3.number_input("step", 0.5, hi, step=step, key=f"{p}_{name}_step", disabled=disabled)


def _sleeve_block(p: str, sweep: bool) -> bool:
    """The call sleeve: how it is sized (one share on page 3; on page 4 the swept range from / to / step of the active sizing) and how it is
    built. Returns whether it is sized by exposure."""
    st.markdown("**The call sleeve**")
    st.radio("Sized by", list(_SIZING), key=f"{p}_sizing", help=_SIZING_HELP)
    by_exposure = _SIZING[str(st.session_state[f"{p}_sizing"])] == "exposure"
    if sweep:
        _range_row(p, "prem", "Premium share swept (% of the capital)", 99.0, 5.0, disabled=by_exposure)
        _range_row(p, "exp", "Exposure share swept (% of the capital)", 500.0, 10.0, disabled=not by_exposure)
    else:
        st.number_input("Exposure (% of the capital)", 0.0, 500.0, step=5.0, key=f"{p}_exp_pct", disabled=not by_exposure, help="SPX-equivalent exposure of the calls at purchase, as a share of the capital.")
        st.number_input("Premium (% of the capital)", 0.0, 99.0, step=5.0, key=f"{p}_call_pct", disabled=by_exposure, help="Premium spent, as a share of the capital.")
    st.radio("Built", list(_BUILD_UNIT), key=f"{p}_build_unit", horizontal=True, help="The calls are bought in equal slots a week or a month apart, the money waiting in SPX and rotated slot by slot (scenario 4 draws the loan slot by slot). Each slot has its own expiry and is rolled on its own.")
    weekly = _BUILD_UNIT[str(st.session_state[f"{p}_build_unit"])] == "week"
    st.session_state[f"{p}_build_steps"] = int(min(int(st.session_state[f"{p}_build_steps"]), 104 if weekly else 24))
    st.number_input("Steps (1 = all on the start day)", 1, 104 if weekly else 24, step=1, key=f"{p}_build_steps", help="Up to two years: 104 weekly or 24 monthly steps.")
    return by_exposure


def _calls_block(p: str, tax_help: str) -> None:
    st.markdown("**The calls**")
    _tenor_widget()
    st.radio("Premium", list(_PREM_MODE), key=f"{p}_prem_mode", help="Fixed: every call costs this share of its notional, whatever the vol of the day, and is marked at the vol that share implies on its purchase day. Market: priced and marked at the tenor's implied vol of the day (Assumptions 19l).")
    fixed_prem = str(st.session_state[f"{p}_prem_mode"]) == "fixed % of notional"
    st.number_input("Premium (% of notional)", 1.0, 60.0, step=0.5, key=f"{p}_prem", disabled=not fixed_prem)
    st.number_input("Tax on the profit at expiry (%)", 0.0, 90.0, step=1.0, key=f"{p}_tax", help=tax_help)


def _rebalance_widget(p: str, by_exposure: bool) -> None:
    """Scenario 3's rule at a slot's expiry; under exposure sizing only the whole-portfolio refill exists (the engine refuses the other)."""
    if by_exposure:
        st.session_state[f"{p}_rebalance"] = next(iter(_REBAL))
    st.radio("Scenario 3 at a slot's expiry, refill from:", list(_REBAL), key=f"{p}_rebalance", disabled=by_exposure, help="The user's reading (2026-09-29): the slot is refilled to its share of the whole portfolio, SPX sold when the call expired worthless and bought when it paid. The literal alternative splits only the after-tax proceeds (premium sizing only).")


def scenarios_sidebar() -> ScenarioSetup:
    """Page 3's sidebar: the investor's five scenarios. Drawn by the entry script when that page is shown; read back with ``scenario_setup``."""
    for k, v in _SCEN_DEFAULTS.items():
        _restore(f"f_{k}", v)
    for k, v in (("f_loan_pct", 25.0), ("f_rate_mode", "fixed rate"), ("f_rate", 5.5), ("f_spread", 75.0), ("f_lv", 75.0), ("f_lv_calls", 0.0), ("f_margin", 90.0),
                 ("f_repay", "repaid from SPX after one tenor"), ("f_repay4", next(iter(_REPAY4)))):
        _restore(k, v)
    with st.sidebar:
        st.caption("Separate exercise — its inputs are its own; page 1's setup is not used here.")
        _capital_widget("f", "Scenario 1 holds it all in SPX; the other four are built from it as below.")
        st.number_input("Loan (% of the capital)", 0.0, 99.0, step=5.0, key="f_loan_pct", help="Scenario 2: borrowed and invested in SPX on top of the capital. Scenario 4's loan is the call sleeve's premium.")
        by_exposure = _sleeve_block("f", sweep=False)
        st.markdown("**The loan**")
        st.radio("Interest at", list(_RATE_MODE), key="f_rate_mode", help="The investor's flat 5.5 %, or the 3-month LIBOR / Term SOFR of the day plus a spread. Simple interest ACT/360, capitalised daily.")
        fixed_rate = str(st.session_state["f_rate_mode"]) == "fixed rate"
        st.number_input("Flat rate (%)", 0.0, 30.0, step=0.25, key="f_rate", disabled=not fixed_rate)
        st.number_input("Spread over the base rate (bp)", 0.0, 500.0, step=5.0, key="f_spread", disabled=fixed_rate)
        _calls_block("f", "Taken off payoff − premium when positive, at every call expiry, scenarios 3, 4 and 5 alike; no loss carry-forward; no tax on the SPX.")
        st.markdown("**The scenarios' rules**")   # visible, in scenario order 2 → 3 → 4 (user, 2026-10-01)
        st.radio("Scenario 2: the loan is", list(_REPAY), key="f_repay", help="The investor's note: 'the loan rolling up, then paid off' — repaid from SPX on the day the first call of the other scenarios expires (user's reading, 2026-09-29), or never (the NAV is net of it throughout).")
        _rebalance_widget("f", by_exposure)
        st.radio("Scenario 4 at a slot's expiry, the loan is", list(_REPAY4), key="f_repay4", help="Surplus (the note's 'rolling up, then paid off'): the slot's share of the loan is repaid from the call's proceeds, SPX sold for any shortfall, and only the cash left buys a new call — a worthless call ends its slot. Refilled from SPX: the same repayment, then the slot is refilled to its share of the portfolio by selling SPX; the sleeve lives on, unlevered. Never repaid: the after-tax proceeds buy SPX and every replacement call is bought on a new loan; the loan runs to the end and the LTV with it.")
        with st.expander("Advanced"):
            _wht_widget("On the dividends of the SPX held. Common to every page.")
            st.number_input("Lending value of the SPX (%)", 1.0, 100.0, step=5.0, key="f_lv", help="The most the bank lends against the SPX. LTV = loan ÷ lending value.")
            st.number_input("Lending value of the calls (%)", 0.0, 100.0, step=5.0, key="f_lv_calls")
            st.number_input("Margin call at LTV (%)", 1.0, 100.0, step=5.0, key="f_margin", help="The bank calls when the loan exceeds this share of the lending value (PLACEHOLDER 90 %, as on page 1). Flagged on the charts; no forced sale is modelled.")
        st.caption("Historical data only, September 1997 to today; every start is held to the last data day.")
    _remember("f_capital", "f_call_pct", "f_loan_pct", "f_rate_mode", "f_rate", "f_spread", "s_tenor", "f_prem_mode", "f_prem", "f_tax", "s_wht", "f_lv", "f_lv_calls", "f_margin", "f_repay", "f_repay4", "f_rebalance",
              "f_sizing", "f_exp_pct", "f_build_unit", "f_build_steps")
    return scenario_setup()


def scenario_setup() -> ScenarioSetup:
    """The ScenarioSetup the sidebar widgets currently show."""
    s = st.session_state
    return ScenarioSetup(float(s["f_capital"]) * M, float(s["f_call_pct"]) / 100.0, float(s["f_loan_pct"]) / 100.0,
                         float(s["f_rate"]) / 100.0 if _RATE_MODE[str(s["f_rate_mode"])] == "fixed" else None, float(s["f_spread"]) / 1e4,
                         float(s["f_prem"]) / 100.0 if _PREM_MODE[str(s["f_prem_mode"])] == "fixed" else None, float(s["f_tax"]) / 100.0,
                         ("3", "4", "5"), float(s["s_tenor"]), float(s["s_wht"]) / 100.0, float(s["f_lv"]) / 100.0, float(s["f_lv_calls"]) / 100.0,
                         float(s["f_margin"]) / 100.0, _REPAY[str(s["f_repay"])], _REPAY4[str(s["f_repay4"])], _REBAL[str(s["f_rebalance"])],
                         _SIZING[str(s["f_sizing"])], float(s["f_exp_pct"]) / 100.0, int(s["f_build_steps"]), _BUILD_UNIT[str(s["f_build_unit"])])


@st.cache_data(show_spinner=False)
def scenarios_cached(start: str, end: str, s: ScenarioSetup) -> tuple[dict[str, pd.DataFrame], dict[str, fsc.ScenarioInfo]]:
    """The five scenarios from ``start`` to ``end`` on the sidebar setup."""
    return fsc.simulate_scenarios(start, end, capital=s.capital, loan_frac=s.loan_frac, call_frac=s.call_frac, loan_rate=s.loan_rate, spread=s.spread, premium=s.premium,
                                  tenor=s.tenor, tax_rate=s.tax_rate, tax_on=s.tax_on, lv_equity=s.lv_equity, lv_calls=s.lv_calls, margin_call=s.margin_call, wht=s.wht,
                                  repay=s.repay, repay_calls=s.repay_calls, rebalance=s.rebalance, sizing=s.sizing, exposure_frac=s.exposure_frac, build_steps=s.build_steps, build_unit=s.build_unit)


@dataclass(frozen=True)
class SweepSetup:
    """Page 4's inputs: scenario 3 alone, the size of its call sleeve swept."""
    capital: float               # USD
    sizing: str                  # "premium" | "exposure": which share is swept
    shares: tuple[float, ...]    # the grid of shares (fractions of the capital), 0 first; empty when the range in the sidebar is invalid
    premium: float | None        # premium as a fraction of notional; None = the market's vol of the day
    tax_rate: float              # fraction of the call's profit taken off at expiry
    tenor: float                 # years (shared)
    wht: float                   # fraction (shared)
    rebalance: str               # scenario 3 at expiry: "portfolio" | "proceeds"
    build_steps: int             # the calls bought in this many steps (1 = one shot)
    build_unit: str              # "week" | "month" between steps


def sweep_sidebar() -> SweepSetup:
    """Page 4's sidebar: the page-3 inputs that bear on scenario 3 (its own keys, page 3's defaults) and the swept range. Read back with ``sweep_setup``."""
    for k, v in {**_SCEN_DEFAULTS, **_SWEEP_DEFAULTS}.items():
        _restore(f"c_{k}", v)
    with st.sidebar:
        st.caption("Separate exercise — scenario 3 of the five scenarios alone, the size of its call sleeve swept. Its inputs are its own (page 3's defaults); the tenor and the withholding tax are shared with every page.")
        _capital_widget("c", "Scenario 3 holds it in SPX less the premium spent on the call sleeve.")
        by_exposure = _sleeve_block("c", sweep=True)
        _calls_block("c", "Taken off payoff − premium when positive, at every call expiry; no loss carry-forward; no tax on the SPX.")
        st.markdown("**Scenario 3's rule**")
        _rebalance_widget("c", by_exposure)
        with st.expander("Advanced"):
            _wht_widget("On the dividends of the SPX held. Common to every page.")
        st.caption("Historical data only, September 1997 to today; every start is held to the last data day.")
    _remember(*(f"c_{k}" for k in {**_SCEN_DEFAULTS, **_SWEEP_DEFAULTS}), "s_tenor", "s_wht")
    return sweep_setup()


def sweep_setup() -> SweepSetup:
    """The SweepSetup the sidebar widgets currently show; ``shares`` is empty when the swept range is invalid (the page says so)."""
    s = st.session_state
    sizing = _SIZING[str(s["c_sizing"])]
    name = "prem" if sizing == "premium" else "exp"
    lo, hi, step = (float(s[f"c_{name}_{k}"]) / 100.0 for k in ("from", "to", "step"))
    valid = step > 0.0 and 0.0 <= lo <= hi and (sizing == "exposure" or hi < 1.0)
    shares = css.share_grid(lo, hi, step) if valid else ()
    return SweepSetup(float(s["c_capital"]) * M, sizing, shares, float(s["c_prem"]) / 100.0 if _PREM_MODE[str(s["c_prem_mode"])] == "fixed" else None,
                      float(s["c_tax"]) / 100.0, float(s["s_tenor"]), float(s["s_wht"]) / 100.0, _REBAL[str(s["c_rebalance"])], int(s["c_build_steps"]), _BUILD_UNIT[str(s["c_build_unit"])])


def sweep_kwargs(s: SweepSetup) -> dict[str, object]:
    """The sweep setup as ``simulate_scenarios`` keyword arguments (the loan, lending values and margin call are irrelevant to scenario 3)."""
    return {"capital": s.capital, "premium": s.premium, "tenor": s.tenor, "tax_rate": s.tax_rate, "tax_on": ("3", "4", "5"), "wht": s.wht,
            "rebalance": s.rebalance, "build_steps": s.build_steps, "build_unit": s.build_unit}


@st.cache_data(show_spinner=False)
def sweep_cached(start: str, end: str, s: SweepSetup) -> pd.DataFrame:
    """Scenario 3 from ``start`` to ``end`` at every share of the setup: one summary row per share; a progress bar the first time."""
    bar = st.progress(0.0, text=f"Sweeping {len(s.shares)} call shares from {start} …")

    def _p(i: int, n: int) -> None:
        bar.progress(i / n, text=f"Sweeping call share {i} of {n} from {start} …")

    out = css.sweep_call_share(start, end, s.shares, s.sizing, progress=_p, **sweep_kwargs(s))
    bar.empty()
    return out


@st.cache_data(show_spinner=False)
def sweep_grid_cached(end: str, s: SweepSetup) -> pd.DataFrame:
    """Scenario 3 from every month-end start (``call_share_sweep.month_end_starts``) at every share: the long table; a progress bar the first time."""
    starts = css.month_end_starts(end, s.tenor, s.build_steps, s.build_unit)
    bar = st.progress(0.0, text=f"Sweeping {len(starts)} month-end starts × {len(s.shares)} shares …")

    def _p(i: int, n: int) -> None:
        bar.progress(i / n, text=f"Run {i:,} of {n:,} ({len(starts)} starts × {len(s.shares)} shares) …")

    out = css.sweep_grid(starts, end, s.shares, s.sizing, progress=_p, **sweep_kwargs(s))
    bar.empty()
    return out


def data_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    d = lvs._load(str(lvs.DAILY_FILE))
    return pd.Timestamp(d.date.iloc[0]), pd.Timestamp(d.date.iloc[-1])


def _grid(first: str, last: str, freq: str) -> pd.DatetimeIndex:
    """The start dates from ``first`` to ``last``: every trading day of the daily file ('D') or every Friday ('W-FRI')."""
    if freq == "D":
        d = lvs._load(str(lvs.DAILY_FILE))
        return pd.DatetimeIndex(d.date[(d.date >= pd.Timestamp(first)) & (d.date <= pd.Timestamp(last))])
    return pd.date_range(first, last, freq=freq)


def _engine_kwargs(s: Setup) -> dict[str, object]:
    """The sidebar setup as ``simulate`` keyword arguments."""
    return {"equity0": s.equity, "loan0": s.loan, "spread": s.spread, "ltv_equity": s.lv_equity, "ltv_call": s.lv_calls, "ltv_cash": s.lv_cash, "margin_call": s.margin_call, "tenor": s.tenor, "wht": s.wht,
            "surplus": s.surplus, "cash_buffer": 0.0, "delta": None, "build_tranches": s.build, "replace_worthless": s.replace_worthless, "roll": s.roll,
            "rebalance": s.rebalance, "band": s.band, "unwind_haircut": s.haircut, "below": s.below}


@st.cache_data(show_spinner=False)
def all_starts_cached(first: str, last: str, freq: str, s: Setup, until: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The setup put on at every start of the grid ('D' every trading day, 'W-FRI' every week) and held to ``until``: one summary row per
    start, plus the month-end paths of the room before a margin call (A) and the dry powder (B) for every start; a progress bar the first time."""
    starts = _grid(first, last, freq)
    bar = st.progress(0.0, text=f"Simulating {len(starts):,} start dates …")

    def _p(i: int, n: int) -> None:
        bar.progress(i / n, text=f"Simulating start {i:,} of {n:,} …")

    bottoms = tuple(pd.Timestamp(d) for d in lvs.corrections(0.20, wht=s.wht, on="price")["trough"])   # sampled exactly, for the worst trajectory at each bottom
    out = lvs.rolling_paths(starts, until, columns=("headroom_A", "dry_powder_B", "nav_A", "nav_B", "ltv_A", "E_A", "loan", "E_B", "call_val", "cash_B", "loan_B", "n_calls", "n_bought", "call_notional", "interest_cum_A", "interest_cum_B", "premiums_cum_B", "payoffs_cum_B", "div_cum_A", "div_cum_B", "exposure_A", "exposure_B"), sample="M", mark_days=bottoms, progress=_p,
                            **_engine_kwargs(s))
    bar.empty()
    return out


@st.cache_data
def corrections_cached(threshold: float, wht: float, on: str = "price") -> pd.DataFrame:
    """The corrections of at least ``threshold`` on the record: on the SPX price index (default) or on SPX with dividends net of ``wht``."""
    return lvs.corrections(threshold, wht=wht, on=on)


def start_grid(s: Setup, grid: str) -> tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """(freq, first day, last start, last strike, last day) for the all-start-dates pages: the last start is the one whose rotation was
    complete a month before the last data day; ``last strike`` is the last day on which a tranche still expires inside the data (starts
    whose last tranche was struck after it have calls not yet expired). ``grid`` is the sidebar's start-date choice."""
    first_day, last_day = data_bounds()
    freq = "D" if grid.startswith("every trading") else "W-FRI"
    last_strike = last_day - pd.DateOffset(months=round(s.tenor * 12))
    last_start = last_day - pd.DateOffset(days=7 * (s.build - 1) + 30)
    return freq, first_day, last_start, last_strike, last_day


def table_height(n_rows: int) -> int:
    """Pixel height that shows every row of st.dataframe without an inner scrollbar (35 px per row + header + border)."""
    return 35 * (n_rows + 1) + 3
