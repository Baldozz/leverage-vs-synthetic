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

import fosim.analytics.call_vs_cash as cvc  # noqa: E402
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
    wht: float             # dividend withholding tax, fraction
    tenor: float           # years
    build: int             # weekly steps of the rotation (1 = one shot)
    surplus: str           # "cash" | "equity" | "calls"
    replace_worthless: bool = True    # a call that expires worthless: replaced (True) or lapses (False)
    roll: str = "target"              # what an expiring call is replaced on: "target" (the gap to Keep's exposure), "delta" (the same dollar delta), "units" (the same index units)
    rebalance: str = "quarterly"      # target rule: when the exposure is checked against the band ("quarterly", "monthly", "none")
    band: float = 0.10                # target rule: the band around Keep's exposure, fraction
    haircut: float = 0.01             # target rule: vol points (fraction) taken off the mark on calls sold early
    below: str = "calls"              # target rule: what is bought from the T-bills below the band ("calls", "spx")


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
    for k, v in (("s_equity", 1000.0), ("s_loan", 250.0), ("s_spread", 75.0), ("s_lv", 75.0), ("s_weeks", 52), ("s_worthless", "replaced, SPX sold to pay it"), ("s_roll", "Keep's exposure"),
                 ("s_rebalance", "every quarter"), ("s_band", 10.0), ("s_haircut", 1.0), ("s_below", "ATM calls from the T-bills"),
                 ("s_surplus", "kept in T-bills"), ("s_lv_calls", 0.0), ("s_lv_cash", 90.0), ("s_grid", "every trading day")):
        _restore(k, v)
    with st.sidebar:
        st.markdown("**Today**")
        st.number_input("SPX exposure (USD m)", 10.0, 100000.0, step=50.0, key="s_equity")
        st.number_input("Lombard loan (USD m)", 0.0, 100000.0, step=10.0, key="s_loan", help="Borrowed against the SPX; interest capitalised.")
        st.number_input("Spread over SOFR (bp)", 0.0, 500.0, step=5.0, key="s_spread", help="Base rate: 3-month LIBOR to 2018, Term SOFR after.")
        st.number_input("Lending value of the SPX (%)", 1.0, 100.0, step=5.0, key="s_lv", help="The most the bank lends against the SPX. LTV = loan ÷ lending value; margin call above 100 %.")
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
    _remember("s_equity", "s_loan", "s_spread", "s_lv", "s_tenor", "s_weeks", "s_wht", "s_roll", "s_rebalance", "s_band", "s_haircut", "s_below", "s_worthless", "s_surplus", "s_lv_calls", "s_lv_cash", "s_grid")
    return setup()


def setup() -> Setup:
    """The Setup the sidebar widgets currently show (pages run after the entry script has drawn them)."""
    s = st.session_state
    return Setup(float(s["s_equity"]) * M, float(s["s_loan"]) * M, float(s["s_spread"]) / 1e4, float(s["s_lv"]) / 100.0, float(s["s_lv_calls"]) / 100.0, float(s["s_lv_cash"]) / 100.0,
                 float(s["s_wht"]) / 100.0, float(s["s_tenor"]), int(s["s_weeks"]), _SURPLUS[str(s["s_surplus"])], str(s["s_worthless"]).startswith("replaced"), _ROLL[str(s["s_roll"])],
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
    return {"equity0": s.equity, "loan0": s.loan, "spread": s.spread, "ltv_equity": s.lv_equity, "ltv_call": s.lv_calls, "ltv_cash": s.lv_cash, "tenor": s.tenor, "wht": s.wht,
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
