"""Shared by the pages: the sidebar setup, cached engine calls, labels and colours."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

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
    wht: float             # dividend withholding tax, fraction
    tenor: float           # years
    build: int             # weekly steps of the rotation (1 = one shot)
    surplus: str           # "cash" | "equity" | "calls"


_SURPLUS = {"kept in T-bills": "cash", "reinvested in SPX": "equity", "reinvested in more calls": "calls"}


def sidebar_setup() -> Setup:
    """The one set of assumptions every page uses. Drawn in the sidebar by the entry script; read back by the pages."""
    with st.sidebar:
        st.markdown("**Today**")
        st.number_input("SPX exposure (USD m)", 10.0, 100000.0, 1000.0, 50.0, key="s_equity")
        st.number_input("Lombard loan (USD m)", 0.0, 100000.0, 250.0, 10.0, key="s_loan", help="Borrowed against the SPX; interest capitalised.")
        st.number_input("Spread over SOFR (bp)", 0.0, 500.0, 75.0, 5.0, key="s_spread", help="Base rate: 3-month LIBOR to 2018, Term SOFR after.")
        st.number_input("Lending value of the SPX (%)", 1.0, 100.0, 75.0, 5.0, key="s_lv", help="The most the bank lends against the SPX. LTV = loan ÷ lending value; margin call above 100 %.")
        st.markdown("**The rotation**")
        st.selectbox("Call tenor (years)", [5, 7, 10], key="s_tenor")
        st.number_input("Weeks to complete the rotation", 1, 156, 52, 1, key="s_weeks", help="Each week: sell SPX, buy one tranche of ATM calls sized by delta, repay an equal share of the loan. 1 = all on the start date.")
        with st.expander("Advanced"):
            st.number_input("Dividend withholding tax (%)", 0.0, 50.0, 15.0, 1.0, key="s_wht")
            st.radio("Payoff left after a roll", list(_SURPLUS), key="s_surplus", help="Each expiring tranche is rolled the same day into a new ATM call on the same index units; this is what happens to any payoff left over.")
            st.number_input("Lending value of the calls (%)", 0.0, 100.0, 0.0, 5.0, key="s_lv_calls")
        st.caption("Historical data only, September 1997 to today.")
    return setup()


def setup() -> Setup:
    """The Setup from the sidebar widgets' session state (pages run after the entry script has drawn them)."""
    s = st.session_state
    return Setup(float(s["s_equity"]) * M, float(s["s_loan"]) * M, float(s["s_spread"]) / 1e4, float(s["s_lv"]) / 100.0, float(s["s_lv_calls"]) / 100.0, float(s["s_wht"]) / 100.0,
                 float(s["s_tenor"]), int(s["s_weeks"]), _SURPLUS[str(s["s_surplus"])])


def data_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    d = lvs._load(str(lvs.DAILY_FILE))
    return pd.Timestamp(d.date.iloc[0]), pd.Timestamp(d.date.iloc[-1])


@st.cache_data(show_spinner="Simulating …")
def simulate_cached(start: str, end: str, s: Setup) -> tuple[pd.DataFrame, lvs.StressInfo]:
    """Both portfolios from ``start`` to ``end`` on the sidebar setup."""
    return lvs.simulate(start, end, equity0=s.equity, loan0=s.loan, spread=s.spread, ltv_equity=s.lv_equity, ltv_call=s.lv_calls, tenor=s.tenor, wht=s.wht,
                        surplus=s.surplus, cash_buffer=0.0, delta=None, build_tranches=s.build)


def table_height(n_rows: int) -> int:
    """Pixel height that shows every row of st.dataframe without an inner scrollbar (35 px per row + header + border)."""
    return 35 * (n_rows + 1) + 3
