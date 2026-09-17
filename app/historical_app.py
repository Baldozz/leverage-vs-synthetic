"""Keep the loan or rotate into calls — the decision app (two pages).

Page 1 (views/rotation.py): both portfolios from one start date, held to today — the weekly rotation, how they evolve,
the room before a margin call vs. the capacity to borrow, the rolls.  Engine: fosim.analytics.leverage_stress.
Page 2 (views/premium_history.py): the call-vs-cash backtest behind the premiums (J.P. Morgan chart construction).
Engine: fosim.analytics.call_vs_cash.

Run:  .venv/bin/streamlit run app/historical_app.py --server.runOnSave true
"""

from __future__ import annotations

import streamlit as st

from common import sidebar_setup

st.set_page_config(page_title="Keep the loan or rotate into calls", layout="wide")
sidebar_setup()
st.navigation([
    st.Page("views/rotation.py", title="Keep the loan or rotate", default=True),
    st.Page("views/premium_history.py", title="Call premium history"),
]).run()
