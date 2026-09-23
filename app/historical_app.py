"""Keep the loan or rotate into calls — the decision app (two pages).

Page 1 (views/all_starts.py): both portfolios from every start date since 1997 held to today — the fan of trajectories with
the final-NAV distribution, its statistics, the corrections and the worst trajectories, the final value by start date.
Engine: fosim.analytics.leverage_stress.
Page 2 (views/premium_history.py): the call-vs-cash backtest behind the premiums (J.P. Morgan chart construction).
Engine: fosim.analytics.call_vs_cash.

Run:  .venv/bin/streamlit run app/historical_app.py --server.runOnSave true
"""

from __future__ import annotations

import streamlit as st

from common import premium_sidebar, sidebar_setup

st.set_page_config(page_title="Keep the loan or rotate into calls", layout="wide")
pg = st.navigation([
    st.Page("views/all_starts.py", title="Historical simulation", default=True),
    st.Page("views/premium_history.py", title="Call premium history"),
])
if pg.title == "Call premium history":   # each page has its own sidebar; the call tenor and the withholding tax are common to both
    premium_sidebar()
else:
    sidebar_setup()
pg.run()
