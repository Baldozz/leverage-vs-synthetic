"""Keep the loan or rotate into calls — the decision app (three pages).

Page 1 (views/all_starts.py): both portfolios from every start date since 1997 held to today — the fan of trajectories with
the final-NAV distribution, its statistics, the corrections and the worst trajectories, the final value by start date.
Engine: fosim.analytics.leverage_stress.
Page 2 (views/premium_history.py): the call-vs-cash backtest behind the premiums (J.P. Morgan chart construction).
Engine: fosim.analytics.call_vs_cash.
Page 3 (views/five_scenarios.py), a SEPARATE, simplified exercise independent of pages 1 and 2 (own menu group): the investor's five ways to
hold a capital (long, levered long, 75/25 with calls, long plus calls on a loan, all in calls) from three start dates, margin calls flagged.
Engine: fosim.analytics.five_scenarios.

Run:  .venv/bin/streamlit run app/historical_app.py --server.runOnSave true
"""

from __future__ import annotations

import streamlit as st

from common import premium_sidebar, scenarios_sidebar, sidebar_setup

st.set_page_config(page_title="Keep the loan or rotate into calls", layout="wide")
pg = st.navigation({   # two separate parts: the decision tool (pages 1 and 2) and the five-scenarios exercise (page 3), grouped apart in the menu
    "Keep the loan or rotate into calls": [
        st.Page("views/all_starts.py", title="Historical simulation", default=True),
        st.Page("views/premium_history.py", title="Call premium history"),
    ],
    "Separate exercise": [st.Page("views/five_scenarios.py", title="Five scenarios")],
})
if pg.title == "Call premium history":   # each page has its own sidebar; the call tenor and the withholding tax are common to all
    premium_sidebar()
elif pg.title == "Five scenarios":
    scenarios_sidebar()
else:
    sidebar_setup()
pg.run()
