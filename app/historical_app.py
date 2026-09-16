"""Call-vs-cash backtest — the primary analysis.

For every trading day since September 1997 an at-the-money call on SPXFP (S&P 500 futures excess-return
index) maturing `tenor` years later is bought for a fixed USD amount, or the same amount is invested in the
index; both legs are read at the option's maturity. Same construction as the J.P. Morgan / Bloomberg chart
("Backtested PnL of SPXFP 5yr ATM Call Option vs. Cash Investment"), with our own Bloomberg series.
Engine: fosim.analytics.call_vs_cash. The strategy simulator (A/B/C replay) lives in app/strategy_replay_app.py.

Run:  .venv/bin/streamlit run app/historical_app.py --server.runOnSave true
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fosim.analytics.call_vs_cash as cvc  # noqa: E402

M = 1e6

st.set_page_config(page_title="SPXFP long-dated call vs. cash investment — backtest", layout="wide")


@st.cache_data
def call_vs_cash_backtest(tenor: float, premium_usd: float, start: str, end: str, prem_fixed: float | None, cash_leg: str, wht: float) -> tuple[pd.DataFrame, cvc.BacktestInfo]:
    return cvc.backtest(tenor, premium_usd, start, end, prem_fixed, cash_leg, wht)


st.title("Long-dated ATM call on SPXFP vs. cash investment — historical backtest")
st.subheader("Backtested P&L of a long-dated ATM call on SPXFP vs. a cash investment in the index")
st.caption("Every trading day is a hypothetical strike date: buy an ATM call on SPXFP for the premium below (notional = premium ÷ premium-% of the day) "
           "or invest the same amount in the index; both are read at the option's maturity. Plotted against the maturity date. "
           "Same construction as the J.P. Morgan / Bloomberg chart (their strikes: May-2006 → Aug-2021) with our own data: the premium is the one implied by the vol series and the 5-year Treasury of the day, not a dealer quote.")
c1, c2, c3, c4 = st.columns(4)
with c1:
    bt_prem_usd = st.number_input("Premium / investment (USD m)", 1.0, 1000.0, 10.0, 1.0, key="bt_prem_usd") * M
    bt_tenor = st.number_input("Call tenor (years)", 1.0, 10.0, 5.0, 1.0, key="bt_tenor")
with c2:
    bt_start = st.date_input("First strike date", pd.Timestamp("1997-09-09"), min_value=pd.Timestamp("1997-09-09"), max_value=pd.Timestamp("2025-12-31"), key="bt_start")
    bt_end = st.date_input("Last strike date", pd.Timestamp("2021-08-31"), min_value=pd.Timestamp("1997-09-09"), max_value=pd.Timestamp("2025-12-31"), key="bt_end")
with c3:
    bt_mode = st.radio("Premium paid", ["Market: vol and 5y rate of the day", "Fixed % of notional"], key="bt_mode")
    bt_fixed = st.number_input("Fixed premium (% of notional)", 1.0, 60.0, 15.0, 0.5, key="bt_fixed", disabled=bt_mode.startswith("Market")) / 100.0
with c4:
    bt_leg = st.radio("Cash investment in", ["SPXFP", "SPX with dividends reinvested"], key="bt_leg", help="The J.P. Morgan chart invests the cash leg in SPXFP itself.")
    bt_wht = st.number_input("Dividend withholding tax (%)", 0.0, 50.0, 15.0, 1.0, key="bt_wht") / 100.0
bt, bt_info = call_vs_cash_backtest(float(bt_tenor), float(bt_prem_usd), str(bt_start), str(bt_end), None if bt_mode.startswith("Market") else float(bt_fixed), "SPXFP" if bt_leg == "SPXFP" else "SPX_TR", float(bt_wht))
if bt_mode.startswith("Market") and bt_info.fallback:
    st.warning(f"No {bt_tenor:g}-year Treasury / implied-vol series in the data: the premium is computed with the 5-year columns ({bt_info.rate_col}, {bt_info.vol_col}). "
               f"Add columns `ust_{round(bt_tenor)}y` and `iv_{round(bt_tenor)}y` to data/market_data_daily.csv for a tenor-consistent premium.")
if bt.empty:
    st.warning("No strike date in that range has reached maturity within the data (ends 2026-09-14).")
else:
    st.markdown(f"**{bt_tenor:g}-year ATM call on SPXFP vs. {bt_prem_usd / M:,.0f} m in {'SPXFP' if bt_leg == 'SPXFP' else 'SPX (dividends reinvested)'} — "
                f"{len(bt):,} strike dates {bt.strike_date.min().date()} → {bt.strike_date.max().date()}, premium {'of the day' if bt_mode.startswith('Market') else f'fixed at {bt_fixed:.1%}'}**")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt.call_pnl / M, name=f"{bt_tenor:g}y SPXFP ATM call payoff − {bt_prem_usd / M:,.0f} m premium", line={"color": "#0b2a6f", "width": 1.6}))
    fig.add_trace(go.Scatter(x=bt.index, y=bt.cash_pnl / M, name=f"{bt_tenor:g}y cash investment − {bt_prem_usd / M:,.0f} m principal", line={"color": "#c00000", "width": 1.6}))
    fig.add_hline(y=0, line={"color": "#999", "width": 1})
    fig.update_layout(template="plotly_white", height=460, legend={"orientation": "h", "y": 1.08}, xaxis_title="Maturity date", yaxis_title="P&L (USD m)", margin={"l": 40, "r": 20, "t": 30, "b": 40})
    st.plotly_chart(fig, width="stretch")
    beats = float((bt.call_pnl > bt.cash_pnl).mean())
    lost_all = float((bt.call_pnl <= -bt_prem_usd + 1e-6).mean())
    summ = pd.DataFrame({
        "call": {"average P&L (m)": bt.call_pnl.mean() / M, "median P&L (m)": bt.call_pnl.median() / M, "best (m)": bt.call_pnl.max() / M, "worst (m)": bt.call_pnl.min() / M, "share of strike dates with a loss": float((bt.call_pnl < 0).mean())},
        "cash investment": {"average P&L (m)": bt.cash_pnl.mean() / M, "median P&L (m)": bt.cash_pnl.median() / M, "best (m)": bt.cash_pnl.max() / M, "worst (m)": bt.cash_pnl.min() / M, "share of strike dates with a loss": float((bt.cash_pnl < 0).mean())},
    })
    st.dataframe(summ.style.format("{:,.1f}", subset=pd.IndexSlice[[i for i in summ.index if i.endswith("(m)")], :]).format("{:.0%}", subset=pd.IndexSlice[["share of strike dates with a loss"], :]), width="stretch")
    st.markdown(f"- {len(bt):,} strike dates from {bt.strike_date.min().date()} to {bt.strike_date.max().date()} (maturities {bt.index.min().date()} → {bt.index.max().date()}).\n"
                f"- The call beats the cash investment on **{beats:.0%}** of strike dates and expires worthless (whole premium lost) on **{lost_all:.0%}**.\n"
                f"- Premium paid: average **{bt.prem.mean():.1%}** of notional (min {bt.prem.min():.1%}, max {bt.prem.max():.1%}) → notional bought per {bt_prem_usd / M:,.0f} m of premium averages **{bt.notional.mean() / M:,.0f} m** ({bt.notional.min() / M:,.0f}–{bt.notional.max() / M:,.0f} m), i.e. the call's leverage on the same USD.")
    # ---------------- return distribution (total return on the USD committed, over the tenor)
    st.subheader(f"Distribution of the {bt_tenor:g}-year return on the {bt_prem_usd / M:,.0f} m committed")
    ret = pd.DataFrame({"call": bt.call_pnl / bt_prem_usd, "cash investment": bt.cash_pnl / bt_prem_usd}, index=bt.index)
    lo, hi = float(ret.min().min()), float(ret.max().max())
    step = 0.25 if hi - lo > 4 else 0.10
    edges = np.arange(np.floor(lo / step) * step, np.ceil(hi / step) * step + step, step)
    fig = go.Figure()
    for col, colr in (("call", "#0b2a6f"), ("cash investment", "#c00000")):
        cnt, _ = np.histogram(ret[col], bins=edges)
        fig.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=cnt / cnt.sum(), name=col, marker={"color": colr}, opacity=0.55, width=step))
    fig.update_layout(barmode="overlay", template="plotly_white", height=380, legend={"orientation": "h"}, xaxis={"title": f"total return over {bt_tenor:g} years (× the amount committed)", "tickformat": ".0%"}, yaxis={"title": "share of strike dates", "tickformat": ".0%"}, margin={"l": 40, "r": 20, "t": 20, "b": 40})
    st.plotly_chart(fig, width="stretch")
    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    dist = pd.DataFrame({c: {**{f"{int(q * 100)}th percentile": float(ret[c].quantile(q)) for q in qs}, "mean": float(ret[c].mean()), "std": float(ret[c].std()),
                             "probability of a loss": float((ret[c] < 0).mean()), "probability of losing everything": float((ret[c] <= -1 + 1e-9).mean()),
                             "mean annualised (CAGR)": float(np.sign(1 + ret[c].mean()) * abs(1 + ret[c].mean()) ** (1 / bt_tenor) - 1)} for c in ret.columns})
    st.dataframe(dist.style.format("{:+.0%}", subset=pd.IndexSlice[[i for i in dist.index if "probability" not in i and i != "std"], :]).format("{:.0%}", subset=pd.IndexSlice[[i for i in dist.index if "probability" in i or i == "std"], :]), width="stretch")
    st.caption("Returns are on the same USD committed (premium for the call, principal for the cash investment), held to the option's maturity; the call's return is floored at −100 %. "
               "Strike dates are daily, so consecutive observations overlap almost entirely: the distribution describes the range of outcomes, not independent draws.")
    st.caption("For illustrative purposes only: hypothetical strike dates, cash-settled at maturity, no bid/ask, no early unwind. The 5-year implied vol before May-2005 is a VIX proxy and the 5-year vol is extrapolated from the 24-month series throughout (Assumptions 19l).")
