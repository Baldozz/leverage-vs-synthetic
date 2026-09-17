"""Any start date since 1997 — the same two portfolios put on at every start of the grid and held to today."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    BLUE,
    GREY,
    LAYOUT,
    RED,
    A,
    B,
    M,
    all_starts_cached,
    data_bounds,
    setup,
    table_height,
)

s = setup()
first_day, last_day = data_bounds()
freq = "D" if str(st.session_state.get("s_grid", "every trading day")).startswith("every trading") else "W-FRI"
last_strike = last_day - pd.DateOffset(months=round(s.tenor * 12))            # a call struck later would not have expired yet
last_start = last_strike - pd.Timedelta(7 * (s.build - 1), unit="D")         # the last weekly tranche is struck on last_strike

st.title("Any start date since 1997")
st.caption(f"The same two portfolios put on at every {'trading day' if freq == 'D' else 'week'} from {first_day:%d %b %Y} to {last_start:%d %b %Y} and held to {last_day:%d %b %Y}. "
           f"The last start is the one whose last weekly tranche, struck on {last_strike:%d %b %Y}, still expires inside the data. Same rotation as page 1: "
           f"{s.tenor:g}-year ATM calls on SPXFP, {s.build} weekly step{'s' if s.build > 1 else ''}, the loan gone after the build; a call that expires in the money is replaced, "
           + ("a call that expires worthless lapses." if not s.replace_worthless else "a worthless call is replaced too."))

rs = all_starts_cached(str(first_day.date()), str(last_start.date()), freq, s, str(last_day.date()))
if rs.empty:
    st.warning("No start date fits: shorten the rotation or the tenor.")
    st.stop()
years = rs["years"]
ann = pd.DataFrame({A: (1.0 + rs["A return"]) ** (1.0 / years) - 1.0, B: (1.0 + rs["B return"]) ** (1.0 / years) - 1.0}, index=rs.index)
lost_all = rs["B: call notional end"] <= 0.0                       # no call left today: the whole option part lapsed
some_lapsed = rs["B: lapsed"] > 0
st.markdown(f"**{len(rs):,} start dates, {rs.index[0]:%d %b %Y} → {rs.index[-1]:%d %b %Y}, each held to {last_day:%d %b %Y} ({years.min():.0f} to {years.max():.0f} years).**")

# ---------------- annualised return to today, by start date
st.subheader("Annualised return to today, by start date")
fig = go.Figure()
runs = (lost_all != lost_all.shift()).cumsum()
for _, g in rs[lost_all].groupby(runs[lost_all]):
    fig.add_vrect(x0=str(g.index[0].date()), x1=str(g.index[-1].date()), fillcolor=GREY, opacity=0.25, line_width=0)
fig.add_trace(go.Scatter(x=ann.index, y=ann[A], name=A, line={"color": RED, "width": 1.4}))
fig.add_trace(go.Scatter(x=ann.index, y=ann[B], name=B, line={"color": BLUE, "width": 1.4}))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker={"color": GREY, "opacity": 0.4, "size": 12, "symbol": "square"}, name="start dates whose calls all expired worthless"))
fig.add_hline(y=0, line={"color": GREY, "width": 1})
fig.update_layout(height=420, xaxis_title="Start date", yaxis={"title": "% change of the NAV to today, annualised", "tickformat": ".0%"}, **LAYOUT)
st.plotly_chart(fig, width="stretch")
st.caption(f"Each start begins with the same NAV ({(s.equity - s.loan) / M:,.0f} m); the % change to {last_day:%d %b %Y} is annualised over that start's own holding period, so starts of different length are comparable.")

# ---------------- distribution over all start dates
st.subheader("Distribution of the annualised return over all start dates")
step = 0.005
lo, hi = float(ann.min().min()), float(ann.max().max())
edges = np.arange(np.floor(lo / step) * step, np.ceil(hi / step) * step + step, step)
fig = go.Figure()
for col, colr in ((A, RED), (B, BLUE)):
    cnt, _ = np.histogram(ann[col], bins=edges)
    fig.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=cnt / cnt.sum(), name=col, marker={"color": colr}, opacity=0.55, width=step))
fig.update_layout(barmode="overlay", height=360, xaxis={"title": "annualised return to today", "tickformat": ".0%"}, yaxis={"title": "share of start dates", "tickformat": ".0%"}, **LAYOUT)
st.plotly_chart(fig, width="stretch")
qs = [0.05, 0.25, 0.50, 0.75, 0.95]
dist = pd.DataFrame({c: {**{f"{int(q * 100)}th percentile": float(ann[c].quantile(q)) for q in qs}, "mean": float(ann[c].mean()), "worst start": float(ann[c].min()), "best start": float(ann[c].max())} for c in ann.columns})
dist[f"{B} − {A} (points)"] = dist[B] - dist[A]
st.dataframe(dist.style.format("{:+.1%}", subset=[A, B]).format(lambda v: f"{v * 100:+.1f}", subset=[f"{B} − {A} (points)"]), width="stretch", height=table_height(len(dist)))
diff = ann[B] - ann[A]
st.markdown(f"- Same start, same end: the rotation ends **ahead on {float((diff > 0).mean()):.0%}** of start dates; annualised difference {B} − {A}: median **{diff.median() * 100:+.1f} points**, "
            f"5th–95th percentile {diff.quantile(0.05) * 100:+.1f} to {diff.quantile(0.95) * 100:+.1f}, worst {diff.min() * 100:+.1f} ({diff.idxmin():%d %b %Y} start), best {diff.max() * 100:+.1f} ({diff.idxmax():%d %b %Y} start).")

# ---------------- reading
st.markdown(f"- The rotated portfolio lost **all** its calls on **{lost_all.mean():.0%}** of start dates ({int(lost_all.sum()):,}) and **some** of them on {some_lapsed.mean():.0%}; "
            f"on the other {1 - some_lapsed.mean():.0%} every call expired in the money and was replaced.\n"
            f"- Annualised return to today, median over start dates: **{ann[A].median():+.1%}** ({A}) vs **{ann[B].median():+.1%}** ({B}); worst start {ann[A].min():+.1%} vs {ann[B].min():+.1%}.\n"
            f"- Margin calls ({A}): **{int(rs['A: margin call'].sum()):,}** start dates; least room before a margin call {rs['A: min headroom'].min() / M:,.0f} m. "
            f"Least capacity to borrow ({B}): {rs['B: min dry powder'].min() / M:,.0f} m.")
st.caption("Consecutive start dates overlap almost entirely, so the chart shows the range of historical outcomes, not independent draws. Same pricing and data caveats as page 1.")
