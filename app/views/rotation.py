"""Keep the loan or rotate into calls — both portfolios from one start date, held to today."""

from __future__ import annotations

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
    data_bounds,
    setup,
    simulate_cached,
    table_height,
)

s = setup()
first_day, last_day = data_bounds()

st.title("Keep the loan, or rotate into calls")
st.caption(f"**{A}**: {s.equity / M:,.0f} m in SPX with a {s.loan / M:,.0f} m Lombard loan at SOFR + {s.spread * 1e4:.0f} bp, interest capitalised; lending value {s.lv_equity:.0%}. "
           f"**{B}**: every week sell SPX, buy {s.tenor:g}-year ATM calls on SPXFP sized by delta so the SPX-equivalent exposure stays {s.equity / M:,.0f} m, and repay an equal share of the loan; "
           f"after {s.build} week{'s' if s.build > 1 else ''} the loan is gone. Each call is rolled at expiry into a new ATM call on the same index units.")

latest_start = last_day - pd.Timedelta(7 * (s.build - 1) + 30, unit="D")
start = st.date_input("Start date", first_day.date(), min_value=first_day.date(), max_value=latest_start.date(), key="r_start", help="Both portfolios are put on this day and held to the last day of the data.")
p, info = simulate_cached(str(start), str(last_day.date()), s)
if info.fallback:
    st.warning(f"No {s.tenor:g}-year Treasury / implied-vol series in the data: the calls are priced with the 5-year columns.")

# ---------------- the rotation
st.subheader("The rotation")
build_premium = sum(b.premium_paid for b in info.builds)
build_repaid = sum(b.equity_sold - b.premium_paid for b in info.builds)   # each step: SPX sold − premium = loan repaid
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("SPX sold", f"{info.rotation / M:,.0f} m", f"{info.rotation / s.equity:.0%} of the exposure", delta_color="off")
c2.metric("Call notional bought", f"{info.notional0 / M:,.0f} m", f"delta {info.delta0:.0%}", delta_color="off")
c3.metric("Premium paid", f"{build_premium / M:,.0f} m", f"{info.premium0:.1%} of notional", delta_color="off")
c4.metric("Loan repaid", f"{build_repaid / M:,.0f} m", f"{info.loan_base0 + s.spread:.2%} at the start", delta_color="off")
c5.metric("Rotation complete", f"{info.build_end:%d %b %Y}", f"{len(info.builds)} weekly step{'s' if len(info.builds) > 1 else ''}", delta_color="off")

# ---------------- how the two portfolios evolve
st.subheader("How the two portfolios evolve")
nav0 = float(p["nav_A"].iloc[0])
fig = go.Figure()
fig.add_trace(go.Scatter(x=p.index, y=p["nav_A"] / M, name=A, line={"color": RED, "width": 1.6}))
fig.add_trace(go.Scatter(x=p.index, y=p["nav_B"] / M, name=B, line={"color": BLUE, "width": 1.6}))
fig.add_trace(go.Scatter(x=p.index, y=nav0 * p["spx_tr"] / M, name="SPX with dividends, same starting NAV", line={"color": GREY, "width": 1, "dash": "dot"}))
fig.update_layout(height=400, yaxis_title="Net asset value (USD m)", **LAYOUT)
st.plotly_chart(fig, width="stretch")

fig = go.Figure()
for col, name, colr in (("E_B", f"{B}: SPX", "#7f9ccf"), ("call_val", f"{B}: calls", BLUE), ("cash_B", f"{B}: cash", "#bcd"), ("loan_B", f"{B}: loan", "#e0b0b0")):
    fig.add_trace(go.Scatter(x=p.index, y=(-p[col] if col == "loan_B" else p[col]) / M, name=name, stackgroup="B" if col != "loan_B" else "L", line={"width": 0.5, "color": colr}))
fig.add_trace(go.Scatter(x=p.index, y=p["E_A"] / M, name=f"{A}: SPX", line={"color": RED, "width": 1.6}))
fig.add_trace(go.Scatter(x=p.index, y=-p["loan"] / M, name=f"{A}: loan", line={"color": RED, "width": 1.6, "dash": "dash"}))
fig.update_layout(height=400, yaxis_title="What each portfolio holds (USD m)", **LAYOUT)
st.plotly_chart(fig, width="stretch")

fig = go.Figure()
fig.add_trace(go.Scatter(x=p.index, y=p["headroom_A"] / M, name=f"{A}: room before a margin call (lending value − loan)", line={"color": RED, "width": 1.6}))
fig.add_trace(go.Scatter(x=p.index, y=p["dry_powder_B"] / M, name=f"{B}: capacity to borrow (lending value − loan + cash)", line={"color": BLUE, "width": 1.6}))
fig.add_hline(y=0, line={"color": GREY, "width": 1})
fig.update_layout(height=400, yaxis_title="USD m", **LAYOUT)
st.plotly_chart(fig, width="stretch")

worst_a = p["headroom_A"].idxmin()
worst_b = p["dry_powder_B"].idxmin()
st.markdown(f"- **{A}**: LTV peaks at **{p['ltv_A'].max():.0%}** on {p['ltv_A'].idxmax():%d %b %Y}; the least room before a margin call is **{p.loc[worst_a, 'headroom_A'] / M:,.0f} m** on {worst_a:%d %b %Y}"
            + (" — **a margin call**." if p["headroom_A"].min() < 0 else ".") + "\n"
            f"- **{B}**: the least capacity to borrow is **{p.loc[worst_b, 'dry_powder_B'] / M:,.0f} m** on {worst_b:%d %b %Y}; the loan is gone from {info.build_end:%d %b %Y}.\n"
            f"- Today: NAV **{p['nav_A'].iloc[-1] / M:,.0f} m** ({A}) vs **{p['nav_B'].iloc[-1] / M:,.0f} m** ({B}); interest paid {p['interest_A'].sum() / M:,.0f} m ({A}) vs premiums paid "
            f"{(build_premium + sum(r.premium_paid for r in info.rolls)) / M:,.0f} m against {sum(r.payoff for r in info.rolls) / M:,.0f} m of payoffs received ({B}).")

# ---------------- rolls
st.subheader("Rolls")
if info.rolls:
    rl = pd.DataFrame({"Date": [r.date for r in info.rolls], "Payoff received (m)": [r.payoff / M for r in info.rolls], "New premium paid (m)": [r.premium_paid / M for r in info.rolls],
                       "Premium (% of notional)": [r.premium_frac for r in info.rolls], "SPX sold to pay it (m)": [r.equity_sold / M for r in info.rolls]})
    by_year = rl.groupby(rl["Date"].dt.year).agg(**{"Tranches rolled": ("Date", "size"), "Payoff received (m)": ("Payoff received (m)", "sum"), "New premium paid (m)": ("New premium paid (m)", "sum"),
                                                     "Premium (% of notional)": ("Premium (% of notional)", "mean"), "SPX sold to pay it (m)": ("SPX sold to pay it (m)", "sum")})
    by_year.index.name = "Year"
    st.dataframe(by_year.style.format("{:,.1f}").format("{:.1%}", subset=["Premium (% of notional)"]).format("{:d}", subset=["Tranches rolled"]), width="stretch", height=table_height(len(by_year)))
    with st.expander("Every roll"):
        rl["Date"] = rl["Date"].dt.date
        st.dataframe(rl.style.format("{:,.1f}").format("{:.1%}", subset=["Premium (% of notional)"]), width="stretch", hide_index=True, height=min(table_height(len(rl)), 600))
else:
    st.markdown("No call has expired yet from this start date.")
st.caption("Calls are priced and marked with Black–Scholes on SPXFP (q = r) at the implied vol of the day for the tenor, extrapolated from the 24-month Bloomberg series; "
           "the delta is the model delta of the ATM call. No bid/ask, no early unwind. Dividends reinvested net of withholding; cash earns the 3-month T-bill.")
