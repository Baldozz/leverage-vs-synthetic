"""Five scenarios — the investor's five ways to hold the capital, from a few start dates, held to today (page 3 of app/historical_app.py).

1 long the market; 2 levered long (a loan on top, repaid after one tenor); 3 75 % SPX + 25 % rolling 5-year calls (tax at expiry, back to
75/25); 4 long + calls bought on a loan (tax, the loan repaid from the proceeds, the rest in new calls); 5 all in calls. Margin calls flagged on
2 and 4. Engine: fosim.analytics.five_scenarios. Inputs: the page's sidebar (common.scenarios_sidebar); the tenor and the withholding tax are shared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import fosim.analytics.five_scenarios as fsc
import fosim.analytics.leverage_stress as lvs
from common import (
    GREY,
    LAYOUT,
    M,
    corrections_cached,
    data_bounds,
    scenario_setup,
    scenarios_cached,
)

COLOURS = {"1": "#2a78d6", "2": "#eb6834", "3": "#1baf7a", "4": "#eda100", "5": "#e87ba4"}   # the dataviz reference palette, slots 1–5, validated on the light surface
CALL_RED = "#c00000"          # status colour, reserved for the margin call (icon + label, never colour alone)
INK = "#333333"
ALIGNED = {"l": 80, "r": 120, "t": 30, "b": 40}
NAMES = {sid: f"{sid} {lab}" for sid, lab in fsc.LABELS.items()}


@st.cache_data
def named_starts() -> dict[str, pd.Timestamp]:
    return fsc.default_starts()


s = scenario_setup()
first_day, last_day = data_bounds()
corr = corrections_cached(0.20, 0.15, "price")
spx_px = lvs._load(str(lvs.DAILY_FILE)).set_index("date")["spx_px_last"]
starts = named_starts()
labels = {f"{lab} — {d:%d %b %Y}, SPX {float(spx_px.loc[d]):,.0f}": (lab, d) for lab, d in starts.items()}

st.title("Five scenarios")
st.caption("A separate, simplified exercise, independent of the two *Keep the loan or rotate into calls* pages: a different portfolio, the investor's own rules as written, "
           "no delta-sized rotation, no roll matching, no band. Only the market data and the pricer are shared.")
c_starts, c_extra, c_log = st.columns([3, 1.6, 0.9])
picked = c_starts.multiselect("Start dates", list(labels), default=list(labels), key="f_starts", help="Each start is held to the last data day. The three named starts are the investor's: when the data starts, the top of the dot-com bubble, the bottom of the GFC.")
extra = c_extra.checkbox("Add another start date", key="f_extra")
extra_date = c_extra.date_input("Start", first_day.date(), min_value=first_day.date(), max_value=(last_day - pd.DateOffset(days=30)).date(), key="f_extra_date", disabled=not extra, label_visibility="collapsed")
log_scale = c_log.checkbox("Log scale", key="f_log", help="The NAV axis in log scale: the all-in-calls portfolio can end a hundred times the others.")
runs: list[tuple[str, pd.Timestamp]] = [labels[k] for k in labels if k in picked]
if extra:
    runs.append(("chosen start", pd.Timestamp(extra_date)))
if not runs:
    st.info("Pick at least one start date.")
    st.stop()

results = {lab: scenarios_cached(str(d.date()), str(last_day.date()), s) for lab, d in runs}
summaries = {lab: fsc.summary(*results[lab]) for lab, _ in runs}
info0 = results[runs[0][0]][1]
c = info0["3"].prem_frac0 if s.call_frac > 0 else (s.premium or float("nan"))
kC, lC = s.call_frac * s.capital / M, s.loan_frac * s.capital / M
rate_txt = f"{s.loan_rate:.2%}" if s.loan_rate is not None else f"the 3-month base rate + {s.spread * 1e4:.0f} bp"
prem_txt = f"{s.premium:.1%} of notional" if s.premium is not None else f"the market's premium of the day ({c:.1%} on the first start)"
st.caption(f"Capital {s.capital / M:,.0f} m, held to {last_day:%d %b %Y}. **1 Long the market**: {s.capital / M:,.0f} m in SPX. "
           f"**2 Levered long**: {(1 + s.loan_frac) * s.capital / M:,.1f} m in SPX − a {lC:,.1f} m loan at {rate_txt}, interest capitalised, "
           + ("repaid from SPX after " + f"{s.tenor:g} years. " if s.repay == "tenor" else "rolled up to the end. ")
           + f"**3 {fsc.LABELS['3']}**: {(1 - s.call_frac) * s.capital / M:,.1f} m in SPX + {kC:,.1f} m of premium in rolling {s.tenor:g}-year ATM calls on SPXFP "
           f"(each call costs {prem_txt}: notional {kC / c:,.0f} m, i.e. {kC / c / 2:,.0f} m of delta at 50 %); at every expiry {s.tax_rate:.0%} of the profit is paid as tax and "
           + ("the whole portfolio goes back to " if s.rebalance == "portfolio" else "the after-tax proceeds are split ") + f"{1 - s.call_frac:.0%} SPX / {s.call_frac:.0%} calls. "
           f"**4 {fsc.LABELS['4']}**: {s.capital / M:,.0f} m in SPX + {kC:,.1f} m of premium bought with a {kC:,.1f} m loan at {rate_txt}; at expiry the tax, "
           "then the rolled-up loan repaid from the proceeds (SPX sold for any shortfall), the rest in new calls, no new loan. "
           f"**5 All in calls**: {s.capital / M:,.0f} m of premium (notional {s.capital / M / c:,.0f} m, {s.capital / M / c / 2:,.0f} m of delta); the payoff rolls into new calls"
           + (", taxed too" if "5" in s.tax_on else ", untaxed") + "; a worthless expiry ends it. Margin calls are flagged on 2 and 4, the two with a loan.")

# ---------------- at a glance: one row per start, one column per scenario
st.markdown("""<style>
div[data-testid="stTable"] table { font-size: 0.9rem; table-layout: fixed; width: 100%; }
div[data-testid="stTable"] th, div[data-testid="stTable"] td { white-space: normal !important; overflow-wrap: anywhere; padding: 0.35rem 0.5rem; vertical-align: top; line-height: 1.3; }
div[data-testid="stTable"] th:nth-child(1), div[data-testid="stTable"] td:nth-child(1) { width: 16%; font-weight: 500; }
</style>""", unsafe_allow_html=True)
FLAG = "background-color: #f8d7d7; color: #1a1a1a"


def cell(sm: pd.DataFrame, sid: str) -> str:
    """'4,874 m · +7.9 % a year' with the margin call or the wipe-out appended."""
    txt = f"{sm.loc['NAV today', sid] / M:,.0f} m · {sm.loc['annualised return', sid]:+.1%} a year"
    first = sm.loc["margin call first on", sid]
    if pd.notna(first):
        txt += f" · ⚠ margin call {first:%d %b %Y} (LTV {sm.loc['LTV at the first margin call', sid]:.0%}, {sm.loc['days in margin call', sid]} days)"
    wiped = sm.loc["wiped out on", sid]
    if pd.notna(wiped):
        txt += f" · wiped out {wiped:%d %b %Y}"
    return txt


glance = pd.DataFrame({NAMES[sid]: [cell(summaries[lab], sid) for lab, _ in runs] for sid in fsc.SCENARIOS}, index=pd.Index([f"{lab}, {d:%d %b %Y}" for lab, d in runs], name="start"))
flags = pd.DataFrame({NAMES[sid]: [pd.notna(summaries[lab].loc["margin call first on", sid]) or pd.notna(summaries[lab].loc["wiped out on", sid]) for lab, _ in runs] for sid in fsc.SCENARIOS}, index=glance.index)
st.table(glance.style.apply(lambda col: [FLAG if flags.loc[i, col.name] else "" for i in col.index], axis=0), border="horizontal")
st.caption(f"NAV on {last_day:%d %b %Y} and the annualised return from each start, per scenario. ⚠ marks a margin call (the loan above {s.margin_call:.0%} of the lending value; "
           "the first day, the LTV that day, the days spent in call); 'wiped out' the day the all-in-calls portfolio's call expired worthless. Shaded cells carry one of the two. The charts and tables below give each start in detail.")
st.download_button("Download every start's daily paths as CSV", fsc.paths_table({lab: results[lab][0] for lab, _ in runs}).to_csv(index=False).encode(),
                   f"five_scenarios_to_{last_day:%Y-%m-%d}.csv", "text/csv")


def spans(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The contiguous runs of True as (first day, last day)."""
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    prev = None
    for day, v in mask.items():
        if v and start is None:
            start = day
        if not v and start is not None:
            out.append((start, prev))
            start = None
        prev = day
    if start is not None:
        out.append((start, prev))
    return out


def fmt_row(sm: pd.DataFrame, row: str, sid: str, info: fsc.ScenarioInfo) -> str:
    """One cell of the per-start table."""
    v = sm.loc[row, sid] if row in sm.index else None
    if row == "NAV today":
        return f"{v / M:,.0f} m ({sm.loc['total return', sid]:+.0%} from the start)"
    if row == "annualised return":
        return f"{v:+.1%}"
    if row == "lowest NAV":
        return f"{v / M:,.0f} m on {sm.loc['lowest NAV on', sid]:%d %b %Y} ({sm.loc['max drawdown', sid]:+.0%} from the high)"
    if row == "margin call":
        if sid not in fsc.LEVERED:
            return "no loan"
        first = sm.loc["margin call first on", sid]
        if pd.isna(first):
            return f"none (highest LTV {sm.loc['max LTV', sid]:.0%})"
        return f"⚠ first on {first:%d %b %Y} at LTV {sm.loc['LTV at the first margin call', sid]:.0%}, {sm.loc['days in margin call', sid]} days in call (highest LTV {sm.loc['max LTV', sid]:.0%})"
    if row == "loan repaid on":
        if sid not in fsc.LEVERED:
            return "—"
        d = sm.loc[row, sid]
        return f"{d:%d %b %Y}" if pd.notna(d) else ("never: rolled up to the end" if sid == "2" and s.repay == "never" else "not yet")
    if row == "calls expired":
        if sid in ("1", "2"):
            return "—"
        txt = f"{sm.loc[row, sid]} ({sm.loc['expired worthless', sid]} worthless)"
        if pd.notna(sm.loc["wiped out on", sid]):
            txt += f"; wiped out on {sm.loc['wiped out on', sid]:%d %b %Y}"
        if info.unexpired:
            txt += "; the last call has not expired: a model mark today"
        return txt
    if row in ("premiums paid", "payoffs received", "tax paid", "interest paid"):
        return "—" if v == 0.0 else f"{v / M:,.0f} m"
    if row == "SPX traded at expiries":
        b, sd = sm.loc["SPX bought at expiries", sid], sm.loc["SPX sold at expiries", sid]
        return "—" if b == 0.0 and sd == 0.0 else f"bought {b / M:,.0f} m, sold {sd / M:,.0f} m"
    if row == "holdings today":
        parts = [f"SPX {sm.loc['SPX today', sid] / M:,.0f} m"]
        if sm.loc["calls today", sid] > 0.0:
            parts.append(f"calls {sm.loc['calls today', sid] / M:,.0f} m (notional {sm.loc['call notional today', sid] / M:,.0f} m)")
        if sm.loc["cash today", sid] > 0.5:
            parts.append(f"cash {sm.loc['cash today', sid] / M:,.0f} m")
        if sm.loc["loan today", sid] > 0.5:
            parts.append(f"loan {sm.loc['loan today', sid] / M:,.0f} m")
        return " · ".join(parts)
    return str(v)


def spread(values: dict[str, float], gap: float, log: bool) -> dict[str, float]:
    """Label positions for the values at the right edge: the same order, pushed apart so that two labels are at least ``gap`` apart
    (in log10 units when the axis is log), the group kept centred on where it would be."""
    f = (lambda v: float(np.log10(max(v, 1e-3)))) if log else (lambda v: v)
    g = (lambda y: float(10.0 ** y)) if log else (lambda y: y)
    order = sorted(values, key=lambda k: values[k])
    ys = [f(values[k]) for k in order]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = (sum(ys) - sum(f(values[k]) for k in order)) / len(ys) if ys else 0.0   # the pushed group drifts up: bring its centre back
    lo = min(f(v) for v in values.values()) if values else 0.0
    ys = [y - shift for y in ys]
    if ys and ys[0] < lo - gap:
        ys = [y + (lo - gap - ys[0]) for y in ys]
    return {k: g(y) for k, y in zip(order, ys, strict=True)}


ROWS = ("NAV today", "annualised return", "lowest NAV", "margin call", "loan repaid on", "calls expired", "premiums paid", "payoffs received", "tax paid", "interest paid", "SPX traded at expiries", "holdings today")

# ---------------- one section per start: the NAV of the five over time (top), the LTV of the two with a loan against the call level (bottom), the table
for lab, d in runs:
    paths, infos = results[lab]
    sm = summaries[lab]
    st.subheader(f"Start {d:%d %b %Y} — {lab}, SPX {float(spx_px.loc[paths['1'].index[0]]):,.0f}")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
    x_end = paths["1"].index[-1]
    ends = {sid: float(paths[sid]["nav"].iloc[-1]) / M for sid in fsc.SCENARIOS}
    top = max(max(ends.values()), max(float(paths[sid]["nav"].max()) / M for sid in fsc.SCENARIOS))
    label_y = spread(ends, np.log10(top / 0.5) * 0.04 if log_scale else top * 0.04, log_scale)   # a label is ≈ 4 % of the axis tall
    for sid in fsc.SCENARIOS:
        p, info, colr = paths[sid], infos[sid], COLOURS[sid]
        nav = p["nav"] / M
        fig.add_trace(go.Scatter(x=nav.index, y=nav, name=NAMES[sid], line={"color": colr, "width": 2}, legendgroup=sid,
                                 hovertemplate="%{x|%d %b %Y}<br>%{fullData.name}: %{y:,.0f} m<extra></extra>"), row=1, col=1)
        if info.expiries:   # hollow markers at the expiries: what the call paid, the tax, what was done with the rest
            ex = [e for e in info.expiries]
            fig.add_trace(go.Scatter(x=[e.date for e in ex], y=[nav.loc[e.date] for e in ex], mode="markers", name=f"{sid}: expiry", legendgroup=sid, showlegend=False,
                                     marker={"symbol": "circle-open", "size": 9, "color": colr, "line": {"width": 2}},
                                     customdata=[[e.payoff / M, e.tax / M, e.loan_repaid / M, e.spx_traded / M, e.new_premium / M, e.new_notional / M] for e in ex],
                                     hovertemplate=(f"{NAMES[sid]}, call expiry %{{x|%d %b %Y}}<br>payoff %{{customdata[0]:,.0f}} m · tax %{{customdata[1]:,.0f}} m · loan repaid %{{customdata[2]:,.0f}} m"
                                                    "<br>SPX traded %{customdata[3]:+,.0f} m · new call: premium %{customdata[4]:,.0f} m on %{customdata[5]:,.0f} m notional<extra></extra>")), row=1, col=1)
        if sid == "2" and info.loan_repaid_on is not None:
            fig.add_trace(go.Scatter(x=[info.loan_repaid_on], y=[nav.loc[info.loan_repaid_on]], mode="markers", name="2: loan repaid", legendgroup=sid, showlegend=False,
                                     marker={"symbol": "triangle-up", "size": 11, "color": colr, "line": {"color": "white", "width": 1}},
                                     hovertemplate=f"{NAMES[sid]}: loan repaid from SPX on %{{x|%d %b %Y}}<extra></extra>"), row=1, col=1)
        if info.wiped_out_on is not None:
            fig.add_annotation(x=info.wiped_out_on, y=0.0, text=f"<b>{sid} wiped out</b> {info.wiped_out_on:%d %b %Y}", showarrow=True, arrowhead=2, arrowcolor=colr, ax=40, ay=-40,
                               font={"size": 11, "color": INK}, bgcolor="rgba(255, 255, 255, 0.9)", bordercolor=colr, borderwidth=1, row=1, col=1)
        if sid in fsc.LEVERED:
            ltv = p["ltv"] * 100.0
            fig.add_trace(go.Scatter(x=ltv.index, y=ltv, name=f"{NAMES[sid]}: LTV", line={"color": colr, "width": 2}, legendgroup=sid, showlegend=False,
                                     hovertemplate="%{x|%d %b %Y}<br>%{fullData.name} %{y:.0f} %<extra></extra>"), row=2, col=1)
            if info.margin_call_first is not None:   # the margin call: a red ✕ on the first day, the days in call shaded, a label
                first = info.margin_call_first
                for a, b in spans(p["margin_call"]):
                    fig.add_vrect(x0=a, x1=b + pd.DateOffset(days=1), fillcolor="rgba(192, 0, 0, 0.12)", line_width=0, row="all", col=1)
                fig.add_trace(go.Scatter(x=[first], y=[nav.loc[first]], mode="markers", name=f"{sid}: margin call", legendgroup=sid, showlegend=False,
                                         marker={"symbol": "x", "size": 13, "color": CALL_RED, "line": {"color": "white", "width": 1}},
                                         hovertemplate=f"{NAMES[sid]}: margin call on %{{x|%d %b %Y}}, LTV {float(p['ltv'].loc[first]):.0%}, {info.days_in_margin_call} days in call<extra></extra>"), row=1, col=1)
                fig.add_annotation(x=first, y=nav.loc[first], text=f"<b>⚠ {sid} margin call</b> {first:%d %b %Y} — LTV {float(p['ltv'].loc[first]):.0%}", showarrow=True, arrowhead=2, arrowcolor=CALL_RED,
                                   ax=0, ay=-50 if sid == "2" else -90, font={"size": 11, "color": INK}, bgcolor="rgba(255, 255, 255, 0.92)", bordercolor=CALL_RED, borderwidth=1, row=1, col=1)
        fig.add_annotation(x=x_end, y=np.log10(label_y[sid]) if log_scale else label_y[sid], xref="x", yref="y", text=f"<span style='color:{colr}'>●</span> {sid}: {ends[sid]:,.0f} m", showarrow=False,
                           xanchor="left", xshift=6, font={"size": 11, "color": INK}, row=1, col=1)
    for r_ in corr[corr["trough"] <= last_day].itertuples():   # the market peaks (dotted) and bottoms (dashed) inside the run
        for day, dash, what in ((r_.peak, "dot", "peak"), (r_.trough, "dash", "bottom")):
            if day >= paths["1"].index[0]:
                fig.add_vline(x=day.timestamp() * 1000, line={"color": GREY, "width": 1, "dash": dash}, row="all", col=1)
                fig.add_annotation(x=day, y=1 if what == "peak" else 0.955, yref="y domain", text=f"{what} {day:%b %Y}", showarrow=False, font={"size": 10, "color": GREY},
                                   xanchor="right" if what == "peak" else "left", yanchor="top", row=1, col=1)
    call_pct = s.margin_call * 100.0
    fig.add_hline(y=call_pct, line={"color": CALL_RED, "width": 1, "dash": "dot"}, annotation={"text": f"margin call (LTV {s.margin_call:.0%})", "font": {"size": 10, "color": CALL_RED}}, annotation_position="top left", row=2, col=1)
    fig.add_hrect(y0=call_pct, y1=120.0, fillcolor="rgba(192, 0, 0, 0.06)", line_width=0, row=2, col=1)
    if log_scale:
        fig.update_yaxes(title_text="NAV (USD m, log)", type="log", row=1, col=1)
    else:
        fig.update_yaxes(title_text="NAV (USD m)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="LTV", range=[0, 120], ticksuffix=" %", row=2, col=1)
    fig.update_xaxes(range=[paths["1"].index[0] - pd.DateOffset(days=30), x_end + pd.DateOffset(days=30)], row=1, col=1)
    fig.update_xaxes(title_text="Date", range=[paths["1"].index[0] - pd.DateOffset(days=30), x_end + pd.DateOffset(days=30)], row=2, col=1)
    fig.update_layout(height=660, title={"text": f"From {d:%d %b %Y}: the NAV of the five scenarios, and the LTV of the two with a loan", "x": 0.02, "font": {"size": 15}},
                      **{**LAYOUT, "legend": {"orientation": "h", "y": -0.1, "x": 0, "yanchor": "top"}, "margin": ALIGNED}, hovermode="x unified")
    fig.update_yaxes(automargin=False)
    st.plotly_chart(fig, width="stretch")
    table = pd.DataFrame({NAMES[sid]: [fmt_row(sm, row, sid, infos[sid]) for row in ROWS] for sid in fsc.SCENARIOS}, index=pd.Index(ROWS, name=" "))
    st.table(table.style.apply(lambda col: [FLAG if (i == "margin call" and str(col[i]).startswith("⚠")) or (i == "calls expired" and "wiped out" in str(col[i])) else "" for i in col.index], axis=0), border="horizontal")
    st.caption(f"Top: the NAV (SPX + calls at their model mark + cash − loan) of each scenario from {d:%d %b %Y} to {last_day:%d %b %Y}, in USD m, the final value labelled at the right edge; "
               "hollow circles: a call expiry (hover: payoff, tax, loan repaid, SPX traded, the new call); the triangle: scenario 2's loan repaid from SPX; a red ✕ with a label and the shaded days: a margin call. "
               f"Bottom: the LTV (loan ÷ lending value, {s.lv_equity:.0%} of the SPX and {s.lv_calls:.0%} of the calls) of scenarios 2 and 4; the bank calls above the dotted {s.margin_call:.0%} line; it drops to zero when the loan is repaid. "
               "Dotted verticals: the market peaks; dashed: the bottoms. The table reads the same run: the NAV today, the return, the low, the margin call, the loan, the calls and the money flows since the start.")

st.caption(f"Calls: {s.tenor:g}-year at-the-money calls on SPXFP, cash-settled at expiry, " + (f"each costing {s.premium:.1%} of notional and marked between expiries with Black–Scholes (q = r, the {s.tenor:g}-year Treasury of the day) at the vol that premium implied on its purchase day; "
           if s.premium is not None else f"priced and marked at the {s.tenor:g}-year implied vol of the day (extrapolated from the 24-month Bloomberg series, Assumptions 19l); ")
           + f"the delta quoted above is the investor's 50 % of notional (the model's at the first start: {info0['3'].delta0:.0%} of notional). Tax: {s.tax_rate:.0%} of payoff − premium when positive, at expiry, no loss carry-forward, no tax on the SPX. "
           f"The SPX is held with dividends reinvested net of {s.wht:.0%} withholding; the loan accrues simple interest ACT/360, capitalised daily. A margin call is flagged, not acted on: nothing is sold for it. "
           "Consecutive starts overlap: the three are illustrations of history, not probabilities. For illustrative purposes only.")
