"""Five scenarios — the investor's five ways to hold the capital, from five named start dates (the data start, the dot-com peak and bottom,
the GFC peak and bottom) or any other, held to today (page 3 of app/historical_app.py). Short tables, the explanations in the notes at the bottom.

1 long the market; 2 levered long (a loan on top, repaid after one tenor); 3 SPX + a sleeve of rolling 5-year calls (tax at expiry, each slot
refilled to its share of the portfolio); 4 long + the sleeve bought on a loan (tax, the slot's share of the loan repaid from the proceeds, the rest
in new calls); 5 all in calls. The sleeve is sized by exposure (delta × notional, 86 % of the capital) or by premium, and built in weekly or
monthly slots, the money waiting in SPX. Margin calls flagged on 2 and 4. Engine: fosim.analytics.five_scenarios. Inputs: the page's sidebar
(common.scenarios_sidebar); the tenor and the withholding tax are shared.
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
st.caption("A separate, simplified exercise, independent of the two *Keep the loan or rotate into calls* pages: a different portfolio, the investor's own rules as written. "
           "The notes at the bottom say what each scenario does and how to read the charts and tables.")
c_starts, c_extra, c_log = st.columns([3, 1.6, 0.9])
picked = c_starts.multiselect("Start dates", list(labels), default=list(labels), key="f_starts", help="Each start is held to the last data day. The five named starts: when the data starts, the top and the bottom of the dot-com bubble, the top and the bottom of the GFC.")
extra = c_extra.checkbox("Add another start date", key="f_extra")
extra_date = c_extra.date_input("Start", first_day.date(), min_value=first_day.date(), max_value=(last_day - pd.DateOffset(days=30)).date(), key="f_extra_date", disabled=not extra, label_visibility="collapsed")
log_scale = c_log.checkbox("Log scale", value=True, key="f_log", help="The NAV and exposure axes in log scale (default): the all-in-calls portfolio can end a hundred times the others, which flattens the rest on a linear axis.")
runs: list[tuple[str, pd.Timestamp]] = [labels[k] for k in labels if k in picked]
if extra:
    runs.append(("chosen start", pd.Timestamp(extra_date)))
if not runs:
    st.info("Pick at least one start date.")
    st.stop()

results = {lab: scenarios_cached(str(d.date()), str(last_day.date()), s) for lab, d in runs}
summaries = {lab: fsc.summary(*results[lab]) for lab, _ in runs}
info0 = results[runs[0][0]][1]
i3, i4, i5 = info0["3"], info0["4"], info0["5"]
c = i3.prem_frac0
lC = s.loan_frac * s.capital / M
rate_txt = f"{s.loan_rate:.2%}" if s.loan_rate is not None else f"the 3-month base rate + {s.spread * 1e4:.0f} bp"
prem_txt = f"{s.premium:.1%} of notional" if s.premium is not None else f"the market's premium of the day ({c:.1%} at the first start)"
unit = "weekly" if s.build_unit == "week" else "monthly"
sleeve_txt = f"{s.exposure_frac:.0%} of the capital in exposure" if s.sizing == "exposure" else f"{s.call_frac:.0%} of the capital in premium"
build_short = "in one shot" if s.build_steps == 1 else f"in {s.build_steps} {unit} slots"
st.caption(f"Capital {s.capital / M:,.0f} m · loan {lC:,.1f} m at {rate_txt} · call sleeve {sleeve_txt}, built {build_short} · {s.tenor:g}-year ATM calls on SPXFP at {prem_txt} · "
           f"tax {s.tax_rate:.0%} of the profit at expiry · every start held to {last_day:%d %b %Y}.")

# ---------------- at a glance: one row per start, one column per scenario
st.markdown("""<style>
div[data-testid="stTable"] table { font-size: 0.9rem; table-layout: fixed; width: 100%; }
div[data-testid="stTable"] th, div[data-testid="stTable"] td { white-space: normal !important; overflow-wrap: anywhere; padding: 0.35rem 0.5rem; vertical-align: top; line-height: 1.3; }
div[data-testid="stTable"] th:nth-child(1), div[data-testid="stTable"] td:nth-child(1) { width: 16%; font-weight: 500; }
</style>""", unsafe_allow_html=True)
FLAG = "background-color: #f8d7d7; color: #1a1a1a"


def cell(sm: pd.DataFrame, sid: str) -> str:
    """'4,874 m · +7.9 %' with the margin call (⚠) or the wipe-out (✕) appended."""
    txt = f"{sm.loc['NAV today', sid] / M:,.0f} m · {sm.loc['annualised return', sid]:+.1%}"
    first = sm.loc["margin call first on", sid]
    if pd.notna(first):
        txt += f" · ⚠ {first:%b %Y}"
    wiped = sm.loc["wiped out on", sid]
    if pd.notna(wiped):
        txt += f" · ✕ {wiped:%b %Y}"
    return txt


glance = pd.DataFrame({NAMES[sid]: [cell(summaries[lab], sid) for lab, _ in runs] for sid in fsc.SCENARIOS}, index=pd.Index([f"{lab}, {d:%d %b %Y}" for lab, d in runs], name="start"))
flags = pd.DataFrame({NAMES[sid]: [pd.notna(summaries[lab].loc["margin call first on", sid]) or pd.notna(summaries[lab].loc["wiped out on", sid]) for lab, _ in runs] for sid in fsc.SCENARIOS}, index=glance.index)
st.table(glance.style.apply(lambda col: [FLAG if flags.loc[i, col.name] else "" for i in col.index], axis=0), border="horizontal")
st.caption(f"NAV on {last_day:%d %b %Y} · annualised return since the start. ⚠ a margin call (first month), ✕ wiped out (shaded).")
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
    """One cell of the per-start table: a number and, where it matters, a date."""
    v = sm.loc[row, sid] if row in sm.index else None
    if row == "NAV today":
        return f"{v / M:,.0f} m · {sm.loc['annualised return', sid]:+.1%} a year"
    if row == "lowest NAV":
        return f"{v / M:,.0f} m, {sm.loc['lowest NAV on', sid]:%b %Y}"
    if row == "exposure today":
        return f"{v / M:,.0f} m"
    if row == "margin call":
        if sid not in fsc.LEVERED:
            return "—"
        first = sm.loc["margin call first on", sid]
        if pd.isna(first):
            return f"none, LTV up to {sm.loc['max LTV', sid]:.0%}"
        return f"⚠ {first:%d %b %Y}, LTV {sm.loc['LTV at the first margin call', sid]:.0%}, {sm.loc['days in margin call', sid]} days"
    if row == "loan repaid":
        if sid not in fsc.LEVERED:
            return "—"
        d = sm.loc["loan repaid on", sid]
        return f"{d:%d %b %Y}" if pd.notna(d) else ("never" if sid == "2" and s.repay == "never" else "not yet")
    if row == "calls expired":
        if sid in ("1", "2"):
            return "—"
        txt = f"{sm.loc[row, sid]} ({sm.loc['expired worthless', sid]} worthless)"
        if pd.notna(sm.loc["wiped out on", sid]):
            txt += f" · ✕ {sm.loc['wiped out on', sid]:%b %Y}"
        return txt
    if row in ("premiums paid", "payoffs received", "tax paid", "interest paid"):
        return "—" if v == 0.0 else f"{v / M:,.0f} m"
    if row == "holdings today":
        parts = [f"SPX {sm.loc['SPX today', sid] / M:,.0f} m"]
        if sm.loc["calls today", sid] > 0.0:
            parts.append(f"calls {sm.loc['calls today', sid] / M:,.0f} m ({int(sm.loc['calls alive today', sid])})")
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


ROWS = ("NAV today", "lowest NAV", "exposure today", "margin call", "loan repaid", "calls expired", "premiums paid", "payoffs received", "tax paid", "interest paid", "holdings today")

# ---------------- one section per start: the NAV of the five over time (top), the LTV of the two with a loan against the call level (bottom), the table
for lab, d in runs:
    paths, infos = results[lab]
    sm = summaries[lab]
    st.subheader(f"Start {d:%d %b %Y} — {lab}, SPX {float(spx_px.loc[paths['1'].index[0]]):,.0f}")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.54, 0.26, 0.20], vertical_spacing=0.05)
    x_end = paths["1"].index[-1]
    ends = {sid: float(paths[sid]["nav"].iloc[-1]) / M for sid in fsc.SCENARIOS}
    top = max(max(ends.values()), max(float(paths[sid]["nav"].max()) / M for sid in fsc.SCENARIOS))
    tiny = s.capital / 1000.0   # USD: below this a portfolio counts as gone
    lo_nav = min(float(paths[sid]["nav"][paths[sid]["nav"] > tiny].min()) for sid in fsc.SCENARIOS) / M
    floor = max(tiny / M, 0.5 * lo_nav)   # the log axis stops half a step under the lowest NAV that still counts: a wiped-out portfolio falls off the chart instead of stretching it
    y_top = np.log10(top * 1.5)
    label_y = spread({sid: max(v, floor * 2.0) if log_scale else v for sid, v in ends.items()}, (y_top - np.log10(floor)) * 0.04 if log_scale else top * 0.04, log_scale)   # a label is ≈ 4 % of the axis tall; a wiped-out portfolio's label sits on the floor
    for sid in fsc.SCENARIOS:
        p, info, colr = paths[sid], infos[sid], COLOURS[sid]
        nav = p["nav"] / M
        fig.add_trace(go.Scatter(x=nav.index, y=nav, name=NAMES[sid], line={"color": colr, "width": 2}, legendgroup=sid,
                                 hovertemplate="%{x|%d %b %Y}<br>%{fullData.name}: %{y:,.0f} m<extra></extra>"), row=1, col=1)
        exposure = p["exposure"] / M
        fig.add_trace(go.Scatter(x=exposure.index, y=exposure, name=f"{NAMES[sid]}: exposure", line={"color": colr, "width": 1.5}, legendgroup=sid, showlegend=False,
                                 hovertemplate="%{x|%d %b %Y}<br>%{fullData.name} %{y:,.0f} m<extra></extra>"), row=2, col=1)
        if info.expiries:   # hollow markers at the expiries: what the call paid, the tax, what was done with the rest
            ex = [e for e in info.expiries]
            fig.add_trace(go.Scatter(x=[e.date for e in ex], y=[nav.loc[e.date] for e in ex], mode="markers", name=f"{sid}: expiry", legendgroup=sid, showlegend=False,
                                     marker={"symbol": "circle-open", "size": 9 if info.n_tranches == 1 else 5, "color": colr, "line": {"width": 2 if info.n_tranches == 1 else 1}},
                                     customdata=[[e.payoff / M, e.tax / M, e.loan_repaid / M, e.spx_traded / M, e.new_premium / M, e.new_notional / M, e.slot + 1] for e in ex],
                                     hovertemplate=(f"{NAMES[sid]}, slot %{{customdata[6]}} expiry %{{x|%d %b %Y}}<br>payoff %{{customdata[0]:,.0f}} m · tax %{{customdata[1]:,.0f}} m · loan repaid %{{customdata[2]:,.0f}} m"
                                                    "<br>SPX traded %{customdata[3]:+,.0f} m · new call: premium %{customdata[4]:,.0f} m on %{customdata[5]:,.0f} m notional<extra></extra>")), row=1, col=1)
        if sid == "2" and info.loan_repaid_on is not None:
            fig.add_trace(go.Scatter(x=[info.loan_repaid_on], y=[nav.loc[info.loan_repaid_on]], mode="markers", name="2: loan repaid", legendgroup=sid, showlegend=False,
                                     marker={"symbol": "triangle-up", "size": 11, "color": colr, "line": {"color": "white", "width": 1}},
                                     hovertemplate=f"{NAMES[sid]}: loan repaid from SPX on %{{x|%d %b %Y}}<extra></extra>"), row=1, col=1)
        if info.wiped_out_on is not None:
            fig.add_annotation(x=info.wiped_out_on, y=np.log10(floor) if log_scale else 0.0, text=f"<b>{sid} wiped out</b> {info.wiped_out_on:%d %b %Y}", showarrow=True, arrowhead=2, arrowcolor=colr, ax=40, ay=-40,
                               font={"size": 11, "color": INK}, bgcolor="rgba(255, 255, 255, 0.9)", bordercolor=colr, borderwidth=1, row=1, col=1)
        if sid in fsc.LEVERED:
            ltv = p["ltv"] * 100.0
            fig.add_trace(go.Scatter(x=ltv.index, y=ltv, name=f"{NAMES[sid]}: LTV", line={"color": colr, "width": 2}, legendgroup=sid, showlegend=False,
                                     hovertemplate="%{x|%d %b %Y}<br>%{fullData.name} %{y:.0f} %<extra></extra>"), row=3, col=1)
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
                                   xanchor="left", xshift=3, yanchor="top", row=1, col=1)
    build_end = infos["3"].build_end
    if build_end > paths["1"].index[0]:   # the build: a light band from the start to the last slot's purchase day
        fig.add_vrect(x0=paths["1"].index[0], x1=build_end, fillcolor="rgba(120, 120, 120, 0.08)", line_width=0, row="all", col=1)
        fig.add_annotation(x=build_end, y=1, yref="y domain", text=f"build to {build_end:%b %Y}", showarrow=False, font={"size": 10, "color": GREY}, xanchor="left", xshift=3, yanchor="top", row=2, col=1)
    call_pct = s.margin_call * 100.0
    fig.add_hline(y=call_pct, line={"color": CALL_RED, "width": 1, "dash": "dot"}, annotation={"text": f"margin call (LTV {s.margin_call:.0%})", "font": {"size": 10, "color": CALL_RED}}, annotation_position="top left", row=3, col=1)
    fig.add_hrect(y0=call_pct, y1=120.0, fillcolor="rgba(192, 0, 0, 0.06)", line_width=0, row=3, col=1)
    if log_scale:
        fig.update_yaxes(title_text="NAV (USD m, log)", type="log", range=[np.log10(floor), y_top], row=1, col=1)
        exp_top = max(float(paths[sid]["exposure"].max()) / M for sid in fsc.SCENARIOS)
        lo_exp = min(float(paths[sid]["exposure"][paths[sid]["exposure"] > tiny].min()) for sid in fsc.SCENARIOS) / M
        fig.update_yaxes(title_text="Exposure (USD m, log)", type="log", range=[np.log10(max(tiny / M, 0.5 * lo_exp)), np.log10(max(exp_top, floor * 10.0) * 1.5)], row=2, col=1)
    else:
        fig.update_yaxes(title_text="NAV (USD m)", rangemode="tozero", row=1, col=1)
        fig.update_yaxes(title_text="Exposure (USD m)", rangemode="tozero", row=2, col=1)
    fig.update_yaxes(title_text="LTV", range=[0, 120], ticksuffix=" %", row=3, col=1)
    x_range = [paths["1"].index[0] - pd.DateOffset(days=30), x_end + pd.DateOffset(days=30)]
    fig.update_xaxes(range=x_range, row=1, col=1)
    fig.update_xaxes(range=x_range, row=2, col=1)
    fig.update_xaxes(title_text="Date", range=x_range, row=3, col=1)
    fig.update_layout(height=820, title={"text": f"From {d:%d %b %Y}: NAV, exposure, LTV", "x": 0.02, "font": {"size": 15}},
                      **{**LAYOUT, "legend": {"orientation": "h", "y": -0.1, "x": 0, "yanchor": "top"}, "margin": ALIGNED}, hovermode="x unified")
    fig.update_yaxes(automargin=False)
    st.plotly_chart(fig, width="stretch")
    table = pd.DataFrame({NAMES[sid]: [fmt_row(sm, row, sid, infos[sid]) for row in ROWS] for sid in fsc.SCENARIOS}, index=pd.Index(ROWS, name=" "))
    st.table(table.style.apply(lambda col: [FLAG if (i == "margin call" and str(col[i]).startswith("⚠")) or (i == "calls expired" and "✕" in str(col[i])) else "" for i in col.index], axis=0), border="horizontal")

# ---------------- the notes: the setup (the assumptions of Assumptions 40), the five scenarios, the rules at expiry, how to read the page
cap = s.capital / M
if s.sizing == "exposure":
    sleeve_lines = (f"- **Sleeve**: {s.exposure_frac:.0%} of the capital in exposure = {s.exposure_frac * cap:,.0f} m of delta × notional; premium = exposure × {c:.1%} ÷ the delta of the day "
                    f"(first slot at the first start: delta {i3.delta0:.0%}, premium {i3.premium0 / M:,.1f} m; the whole build {i3.premium_build / M:,.0f} m).")
    refill = f"the slot refilled to its share of the portfolio ({s.exposure_frac:.0%} of the NAV in exposure ÷ {s.build_steps}); SPX sold or bought for the difference"
else:
    sleeve_lines = (f"- **Sleeve**: {s.call_frac:.0%} of the capital in premium = {s.call_frac * cap:,.1f} m, notional {s.call_frac * cap / c:,.0f} m; "
                    f"the note counts {s.call_frac * cap / c / 2:,.0f} m of delta at 50 %, the model's delta at the first start is {i3.delta0:.0%} ({s.call_frac * cap / c * i3.delta0:,.0f} m).")
    refill = (f"the slot refilled to its share of the portfolio ({s.call_frac:.0%} of the NAV in premium ÷ {s.build_steps}); SPX sold or bought for the difference"
              if s.rebalance == "portfolio" else f"the after-tax proceeds split {1 - s.call_frac:.0%} SPX / {s.call_frac:.0%} calls; the SPX held untouched")
build_line = ("- **Build**: one shot on the start day." if s.build_steps == 1
              else f"- **Build**: {s.build_steps} {unit} slots (first start: {i3.start:%d %b %Y} → {i3.build_end:%d %b %Y}); the money waits in SPX and is rotated slot by slot; each slot has its own expiry.")
marks_line = (f"marked between expiries with Black–Scholes (q = r = the {s.tenor:g}-year Treasury of the day) at the vol that premium implied on the purchase day, constant for that call"
              if s.premium is not None else f"priced and marked at the {s.tenor:g}-year implied vol of the day (extrapolated from the 24-month Bloomberg series, Assumptions 19l)")
st.markdown("##### Notes")
st.caption(f"""**Setup** (the sidebar; Assumptions 40)
- **Capital**: {cap:,.0f} m; every start held to {last_day:%d %b %Y}; historical data only, September 1997 → today.
- **Loan**: {lC:,.1f} m ({s.loan_frac:.0%} of the capital) at {rate_txt}; simple interest ACT/360, capitalised daily; """
           + (f"repaid from SPX after {s.tenor:g} years (scenario 2)." if s.repay == "tenor" else "rolled up to the end (scenario 2).") + f"""
- **Calls**: {s.tenor:g}-year at-the-money calls on SPXFP, cash-settled at expiry, each costing {prem_txt}; {marks_line}; calls alive today are at that mark.
{sleeve_lines}
{build_line}
- **Tax**: {s.tax_rate:.0%} of a call's profit (payoff − premium, when positive) at expiry, scenarios 3 and 4""" + (" and 5" if "5" in s.tax_on else "") + """; no loss carry-forward; no tax on the SPX.
- **Margin call**: the loan above """ + f"{s.margin_call:.0%} of the lending value ({s.lv_equity:.0%} of the SPX, {s.lv_calls:.0%} of the calls); flagged (⚠, shaded days), nothing sold for it."
           + f"""
- **SPX**: held with dividends reinvested, net of {s.wht:.0%} withholding.""")
st.caption(f"""**The five scenarios**
- **1 Long the market**: all in SPX.
- **2 Levered long**: {(1 + s.loan_frac) * cap:,.1f} m in SPX on the {lC:,.1f} m loan, drawn on the start day.
- **3 {fsc.LABELS['3']}**: all in SPX; the sleeve's premium sold out of it slot by slot ({i3.premium_build / M:,.0f} m at the first start).
- **4 {fsc.LABELS['4']}**: all in SPX; the same sleeve bought with a loan drawn slot by slot ({i4.loan0 / M:,.0f} m at the first start), repaid slot by slot at the first expiries.
- **5 All in calls**: the SPX rotated into calls slot by slot, all of it ({i5.premium_build / M:,.0f} m of premium at the first start).""")
st.caption(f"""**At a slot's expiry** (payoff in cash, tax where due)
- **3**: {refill}.
- **4**: the slot's share of the loan repaid from the proceeds (SPX sold for a shortfall); the rest in new calls; no new loan.
- **5**: the whole payoff into a new call; a worthless slot lapses; the last one lapsing with nothing left ends the scenario (✕ wiped out).""")
departures = [f"- The sleeve is built over {s.build_steps} {unit} slots, the money waiting in SPX; the note buys everything on the start day (one strike, one expiry). Set *Steps* to 1 for the note's version."
              if s.build_steps > 1 else None,
              "- Scenario 5 is taxed like 3 and 4, so that every assumption applies to every strategy; the note taxes 3 and 4 only. Untick *Tax scenario 5 too* for the note's version."
              if "5" in s.tax_on else None,
              "- Scenario 3 refills an expired slot from the whole portfolio (SPX sold when the call was worthless); the note redistributes the proceeds only, so a worthless call buys nothing and the sleeve can die out. The sidebar offers the note's version."
              if s.rebalance == "portfolio" and s.sizing == "premium" else None,
              (f"- The sleeve is sized by exposure ({s.exposure_frac:.0%} of the capital in delta × notional, ≈ {i3.premium_build / M:,.0f} m of premium); the note fixes the premium at 25 % of the capital (162.5 m, 1,120 m of notional)."
               if s.sizing == "exposure" else None),
              "- Exposure is counted at the model's delta (≈ 44–53 % for a 5-year at-the-money call, with the rate of the day); the note assumes 50 %."]
st.caption("**Departures from the investor's note**\n" + "\n".join(d for d in departures if d))
st.caption("""**Reading the charts**
- **Top — NAV** = SPX + calls at their mark + cash − loan; final value at the right edge. Hollow circles: slot expiries (hover for payoff, tax, loan repaid, SPX traded, new call). Triangle: scenario 2's loan repaid. Grey band: the build. Dotted / dashed verticals: market peaks / bottoms. ⚠ and shading: margin call.
- **Middle — exposure** = SPX + the calls' dollar delta (units × model delta × index): what a 1 % move in the index does to the NAV, × 100. A slot carries its target exposure the day it is bought; the delta then drifts with the index until the slot is refilled.
- **Bottom — LTV** = loan ÷ lending value, scenarios 2 and 4, against the call level; zero once the loan is repaid.""")
st.caption("""**Reading the tables**
- **Overview**: NAV today · annualised return since the start; ⚠ margin call (first month); ✕ wiped out.
- **Per start**: the low and its month; exposure today; margin call (or the highest LTV); loan repaid; calls expired (worthless in brackets); money flows since the start; holdings today (calls alive in brackets).
- **CSV**: every start's daily path.""")
st.caption("""**Caveats**
- Consecutive starts overlap: the named starts are illustrations of history, not probabilities.
- The long-dated vol behind the market-premium option is extrapolated, not observed; a fixed premium ignores the vol of the day.
- For illustrative purposes only.""")
