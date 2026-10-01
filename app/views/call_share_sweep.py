"""Call-share sweep — page 3's scenario 3 (SPX + a sleeve of rolling calls) alone, the size of its call sleeve swept: how the share of the
capital spent on calls shapes the return / risk profile (page 4 of app/historical_app.py, in the separate-exercise group with page 3).

The share runs 0 % (the long portfolio, exactly scenario 1), then the sidebar's range (5 → 50 % of the capital in premium by default). From
the five named starts (interactive) the page shows the risk–return frontier of each start, the measures against the share, and a table; from
every month-end start (Launch button, ≈ 20 minutes the first time, cached) the surface of the annualised return over (start, share), a
heatmap, the bands across the starts and the across-starts table. Engine: fosim.analytics.call_share_sweep over fosim.analytics.five_scenarios.
Inputs: the page's sidebar (common.sweep_sidebar); the tenor and the withholding tax are shared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import fosim.analytics.call_share_sweep as css
import fosim.analytics.five_scenarios as fsc
import fosim.analytics.leverage_stress as lvs
from common import (
    GREY,
    LAYOUT,
    M,
    corrections_cached,
    data_bounds,
    sweep_cached,
    sweep_grid_cached,
    sweep_setup,
)

PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")   # the dataviz reference palette, slots 1–5: here one per named start, in date order, fixed
SEQ_BLUES = ["#dbe9fb", "#9ec4f3", "#5a97e3", "#2a78d6", "#1c5cab", "#0d366b"]   # a single-hue sequential ramp for the surface and the heatmap
INK = "#333333"
ALIGNED = {"l": 80, "r": 120, "t": 30, "b": 40}
MODES = ("the five named starts", "every month-end start")
MEASURES = (("annualised return", "annualised return", True), ("volatility", "annualised volatility", True), ("max drawdown", "max drawdown", True),
            ("VaR 99 %", "1-day VaR 99 %", True), ("Sharpe ratio", "Sharpe ratio", False), ("CVaR 99 %", "1-day CVaR 99 %", True))


@st.cache_data
def named_starts() -> dict[str, pd.Timestamp]:
    return fsc.default_starts()


def pct(k: float) -> str:
    return f"{k * 100:g} %"


def fmt(v: float, spec: str) -> str:
    return "—" if pd.isna(v) else format(float(v), spec)


s = sweep_setup()
first_day, last_day = data_bounds()
starts = named_starts()
START_COLOURS = dict(zip(starts, PALETTE, strict=False))
spx_px = lvs._load(str(lvs.DAILY_FILE)).set_index("date")["spx_px_last"]
labels = {f"{lab} — {d:%d %b %Y}, SPX {float(spx_px.loc[d]):,.0f}": (lab, d) for lab, d in starts.items()}
end = str(last_day.date())

st.title("Call-share sweep — scenario 3")
st.caption("A separate exercise on page 3's scenario 3 alone — SPX plus a sleeve of rolling calls — with the size of the sleeve swept: what a bigger or smaller share of the capital in calls does to the "
           "return and the risk of the mix, read on each start's daily path. The notes at the bottom say how.")
if not s.shares:
    st.error("The swept range in the sidebar needs *from* ≤ *to*, a step above zero and, for a premium share, *to* below 100 %.")
    st.stop()
c_starts, c_mode, c_go = st.columns([3, 1.8, 1.0])
picked = c_starts.multiselect("Start dates", list(labels), default=list(labels), key="c_starts", help="Each start is held to the last data day. The five named starts: when the data starts, the top and the bottom of the dot-com bubble, the top and the bottom of the GFC.")
mode = c_mode.radio("Starts", MODES, key="c_mode", horizontal=True, help="The named starts run as the sidebar changes (a few seconds). Every month-end start — the last trading day of each month up to the last start whose whole build has reached its first expiry — runs on the Launch button (about twenty minutes the first time, then cached) and adds the surface, the heatmap and the bands across the starts.")
grid_mode = mode == MODES[1]
runs: list[tuple[str, pd.Timestamp]] = [labels[k] for k in labels if k in picked]
n_grid = len(css.month_end_starts(last_day, s.tenor, s.build_steps, s.build_unit))
current = (s, mode)
launched = st.session_state.get("c_launched")
if grid_mode:
    c_go.markdown("<div style='height: 1.75rem'></div>", unsafe_allow_html=True)   # the button level with the widgets
    if c_go.button("Launch", key="c_launch", type="primary", width="stretch", disabled=current == launched,
                   help=f"Runs scenario 3 from every month-end start on the sidebar setup: {n_grid} starts × {len(s.shares)} shares, about {max(1, round(n_grid * len(s.shares) * 0.4 / 60))} minutes the first time. Greyed out while nothing has changed since the last run."):
        st.session_state["c_launched"] = current
        st.rerun()
    if launched is None:
        st.info(f"Press **Launch** to run scenario 3 from every month-end start ({n_grid} starts × {len(s.shares)} shares). The named starts below run on the sidebar as it is.")
    elif current != launched:
        st.caption("The parameters have changed: the page still shows the last launched run. Press **Launch** to update it.")
s_run: object = launched[0] if (grid_mode and launched is not None) else s
assert isinstance(s_run, type(s))
if not runs and not (grid_mode and launched is not None):
    st.info("Pick at least one start date.")
    st.stop()

sh = s_run.shares
step_txt = f" step {(sh[2] - sh[1]) * 100:g}" if len(sh) >= 3 else ""
shares_txt = f"0 % then {sh[1] * 100:g} → {sh[-1] * 100:g} %{step_txt}" if len(sh) >= 2 else "0 % only"
prem_txt = f"{s_run.premium:.1%} of notional" if s_run.premium is not None else "the market's premium of the day"
unit = "weekly" if s_run.build_unit == "week" else "monthly"
build_short = "in one shot" if s_run.build_steps == 1 else f"in {s_run.build_steps} {unit} slots"
st.caption(f"Capital {s_run.capital / M:,.0f} m · scenario 3 (SPX + calls) alone, the sleeve sized by {s_run.sizing}, swept {shares_txt} of the capital · built {build_short} · "
           f"{s_run.tenor:g}-year ATM calls on SPXFP at {prem_txt} · tax {s_run.tax_rate:.0%} of the profit at expiry · every start held to {last_day:%d %b %Y}.")

sweeps = {lab: sweep_cached(str(d.date()), end, s_run) for lab, d in runs}
long_named = pd.concat([f.reset_index().assign(start=lab) for lab, f in sweeps.items()], ignore_index=True) if sweeps else pd.DataFrame()

# ---------------- A. the risk–return frontier: one curve per start, the points labelled by the share, the diamond = 0 % = the long portfolio
st.subheader("Risk and return by call share")
fig = go.Figure()
for lab, _ in runs:
    df, c = sweeps[lab], START_COLOURS.get(lab, GREY)
    x, y = df["volatility"].astype(float) * 100.0, df["annualised return"].astype(float) * 100.0
    custom = np.column_stack([df["Sharpe ratio"].astype(float), df["max drawdown"].astype(float)])
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers+text", name=lab, text=[pct(k) for k in df.index], textposition="top center", textfont={"size": 10, "color": INK},
                             line={"color": c, "width": 2}, marker={"size": 8, "color": c}, customdata=custom, legendgroup=lab,
                             hovertemplate=f"{lab} · share %{{text}}<br>return %{{y:.1f}} % a year · volatility %{{x:.1f}} %<br>Sharpe %{{customdata[0]:.2f}} · max drawdown %{{customdata[1]:.0%}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[x.iloc[0]], y=[y.iloc[0]], mode="markers", name=f"{lab}: long", legendgroup=lab, showlegend=False,
                             marker={"symbol": "diamond", "size": 12, "color": "white", "line": {"color": c, "width": 2}},
                             hovertemplate=f"{lab} · 0 % = long the market<br>return %{{y:.1f}} % a year · volatility %{{x:.1f}} %<extra></extra>"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="◇ 0 % = long the market (scenario 1)", marker={"symbol": "diamond", "size": 10, "color": "white", "line": {"color": GREY, "width": 2}}))
fig.update_layout({**LAYOUT, "height": 480, "margin": ALIGNED, "xaxis": {"title": "annualised volatility of the daily NAV returns", "ticksuffix": " %"},
                   "yaxis": {"title": "annualised return", "ticksuffix": " %", "zeroline": True, "zerolinecolor": GREY}})
st.plotly_chart(fig, width="stretch")

# ---------------- B. the measures against the share, one line per start
fig2 = make_subplots(rows=2, cols=3, subplot_titles=[m[1] for m in MEASURES], shared_xaxes=True, vertical_spacing=0.16, horizontal_spacing=0.07)
for i, (row_name, _, is_pct) in enumerate(MEASURES):
    r, col = divmod(i, 3)
    for lab, _ in runs:
        df, c = sweeps[lab], START_COLOURS.get(lab, GREY)
        y = df[row_name].astype(float) * (100.0 if is_pct else 1.0)
        fig2.add_trace(go.Scatter(x=[k * 100.0 for k in df.index], y=y, mode="lines+markers", name=lab, legendgroup=lab, showlegend=i == 0,
                                  line={"color": c, "width": 2}, marker={"size": 6, "color": c}, hovertemplate=f"{lab} · share %{{x:g}} %<br>%{{y:.2f}}{' %' if is_pct else ''}<extra></extra>"), row=r + 1, col=col + 1)
    fig2.update_yaxes(ticksuffix=" %" if is_pct else "", row=r + 1, col=col + 1)
    fig2.update_xaxes(ticksuffix=" %", row=r + 1, col=col + 1)
for col in (1, 2, 3):
    fig2.update_xaxes(title_text="call share (% of the capital)", row=2, col=col)
fig2.add_vline(x=0.0, line={"dash": "dot", "color": GREY, "width": 1}, row=1, col=1)
fig2.add_annotation(x=0.0, y=1.0, yref="y domain", text="0 % = long", showarrow=False, xanchor="left", font={"size": 10, "color": GREY}, row=1, col=1)
fig2.update_layout({**LAYOUT, "height": 560, "margin": ALIGNED})
st.plotly_chart(fig2, width="stretch")

# ---------------- C. every month-end start (after Launch): the surface / heatmap of the return, the bands across the starts, the across-starts table
grid = pd.DataFrame()
if grid_mode and launched is not None:
    grid = sweep_grid_cached(end, s_run)
    starts_g = sorted(pd.Timestamp(d) for d in grid["start"].unique())
    st.subheader(f"Every month-end start, {starts_g[0]:%b %Y} → {starts_g[-1]:%b %Y} ({len(starts_g)} starts)")
    view = st.radio("View", ("surface", "heatmap"), key="c_view", horizontal=True, help="The same numbers: the annualised return of scenario 3 by start date and call share, as a 3-D surface to see the shape or as a heatmap to read values.")
    Z = css.surface_table(grid, "annualised return") * 100.0
    xs = [pd.Timestamp(c).strftime("%Y-%m-%d") for c in Z.columns]
    ys = [float(k) * 100.0 for k in Z.index]
    zs = Z.to_numpy().tolist()
    colorbar = {"title": {"text": "return, % a year"}, "ticksuffix": " %"}
    hover = "start %{x|%b %Y} · share %{y:g} %<br>return %{z:.1f} % a year<extra></extra>"
    if view == "surface":
        fig3 = go.Figure(go.Surface(x=xs, y=ys, z=zs, colorscale=SEQ_BLUES, colorbar=colorbar, hovertemplate=hover, name="annualised return"))
        fig3.update_layout({**LAYOUT, "height": 640, "margin": {"l": 0, "r": 0, "t": 30, "b": 0}, "showlegend": False,
                            "scene": {"xaxis": {"type": "date", "title": {"text": "start"}, "tickformat": "%Y"}, "yaxis": {"title": {"text": "call share (% of the capital)"}, "ticksuffix": " %"},
                                      "zaxis": {"title": {"text": "annualised return (%)"}}, "camera": {"eye": {"x": 1.7, "y": -1.7, "z": 0.8}}, "aspectratio": {"x": 1.6, "y": 1.0, "z": 0.7}}})
    else:
        fig3 = go.Figure(go.Heatmap(x=xs, y=ys, z=zs, colorscale=SEQ_BLUES, colorbar=colorbar, hovertemplate=hover, name="annualised return"))
        corr = corrections_cached(0.20, s_run.wht, "price")
        for _, row in corr.iterrows():
            for day, dash in ((row["peak"], "dot"), (row["trough"], "dash")):
                fig3.add_vline(x=pd.Timestamp(day).strftime("%Y-%m-%d"), line={"dash": dash, "color": INK, "width": 1})
        for lab, d in starts.items():
            fig3.add_annotation(x=d.strftime("%Y-%m-%d"), y=1.0, yref="y domain", text=lab, showarrow=False, xanchor="left", yanchor="bottom", font={"size": 10, "color": START_COLOURS.get(lab, GREY)})
        fig3.update_layout({**LAYOUT, "height": 480, "margin": ALIGNED, "showlegend": False, "xaxis": {"title": "start date"}, "yaxis": {"title": "call share (% of the capital)", "ticksuffix": " %"}})
    st.plotly_chart(fig3, width="stretch")
    st.caption("Dotted / dashed verticals on the heatmap: market peaks / bottoms (the SPX price index corrections of 20 % or more); the named starts labelled at the top.")

    bands = {m: css.grid_bands(grid, m) for m in ("annualised return", "volatility")}
    fig4 = make_subplots(rows=1, cols=2, subplot_titles=("annualised return", "annualised volatility"), horizontal_spacing=0.08)
    for col, (m, b) in enumerate(bands.items(), start=1):
        xb = [float(k) * 100.0 for k in b.index]
        fig4.add_trace(go.Scatter(x=xb, y=b["p95"] * 100.0, mode="lines", name="95th percentile", line={"color": PALETTE[0], "width": 1, "dash": "dot"}, showlegend=col == 1, legendgroup="band",
                                  hovertemplate="share %{x:g} % · 95th %{y:.1f} %<extra></extra>"), row=1, col=col)
        fig4.add_trace(go.Scatter(x=xb, y=b["p5"] * 100.0, mode="lines", name="5th percentile", line={"color": PALETTE[0], "width": 1, "dash": "dot"}, fill="tonexty", fillcolor="rgba(42, 120, 214, 0.15)",
                                  showlegend=col == 1, legendgroup="band", hovertemplate="share %{x:g} % · 5th %{y:.1f} %<extra></extra>"), row=1, col=col)
        fig4.add_trace(go.Scatter(x=xb, y=b["median"] * 100.0, mode="lines+markers", name=f"median of the {len(starts_g)} starts", line={"color": PALETTE[0], "width": 2.5}, marker={"size": 6},
                                  showlegend=col == 1, legendgroup="median", hovertemplate="share %{x:g} % · median %{y:.1f} %<extra></extra>"), row=1, col=col)
        for lab, _ in runs:
            df = sweeps[lab]
            fig4.add_trace(go.Scatter(x=[k * 100.0 for k in df.index], y=df[m].astype(float) * 100.0, mode="lines", name=lab, legendgroup=lab, showlegend=col == 1, opacity=0.6,
                                      line={"color": START_COLOURS.get(lab, GREY), "width": 1}, hovertemplate=f"{lab} · share %{{x:g}} % · %{{y:.1f}} %<extra></extra>"), row=1, col=col)
        fig4.update_xaxes(title_text="call share (% of the capital)", ticksuffix=" %", row=1, col=col)
        fig4.update_yaxes(ticksuffix=" %", row=1, col=col)
    fig4.update_layout({**LAYOUT, "height": 400, "margin": ALIGNED})
    st.plotly_chart(fig4, width="stretch")
    worst_dd = css.surface_table(grid, "max drawdown").min(axis=1)
    b_sh = css.grid_bands(grid, "Sharpe ratio")
    across = pd.DataFrame({
        "median return": [f"{v:+.1%}" for v in bands["annualised return"]["median"]],
        "5th – 95th": [f"{lo:+.1%} … {hi:+.1%}" for lo, hi in zip(bands["annualised return"]["p5"], bands["annualised return"]["p95"], strict=True)],
        "median volatility": [f"{v:.1%}" for v in bands["volatility"]["median"]],
        "median Sharpe": [fmt(v, ".2f") for v in b_sh["median"]],
        "worst max drawdown": [f"{v:.0%}" for v in worst_dd.reindex(bands["volatility"].index)],
        "ahead of long": [fmt(v, ".0%") for v in bands["annualised return"]["ahead of 0 %"]],
    }, index=pd.Index([pct(float(k)) for k in bands["annualised return"].index], name="call share"))
    st.table(across)
    st.caption(f"Across the {len(starts_g)} month-end starts, per call share: the median and the 5th–95th percentiles of the annualised return, the median volatility and Sharpe ratio, "
               "the deepest max drawdown of any start, and the share of the starts on which the mix beats the long portfolio (0 %) on its own start.")

# ---------------- D. the table per start × share, and the CSV
st.markdown("""<style>
div[data-testid="stTable"] table { font-size: 0.9rem; table-layout: fixed; width: 100%; }
div[data-testid="stTable"] th, div[data-testid="stTable"] td { white-space: normal !important; overflow-wrap: anywhere; padding: 0.35rem 0.5rem; vertical-align: top; line-height: 1.3; }
div[data-testid="stTable"] th:nth-child(1), div[data-testid="stTable"] td:nth-child(1) { width: 16%; font-weight: 500; }
</style>""", unsafe_allow_html=True)
if runs:
    cells: dict[str, list[str]] = {}
    index: list[str] = []
    for lab, _ in runs:
        df = sweeps[lab]
        for name, row_name, spec in (("return", "annualised return", "+.1%"), ("vol", "volatility", ".1%"), ("Sharpe", "Sharpe ratio", ".2f"), ("max DD", "max drawdown", ".0%")):
            index.append(f"{lab} · {name}")
            for k in df.index:
                cells.setdefault(pct(float(k)), []).append(fmt(df.loc[k, row_name], spec))
    st.table(pd.DataFrame(cells, index=pd.Index(index, name=" ")))
    st.caption("Per start and call share: the annualised return, the annualised volatility of the daily NAV returns, the Sharpe ratio against the 3-month T-bill and the max drawdown, all to the last data day.")
export = pd.concat([long_named.assign(starts="named")] + ([grid.assign(starts="month-end")] if len(grid) else []), ignore_index=True) if len(long_named) or len(grid) else pd.DataFrame()
if len(export):
    st.download_button("Download the sweep as CSV", export.to_csv(index=False).encode(), f"call_share_sweep_to_{last_day:%Y-%m-%d}.csv", "text/csv")

# ---------------- E. the notes
st.markdown("##### Notes")
rule = "its share of the whole portfolio, SPX sold or bought for the difference" if s_run.rebalance == "portfolio" else "the after-tax proceeds only, the SPX untouched"
st.caption(f"""**Setup** (the sidebar; Assumptions 41)
- **Scenario 3 alone**: the capital ({s_run.capital / M:,.0f} m) in SPX less the premium of a sleeve of rolling {s_run.tenor:g}-year ATM calls on SPXFP at {prem_txt}, built {build_short}, the money waiting in SPX; at a slot's expiry {s_run.tax_rate:.0%} tax on the profit, then the slot refilled to {rule}. Page 3's other four scenarios are not run.
- **The swept share**: {"the premium spent, as a share of the capital" if s_run.sizing == "premium" else "the SPX-equivalent exposure carried, as a share of the capital (the premium follows from the model's delta)"} — {shares_txt}. Everything else is fixed across the sweep; the premium per call stays {prem_txt} whatever the share.
- **0 %** buys no call at all: scenario 3 at 0 % is the long portfolio, page 3's scenario 1, to the cent — the diamond on the frontier.
- **Starts**: the five named starts of page 3 (the data start, the dot-com peak and bottom, the GFC peak and bottom), each held to {last_day:%d %b %Y}; or every month-end start — the last trading day of each calendar month, up to the last start whose whole build has reached its first expiry on the end day ({n_grid} starts on this setup).""")
st.caption("""**Reading the charts**
- **Frontier**: annualised volatility across, annualised return up — up and to the left is better. One curve per start, the points in share order and labelled; a curve that folds back means more calls stopped adding return. The diamond is 0 %, the long portfolio.
- **Measures against the share**: the same numbers one at a time. Max drawdown is negative (the fall from the running peak); the 1-day VaR and CVaR at 99 % are losses, positive, as a % of the NAV; Sharpe = (annualised return − the 3-month T-bill averaged over the period) ÷ volatility.
- **Surface / heatmap** (month-end starts): the annualised return by start date and call share; each node is a full run, the surface shades between nodes. The heatmap reads values better; the surface shows the shape.
- **Bands**: the median and the empirical 5th–95th percentiles across the month-end starts at each share, the named starts drawn thin on top — a spread of history, not a confidence interval.""")
st.caption("""**Reading the tables**
- **Per start × share**: return · volatility · Sharpe · max drawdown, as on page 3's tables (the risk measures on the daily NAV returns, to the wipe-out day if ever).
- **Across the starts** (month-end mode): the percentiles across starts at each share; *ahead of long* = the share of the starts on which that mix beats 0 % on the same start.
- **CSV**: every (start, share) row of the sweep with the whole summary of the run.""")
st.caption("""**Caveats**
- Consecutive month-end starts overlap: the bands are a spread of one history, not independent draws; recent starts whose build has not reached an expiry are left out.
- A bigger sleeve does not move the premium in the model: every call costs the same share of notional whatever the size of the order.
- Between expiries the calls are marked by the model (Black–Scholes at the vol the premium implies), so the volatility and the tail of the call-heavy mixes are model figures.
- For illustrative purposes only.""")
