"""Call-share sweep — page 3's scenario 3 (SPX + a sleeve of rolling calls) alone, the size of its call sleeve swept: how the composition of the
mix — % SPX / % calls — shapes its historical return and volatility (page 4 of app/historical_app.py, in the separate-exercise group with page 3).

The composition runs 100/0 (the long portfolio, exactly scenario 1), then the sidebar's range (95/5 … 50/50 by default, 5 % steps). From the
five named starts (interactive) the page draws the annualised return and the annualised volatility against the composition, one line per start,
and a table per start × composition; from every month-end start (Launch button, ≈ 20 minutes the first time, cached) the median across the
starts with its 5–95 % band joins the chart and an across-starts table follows. Engine: fosim.analytics.call_share_sweep over
fosim.analytics.five_scenarios. Inputs: the page's sidebar (common.sweep_sidebar); the tenor and the withholding tax are shared.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import fosim.analytics.call_share_sweep as css
import fosim.analytics.five_scenarios as fsc
import fosim.analytics.leverage_stress as lvs
from common import (
    LAYOUT,
    M,
    data_bounds,
    run_key,
    sweep_cached,
    sweep_grid_cached,
    sweep_setup,
)

ALIGNED = {"l": 80, "r": 120, "t": 40, "b": 60}
MODES = ("the five named starts", "every month-end start")


@st.cache_data
def named_starts() -> dict[str, pd.Timestamp]:
    return fsc.default_starts()


def fmt(v: float, spec: str) -> str:
    return "—" if pd.isna(v) else format(float(v), spec)


s = sweep_setup()
first_day, last_day = data_bounds()
starts = named_starts()
spx_px = lvs._load(str(lvs.DAILY_FILE)).set_index("date")["spx_px_last"]
labels = {f"{lab} — {d:%d %b %Y}, SPX {float(spx_px.loc[d]):,.0f}": (lab, d) for lab, d in starts.items()}
end = str(last_day.date())

st.title("Call-share sweep — scenario 3")
st.caption("A separate exercise on page 3's scenario 3 alone — SPX plus a sleeve of rolling calls — with the size of the sleeve swept on page 3's assumptions: what the composition of the mix, "
           "% SPX / % calls, does to its historical return, volatility, drawdown, Sharpe ratio, VaR and CVaR, read on each start's daily path; the best composition within a CVaR budget starred. The notes at the bottom say how.")
if not s.shares:
    st.error("The swept range in the sidebar needs *from* ≤ *to*, a step above zero and, for a premium share, *to* below 100 %.")
    st.stop()
c_starts, c_mode, c_go = st.columns([3, 1.8, 1.0])
picked = c_starts.multiselect("Start dates", list(labels), default=list(labels), key="c_starts", help="Each start is held to the last data day. The five named starts: when the data starts, the top and the bottom of the dot-com bubble, the top and the bottom of the GFC.")
mode = c_mode.radio("Starts", MODES, key="c_mode", horizontal=True, help="The named starts run as the sidebar changes (a few seconds). Every month-end start — the last trading day of each month up to the last start whose whole build has reached its first expiry — runs on the Launch button (about twenty minutes the first time, then cached) and adds the median across the starts with its 5–95 % band and the across-starts table.")
grid_mode = mode == MODES[1]
runs: list[tuple[str, pd.Timestamp]] = [labels[k] for k in labels if k in picked]
n_grid = len(css.month_end_starts(last_day, s.tenor, s.build_steps, s.build_unit))
current = (run_key(s), mode)   # the ★ settings are read off the sweeps: changing them never asks for a new launch
launched = st.session_state.get("c_launched")
if grid_mode:
    c_go.markdown("<div style='height: 1.75rem'></div>", unsafe_allow_html=True)   # the button level with the widgets
    if c_go.button("Launch", key="c_launch", type="primary", width="stretch", disabled=current == launched,
                   help=f"Runs scenario 3 from every month-end start on the sidebar setup: {n_grid} starts × {len(s.shares)} compositions, about {max(1, round(n_grid * len(s.shares) * 0.4 / 60))} minutes the first time. Greyed out while nothing has changed since the last run."):
        st.session_state["c_launched"] = current
        st.rerun()
    if launched is None:
        st.info(f"Press **Launch** to run scenario 3 from every month-end start ({n_grid} starts × {len(s.shares)} compositions). The named starts below run on the sidebar as it is.")
    elif current != launched:
        st.caption("The parameters have changed: the page still shows the last launched run. Press **Launch** to update it.")
s_run: object = launched[0] if (grid_mode and launched is not None) else s
assert isinstance(s_run, type(s))
s_run = replace(s_run, risk_budget=s.risk_budget)   # the launched run, the ★ budget as set now
if not runs and not (grid_mode and launched is not None):
    st.info("Pick at least one start date.")
    st.stop()

by_premium = s_run.sizing == "premium"


def comp(k: float) -> str:
    """The composition label of a share: '95 / 5' (% SPX / % calls) under premium sizing, the exposure share under exposure sizing."""
    return f"{(1.0 - k) * 100:g} / {k * 100:g}" if by_premium else f"{k * 100:g} %"


x_title = "composition: % SPX / % calls" if by_premium else "SPX-equivalent exposure of the calls (% of the capital)"
sh = s_run.shares
step_txt = f" step {(sh[2] - sh[1]) * 100:g}" if len(sh) >= 3 else ""
shares_txt = f"0 % then {sh[1] * 100:g} → {sh[-1] * 100:g} %{step_txt}" if len(sh) >= 2 else "0 % only"
prem_txt = f"{s_run.premium:.1%} of notional" if s_run.premium is not None else "the market's premium of the day"
unit = "weekly" if s_run.build_unit == "week" else "monthly"
build_short = "in one shot" if s_run.build_steps == 1 else f"in {s_run.build_steps} {unit} slots"
st.caption(f"Capital {s_run.capital / M:,.0f} m · scenario 3 (SPX + calls) alone, the sleeve sized by {s_run.sizing}, swept {shares_txt} of the capital · built {build_short} · "
           f"{s_run.tenor:g}-year ATM calls on SPXFP at {prem_txt} · tax {s_run.tax_rate:.0%} of the profit at expiry · every start held to {last_day:%d %b %Y}.")

sweeps = {lab: sweep_cached(str(d.date()), end, run_key(s_run)) for lab, d in runs}
long_named = pd.concat([f.reset_index().assign(start=lab) for lab, f in sweeps.items()], ignore_index=True) if sweeps else pd.DataFrame()
grid = sweep_grid_cached(end, run_key(s_run)) if (grid_mode and launched is not None) else pd.DataFrame()
MEASURES = ("annualised return", "volatility", "max drawdown", "Sharpe ratio", "VaR 95 %", "CVaR 95 %")
bands = {m: css.grid_bands(grid, m) for m in MEASURES} if len(grid) else {}
n_starts_g = int(grid["start"].nunique()) if len(grid) else 0

# ---------------- one heat map per start: the measures (rows) by composition — SPX exposure / call exposure (columns)
BLUES = ["#e4eefb", "#9ec4f3", "#2a78d6", "#0d366b"]
ORANGES = ["#fdeedd", "#f6b97e", "#eb6834", "#8a2a05"]
REDS_DOWN = ["#8a1515", "#e07070", "#f6c6c6", "#fdeeee"]   # a negative number: the most negative darkest (drawdown, worst window)
REDS_UP = ["#fdeeee", "#f6c6c6", "#e07070", "#8a1515"]     # a probability of loss: the higher darkest
PURPLES = ["#efe7f7", "#c4a7e3", "#8a5cc6", "#4b2a80"]     # the 1-day tail loss: the heavier darkest
GREENS = ["#e3f4ec", "#8fd4b1", "#1baf7a", "#0b5c3f"]
INK = "#333333"
# (column, row label, colour-bar title, colour scale, number format, multiplier, suffix)
ROWS: list[tuple[str, str, str, list[str], str, float, str]] = [
    ("annualised return", "annualised return", "return<br>% a year", BLUES, "+.1f", 100.0, " %"),
    ("volatility", "volatility", "volatility<br>% a year", ORANGES, ".1f", 100.0, " %"),
    ("max drawdown", "max drawdown", "max drawdown<br>% of the NAV", REDS_DOWN, ".0f", 100.0, " %"),
    ("Sharpe ratio", "Sharpe ratio", "Sharpe<br>ratio", GREENS, ".2f", 1.0, ""),
    ("VaR 95 %", "1-day VaR 95 %", "1-day VaR 95 %<br>% of the NAV", REDS_UP, ".2f", 100.0, " %"),
    ("CVaR 95 %", "1-day CVaR 95 %", "1-day CVaR 95 %<br>% of the NAV", PURPLES, ".2f", 100.0, " %"),
]

def comp_axis(k: float) -> str:
    """The column label of a composition: the SPX exposure and the call exposure, % of the capital."""
    return f"SPX {(1.0 - k) * 100:g} %<br>calls {k * 100:g} %" if by_premium else f"calls {k * 100:g} %<br>of exposure"


def cell(v: float, spec: str, suffix: str) -> str:
    if pd.isna(v):
        return "—"
    if np.isinf(v):
        return "∞"
    return f"{v:{spec}}{suffix}"


def heat_map(df: pd.DataFrame, title: str) -> go.Figure:
    """One row per measure (``ROWS``), one column per composition, the value written in each cell, each row on its own colour scale; the ★
    composition (``call_share_sweep.best_share`` on the sidebar's measure and budget) framed."""
    xs = [comp_axis(float(k)) for k in df.index]
    n = len(ROWS)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.025)
    for i, (col, label, bar, scale, spec, mult, suffix) in enumerate(ROWS, start=1):
        v = df[col].astype(float).to_numpy() * mult
        finite = v[np.isfinite(v)]
        z = np.where(np.isinf(v), (finite.max() * 1.1 if finite.size else 0.0), v)   # an infinite Omega coloured as the top of its row
        fig.add_trace(go.Heatmap(x=xs, y=[label], z=[z], text=[[cell(x, spec, suffix) for x in v]], texttemplate="%{text}", textfont={"size": 12}, colorscale=scale,
                                 colorbar={"title": {"text": bar}, "len": 0.9 / n, "y": 1.0 - (i - 0.5) / n, "thickness": 12}, hovertemplate=f"%{{x}}<br>{label} %{{text}}<extra></extra>"), row=i, col=1)
        fig.update_xaxes(type="category", row=i, col=1)
    fig.update_xaxes(title_text=x_title, row=n, col=1)
    j = list(df.index).index(css.best_share(df, s_run.risk_budget, "CVaR 95 %"))
    fig.add_vrect(x0=j - 0.5, x1=j + 0.5, line={"color": INK, "width": 3}, fillcolor="rgba(0, 0, 0, 0)", row="all", col=1)
    fig.add_annotation(x=j, y=1.0, xref="x", yref="paper", yanchor="bottom", yshift=4, text=f"★ best within the {STAR_TXT} budget: {xs[j].replace('<br>', ' / ')}", showarrow=False, font={"size": 11, "color": INK})
    fig.update_layout({**LAYOUT, "height": 110 * n + 140, "margin": {**ALIGNED, "t": 70}, "showlegend": False, "title": {"text": title, "x": 0.02, "font": {"size": 15}}})
    return fig


STAR_TXT = "1-day CVaR 95 %"
BUDGET_TXT = f"{s_run.risk_budget * 100:g} points of the NAV"
for lab, d in runs:
    st.subheader(f"Start {d:%d %b %Y} — {lab}, SPX {float(spx_px.loc[d]):,.0f}")
    st.plotly_chart(heat_map(sweeps[lab], f"From {d:%d %b %Y}: return, volatility, max drawdown, Sharpe ratio, VaR and CVaR by composition"), width="stretch")
if bands:
    starts_g = sorted(pd.Timestamp(d) for d in grid["start"].unique())
    st.subheader(f"Every month-end start, {starts_g[0]:%b %Y} → {starts_g[-1]:%b %Y} ({n_starts_g} starts): the median")
    med = pd.DataFrame({m: bands[m]["median"] for m in bands})
    st.plotly_chart(heat_map(med, f"Median of the {n_starts_g} month-end starts, by composition"), width="stretch")

# ---------------- the tables: per start × composition; across the month-end starts once launched
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
        for col, label, _bar, _scale, spec, mult, suffix in ROWS:
            index.append(f"{lab} · {label}")
            for k in df.index:
                cells.setdefault(comp(float(k)), []).append(cell(float(df.loc[k, col]) * mult, spec, suffix))
    st.table(pd.DataFrame(cells, index=pd.Index(index, name=" ")))
    st.caption("Per start and composition: the heat maps' numbers, all to the last data day.")
if bands:
    worst_dd = css.surface_table(grid, "max drawdown").min(axis=1)
    across = pd.DataFrame({
        "median return": [f"{v:+.1%}" for v in bands["annualised return"]["median"]],
        "5th – 95th": [f"{lo:+.1%} … {hi:+.1%}" for lo, hi in zip(bands["annualised return"]["p5"], bands["annualised return"]["p95"], strict=True)],
        "median volatility": [f"{v:.1%}" for v in bands["volatility"]["median"]],
        "median Sharpe": [fmt(v, ".2f") for v in bands["Sharpe ratio"]["median"]],
        "median CVaR 95 %": [f"{v:.2%}" for v in bands["CVaR 95 %"]["median"]],
        "worst max drawdown": [f"{v:.0%}" for v in worst_dd.reindex(bands["annualised return"].index)],
        "ahead of long": [fmt(v, ".0%") for v in bands["annualised return"]["ahead of 0 %"]],
    }, index=pd.Index([comp(float(k)) for k in bands["annualised return"].index], name="composition"))
    st.table(across)
    st.caption(f"Across the {n_starts_g} month-end starts, {starts_g[0]:%b %Y} → {starts_g[-1]:%b %Y}, per composition: the median and the 5th–95th percentiles of the annualised return, the median volatility, "
               "Sharpe ratio and 1-day CVaR 95 %, the deepest max drawdown of any start, and the share of the starts on which the mix beats the long portfolio (100 / 0) on its own start.")
export = pd.concat([long_named.assign(starts="named")] + ([grid.assign(starts="month-end")] if len(grid) else []), ignore_index=True) if len(long_named) or len(grid) else pd.DataFrame()
if len(export):
    st.download_button("Download the sweep as CSV", export.to_csv(index=False).encode(), f"call_share_sweep_to_{last_day:%Y-%m-%d}.csv", "text/csv")

# ---------------- the notes
st.markdown("##### Notes")
rule = "its share of the whole portfolio, SPX sold or bought for the difference" if s_run.rebalance == "portfolio" else "the after-tax proceeds only, the SPX untouched"
st.caption(f"""**Setup** (the sidebar; Assumptions 41)
- **Scenario 3 alone**: the capital ({s_run.capital / M:,.0f} m) in SPX less the premium of a sleeve of rolling {s_run.tenor:g}-year ATM calls on SPXFP at {prem_txt}, built {build_short}, the money waiting in SPX; at a slot's expiry {s_run.tax_rate:.0%} tax on the profit, then the slot refilled to {rule}. Page 3's other four scenarios are not run.
- **The composition**: {"the premium spent, as a share of the capital, the rest in SPX — 95 / 5 means 95 % SPX, 5 % calls" if by_premium else "the SPX-equivalent exposure carried by the calls, as a share of the capital (the premium follows from the model's delta, the rest in SPX)"} — {shares_txt}. Everything else is fixed across the sweep; the premium per call stays {prem_txt} whatever the size of the sleeve.
- **100 / 0** buys no call at all: scenario 3 is then the long portfolio, page 3's scenario 1, to the cent.
- **Starts**: the five named starts of page 3 (the data start, the dot-com peak and bottom, the GFC peak and bottom), each held to {last_day:%d %b %Y}; or every month-end start — the last trading day of each calendar month, up to the last start whose whole build has reached its first expiry on the end day ({n_grid} starts on this setup).""")
st.caption(f"""**Reading the heat maps**
- One per start. **Columns**: the composition — the SPX exposure and the call exposure, % of the capital. **Rows**: the annualised return of that mix from the start to the last data day; the annualised volatility of its daily NAV returns (standard deviation × √252); the max drawdown (the deepest fall from a running peak); the Sharpe ratio ((return − the 3-month T-bill averaged over the period) ÷ volatility); the 1-day VaR 95 % (the loss not exceeded on 95 % of days) and CVaR 95 % (the average loss on the worst 5 % of days), % of the NAV, historical. The value is written in each cell, the colour scales it row by row. Each cell is one full run of scenario 3.
- **★ best**: the framed column is the composition with the highest return among those whose {STAR_TXT} exceeds the long portfolio's (100 / 0) by at most {BUDGET_TXT} on the same start — the upside bought without breaking the tail-loss budget (the sidebar sets the budget).
- **Beyond 50 / 50** (the default range runs to 5 / 95): from the starts after the dot-com bust the return keeps rising with the call share while the drawdown runs towards −100 %; from the 2000 peak the mix turns negative past 80 % calls. 0 / 100 is page 3's scenario 5 (all in calls, its own rules), not shown here.
- **Month-end starts** (once launched): one more heat map with the median across the starts at each composition, the same rule on the medians.""")
st.caption("""**Reading the tables**
- **Per start × composition**: the heat maps' numbers, as on page 3's tables (the risk measures on the daily NAV returns, to the wipe-out day if ever).
- **Across the starts** (month-end mode): the percentiles across starts at each composition; *ahead of long* = the share of the starts on which that mix beats 100 / 0 on the same start.
- **CSV**: every (start, composition) row of the sweep with the whole summary of the run.""")
st.caption("""**Caveats**
- Consecutive month-end starts overlap: the band is a spread of one history, not independent draws; recent starts whose build has not reached an expiry are left out.
- A bigger sleeve does not move the premium in the model: every call costs the same share of notional whatever the size of the order.
- Between expiries the calls are marked by the model (Black–Scholes at the vol the premium implies): a window ending between expiries reads a model price.
- For illustrative purposes only.""")
