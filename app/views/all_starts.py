"""Historical simulation — the same two portfolios put on at every start of the grid since 1997 and held to today or to a market bottom."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import fosim.analytics.leverage_stress as lvs
from common import (
    BLUE,
    GREY,
    LAYOUT,
    ORANGE,
    RED,
    A,
    B,
    M,
    Setup,
    all_starts_cached,
    corrections_cached,
    data_bounds,
    setup,
    start_grid,
    table_height,
)

corr = corrections_cached(0.20, 0.15, "price")   # the SPX price index: independent of the setup
first_day, last_day = data_bounds()
ends = {f"today ({last_day:%d %b %Y})": last_day, **{f"the {lab} bottom ({tr_:%d %b %Y}, SPX {lvl:,.0f})": tr_ for lab, tr_, lvl in zip(corr.index, corr["trough"], corr["trough level"], strict=True)}}


def window(s_: Setup, grid_: str, end_label_: str) -> tuple[str, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, bool, list[int]]:
    """(freq, last start, last strike, end day, at a bottom, the start years) of one run: every start up to a week before the end day — today or a
    bottom, the same rule; the rotations still under way on that day run with the steps taken so far."""
    freq_, _, _, last_strike_, _ = start_grid(s_, grid_)
    end_day_ = ends[end_label_]
    last_start_ = end_day_ - pd.DateOffset(days=7)
    return freq_, last_start_, last_strike_, end_day_, end_day_ != last_day, list(range(first_day.year, last_start_.year + 1))


# ---------------- the controls: the run is defined by the sidebar, the grid, the start year and the end day, and runs only when the button is pressed
st.title("Historical simulation")
s_now, grid_now = setup(), str(st.session_state["s_grid"])
c_years, c_end, c_go = st.columns([2, 2, 1.2])   # start years, the final day, the launch
end_label_now = c_end.selectbox("Held until", list(ends), key="r_end", help="Every start runs to this day: today, or the bottom of a market correction to see how the portfolios looked at the worst moment.")
_, _, _, _, _, years_now = window(s_now, grid_now, end_label_now)
picked_now = int(c_years.selectbox("Starts from", years_now, key="r_years", help="Every start from 1 January of that year to the last start; the first year = every start."))
current = (s_now, grid_now, picked_now, end_label_now)
launched = st.session_state.get("launched")
c_go.markdown("<div style='height: 1.75rem'></div>", unsafe_allow_html=True)   # the button level with the widgets
if c_go.button("Launch simulation", key="r_launch", type="primary", width="stretch", disabled=current == launched,
               help="Runs the page on the sidebar setup, the start year and the end day above. Greyed out while nothing has changed since the last run."):
    st.session_state["launched"] = current
    st.rerun()   # redraw at once: the button greyed out, the page on the new run
if launched is None:
    st.info("Set the portfolios in the sidebar and the start year and end day above, then press **Launch simulation**.")
    st.stop()
if current != launched:
    st.caption("The parameters have changed: the page still shows the last launched run. Press **Launch simulation** to update it.")
s, grid, picked, end_label = launched
freq, last_start, last_strike, end_day, at_bottom, years_all = window(s, grid, end_label)
sel_years = [y for y in years_all if y >= picked]
if last_start <= first_day:
    st.warning("No start date fits before that day: shorten the rotation.")
    st.stop()

rs, paths = all_starts_cached(str(first_day.date()), str(last_start.date()), freq, s, str(end_day.date()))
if rs.empty:
    st.warning("No start date fits: shorten the rotation or the tenor.")
    st.stop()
expiry_cut = last_strike - pd.DateOffset(days=7 * (s.build - 1))   # the last start whose 52nd tranche still expires inside the data
unexpired = pd.Series(rs.index > expiry_cut, index=rs.index)      # later starts: some calls have not reached expiry, their NAV is partly a model mark
rotating = rs["end: loan B"] > 0.5e6                              # a loan left on the end day: the rotation was still under way
st.caption(f"The same two portfolios put on at every {'trading day' if freq == 'D' else 'week'} from {first_day:%d %b %Y} to {last_start:%d %b %Y} (a week before the end day) and held to {end_day:%d %b %Y}. "
           + (f"**{int(rotating.sum()):,} of the {len(rs):,} starts (after {rs.index[rotating][0]:%d %b %Y}) were still rotating on {end_day:%d %b %Y}**, with the loan partly repaid and only the tranches "
              "bought so far. " if rotating.any() else "")
           + (f"**{int(unexpired.sum()):,} starts (after {expiry_cut:%d %b %Y}) have calls that have not yet expired**: their NAV on the end day is a model mark, "
              "not a settled outcome. " if unexpired.any() else "Every start has had all its calls expire at least once. ")
           + f"The rotation: {s.tenor:g}-year ATM calls on SPXFP, {s.build} weekly step{'s' if s.build > 1 else ''}, the loan gone after the build; "
           + {"target": "an expiring call is replaced to close the gap to Keep's exposure" + (f", the exposure checked {'every quarter' if s.rebalance == 'quarterly' else 'every month'} and kept within ±{s.band:.0%} of Keep's" if s.rebalance != "none" else ""),
              "delta": "an in-the-money call is replaced on the same dollar delta", "units": "an in-the-money call is replaced on the same index units"}[s.roll]
           + ("; a call that expires worthless lapses." if not s.replace_worthless else "; a worthless call is replaced too."))
if s.roll == "target" and s.rebalance != "none":
    n_up, n_down, n_part = int(rs["B: rebalances up"].sum()), int(rs["B: rebalances down"].sum()), int(rs["B: partial rebalances"].sum())
    st.caption(f"Band trades across the {len(rs):,} starts: **{n_up:,} up** (calls sold, most in the money first, at the mark less {s.haircut * 100:g} vol point{'s' if s.haircut != 0.01 else ''}), "
               f"**{n_down:,} down** ({'ATM calls' if s.below == 'calls' else 'SPX'} bought from the T-bills), **{n_part:,} partial** (the T-bills ran out); "
               f"unwind cost {rs['B: unwind cost'].sum() / M:,.0f} m in all, {rs['B: rolls skipped'].sum():,} expiries with nothing to buy (exposure already at Keep's).")
years = rs["years"]
ann = pd.DataFrame({A: (1.0 + rs["A return"]) ** (1.0 / years) - 1.0, B: (1.0 + rs["B return"]) ** (1.0 / years) - 1.0}, index=rs.index)
lost_all = rs["B: call notional end"] <= 0.0                       # no call left today: the whole option part lapsed
some_lapsed = rs["B: lapsed"] > 0
st.markdown(f"**{len(rs):,} start dates, {rs.index[0]:%d %b %Y} → {rs.index[-1]:%d %b %Y}, each held to {end_day:%d %b %Y} ({years.min():.1f} to {years.max():.1f} years).**")
st.download_button("Download every start date as CSV", lvs.starts_table(rs).to_csv(index=False).encode(), f"any_start_date_since_1997_to_{end_day:%Y-%m-%d}_{'daily' if freq == 'D' else 'weekly'}.csv", "text/csv")

# ---------------- every start date's path, Monte Carlo style, with the distribution of the final NAV on the right
st.subheader(f"Every start date's path to {'today' if not at_bottom else end_label.split(' (')[0]}")
sel_all = rs.index[[d.year in sel_years for d in rs.index]]   # every selected start: the profiles and every table below use all of them
step = 5 if freq == "D" else 1   # with daily starts draw every fifth line (≈ weekly): 4,500 lines would be slow and unreadable
cols = sel_all[::step]
LIGHT_ORANGE = "#f0b46e"


def shade(base: str, light: str, t: float) -> str:
    """Colour between ``light`` (t = 0, the oldest start year) and ``base`` (t = 1, the newest), hex."""
    b_, l_ = [int(base[i:i + 2], 16) for i in (1, 3, 5)], [int(light[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(lo + (hi - lo) * t):02x}" for lo, hi in zip(l_, b_, strict=True))


def fan(w: pd.DataFrame, starts: pd.Index, colr: str, name: str, opacity: float = 0.55) -> go.Scattergl:
    """One trace for many paths: the paths joined with NaN gaps (fast to draw), hover shows the start date."""
    xs: list[object] = []
    ys: list[object] = []
    tags: list[str] = []
    for s0 in starts:
        y = w[s0].dropna()
        xs.extend([*y.index, None])
        ys.extend([*(y.to_numpy() / M), None])
        tags.extend([f"{s0:%d %b %Y}"] * (len(y) + 1))
    return go.Scattergl(x=xs, y=ys, mode="lines", line={"color": colr, "width": 0.8}, opacity=opacity, name=name, customdata=tags, legendgroup=name, showlegend=False,
                        hovertemplate="start %{customdata}<br>%{y:,.0f} m<extra></extra>")


def profile(vals: pd.Series, edges: np.ndarray, colr: str, name: str) -> go.Scatter:
    """The distribution of the final NAV drawn vertically: share of the selected starts in each bin, filled towards the axis."""
    cnt, _ = np.histogram(vals, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    return go.Scatter(x=cnt / max(cnt.sum(), 1), y=centers, mode="lines", fill="tozerox", line={"color": colr, "width": 1}, opacity=0.5, name=name, legendgroup=name, showlegend=False,
                      hovertemplate="%{x:.1%} of the starts end near %{y:,.0f} m<extra></extra>")


fin_all = pd.concat([rs.loc[sel_all, "NAV A end"], rs.loc[sel_all, "NAV B end"]]) / M
bin_w = max(float(np.ceil((fin_all.max() - fin_all.min()) / 40 / 50) * 50), 50.0)   # ≈ 40 bins, rounded to 50 m
edges = np.arange(np.floor(fin_all.min() / bin_w) * bin_w, np.ceil(fin_all.max() / bin_w) * bin_w + bin_w, bin_w)
years_shown = sorted({d.year for d in cols})
y0, y1 = years_all[0], years_all[-1]
LIGHT_RED, LIGHT_BLUE, LIGHT_YELLOW = "#f6c6c6", "#c3cfe8", "#fbe8b0"
bounds = (picked, end_day.year)   # the zoom window follows the launched start year and end day: a widget per range, so it resets to the whole range when they change
zoom = st.slider("Zoom: years shown on the fans and on the charts by start date below (the NAV axes follow the highest value inside the window; drag a rectangle on a chart to zoom further, double-click to reset)",
                 bounds[0], bounds[1], bounds, key=f"r_zoom_{bounds[0]}_{bounds[1]}") if bounds[0] < bounds[1] else bounds
x_lo, x_hi = pd.Timestamp(year=zoom[0], month=1, day=1), min(pd.Timestamp(year=zoom[1], month=12, day=31), end_day)
for title, w, end_col, base, light in ((A, paths["nav_A"], "NAV A end", RED, LIGHT_RED), (B, paths["nav_B"], "NAV B end", BLUE, LIGHT_BLUE)):
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.8, 0.2], horizontal_spacing=0.02)
    in_window = w.loc[(w.index >= x_lo) & (w.index <= x_hi), cols]
    y_top = float(np.nanmax(in_window.to_numpy())) / M * 1.05 if in_window.size and not in_window.isna().all().all() else None
    for yr in years_shown:   # oldest first, so the newest years are drawn on top
        yc = cols[[d.year == yr for d in cols]]
        colr = shade(base, light, (yr - y0) / max(y1 - y0, 1))
        yc_keep = yc if title == A else yc[(~some_lapsed.loc[yc]).to_numpy()]
        if len(yc_keep):
            fig.add_trace(fan(w, yc_keep, colr, f"{yr}"), row=1, col=1)
    if title == B:   # the rotations that stopped buying calls after a correction, drawn last so they sit in front: light yellow (oldest) to orange (newest)
        for yr in years_shown:
            stopped = cols[[d.year == yr and some_lapsed.loc[d] for d in cols]]
            if len(stopped):
                fig.add_trace(fan(w, stopped, shade(ORANGE, LIGHT_YELLOW, (yr - y0) / max(y1 - y0, 1)), f"{yr}, stopped buying calls", opacity=0.9), row=1, col=1)
    for r in corr[corr["trough"] <= end_day].itertuples():   # the market bottoms up to the end day
        fig.add_vline(x=r.trough.timestamp() * 1000, line={"color": GREY, "width": 1, "dash": "dash"}, row=1, col=1)
        fig.add_annotation(x=r.trough, y=1, yref="y domain", text=f"{r.trough:%b %Y}", showarrow=False, font={"size": 10, "color": GREY}, xanchor="left", yanchor="top", row=1, col=1)
    # colour scale legends for the start years (and, for the rotation, the yellow scale of the starts that stopped buying calls)
    scales = [(light, base, "start year", 0.2)] + ([(LIGHT_YELLOW, ORANGE, "start year, stopped buying calls", 0.62)] if title == B else [])
    for lo_c, hi_c, name, x in scales:   # horizontal colour bars under the chart
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker={"colorscale": [[0, lo_c], [1, hi_c]], "cmin": y0, "cmax": y1, "color": [y0, y1], "showscale": True,
                                         "colorbar": {"title": {"text": name, "font": {"size": 11}, "side": "right"}, "orientation": "h", "thickness": 10, "len": 0.36, "x": x, "xanchor": "left",
                                                      "y": -0.22, "yanchor": "top", "tickvals": [y0, (y0 + y1) // 2, y1]}}), row=1, col=1)
    fin_col = rs.loc[sel_all, end_col] / M
    fig.add_trace(profile(fin_col, edges, base, title), row=1, col=2)
    if title == B:
        stopped_all = sel_all[some_lapsed.loc[sel_all].to_numpy()]
        if len(stopped_all):
            fig.add_trace(profile(rs.loc[stopped_all, end_col] / M, edges, ORANGE, "stopped"), row=1, col=2)
    marks = [("min", float(fin_col.min())), ("median", float(fin_col.median())), ("max", float(fin_col.max()))]   # the lowest, median and highest final NAV, marked on the profile
    for _, level in marks:
        fig.add_hline(y=level, line={"color": GREY, "width": 1, "dash": "dot"}, row=1, col=2)
    fig.update_yaxes(title_text="NAV (USD m)", rangemode="tozero", range=[0, y_top] if y_top else None, row=1, col=1)
    fig.update_yaxes(showticklabels=True, side="right", tickvals=[lv for _, lv in marks], ticktext=[f"{nm} {lv:,.0f}" for nm, lv in marks], tickfont={"size": 10}, row=1, col=2)
    fig.update_xaxes(title_text="Date", range=[x_lo, x_hi], row=1, col=1)
    fig.update_xaxes(title_text=f"share of starts, NAV on {end_day:%d %b %Y}", tickformat=".0%", row=1, col=2)
    fig.update_layout(height=500, title={"text": f"{title} — {len(sel_all):,} start dates", "x": 0.02, "font": {"size": 15}}, **{**LAYOUT, "margin": {"l": 40, "r": 90, "t": 30, "b": 90}})
    st.plotly_chart(fig, width="stretch")
n_stop = int(some_lapsed.loc[sel_all].sum())
n_all = int(lost_all.loc[sel_all].sum())
st.caption(f"One line per start date ({len(cols):,} lines{', one start in five on the daily grid' if step > 1 else ''}), month-end values, each leaving from {(s.equity - s.loan) / M:,.0f} m on its own start day and running to {end_day:%d %b %Y}; "
           f"the profile on the right is the distribution of the final NAV over the {len(sel_all):,} selected starts, on the same scale. One colour per start year, light for the oldest and strong for the newest, newer years drawn in front. "
           f"Yellow to orange, same light-to-strong logic and drawn in front, are the rotations that stopped buying calls after a correction (orange profile on the right): {n_stop:,} of the {len(sel_all):,} starts had at least one call expire worthless and not replaced, {n_all:,} of them lost every call. "
           "Dashed lines: the market bottoms (table below).")

# ---------------- NAV on the end day by start date: Keep, Rotate and the net on the same start, one point per selected start (gaps where starts are not selected)
st.subheader(f"NAV on {end_day:%d %b %Y} by start date")
end_a = (rs["NAV A end"] / M).where(rs.index.isin(sel_all))
end_b = (rs["NAV B end"] / M).where(rs.index.isin(sel_all))
net = end_b - end_a
in_win = (rs.index >= x_lo) & (rs.index <= x_hi) & rs.index.isin(sel_all)   # the zoom slider: the same window of start dates, the axes following the values inside it
hov = "start %{x|%d %b %Y}<br>%{fullData.name}: %{y:,.0f} m<extra></extra>"


def mark_lowest(fig_: go.Figure, ya: pd.Series, yb: pd.Series, name_a: str, name_b: str, colr: str, dy: int) -> None:
    """The lowest value of ``ya`` inside the zoom window, marked on the top panel and labelled with ``yb`` on the same start."""
    w_ = ya[in_win].dropna()
    if w_.empty:
        return
    i = w_.idxmin()
    fig_.add_trace(go.Scatter(x=[i], y=[w_[i]], mode="markers", name=f"lowest {name_a}", marker={"color": colr, "size": 10, "line": {"color": "white", "width": 1.5}}, showlegend=False,
                              hovertemplate=f"lowest {name_a}: %{{y:,.0f}} m ({i:%d %b %Y} start)<br>{name_b} there: {yb[i]:,.0f} m<extra></extra>"), row=1, col=1)
    other = BLUE if colr == RED else RED
    fig_.add_annotation(x=i, y=w_[i], text=f"<span style='color:{colr}'><b>{w_[i]:,.0f} m</b></span> · {i:%d %b %Y} · <span style='color:{other}'>{yb[i]:,.0f} m</span>", showarrow=True,
                        arrowhead=2, arrowcolor=colr, ax=0, ay=dy, font={"size": 11, "color": "#333333"}, bgcolor="rgba(255, 255, 255, 0.9)", bordercolor=GREY, borderwidth=1, row=1, col=1)


fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
fig.add_trace(go.Scatter(x=rs.index, y=end_a, name=A, line={"color": RED, "width": 1.4}, connectgaps=False, hovertemplate=hov), row=1, col=1)
fig.add_trace(go.Scatter(x=rs.index, y=end_b, name=B, line={"color": BLUE, "width": 1.4}, connectgaps=False, hovertemplate=hov), row=1, col=1)
fig.add_trace(go.Scatter(x=rs.index, y=net.clip(lower=0.0), name="rotation ahead", fill="tozeroy", mode="none", fillcolor="rgba(11, 42, 111, 0.35)", connectgaps=False, hoverinfo="skip"), row=2, col=1)
fig.add_trace(go.Scatter(x=rs.index, y=net.clip(upper=0.0), name="rotation behind", fill="tozeroy", mode="none", fillcolor="rgba(192, 0, 0, 0.35)", connectgaps=False, hoverinfo="skip"), row=2, col=1)
fig.add_trace(go.Scatter(x=rs.index, y=net, name=f"net: {B} − {A}", line={"color": "#333333", "width": 1.2}, connectgaps=False, hovertemplate=hov), row=2, col=1)


def mark_corrections(fig_: go.Figure) -> None:
    """Every market peak (dotted) and bottom (dashed) up to the end day, on both panels, whatever the start-date range."""
    for r_ in corr[corr["trough"] <= end_day].itertuples():   # a start at a peak is the worst case for both, a start at a bottom the best
        for day, dash, what in ((r_.peak, "dot", "peak"), (r_.trough, "dash", "bottom")):
            fig_.add_vline(x=day.timestamp() * 1000, line={"color": GREY, "width": 1, "dash": dash}, row="all", col=1)
            fig_.add_annotation(x=day, y=1, yref="y domain", text=f"{what} {day:%b %Y}", showarrow=False, font={"size": 10, "color": GREY}, xanchor="left", yanchor="top", row=1, col=1)


mark_corrections(fig)
mark_lowest(fig, end_a, end_b, A, B, RED, -45)
mark_lowest(fig, end_b, end_a, B, A, BLUE, -85)
nav_win = pd.concat([end_a[in_win], end_b[in_win]])
net_win = net[in_win]
fig.update_yaxes(title_text="NAV (USD m)", rangemode="tozero", range=[0, float(nav_win.max()) * 1.05] if nav_win.notna().any() else None, row=1, col=1)
fig.update_yaxes(title_text="net (USD m)", zeroline=True, zerolinecolor=GREY,
                 range=[min(float(net_win.min()), 0.0) * 1.1 - 1.0, max(float(net_win.max()), 0.0) * 1.1 + 1.0] if net_win.notna().any() else None, row=2, col=1)
x_win = [max(x_lo, pd.Timestamp(rs.index[0]) - pd.DateOffset(days=30)), x_hi + pd.DateOffset(days=30)]   # to the end day, so every peak and bottom up to it is in view
fig.update_xaxes(range=x_win, row=1, col=1)
fig.update_xaxes(title_text="Start date", range=x_win, row=2, col=1)
fig.update_layout(height=640, title={"text": f"On {end_day:%d %b %Y}: the NAV of each portfolio and the net, by start date — {len(sel_all):,} starts", "x": 0.02, "font": {"size": 15}},
                  **{**LAYOUT, "legend": {"orientation": "h", "y": -0.12, "x": 0, "yanchor": "top"}, "margin": {"l": 40, "r": 20, "t": 30, "b": 40}})
st.plotly_chart(fig, width="stretch")
st.caption(f"Top: the NAV on {end_day:%d %b %Y} of the two portfolios put on at each start date (x axis), one point per selected start, both leaving from {(s.equity - s.loan) / M:,.0f} m on that day. "
           "Earlier starts have been held longer, so the level falls along the axis; read the two lines against each other. Bottom: the net, Rotate − Keep on the same start, shaded blue where the rotation "
           "ends ahead and red where it ends behind. Dotted lines: the market peaks (a start there is bought at the top); dashed lines: the bottoms (a start there is bought at the low). "
           "Gaps: starts outside the selected years. The zoom slider above sets the window of start dates; the two labels mark the lowest NAV of each portfolio inside it, "
           "with the other portfolio's NAV on that same start. The table below gives the distribution of the same values as annualised returns.")

# ---------------- dry powder on the end day by start date: Keep's room before a margin call and its LTV, Rotate's dry powder, the net on the same start
st.subheader(f"Dry powder on {end_day:%d %b %Y} by start date")
dp_a = (rs["A: headroom end"] / M).where(rs.index.isin(sel_all))
dp_b = (rs["B: dry powder end"] / M).where(rs.index.isin(sel_all))
ltv_pct = (rs["end: loan A"] / (s.lv_equity * rs["end: equity A"]) * 100.0).where(rs.index.isin(sel_all))   # Keep's LTV = loan ÷ lending value, %; the margin call is at 100
net_dp = dp_b - dp_a
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05, specs=[[{"secondary_y": True}], [{}]])
fig.add_trace(go.Scatter(x=rs.index, y=dp_a, name=f"{A}: room before a margin call", line={"color": RED, "width": 1.4}, connectgaps=False, hovertemplate=hov), row=1, col=1)
fig.add_trace(go.Scatter(x=rs.index, y=dp_b, name=f"{B}: dry powder", line={"color": BLUE, "width": 1.4}, connectgaps=False, hovertemplate=hov), row=1, col=1)
fig.add_trace(go.Scatter(x=rs.index, y=ltv_pct, name=f"{A}: LTV (right axis; yellow far from a margin call, orange close to it)", mode="markers",
                         marker={"size": 4, "color": ltv_pct, "colorscale": [[0.0, LIGHT_YELLOW], [1.0, ORANGE]], "cmin": 0.0, "cmax": 100.0, "showscale": True,
                                 "colorbar": {"title": {"text": "LTV"}, "ticksuffix": " %", "thickness": 10, "len": 0.5, "y": 0.78}},
                         hovertemplate="start %{x|%d %b %Y}<br>Keep the loan: LTV %{y:.0f} %<extra></extra>"), row=1, col=1, secondary_y=True)
fig.add_hline(y=100.0, line={"color": ORANGE, "width": 1, "dash": "dot"}, annotation={"text": "margin call (LTV 100 %)", "font": {"size": 10, "color": ORANGE}}, annotation_position="top left",
              row=1, col=1, secondary_y=True)
fig.add_trace(go.Scatter(x=rs.index, y=net_dp.clip(lower=0.0), name="rotation has more", fill="tozeroy", mode="none", fillcolor="rgba(11, 42, 111, 0.35)", connectgaps=False, hoverinfo="skip"), row=2, col=1)
fig.add_trace(go.Scatter(x=rs.index, y=net_dp.clip(upper=0.0), name="rotation has less", fill="tozeroy", mode="none", fillcolor="rgba(192, 0, 0, 0.35)", connectgaps=False, hoverinfo="skip"), row=2, col=1)
fig.add_trace(go.Scatter(x=rs.index, y=net_dp, name=f"net dry powder: {B} − {A}", line={"color": "#333333", "width": 1.2}, connectgaps=False, hovertemplate=hov), row=2, col=1)
mark_corrections(fig)
mark_lowest(fig, dp_a, dp_b, A, B, RED, -45)
mark_lowest(fig, dp_b, dp_a, B, A, BLUE, -85)
dp_win = pd.concat([dp_a[in_win], dp_b[in_win]])
net_dp_win = net_dp[in_win]
fig.update_yaxes(title_text="dry powder (USD m)", rangemode="tozero", range=[0, float(dp_win.max()) * 1.05] if dp_win.notna().any() else None, row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="LTV of the loan portfolio", range=[0, 110], ticksuffix=" %", showgrid=False, row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="net (USD m)", zeroline=True, zerolinecolor=GREY,
                 range=[min(float(net_dp_win.min()), 0.0) * 1.1 - 1.0, max(float(net_dp_win.max()), 0.0) * 1.1 + 1.0] if net_dp_win.notna().any() else None, row=2, col=1)
fig.update_xaxes(range=x_win, row=1, col=1)
fig.update_xaxes(title_text="Start date", range=x_win, row=2, col=1)
fig.update_layout(height=640, title={"text": f"On {end_day:%d %b %Y}: the dry powder of each portfolio and the net, by start date — {len(sel_all):,} starts", "x": 0.02, "font": {"size": 15}},
                  **{**LAYOUT, "legend": {"orientation": "h", "y": -0.12, "x": 0, "yanchor": "top"}, "margin": {"l": 40, "r": 20, "t": 30, "b": 40}})
st.plotly_chart(fig, width="stretch")
st.caption(f"Top: on {end_day:%d %b %Y}, for each start date, the room before a margin call of the loan portfolio ({s.lv_equity:.0%} × SPX − loan, red) and the dry powder of the rotation "
           f"({s.lv_equity:.0%} × SPX + {s.lv_calls:.0%} × calls + {s.lv_cash:.0%} × T-bills − loan, blue), both in USD m; the dots are the loan portfolio's LTV = loan ÷ lending value on the right axis, "
           "yellow far from a margin call and orange close to it (the call comes at 100 %, the dotted line; the further SPX fall to it is 1 − LTV). "
           "Bottom: the net, Rotate − Keep on the same start, shaded blue where the rotation has more dry powder and red where it has less. "
           "Dotted vertical lines: the market peaks; dashed: the bottoms. Same selection and zoom window as the NAV chart above; the labels mark each portfolio's lowest dry powder inside the window with the other's on that start.")

# ---------------- the rotation's exposure relative to Keep's, over time, across the selected starts
st.subheader("Exposure of the rotation relative to Keep's, over time")
ratio_w = (paths["exposure_B"][sel_all] / paths["exposure_A"][sel_all]) if len(sel_all) else pd.DataFrame()
if not ratio_w.empty:
    alive = ratio_w.notna().sum(axis=1)
    ratio_w = ratio_w[alive > 0]
    med, lo5, hi95 = ratio_w.median(axis=1), ratio_w.quantile(0.05, axis=1), ratio_w.quantile(0.95, axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hi95.index, y=hi95, name="95th percentile", line={"color": BLUE, "width": 0.8, "dash": "dot"}, hovertemplate="%{x|%d %b %Y}: %{y:.2f}<extra>95th percentile</extra>"))
    fig.add_trace(go.Scatter(x=lo5.index, y=lo5, name="5th percentile", line={"color": BLUE, "width": 0.8, "dash": "dot"}, fill="tonexty", fillcolor="rgba(11, 42, 111, 0.15)", hovertemplate="%{x|%d %b %Y}: %{y:.2f}<extra>5th percentile</extra>"))
    fig.add_trace(go.Scatter(x=med.index, y=med, name="median across the starts alive", line={"color": BLUE, "width": 1.8}, hovertemplate="%{x|%d %b %Y}: %{y:.2f}<extra>median</extra>"))
    fig.add_hline(y=1.0, line={"color": RED, "width": 1.2}, annotation={"text": "Keep's exposure", "font": {"size": 10, "color": RED}}, annotation_position="top left")
    if s.roll == "target" and s.rebalance != "none":
        for edge in (1.0 + s.band, 1.0 - s.band):
            fig.add_hline(y=edge, line={"color": ORANGE, "width": 1, "dash": "dash"}, annotation={"text": f"band edge {edge:.2f}", "font": {"size": 10, "color": ORANGE}}, annotation_position="top left")
    fig.update_yaxes(title_text="rotation's exposure ÷ Keep's", rangemode="tozero")
    fig.update_xaxes(title_text="Date")
    fig.update_layout(height=380, title={"text": f"Live exposure of the rotation ÷ Keep's SPX value, month by month across the {len(sel_all):,} selected starts", "x": 0.02, "font": {"size": 15}},
                      **{**LAYOUT, "legend": {"orientation": "h", "y": -0.2, "x": 0, "yanchor": "top"}})
    st.plotly_chart(fig, width="stretch")
    st.caption("The rotation's live exposure (its SPX plus the dollar delta of its calls, model delta of the day) divided by Keep's SPX value, on each month-end, across the selected starts alive on that date: "
               "median and 5th–95th percentile band. It is 1 on every start day and, under *Keep's exposure*, 1 again after every roll and inside the band on every funded check day "
               "(dashed lines); between checks it drifts with the calls' delta — up in a rally, down in a fall. Under the two earlier rules it shows how far the exposure runs from Keep's.")

# ---------------- statistics of the final NAV over the selected starts (every start of the selected years, not subsampled)
nav0 = (s.equity - s.loan) / M
fin = pd.DataFrame({A: rs.loc[sel_all, "NAV A end"] / M, B: rs.loc[sel_all, "NAV B end"] / M})
D = f"Δ {B} − {A}"
LABEL = " "   # the first column: section names and the label of each row
PCT = (("5th percentile", 0.05), ("25th percentile", 0.25), ("median", 0.5), ("75th percentile", 0.75), ("95th percentile", 0.95))
SECTION_STYLE = "background-color: #dfe6f0; color: #1a1a1a; font-weight: 600"
st.markdown("""<style>
div[data-testid="stTable"] table { font-size: 0.95rem; table-layout: fixed; width: 100%; }
div[data-testid="stTable"] th, div[data-testid="stTable"] td { white-space: normal !important; overflow-wrap: anywhere; padding: 0.35rem 0.6rem; vertical-align: top; line-height: 1.3; }
div[data-testid="stTable"] th:nth-child(1), div[data-testid="stTable"] td:nth-child(1) { width: 23%; font-weight: 500; }
div[data-testid="stTable"] th:nth-child(2), div[data-testid="stTable"] td:nth-child(2) { width: 32%; }
div[data-testid="stTable"] th:nth-child(3), div[data-testid="stTable"] td:nth-child(3) { width: 32%; }
div[data-testid="stTable"] th:nth-child(4), div[data-testid="stTable"] td:nth-child(4) { width: 13%; text-align: right; }
</style>""", unsafe_allow_html=True)


def pm(x: float) -> str:
    """Signed USD m, '-0' avoided."""
    return f"{round(float(x)) + 0.0:+,.0f} m"


def block(title: str, a: pd.Series, b: pd.Series, suffix: Callable[[float], str] = lambda _v: "", mean: bool = False,
          cell: Callable[[float], str] = lambda x: f"{x:,.0f} m", delta: Callable[[float], str] = pm) -> list[list[str]]:
    """Rows [label, Keep, Rotate, Δ] of one section: a section row with the title, then lowest, the five percentiles, highest (and the mean) of each
    portfolio's own distribution over the starts (a, b one value per start), with Δ = the Rotate cell − the Keep cell of the row; last, the
    count of starts on which the rotation is ahead (b > a on the same start)."""
    stats = [("lowest", a.min(), b.min()), *((n, a.quantile(q), b.quantile(q)) for n, q in PCT), ("highest", a.max(), b.max())]
    if mean:
        stats.append(("mean", a.mean(), b.mean()))
    rows = [[title, "", "", ""]]
    for name, fa, fb in stats:
        sfx = suffix if name in ("lowest", "highest") else (lambda _v: "")
        rows.append([name, f"{cell(float(fa))}{sfx(float(fa))}", f"{cell(float(fb))}{sfx(float(fb))}", delta(float(fb - fa))])
    rows.append(["rotation ahead on", "", "", f"{int((b > a).sum()):,} of {len(a):,} starts"])
    return rows


def show(rows: list[list[str]]) -> pd.DataFrame:
    """The rows as one table (text wraps, no inner scrolling), the section rows (title only) shaded."""
    df = pd.DataFrame(rows, columns=[LABEL, A, B, D])
    heads = {i for i, r in enumerate(rows) if r[1] == "" and r[2] == "" and r[3] == ""}
    sty = df.style.apply(lambda col: [SECTION_STYLE if i in heads else "" for i in range(len(col))], axis=0)
    st.table(sty, hide_index=True, border="horizontal")
    return df


sel_year = sel_all[(rs.loc[sel_all, "years"] >= 1.0).to_numpy()]   # annualising a few weeks of noise is meaningless: the annualised figures use the starts held at least a year
n_young = len(sel_all) - len(sel_year)
st.subheader(f"On {end_day:%d %b %Y}: the {len(sel_all):,} selected starts")
ann_sel = ann.loc[sel_year]
show(block("Annualised return to the end day" + (f" ({len(sel_year):,} starts held at least a year)" if n_young else ""), ann_sel[A], ann_sel[B], cell=lambda x: f"{x:+.1%}", delta=lambda d: f"{d * 100:+.1f} pp"))
ratio = (fin[B] / fin[A] - 1.0).loc[sel_year]
young_note = f"; the {n_young:,} starts held less than a year are left out of this block (a few weeks of noise annualised would swamp the lowest and highest rows). " if n_young else ". "
st.markdown(f"- Annualised return = (NAV on the end day ÷ {nav0:,.0f} m)^(1 / years held) − 1, comparable across starts whatever their holding period{young_note}"
            f"Keep and Rotate: each portfolio's own distribution over the starts; Δ = Rotate − Keep on each row (lowest vs lowest, median vs median — usually different start dates); "
            f"'rotation ahead on' compares each start with itself.\n"
            f"- Same start, same end: the rotation ends **ahead on {float((ratio > 0).mean()):.0%}** of these starts; its final NAV relative to keeping the loan: median **{ratio.median():+.1%}**, "
            f"5th–95th percentile {ratio.quantile(0.05):+.1%} to {ratio.quantile(0.95):+.1%}, worst {ratio.min():+.1%} ({ratio.idxmin():%d %b %Y} start), best {ratio.max():+.1%} ({ratio.idxmax():%d %b %Y} start).\n"
            f"- Consecutive starts overlap almost entirely: the percentiles are the range of history, not probabilities.")

# ---------------- the market bottoms: one tab per bottom
st.subheader("At each market bottom" if not at_bottom else f"At {end_label.split(' (')[0]}")
corr_shown = corr if not at_bottom else corr[corr["trough"] == end_day]   # held to today: every bottom; held to a bottom: that one only
shown = [(lab, r, [c for c in sel_all if c in paths["nav_A"].columns and pd.notna(paths["nav_A"].loc[r.trough, c])])
         for lab, r in zip(corr_shown.index, corr_shown.itertuples(), strict=True) if r.trough <= end_day and r.trough in paths["nav_A"].index]
shown = [(lab, r, alive) for lab, r, alive in shown if alive]
alive_at = {lab: alive for lab, _, alive in shown}


lowest = {k: [paths[k].loc[corr_shown.loc[lab, "trough"], alive_at[lab]].min() / M if lab in alive_at else np.nan for lab in corr_shown.index] for k in ("nav_A", "nav_B")}
bottoms = pd.DataFrame({
    "Correction": corr_shown.index, "Previous peak": [d.date() for d in corr_shown["peak"]], "SPX at the peak": corr_shown["peak level"].to_numpy(),
    "Bottom": [d.date() for d in corr_shown["trough"]], "SPX at the bottom": corr_shown["trough level"].to_numpy(), "Fall": corr_shown["drawdown"].to_numpy(),
    "Back to the peak": [d.date() if pd.notna(d) else "not yet" for d in corr_shown["recovered"]],
    f"Lowest NAV, {A}": lowest["nav_A"], f"Lowest NAV, {B}": lowest["nav_B"],
})
st.dataframe(bottoms.style.format("{:,.0f}", subset=["SPX at the peak", "SPX at the bottom"]).format("{:+.0%}", subset=["Fall"]).format("{:,.0f} m", subset=[f"Lowest NAV, {A}", f"Lowest NAV, {B}"], na_rep="—"),
             width="stretch", hide_index=True, height=table_height(len(bottoms)))
st.caption(("Falls of 20 % or more in the SPX price index since 1997" if not at_bottom else "The correction chosen in *Held until* (held to today, the table lists all four)")
           + ": the day of the previous peak, the lowest close, and the day the index regained the peak. The bottoms are the dashed lines on the charts above. "
           "Lowest NAV: on the bottom day, the lowest NAV of each portfolio over the selected starts running that day (usually different start dates; both portfolios reach their low of the correction on that day); "
           "— when no selected start was running. "
           "One tab per bottom below: the worst trajectory in detail (the lowest NAV that day in either portfolio, both strategies read on that same start) and the count of starts on which "
           "the rotation has more dry powder that day.")
spx_px = lvs._load(str(lvs.DAILY_FILE)).set_index("date")["spx_px_last"]


def vs_peak(start: pd.Timestamp, r: object) -> str:
    """'15 days before the peak, SPX 1,521 (−0.4 % vs the peak)'."""
    peak, level = pd.Timestamp(r.peak), float(r._5)   # itertuples: 'peak level' is the 5th field
    days = (start - peak).days
    when = f"{-days} days before the peak" if days < 0 else (f"{days} days after the peak" if days > 0 else "on the peak day")
    px = float(spx_px.loc[start])
    return f"{when}, SPX {px:,.0f} ({px / level - 1:+.1%} vs the peak)"


def trajectory(w: pd.Timestamp, title: str, same_start: str, at_: dict[str, pd.Series], r: object) -> list[list[str]]:
    """One start read on both strategies on the bottom day (``at_``: the path columns on that day, over the starts): a section row, then its
    balance sheet and costs since the start, Δ on every row."""
    v = lambda k: float(at_[k][w]) / M  # noqa: E731
    lv_a, lv_b = s.lv_equity * v("E_A"), s.lv_equity * v("E_B") + s.lv_calls * v("call_val") + s.lv_cash * v("cash_B")
    n_bought = int(at_["n_bought"][w])   # the build steps done by that day plus every replacement at expiry — fewer than the weekly steps when the bottom came before the rotation was complete
    n_alive = f"{int(at_['n_calls'][w])} call{'s' if at_['n_calls'][w] != 1 else ''} alive of the {n_bought} bought" + (f" so far: {n_bought} of the {s.build} weekly steps done" if n_bought < s.build else "")
    return [
        [title, "", "", ""],
        ["started on", f"{w:%d %b %Y}, {vs_peak(w, r)}", same_start, ""],
        ["NAV", f"{v('nav_A'):,.0f} m ({v('nav_A') / nav0 - 1:+.0%} from the start)", f"{v('nav_B'):,.0f} m ({v('nav_B') / nav0 - 1:+.0%} from the start)", pm(v("nav_B") - v("nav_A"))],
        ["SPX held, market value", f"{v('E_A'):,.0f} m", f"{v('E_B'):,.0f} m", pm(v("E_B") - v("E_A"))],
        ["T-bills (cash)", "—", f"{v('cash_B'):,.0f} m", pm(v("cash_B"))],
        ["calls held, market value", "—", f"{v('call_val'):,.0f} m ({n_alive}, on a notional of {v('call_notional'):,.0f} m of index)", pm(v("call_val"))],
        ["loan", f"{v('loan'):,.0f} m", f"{v('loan_B'):,.0f} m" + (" (rotation still under way)" if at_["loan_B"][w] > 0.5e6 else ""), pm(v("loan_B") - v("loan"))],
        ["lending value of the holdings", f"{lv_a:,.0f} m ({s.lv_equity:.0%} of the SPX)", f"{lv_b:,.0f} m ({s.lv_equity:.0%} of the SPX + {s.lv_calls:.0%} of the calls + {s.lv_cash:.0%} of the T-bills)", pm(lv_b - lv_a)],
        ["dry powder = lending value − loan", f"{lv_a:,.0f} − {v('loan'):,.0f} = {v('headroom_A'):,.0f} m (LTV {at_['ltv_A'][w]:.0%}: a further {max(1 - at_['ltv_A'][w], 0):.0%} SPX fall to a margin call)",
         f"{lv_b:,.0f} − {v('loan_B'):,.0f} = {v('dry_powder_B'):,.0f} m", pm(v("dry_powder_B") - v("headroom_A"))],
        ["interest cumulated", f"{v('interest_cum_A'):,.0f} m", f"{v('interest_cum_B'):,.0f} m (during the build)", pm(v("interest_cum_B") - v("interest_cum_A"))],
    ]


if not shown:
    st.markdown("No market bottom falls inside the selected runs.")
for (lab, r, alive), tab in zip(shown, st.tabs([f"{lab} bottom — {r.trough:%d %b %Y}" for lab, r, _ in shown]) if shown else [], strict=True):
  with tab:
    na, nb = paths["nav_A"].loc[r.trough, alive], paths["nav_B"].loc[r.trough, alive]
    wa, wb = na.idxmin(), nb.idxmin()
    st.markdown(f"**{lab} bottom — {r.trough:%d %b %Y}**, SPX {corr.loc[lab, 'trough level']:,.0f}, {r.drawdown:+.0%} from the {r.peak:%d %b %Y} peak; {len(alive):,} starts running that day")
    at_ = {k: paths[k].loc[r.trough] for k in ("nav_A", "nav_B", "E_A", "loan", "E_B", "call_val", "cash_B", "loan_B", "n_calls", "n_bought", "call_notional", "headroom_A", "dry_powder_B", "ltv_A", "interest_cum_A", "interest_cum_B")}
    room_all, dp_all = at_["headroom_A"][alive] / M, at_["dry_powder_B"][alive] / M
    # the lowest NAV that day in either portfolio (usually the last start before the peak), both strategies read on that one start so every Δ is like for like
    w = wa if na.min() <= nb.min() else wb
    who = A if na.min() <= nb.min() else B
    rows = trajectory(w, "Worst trajectory: the lowest NAV that day", f"same start (the lowest NAV that day: {who})", at_, r)
    rows.append(["rotation has more dry powder on", "", "", f"{int((dp_all > room_all).sum()):,} of {len(alive):,} starts running that day"])
    show(rows)
    st.caption(f"Market values on the bottom day, from the daily simulation of each trajectory. LTV = loan ÷ ({s.lv_equity:.0%} × SPX); the margin call comes when the SPX falls by a further 1 − LTV. "
               f"Dry powder = the lending value of what is held − loan: {s.lv_equity:.0%} on the SPX, {s.lv_calls:.0%} on the calls (at market value, not notional), {s.lv_cash:.0%} on the T-bills (sidebar, Advanced). "
               "Worst trajectory: the start with the lowest NAV that day in either portfolio, both strategies read on that same start so every Δ is like for like — its balance sheet and the interest "
               "capitalised on each loan since the start (Rotate: only while the build was repaying it). The rotation's T-bills are the payoffs of the calls that expired in the money, less the premiums of "
               "their replacements. Dividends on the SPX are reinvested in the SPX the same day, net of withholding, in both portfolios: they sit inside the SPX market value and never appear as cash; "
               "the interest on the loan is added to the loan, not paid from them. The last row counts the starts (of all those running that day) where the rotation has more dry powder.")

low = pd.DataFrame({A: rs.loc[sel_all, "A: min headroom"] / M, B: rs.loc[sel_all, "B: min dry powder"] / M})
st.markdown(f"- Margin calls keeping the loan: **{int(rs.loc[sel_all, 'A: margin call'].sum()):,}** of the {len(sel_all):,} selected starts; the closest was {low[A].min():,.0f} m of room "
            f"({rs.loc[sel_all, 'A: min headroom'].idxmin():%d %b %Y} start, on {rs.loc[rs.loc[sel_all, 'A: min headroom'].idxmin(), 'A: min headroom date']:%d %b %Y}). "
            f"The rotation carries a loan only during the {s.build}-week build, so a margin call is impossible once it is complete.\n"
            f"- Over the whole holding period, at its lowest point the rotation has more dry powder than keeping the loan on **{float((low[B] > low[A]).mean()):.0%}** of the selected starts (median difference {(low[B] - low[A]).median():+,.0f} m).")

# ---------------- reading
st.markdown(f"- Of the {len(sel_all):,} selected starts, the rotated portfolio lost **all** its calls on **{lost_all.loc[sel_all].mean():.0%}** ({int(lost_all.loc[sel_all].sum()):,}) and **some** of them on "
            f"{some_lapsed.loc[sel_all].mean():.0%}; on the other {1 - some_lapsed.loc[sel_all].mean():.0%} every call expired in the money and was replaced.\n"
            + f"- Annualised over each start's own holding period, the median return to {end_day:%d %b %Y} over the selected starts is {ann_sel[A].median():+.1%} ({A}) vs {ann_sel[B].median():+.1%} ({B}).")
st.caption("Consecutive start dates overlap almost entirely, so the charts show the range of historical outcomes, not independent draws. Calls are priced and marked with Black–Scholes on SPXFP (q = r) at the implied vol of the day, "
           "extrapolated from the 24-month Bloomberg series; the delta is the model delta of the ATM call. No bid/ask, no early unwind. Dividends reinvested net of withholding; cash earns the 3-month T-bill.")
