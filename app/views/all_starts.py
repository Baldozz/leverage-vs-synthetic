"""Any start date since 1997 — the same two portfolios put on at every start of the grid and held to today."""

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
    all_starts_cached,
    corrections_cached,
    setup,
    start_grid,
    table_height,
)

s = setup()
freq, first_day, last_start, last_strike, last_day = start_grid(s)
corr = corrections_cached(0.20, s.wht, "price")

st.title("Any start date since 1997")
ends = {f"today ({last_day:%d %b %Y})": last_day, **{f"the {lab} bottom ({tr_:%d %b %Y}, SPX {lvl:,.0f})": tr_ for lab, tr_, lvl in zip(corr.index, corr["trough"], corr["trough level"], strict=True)}}
c_years, c_end = st.columns(2)   # start years on the left, the final day on the right
end_label = c_end.selectbox("Held until", list(ends), key="r_end", help="Every start runs to this day: today, or the bottom of a market correction to see how the portfolios looked at the worst moment.")
end_day = ends[end_label]
at_bottom = end_day != last_day
if at_bottom:   # every start that had completed its rotation at least a month before the bottom
    last_start = min(last_start, end_day - pd.Timedelta(7 * (s.build - 1) + 30, unit="D"))
years_all = list(range(first_day.year, last_start.year + 1))
picked = c_years.multiselect("Start years", years_all, default=[], key="r_years", placeholder="All years — or pick one or more", help="Empty = every start year.")
sel_years = picked or years_all
st.caption(f"The same two portfolios put on at every {'trading day' if freq == 'D' else 'week'} from {first_day:%d %b %Y} to {last_start:%d %b %Y} and held to {end_day:%d %b %Y}. "
           + ("The last start is the one whose rotation was complete a month before the bottom. " if at_bottom else f"The last start is the one whose last weekly tranche, struck on {last_strike:%d %b %Y}, still expires inside the data. ")
           + f"The rotation: {s.tenor:g}-year ATM calls on SPXFP, {s.build} weekly step{'s' if s.build > 1 else ''}, the loan gone after the build; a call that expires in the money is replaced, "
           + ("a call that expires worthless lapses." if not s.replace_worthless else "a worthless call is replaced too."))
if last_start <= first_day:
    st.warning("No start date fits before that day: shorten the rotation.")
    st.stop()

rs, paths = all_starts_cached(str(first_day.date()), str(last_start.date()), freq, s, str(end_day.date()))
if rs.empty:
    st.warning("No start date fits: shorten the rotation or the tenor.")
    st.stop()
years = rs["years"]
ann = pd.DataFrame({A: (1.0 + rs["A return"]) ** (1.0 / years) - 1.0, B: (1.0 + rs["B return"]) ** (1.0 / years) - 1.0}, index=rs.index)
lost_all = rs["B: call notional end"] <= 0.0                       # no call left today: the whole option part lapsed
some_lapsed = rs["B: lapsed"] > 0
st.markdown(f"**{len(rs):,} start dates, {rs.index[0]:%d %b %Y} → {rs.index[-1]:%d %b %Y}, each held to {end_day:%d %b %Y} ({years.min():.1f} to {years.max():.1f} years).**")
st.download_button("Download every start date as CSV", lvs.starts_table(rs).to_csv(index=False).encode(), f"any_start_date_since_1997_to_{end_day:%Y-%m-%d}_{'daily' if freq == 'D' else 'weekly'}.csv", "text/csv")

# ---------------- every start date's path, Monte Carlo style, with the distribution of the final NAV on the right
st.subheader(f"Every start date's path to {'today' if not at_bottom else end_label.split(' (')[0]}")
step = 5 if freq == "D" else 1   # with daily starts draw every fifth (≈ weekly); the distributions further down still use every start
cols = rs.index[::step]
cols = cols[[d.year in sel_years for d in cols]]
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


fin_all = pd.concat([rs.loc[cols, "NAV A end"], rs.loc[cols, "NAV B end"]]) / M
bin_w = max(float(np.ceil((fin_all.max() - fin_all.min()) / 40 / 50) * 50), 50.0)   # ≈ 40 bins, rounded to 50 m
edges = np.arange(np.floor(fin_all.min() / bin_w) * bin_w, np.ceil(fin_all.max() / bin_w) * bin_w + bin_w, bin_w)
years_shown = sorted({d.year for d in cols})
y0, y1 = years_all[0], years_all[-1]
LIGHT_RED, LIGHT_BLUE, LIGHT_YELLOW = "#f6c6c6", "#c3cfe8", "#fbe8b0"
zoom = st.slider("Zoom: years shown on the fans (the NAV axis follows the highest path inside the window; drag a rectangle on a chart to zoom further, double-click to reset)",
                 first_day.year, end_day.year, (first_day.year, end_day.year), key="r_zoom")
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
    fig.add_trace(profile(rs.loc[cols, end_col] / M, edges, base, title), row=1, col=2)
    if title == B:
        stopped_all = cols[some_lapsed.loc[cols].to_numpy()]
        if len(stopped_all):
            fig.add_trace(profile(rs.loc[stopped_all, end_col] / M, edges, ORANGE, "stopped"), row=1, col=2)
    fig.update_yaxes(title_text="NAV (USD m)", rangemode="tozero", range=[0, y_top] if y_top else None, row=1, col=1)
    fig.update_xaxes(title_text="Date", range=[x_lo, x_hi], row=1, col=1)
    fig.update_xaxes(title_text=f"share of starts, NAV on {end_day:%d %b %Y}", tickformat=".0%", row=1, col=2)
    fig.update_layout(height=500, title={"text": f"{title} — {len(cols):,} start dates", "x": 0.02, "font": {"size": 15}}, **{**LAYOUT, "margin": {"l": 40, "r": 20, "t": 30, "b": 90}})
    st.plotly_chart(fig, width="stretch")
n_stop = int(some_lapsed.loc[cols].sum())
n_all = int(lost_all.loc[cols].sum())
st.caption(f"One line per start date ({len(cols):,} lines{', every fifth trading-day start' if step > 1 else ''}), month-end values, each leaving from {(s.equity - s.loan) / M:,.0f} m on its own start day and running to {end_day:%d %b %Y}; "
           f"the profile on the right is the distribution of the final NAV over the same starts, on the same scale. One colour per start year, light for the oldest and strong for the newest, newer years drawn in front. "
           f"Yellow to orange, same light-to-strong logic and drawn in front, are the rotations that stopped buying calls after a correction (orange profile on the right): {n_stop:,} of the {len(cols):,} starts had at least one call expire worthless and not replaced, {n_all:,} of them lost every call. "
           "Dashed lines: the market bottoms (table below).")

# ---------------- statistics of the final NAV over the selected starts (every start of the selected years, not subsampled)
sel_all = rs.index[[d.year in sel_years for d in rs.index]]
nav0 = (s.equity - s.loan) / M
fin = pd.DataFrame({A: rs.loc[sel_all, "NAV A end"] / M, B: rs.loc[sel_all, "NAV B end"] / M})
D = f"Δ {B} − {A}"
LABEL = " "   # the first column: section names and the statistic of each row
PCT = (("5th percentile", 0.05), ("25th percentile", 0.25), ("median", 0.5), ("75th percentile", 0.75), ("95th percentile", 0.95))
SECTION_STYLE = "background-color: #dfe6f0; color: #1a1a1a; font-weight: 600"


def block(title: str, a: pd.Series, b: pd.Series, suffix: Callable[[float], str] = lambda _v: "", mean: bool = False) -> list[list[str]]:
    """Rows [label, Keep, Rotate, Δ] of one section: a section row with the title, then lowest, the five percentiles, highest (and the mean) of each
    portfolio's own distribution over the starts (a, b in USD m, one value per start), with Δ = the Rotate cell − the Keep cell of the row; last, the
    count of starts on which the rotation is ahead (b > a on the same start)."""
    stats = [("lowest", a.min(), b.min()), *((n, a.quantile(q), b.quantile(q)) for n, q in PCT), ("highest", a.max(), b.max())]
    if mean:
        stats.append(("mean", a.mean(), b.mean()))
    rows = [[title, "", "", ""]]
    for name, fa, fb in stats:
        sfx = suffix if name in ("lowest", "highest") else (lambda _v: "")
        rows.append([name, f"{fa:,.0f} m{sfx(float(fa))}", f"{fb:,.0f} m{sfx(float(fb))}", f"{round(float(fb - fa)) + 0.0:+,.0f} m"])
    rows.append(["rotation ahead on", "", "", f"{int((b > a).sum()):,} of {len(a):,} starts"])
    return rows


def show(rows: list[list[str]]) -> pd.DataFrame:
    """The rows as one table, the section rows (title only) shaded."""
    df = pd.DataFrame(rows, columns=[LABEL, A, B, D])
    heads = {i for i, r in enumerate(rows) if r[1] == "" and r[2] == "" and r[3] == ""}
    sty = df.style.apply(lambda col: [SECTION_STYLE if i in heads else "" for i in range(len(col))], axis=0)
    st.dataframe(sty, width="stretch", hide_index=True, height=table_height(len(df)),
                 column_config={LABEL: st.column_config.Column(width="medium"), A: st.column_config.Column(width="large"), B: st.column_config.Column(width="large"), D: st.column_config.Column(width="small")})
    return df


st.subheader(f"Final NAV on {end_day:%d %b %Y}: statistics over the {len(sel_all):,} selected starts")
show(block("Final NAV", fin[A], fin[B], mean=True))
ratio = fin[B] / fin[A] - 1.0
st.markdown(f"- USD m, from {nav0:,.0f} m at the start. Worst start: {fin[A].idxmin():%d %b %Y} ({A}), {fin[B].idxmin():%d %b %Y} ({B}); best start: {fin[A].idxmax():%d %b %Y} ({A}), {fin[B].idxmax():%d %b %Y} ({B}).\n"
            f"- Same start, same end: the rotation ends **ahead on {float((ratio > 0).mean()):.0%}** of the selected starts; its final NAV relative to keeping the loan: median **{ratio.median():+.1%}**, "
            f"5th–95th percentile {ratio.quantile(0.05):+.1%} to {ratio.quantile(0.95):+.1%}, worst {ratio.min():+.1%} ({ratio.idxmin():%d %b %Y} start), best {ratio.max():+.1%} ({ratio.idxmax():%d %b %Y} start).\n"
            f"- Keep and Rotate: each portfolio's own distribution over the starts; Δ = Rotate − Keep on each row (e.g. lowest vs lowest, median vs median — the two cells "
            f"are usually different start dates). 'Rotation ahead on' compares each start with itself.")

# ---------------- the market bottoms and the ten worst trajectories
st.subheader("At each market bottom: NAV and dry powder of the starts running that day, and the worst trajectory in detail")
bottoms = pd.DataFrame({
    "Correction": corr.index, "Previous peak": [d.date() for d in corr["peak"]], "SPX at the peak": corr["peak level"].to_numpy(),
    "Bottom": [d.date() for d in corr["trough"]], "SPX at the bottom": corr["trough level"].to_numpy(), "Fall": corr["drawdown"].to_numpy(),
    "Back to the peak": [d.date() if pd.notna(d) else "not yet" for d in corr["recovered"]],
})
st.dataframe(bottoms.style.format("{:,.0f}", subset=["SPX at the peak", "SPX at the bottom"]).format("{:+.0%}", subset=["Fall"]), width="stretch", hide_index=True, height=table_height(len(bottoms)))
st.caption("Falls of 20 % or more in the SPX price index since 1997: the day of the previous peak, the lowest close, and the day the index regained the peak. The bottoms are the dashed lines on the charts above.")

st.markdown("**At each bottom** — the NAV of the selected starts running that day; the worst trajectory (the lowest NAV that day in either portfolio) with both strategies on that start, in detail, with what each had cost to get there; the dry powder of the starts running that day.")
spx_px = lvs._load(str(lvs.DAILY_FILE)).set_index("date")["spx_px_last"]


def vs_peak(start: pd.Timestamp, r: object) -> str:
    """'15 days before the peak, SPX 1,521 (−0.4 % vs the peak)'."""
    peak, level = pd.Timestamp(r.peak), float(r._5)   # itertuples: 'peak level' is the 5th field
    days = (start - peak).days
    when = f"{-days} days before the peak" if days < 0 else (f"{days} days after the peak" if days > 0 else "on the peak day")
    px = float(spx_px.loc[start])
    return f"{when}, SPX {px:,.0f} ({px / level - 1:+.1%} vs the peak)"


def pm(x: float) -> str:
    """Signed USD m, '-0' avoided."""
    return f"{round(float(x)) + 0.0:+,.0f} m"


n_wb = 0
for lab, r in zip(corr.index, corr.itertuples(), strict=True):
    if r.trough > end_day or r.trough not in paths["nav_A"].index:
        continue
    alive = [c for c in sel_all if c in paths["nav_A"].columns and pd.notna(paths["nav_A"].loc[r.trough, c])]
    if not alive:
        continue
    na, nb = paths["nav_A"].loc[r.trough, alive], paths["nav_B"].loc[r.trough, alive]
    wa, wb = na.idxmin(), nb.idxmin()
    st.markdown(f"**{lab} bottom — {r.trough:%d %b %Y}**, SPX {corr.loc[lab, 'trough level']:,.0f}, {r.drawdown:+.0%} from the {r.peak:%d %b %Y} peak; {len(alive):,} starts running that day")
    at_ = {k: paths[k].loc[r.trough] for k in ("E_A", "loan", "E_B", "call_val", "cash_B", "loan_B", "n_calls", "call_notional", "headroom_A", "dry_powder_B", "ltv_A", "interest_cum_A", "premiums_cum_B", "payoffs_cum_B")}
    v = lambda k, c, at_=at_: float(at_[k][c]) / M  # noqa: E731  (bound per bottom)
    w = wa if na.min() <= nb.min() else wb   # the lowest NAV that day in either portfolio; both strategies are then read on that one start, so every Δ is like for like
    who = A if na.min() <= nb.min() else B
    lv_a, lv_b = s.lv_equity * v("E_A", w), s.lv_equity * v("E_B", w) + s.lv_calls * v("call_val", w)
    room_all, dp_all = at_["headroom_A"][alive] / M, at_["dry_powder_B"][alive] / M
    n_alive = f"{int(at_['n_calls'][w])} call{'s' if at_['n_calls'][w] != 1 else ''} alive"
    net = v("premiums_cum_B", w) - v("payoffs_cum_B", w)
    worst = [
        ["Worst trajectory", "", "", ""],
        ["started on", f"{w:%d %b %Y}, {vs_peak(w, r)}", f"same start (the lowest NAV that day: {who})", ""],
        ["NAV", f"{na[w] / M:,.0f} m ({na[w] / M / nav0 - 1:+.0%} from the start)", f"{nb[w] / M:,.0f} m ({nb[w] / M / nav0 - 1:+.0%} from the start)", pm((nb[w] - na[w]) / M)],
        ["SPX held, market value", f"{v('E_A', w):,.0f} m", f"{v('E_B', w):,.0f} m", pm(v("E_B", w) - v("E_A", w))],
        ["calls held, market value", "—", f"{v('call_val', w):,.0f} m ({n_alive} of the {s.build} bought, on a notional of {v('call_notional', w):,.0f} m of index)", pm(v("call_val", w))],
        ["cash", "—", f"{v('cash_B', w):,.0f} m", pm(v("cash_B", w))],
        ["loan", f"{v('loan', w):,.0f} m", f"{v('loan_B', w):,.0f} m" + (" (rotation still under way)" if at_["loan_B"][w] > 0.5e6 else ""), pm(v("loan_B", w) - v("loan", w))],
        ["lending value of the holdings", f"{lv_a:,.0f} m ({s.lv_equity:.0%} of the SPX)", f"{lv_b:,.0f} m ({s.lv_equity:.0%} of the SPX + {s.lv_calls:.0%} of the calls)", pm(lv_b - lv_a)],
        ["dry powder = lending value − loan + cash", f"{lv_a:,.0f} − {v('loan', w):,.0f} = {v('headroom_A', w):,.0f} m (LTV {at_['ltv_A'][w]:.0%}: a further {max(1 - at_['ltv_A'][w], 0):.0%} SPX fall to a margin call)",
         f"{lv_b:,.0f} − {v('loan_B', w):,.0f} + {v('cash_B', w):,.0f} = {v('dry_powder_B', w):,.0f} m", pm(v("dry_powder_B", w) - v("headroom_A", w))],
        ["cost since the start", f"interest {v('interest_cum_A', w):,.0f} m", f"premiums {v('premiums_cum_B', w):,.0f} m − payoffs {v('payoffs_cum_B', w):,.0f} m = {net:,.0f} m", pm(net - v("interest_cum_A", w))],
    ]
    show(block("NAV", na / M, nb / M, lambda val: f" ({val / nav0 - 1:+.0%} from the start)") + worst + block("Dry powder", room_all, dp_all))
    n_wb += 1
if n_wb == 0:
    st.markdown("No market bottom falls inside the selected runs.")

st.caption(f"Market values on the bottom day, from the daily simulation of each trajectory. LTV = loan ÷ ({s.lv_equity:.0%} × SPX); the margin call comes when the SPX falls by a further 1 − LTV. "
           f"Dry powder = the lending value of what is held − loan + cash: {s.lv_equity:.0%} on the SPX, {s.lv_calls:.0%} on the calls (sidebar, Advanced); the calls are at market value, not notional. "
           "NAV and Dry powder: Keep and Rotate are each portfolio's own distribution over the starts running that day; Δ = Rotate − Keep on each row "
           "(lowest vs lowest, median vs median, … — the two cells are usually different start dates); 'rotation ahead on' compares each start with itself and counts those where the rotation is ahead. "
           "Worst trajectory: the start with the lowest NAV that day in either portfolio, and both strategies on that same start — its balance sheet and what it had cost since the start "
           "(Keep: the interest capitalised on the loan; Rotate: the premiums paid for every call bought less the payoffs received at expiry; a positive cost Δ = the rotation had cost more).")
low = pd.DataFrame({A: rs.loc[sel_all, "A: min headroom"] / M, B: rs.loc[sel_all, "B: min dry powder"] / M})
st.markdown(f"- Margin calls keeping the loan: **{int(rs.loc[sel_all, 'A: margin call'].sum()):,}** of the {len(sel_all):,} selected starts; the closest was {low[A].min():,.0f} m of room "
            f"({rs.loc[sel_all, 'A: min headroom'].idxmin():%d %b %Y} start, on {rs.loc[rs.loc[sel_all, 'A: min headroom'].idxmin(), 'A: min headroom date']:%d %b %Y}). "
            f"The rotation carries a loan only during the {s.build}-week build, so a margin call is impossible once it is complete.\n"
            f"- Over the whole holding period, at its lowest point the rotation has more dry powder than keeping the loan on **{float((low[B] > low[A]).mean()):.0%}** of the selected starts (median difference {(low[B] - low[A]).median():+,.0f} m).")

# ---------------- reading
st.markdown(f"- The rotated portfolio lost **all** its calls on **{lost_all.mean():.0%}** of start dates ({int(lost_all.sum()):,}) and **some** of them on {some_lapsed.mean():.0%}; "
            f"on the other {1 - some_lapsed.mean():.0%} every call expired in the money and was replaced.\n"
            f"- Annualised over each start's own holding period, the median return to today is {ann[A].median():+.1%} ({A}) vs {ann[B].median():+.1%} ({B}).\n"
            f"- Margin calls ({A}): **{int(rs['A: margin call'].sum()):,}** start dates over all {len(rs):,} starts.")
st.caption("Consecutive start dates overlap almost entirely, so the charts show the range of historical outcomes, not independent draws. Calls are priced and marked with Black–Scholes on SPXFP (q = r) at the implied vol of the day, "
           "extrapolated from the 24-month Bloomberg series; the delta is the model delta of the ATM call. No bid/ask, no early unwind. Dividends reinvested net of withholding; cash earns the 3-month T-bill.")
