"""Historical replay app — "if we had run A, B and C from a given date, what would the P&L have been?"

No Monte Carlo: one deterministic path through the user's S&P 500 history (SPX for the plain-vanilla
holding, SPXFP — futures excess-return index — as the option underlying). Illiquids follow their
expected drift with the configured cash-flow model.

Run:  .venv/bin/streamlit run app/historical_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fosim.analytics.episodes import (  # noqa: E402
    episode_table,
    return_distribution,
    risk_table,
    rolling_vol,
)
from fosim.analytics.inception import balance_sheets, inception_market  # noqa: E402
from fosim.config import PROJECT_ROOT, SimConfig  # noqa: E402
from fosim.engine.state import COMPONENTS  # noqa: E402
from fosim.reporting.excel_export import export_path_audit  # noqa: E402
from fosim.runner import run_config  # noqa: E402

M = 1e6
COLORS = {"A": "#1f4e79", "B": "#c55a11", "C": "#548235"}
DATA = PROJECT_ROOT / "data" / "SPX_SPXFP_aligned_monthly.csv"
CFG = PROJECT_ROOT / "config" / "historical_1997.yaml"

st.set_page_config(page_title="Historical replay — leverage vs. call replacement", layout="wide")


@st.cache_data
def history() -> pd.DataFrame:
    return pd.read_csv(DATA, parse_dates=["date"])


def base_raw() -> dict[str, Any]:
    with open(CFG) as f:
        d: dict[str, Any] = yaml.safe_load(f)
    return d


def pct_input(label: str, help_: str, key: str) -> float | None:
    """Blank text field in percent; blank → None (model placeholder is used and flagged)."""
    txt = st.text_input(label, value="", placeholder="blank = model placeholder", help=help_, key=key)
    if txt.strip() == "":
        return None
    try:
        return float(txt.replace("%", "").replace(",", ".")) / 100.0
    except ValueError:
        st.error(f"{label}: enter a number in percent, e.g. 4.5")
        return None


st.title("Historical replay — what would each strategy have done?")
st.caption("One deterministic path through the actual S&P 500 history. No Monte Carlo. Illiquids follow their expected drift.")

tab_in, tab_out, tab_risk = st.tabs(["Assumptions", "Results", "Risk"])

with tab_in:
    hist_m = history()
    dates = hist_m["date"].dt.date.tolist()
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Balance sheet of each strategy at the start (USD m)")
        st.caption("Enter the amounts directly. Illiquids are identical in every strategy (they cannot be traded). NAV = equities + cash + illiquids − loan.")
        default_tbl = pd.DataFrame(
            {"strategy": ["A  levered equities", "B  call ladder + dry powder", "C  unlevered"], "equities (SPX)": [875.0, 0.0, 625.0], "loan": [250.0, 0.0, 0.0], "cash": [0.0, 625.0, 0.0]}
        ).set_index("strategy")
        tbl = st.data_editor(default_tbl, width="stretch", num_rows="fixed")
        illiq = st.number_input("Illiquids (USD m, common to all strategies)", 0.0, 100_000.0, 375.0, 5.0)
        navs = tbl["equities (SPX)"] + tbl["cash"] + illiq - tbl["loan"]
        st.caption("Starting NAV: " + " · ".join(f"{n.split()[0]} {v:,.0f} m" for n, v in navs.items()))
        st.subheader("Strategy B — how the option book is built")
        b1, b2, b3 = st.columns(3)
        freq = b1.radio("Buy fresh ATM calls every", ["week", "month"], horizontal=True)
        per = b2.number_input("Notional bought per purchase (USD m)", 0.1, 10_000.0, 8.75, 0.25)
        target = b3.number_input("Target total notional (USD m)", 1.0, 100_000.0, 875.0, 5.0)
        n_buys = int(np.ceil(target / per))
        st.caption(f"{n_buys} purchases to reach the target (≈ {n_buys / (52 if freq == 'week' else 12):.1f} years); expired tranches are replaced so the book stays at the target. Tranches are held to expiry (cash-settled).")
        tenor = st.selectbox("Option tenor (years)", [1, 2, 3, 5, 7, 10], index=3)
        dry = st.checkbox("Deploy dry powder into SPX after drawdowns (−20/−30/−40 % tiers of deployable cash)", value=False)
    with c2:
        st.subheader("Period")
        start = st.selectbox("Start investing at", dates, index=0)
        end = st.selectbox("End", dates, index=len(dates) - 1)
        n_months = round((pd.Timestamp(end) - pd.Timestamp(start)).days / 30.4375)
        st.caption(f"{n_months} months of history · {'weekly' if freq == 'week' else 'monthly'} steps")
        st.subheader("Historical inputs — blank = model placeholder")
        lombard = pct_input("All-in Lombard cost (% p.a.)", "Base rate + spread on the loan, held constant. The loan is rolled monthly: interest accrues weekly on the balance and is added to the principal at each month-end (not paid from cash).", "lombard")
        cash_y = pct_input("Cash yield on dry powder (% p.a.)", "Also the risk-free rate used to price and discount the options.", "cashy")
        prem = pct_input(f"{tenor}y ATM call premium (% of notional, underlying SPXFP)", "Price of a fresh at-the-money long-dated call at every purchase, held constant. Blank → parametric implied-vol placeholder.", "prem")
        div = st.number_input("SPX dividend yield received by A/C (% p.a.)", 0.0, 6.0, 1.8, 0.1) / 100.0
        wht = st.number_input("Dividend withholding tax (%)", 0.0, 50.0, 15.0, 1.0) / 100.0
        missing = [x for x, v in (("Lombard cost = cash + 1 %", lombard), ("cash yield 3.9 %", cash_y), ("option premium from a 21 % implied vol", prem)) if v is None]
        st.markdown(("**Placeholders in use:** " + ", ".join(missing)) if missing else "**All three historical inputs supplied.**")

    run_btn = st.button("Run historical replay", type="primary")
    if run_btn:
        d = base_raw()
        weekly = freq == "week"
        d["run"]["dt"] = "weekly" if weekly else "monthly"
        d["run"]["horizon_years"] = n_months / 12.0
        d["historical"]["file"] = "data/SPX_SPXFP_aligned_weekly.csv" if weekly else "data/SPX_SPXFP_aligned_monthly.csv"
        d["historical"]["start"] = str(start)
        rows_ = {n.split()[0]: r for n, r in tbl.iterrows()}
        d["custom_inception"] = {
            "enabled": True, "illiquids": float(illiq) * M,
            "A": {"equities": float(rows_["A"]["equities (SPX)"]) * M, "loan": float(rows_["A"]["loan"]) * M, "cash": float(rows_["A"]["cash"]) * M},
            "B": {"equities": float(rows_["B"]["equities (SPX)"]) * M, "loan": float(rows_["B"]["loan"]) * M, "cash": float(rows_["B"]["cash"]) * M},
            "C": {"equities": float(rows_["C"]["equities (SPX)"]) * M, "loan": float(rows_["C"]["loan"]) * M, "cash": float(rows_["C"]["cash"]) * M},
            "reference_equity_exposure": max(float(rows_["A"]["equities (SPX)"]) * M, 1.0),
        }
        d["loan_terms"]["facility_limit"] = max(d["loan_terms"]["facility_limit"], float(rows_["A"]["loan"]) * M * 1.6, 1.0)
        d["options"]["ladder_mode"] = "fixed_notional_schedule"
        d["options"]["purchase_frequency"] = "weekly" if weekly else "monthly"
        d["options"]["notional_per_purchase"] = float(per) * M
        d["options"]["target_total_notional"] = float(target) * M
        d["options"]["tenor_years"] = float(tenor)
        d["options"]["hold_months_before_roll"] = min(12, int(tenor * 12))
        d["dry_powder"]["enabled"] = dry
        spx = next(i for i, e in enumerate(d["equity_indices"]) if e["name"] == "SPX")
        d["equity_indices"][spx]["dividend_yield"] = div
        d["equity_indices"][spx]["dividend_wht"] = wht
        if tenor > max(d["implied_vol"]["term_structure"]["tenors"]):
            d["implied_vol"]["term_structure"]["tenors"].append(float(tenor))
            d["implied_vol"]["term_structure"]["theta"].append(d["implied_vol"]["term_structure"]["theta"][-1])
            d["implied_vol"]["term_structure"]["beta"].append(d["implied_vol"]["term_structure"]["beta"][-1])
        if cash_y is not None:
            d["rates"]["usd"]["r0"] = cash_y
            d["rates"]["cash"]["spread"] = 0.0
        if lombard is not None:
            d["loan_terms"]["rate_type"] = "fixed"
            d["loan_terms"]["fixed_years"] = n_months / 12.0 + 1
            d["loan_terms"]["fixed_rate"] = lombard
            d["loan_terms"]["spread_tiers"] = [{"utilisation_below": 1.0, "spread": 0.0}]
        if prem is not None:
            d["pricing"]["scenario_overrides"] = [{"index": "SPXFP", "month_from": 0, "tenor": float(tenor), "premium_pct": prem}]
            d["pricing"]["bid_ask_vol_pts_new"] = 0.0
            d["pricing"]["bid_ask_vol_pts_unwind"] = 0.0
        try:
            cfg = SimConfig.model_validate(d)
            with st.spinner("Replaying history…"):
                out = run_config(cfg, ledger_paths=[0], base_dir=PROJECT_ROOT)
            st.session_state["hist_run"] = (out, {"lombard": lombard, "cash": cash_y, "prem": prem, "start": str(start), "end": str(end), "weekly": weekly})
            st.success("Done — see the Results tab.")
        except Exception as e:
            st.error(f"Could not run: {e}")

with tab_out:
    if "hist_run" not in st.session_state:
        st.info("Set the assumptions and press **Run historical replay**.")
    else:
        out, meta = st.session_state["hist_run"]
        cfg = out.cfg
        res = out.results
        hist = pd.read_csv(PROJECT_ROOT / "data" / ("SPX_SPXFP_aligned_weekly.csv" if meta.get("weekly") else "SPX_SPXFP_aligned_monthly.csv"), parse_dates=["date"])
        dates_all = hist["date"][hist["date"] >= pd.Timestamp(meta["start"])].reset_index(drop=True)
        t_dates = dates_all.iloc[: out.paths.n_steps + 1]
        # ---------------- headline table
        rows = []
        for k, r in res.items():
            nav = r.nav[0]
            yrs = out.paths.grid.horizon_years
            comp = r.recorder.components[0, 1:, :].sum(axis=0)
            ci = {c: comp[i] for i, c in enumerate(COMPONENTS)}
            ev = r.events
            rows.append({
                "strategy": k, "final NAV (m)": nav[-1] / M, "P&L (m)": (nav[-1] - nav[0]) / M, "CAGR": (nav[-1] / nav[0]) ** (1 / yrs) - 1,
                "max drawdown": float((nav / np.maximum.accumulate(nav) - 1).min()), "min NAV (m)": nav.min() / M,
                "margin calls": int(ev.margin_calls[0]), "forced liquidations": int(ev.forced_liquidations[0]), "slippage loss (m)": ev.slippage_loss[0] / M,
                "loan interest (m)": ci["loan_interest"] / M, "cash interest (m)": ci["cash_interest"] / M, "dividends net (m)": ci["dividends"] / M,
                "option MTM incl. payoffs (m)": ci["option_mtm"] / M, "option bid/ask + costs (m)": (ci["option_bid_ask"] + ci["transaction_costs"]) / M,
                "premium paid total (m)": float(sum(t for t in r.ledger.to_frame().query("account == 'cash' and description.str.contains('premium at ask')", engine="python")["amount_usd"])) / -M if k == "B" else 0.0,
                "dry powder deployed (m)": ev.dry_powder_deployed_usd[0] / M,
            })
        tbl = pd.DataFrame(rows).set_index("strategy")
        rt0 = risk_table(res, out.paths)
        tbl.insert(3, "ann. volatility", rt0["ann_vol"])
        tbl.insert(4, "Sharpe (CAGR / vol)", rt0["sharpe"])
        tbl.insert(5, "Sortino", rt0["sortino"])
        st.subheader(f"P&L if deployed on {meta['start']} and held to {t_dates.iloc[-1].date()}")
        st.dataframe(tbl.style.format({"CAGR": "{:.2%}", "ann. volatility": "{:.1%}", "Sharpe (CAGR / vol)": "{:.2f}", "Sortino": "{:.2f}", "max drawdown": "{:.1%}"}, precision=1), width="stretch")
        # ---------------- NAV chart
        fig = go.Figure()
        for k, r in res.items():
            fig.add_trace(go.Scatter(x=t_dates, y=r.nav[0] / M, name=f"NAV {k}", line={"color": COLORS.get(k), "width": 2}))
        spx_i = cfg.index_names.index("SPX")
        fig.add_trace(go.Scatter(x=t_dates, y=out.paths.S[0, :, spx_i] / out.paths.S[0, 0, spx_i] * res["C"].nav[0, 0] / M, name="SPX price index (rebased to capital)", line={"dash": "dot", "color": "#999"}))
        for k, r in res.items():
            led = r.ledger.to_frame()
            for _, e in led[led.account == "event"].iterrows():
                fig.add_vline(x=t_dates.iloc[int(e.step)], line={"color": COLORS.get(k, "#000"), "dash": "dash", "width": 1})
        fig.update_layout(title="NAV (USD m) — dashed verticals: margin events of A", template="plotly_white", height=460, legend={"orientation": "h"})
        st.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for k, r in res.items():
                nav = r.nav[0]
                fig.add_trace(go.Scatter(x=t_dates, y=(nav / np.maximum.accumulate(nav) - 1) * 100, name=k, line={"color": COLORS.get(k)}))
            fig.update_layout(title="Drawdown from peak (%)", template="plotly_white", height=360, legend={"orientation": "h"})
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_dates, y=res["A"].series("utilisation")[0], name="A margin utilisation", line={"color": COLORS["A"]}))
            fig.add_trace(go.Scatter(x=t_dates, y=res["B"].series("cash")[0] / res["B"].nav[0], name="B cash / NAV", line={"color": COLORS["B"]}))
            fig.add_trace(go.Scatter(x=t_dates, y=res["B"].series("equity_exposure")[0] / np.maximum(res["A"].series("equity_exposure")[0], 1), name="B exposure / A exposure", line={"color": "#888"}))
            fig.update_layout(title="A: margin utilisation (call at 1.0) · B: cash share and exposure vs A", template="plotly_white", height=360, legend={"orientation": "h"})
            st.plotly_chart(fig, width="stretch")
        # ---------------- yearly table
        yrs_idx = t_dates.dt.year.to_numpy()
        yrows = []
        for y in sorted(set(yrs_idx)):
            idx = np.flatnonzero(yrs_idx == y)
            k0, k1 = max(idx[0] - 1, 0), idx[-1]
            row = {"year": y}
            for k, r in res.items():
                row[f"{k} return"] = r.nav[0, k1] / r.nav[0, k0] - 1
            row["SPX price"] = out.paths.S[0, k1, spx_i] / out.paths.S[0, k0, spx_i] - 1
            row["A margin calls"] = int((res["A"].ledger.to_frame().query("account == 'event' and step >= @k0 and step <= @k1 and description.str.contains('MARGIN')", engine="python")).shape[0])
            yrows.append(row)
        st.subheader("Calendar-year returns")
        st.dataframe(pd.DataFrame(yrows).set_index("year").style.format({c: "{:.1%}" for c in ("A return", "B return", "C return", "SPX price")}), width="stretch", height=420)
        # ---------------- drawdown episodes: what was left at each trough
        st.subheader("After each major drawdown of the SPX: dry powder and equity exposure left at the trough")
        thr = st.select_slider("Episode threshold (SPX drawdown)", [-0.10, -0.15, -0.20, -0.30], value=-0.15, format_func=lambda v: f"{v:.0%}")
        ep = episode_table(res, out.paths, cfg, t_dates, threshold=thr)
        if len(ep):
            show = ep.copy()
            for c in ("nav", "cash_dry_powder", "equity_exposure", "held_equities", "deployed_spot", "option_book_value", "loan"):
                show[c] = show[c] / M
            show = show[["episode", "index_drawdown", "recovered", "strategy", "nav", "drawdown_from_own_peak", "cash_dry_powder", "cash_pct_nav", "equity_exposure", "exposure_pct_nav", "held_equities", "option_book_value", "n_tranches", "deployed_spot", "loan", "utilisation"]]
            st.dataframe(show.style.format({"index_drawdown": "{:.1%}", "drawdown_from_own_peak": "{:.1%}", "cash_pct_nav": "{:.1%}", "exposure_pct_nav": "{:.1%}", "utilisation": "{:.2f}"}, precision=0, na_rep="–"), width="stretch", height=min(60 + 36 * len(show), 700))
            st.caption("USD m at the SPX trough. Equity exposure: A/C = equities held; B = option dollar-delta + deployed spot. Utilisation = loan / lending value (margin call at 1.00) — A only. Dry powder = cash.")
        else:
            st.info("No SPX drawdown deeper than the threshold in this window.")
        st.subheader("Risk measures (from the strategies' NAV paths)")
        rt = risk_table(res, out.paths)
        st.dataframe(rt.style.format({c: "{:.1%}" for c in rt.columns if c not in ("sharpe", "sortino")} | {"sharpe": "{:.2f}", "sortino": "{:.2f}"}), width="stretch")
        st.caption("Step = one simulation step (week or month). VaR = loss quantile of step returns (and of rolling 1-year returns). Sharpe = CAGR ÷ annualised vol of step returns (no cash-rate deduction); Sortino = CAGR ÷ annualised downside deviation below zero. Illiquid marks are appraisal-smoothed, which flatters both.")
        # ---------------- inputs used
        mk = inception_market(cfg)
        bs = balance_sheets(cfg)
        vol_used = float(bs.attrs["tranche"]["vol"])
        prem_note = f"(placeholder from a {vol_used:.1%} implied vol)" if meta["prem"] is None else "(your input)"
        st.subheader("Inputs actually used")
        st.markdown(
            f"- Lombard cost: **{(meta['lombard'] if meta['lombard'] is not None else mk.loan_rate):.2%}** p.a. {'(placeholder)' if meta['lombard'] is None else ''}\n"
            f"- Cash yield / risk-free rate: **{(meta['cash'] if meta['cash'] is not None else mk.cash_rate):.2%}** p.a. {'(placeholder)' if meta['cash'] is None else ''}\n"
            f"- {cfg.options.tenor_years:g}y ATM call premium on SPXFP: **{float(bs.attrs['tranche']['premium_pct']):.2%}** of notional {prem_note}; delta {float(bs.attrs['tranche']['delta']):.3f}\n"
            f"- Inception: A equities {bs.loc['A', 'equities'] / M:,.0f} m / loan {bs.loc['A', 'loan'] / M:,.0f} m; B target book {cfg.options.target_total_notional / M:,.0f} m notional bought {cfg.options.notional_per_purchase / M:,.2f} m per {cfg.options.purchase_frequency[:-2]} (premium at the target ≈ {bs.loc['B', 'options_premium'] / M:,.0f} m); illiquids {bs.loc['A', 'illiquids'] / M:,.0f} m in all strategies.\n"
            f"- Loan: rolled monthly, interest capitalised into the principal (ACT/360); A's loan at the end: {res['A'].series('loan')[0, -1] / M:,.0f} m.\n"
            f"- Illiquids: deterministic expected drift (HF 6 %, PE 10 %, Infra 8 % p.a. placeholders) with capital calls/distributions; identical across strategies."
        )
        tmp = PROJECT_ROOT / "reports" / "historical_audit.xlsx"
        if st.button("Export the full ledger / balance sheet to Excel"):
            export_path_audit(res, out.paths, 0, tmp)
            st.download_button("Download", tmp.read_bytes(), tmp.name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


with tab_risk:
    st.subheader("Rolling volatility (annualised, 52-week window)")
    if "hist_run" in st.session_state:
        out, meta = st.session_state["hist_run"]
        hist_r = pd.read_csv(PROJECT_ROOT / "data" / ("SPX_SPXFP_aligned_weekly.csv" if meta.get("weekly") else "SPX_SPXFP_aligned_monthly.csv"), parse_dates=["date"])
        t_dates = hist_r["date"][hist_r["date"] >= pd.Timestamp(meta["start"])].reset_index(drop=True).iloc[: out.paths.n_steps + 1]
        spy = out.paths.grid.steps_per_year
        win = spy  # one year of steps
        fig = go.Figure()
        for k, r in out.results.items():
            fig.add_trace(go.Scatter(x=t_dates, y=rolling_vol(r.nav[0], win, spy) * 100, name=f"NAV {k}", line={"color": COLORS.get(k)}))
        spx_i = out.cfg.index_names.index("SPX")
        fig.add_trace(go.Scatter(x=t_dates, y=rolling_vol(out.paths.S[0, :, spx_i], win, spy) * 100, name="SPX", line={"dash": "dot", "color": "#999"}))
        fig.update_layout(template="plotly_white", height=400, yaxis_title="vol % p.a.", legend={"orientation": "h"})
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Run a replay on the Assumptions tab to see the strategies' rolling volatility; the index distributions below do not need a run.")
    st.subheader("Distribution of the underlying index returns (daily data, 1997-09 → 2026-09)")
    daily = pd.read_csv(PROJECT_ROOT / "data" / "SPX_SPXFP_aligned_daily.csv", parse_dates=["date"])
    which = st.radio("Index", ["SPX (price index, held by A/C)", "SPXFP (futures excess return, option underlying)"], horizontal=True)
    col = "spx_rebased" if which.startswith("SPX ") else "spxfp"
    lvl = daily[col].to_numpy(dtype=float)
    cols = st.columns(3)
    stats_rows = {}
    for (label, h), c in zip((("Daily", 1), ("Rolling 3-year", 756), ("Rolling 5-year", 1260)), cols, strict=True):
        d = return_distribution(lvl, h)
        stats_rows[label] = {k: v for k, v in d.items() if k != "returns"}
        ret = d["returns"] * 100
        fig = go.Figure(go.Histogram(x=ret, nbinsx=80, marker_color="#7f9cc0", name="returns"))
        fig.add_vline(x=-d["var95"] * 100, line={"color": "#c55a11", "width": 2, "dash": "dash"})
        fig.add_vline(x=-d["var99"] * 100, line={"color": "#b00020", "width": 2, "dash": "dot"})
        fig.add_annotation(x=-d["var95"] * 100, y=1.0, yref="paper", text=f"VaR 95%: {-d['var95']:.1%}", showarrow=False, xanchor="left", yanchor="top", font={"color": "#c55a11", "size": 11}, bgcolor="rgba(255,255,255,0.8)")
        fig.add_annotation(x=-d["var99"] * 100, y=0.86, yref="paper", text=f"VaR 99%: {-d['var99']:.1%}", showarrow=False, xanchor="right", yanchor="top", font={"color": "#b00020", "size": 11}, bgcolor="rgba(255,255,255,0.8)")
        fig.update_layout(title=f"{label} returns (n = {d['n']:,})", template="plotly_white", height=340, xaxis_title="return %", yaxis_title="count", showlegend=False, margin={"t": 60})
        c.plotly_chart(fig, width="stretch")
    st.subheader("Value at risk of the index (losses, % of value)")
    var_tbl = pd.DataFrame(
        {lab: {"VaR 95%": v["var95"], "VaR 99%": v["var99"], "CVaR 95%": v["cvar95"], "CVaR 99%": v["cvar99"], "worst observed": -v["worst"], "P(return < 0)": v["p_negative"], "mean return": v["mean"], "median return": v["median"], "std dev": v["std"], "observations": v["n"]} for lab, v in stats_rows.items()}
    )
    st.dataframe(var_tbl.style.format("{:.1%}", subset=pd.IndexSlice[[r for r in var_tbl.index if r != "observations"], :]).format("{:,.0f}", subset=pd.IndexSlice[["observations"], :]), width="stretch")
    st.markdown("**Other moments**")
    st.dataframe(pd.DataFrame({lab: {"skew": v["skew"], "excess kurtosis": v["kurtosis"], "best observed": v["best"]} for lab, v in stats_rows.items()}).style.format("{:.3g}"), width="stretch")
    st.caption("VaR = loss quantile of the historical returns (positive numbers are losses); CVaR = average loss beyond VaR. Rolling 3- and 5-year windows overlap heavily (about 30 independent 1-year blocks), so tail quantiles are indicative, not precise.")
