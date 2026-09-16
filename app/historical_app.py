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

import fosim.analytics.call_vs_cash as cvc  # noqa: E402
from fosim.analytics.episodes import (  # noqa: E402
    episode_table,
    return_distribution,
    risk_table,
    rolling_vol,
)
from fosim.analytics.inception import balance_sheets, inception_market  # noqa: E402
from fosim.config import PROJECT_ROOT, SimConfig  # noqa: E402
from fosim.engine.state import COMPONENTS  # noqa: E402
from fosim.pricing.black_scholes import bsm_greeks, bsm_price  # noqa: E402
from fosim.pricing.implied_vol import implied_vol  # noqa: E402
from fosim.reporting.excel_export import export_path_audit  # noqa: E402
from fosim.runner import run_config  # noqa: E402

M = 1e6
COLORS = {"A": "#1f4e79", "B": "#c55a11", "C": "#548235"}
DATA = PROJECT_ROOT / "data" / "SPX_SPXFP_aligned_monthly.csv"
MKT = {f: PROJECT_ROOT / "data" / f"market_data_{f}.csv" for f in ("daily", "weekly", "monthly")}
MKT_COLS = {"SPXFP": "spxfp", "SPX": "spx_rebased", "LOAN_BASE": "loan_base", "CASH_YIELD": "tbill_3m", "OPTION_RATE": "ust_5y", "DIV_YIELD": "spx_div_yld", "IV": "iv_5y"}
CFG = PROJECT_ROOT / "config" / "historical_1997.yaml"

st.set_page_config(page_title="Historical replay — leverage vs. call replacement", layout="wide")


@st.cache_data
def history() -> pd.DataFrame:
    return pd.read_csv(DATA, parse_dates=["date"])


@st.cache_data
def series_premium(tenor: float) -> pd.Series:
    """ATM call premium on SPXFP (% of notional) implied by the vol series and the 5y Treasury of each week (q = r)."""
    m = pd.read_csv(MKT["weekly"], parse_dates=["date"]).dropna(subset=["ust_5y", "iv_5y"]).set_index("date")
    r = m.ust_5y.to_numpy() / 100.0
    px = bsm_price(np.full(len(m), 100.0), np.full(len(m), 100.0), r, r, m.iv_5y.to_numpy() / 100.0, tenor)
    return pd.Series(np.asarray(px, dtype=np.float64) / 100.0, index=m.index, name="premium")


@st.cache_data
def call_vs_cash_backtest(tenor: float, premium_usd: float, start: str, end: str, prem_fixed: float | None, cash_leg: str, wht: float) -> tuple[pd.DataFrame, cvc.BacktestInfo]:
    return cvc.backtest(tenor, premium_usd, start, end, prem_fixed, cash_leg, wht)


@st.cache_data
def market_last(tenor: float) -> dict[str, float]:
    m = pd.read_csv(MKT["weekly"], parse_dates=["date"]).dropna(subset=["loan_base", "tbill_3m", "ust_5y", "iv_5y"]).iloc[-1]
    return {"date": str(m.date.date()), "loan_base": float(m.loan_base), "tbill": float(m.tbill_3m), "ust5y": float(m.ust_5y), "iv_5y": float(m.iv_5y),
            "prem": float(series_premium(tenor).iloc[-1])}


def base_raw() -> dict[str, Any]:
    with open(CFG) as f:
        d: dict[str, Any] = yaml.safe_load(f)
    return d


def pct_input(label: str, help_: str, key: str, default: str = "") -> float | None:
    """Blank text field in percent; blank → None (model placeholder is used and flagged)."""
    txt = st.text_input(label, value=default, placeholder="blank = model placeholder", help=help_, key=key)
    if txt.strip() == "":
        return None
    try:
        return float(txt.replace("%", "").replace(",", ".")) / 100.0
    except ValueError:
        st.error(f"{label}: enter a number in percent, e.g. 4.5")
        return None


st.title("Historical replay — what would each strategy have done?")
st.caption("One deterministic path through the actual S&P 500 history. No Monte Carlo. Illiquids follow their expected drift.")

tab_in, tab_out, tab_risk, tab_bt = st.tabs(["Assumptions", "Results", "Risk", "Call vs cash backtest"])

with tab_in:
    hist_m = history()
    dates = hist_m["date"].dt.date.tolist()
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Balance sheet of each strategy at the start (USD m)")
        st.caption("Enter the amounts directly. Illiquids are identical in every strategy (they cannot be traded). NAV = equities + cash + illiquids − loan.")
        default_tbl = pd.DataFrame(
            {"strategy": ["A  levered equities", "B  call ladder + dry powder", "C  unlevered"], "equities (SPX)": [1000.0, 0.0, 750.0], "loan": [250.0, 0.0, 0.0], "cash": [0.0, 750.0, 0.0]}
        ).set_index("strategy")
        tbl = st.data_editor(default_tbl, width="stretch", num_rows="fixed")
        illiq = st.number_input("Illiquids (USD m, common to all strategies)", 0.0, 100_000.0, 250.0, 5.0)
        navs = tbl["equities (SPX)"] + tbl["cash"] + illiq - tbl["loan"]
        st.caption("Starting NAV: " + " · ".join(f"{n.split()[0]} {v:,.0f} m" for n, v in navs.items()))
        st.subheader("Strategy B — how the option book is built")
        b1, b2, b3 = st.columns(3)
        freq = b1.radio("Buy fresh ATM calls every", ["week", "month"], horizontal=True)
        per = b2.number_input("Notional bought per purchase (USD m)", 0.1, 10_000.0, 10.0, 0.25)
        target = b3.number_input("Target total notional (USD m)", 1.0, 100_000.0, 1000.0, 5.0)
        n_buys = int(np.ceil(target / per))
        st.caption(f"{n_buys} purchases to reach the target (≈ {n_buys / (52 if freq == 'week' else 12):.1f} years); expired tranches are replaced so the book stays at the target. Tranches are held to expiry (cash-settled).")
        tenor = st.selectbox("Option tenor (years)", [1, 2, 3, 5, 7, 10], index=3)
        dry = st.checkbox("Deploy dry powder into MORE CALLS after SPX drawdowns (B only)", value=False, help="At each month-end, if the SPX drawdown from its running peak reaches a tier, B spends the tier's share of deployable cash (cash minus the liquidity reserve) on additional ATM calls of the same tenor. Each tier fires once per drawdown episode; re-armed at a new peak.")
        if dry:
            t1, t2, t3 = st.columns(3)
            tier_dd = [t1.number_input("Tier 1 drawdown", -0.9, -0.01, -0.20, 0.05, format="%.2f"), t2.number_input("Tier 2 drawdown", -0.9, -0.01, -0.30, 0.05, format="%.2f"), t3.number_input("Tier 3 drawdown", -0.9, -0.01, -0.40, 0.05, format="%.2f")]
            tier_pct = [t1.number_input("Tier 1: % of deployable cash", 0.0, 1.0, 0.33, 0.05), t2.number_input("Tier 2: % of deployable cash", 0.0, 1.0, 0.50, 0.05), t3.number_input("Tier 3: % of deployable cash", 0.0, 1.0, 1.00, 0.05)]
            reserve_pct = st.number_input("Liquidity reserve kept back (% of NAV)", 0.0, 0.5, 0.03, 0.01)
        st.subheader("Lombard lending values (advance rates)")
        l1, l2, l3 = st.columns(3)
        ltv_eq = l1.number_input("LTV on listed equities", 0.0, 1.0, 0.50, 0.05)
        ltv_ill = l2.number_input("LTV on illiquids", 0.0, 1.0, 0.0, 0.05)
        ltv_cash = l3.number_input("LTV on cash", 0.0, 1.0, 0.90, 0.05)
        st.caption("Lending value = Σ LTV × market value; margin call when the loan (incl. accrued interest) exceeds it. Long options have no lending value.")
    with c2:
        st.subheader("Period")
        start = st.selectbox("Start investing at", dates, index=0)
        end = st.selectbox("End", dates, index=len(dates) - 1)
        n_months = round((pd.Timestamp(end) - pd.Timestamp(start)).days / 30.4375)
        st.caption(f"{n_months} months of history · {'weekly' if freq == 'week' else 'monthly'} steps")
        st.subheader("Market data")
        use_mkt = st.checkbox("Use the Bloomberg history (3m LIBOR/SOFR, T-bill, 5y Treasury, implied vol, dividend yield)", value=True,
                              help="Rates, the option discount rate, the dividend yield and (in market-premium mode) the implied vol vary week by week as they did. Off = constant inputs below.")
        ml = market_last(float(tenor))
        if use_mkt:
            sp = series_premium(float(tenor))
            sp_win = sp.loc[str(start):str(end)]
            st.caption(f"Latest observations ({ml['date']}): 3m base {ml['loan_base']:.2f} %, 3m T-bill {ml['tbill']:.2f} %, 5y Treasury {ml['ust5y']:.2f} %, "
                       f"{tenor}y ATM vol (series) {ml['iv_5y']:.1f} % → {tenor}y ATM call premium **{ml['prem']:.1%}** of notional. "
                       f"Over the selected window the series-implied premium averages **{sp_win.mean():.1%}** (min {sp_win.min():.1%}, max {sp_win.max():.1%}).")
            lombard = pct_input("Lombard spread over the 3m base (% p.a.)", "All-in loan rate = 3m LIBOR / Term SOFR of the day + this spread, re-fixed at each monthly roll and capitalised.", "lombard", default=f"{4.3 - ml['loan_base']:.2f}")
            cash_y = pct_input("Cash yield: spread vs 3m T-bill (% p.a., negative = below bills)", "Dry powder earns the T-bill yield of the day plus this spread.", "cashy", default=f"{3.5 - ml['tbill']:.2f}")
            prem = st.text_input(f"{tenor}y ATM call premium TODAY (% of notional, underlying SPXFP) — optional dealer quote", value="", placeholder=f"blank = as implied by the data ({ml['prem']:.1%} today)",
                                 help="Leave blank to price the calls exactly as the vol series and the 5y Treasury imply. Enter today's dealer quote to re-scale the whole vol series so that the quote is reproduced on the last date (Market mode) or to pay that premium at every purchase (Fixed mode).", key="prem")
            prem = None if prem.strip() == "" else float(prem.replace("%", "").replace(",", ".")) / 100.0
            prem_mode = st.radio("Option premium through history", ["Market: vol and 5y rate of the day (premium varies purchase by purchase)",
                                                                     f"Fixed: one premium at every purchase (blank = window average {sp_win.mean():.1%})"], horizontal=False)
            div = None
            st.caption("Dividends: trailing 12-month S&P yield of the day.")
        else:
            lombard = pct_input("All-in Lombard cost (% p.a.)", "Base rate + spread on the loan, held constant. The loan is rolled monthly: interest accrues weekly on the balance and is added to the principal at each month-end (not paid from cash).", "lombard", default="4.3")
            cash_y = pct_input("Cash yield on dry powder (% p.a.)", "Also the risk-free rate used to price and discount the options.", "cashy", default="3.5")
            prem = pct_input(f"{tenor}y ATM call premium (% of notional, underlying SPXFP)", "Price of a fresh at-the-money long-dated call at every purchase, held constant. Blank → parametric implied-vol placeholder.", "prem", default="15")
            prem_mode = "Fixed"
            div = st.number_input("SPX dividend yield received by A/C (% p.a.)", 0.0, 6.0, 1.8, 0.1) / 100.0
        wht = st.number_input("Dividend withholding tax (%)", 0.0, 50.0, 15.0, 1.0) / 100.0

    run_btn = st.button("Run historical replay", type="primary")
    if run_btn:
        d = base_raw()
        weekly = freq == "week"
        d["run"]["dt"] = "weekly" if weekly else "monthly"
        d["run"]["horizon_years"] = n_months / 12.0
        if use_mkt:
            d["historical"]["file"] = str(MKT["weekly" if weekly else "monthly"].relative_to(PROJECT_ROOT))
            cols = dict(MKT_COLS)
            if not prem_mode.startswith("Market"):
                cols.pop("IV")  # fixed premium: constant vol from the quote, series only for rates and dividends
            d["historical"]["columns"] = cols
        else:
            d["historical"]["file"] = "data/SPX_SPXFP_aligned_weekly.csv" if weekly else "data/SPX_SPXFP_aligned_monthly.csv"
            d["historical"]["columns"] = {"SPXFP": "spxfp", "SPX": "spx_rebased"}
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
        d["dry_powder"]["instrument"] = "call_tranches"
        d["dry_powder"]["trigger_index"] = "SPX"
        if dry:
            order = sorted(range(3), key=lambda i: -tier_dd[i])  # mildest first
            d["dry_powder"]["tiers"] = [{"index_drawdown": float(tier_dd[i]), "iv_short_above": None, "deploy_pct_of_deployable": float(tier_pct[i])} for i in order]
            d["dry_powder"]["liquidity_reserve"]["floor_pct_nav"] = float(reserve_pct)
        d["leverage"]["ltv_base"] = {"equity_index": float(ltv_eq), "illiquid": float(ltv_ill), "cash": float(ltv_cash)}
        spx = next(i for i, e in enumerate(d["equity_indices"]) if e["name"] == "SPX")
        d["equity_indices"][spx]["dividend_yield"] = div if div is not None else 0.018  # overridden by the DIV_YIELD series when present
        d["equity_indices"][spx]["dividend_wht"] = wht
        if tenor > max(d["implied_vol"]["term_structure"]["tenors"]):
            d["implied_vol"]["term_structure"]["tenors"].append(float(tenor))
            d["implied_vol"]["term_structure"]["theta"].append(d["implied_vol"]["term_structure"]["theta"][-1])
            d["implied_vol"]["term_structure"]["beta"].append(d["implied_vol"]["term_structure"]["beta"][-1])
        if use_mkt:
            # spreads over the observed series; the loan floats with the 3m base
            d["loan_terms"]["rate_type"] = "floating"
            d["loan_terms"]["fixed_years"] = None
            d["loan_terms"]["fixed_rate"] = None
            d["loan_terms"]["spread_tiers"] = [{"utilisation_below": 1.0, "spread": float(lombard or 0.0)}]
            d["rates"]["cash"]["spread"] = -float(cash_y or 0.0)  # engine subtracts the spread from the series
            d["rates"]["usd"]["r0"] = ml["tbill"] / 100.0  # fallback short rate (unused when the series are present)
            d["pricing"]["bid_ask_vol_pts_new"] = 0.0
            d["pricing"]["bid_ask_vol_pts_unwind"] = 0.0
            r_anchor = ml["ust5y"] / 100.0
            # blank quote → the series as fitted (scale 1); a quote re-scales the series so that it is reproduced on the last date
            vol_anchor = implied_vol(prem * 100.0, 100.0, 100.0, r_anchor, r_anchor, float(tenor), "call") if prem is not None else ml["iv_5y"] / 100.0
            if prem_mode.startswith("Market"):
                d["historical"]["iv_scale"] = vol_anchor / (ml["iv_5y"] / 100.0)
                d["implied_vol"]["short"] = {**d["implied_vol"]["short"], "theta": vol_anchor, "iv0": None, "floor": 0.03, "cap": 2.0}
                d["implied_vol"]["term_structure"] = {"tenors": [1, 2, 5], "theta": [vol_anchor] * 3, "beta": [1.0, 1.0, 1.0]}
                d["pricing"]["scenario_overrides"] = []
            else:
                prem_fixed = prem if prem is not None else float(series_premium(float(tenor)).loc[str(start):str(end)].mean())
                d["pricing"]["scenario_overrides"] = [{"index": "SPXFP", "month_from": 0, "tenor": float(tenor), "premium_pct": prem_fixed}]
                prem = prem_fixed
        else:
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
            st.session_state["hist_run"] = (out, {"lombard": lombard, "cash": cash_y, "prem": prem, "start": str(start), "end": str(end), "weekly": weekly, "use_mkt": use_mkt, "prem_mode": prem_mode})
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
        # ---------------- equity exposure evolution
        fig = go.Figure()
        for k, r in res.items():
            fig.add_trace(go.Scatter(x=t_dates, y=r.series("equity_exposure")[0] / M, name=f"{k} equity exposure", line={"color": COLORS.get(k), "width": 2}))
        fig.update_layout(title="Equity exposure (USD m) — A/C: SPX held × β; B: option dollar-delta", template="plotly_white", height=420, legend={"orientation": "h"})
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
            fig.update_layout(title="A: margin utilisation (call at 1.0) · B: cash share of NAV", template="plotly_white", height=360, legend={"orientation": "h"})
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
        prem_used = float(bs.attrs["tranche"]["premium_pct"])
        delta_used = float(bs.attrs["tranche"]["delta"])
        if meta["prem"] is not None:
            # the engine converts the premium input into an implied vol (S = K = 100, r = cash yield, q = r for SPXFP)
            r_opt = cfg.rates.usd.r0 + cfg.pricing.option_funding_spread
            vol_used = implied_vol(meta["prem"] * 100.0, 100.0, 100.0, r_opt, r_opt, cfg.options.tenor_years, "call")
            prem_used = float(meta["prem"])
            delta_used = float(bsm_greeks(100.0, 100.0, r_opt, r_opt, vol_used, cfg.options.tenor_years).delta)
        prem_note = f"(placeholder from a {vol_used:.1%} implied vol)" if meta["prem"] is None else f"(your input; implied vol {vol_used:.1%})"
        st.subheader("Inputs actually used")
        trs = res["B"].ledger.tranche_frame().drop_duplicates(["slot", "purchase_step"]).query("notional_usd > 0").copy()
        trs["prem_pct"] = trs.cost_usd / trs.notional_usd
        trs["date"] = pd.to_datetime(np.asarray(t_dates)[trs.purchase_step.to_numpy(dtype=int)])
        paid_avg = float(trs.cost_usd.sum() / trs.notional_usd.sum()) if len(trs) else float("nan")
        if meta.get("use_mkt"):
            sr = out.paths.series
            lr = res["A"].series("loan_rate")[0]
            st.markdown(
                f"- Lombard cost: 3m LIBOR / Term SOFR of the day + **{(meta['lombard'] or 0):.2%}**; realised average **{lr.mean():.2%}** p.a. (min {lr.min():.2%}, max {lr.max():.2%}), rolled monthly and capitalised.\n"
                f"- Cash yield: 3m T-bill of the day {'+' if (meta['cash'] or 0) >= 0 else '−'} **{abs(meta['cash'] or 0):.2%}**; realised average **{res['B'].series('cash_rate')[0].mean():.2%}** p.a. (min {res['B'].series('cash_rate')[0].min():.2%}).\n"
                f"- Option discount rate: 5y Treasury of the day (average {sr['option_rate'][0].mean():.2%}). Dividends: trailing 12m S&P yield of the day (average {np.expm1(sr['div_yield'][0]).mean():.2%}).\n"
                + (f"- {cfg.options.tenor_years:g}y ATM call premium: **vol and 5y rate of the day** — premium paid averages **{paid_avg:.1%}** of notional across {len(trs)} purchases (min {trs.prem_pct.min():.1%}, max {trs.prem_pct.max():.1%}; by year below); "
                   + (f"vol series scaled ×{cfg.historical.iv_scale:.3f} so that today's quote ({meta['prem']:.1%}) is reproduced" if meta['prem'] is not None else "vol series as fitted from the 24m implied vol (no anchor)")
                   + f"; 5y vol average {out.paths.iv_short[0].mean():.1%} (min {out.paths.iv_short[0].min():.1%}, max {out.paths.iv_short[0].max():.1%}).\n"
                   if str(meta.get('prem_mode', '')).startswith('Market') else f"- {cfg.options.tenor_years:g}y ATM call premium: **fixed at {meta['prem']:.2%}** at every purchase (constant vol {vol_used:.1%}).\n")
            )
            if str(meta.get("prem_mode", "")).startswith("Market") and len(trs):
                by_year = trs.groupby(trs.date.dt.year).prem_pct.agg(["mean", "min", "max"]).rename(columns={"mean": "average", "min": "lowest", "max": "highest"})
                st.dataframe(by_year.T.style.format("{:.1%}"), width="stretch")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_dates, y=lr * 100, name="Lombard rate (%)", line={"color": COLORS["A"]}))
            fig.add_trace(go.Scatter(x=t_dates, y=res["B"].series("cash_rate")[0] * 100, name="cash yield (%)", line={"color": COLORS["B"]}))
            fig.add_trace(go.Scatter(x=t_dates, y=sr["option_rate"][0] * 100, name="5y Treasury (%)", line={"color": COLORS["C"]}))
            fig.add_trace(go.Scatter(x=t_dates, y=out.paths.iv_short[0] * 100, name="5y ATM vol used (%)", line={"color": "#888", "dash": "dot"}, yaxis="y2"))
            if len(trs):
                fig.add_trace(go.Scatter(x=trs.date, y=trs.prem_pct * 100, name="call premium paid (% notional)", mode="markers", marker={"size": 4, "color": "#d62728"}, yaxis="y2"))
            fig.update_layout(title="Market series used in this replay", template="plotly_white", height=360, legend={"orientation": "h"}, yaxis2={"overlaying": "y", "side": "right", "title": "vol %"})
            st.plotly_chart(fig, width="stretch")
        st.markdown(("" if meta.get("use_mkt") else
            f"- Lombard cost: **{(meta['lombard'] if meta['lombard'] is not None else mk.loan_rate):.2%}** p.a. {'(placeholder)' if meta['lombard'] is None else ''}\n"
            f"- Cash yield / risk-free rate: **{(meta['cash'] if meta['cash'] is not None else mk.cash_rate):.2%}** p.a. {'(placeholder)' if meta['cash'] is None else ''}\n"
            f"- {cfg.options.tenor_years:g}y ATM call premium on SPXFP: **{prem_used:.2%}** of notional {prem_note}; delta {delta_used:.3f}\n")
            + f"- Inception: A equities {bs.loc['A', 'equities'] / M:,.0f} m / loan {bs.loc['A', 'loan'] / M:,.0f} m; B target book {cfg.options.target_total_notional / M:,.0f} m notional bought {cfg.options.notional_per_purchase / M:,.2f} m per {cfg.options.purchase_frequency[:-2]} (premium at the target ≈ {bs.loc['B', 'options_premium'] / M:,.0f} m); illiquids {bs.loc['A', 'illiquids'] / M:,.0f} m in all strategies.\n"
            f"- Loan: rolled monthly, interest capitalised into the principal (ACT/360); A's loan at the end: {res['A'].series('loan')[0, -1] / M:,.0f} m. LTVs: equities {cfg.leverage.ltv_base.equity_index:.0%}, illiquids {cfg.leverage.ltv_base.illiquid:.0%}, cash {cfg.leverage.ltv_base.cash:.0%}.\n"
            + (f"- Dry powder: B bought additional calls at {len(cfg.dry_powder.tiers)} drawdown tiers ({', '.join(f'{t.index_drawdown:.0%}→{t.deploy_pct_of_deployable:.0%}' for t in cfg.dry_powder.tiers)}); deployed {res['B'].events.dry_powder_deployed_usd[0] / M:,.0f} m in {int(res['B'].events.dry_powder_deployments[0])} deployments.\n" if cfg.dry_powder.enabled else "- Dry powder deployment: off.\n")
            + "- Illiquids: deterministic expected drift (HF 6 %, PE 10 %, Infra 8 % p.a. placeholders) with capital calls/distributions; identical across strategies."
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


with tab_bt:
    st.subheader("Backtested P&L of a 5-year ATM call on SPXFP vs. a cash investment in the index")
    st.caption("Every trading day is a hypothetical strike date: buy an ATM call on SPXFP for the premium below (notional = premium ÷ premium-% of the day) "
               "or invest the same amount in the index; both are read at the option's maturity. Plotted against the maturity date. "
               "Same construction as the J.P. Morgan / Bloomberg chart (their strikes: May-2006 → Aug-2021) with our own data: the premium is the one implied by the vol series and the 5-year Treasury of the day, not a dealer quote.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        bt_prem_usd = st.number_input("Premium / investment (USD m)", 1.0, 1000.0, 10.0, 1.0) * M
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
                   f"Add columns `ust_{int(round(bt_tenor))}y` and `iv_{int(round(bt_tenor))}y` to data/market_data_daily.csv for a tenor-consistent premium.")
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
