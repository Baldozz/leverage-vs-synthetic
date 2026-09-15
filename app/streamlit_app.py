"""Streamlit UI (SPEC §10) — sober institutional layout, USD m, nine tabs, one-click IC report.

Run:  .venv/bin/streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import datetime as dt
import io
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

from fosim.analytics.attribution import (  # noqa: E402
    component_totals,
    greek_explain,
    realised_vs_implied_diagnostic,
)
from fosim.analytics.dominance import stochastic_dominance  # noqa: E402
from fosim.analytics.exposure import (  # noqa: E402
    delta_value_grid,
    exposure_fans,
    participation_profile,
    realised_capture,
)
from fosim.analytics.inception import (  # noqa: E402
    balance_sheets,
    carry_comparison,
    inception_market,
    margin_trigger_distance,
    put_call_parity_decomposition,
    sizing_table,
    strike_for_mode,
    tranche_terms,
)
from fosim.analytics.lenses import comparison_lenses  # noqa: E402
from fosim.analytics.metrics import drawdown_stats, summary_table  # noqa: E402
from fosim.analytics.sensitivity import (  # noqa: E402
    METRICS,
    breakeven,
    default_inputs,
    heatmap,
    tornado,
)
from fosim.analytics.static_payoff import (  # noqa: E402
    StaticPayoffInputs,
    breakevens,
    reading,
    v_a,
    v_b,
)
from fosim.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, SimConfig  # noqa: E402
from fosim.engine.state import COMPONENTS  # noqa: E402
from fosim.market.stress import (  # noqa: E402
    build_historical_replay,
    build_stylised_paths,
    load_templates,
)
from fosim.reporting.excel_export import export_path_audit  # noqa: E402
from fosim.reporting.html_report import build_ic_report  # noqa: E402
from fosim.runner import RunOutput, run_config, with_overrides  # noqa: E402
from fosim.strategies.base import inception_balance  # noqa: E402

M = 1e6
COLORS = {"A": "#1f4e79", "B": "#c55a11", "C": "#548235", "D": "#7030a0"}
st.set_page_config(page_title="Leverage vs. call replacement", layout="wide")


# ----------------------------------------------------------------------------- helpers
def money(x: float) -> str:
    return f"{x / M:,.1f}"


def fan_chart(t: np.ndarray, series: dict[str, np.ndarray], title: str, ytitle: str = "USD m", scale: float = M) -> go.Figure:  # type: ignore[type-arg]
    fig = go.Figure()
    for name, arr in series.items():
        c = COLORS.get(name, "#444")
        p5, p25, p50, p75, p95 = (np.percentile(arr, q, axis=0) / scale for q in (5, 25, 50, 75, 95))
        fig.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([p95, p5[::-1]]), fill="toself", fillcolor=c, opacity=0.10, line={"width": 0}, name=f"{name} 5–95%"))
        fig.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([p75, p25[::-1]]), fill="toself", fillcolor=c, opacity=0.18, line={"width": 0}, name=f"{name} 25–75%"))
        fig.add_trace(go.Scatter(x=t, y=p50, line={"color": c, "width": 2}, name=f"{name} median"))
    fig.update_layout(title=title, xaxis_title="years", yaxis_title=ytitle, template="plotly_white", height=420, legend={"orientation": "h"})
    return fig


def line_chart(t: np.ndarray, series: dict[str, np.ndarray], title: str, ytitle: str = "USD m", scale: float = M) -> go.Figure:  # type: ignore[type-arg]
    fig = go.Figure()
    for name, arr in series.items():
        fig.add_trace(go.Scatter(x=t, y=np.asarray(arr) / scale, name=name, line={"color": COLORS.get(name, None), "width": 2}))
    fig.update_layout(title=title, xaxis_title="years", yaxis_title=ytitle, template="plotly_white", height=380, legend={"orientation": "h"})
    return fig


def load_raw() -> dict[str, Any]:
    with open(DEFAULT_CONFIG_PATH) as f:
        d: dict[str, Any] = yaml.safe_load(f)
    return d


def get_cfg() -> SimConfig | None:
    try:
        return SimConfig.model_validate(yaml.safe_load(st.session_state["yaml_text"]))
    except Exception as e:
        st.error(f"Configuration invalid: {e}")
        return None


def log_change(msg: str) -> None:
    st.session_state.setdefault("changelog", []).append(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} — {msg}")


if "yaml_text" not in st.session_state:
    st.session_state["yaml_text"] = DEFAULT_CONFIG_PATH.read_text()
    log_change("loaded config/default.yaml (all numbers PLACEHOLDERS)")

# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Leverage vs. call replacement")
st.sidebar.caption("Decision-support simulator — USD only. Every default number is a PLACEHOLDER.")
n_paths_ui = st.sidebar.number_input("Paths for interactive runs", 100, 200_000, 2000, step=100)
ledger_path = st.sidebar.number_input("Audited path id", 0, 10_000, 0)
run_btn = st.sidebar.button("Run simulation", type="primary")
cfg = get_cfg()
if cfg is not None:
    if cfg.run.dt == "monthly":
        st.sidebar.warning("dt = monthly understates intra-month drawdowns and margin-call frequency. Run daily for margin-risk conclusions.")
    if cfg.exposure.policy == "futures_overlay":
        st.sidebar.error("FUTURES OVERLAY ENABLED: variation margin re-introduces margin calls into Strategy B, contradicting its objective.")
    if run_btn:
        with st.spinner("Simulating…"):
            bar = st.sidebar.progress(0.0)
            out = run_config(cfg, n_paths=int(n_paths_ui), ledger_paths=[int(ledger_path)], base_dir=PROJECT_ROOT, progress=lambda k, n: bar.progress(k / n))
            st.session_state["run"] = out
            bar.empty()
        if out.paths.correlation_repaired:
            st.sidebar.warning(f"Correlation matrix was not PSD and was repaired (Frobenius distance {out.paths.meta.get('frobenius_distance', 0):.3e}).")
        st.sidebar.success(f"Done: paths {out.timings['paths_seconds']:.1f}s, engine {out.timings['engine_seconds']:.1f}s")
out_run: RunOutput | None = st.session_state.get("run")
if st.sidebar.button("Build IC report (HTML)") and out_run is not None:
    p = build_ic_report(out_run, PROJECT_ROOT / "reports" / "ic_report.html", tornado_df=st.session_state.get("tornado"))
    st.sidebar.success(f"Written {p}")
    st.sidebar.download_button("Download IC report", p.read_bytes(), "ic_report.html", "text/html")

tabs = st.tabs(["Assumptions", "Inception", "Simulation", "Risk", "Liquidity & dry powder", "Option book & exposure", "Sensitivity", "Stress tests", "Audit"])

# ----------------------------------------------------------------------------- 1 Assumptions
with tabs[0]:
    st.subheader("Assumptions (editable YAML — every default is a PLACEHOLDER)")
    c1, c2 = st.columns([3, 1])
    with c1:
        txt = st.text_area("config YAML", st.session_state["yaml_text"], height=520, label_visibility="collapsed")
        if txt != st.session_state["yaml_text"]:
            st.session_state["yaml_text"] = txt
            log_change("YAML edited in the UI")
            st.rerun()
    with c2:
        up = st.file_uploader("Load YAML", type=["yaml", "yml"])
        if up is not None:
            st.session_state["yaml_text"] = up.read().decode()
            log_change(f"loaded YAML {up.name}")
            st.rerun()
        st.download_button("Save YAML", st.session_state["yaml_text"], "config.yaml", "text/yaml")
        data_up = st.file_uploader("Upload market-data CSV (bootstrap / replay / surface / curve)", type=["csv"])
        if data_up is not None:
            dest = PROJECT_ROOT / "data" / data_up.name
            dest.write_bytes(data_up.read())
            log_change(f"uploaded data file {dest.name}")
            st.success(f"Saved to {dest}")
        if cfg is not None:
            st.markdown("**Quick edits**")
            er = st.number_input("Equity expected return", -0.5, 0.5, float(cfg.equity_indices[0].expected_return), 0.005, format="%.3f")
            lev = st.number_input("Leverage (% NAV)", 0.0, 2.0, float(cfg.leverage.ratio_of_nav), 0.05)
            sizing = st.selectbox("Sizing mode", ["notional_match", "delta_match", "beta_adjusted_delta_match", "premium_budget", "cash_reserve_target", "delta_fraction_of_A"], index=["notional_match", "delta_match", "beta_adjusted_delta_match", "premium_budget", "cash_reserve_target", "delta_fraction_of_A"].index(cfg.options.sizing_mode))
            strike = st.selectbox("Strike mode", ["atm_spot", "atm_forward", "pct_spot", "pct_forward", "delta_target"], index=["atm_spot", "atm_forward", "pct_spot", "pct_forward", "delta_target"].index(cfg.options.strike_mode))
            if st.button("Apply quick edits"):
                raw_d = yaml.safe_load(st.session_state["yaml_text"])
                raw_d["equity_indices"][0]["expected_return"] = float(er)
                raw_d["leverage"]["ratio_of_nav"] = float(lev)
                raw_d["options"]["sizing_mode"] = sizing
                raw_d["options"]["strike_mode"] = strike
                if strike in ("pct_spot", "pct_forward") and not raw_d["options"].get("strike_param"):
                    raw_d["options"]["strike_param"] = 1.0
                if strike == "delta_target" and not (raw_d["options"].get("strike_param") and 0 < raw_d["options"]["strike_param"] < 1):
                    raw_d["options"]["strike_param"] = 0.5
                if sizing in ("premium_budget", "cash_reserve_target", "delta_fraction_of_A") and raw_d["options"].get("sizing_param") is None:
                    raw_d["options"]["sizing_param"] = {"premium_budget": 0.15, "cash_reserve_target": 0.35, "delta_fraction_of_A": 0.5}[sizing]
                st.session_state["yaml_text"] = yaml.safe_dump(raw_d, sort_keys=False)
                log_change(f"quick edits: ER={er}, leverage={lev}, sizing={sizing}, strike={strike}")
                st.rerun()
    st.markdown("**Assumptions changelog**")
    st.code("\n".join(st.session_state.get("changelog", [])))

# ----------------------------------------------------------------------------- 2 Inception
with tabs[1]:
    if cfg is not None:
        st.subheader("Inception balance sheets (USD m)")
        bs = balance_sheets(cfg)
        show = bs[["equities", "options_premium", "illiquids", "cash", "loan", "nav", "gross_equity_exposure", "dollar_delta"]] / M
        show["dollar_delta_pct_nav"] = bs["dollar_delta_pct_nav"]
        st.dataframe(show.style.format("{:,.1f}"), width="stretch")
        tt = bs.attrs["tranche"]
        st.caption(f"New 5y tranche at inception: strike {tt['strike_pct_spot']:.1%} of spot, vol {tt['vol']:.2%}, premium {tt['premium_pct']:.2%} of notional, BSM delta {tt['delta']:.4f} (smile delta {tt['delta_smile']:.4f}), elasticity {tt['elasticity']:.2f}.")
        st.subheader("Sizing × strike mode comparison (B)")
        siz = sizing_table(cfg)
        cols = ["strike_mode", "sizing_mode", "strike_pct_spot", "vol", "notional", "premium", "cash", "cash_pct_nav", "delta", "dollar_delta", "pct_of_A_exposure", "vega_1pt", "theta_year", "rho_100bp", "dq_100bp", "elasticity", "feasible"]
        sz = siz[cols].copy()
        for c in ("notional", "premium", "cash", "dollar_delta", "vega_1pt", "theta_year", "rho_100bp", "dq_100bp"):
            sz[c] = sz[c] / M
        st.dataframe(sz.style.format({c: "{:,.1f}" for c in ("notional", "premium", "cash", "dollar_delta", "vega_1pt", "theta_year", "rho_100bp", "dq_100bp")} | {"strike_pct_spot": "{:.3f}", "vol": "{:.2%}", "cash_pct_nav": "{:.1%}", "delta": "{:.3f}", "pct_of_A_exposure": "{:.1%}", "elasticity": "{:.2f}"}), width="stretch", height=400)
        st.warning("notional_match gives materially LOWER equity exposure than A because the 5y call delta is ≈ 0.64 at the reference inputs, not 0.50 or 1.")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Put–call parity decomposition")
            pcp = put_call_parity_decomposition(cfg)
            st.dataframe(pd.DataFrame({k: [v] for k, v in pcp.items()}).T.rename(columns={0: "value"}).style.format("{:.6g}"), width="stretch")
            st.caption("Long call + cash ≡ long equity + long put − forgone dividends, financed at the option-implied rate.")
        with c2:
            st.subheader("Expected carry (year 1, USD m)")
            st.dataframe((carry_comparison(cfg) / M).style.format("{:,.2f}"), width="stretch")
            st.subheader("Margin-trigger distance d*")
            st.json({k: f"{v:.2%}" for k, v in margin_trigger_distance(cfg).items()})

# ----------------------------------------------------------------------------- 3 Simulation
with tabs[2]:
    if out_run is None:
        st.info("Press **Run simulation** in the sidebar.")
    else:
        o = out_run
        t = o.paths.grid.t
        st.plotly_chart(fan_chart(t, {k: v.nav for k, v in o.results.items()}, "NAV fan chart (5/25/50/75/95%)"), width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for k, v in o.results.items():
                fig.add_trace(go.Histogram(x=v.nav[:, -1] / M, name=k, opacity=0.5, nbinsx=80, marker_color=COLORS.get(k)))
            fig.update_layout(barmode="overlay", title="Terminal NAV (USD m)", template="plotly_white", height=360)
            st.plotly_chart(fig, width="stretch")
        with c2:
            d = (o.results["B"].nav[:, -1] - o.results["A"].nav[:, -1]) / M
            fig = go.Figure(go.Histogram(x=d, nbinsx=80, marker_color="#7f7f7f"))
            fig.update_layout(title=f"NAV_B − NAV_A at horizon (USD m); P(B > A) = {(d > 0).mean():.1%}", template="plotly_white", height=360)
            st.plotly_chart(fig, width="stretch")
        summ = summary_table(o.results, o.paths, o.cfg.analytics.percentiles, o.cfg.analytics.var_levels, o.cfg.analytics.crra_gamma, o.cfg.spending.pct_nav_pa, o.cfg.dry_powder.liquidity_reserve.capital_call_months)
        st.subheader("Summary metrics (USD m where monetary; SE = Monte Carlo standard error)")
        disp = summ.copy()
        for r in disp.index:
            if any(s in r for s in ("nav", "diff", "ce", "volume", "loss", "usd", "cash_at_trough_mean", "cash_at_trough_p5", "transition")) and "pct" not in r and "p_" != r[:2]:
                disp.loc[r] = disp.loc[r] / M
        st.dataframe(disp.style.format("{:,.4g}"), width="stretch", height=600)
        st.caption("cagr_median / cagr_mean are statistics of per-path CAGRs; cagr_of_mean_nav is the CAGR of the mean terminal NAV — not the same thing. Sharpe/Sortino use smoothed illiquid marks and are biased upwards.")
        sd = stochastic_dominance(o.results["B"].nav[:, -1], o.results["A"].nav[:, -1])
        st.markdown(f"**Stochastic dominance (B vs A):** FSD B≻A: {sd['x_fsd_y']}, A≻B: {sd['y_fsd_x']}; SSD B≻A: {sd['x_ssd_y']}, A≻B: {sd['y_ssd_x']}; max FSD violation {sd['max_fsd_violation']:.3f}.")
        with st.expander("Fair comparison lenses (equal capital / equal delta / equal vol) — reruns the simulation"):
            if st.button("Compute lenses"):
                with st.spinner("Running lenses…"):
                    lens = comparison_lenses(o.cfg, min(int(n_paths_ui), 3000))
                for k, v in lens.items():
                    st.markdown(f"**{k}** — {v['note']}")
                    if v["table"] is not None:
                        tbl = v["table"].copy()
                        for r in ("median_terminal_nav", "mean_terminal_nav", "p5", "p95", "dollar_delta0"):
                            tbl.loc[r] = tbl.loc[r] / M
                        st.dataframe(tbl.style.format("{:,.4g}"), width="stretch")

# ----------------------------------------------------------------------------- 4 Risk
with tabs[3]:
    if out_run is not None:
        o = out_run
        t = o.paths.grid.t
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for k, v in o.results.items():
                mdd = drawdown_stats(v.nav, o.paths.grid.dt)["max_dd_per_path"]
                fig.add_trace(go.Histogram(x=np.asarray(mdd) * 100, name=k, opacity=0.5, nbinsx=60, marker_color=COLORS.get(k)))
            fig.update_layout(barmode="overlay", title="Maximum drawdown per path (%)", template="plotly_white", height=360)
            st.plotly_chart(fig, width="stretch")
        with c2:
            margin_strats = [k for k, v in o.results.items() if v.meta.get("uses_margin")]
            st.plotly_chart(fan_chart(t, {k: o.results[k].series("utilisation") for k in margin_strats}, "Margin utilisation (loan / lending value)", "utilisation", 1.0), width="stretch")
        summ = summary_table(o.results, o.paths, o.cfg.analytics.percentiles, o.cfg.analytics.var_levels, o.cfg.analytics.crra_gamma, o.cfg.spending.pct_nav_pa, o.cfg.dry_powder.liquidity_reserve.capital_call_months)
        risk_rows = [r for r in summ.index if r.startswith(("var_", "cvar_", "p_margin", "p_forced", "p_ruin", "max_dd", "forced_sale", "slippage", "min_headroom", "p_liquidity", "time_under", "futures_margin", "ann_vol"))]
        st.dataframe(summ.loc[risk_rows].style.format("{:,.4g}"), width="stretch", height=520)
        st.caption("VaR/CVaR are losses as returns (positive = loss) on 1-year and horizon returns. Ruin = NAV ≤ 0 (path frozen). Long options carry no margin: B's margin statistics are zero unless the futures overlay is enabled.")

# ----------------------------------------------------------------------------- 5 Liquidity
with tabs[4]:
    if out_run is not None:
        o = out_run
        t = o.paths.grid.t
        st.plotly_chart(fan_chart(t, {k: v.series("cash") for k, v in o.results.items()}, "Cash (USD m)"), width="stretch")
        il = o.paths.illiquids
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(line_chart(t, {"capital calls (mean)": il.contributions.sum(axis=2).mean(axis=0), "distributions (mean)": il.distributions.sum(axis=2).mean(axis=0), "unfunded (mean)": il.unfunded.sum(axis=2).mean(axis=0)}, "Illiquid cash flows per step and unfunded commitments (mean, USD m)"), width="stretch")
        with c2:
            b = o.results["B"]
            st.plotly_chart(fan_chart(t, {"B": b.series("spot_mv")}, "B: deployed spot index (dry powder), USD m"), width="stretch")
        summ = summary_table(o.results, o.paths, o.cfg.analytics.percentiles, o.cfg.analytics.var_levels, o.cfg.analytics.crra_gamma, o.cfg.spending.pct_nav_pa, o.cfg.dry_powder.liquidity_reserve.capital_call_months)
        dp_rows = [r for r in summ.index if r.startswith(("cash_at_trough", "p_deployed", "deployed_usd", "roi_on", "lcr", "p_cash_constrained"))]
        st.dataframe(summ.loc[dp_rows, ["B"]].style.format("{:,.4g}"), width="stretch")
        st.caption("Liquidity coverage ratio = cash / (unfunded commitments expected to be called in 12 months + 12 months of spending). Cash at trough = cash at the step of the deepest option-index drawdown on each path.")

# ----------------------------------------------------------------------------- 6 Option book
with tabs[5]:
    if cfg is not None:
        st.subheader("Static hold-to-expiry payoff (closed form, continuous compounding — SPEC §6.6)")
        mk = inception_market(cfg)
        bal = inception_balance(cfg.portfolio.nav0, cfg.portfolio.weights.equities, cfg.portfolio.weights.illiquids, cfg.leverage.ratio_of_nav, cfg.leverage.allocation_of_borrowed_funds)
        K = strike_for_mode(mk, cfg.options.strike_mode, cfg.options.strike_param, cfg.options.delta_definition)
        tt = tranche_terms(mk, K)
        siz = sizing_table(cfg)
        fig = go.Figure()
        S_T = np.linspace(0.3 * mk.S0, 2.2 * mk.S0, 400)
        lines = []
        for zm in ("notional_match", "delta_match"):
            row = siz[(siz.strike_mode == cfg.options.strike_mode) & (siz.sizing_mode == zm)].iloc[0]
            x = StaticPayoffInputs(E0=bal["equity_A"], L=bal["loan"], S0=mk.S0, K=K, N=float(row.notional), c=float(row.premium_pct_notional), T=mk.T, r=mk.cash_rate, r_L=np.log1p(mk.loan_rate), q=np.log1p(cfg.equity_indices[0].dividend_yield))
            fig.add_trace(go.Scatter(x=S_T / mk.S0, y=v_b(x, S_T) / M, name=f"B ({zm})"))
            be = breakevens(x)
            lines.append(f"**{zm}**: breakevens at {', '.join(f'{b / mk.S0:.1%}' for b in be) or 'none'} of spot — {reading(x, be)}")
            if zm == "notional_match":
                fig.add_trace(go.Scatter(x=S_T / mk.S0, y=v_a(x, S_T) / M, name="A (equity sleeve − loan)", line={"color": COLORS["A"]}))
        fig.update_layout(title="Value at expiry vs. index level (equity sleeve only, USD m)", xaxis_title="S_T / S_0", yaxis_title="USD m", template="plotly_white", height=420)
        st.plotly_chart(fig, width="stretch")
        for ln in lines:
            st.markdown(ln)
        st.subheader("Participation profile (static inception book, MTM at horizons)")
        prof = participation_profile(cfg)
        fig = go.Figure()
        for h in sorted(prof.horizon.unique()):
            d = prof[prof.horizon == h]
            fig.add_trace(go.Scatter(x=d.index_level, y=d.V_B / M, name=f"B {h:g}y", line={"dash": "solid"}))
            fig.add_trace(go.Scatter(x=d.index_level, y=d.V_A / M, name=f"A {h:g}y", line={"dash": "dot", "color": COLORS["A"]}))
        fig.update_layout(template="plotly_white", height=400, xaxis_title="index level / S_0", yaxis_title="USD m")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame(prof.attrs["capture"]).T.style.format("{:.2f}"), width="stretch")
        st.subheader("Delta and value grid (inception tranche: index × implied vol × rate shocks)")
        idx_sh = np.array([-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5])
        vol_sh = np.array([-5.0, 0.0, 5.0, 10.0, 20.0])
        rate_sh = st.select_slider("Rate shock (bp)", [-200, -100, 0, 100, 200], value=0)
        row = siz[(siz.strike_mode == cfg.options.strike_mode) & (siz.sizing_mode == cfg.options.sizing_mode)].iloc[0]
        tr0 = pd.DataFrame([{"units": float(row.notional) / mk.S0, "strike": K, "residual_years": mk.T, "vol_mid": float(row.vol)}])
        g = delta_value_grid(tr0, mk.S0, mk.r, mk.q, idx_sh, vol_sh, np.array([float(rate_sh)]))
        st.markdown("Book P&L (USD m):")
        st.dataframe(pd.DataFrame(g["pnl"][:, :, 0] / M, index=[f"{x:+.0%}" for x in idx_sh], columns=[f"{v:+.0f} pts" for v in vol_sh]).style.format("{:,.1f}"), width="stretch")
        st.markdown("Dollar delta (USD m):")
        st.dataframe(pd.DataFrame(g["dollar_delta"][:, :, 0] / M, index=[f"{x:+.0%}" for x in idx_sh], columns=[f"{v:+.0f} pts" for v in vol_sh]).style.format("{:,.1f}"), width="stretch")
    if out_run is not None:
        o = out_run
        t = o.paths.grid.t
        b = o.results["B"]
        st.subheader("Simulated option book")
        step = st.slider("Step for the tranche table (audited path)", 0, o.paths.n_steps, min(12, o.paths.n_steps))
        tr = b.ledger.tranche_frame()
        if len(tr):
            tab = tr[(tr.step == step) & (tr.path == int(ledger_path))].copy()
            for c in ("units", "cost_usd", "notional_usd"):
                tab[c] = tab[c] / M if c != "units" else tab[c]
            st.dataframe(tab[["slot", "index", "origin", "strike", "spot", "moneyness_S_over_K", "residual_years", "vol_mid", "vol_at_purchase_ask", "source", "units", "cost_usd", "notional_usd", "purchase_step", "expiry_step"]].style.format({"strike": "{:.2f}", "spot": "{:.2f}", "moneyness_S_over_K": "{:.3f}", "residual_years": "{:.2f}", "vol_mid": "{:.2%}", "vol_at_purchase_ask": "{:.2%}", "units": "{:,.0f}", "cost_usd": "{:,.2f}", "notional_usd": "{:,.1f}"}), width="stretch")
            st.caption("origin: 0 ladder, 1 rebalance, 2 dry powder, 3 restrike, 4 exit rule. 'source' is the option price source used for the mark.")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fan_chart(t, {"B option book (mid)": b.series("option_mv"), "B option book (bid)": b.series("nav_bid") - b.nav + b.series("option_mv")}, "Option book MTM: mid vs bid (USD m)"), width="stretch")
        with c2:
            st.plotly_chart(fan_chart(t, {"B": b.series("equity_exposure"), "A": o.results["A"].series("equity_exposure"), "C": o.results["C"].series("equity_exposure")}, "Effective equity exposure: A (equity MV) vs B (dollar delta incl. spot) vs C"), width="stretch")
        fans = exposure_fans(o.results, o.paths, o.cfg)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(line_chart(t, {"B % of A exposure (median)": fans["B_pct_of_A_exposure"]["p50"].to_numpy(), "A effective leverage (median)": fans["A_effective_leverage"]["p50"].to_numpy()}, "B's exposure as % of A vs A's effective leverage (medians)", "ratio", 1.0), width="stretch")
        with c2:
            st.plotly_chart(line_chart(t, {"vega per vol pt": fans["B_vega_1pt"]["p50"].to_numpy(), "theta per year": fans["B_theta_year"]["p50"].to_numpy(), "rho per 100bp": fans["B_rho_100bp"]["p50"].to_numpy(), "div sens per 100bp": fans["B_dq_100bp"]["p50"].to_numpy()}, "B aggregate Greeks (medians, USD m)"), width="stretch")
        ge = greek_explain(b, o.paths, o.cfg)
        tot = ge.attrs["totals"]
        st.markdown("**Greek-based P&L explain of the option book (horizon totals, mean over paths, USD m):** " + ", ".join(f"{k} {v / M:,.1f}" for k, v in tot.items()) + f" — residual share of |actual| = {ge.attrs['residual_share_of_abs_actual']:.1%}")
        diag = realised_vs_implied_diagnostic(b, o.paths, o.cfg)
        rc = realised_capture(b, o.paths)
        st.markdown(f"Realised vs implied vol diagnostic ½Γ S²(σ_r² − σ_i²)dt: {diag['gamma_vs_theta_pnl_mean'] / M:,.1f} m (realised {diag['realised_vol_mean']:.1%} vs 5y implied {diag['implied_5y_vol_mean']:.1%}). Realised capture: up-beta {rc['up_beta']:.2f}, down-beta {rc['down_beta']:.2f}. Roll cost (bid/ask + commissions) mean {b.events.roll_cost_usd.mean() / M:,.1f} m.")

# ----------------------------------------------------------------------------- 7 Sensitivity
with tabs[6]:
    if cfg is not None:
        st.subheader("One-at-a-time tornado (common random numbers)")
        metric_name = st.selectbox("Metric", list(METRICS))
        n_sens = st.number_input("Paths per sensitivity run", 200, 20000, 1000, 100)
        all_inputs = default_inputs(cfg)
        chosen = st.multiselect("Inputs", [i.label for i in all_inputs], default=[i.label for i in all_inputs[:8]])
        if st.button("Run tornado"):
            bar = st.progress(0.0)
            df = tornado(cfg, [i for i in all_inputs if i.label in chosen], METRICS[metric_name], int(n_sens), progress=lambda k, n: bar.progress(k / n))
            st.session_state["tornado"] = df
            bar.empty()
        if "tornado" in st.session_state:
            df = st.session_state["tornado"]
            fig = go.Figure()
            fig.add_trace(go.Bar(y=df.input, x=df.BminusA_low - df.BminusA_base, orientation="h", name="low", marker_color="#c55a11"))
            fig.add_trace(go.Bar(y=df.input, x=df.BminusA_high - df.BminusA_base, orientation="h", name="high", marker_color="#1f4e79"))
            fig.update_layout(barmode="overlay", title=f"Δ({metric_name}: B − A) vs base", template="plotly_white", height=500)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(df.style.format("{:,.4g}"), width="stretch")
        st.subheader("2-D heatmap")
        pairs = {"equity ER × 5y implied vol": ("equity_indices.0.expected_return", [0.04, 0.06, 0.08, 0.10], "implied_vol.term_structure.theta", [[0.17, 0.18, 0.19], [0.21, 0.22, 0.23], [0.25, 0.26, 0.27]]), "spread × crash size": ("loan_terms.spread_tiers.0.spread", [0.005, 0.01, 0.02], "equity_indices.0.jump_mean", [-0.30, -0.15, -0.05]), "base rate × implied vol": ("rates.usd.r0", [0.01, 0.03, 0.05], "implied_vol.term_structure.theta", [[0.17, 0.18, 0.19], [0.21, 0.22, 0.23], [0.25, 0.26, 0.27]])}
        pair = st.selectbox("Pair", list(pairs))
        if st.button("Run heatmap"):
            px, xs, py, ys = pairs[pair]
            bar = st.progress(0.0)
            hm = heatmap(cfg, px, xs, py, ys, METRICS[metric_name], int(n_sens), progress=lambda k, n: bar.progress(k / n))
            bar.empty()
            fig = go.Figure(go.Heatmap(z=hm["BminusA"].to_numpy() / (M if "nav" in metric_name or "ce" in metric_name else 1), x=hm["BminusA"].columns, y=hm["BminusA"].index, colorscale="RdBu", zmid=0))
            fig.update_layout(title=f"{metric_name}: B − A", xaxis_title=px, yaxis_title=py, template="plotly_white", height=420)
            st.plotly_chart(fig, width="stretch")
        st.subheader("Breakeven solver (value of an input at which metric_A = metric_B)")
        be_in = st.selectbox("Input", [i for i in all_inputs if not isinstance(i.low, list)], format_func=lambda i: i.label)
        c1, c2 = st.columns(2)
        lo = c1.number_input("bracket low", value=float(be_in.low), format="%.4f")
        hi = c2.number_input("bracket high", value=float(be_in.high), format="%.4f")
        if st.button("Solve breakeven"):
            with st.spinner("Solving…"):
                res = breakeven(cfg, be_in.path, lo, hi, METRICS[metric_name], int(n_sens))
            if res.value is None:
                st.warning(res.message)
            else:
                st.success(f"Breakeven {be_in.label} = {res.value:.4g} (95% CI {res.ci_low:.4g} – {res.ci_high:.4g}); reliable: {res.reliable}. {res.message}")

# ----------------------------------------------------------------------------- 8 Stress
with tabs[7]:
    if cfg is not None:
        st.subheader("Stylised stress scenarios — NOT historical data")
        templates = load_templates()
        name = st.selectbox("Template", list(templates))
        sc = templates[name]
        c = st.columns(4)
        dd = c[0].number_input("Drawdown", -0.95, 0.0, float(sc.drawdown), 0.05)
        mtt = c[1].number_input("Months to trough", 1, 120, int(sc.months_to_trough))
        shape = c[2].selectbox("Recovery", ["V", "U", "L"], index=["V", "U", "L"].index(sc.recovery_shape))
        mtr = c[3].number_input("Months to recover", 1, 240, int(sc.months_to_recover))
        c = st.columns(4)
        ivp = c[0].number_input("IV peak", 0.1, 1.5, float(sc.iv_peak), 0.05)
        ivd = c[1].number_input("IV decay half-life (months)", 1, 60, int(sc.iv_decay_months))
        rsh = c[2].number_input("Rate shock (bp)", -600.0, 600.0, float(sc.rate_shock_bp), 25.0)
        ltvm = c[3].number_input("LTV multiplier at trough", 0.3, 1.0, float(sc.ltv_multiplier), 0.05)
        if st.button("Run stylised scenario"):
            sc2 = sc.model_copy(update={"drawdown": dd, "months_to_trough": int(mtt), "recovery_shape": shape, "months_to_recover": int(mtr), "iv_peak": ivp, "iv_decay_months": int(ivd), "rate_shock_bp": rsh, "ltv_multiplier": ltvm})
            sres = build_stylised_paths(cfg, sc2, base_dir=PROJECT_ROOT)
            cfg_s = with_overrides(cfg, {k: v for k, v in sres.overrides.items() if not k.startswith("stress.")}) if sres.overrides else cfg
            so = run_config(cfg_s, ledger_paths=[0], paths=sres.paths)
            st.session_state["stress"] = (sres.label, so)
        st.subheader("Historical replay (user data only)")
        files = sorted(p.name for p in (PROJECT_ROOT / "data").glob("*.csv"))
        hf = st.selectbox("File", files, index=files.index("SPX_monthly.csv") if "SPX_monthly.csv" in files else 0)
        start = st.text_input("Start date (YYYY-MM-DD)", "2007-10-31")
        col = st.text_input("Index column", "px_last")
        if st.button("Run historical replay"):
            try:
                hres = build_historical_replay(cfg, PROJECT_ROOT / "data" / hf, start, {cfg.index_names[0]: col}, base_dir=PROJECT_ROOT)
                so = run_config(cfg, ledger_paths=[0], paths=hres.paths)
                st.session_state["stress"] = (hres.label, so)
            except Exception as e:
                st.error(str(e))
        if "stress" in st.session_state:
            label, so = st.session_state["stress"]
            st.markdown(f"**{label}**")
            t = so.paths.grid.t
            fig = go.Figure()
            for k, v in so.results.items():
                fig.add_trace(go.Scatter(x=t, y=v.nav[0] / M, name=f"NAV {k}", line={"color": COLORS.get(k)}))
            fig.add_trace(go.Scatter(x=t, y=so.paths.S[0, :, 0] / so.paths.S[0, 0, 0] * so.results["C"].nav[0, 0] / M, name="index (rebased)", line={"dash": "dot", "color": "#999"}))
            for k, v in so.results.items():
                led = v.ledger.to_frame()
                ev = led[led.account == "event"]
                for _, r in ev.iterrows():
                    fig.add_vline(x=t[int(r.step)], line={"color": COLORS.get(k, "#000"), "dash": "dash", "width": 1}, annotation_text=f"{k}: {r.description[:14]}", annotation_position="top")
            fig.update_layout(title="Side-by-side NAV paths with event markers (USD m)", template="plotly_white", height=460)
            st.plotly_chart(fig, width="stretch")
            rows = []
            for k, v in so.results.items():
                nav = v.nav[0]
                rows.append({"strategy": k, "terminal NAV (m)": nav[-1] / M, "min NAV (m)": nav.min() / M, "max drawdown": nav.min() / np.maximum.accumulate(nav).max() - 1, "margin calls": int(v.events.margin_calls[0]), "forced liquidations": int(v.events.forced_liquidations[0]), "slippage loss (m)": v.events.slippage_loss[0] / M, "dry powder deployed (m)": v.events.dry_powder_deployed_usd[0] / M, "ruin": bool(v.events.ruin[0])})
            st.dataframe(pd.DataFrame(rows).style.format("{:,.3g}"), width="stretch")

# ----------------------------------------------------------------------------- 9 Audit
with tabs[8]:
    if out_run is not None:
        o = out_run
        strat = st.selectbox("Strategy", list(o.results))
        led = o.results[strat].ledger.to_frame()
        st.dataframe(led, width="stretch", height=420)
        st.markdown("**P&L components, horizon totals (mean over paths, USD m)**")
        st.dataframe((component_totals(o.results) / M).style.format("{:,.2f}"), width="stretch")
        buf = io.BytesIO()
        tmp = PROJECT_ROOT / "reports" / f"audit_path_{int(ledger_path)}.xlsx"
        if st.button("Export single-path audit to Excel"):
            export_path_audit(o.results, o.paths, int(ledger_path), tmp)
            buf.write(tmp.read_bytes())
            st.download_button("Download audit workbook", buf.getvalue(), tmp.name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.markdown(f"Validation report: `python -m fosim.reporting.validation_report` → `reports/validation_report.html` ({'exists' if (PROJECT_ROOT / 'reports' / 'validation_report.html').exists() else 'not generated yet'}).")
        st.caption("Components: " + ", ".join(COMPONENTS))
    else:
        st.info("Run a simulation first.")
