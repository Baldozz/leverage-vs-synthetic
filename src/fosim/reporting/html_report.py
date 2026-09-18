"""One-click investment-committee report (SPEC §10): HTML with an executive summary, methodology
summary, key inputs, results with confidence intervals, stress tests, sensitivity, limitations and a
model-risk disclaimer. PDF: open the HTML in a browser and print to PDF (no PDF engine dependency).
"""

from __future__ import annotations

import datetime as dt
import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from fosim.analytics.attribution import component_totals
from fosim.analytics.inception import (
    balance_sheets,
    margin_trigger_distance,
    put_call_parity_decomposition,
    sizing_table,
)
from fosim.analytics.metrics import summary_table
from fosim.config.schema import SimConfig
from fosim.runner import RunOutput

USD_M = 1e6


def fmt_m(x: float) -> str:
    return f"{x / USD_M:,.1f}"


def fmt_pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def _table(df: pd.DataFrame, fmt: dict[str, Any] | None = None, index: bool = True) -> str:
    return df.to_html(border=0, classes="tbl", float_format=lambda v: f"{v:,.4g}", index=index, escape=True)


def nav_fan_figure(out: RunOutput) -> go.Figure:
    fig = go.Figure()
    t = out.paths.grid.t
    colors = {"A": "#1f4e79", "B": "#c55a11", "C": "#548235", "D": "#7030a0"}
    for name, res in out.results.items():
        p5, p50, p95 = (np.percentile(res.nav, q, axis=0) / USD_M for q in (5, 50, 95))
        c = colors.get(name, "#444")
        fig.add_trace(go.Scatter(x=t, y=p50, name=f"{name} median", line={"color": c, "width": 2}))
        fig.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([p95, p5[::-1]]), fill="toself", fillcolor=c, opacity=0.12, line={"width": 0}, name=f"{name} 5–95%", showlegend=True))
    fig.update_layout(title="NAV fan chart (USD m)", xaxis_title="years", yaxis_title="USD m", template="plotly_white", height=420)
    return fig


def terminal_hist_figure(out: RunOutput) -> go.Figure:
    fig = go.Figure()
    for name, res in out.results.items():
        fig.add_trace(go.Histogram(x=res.nav[:, -1] / USD_M, name=name, opacity=0.5, nbinsx=80))
    fig.update_layout(barmode="overlay", title="Terminal NAV distribution (USD m)", template="plotly_white", height=380)
    return fig


def build_ic_report(
    out: RunOutput,
    out_file: str | Path,
    stress_results: dict[str, RunOutput] | None = None,
    tornado_df: pd.DataFrame | None = None,
    extra_sections: dict[str, str] | None = None,
) -> Path:
    cfg: SimConfig = out.cfg
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    summ = summary_table(out.results, out.paths, cfg.analytics.percentiles, cfg.analytics.var_levels, cfg.analytics.crra_gamma, cfg.spending.pct_nav_pa, cfg.dry_powder.liquidity_reserve.capital_call_months)
    bs = balance_sheets(cfg)
    pcp = put_call_parity_decomposition(cfg)
    dstar = margin_trigger_distance(cfg)
    comp = component_totals(out.results)
    siz = sizing_table(cfg)
    A, B = summ["A"], summ["B"]
    n = out.paths.n_paths
    exec_lines = [
        f"Monte Carlo: {n:,} paths × {cfg.run.horizon_years:g} years, {cfg.run.dt} steps, seed {cfg.run.seed}. "
        f"Equity model: {cfg.equity_model.type}; rates: {cfg.rates.model}; option price source: {cfg.pricing.source}.",
        f"Median terminal NAV: A {fmt_m(A['terminal_nav_median'])} m (±{fmt_m(A['terminal_nav_median_se'])} SE), "
        f"B {fmt_m(B['terminal_nav_median'])} m (±{fmt_m(B['terminal_nav_median_se'])} SE), C {fmt_m(summ['C']['terminal_nav_median'])} m.",
        f"P(B terminal NAV > A) = {fmt_pct(B['p_B_beats_A'])} (±{fmt_pct(B['p_B_beats_A_se'])}); mean difference B − A = {fmt_m(B['diff_mean'])} m (±{fmt_m(B['diff_se'])}).",
        f"A: probability of ≥1 margin call {fmt_pct(A['p_margin_call'])}, of ≥1 forced liquidation {fmt_pct(A['p_forced_liquidation'])}, of ruin {fmt_pct(A['p_ruin'], 2)}; "
        f"margin-trigger distance at inception d* = {fmt_pct(dstar['d_star_base'])} of equity value.",
        f"B: no margin exposure; maximum loss on the option book = premium paid; inception dollar-delta {fmt_m(out.results['B'].inception.get('dollar_delta0', 0))} m "
        f"vs A {fmt_m(out.results['A'].inception.get('dollar_delta0', 0))} m; cash at inception {fmt_m(out.results['B'].inception.get('cash0', 0))} m.",
        f"Max drawdown (mean): A {fmt_pct(A['max_dd_mean'])}, B {fmt_pct(B['max_dd_mean'])}; CVaR 95% horizon: A {fmt_pct(A['cvar_95_horizon'])}, B {fmt_pct(B['cvar_95_horizon'])}.",
        f"CRRA certainty equivalent (γ = {cfg.analytics.crra_gamma:g}): A {fmt_m(A['crra_ce'])} m, B {fmt_m(B['crra_ce'])} m.",
    ]
    css = (
        "body{font-family:Helvetica,Arial,sans-serif;margin:32px;color:#1a1a1a;max-width:1200px}h1,h2{color:#1f4e79}"
        ".tbl{border-collapse:collapse;font-size:12px}.tbl th,.tbl td{border:1px solid #ddd;padding:4px 8px;text-align:right}.tbl th{background:#f3f6fa}"
        ".warn{background:#fff4e5;border-left:4px solid #c55a11;padding:8px 12px}.box{background:#f7f9fc;padding:10px 14px;border-radius:4px}"
        "li{margin:4px 0}"
    )
    parts = [
        f"<html><head><meta charset='utf-8'><title>Investment committee report — leverage vs. call replacement</title><style>{css}</style></head><body>",
        "<h1>Leverage vs. long-dated call replacement — investment committee report</h1>",
        f"<p>Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Config label: <b>{html.escape(cfg.run.label)}</b>. All amounts USD m.</p>",
        "<div class='warn'><b>All default numeric inputs are PLACEHOLDERS.</b> Expected returns, volatilities, implied-vol levels, Lombard terms and LTVs must be replaced by current market data, dealer quotes and the bank's term sheet before any decision. "
        "Results are Monte Carlo estimates with the stated standard errors; monthly stepping understates intra-month drawdowns and margin-call frequency.</div>",
        "<h2>1. Executive summary</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in exec_lines) + "</ul>",
        "<h2>2. Methodology summary</h2><div class='box'><p>Strategies A (levered cash equities), B (unlevered 5-year call ladder with dry powder), C (unlevered benchmark) and optionally D run on identical simulated market paths (common random numbers). "
        "Real-world (P) dynamics: equity indices (GBM / Merton jumps / coupled stochastic vol), log-OU short-dated implied vol with a term-structure elasticity and skew, flat / Vasicek / curve rates, illiquids with Geltner-smoothed marks and Takahashi–Alexander cash flows. "
        "Risk-neutral (Q) pricing: Black–Scholes–Merton with the surface vol at each tranche's current moneyness, the zero rate for the residual maturity and the pricing dividend curve; purchases at ask, marks at mid, sales at bid. "
        "Lombard mechanics: stress-dependent lending values, utilisation thresholds, cure waterfall (cash, then asset sales) and forced liquidation with stressed slippage. "
        "Every step asserts NAV_{t+dt} − NAV_t = Σ P&amp;L components + external flows. Full details: docs/METHODOLOGY.md.</p></div>",
        "<h2>3. Key inputs</h2>" + _table(pd.DataFrame({
            "input": ["NAV₀", "Leverage", "Allocation of borrowed funds", "Equity expected return", "Realised vol", "Dividend yield / WHT", "5y ATM implied vol (inception)", "Skew ψ(1y)", "Base rate", "Loan spread", "Cash spread", "Equity LTV / stress multiplier", "Option tenor / ladder", "Sizing / strike mode", "Exposure policy", "Dry-powder tiers", "Jump λ / μ_J / σ_J"],
            "value": [fmt_m(cfg.portfolio.nav0) + " m", fmt_pct(cfg.leverage.ratio_of_nav), cfg.leverage.allocation_of_borrowed_funds, fmt_pct(cfg.equity_indices[0].expected_return) + f" ({cfg.equity_indices[0].return_type})", fmt_pct(cfg.equity_indices[0].realized_vol),
                      f"{fmt_pct(cfg.equity_indices[0].dividend_yield)} / {fmt_pct(cfg.equity_indices[0].dividend_wht)}", fmt_pct(float(bs.attrs['tranche']['vol'])), f"{cfg.implied_vol.skew_1y:.2f}", fmt_pct(cfg.rates.usd.r0, 2), fmt_pct(cfg.loan_terms.spread_tiers[0].spread, 2), fmt_pct(cfg.rates.cash.spread, 2),
                      f"{cfg.leverage.ltv_base.equity_index:.2f} / {cfg.leverage.ltv_stress_schedule[0].multiplier if cfg.leverage.ltv_stress_schedule else 1:.2f}", f"{cfg.options.tenor_years:g}y / {cfg.options.ladder_mode} (M={cfg.options.buildup_months}, H={cfg.options.hold_months_before_roll})",
                      f"{cfg.options.sizing_mode} / {cfg.options.strike_mode}", cfg.exposure.policy, ", ".join(f"{t.index_drawdown:+.0%}→{t.deploy_pct_of_deployable:.0%}" for t in cfg.dry_powder.tiers if t.index_drawdown is not None),
                      f"{cfg.equity_model.jump_lambda:g} / {cfg.equity_indices[0].jump_mean:g} / {cfg.equity_indices[0].jump_vol:g}"],
        }), index=False),
        "<h2>4. Inception structure</h2>" + _table(bs.map(lambda v: v / USD_M if abs(v) > 100 else v)),
        "<p>Put–call parity view of one call unit: call " + fmt_pct(pcp['call_pct'], 2) + " of spot = PV(forward − strike) " + fmt_pct(pcp['embedded_financing_leg_pct'], 2) + " + put (insurance) " + fmt_pct(pcp['put_pct'], 2) + "; PV of forgone dividends " + fmt_pct(pcp['pv_forgone_dividends_pct'], 2) +
        f"; financing embedded at the option-implied rate {fmt_pct(pcp['implied_financing_rate'], 2)} vs Lombard {fmt_pct(pcp['lombard_loan_rate_simple'], 2)}; cash yield on unspent premium {fmt_pct(pcp['cash_yield_on_unspent_premium'], 2)}.</p>",
        "<h3>Sizing × strike mode table (B)</h3>" + _table(siz[["strike_mode", "sizing_mode", "strike_pct_spot", "vol", "notional", "premium", "cash", "delta", "dollar_delta", "pct_of_A_exposure", "vega_1pt", "theta_year", "rho_100bp", "dq_100bp", "feasible"]].assign(notional=lambda d: d.notional / USD_M, premium=lambda d: d.premium / USD_M, cash=lambda d: d.cash / USD_M, dollar_delta=lambda d: d.dollar_delta / USD_M, vega_1pt=lambda d: d.vega_1pt / USD_M, theta_year=lambda d: d.theta_year / USD_M, rho_100bp=lambda d: d.rho_100bp / USD_M, dq_100bp=lambda d: d.dq_100bp / USD_M), index=False),
        "<h2>5. Simulation results (with MC standard errors)</h2>",
        pio.to_html(nav_fan_figure(out), include_plotlyjs="cdn", full_html=False),
        pio.to_html(terminal_hist_figure(out), include_plotlyjs=False, full_html=False),
        _table(summ.loc[[r for r in summ.index if not r.endswith("_per_path")]].map(lambda v: v / USD_M if isinstance(v, float) and abs(v) > 1e4 else v)),
        "<h3>P&amp;L attribution (mean over paths, USD m, steps 1..N)</h3>" + _table(comp / USD_M),
    ]
    if stress_results:
        parts.append("<h2>6. Stress tests (stylised, not historical data)</h2>")
        rows = []
        for label, so in stress_results.items():
            for name, res in so.results.items():
                nav = res.nav[0]
                rows.append({"scenario": label, "strategy": name, "terminal_nav_m": nav[-1] / USD_M, "min_nav_m": nav.min() / USD_M, "max_drawdown": nav.min() / np.maximum.accumulate(nav).max() - 1 if nav.max() > 0 else np.nan, "margin_calls": int(res.events.margin_calls[0]), "forced_liquidations": int(res.events.forced_liquidations[0]), "ruin": bool(res.events.ruin[0])})
        parts.append(_table(pd.DataFrame(rows), index=False))
    if tornado_df is not None:
        parts.append("<h2>7. Sensitivity (one-at-a-time, common random numbers)</h2>" + _table(tornado_df.map(lambda v: v / USD_M if isinstance(v, float) and abs(v) > 1e4 else v), index=False))
    for title, body in (extra_sections or {}).items():
        parts.append(f"<h2>{html.escape(title)}</h2>{body}")
    parts.append(
        "<h2>Limitations and model-risk disclaimer</h2><div class='box'><ul>"
        "<li>Black–Scholes with a parametric skew is a pricing approximation; inception premiums must be validated against dealer indications, and long-dated OTC bid/ask can be far wider in stress.</li>"
        "<li>Lombard terms, LTVs and advance-rate cuts are bank-specific and discretionary; the modelled stress schedule is an assumption.</li>"
        "<li>Illiquid marks are appraisal-based (smoothed); economic risk is understated by reported marks.</li>"
        "<li>Monthly stepping understates intra-month drawdowns and margin-call frequency — run daily for margin-risk conclusions.</li>"
        "<li>All-USD simplification: FX risk versus a non-USD reference currency is deliberately ignored. Taxes beyond simple withholding and stamp-duty costs are not modelled.</li>"
        "<li>Model risk in jump and vol dynamics; Monte Carlo sampling error (see standard errors); no behavioural or governance constraints.</li>"
        "</ul><p>This report is decision support produced by a simulation model with placeholder defaults; it is not investment advice and must be read with docs/LIMITATIONS.md.</p></div></body></html>"
    )
    out_file.write_text("\n".join(parts), encoding="utf-8")
    return out_file
