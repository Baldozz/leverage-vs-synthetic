# PLAN.md — Leverage vs. long-dated call replacement simulator

Status: confirmed by the user on 2026-09-15 ("keep everything open — all values, percentages and
indices adjustable; start"). Every ambiguity below is therefore resolved as **a config parameter
whose default is the value the spec names as default**. Nothing is hard-wired.

## 1. Understanding of the problem

A family office with NAV ≈ USD 1,000m (70% listed equities / 30% illiquids) borrows 25% of NAV on a
Lombard facility (Strategy A). It is considering repaying the loan and replacing direct equity
exposure with a monthly ladder of 5-year at-the-money index calls, holding the unspent premium as
dry powder (Strategy B). The tool must answer: is B feasible and preferable to A on return,
drawdown, margin-call / forced-sale risk, liquidity and cost of carry — under the same simulated
market paths, with sensitivities and stress tests — and make every economic driver visible
(exposure gap because Δ < 1, gamma drift, vega, rho, dividends, roll costs, put–call parity view).

Strategy C (unlevered, no options) separates the effect of leverage from the effect of options;
Strategy D (A plus a loan-funded cash buffer) tests whether a buffer alone fixes margin risk.

## 2. Ambiguities and resolutions (all configurable, defaults as listed)

| # | Item | Default | Config key | Note |
|---|------|---------|-----------|------|
| 1 | Allocation of borrowed funds | `pro_rata` (equities 875, illiquids 375, loan 250) | `leverage.allocation_of_borrowed_funds` | `equity_only` → 950/300/250. Illiquid amount in B and C equals A's. |
| 2 | Monthly ladder | `rolling_ladder`, M = H = 12, T₀ = 5y | `options.ladder_mode`, `buildup_months`, `hold_months_before_roll`, `tenor_years` | `bullet`, `monthly_buildup_hold_to_expiry` also available. |
| 3 | Sizing / exposure target | `notional_match` | `options.sizing_mode`, `options.sizing_param`, `exposure.target` | All six modes implemented and shown side by side on the Inception tab. |
| 4 | Strike mode & delta definition | `atm_spot`, BSM delta | `options.strike_mode`, `strike_param`, `delta_definition` | `delta_target` uses the selected delta definition inside the root-finder. |
| 5 | Exposure maintenance | `static` | `exposure.policy` | `rebalance_bands`, `restrike_on_move`, `spot_topup`, `futures_overlay` (red flag). |
| 6 | Option price sources | `parametric` | `pricing.source`, `dealer_quotes`, `vol_surface_file`, `scenario_overrides`, `historical_iv_file` | No dealer quotes exist yet: all vol levels are PLACEHOLDERs. Precedence per §4.2b. |
| 7 | Underlying type; held portfolio | `price_return`; held portfolio β = 1, TE = 3%, α = 0 | `equity_indices[].underlying_type`, `held_equity_portfolio` | Test 38 covers the β = 1 / TE = 0 identity. |
| 8 | Lombard terms | SOFR, monthly reset, floor 0%, one spread tier 100 bp, no fees, limit 400m, floating | `loan_terms.*` | All PLACEHOLDERs pending the bank term sheet. Tier utilisation = loan / facility limit (documented). |
| 9 | Illiquid marks for margin | **reported** (smoothed) | `illiquids_marking.margin_uses`, `.risk_metrics_use` | Official NAV uses reported marks; economic risk metrics use true marks. |
| 10 | Cure grace period vs. step | 2 days < monthly step ⇒ uncured call liquidated at the same step on monthly runs | `leverage.cure_grace_days` | Daily runs honour the grace period in steps. UI warns on monthly runs. |
| 11 | Zero-vol validation scenario | realised vol 0, jumps off, implied vol constant and positive | test 21 | Keeps options priceable while paths are deterministic. |
| 12 | NAV mark of the option book | mid; ask − mid on purchase and mid − bid on sale booked as explicit transaction cost | — | NAV at bid also reported. |
| 13 | Distribution rate cap in Takahashi–Alexander | RD capped at 100% once age > life | `illiquids[].ta_model` | |
| 14 | Historical data files | `data/SPX_daily.csv`, `data/SPXFP_daily.csv` (from the user's Excel) | `bootstrap.file`, `stress.historical_file` | Used only when the user selects bootstrap / historical replay. Series semantics (price vs. futures/excess return) to be confirmed by the user. |
| 15 | Type checking | `mypy --strict` on `src/fosim`, `validation/`; app checked by ruff only | `pyproject.toml` | Streamlit stubs are not strict-clean. |

## 3. Module plan (build order = spec §12.3)

1. `engine/conventions.py` — time grid, day counts, return conversions. `config/schema.py` — pydantic v2 schema, USD guard, cross-field validation. `config/default.yaml`.
2. `pricing/black_scholes.py`, `pricing/greeks.py`, `pricing/implied_vol.py`, `pricing/vol_surface.py` (parametric surface, term structure, skew, bid/ask, dealer-quote calibration, surface-file loader with arbitrage checks, overrides, precedence).
3. `market/correlation.py`, `rates.py`, `equity.py`, `jumps.py`, `implied_vol.py`, `illiquids.py`, `bootstrap.py`, `stress.py`, `paths.py` (immutable `MarketPaths`), `generator.py` (seed spawning, one stream per factor).
4. `instruments/lombard_loan.py`, `cash.py`, `option_ladder.py`, `futures_overlay.py`.
5. `strategies/base.py`, `levered.py` (A), `call_replacement.py` (B), `unlevered.py` (C), `levered_buffer.py` (D); `engine/simulator.py` (order of operations §5.5, accounting identity), `engine/ledger.py`, `engine/events.py`.
6. `validation/independent_recalc.py` — scalar, loop-based, shares no code with `fosim`.
7. `analytics/inception.py` (§6.2), `static_payoff.py` (§6.6), `metrics.py`, `attribution.py`, `exposure.py` (§6.5), `sensitivity.py`, `breakeven.py`, `dominance.py`, `lenses.py` (§5.7.9).
8. `reporting/excel_export.py`, `html_report.py` (IC report), `validation_report.py`.
9. `app/streamlit_app.py` — nine tabs per §10.
10. `docs/METHODOLOGY.md`, `ASSUMPTIONS.md`, `LIMITATIONS.md`, `USER_GUIDE.md`.

## 4. Test plan

`tests/test_NN_*.py` map one-to-one to spec §7 tests 1–40, plus property tests (hypothesis) on
put–call parity and accounting invariants, and a slow-marked 10,000-path identity run. Every
test file's docstring states the spec test number and its purpose, and
`reports/validation_report.html` is generated from the pytest results by one command.

## 5. Architecture

`Config → MarketScenarioGenerator → MarketPaths (frozen) → StrategyEngine × strategies →
Results + Ledger → Analytics → Reports/UI`. Fully vectorised across paths (`[n_paths, ...]`
arrays); option tranches live in fixed slot arrays `[n_paths, n_slots]` so path-dependent
tranche lifetimes remain vectorised. Random numbers: `SeedSequence(seed).spawn(k)` with a fixed
named order of factor streams; all streams are always drawn so toggling a factor never shifts
another factor's draws.
