# USER_GUIDE.md

## The decision app
`.venv/bin/streamlit run app/historical_app.py --server.runOnSave true` — two pages, one sidebar.

**Sidebar (the setup).** Today: SPX exposure (USD m), Lombard loan (USD m), spread over SOFR (bp), lending value of the SPX (%). The rotation: call tenor (5 / 7 / 10 years), weeks to complete the rotation (52; 1 = all on the start date). Advanced: dividend withholding tax, whether a call that expires worthless is replaced (default: not replaced; a call that ends in the money is always replaced on the same index units, paid from the payoff, then cash, then SPX), what happens to a payoff left after a roll (T-bills / SPX / more calls), lending value of the calls.

### Page 1 — Any start date since 1997
The same two portfolios put on at every trading day (Advanced: every week, faster) from 9 Sep 1997 to the last start whose last weekly tranche still expires inside the data (for 5-year calls: start 22 Sep 2020, last strike 14 Sep 2021), each held to the day chosen in **Held until**: today (the last day of the data) or one of the market bottoms (9 Oct 2002, 9 Mar 2009, 23 Mar 2020, 12 Oct 2022; then the last start is the one whose rotation was complete a month before the bottom). A button downloads every start's results as CSV (USD m; a copy for the default setup is in `reports/any_start_date_since_1997_daily.csv`).
1. **Every start date's path to today**: one line per start on the calendar, each leaving from 750 m on its own start day and running to the last day of the data (month-end NAV, linear scale so the lowest levels are visible), one chart per portfolio, with the distribution of the final NAV drawn vertically on the right edge of each chart on the same scale. One colour per start year (red for the loan, blue for the rotation; light for the oldest, strong for the newest, newer years drawn in front, colour bar on the right). The start-year dropdown restricts the lines and everything below to one or several years (empty = all). On the rotation chart the starts that stopped buying calls after a correction are yellow to orange with the same light-to-strong logic (second colour bar), with an orange profile on the right. With the trading-day grid every fifth start is drawn.
2. **Final NAV: statistics over the selected starts**: 5th / 25th / 50th / 75th / 95th percentile, mean, standard deviation, lowest and highest final NAV for each portfolio and the difference; worst and best start dates; the same-start comparison (share of starts where the rotation ends ahead, and its final NAV relative to keeping the loan).
3. **The lowest points**: the corrections of 20 % or more in the SPX price index since 1997 (previous peak and its level, bottom and its level, fall, recovery); at each bottom the worst trajectory of the selected starts running that day (lowest NAV, its start date, fall, LTV or dry powder) and how far its start was from the market peak (days, SPX level vs the peak).\n4. **Dry powder and the distance from a margin call**, rotating or not: at each market bottom and on the end day, for the median and the worst of the selected starts running that day — the levered portfolio's LTV, the further SPX fall that would trigger the margin call and its remaining dry powder; the rotation's dry powder available after the fall; and the delta. Two computed lines: margin calls and the closest one; share of starts where the rotation has more dry powder at its lowest point.
5. **Final value on the end day, by start date** and two computed lines (calls lost, annualised medians, margin calls and least room / capacity).

### Exporting the trajectories
`.venv/bin/python scripts/export_trajectories.py` writes every start's trajectory to one long CSV, one row per start and month-end (`--sample W` for week-ends), with NAV, dry powder and the balance sheet of both portfolios in USD m and SPX with dividends rebased to 1 at the start. Default: weekly starts (`reports/trajectories_weekly_monthly.csv`, ≈ 250,000 rows, 26 MB, one minute); `--grid daily` runs every trading-day start (≈ 1.2 M rows, five minutes). The setup options mirror the sidebar (`--help`).

### Page 2 — Call premium history
The call-vs-cash backtest behind the premiums, with its own inputs. Inputs: premium / investment (USD m), call tenor (years), first and last strike date (defaults 1997-09-09 → 2021-08-31), premium mode (*market*: implied vol and Treasury of the day for the tenor; *fixed*: one % of notional), cash leg (SPXFP, or SPX with dividends reinvested net of the withholding tax you enter).

Output: (1) P&L of the call (payoff − premium) and of the cash investment against the **maturity date**, one point per strike day; (2) bold header stating exactly what was computed (tenor, leg, number of strike dates, premium mode); (3) summary table (average / median / best / worst P&L, share of losing strike dates), hit rate, share of calls expiring worthless, premium and notional actually bought; (4) **distribution of the holding-period return** on the same USD committed: overlaid histogram and percentile table (5th…95th, mean, std, probability of a loss, probability of losing everything, mean annualised).

Caveats shown on the page: strikes are daily so consecutive observations overlap almost entirely (a range of outcomes, not independent draws); no bid/ask, no early unwind; the 5-year vol is extrapolated from the 24-month Bloomberg series (VIX proxy before May 2005). At 7 and 10 years the page uses `ust_7y`/`iv_7y` and `ust_10y`/`iv_10y` (built by `scripts/build_long_tenor_columns.py`; the 7-year yield is interpolated until a `USGG7YR` export is supplied, the vols are extrapolated as at 5 years); for any other tenor without its own columns it warns and falls back to the 5-year series.

## Supporting apps (strategy simulator, kept for reference)
* **Historical strategy replay** — `.venv/bin/streamlit run app/strategy_replay_app.py`: pick capital, allocation, leverage, start date and Strategy B choices; leave the three historical inputs blank (Lombard cost, cash yield, LT option premium) or fill them; enter each strategy's starting balance sheet (equities / loan / cash, common illiquids) in the table, choose how B accumulates its option book (notional bought per week or month, target book, tenor), and press *Run historical replay* to see what A, B and C would have done through the actual S&P 500 history (weekly or monthly steps). No Monte Carlo.
* **Full simulator (9 tabs)** — `.venv/bin/streamlit run app/streamlit_app.py`: Monte Carlo, risk, sensitivity, stress, audit (SPEC §10).

## Install and run
```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                       # full validation suite
.venv/bin/python -m fosim.reporting.validation_report   # → reports/validation_report.html
.venv/bin/streamlit run app/streamlit_app.py     # UI
```
Programmatic use:
```python
from fosim.config import load_config
from fosim.runner import run_config, with_overrides
cfg = load_config("config/default.yaml")
out = run_config(with_overrides(cfg, {"run.n_paths": 5000}), ledger_paths=[0])
out.results["B"].nav            # (paths, steps+1) NAV array
```

## Configuration (`config/default.yaml`)
Every number is a **PLACEHOLDER**. The schema (`src/fosim/config/schema.py`) rejects unknown keys, non-USD currency fields, FX/quanto inputs, inconsistent weights and tenor grids. Key blocks:

| Block | What it controls |
|---|---|
| `run` | paths, horizon, `dt` (monthly/weekly/daily), seed, label |
| `portfolio`, `leverage`, `loan_terms` | NAV, split, leverage ratio and allocation, LTVs and stress schedule, thresholds, slippage, Lombard term sheet (SOFR convention, floor, tiers, fees, limit, fixed/swapped) |
| `held_equity_portfolio`, `equity_indices`, `equity_model` | A's actual book (betas, tracking error, alpha); indices (P-measure returns, realised dividends, underlying type, pricing dividend curve, jump sizes); GBM / Merton / coupled stoch-vol / bootstrap |
| `implied_vol`, `pricing` | log-OU short vol, term structure θ_T and β(T), skew, floor; price sources (dealer quotes, surface CSV, parametric, overrides, historical IV), bid/ask, funding spread, commissions, arbitrage check |
| `options`, `exposure`, `dry_powder` | tenor, strike and sizing modes, ladder mode, build-up/hold months, resize rule, IV pause, extra slots, counterparty; exposure policies and target; dry-powder tiers, reserve, exit rule |
| `illiquids`, `illiquids_marking` | HF/PE/Infra returns, smoothing, reporting frequency, TA cash-flow model, crisis multipliers; which marks feed margin and risk metrics |
| `correlation`, `costs`, `spending`, `analytics` | driver correlation matrix, trading costs, spending, CRRA γ, percentiles, VaR levels |

### Data files
* **Bootstrap / replay**: CSV with `date` and level/return columns; map roles in `bootstrap.columns` (`{SPX: px_last, IV: vix, RATES: sofr}`). Provided: `data/SPX_daily.csv`, `data/SPX_monthly.csv`, `data/SPXFP_daily.csv`, `data/SPXFP_monthly.csv` (from the user's Bloomberg export; no market data is hard-coded).
* **Vol surface**: CSV `tenor, moneyness (K/F), iv` or `tenor, delta, iv`; calendar and butterfly arbitrage are checked (`pricing.arbitrage_check`).
* **Curve**: CSV `tenor, rate` (zero or par; `rates.curve_type`).
* **Dealer quotes**: `pricing.dealer_quotes: [{index, tenor, strike_pct_spot, bid_pct, ask_pct | bid_iv, ask_iv, quote_date, spot, rate, dividend}]`.

## UI tabs
1. **Assumptions** — edit/load/save YAML, upload data, quick edits, timestamped changelog.
2. **Inception** — balance sheets, sizing × strike table, put–call parity, carry, d*.
3. **Simulation** — run (sidebar), NAV fans, terminal and B−A distributions, summary metrics with SEs, dominance, lenses.
4. **Risk** — drawdowns, utilisation fans, VaR/CVaR, margin/liquidation/ruin statistics.
5. **Liquidity & dry powder** — cash fans, illiquid flows, deployment, LCR.
6. **Option book & exposure** — static breakeven chart, participation profile, delta/value grid, tranche table with price source, Greeks, mid vs bid, exposure vs A, Greek explain.
7. **Sensitivity** — tornado, heatmaps, breakeven solver (common random numbers).
8. **Stress tests** — editable stylised templates (labelled not historical) and historical replay from user files, with event markers.
9. **Audit** — ledger viewer, Excel single-path export, validation report.
The sidebar builds the **IC report** (HTML; print to PDF from the browser).

## Reading the results
* `notional_match` B has ≈ 64% of A's dollar-delta at the reference inputs; compare on the equal-delta lens before concluding.
* Monthly steps understate margin-call frequency; daily runs (fewer paths) are the reference for margin-risk statements.
* Sharpe/Sortino use smoothed illiquid marks; `nav_true` series use true marks.
