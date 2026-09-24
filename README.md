# Keep the loan, or rotate into calls — historical decision tool (USD only)

**The question.** An illustrative portfolio (default inputs, all editable): 1 bn in SPX with a 250 m Lombard loan on it (lending value 75 %, SOFR + 75 bp, interest
capitalised). Option 1, **Keep the loan**: nothing changes. Option 2, **Rotate into calls**: week by week sell SPX,
buy long-dated (5-year default) ATM calls on SPXFP sized by the model delta so the SPX-equivalent exposure stays
1 bn, and repay the loan with the proceeds net of premium; the loan is gone after 52 weeks. At every expiry the
replacement closes the gap between Keep's SPX value and the rotation's live exposure (its SPX plus the calls' dollar delta),
paid from the payoff and the T-bills, then by selling SPX; every quarter-end the exposure is brought back inside ±10 % of
Keep's (calls sold, most in the money first, at the mark less 1 vol point; calls bought from the T-bills). The earlier rules
stay in the sidebar: a new call on the same dollar delta (about twice the units) or on the same index units, a worthless call
replaced (or lapsing), SPX sold delta-for-delta, a replacement nobody can fund cut to what can be. Historical
data only, September 1997 to today.

**The app** — `app/historical_app.py`, two pages, each with its own sidebar (the call tenor and the dividend withholding tax are
common to both). Page 1's sidebar is the setup above, every number editable; the page runs only when its **Launch simulation** button is
pressed (sidebar, grid, start year and end day), and the button is greyed out until something changes (Advanced: withholding tax, what an expiring call is replaced on, the band check's frequency, width, haircut and what it buys, whether a
worthless call is replaced, what a payoff left after a roll buys, the lending values of the calls and of the T-bills, the start-date grid); page 2's sidebar
holds the backtest's inputs (premium, premium mode, strike window, cash leg):
1. **Historical simulation** (`app/views/all_starts.py`): both portfolios put on at every trading day (or week) from
   9 Sep 1997 to a week before the end day, each held to the day chosen in *Held until* — today, or one of the market bottoms (Oct 2002, Mar 2009,
   Mar 2020, Oct 2022); the starts still rotating on the end day run with the tranches bought so far and are counted; held to today, the starts whose
   calls have not all expired are counted too (their NAV is a model mark). The page shows: the fan of every trajectory (one colour
   per start year, light to strong, the rotations that stopped buying calls after a correction in yellow to orange) with
   the distribution of the final NAV drawn vertically on the right edge, with a zoom slider; the NAV on the end day by start date — Keep, Rotate and the net Rotate − Keep on the
   same start, one point per start with the market peaks and bottoms marked and each portfolio's lowest point labelled with the other's value on that start, and the same
   chart for dry powder (Keep's room before a margin call with its LTV as yellow-to-orange dots on a right axis, Rotate's dry powder, the net); on the end day, the distribution of the annualised return over the selected starts
   (lowest, percentiles, highest; Δ = Rotate − Keep on each row, plus the count of starts on which the rotation is ahead on the same start); the corrections of 20 % or more in the SPX
   price index (peak, bottom, recovery, the lowest NAV of each portfolio on the bottom day); one tab per market bottom (sections Worst trajectory and Dry powder, shaded section rows; text tables that wrap), Keep the loan vs Rotate into calls with the delta —
   the worst trajectory in detail — the start with the lowest NAV that day in either portfolio, both strategies read on that same start with a Δ on every row (when it
   started and how far from the peak, NAV, SPX, T-bills, calls at market value with the number alive, loan, lending value — 75 % of the
   SPX, 0 % of the calls, 90 % of the T-bills by default — dry powder as lending value − loan, LTV and the further fall to a margin call,
   the interest cumulated on each loan; dividends are reinvested in the SPX in both portfolios and footnoted) and the number of starts on which the rotation has more dry powder that day; the final value by start date; a CSV download of every start's results. Engine
   `src/fosim/analytics/leverage_stress.py` (`simulate` with the daily accounting identity asserted, `rolling_starts`,
   `rolling_paths` with the bottoms sampled exactly, `corrections`, `starts_table`, `paths_table`), tests
   `tests/test_leverage_stress.py` and `tests/test_app.py` (which cross-checks the bottom tables against direct
   simulations), formulas `docs/METHODOLOGY.md` §9, choices `docs/ASSUMPTIONS.md` 19p, limits `docs/LIMITATIONS.md` 17.
   `scripts/export_trajectories.py` writes every start's trajectory (NAV, dry powder, balance sheet, month-end samples)
   to a long CSV under `reports/` (generated files, not committed).
2. **Call premium history** (`app/views/premium_history.py`): for every trading day since
September 1997 an at-the-money call on **SPXFP** (S&P 500 futures excess-return index) maturing 5 years later
is bought for a fixed USD amount (default 10 m; notional = premium ÷ premium-% of the day) and, alternatively,
the same amount is invested in the index. Both are read at the option's maturity and plotted against the
maturity date (same construction as the J.P. Morgan / Bloomberg chart), with the distribution of the
holding-period return of the two legs. The premium of the day is the Black–Scholes ATM price at the 5-year
implied vol and the 5-year Treasury of that day (q = r for an excess-return index) — see
`docs/ASSUMPTIONS.md` 19l/19m for how the 5-year vol series is built from the 24-month Bloomberg series.
Engine: `src/fosim/analytics/call_vs_cash.py`, tests `tests/test_call_vs_cash.py`, formulas `docs/METHODOLOGY.md` §8.

The Black–Scholes pricer behind both pages is verified independently in `tests/test_bsm_reference.py` (Hull's
values, 2,000 random cases against a scipy reference, put–call parity, finite-difference Greeks, implied-vol round
trip, Black-76 equivalence for q = r).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/streamlit run app/historical_app.py --server.runOnSave true   # the app
.venv/bin/python -m pytest                                              # full suite
.venv/bin/ruff check . && .venv/bin/mypy                                # lint / strict typing
```

**Data** (`data/market_data_{daily,weekly,monthly}.csv`, built from the Bloomberg export `BBG_Other Data.xlsx`
aligned to the SPX/SPXFP history): SPX, SPXFP, 3m LIBOR / Term SOFR, 3m T-bill, Fed funds, 5y Treasury,
VIX / VIX3M / VIX6M / VIX1Y, SPX ATM implied vol 12/18/24m (from May 2005), trailing S&P dividend yield;
derived `iv_5y` (extrapolated), `loan_base` (LIBOR→SOFR splice); 10y Treasury (`USGG10YR`, from `10yr for calls analysis.xlsx`),
7y Treasury (`USGG7YR`, from `7Y UST.xlsx`, observed from 26 Feb 2009; interpolated 5y/10y before that, flag `ust_7y_interpolated`) and extrapolated `iv_7y` / `iv_10y`,
all added by `scripts/build_long_tenor_columns.py` (Assumptions 19n). The page uses the tenor's own columns and warns when it has to
fall back to the 5-year ones (e.g. a 3-year tenor).

**Supporting material** (built first, kept for reference): the strategy simulator comparing a levered cash
equity portfolio on a Lombard facility (A) with a laddered book of 5-year calls plus dry powder (B) and an
unlevered benchmark (C) — historical replay in `app/strategy_replay_app.py`, Monte Carlo in
`app/streamlit_app.py` (spec `SPEC.md`, plan `docs/PLAN.md`, conventions `CLAUDE.md`, methodology
`docs/METHODOLOGY.md`, assumptions `docs/ASSUMPTIONS.md`, limitations `docs/LIMITATIONS.md`, guide `docs/USER_GUIDE.md`).

**Every default number in the config files is a PLACEHOLDER**; the backtest uses observed market series, but
the long-dated implied vol is an extrapolation, not dealer quotes — validate against live indications before
any decision. For illustrative purposes only.
