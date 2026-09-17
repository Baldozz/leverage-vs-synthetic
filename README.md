# Keep the loan, or rotate into calls — historical decision tool (USD only)

**The question.** Today: 1 bn in SPX with a 250 m Lombard loan on it (lending value 75 %, SOFR + 75 bp, interest
capitalised). Option 1, **Keep the loan**: nothing changes. Option 2, **Rotate into calls**: week by week sell SPX,
buy long-dated (5-year default) ATM calls on SPXFP sized by the model delta so the SPX-equivalent exposure stays
1 bn, and repay the loan with the proceeds net of premium; the loan is gone after 52 weeks. A call that expires in
the money is replaced by a new ATM call on the same index units (paid from the payoff, then cash, then SPX); a call
that expires worthless lapses. Historical data only, September 1997 to today.

**The app** — `app/historical_app.py`, three pages, one sidebar (the setup above, every number editable):
1. **Keep the loan or rotate** (`app/views/rotation.py`): both portfolios from one start date, held to today — the
   rotation (SPX sold, notional, premium, loan repaid, week by week), how the two evolve (NAV vs SPX, what each
   holds, room before a margin call vs capacity to borrow), the rolls by year. Engine
   `src/fosim/analytics/leverage_stress.py` (daily accounting identity asserted), tests `tests/test_leverage_stress.py`,
   formulas `docs/METHODOLOGY.md` §9, choices `docs/ASSUMPTIONS.md` 19p, limits `docs/LIMITATIONS.md` 17.
2. **Any start date since 1997** (`app/views/all_starts.py`): the same from every trading day (or week) since 1997 to the
   last start whose calls have expired, each held to today — final value by start date and its distribution, and whether the
   rotated portfolio lost its calls (`leverage_stress.rolling_starts`).
3. **Call premium history** (`app/views/premium_history.py`, own inputs): for every trading day since
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
7y Treasury (interpolated 5y/10y until a `USGG7YR` export is supplied, flag `ust_7y_interpolated`) and extrapolated `iv_7y` / `iv_10y`,
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
