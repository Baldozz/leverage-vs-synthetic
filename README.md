# Long-dated SPXFP call vs. cash investment — historical backtest (USD only)

**Primary analysis** — `app/historical_app.py` (a single page): for every trading day since
September 1997 an at-the-money call on **SPXFP** (S&P 500 futures excess-return index) maturing 5 years later
is bought for a fixed USD amount (default 10 m; notional = premium ÷ premium-% of the day) and, alternatively,
the same amount is invested in the index. Both are read at the option's maturity and plotted against the
maturity date (same construction as the J.P. Morgan / Bloomberg chart), with the distribution of the
holding-period return of the two legs. The premium of the day is the Black–Scholes ATM price at the 5-year
implied vol and the 5-year Treasury of that day (q = r for an excess-return index) — see
`docs/ASSUMPTIONS.md` 19l/19m for how the 5-year vol series is built from the 24-month Bloomberg series.
Engine: `src/fosim/analytics/call_vs_cash.py`, tests `tests/test_call_vs_cash.py`, formulas `docs/METHODOLOGY.md` §8.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/streamlit run app/historical_app.py --server.runOnSave true   # the backtest
.venv/bin/python -m pytest                                              # full suite
.venv/bin/ruff check . && .venv/bin/mypy                                # lint / strict typing
```

**Data** (`data/market_data_{daily,weekly,monthly}.csv`, built from the Bloomberg export `BBG_Other Data.xlsx`
aligned to the SPX/SPXFP history): SPX, SPXFP, 3m LIBOR / Term SOFR, 3m T-bill, Fed funds, 5y Treasury,
VIX / VIX3M / VIX6M / VIX1Y, SPX ATM implied vol 12/18/24m (from May 2005), trailing S&P dividend yield;
derived `iv_5y` (extrapolated), `loan_base` (LIBOR→SOFR splice). To run the backtest at 7 or 10 years with
tenor-consistent inputs, add columns `ust_7y` / `iv_7y` and `ust_10y` / `iv_10y` — the tab picks them up
automatically and warns when it has to fall back to the 5-year columns.

**Supporting material** (built first, kept for reference): the strategy simulator comparing a levered cash
equity portfolio on a Lombard facility (A) with a laddered book of 5-year calls plus dry powder (B) and an
unlevered benchmark (C) — historical replay in `app/strategy_replay_app.py`, Monte Carlo in
`app/streamlit_app.py` (spec `SPEC.md`, plan `docs/PLAN.md`, conventions `CLAUDE.md`, methodology
`docs/METHODOLOGY.md`, assumptions `docs/ASSUMPTIONS.md`, limitations `docs/LIMITATIONS.md`, guide `docs/USER_GUIDE.md`).

**Every default number in the config files is a PLACEHOLDER**; the backtest uses observed market series, but
the long-dated implied vol is an extrapolation, not dealer quotes — validate against live indications before
any decision. For illustrative purposes only.
