# LIMITATIONS.md

Read together with the model-risk disclaimer in every IC report.

1. **Pricing is Black–Scholes with a parametric skew**, not market quotes. Inception premiums must be validated against dealer indications (`pricing.source: dealer_quotes`). Long-dated index options are OTC/illiquid; bid/ask in stress can be far wider than the configured multiples, and unwind prices are model marks at bid.
2. **Counterparty and CSA terms** are reduced to an optional default intensity and recovery; collateral, netting and dealer downgrade effects are not modelled.
3. **Lombard terms are bank-specific and discretionary.** LTVs, stress haircuts, thresholds, spreads, floors and the facility limit are placeholders; banks can cut advance rates or the facility at will. Facility renewal re-pricing (`renewal_years`, `renewal_spread_stress`) is not applied by the v1 engine.
4. **Appraisal-based illiquid marks** (Geltner smoothing) understate true volatility and correlation; the official NAV uses reported marks (config switch), risk metrics can use true marks. Fund cash-flow timing follows a stylised Takahashi–Alexander model.
5. **Monthly stepping understates margin risk**: intra-month drawdowns and calls are invisible. Use `run.dt: daily` (fewer paths) for margin-risk conclusions; the UI warns.
6. **All-USD simplification.** FX risk versus the family office's CHF reference currency is deliberately ignored (no FX module exists by design).
7. **Taxes** are limited to dividend withholding and optional stamp duty; treatment of option gains, capital gains and the tax status of the structures is out of scope.
8. **Model risk in jump and vol dynamics**: Merton jumps with a common clock, log-OU implied vol with a term-structure elasticity and linear skew, and a linear implied–realised coupling are stylised. Historical bootstrap depends entirely on the user's data window.
9. **Monte Carlo sampling error**: all headline metrics carry standard errors; tornado/breakeven results at small path counts are noisy (the breakeven solver flags unreliable results).
10. **No behavioural or governance constraints** (e.g. committee reaction to drawdowns, changes in spending, discretionary de-risking).
11. **Basket options, American exercise, quanto features** are out of scope (v1 is European vanilla on USD indices).
12. **Stylised stress scenarios are not historical data**; historical replay uses only user-supplied files and a single path without re-randomisation.
13. **PDF export** of the IC report is via the browser's print-to-PDF; the tool writes HTML.
14. **Performance**: 10,000 paths × 10 years monthly × 1 index runs in ≈ 25 s on an M2 laptop; daily runs and multi-index books scale roughly with steps × indices × tranches.
15. **Long-dated implied vol is extrapolated, not observed.** Bloomberg SPX ATM implied vols stop at 24 months (from May 2005); the 5-year series is the 24-month series raised to a fitted elasticity (VIX proxy before 2005). Backing the premium out of the J.P. Morgan chart suggests dealer 5-year ATM premiums were ≈ 25 % higher than ours at the 2009 and 2013 troughs, so the call-vs-cash backtest overstates the call P&L for those vintages. 7- and 10-year backtests need `ust_7y`/`ust_10y` and, above all, 7y/10y ATM implied vol (or dated dealer quotes to calibrate the extrapolation); until then they run on the 5-year columns and the tab says so.
16. **Overlapping observations.** Daily strike dates over ~24 years give ≈ 5 independent 5-year windows; percentiles of the return distribution describe the range of historical outcomes, not a sampling distribution.
