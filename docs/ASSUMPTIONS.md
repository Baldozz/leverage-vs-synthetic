# ASSUMPTIONS.md — modelling choices not fixed by SPEC.md

Every item is exposed as a config parameter where reasonable (key in brackets) and defaults to the
value the spec names as default. Items marked **(material)** change results materially and were
listed for confirmation in `docs/PLAN.md`; the user's instruction was "keep everything open".

## Balance sheet and strategies
1. **Allocation of borrowed funds** (material) — `leverage.allocation_of_borrowed_funds`, default `pro_rata` (equities 875 / illiquids 375 / loan 250). B and C hold exactly A's illiquid amount (illiquids cannot be traded), so C's equity sleeve is NAV₀ − illiquids (625m pro rata, 700m equity-only), not 70% of NAV₀.
2. **B's transition costs** — B starts from A's balance sheet and pays the configured equity trading cost (`costs.equity_bps`, plus stamp duty) on the sale of A's whole equity book. This is a real cost of switching and is reported as `transition_cost`; the §9 illustrative check uses zero costs.
3. **Ladder time origin** — t = 0 is treated as a month-end for strategy set-up: the first ladder tranche (or the bullet) is bought at t = 0. No spending is charged at t = 0.
4. **Sizing target re-evaluation** — `options.sizing_basis: usd` keeps the USD target fixed at inception values; `pct_nav` re-evaluates it as NAV_t/NAV₀ at each purchase. With `resize_on_roll: target_pct_nav` rolls use NAV_t/NAV₀ scaling even when `sizing_basis` is `usd`.
5. **Delta-based sizing uses the delta of a fresh tranche at the purchase date** (per index, book-weighted), under the configured `delta_definition`.
6. **`beta_adjusted_delta_match`** — because A and B hold identical illiquids, the illiquid beta cancels; the target reduces to A's equity exposure × β_held(book) / Δ.
7. **Cash constraint on purchases** — a tranche purchase is capped at the available cash (max(cash, 0)); the shortfall is flagged (`cash_constrained_purchases`). The core ladder ignores the dry-powder liquidity reserve; exposure-policy purchases and dry-powder deployments respect it.
8. **IV pause rule** — a skipped scheduled purchase stays pending and is retried at each later month-end until IV_ATM(5y) falls below the threshold.
9. **Held portfolio dynamics** — Δln H = Σβ_i Δln S_i + (α − σ_ε²/2)dt + σ_ε√dt ε, dividends = sleeve-weighted realised dividends of price-return indices. β = 1, σ_ε = 0, α = 0 reproduces the index exactly.
10. **`net_total_return` underlying** — the index reinvests dividends net of WHT; the price-index drift leaks q_c − ln(1 + q(1 − WHT)) and A receives no separate dividend cash. The pricing dividend curve is used as configured (the user should set it to the WHT leakage). *The spec's wording "pricing dividend = gross × (1 − WHT)" is ambiguous; confirm before using this mode.*

## Loan and margin
11. **Spread tiers** — `tier_basis: facility_utilisation` means loan ÷ facility limit (distinct from margin utilisation loan ÷ lending value); `loan_size` uses the USD loan. A tier applies when the basis is strictly below `utilisation_below`.
12. **Reset** — the contractual rate is fixed at month-ends where month % `reset_months` == 0 using the short rate at that date; `overnight_compounded` and `term` are treated identically at the reset dates (the difference is second order on monthly steps).
13. **Grace period vs. step** — `cure_grace_days` shorter than one step means an uncured call is cured by an investor sale (normal slippage) in the same step; longer grace periods give ⌊grace/step days⌋ steps before the bank liquidates at stressed slippage (`slippage_bps × slippage_stress_multiplier`). Closeout (`u ≥ u_closeout`) liquidates immediately.
14. **Cash cure** — repayment R from cash is solved so that (L − R)/(LV − ℓ_cash R) = u*.
15. **Cure sales sell held equities first, then deployed spot**; the basket LTV in the sale formula is the (stress-adjusted) equity LTV.
16. **Ruin** — NAV ≤ 0 freezes the path: all positions are liquidated at marks, cash is set to the (negative) NAV, no further accrual, flows or trades.
17. **Swap MTM** (`floating_swapped`) — annual-period pay-fixed swap valued off the zero curve with continuous discount factors (approximation), included in NAV.
18. **Facility renewal** — `renewal_years`/`renewal_spread_stress` are stored but the v1 engine does not re-price or cut the facility at renewal (see LIMITATIONS).

## Market models
19a. **Default equity model = stationary block bootstrap of the user's S&P 500 history** (`data/SPX_SPXFP_aligned_monthly.csv`, 1997-09 → 2026-09, index `SPX`), chosen on the user's instruction on 2026-09-15 to use the Excel workbooks for the equity index. Rows are resampled jointly with the SPXFP series (futures excess-return index; the SPX − SPXFP gap tracks financing minus dividends). Monthly log-returns are re-centred to the configured `expected_return` (still a PLACEHOLDER assumption); `realized_vol` is set to the empirical 15.4% (annualised monthly). Switch `equity_model.type` to `merton`/`gbm`/`coupled_stochvol` for parametric paths; a daily run needs the daily file (`bootstrap.frequency` must equal `run.dt`).
19. **Correlation** — one matrix over [indices…, illiquids…, IV, RATES]; the held-portfolio tracking-error shock is independent by construction. If the block is omitted it is built from the `rho_equity` fields (indices mutually uncorrelated, with a warning).
20. **Coupled stochastic vol** uses σ_t = IV_short,t (1 − vrp) at the start of the step; jumps combine additively.
21. **Illiquid jump beta** — β_J × book-weighted equity log jump, with its own compensator so the expected return is unchanged.
22. **Geltner smoothing on a unit growth index** — NAV_rep = NAV_true · I_obs,last / I_true (flat between reports, jumps at reports; flows enter both marks exactly). Chosen over "smooth the flow-adjusted NAV" because the latter is ill-defined when a fund distributes its whole NAV.
23. **Takahashi–Alexander** — RD_t = min((age/L)^B, 1) per year converted to per step; crisis multipliers use the start-of-step book drawdown; new-commitment pacing adds pacing × NAV₀ per year to unfunded commitments.
24. **Vasicek term premium** — additive `term_premium` on zero rates, phased in linearly from τ = 0 to τ = 1y (default 0 ⇒ P = Q for rates).
25. **Curve mode** — par bootstrap assumes annual coupons and whole-year pillars; log-DF linear interpolation; flat zero-rate extrapolation beyond the last pillar.
26. **Dividend stress** — `dividend_stress_cut` applies to realised dividends (and is intended for the pricing curve) when the book drawdown ≤ `dividend_stress_trigger` (default −25%).
27. **Stylised stress paths** — geometric decline to the trough, V (linear) / U (smoothstep) / L (partial) recovery in level space; IV linear to the peak then exponential decay with the given half-life; rate shock linear to the trough, held thereafter; illiquid markdown spread evenly over the decline; expected drift resumes after the recovery window (`post_scenario_drift`).

## Two-index set-up and historical replay (added 2026-09-15 on the user's instruction)
19b. **Held book vs option underlying.** A and C hold the **SPX price index** (`weight_in_equity_sleeve: 1`, dividends paid in cash net of WHT); B's options are written on **SPXFP** (S&P 500 futures excess-return index, `underlying_type: excess_return`, `option_book_weight: 1`). An excess-return index has a flat forward, so calls are priced with q = r (Black-76) and it pays no dividends; its `expected_return` is the excess return itself (PLACEHOLDER 3 %). `held_equity_portfolio.betas` define the held book's composition (SPX with β = 1); `beta_to_option_book` (empirical 0.988) translates A's equity exposure into SPXFP-delta units for exposure comparisons and `beta_adjusted_delta_match`.
19c. **Historical replay** (`equity_model.type: historical_replay`, `config/historical_1997.yaml`): one deterministic path from `historical.start` for `run.horizon_years`; the held book is taken from `historical.held_column` (SPX exactly, no tracking error); implied vol is constant at its placeholder level unless an IV column is supplied; illiquids follow their expected drift with the TA cash-flow model. Three historical inputs are left blank for the user (all-in Lombard cost, cash yield = risk-free rate, LT option premium % of notional); blank → model placeholders, flagged in the UI.
19d. **Premium-% overrides** are converted to an implied vol once, at S = 100, K = strike_pct_spot × 100 with the option tenor's inception rate (and q = r for an ER underlying); that vol level then drives marks and rolls for the covered months (skew still applied).

19e. **Custom balance sheets** (`custom_inception`, written by the historical app's table): each strategy starts with the entered equities / loan / cash (illiquids common); B then has no transition trades; B's sizing modes use `reference_equity_exposure` (default A's equities). If B is given a loan, margin mechanics apply to it with long options contributing zero lending value.
19f. **Fixed-notional purchase schedule** (`options.ladder_mode: fixed_notional_schedule`): every purchase step (`purchase_frequency` weekly → every weekly step, needs `run.dt: weekly`; monthly → month-ends) B buys `notional_per_purchase` of fresh calls (strike per `strike_mode`) while the active book is below `target_total_notional`; tranches are held to expiry and cash-settled, and the freed notional is re-bought. A cash-constrained purchase cut below 1 % of the requested notional is skipped and flagged instead of creating a dust tranche. The slot pool is sized to one purchase per step over the option's life (262 for weekly 5-year), so partial purchases can never overflow it. On the weekly grid a "month" is 52/12 steps, so month-end actions (dry powder, spending) drift slightly against calendar months (documented approximation).
19f-bis. **Uncapped accumulation and %-of-NAV sizing**: with `target_total_notional: null` B keeps buying every purchase step for the whole period (payoffs recycled through cash); `notional_pct_nav` sizes each purchase as a fraction of current NAV instead of a fixed USD amount. The slot pool is still bounded by one purchase per step over the option's life.
19h. **Distance to margin call** (historical view): d_t = 1 − (loan/h − ℓ_I·I − ℓ_cash·cash)/(ℓ_E·E) — the further equity fall that would make utilisation exactly 1 given the loan, cash, illiquid marks and the stress multiplier in force; reported as a time series, its minimum with date, and at each drawdown trough. Historical Sharpe = CAGR / annualised vol (no cash deduction) and Sortino = CAGR / downside deviation, at the user's request; the Monte Carlo metrics keep the spec's excess-over-cash definition.
19g. **Weekly history**: `data/SPX_SPXFP_aligned_weekly.csv` = Friday closes of the aligned daily file (1,515 weeks, 1997-09-12 → 2026-09-18).

## Options
28. **Bid/ask** — half the configured spread in vol points on each side; separate spreads for new trades and unwinds; both multiplied by `stress_bid_ask_multiplier` when IV_short > `stress_iv_short_above`. Dealer quotes with bid and ask override the spread at their tenor/strike.
29. **Dealer-quote calibration** — additive level shift to the parametric surface at the quoted tenor (interpolated linearly across quoted tenors, flat beyond); exact quote used at the quoted (tenor, strike) within ±½ month / ±0.01 log-moneyness.
30. **Grid surface in time** — the CSV surface is the inception surface; it is scaled through time by the parametric factor (IV_s,t/θ_s)^{β(T)}.
31. **Scenario overrides** replace the ATM level for the covered months and tenor (skew still applied); `premium_pct` overrides are converted at use.
32. **Counterparty default** — per step with probability 1 − e^{−λ dt}; loss = (1 − recovery) × book value, booked as a novation cost (the book is kept). Own random stream.
33. **Option settlement and marks** — cash settlement at expiry equals the T = 0 mark (intrinsic), so steps 6 and 7 of §5.5 commute; NAV is at mid, `nav_bid` is also recorded.
34. **Futures overlay** — futures P&L ≈ units × ΔS (basis ignored); IM = `initial_margin_pct` × |notional| held in cash; a "margin call" for B is recorded when cash < IM.
35. **Tolerance of the accounting identity** — 1e-6 USD per USD 1bn of NAV (i.e. 1e-15 relative), the float64 limit; failure raises.

## Analytics
36. **Greek explain** uses the 5y ATM vol change and the 5y zero-rate change as the book's vol and rate drivers; the residual is shown explicitly.
37. **Liquidity coverage ratio** projects capital calls with the fund's own rc (the summary uses 25% p.a. when the fund config is not available to the metric).
38. **Illiquid equity beta** for the beta-adjusted exposure charts is ρ σ_ill / σ_eq.
39. **Comparison lenses** scale B's `delta_fraction_of_A` on common random numbers (bracket [0.25, 3]).
