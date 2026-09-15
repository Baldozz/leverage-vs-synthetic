# Claude Code Prompt — Leverage vs. Long-Dated Call Replacement Simulator (v3, USD only)

> **v3 scope:** everything is denominated in **USD** — assets, loan, cash, option underlyings, option premiums and payoffs, and reporting. FX risk, FX hedging, non-USD loans, quanto options and CHF reporting are **out of scope**.
>
> **v2 changes (retained):** full treatment of the synthetic (delta-adjusted) exposure of ATM calls and its implications (§5.7, §6.5, §6.6). Expanded flexibility for option prices (§4.2b), strikes, underlyings (§4.1b, §5.2), dividends, and rates and loan terms (§4.3). New validation anchors (§7, tests 24–40).
>
> **How to use:** copy everything below the line into Claude Code, started from an empty project folder. Claude Code is instructed to plan first, confirm ambiguities with you, then build test-first. All numeric defaults are **placeholders** and must be replaced with current market data and your bank's actual Lombard terms before any decision is taken.

---

## 0. Role and standard of care

You are a senior quantitative developer and financial engineer (PhD/MSc Finance level) working for a **Swiss single-family office**. You are building a decision-support simulator that will be used by the investment committee. The standard of care is that of an independent model-validation function:

- **Correctness is the overriding priority.** Speed, UI polish and feature breadth come after.
- Every formula must be written down in `docs/METHODOLOGY.md` with notation, units, compounding and day-count conventions, and a reference to where it is implemented and tested.
- **No silent assumptions.** Any modelling choice not fixed by this prompt must be (a) listed in `docs/ASSUMPTIONS.md`, (b) exposed as a config parameter where reasonable, and (c) raised with me before implementation if it materially affects results.
- **Never weaken, skip or loosen a test to make it pass.** If a test fails, find the bug. If you believe the test itself is wrong, stop and explain why.
- Never confuse the **real-world measure P** (used to simulate paths) with the **risk-neutral measure Q** (used to price options). This separation must be explicit in code and docs.
- No look-ahead: every decision rule at time t may use only information available at t.

---

## 1. Business context and the decision

Current setup ("Strategy A"):

- Own capital (NAV) of **USD 1,000m**, strategic split **70% listed equities / 30% illiquids** (hedge funds, private equity, infrastructure fund).
- Leverage of **25% of NAV** via a Lombard/margin facility, i.e. **USD 250m loan**, **USD 1,250m gross exposure**.

Proposed setup ("Strategy B"):

- Repay the loan, stop borrowing.
- Replace the direct listed-equity exposure with **5-year at-the-money call options** on equity indices, **built and rolled on a monthly schedule** (ladder of tranches).
- The unspent cash (equity sleeve minus option premium) is held in money-market instruments and acts as **dry powder** to be deployed after a large market drawdown.

Objective of the tool: quantify whether B is feasible and preferable to A in terms of **return, drawdown, margin-call / forced-liquidation risk, liquidity (dry powder) and cost of carry**, across expected-return, volatility, rate and crash scenarios, with full sensitivity analysis.

---

## 2. Strategies to implement

All strategies must run on **exactly the same simulated market paths** (common random numbers).

### Strategy A — Levered cash portfolio (status quo)
- Inception balance sheet controlled by `leverage.allocation_of_borrowed_funds`:
  - `pro_rata` (**default**): gross 1,250m split 70/30 → equities 875m, illiquids 375m, loan 250m.
  - `equity_only`: equities 950m, illiquids 300m, loan 250m.
- Loan: floating rate = base rate + spread; interest either paid in cash or capitalised (config).
- Dividends received in cash net of withholding tax.
- Collateral / margin mechanics per §5.1.
- Optional rebalancing policy: `buy_and_hold` (default) or `maintain_target_leverage` (monthly; note this is pro-cyclical).

### Strategy B — Unlevered, call-option equity replacement with dry powder
- Transition at t0: repay the loan by selling equities, then sell the remaining direct equities and buy calls per the ladder rules (§5.2). Illiquids are **held unchanged** (they cannot be sold); illiquid holdings are identical to A's at inception.
- Option sizing mode (`options.sizing_mode`). Each mode implies a different equity exposure, premium and dry powder, so the tool must show all three side by side (§5.7):
  - `notional_match` (**default**): total target notional = A's gross equity exposure (e.g. 875m). **Warning:** this gives materially *lower* equity exposure than A, because option delta < 1.
  - `delta_match`: notional chosen so that the book's dollar-delta equals A's equity exposure (at inception, or maintained per §5.7).
  - `beta_adjusted_delta_match`: as above, but matching total portfolio equity beta, including the illiquids' equity beta and any basis between held stocks and option underlyings (§4.1b).
  - `premium_budget`: premium spend = X% of NAV.
  - `cash_reserve_target`: size so that cash ≥ X% of NAV after premium.
  - `delta_fraction_of_A`: target dollar-delta = user fraction (e.g. 50%) of A's equity exposure.
  - Target may be expressed in USD or as % of current NAV (re-evaluated at each tranche purchase).
- Strike selection mode per §5.2 (`atm_spot` default; `atm_forward`, `pct_spot`, `pct_forward`, `delta_target`).
- Cash earns the money-market rate (base rate minus a configurable spread).
- Dry-powder deployment rules per §5.3.

### Strategy C — Unlevered cash benchmark
- Same 70/30 (or A's illiquid amount) with zero leverage and no options. Required for attribution (effect of leverage vs. effect of options).

### Strategy D (optional, config switch) — A with a cash buffer
- Strategy A that holds a configurable cash buffer funded from the loan, to test whether a buffer alone solves the margin-call problem.

---

## 3. Conventions and notation (must be enforced everywhere)

- **Single currency: USD.** All assets, the loan, cash, option underlyings, premiums, payoffs and reports are in USD. There is no FX risk factor, FX translation or FX hedging anywhere in the model. The config schema must **reject** any non-USD currency field with a clear error (see test 33). All amounts stored in USD as `float64`; displayed in USD m. Any CHF view for the family office is out of scope and must not be implemented implicitly.
- Time in **years**. Simulation step `dt` configurable: `monthly` (1/12, default), `weekly` (1/52), `daily` (1/252). Strategy events (tranche purchases, rolls, deployment checks, rebalancing) occur on **month-end** steps regardless of `dt`.
  - Document that monthly stepping **understates** intra-month drawdowns and margin-call frequency; the UI must show a warning when `dt = monthly` and recommend a daily run for margin-risk conclusions.
- Calendar days per step for accrual = 365/steps_per_year (document this approximation).
- **Money-market accrual** (loan, cash): simple interest, ACT/360 by default (config: ACT/360, ACT/365).
- **Option pricing:** continuous compounding, ACT/365 year fractions, continuous dividend yield.
- **Return inputs:** user specifies whether expected returns are `arithmetic` (E[1+R_annual]) or `geometric` (median/CAGR). Conversion for lognormal assets with annual volatility σ:
  - arithmetic μ_a → continuous drift m = ln(1+μ_a); log-mean per year = m − σ²/2
  - geometric g → log-mean per year = ln(1+g); m = ln(1+g) + σ²/2
- **Equity total return = price return + dividend yield.** Indices underlying the options are assumed to be **price-return** indices (config field `underlying_type`, see §4.1b; for `total_return`, set q = 0 for pricing and no separate dividends are paid). Continuous dividend yield q_c = ln(1+q_annual).
- Per-step dividend cash on a holding of units u: `D = u · S_{t+dt} · (exp(q_c·dt) − 1) · (1 − WHT)`. This makes the gross total-return expectation exact: E[S_{t+dt}·e^{q_c dt}] = S_t·e^{m·dt}.
- Annual rates of flows (e.g. capital call rates) converted to per-step: `rate_step = 1 − (1 − rate_annual)^{dt}`.

---

## 4. Market models (scenario generator)

The scenario generator produces an **immutable `MarketPaths` object** that is independent of any strategy. Use `numpy.random.Generator(PCG64)` with `SeedSequence.spawn` so that **each risk factor has its own stream**: toggling a module (e.g. jumps on/off) must not change the random draws of unrelated factors.

### 4.1 Equity indices (1..n, user-defined)
Each index (USD-denominated) has: `name`, `spot`, `expected_return` (+ type), `dividend_yield`, `realized_vol`, `underlying_type`, `pricing_dividend_curve`, jump parameters.

Models (config `equity.model`):
1. **GBM** (exact discretisation): `ln S_{t+dt} = ln S_t + (m − q_c − σ²/2)·dt + σ·√dt·Z`
2. **Merton jump-diffusion**: add compound Poisson log-jumps. A **single common jump clock** N ~ Poisson(λ·dt) shared across indices (systemic crashes), index-specific jump sizes J ~ N(μ_J, σ_J²). Drift compensator: subtract λ·κ·dt with κ = exp(μ_J + σ_J²/2) − 1, so that the expected return equals the user input. Test this.
3. **Coupled stochastic volatility**: realised vol σ_t = IV_short,t × (1 − vrp), where `vrp` is the variance-risk-premium ratio; use σ_t at the start of the step (conditional log-normal step with drift −σ_t²/2). Combinable with jumps.
4. **Stationary block bootstrap** (Politis–Romano) from a user-supplied CSV of **jointly sampled** monthly/daily series (index returns, short-dated implied vol such as VIX, short rate) to preserve cross-dependence. Expected block length is a parameter. Optional re-centring of means to user expected returns.

Correlations: user-supplied correlation matrix over all Brownian drivers (equity indices, held-portfolio idiosyncratic shock, illiquids, IV shock, rate shock). Validate symmetry, unit diagonal, eigenvalues. If not positive semi-definite, compute the nearest correlation matrix (Higham 2002), **warn loudly**, and log the Frobenius distance. Cholesky factorisation on the validated matrix.

### 4.1b Held portfolio vs. option underlying (basis risk) and dividends
- Strategy A holds an **actual equity portfolio**, not necessarily the option index. Model it as a separate asset with `beta` to each option index, an idiosyncratic (tracking-error) vol, and an expected alpha (default 0). Strategy B gives up this alpha and tracking error; the tool must report it explicitly.
- Option underlying per index: `price_return` (default), `total_return` (q = 0 in pricing, no dividends), or `net_total_return` (pricing dividend = gross × (1 − WHT embedded in the index)). The flag must be consistent between pricing and simulation. Mismatches raise a validation error.
- Dividend inputs per index:
  - realised dividend yield (P-measure, cash received in A);
  - pricing dividend yield as a term structure q(T) (Q-measure: implied from dividend futures or put–call parity), with a flat scalar allowed;
  - optional dividend shock in stress (dividend cuts lower A's income and raise the value of existing calls: ∂C/∂q < 0).
- Allow any number of USD-denominated indices (e.g. US large-cap, US equal-weight, US tech, or USD-quoted global indices), with user weights inside the option book. The book weights may differ from A's equity weights; report the resulting sector/style exposure mismatch.

### 4.2 Implied volatility
- Short-dated ATM implied vol (e.g. 1m) `IV_s`: log-OU with exact discretisation
  `x_{t+dt} = x_t·e^{−κ dt} + ln θ·(1 − e^{−κ dt}) + η·√((1 − e^{−2κ dt})/(2κ))·Z_IV`, x = ln IV_s,
  with corr(Z_IV, Z_equity) = ρ (typically strongly negative), plus an optional additive jump `j_IV` to x when an equity jump occurs. Floor/cap parameters.
- Term structure: `IV_ATM(T, t) = θ_T · (IV_s,t / θ_s)^{β(T)}` where θ_T is the long-run ATM level for tenor T and β(T) ∈ [0,1] is the elasticity of tenor-T vol to short vol (long-dated vol reacts less). Interpolate θ_T and β(T) linearly in T between user grid points.
- Skew: `σ(K, T, t) = IV_ATM(T, t) + ψ(T)·k`, k = ln(K/F), ψ(T) = ψ_1y/√T (ψ_1y < 0 for equities). Mode `sticky_moneyness` (default) or `sticky_strike`. Floor at a minimum vol.
- A tranche struck ATM at purchase becomes ITM/OTM later: its mark must use the skew at its **current** log-moneyness.
- Bid/ask on options expressed in vol points (configurable, may widen with IV level).

### 4.2b Option price sources (flexibility requirement)
Configurable per index (`pricing.source`), in order of precedence where several are supplied:
1. `dealer_quotes`: user-entered inception quotes (premium in % of notional, or implied vol; bid and ask; tenor; strike; quote date; underlying spot, rate and dividend used by the dealer). Back out implied vol with `brentq` on the BSM price (checking the quote lies within no-arbitrage bounds `max(S e^{−qT} − K e^{−rT}, 0) ≤ C ≤ S e^{−qT}`), then use it to calibrate the surface level at that tenor/strike. Show a quote-vs-model reconciliation table.
2. `vol_surface_file`: user CSV grid of implied vols (tenor × moneyness or tenor × delta). Interpolate linearly in **total implied variance** w = σ²T along tenor and with a monotone scheme across strikes. Check calendar arbitrage (w non-decreasing in T at fixed forward moneyness) and butterfly arbitrage (non-negative implied density via the call-price second derivative in strike). Report violations and refuse to run unless the user explicitly overrides.
3. `parametric`: the §4.2 model (default when nothing else is supplied).
4. `scenario_overrides`: user-specified implied vol level or premium % for future roll dates or stress scenarios (e.g. "5y ATM vol = 35% at months 6–12").
5. `historical_iv_file`: user data for historical replay; long-dated vol is mapped from short-dated vol via β(T) if not supplied.
- Additional pricing controls: `option_funding_spread` over the discount curve (dealer-implied financing); a separate bid/ask for **new trades vs. unwinds**; a stress multiplier on bid/ask; a flat commission per trade in bps of notional.
- European exercise only for v1 (index options); raise an error if an American underlying is configured.
- All prices must show the source used (model, quote, surface, override) in the tranche table and the audit ledger.

### 4.3 Interest rates
- Mode `flat` (default): deterministic constant or user term structure.
- Mode `vasicek`: `dr = a(b − r)dt + σ_r dW`, exact discretisation
  `r_{t+dt} = r_t·e^{−a dt} + b(1 − e^{−a dt}) + σ_r·√((1 − e^{−2a dt})/(2a))·Z_r`.
  Zero-coupon bond price for maturity τ: `P(τ) = exp(A(τ) − B(τ)·r)`, `B(τ) = (1 − e^{−aτ})/a`, `A(τ) = (B(τ) − τ)(a²b − σ_r²/2)/a² − σ_r²B(τ)²/(4a)`; zero rate y(τ) = −ln P(τ)/τ. Document the assumption P = Q for rates (or add a term-premium parameter). Negative rates are permitted.
- Mode `curve`: user USD zero or par curves by tenor (bootstrap par to zero). Interpolate linearly in log discount factors. Deterministic scenario shifts are available: parallel, steepener/flattener, and custom tenor-bucket shocks.
- Derived rates: loan rate = max(loan base, floor) + spread tier (`loan_terms`); cash yield = cash base − `rates.cash.spread`; option-pricing rate for tenor T = zero rate y(T) (+ optional `option_funding_spread`).
- **Loan flexibility (Lombard term sheet):**
  - loan in USD only; base rate index USD SOFR (overnight compounded or term SOFR) with its tenor and reset frequency;
  - floor on the base rate (e.g. 0%); tiered spread schedule by utilisation or loan size; commitment/non-utilisation fee on undrawn limit; arrangement fee;
  - `rate_type`: floating, fixed for N years, or floating + swap hedge (swap MTM tracked and included in NAV and liquidity);
  - facility limit and maturity/rollover assumption with a configurable spread at renewal (the bank may re-price or cut the facility in stress).
- **Cash flexibility:** yield instrument (T-bills, MMF, deposits) with its own spread and optional counterparty concentration limit.
- Option pricing uses the USD discount curve (+ optional funding spread). The loan uses the SOFR base plus spread; cash uses its own instrument yield. The model must never silently use one rate for all three purposes; the UI shows the three rates used at each date.
- Optional correlation of Z_r with equity shocks (central-bank cuts in crashes).
- A single USD rate model drives all three rates (with their respective spreads).

### 4.4 FX — out of scope
- No FX model. All instruments are USD. Section kept only so section numbering stays stable; do not create an FX module.

### 4.5 Illiquid assets (HF, PE, Infrastructure — each configurable)
- **True (economic) returns**: correlated lognormal with user expected return, vol, correlation to equities, plus optional beta to equity jump events.
- **Reported NAV (appraisal smoothing)**: Geltner-type AR(1): `r_obs,t = (1 − φ)·r_true,t + φ·r_obs,t−1`, φ per asset and per observation frequency (e.g. quarterly marks held flat between reporting dates). Report both true and reported NAV; margin lending value and official NAV use **reported** marks, economic risk metrics use **true** marks (config switch). Long-run mean of r_obs must equal that of r_true (test).
- **Closed-end cash flows (PE, infrastructure)** — Takahashi–Alexander (Yale) model:
  - contributions `C_t = RC_step · U_t`, `U_{t+1} = U_t − C_t` (U = unfunded commitments)
  - distributions `D_t = RD_t · NAV_t`, `RD_t = (age_t / L)^B` converted to per-step
  - `NAV_{t+1} = NAV_t·(1 + G_t) + C_t − D_t`, G_t = simulated true return
  - optional crisis multipliers: contributions accelerate, distributions slow when index drawdown exceeds a threshold
  - optional new-commitment pacing (default 0).
- **Hedge funds**: redemption frequency, notice period, gate %, lock-up — used only if a liquidity-raising rule tries to redeem.
- Illiquids are not sellable by default in either strategy.

### 4.6 Reporting currency
- USD only. No CHF or other reporting layer.

### 4.7 Deterministic stress scenarios
- **Stylised scenario builder** (clearly labelled "stylised, not historical data"): peak-to-trough drawdown, months to trough, recovery shape (V / U / L), months to recover, IV peak and decay, rate shock, LTV haircut changes, illiquid markdown size and reporting lag, capital-call acceleration.
- Templates (all parameters editable, labelled as stylised): `GFC_like`, `COVID_like_V`, `DotCom_like_grind`, `Rates_and_equity_2022_like`, `Lost_decade_L`.
- **Historical replay**: only from user-supplied data files. Do not hard-code historical market data.

---

## 5. Instrument mechanics and accounting

### 5.1 Lombard loan and margin (Strategy A, D)
- Lending value: `LV_t = Σ_i ℓ_i(t)·MV_i(t)`; default ℓ(equity index holdings) = placeholder, ℓ(illiquids) = 0, ℓ(cash) = placeholder.
- **Stress-dependent LTVs:** `ℓ_i(t) = ℓ_i^base · h(state_t)`, where h is a user step-function of IV_s and/or index drawdown (banks cut advance rates in stress).
- Utilisation `u_t = L_t / LV_t`. Thresholds: `u_warn` (e.g. 0.90), `u_call` (1.00), `u_closeout` (e.g. 1.10), target after cure `u*` (e.g. 0.80), cure grace period in days.
- Cure waterfall: (1) available cash, (2) sale of marginable assets. If `u ≥ u_closeout` or the grace period expires uncured, the bank liquidates with **stressed slippage** `s` (base bps × stress multiplier).
- Sale size to restore target (selling a pro-rata basket with average LTV ℓ and slippage s):
  `x = (L − u*·LV) / ((1 − s) − u*·ℓ)`, valid when denominator > 0 (assert).
  Proceeds `x·(1 − s)` repay the loan.
- Closed-form trigger check (unit test): with loan L, equities E₀, illiquids I (reported), constant LTVs, no accrual, the margin call is triggered at equity decline
  `d* = 1 − (L − ℓ_I·I) / (ℓ_E·E₀)`.
  Example: L = 250, E₀ = 875, ℓ_E = 0.50, ℓ_I = 0 → d* = 42.86%. If the bank cuts ℓ_E to 0.40, d* = 28.57%.
- Record every event: date, utilisation, cash used, assets sold, slippage cost, realised loss.
- If NAV ≤ 0 on any path (gap risk through jumps), flag **ruin**, freeze the path, and report ruin probability.

### 5.2 Option ladder (Strategy B)
- Instrument per tranche j: European call, index i, maturity T₀ = 5y (configurable), units `n_j = Notional_j / S_i(t_j)`.
- **Strike mode** (`options.strike_mode`):
  - `atm_spot` (default): K = S(t_j).
  - `atm_forward`: K = F(t_j, T₀) = S e^{(r(T₀) − q(T₀))T₀}.
  - `pct_spot` / `pct_forward`: K = x% of S or F.
  - `delta_target`: solve K such that the call's delta equals a target (e.g. 0.50), using `brentq` with the skewed vol σ(K) re-evaluated inside the root-finder. Assert the bracket and convergence.
- **Important:** a 5-year spot-ATM call does **not** have a 50% delta. Its BSM delta is e^{−qT}N(d₁), with d₁ = (r − q + σ²/2)T/(σ√T). With positive carry (r > q) this is typically well above 0.5; with negative carry (e.g. low USD rates combined with a high dividend yield) it can be below. A 50-delta 5y call is typically struck well out of the money. The tool must always compute delta from inputs, never assume it.
- **Basket calls** (optional v2): Lévy/moment-matched lognormal approximation, flagged as an approximation and validated against MC. Note that a basket call costs less than the sum of single-index calls when correlation < 1.
- **Pricing (Q-measure):** Black–Scholes–Merton with continuous dividend yield:
  `C = S e^{−qT} N(d₁) − K e^{−rT} N(d₂)`, `d₁ = [ln(S/K) + (r − q + σ²/2)T] / (σ√T)`, `d₂ = d₁ − σ√T`,
  σ from the surface (§4.2) at the tranche's current moneyness and residual maturity, r = zero rate for residual maturity, q = pricing dividend yield from `pricing_dividend_curve` at the residual maturity (separate from the realised `dividend_yield` used for cash dividends in A).
- Purchase at **ask** (σ + half spread), mark-to-market at **mid**, liquidation/early-roll sale at **bid**; report NAV at mid and at bid.
- Handle T → 0 (intrinsic value), σ → 0, deep ITM/OTM numerically stably (use `scipy.special.ndtr`/log-space where needed).
- Ladder modes (`options.ladder_mode`):
  1. `bullet`: full notional at t0; roll at expiry or when residual maturity ≤ R.
  2. `monthly_buildup_hold_to_expiry`: deploy 1/M of target each month for M months; each tranche replaced at its expiry by a new 5y ATM tranche.
  3. `rolling_ladder` (**default**): build over M = 12 months; each tranche is held H = 12 months, then sold at bid and replaced by a fresh 5y ATM tranche bought at ask. Result: a book of M tranches with residual maturities between 4 and 5 years, re-struck monthly. M, H, T₀ configurable.
- During build-up, the not-yet-deployed part of the equity sleeve sits in cash (default) or stays in spot equities being sold down 1/M per month (config).
- Re-sizing at each roll: keep original notional, reset to target % of NAV, or reset to target dollar-delta (config).
- Optional rule: pause or reduce new purchases when IV_ATM(5y) > threshold.
- Optional counterparty risk for OTC options: default intensity λ_cp and recovery rate; off by default.
- Settlement: cash-settled payoff `n_j·max(S_T − K_j, 0)` credited to cash at expiry.
- Long options require **no margin**; document this and assert no margin logic is applied to B's option book.
- Greeks per tranche and aggregated — analytical, unit-tested against central finite differences. Reporting conventions (documented and shown in the UI):
  - dollar delta = Δ·n·S (USD per 100% index move), plus **% of NAV**;
  - dollar gamma = Γ·n·S²·0.01 (change in dollar delta for a 1% move), plus change in delta (pp) per 1% move;
  - vega per 1 vol point = ∂C/∂σ·0.01·n;
  - theta per year and per calendar day (ACT/365) = −∂C/∂T;
  - rho per 1 bp and per 100 bp;
  - dividend sensitivity ∂C/∂q per 100 bp;
  - option elasticity Ω = Δ·S/C (premium leverage).
- **Two deltas:** BSM delta (sticky strike) and smile-consistent delta. Under sticky moneyness with σ = σ_ATM + ψ·ln(K/F): ∂σ/∂S = −ψ/S, so smile delta = BSM delta + vega_raw·(−ψ/S). Report both. Exposure targeting uses the one selected in config.

### 5.3 Dry-powder deployment (Strategy B)
- Triggers evaluated at month-end on information available at t: index drawdown from running peak (e.g. −20% / −30% / −40% tiers) and/or IV_s level.
- Each tier deploys a configurable % of **deployable cash** = cash − liquidity reserve.
- Liquidity reserve = max(fixed floor, projected capital calls over the next N months under stress multipliers, planned spending).
- Deployment instrument: spot index (default) or additional call tranches (warn that calls are expensive when IV is high).
- Optional exit rule: when the index regains its prior peak, convert deployed spot equity back into the target structure over K months.
- Each tier triggers at most once per drawdown episode; re-arm after a new peak (document).

### 5.4 Other flows and costs (both strategies)
- Equity trading costs in bps (normal / stressed); transfer stamp duty on cash securities as a configurable bps cost, off by default, with a note to confirm applicability with the tax advisor.
- Dividend withholding tax rate (configurable).
- Family-office spending/distributions (% of NAV p.a. or fixed USD), default 0.
- Management fees on illiquids are embedded in their net expected returns (document).
- Taxes on gains are out of scope for v1 (document as limitation).

### 5.5 Order of operations within each step (must be implemented exactly and documented)
1. Read state at t (positions, cash, loan, market state).
2. Evolve the market to t + dt (from `MarketPaths`).
3. Accrue interest on **start-of-step** cash and loan at rates fixed at t; pay or capitalise loan interest.
4. Credit dividends (net of WHT) on start-of-step holdings, per §3.
5. Illiquid true/reported NAV update; capital calls and distributions.
6. Option expiries: settle payoffs to cash.
7. Mark all positions to market at t + dt.
8. Liquidity check: if cash < 0, apply the strategy's liquidity waterfall (A: loan headroom then asset sales; B: sell deployed spot equity, then options at bid) and record a liquidity-shortfall event.
9. Margin check (A, D) and cure/liquidation per §5.1.
10. Month-end only: ladder purchases/rolls (B), dry-powder triggers (B), rebalancing (per config), spending.
11. Record NAV, exposures, Greeks, utilisation, cash, and the ledger.
12. **Assert the accounting identity**: `NAV_{t+dt} − NAV_t = Σ P&L components + external flows` to within 1e-6 USD per path per step. Failure raises an exception with full diagnostic context.

### 5.6 Ledger and audit trail
- Double-entry style ledger: every cash flow (trade, premium, fee, interest, dividend, capital call, distribution, payoff, margin sale, spending) is a ledger row with timestamp, path id, strategy, account, amount, description.
- For a user-selected path, export the full ledger and step-by-step balance sheet to Excel, so an analyst can re-check the calculation by hand.

---

### 5.7 Synthetic exposure (delta) management — implications that must be modelled
The call book gives a **non-linear, path-dependent** equity exposure. The following must be modelled, documented and made visible:

1. **Exposure gap at inception.** With notional matching, B's dollar-delta is Δ × notional (e.g. ≈ 64% of A's exposure at σ = 20%, r = 4%, q = 1.5%, not 100%). Matching A's exposure requires notional/Δ, which means more premium and less dry powder. The Inception tab shows, for every sizing and strike mode: notional, premium, cash, dollar-delta, % of A's exposure, vega, theta, rho, dividend sensitivity.
2. **Exposure drift (gamma).** Delta rises in rallies (toward e^{−qT}) and falls in sell-offs (toward 0). B automatically de-risks in a crash and re-risks in a recovery, but only once the index is back near or above the strikes. Deep OTM tranches after a crash give weak participation in the rebound. Rolling/re-striking (§5.2) and dry-powder deployment (§5.3) are the mechanisms that restore exposure; their costs (buying at high implied vol, bid/ask) must be captured.
3. **Contrast with A.** A has constant delta per unit but rising effective leverage as markets fall (loan fixed, equity shrinking), and forced sales at the lows (negative convexity). Track effective leverage of A and effective delta of B on the same chart.
4. **Vol exposure.** B is long vega. Implied vol spikes in crashes cushion MTM losses, while vol crush in calm rallies hurts. B is also long gamma, and pays theta for it. Relative to a delta-equivalent position, performance depends on realised vs. implied vol (diagnostic: ½·Γ·S²·(σ_realised² − σ_implied²)·dt).
5. **Rates exposure.** Long 5y calls have large positive rho (≈ +2.1% of notional per +100 bp at the reference inputs). The rate path, and its correlation with equities, materially changes B's MTM. Rate cuts in a crash reduce call values on top of the equity loss.
6. **Dividend exposure.** Call value falls as pricing dividends rise (≈ −3.2% of notional per +100 bp at reference inputs). B forgoes realised dividends; A receives them net of WHT.
7. **Maximum loss.** B's maximum loss on the option book is the premium paid. There are no margin calls on long options, and cash is not at risk except via counterparty exposure.
8. **Exposure maintenance policy** (`exposure.policy`):
   - `static`: no action between scheduled rolls (default).
   - `rebalance_bands`: at month-end, if book delta deviates from target by more than ±b pp, buy additional tranches (uses cash) or sell tranches at bid. Record transaction costs and the dry-powder impact.
   - `restrike_on_move`: early roll of tranches whose moneyness S/K leaves a band [m_low, m_high].
   - `spot_topup`: close the delta gap with spot index purchases from cash, subject to the liquidity reserve.
   - `futures_overlay`: **off by default and flagged in red.** Futures reintroduce variation margin, which contradicts the objective of Strategy B. If enabled, model daily/monthly variation margin as a cash drain, include it in the liquidity checks, and report margin-call statistics for B.
9. **Fair comparison lenses.** Always present results under (i) equal capital (default), (ii) equal inception dollar-delta, and (iii) equal expected return or equal volatility (solved by scaling), so that differences are not simply exposure differences.

## 6. Analytics

### 6.1 Metrics (per strategy, with Monte Carlo standard errors / 95% confidence intervals)
- Terminal NAV: mean, median, percentiles (1, 5, 10, 25, 75, 90, 95, 99).
- Per-path CAGR `(NAV_T/NAV_0)^{1/T} − 1`: median and mean (label clearly; do not confuse "mean of CAGR" with "CAGR of the mean").
- Annualised volatility, Sharpe and Sortino from step returns (excess over cash rate); warn that smoothed illiquid marks bias these.
- Max drawdown distribution; time under water; time to recovery.
- VaR and CVaR (95%, 99%) of 1-year and horizon returns.
- **Leverage and liquidity risk:** probability of ≥1 margin call, ≥1 forced liquidation, and ruin; distribution of forced-sale volume and realised losses; minimum utilisation headroom.
- **Dry powder:** cash available at the index trough (USD and % NAV), amount deployed, return on deployed capital, liquidity coverage ratio = cash / (unfunded commitments due in 12m + spending).
- Probability that B's terminal NAV exceeds A's; distribution of NAV_B − NAV_A.
- CRRA certainty equivalent: `CE = (E[W^{1−γ}])^{1/(1−γ)}`, log utility at γ = 1; γ user-configurable; paths with W ≤ 0 handled explicitly and reported.
- Empirical first- and second-order stochastic dominance tests between strategies.

### 6.2 Inception carry and structure analytics (closed form, shown before any simulation)
- Inception balance sheets of A, B, C side by side (gross exposure, dollar-delta, cash, loan, % NAV).
- **Put–call parity decomposition** of B: `C = S e^{−qT} − K e^{−rT} + P`, i.e. long call + cash ≡ long equity + long put − forgone dividends, with the financing embedded in the call at the option-implied rate. Report: put premium (insurance cost), PV of forgone dividends, implied financing rate vs. the Lombard loan rate, cash yield on the unspent premium.
- Annualised expected carry of each strategy under the user's inputs.

### 6.3 Performance attribution
- Full-revaluation P&L by component: equity/option market move, dividends, financing cost, cash interest, illiquids, transaction costs, slippage from forced sales, dry-powder deployment.
- For the option book, additionally a Greek-based explain (delta, gamma, vega, theta, rho) with the **unexplained residual shown explicitly**.

### 6.4 Sensitivity analysis
- One-at-a-time tornado charts on a user-chosen metric (median terminal NAV, CVaR, P(margin call), CE) for all key inputs: expected returns (each asset), realised vol, implied vol level, VRP, skew, borrowing spread, base rate, cash spread, LTVs and stress haircuts, crash intensity/size, option bid/ask, ladder parameters, deployment thresholds, capital-call rates.
- Two-dimensional heatmaps for any pair of inputs (default pairs: equity expected return × 5y implied vol; borrowing spread × crash severity; base rate × implied vol).
- **Breakeven solver:** value of a chosen input at which the chosen metric is equal for A and B (root-finding with `scipy.optimize.brentq` on common random numbers; report the breakeven with a confidence interval, and warn when MC noise makes it unreliable).
- All sensitivity runs reuse the same random draws (common random numbers).
- Optional v2: Sobol global sensitivity indices (SALib).
- Additional mandatory sensitivity dimensions: strike mode and moneyness, sizing mode, exposure policy, option price source, pricing vs. realised dividend yield, base-rate floor and spread tiers, curve shifts, bid/ask in stress, implied–realised vol spread.

### 6.5 Exposure and convexity analytics
- Time series (fan charts) of: B's dollar-delta and % of A's exposure, beta-adjusted total portfolio exposure for A, B and C, A's effective leverage and utilisation, vega/rho/theta of B.
- **Delta and value grid:** option book value, P&L and delta for instantaneous index shocks (e.g. −50% … +50%) × implied vol shocks (e.g. −5 … +20 vol pts) × rate shocks, at inception and at any chosen date.
- **Participation profile:** A vs. B vs. C portfolio value against index level at horizons 1y, 3y and 5y (mark-to-market for unexpired tranches with scenario vol and rates). Report upside and downside capture ratios.
- Realised capture over simulated paths: regression of step returns of B on index returns in up and down months (up-beta, down-beta).

### 6.6 Static hold-to-expiry payoff analysis (closed form, no simulation)
For a single bullet tranche held to expiry, with constant rates, cash at the risk-free rate, dividends reinvested, no costs, WHT or margin events:
- A (equity sleeve): `V_A(S_T) = E₀·(S_T/S₀)·e^{qT} − L·e^{r_L T}`
- B: `V_B(S_T) = N·max(S_T/S₀ − K/S₀, 0) + (E₀ − L − N·c)·e^{rT}`, c = premium per unit notional
- Solve for the index levels where V_A = V_B (0, 1 or 2 breakevens) and show them per sizing and strike mode. Explain the economic reading: notional-matched B beats A only below a lower breakeven (insurance), while delta-matched B beats A below a lower breakeven or above an upper breakeven (convexity).
- Continuous compounding for this module only (stated on screen). The full simulation then shows how path dependency (margin calls, rolls, vol, dry powder) moves these conclusions.

---

## 7. Validation and testing (mandatory, `pytest`, run in CI via a single command)

Build **test-first** for every pricing and accounting module. The following must exist and pass before the UI is built:

**Pricing**
1. Black–Scholes reference (Hull textbook): S = 42, K = 40, r = 10%, σ = 20%, T = 0.5, q = 0 → call ≈ 4.76, put ≈ 0.81.
2. 5y ATM anchor: S = K = 100, r = 4%, q = 1.5%, σ = 20%, T = 5 → call ≈ 21.49, put ≈ 10.59, call delta e^{−qT}N(d₁) ≈ 0.6425 (tolerance ±0.01 on prices; recompute independently with scipy in the test).
3. Put–call parity to 1e-10 on a random grid (hypothesis property test).
4. All Greeks vs. central finite differences.
5. Limits: T → 0 gives intrinsic; σ → 0 gives discounted forward intrinsic; monotonicity in S, σ, T (for q = 0), K.
6. Q-measure Monte Carlo price of a call converges to BSM within 3 standard errors.

**Simulation**
7. GBM: sample mean of S_T/S_0 matches e^{(m − q)T} and log-variance matches σ²T within 3 SE.
8. Jump compensator: expected total return unchanged when jumps are switched on.
9. Correlation matrix recovered from simulated shocks within sampling error; nearest-PSD routine tested on a known non-PSD matrix.
10. Vasicek: simulated mean and variance vs. closed form; MC bond price E[exp(−∫r dt)] vs. P(τ) within SE.
11. Log-OU IV: stationary mean/variance of ln IV vs. closed form.
12. Geltner smoothing: long-run mean of observed returns equals true; observed vol lower.
13. Takahashi–Alexander: unfunded commitments never negative; cumulative contributions ≤ commitment.
14. Common random numbers: illiquid and index paths bit-identical across strategies and across module toggles for unrelated factors; reproducible with seed.

**Accounting and strategy logic**
15. Accounting identity asserted at every step (§5.5).
16. Leverage = 0 → Strategy A is identical to Strategy C.
17. Option notional = 0 and no deployment → Strategy B equals illiquids + cash.
18. Margin trigger at d* = 42.86% for the §5.1 example (deterministic path).
19. Cure sale formula: L = 250, E = 875 falling 45% (E = 481.25), ℓ = 0.50, u* = 0.80, s = 1% → x ≈ 97.46 and post-sale utilisation = 0.80 to 1e-9.
20. Loan interest hand-check: L = 250, rate 5%, ACT/360, 12 monthly steps of 365/12 days, flat markets: total interest paid ≈ 12.674; if capitalised monthly ≈ 12.972.
21. Deterministic zero-vol scenario for A, B and C with a **second, independent minimal implementation** (`validation/independent_recalc.py`, sharing no code with the engine) matching engine NAVs to 1e-8.
22. No look-ahead: a test that perturbs market data after t and verifies all decisions up to t are unchanged.
23. Option ladder: tranche count, maturities, strikes and roll dates match a hand-built schedule for mode 3.

**Synthetic exposure, pricing flexibility, rates** (reference inputs unless stated: S = K = 100, r = 4%, q = 1.5%, σ = 20%, T = 5, flat vol, continuous compounding)
24. Greeks of the 5y ATM call: delta ≈ 0.6425; gamma ≈ 0.00729 per index point; vega ≈ 0.729 per vol point (per 100 notional); theta ≈ −2.205 per year; rho ≈ +2.138 per 100 bp; ∂C/∂q ≈ −3.213 per 100 bp; elasticity Ω ≈ 2.99.
25. Instantaneous shocks with 5y remaining and σ = 20%: delta ≈ 0.242 (S = 60), 0.356 (S = 70), 0.708 (S = 110), 0.800 (S = 130); call value ≈ 6.19 at S = 70. With σ = 30% at S = 70: value ≈ 11.91, delta ≈ 0.460 (vol spike cushions the loss).
26. Forward-ATM strike K = F ≈ 113.31: premium ≈ 16.42, delta ≈ 0.546.
27. `delta_target` = 0.50: strike ≈ 119.87, premium ≈ 14.34 (flat vol). With skew enabled, the solved strike must reproduce delta = 0.50 to 1e-8.
28. Sizing (E₀ = 875, L = 250, equity sleeve capital 625): notional match → premium ≈ 188.04, cash ≈ 436.96; delta match → notional ≈ 1,361.8, premium ≈ 292.66, cash ≈ 332.34; 50-delta strike with dollar-delta 875 → notional 1,750, premium ≈ 250.88, cash ≈ 374.12.
29. Static payoff (§6.6) with r_L = 5%, cash at r = 4%, dividends reinvested: at S_T = 100, V_A ≈ 622.14, V_B ≈ 533.71 (notional match), ≈ 405.92 (delta match). Breakevens: notional match → single lower breakeven S_T ≈ 90.62, no upper; delta match → S_T ≈ 77.07 and ≈ 151.64.
30. Smile delta equals BSM delta when ψ = 0; smile delta matches a finite-difference delta computed with re-evaluated surface vol under sticky moneyness.
31. Implied vol inversion recovers σ to 1e-8 across a moneyness/tenor grid; quotes outside no-arbitrage bounds are rejected.
32. Surface loader: a synthetic surface with a deliberate calendar arbitrage and one with a butterfly arbitrage are both detected.
33. USD-only guard: a config containing any non-USD currency field, FX series or quanto option type is rejected by the schema with a clear message; no FX module is importable.
34. Curve: discount factors from bootstrapped par curve reprice input par instruments to 1e-10; interpolation is monotone in log DF.
35. Loan: base-rate floor applied correctly when SOFR is below the floor; spread tier switches at the configured utilisation thresholds; commitment fee on undrawn amount hand-checked.
36. Exposure policy `rebalance_bands`: after execution, book delta is within the band (or cash limit binding, flagged); transaction costs appear in the ledger.
37. `futures_overlay` enabled: variation margin flows equal futures P&L each step, and B's margin statistics are populated.
38. Held portfolio with beta = 1, tracking error 0, alpha 0 is identical to holding the option index.
39. `total_return` underlying: pricing uses q = 0 and A receives no separate dividends; mismatch between flags raises an error.
40. Price-source precedence: a dealer quote overrides the surface at its tenor/strike, and the tranche table shows the source.

Produce `reports/validation_report.html` listing every test, its purpose, and pass/fail, generated by one command.

---

## 8. Architecture and technology

- Python ≥ 3.11; `numpy`, `scipy`, `pandas`, `pydantic` v2 (config schema and validation), `PyYAML`, `plotly`, `streamlit`, `openpyxl`, `pytest`, `hypothesis`; `numba` only if profiling proves it necessary. Pin versions in `pyproject.toml`.
- Fully vectorised across paths; float64; type hints; `ruff` + `mypy --strict` clean.
- Suggested layout:

```
fo_leverage_sim/
  pyproject.toml   README.md   CLAUDE.md
  config/            default.yaml, stress_templates.yaml, schema.py
  src/fosim/
    market/          equity.py, jumps.py, implied_vol.py, rates.py,
                     illiquids.py, bootstrap.py, correlation.py, stress.py, paths.py
    pricing/         black_scholes.py, greeks.py, vol_surface.py
    instruments/     lombard_loan.py, option_ladder.py, cash.py
    strategies/      base.py, levered.py, call_replacement.py, unlevered.py, levered_buffer.py
    engine/          simulator.py, ledger.py, events.py, conventions.py
    analytics/       metrics.py, attribution.py, sensitivity.py, breakeven.py, dominance.py
    reporting/       excel_export.py, html_report.py
  app/streamlit_app.py
  validation/independent_recalc.py
  tests/
  docs/              METHODOLOGY.md, ASSUMPTIONS.md, LIMITATIONS.md, USER_GUIDE.md
```

- Pipeline: `Config → MarketScenarioGenerator → MarketPaths (immutable) → StrategyEngine × strategies → Results + Ledger → Analytics → Reports/UI`.
- Performance target: 10,000 paths × 10 years × monthly steps × 3 indices × 12 tranches runs in under ~2 minutes on a laptop; daily runs may use fewer paths. Show a progress bar and MC standard errors so the user can judge convergence.

---

## 9. Configuration (illustrative YAML — every number is a PLACEHOLDER)

```yaml
run:
  n_paths: 10000
  horizon_years: 10
  dt: monthly            # monthly | weekly | daily
  seed: 20260915
  currency: USD                 # fixed; any other value is rejected

portfolio:
  nav0: 1_000_000_000
  weights: {equities: 0.70, illiquids: 0.30}

leverage:
  ratio_of_nav: 0.25
  allocation_of_borrowed_funds: pro_rata   # pro_rata | equity_only
  # loan rate, spread tiers and fees: see loan_terms below
  day_count: ACT/360
  interest: pay_cash                        # pay_cash | capitalise
  ltv_base: {equity_index: 0.50, illiquid: 0.00, cash: 0.90}   # PLACEHOLDER
  ltv_stress_schedule: [{iv_short_above: 0.35, multiplier: 0.80}]
  thresholds: {warn: 0.90, call: 1.00, closeout: 1.10, target_after_cure: 0.80}
  cure_grace_days: 2
  rebalancing: buy_and_hold

held_equity_portfolio:
  betas: {IDX_1: 1.0}
  tracking_error_vol: 0.03
  expected_alpha: 0.0

equity_indices:
  - name: IDX_1                 # user choice, e.g. S&P 500 price index (USD)
    underlying_type: price_return   # price_return | total_return | net_total_return
    option_book_weight: 1.0
    pricing_dividend_curve: {tenors: [1, 5], q: [0.015, 0.015]}
    dividend_stress_cut: 0.0
    weight_in_equity_sleeve: 1.0
    spot: 100.0
    expected_return: 0.07       # PLACEHOLDER
    return_type: arithmetic     # arithmetic | geometric
    dividend_yield: 0.015
    realized_vol: 0.16
    dividend_wht: 0.15          # PLACEHOLDER — confirm with tax advisor

equity_model: {type: merton, jump_lambda: 0.10, jump_mean: -0.15, jump_vol: 0.10}

implied_vol:
  short: {theta: 0.18, kappa: 4.0, eta: 1.0, rho_equity: -0.70, jump_add: 0.5, floor: 0.08, cap: 1.5}
  term_structure: {tenors: [1, 2, 5], theta: [0.19, 0.20, 0.21], beta: [0.6, 0.45, 0.35]}
  skew_1y: -0.10
  skew_mode: sticky_moneyness
  bid_ask_vol_pts: 0.005
  vrp_ratio: 0.15

rates:
  model: flat                   # flat | vasicek | curve
  usd: {r0: 0.040, a: 0.3, b: 0.035, sigma: 0.01, rho_equity: 0.2}
  curves_file: null             # USD zero/par curve
  scenario_shift: {type: none, bp: 0}   # parallel | steepener | flattener | buckets
  cash: {instrument: tbills, spread: 0.0010}

loan_terms:
  base_index: SOFR              # overnight_compounded | term_1m | term_3m via reset_months
  reset_months: 1
  base_floor: 0.0
  spread_tiers: [{utilisation_below: 1.0, spread: 0.0100}]   # PLACEHOLDER
  commitment_fee_bps: 0
  facility_limit: 400_000_000
  rate_type: floating           # floating | fixed | floating_swapped
  fixed_years: null
  renewal_spread_stress: 0.0050

pricing:
  source: parametric          # dealer_quotes | vol_surface_file | parametric | scenario_overrides | historical_iv_file
  dealer_quotes: []           # e.g. {index: IDX_1, tenor: 5, strike_pct_spot: 1.0, bid_pct: null, ask_pct: null, quote_date: null}
  vol_surface_file: null
  option_funding_spread: 0.0
  bid_ask_vol_pts_new: 0.005
  bid_ask_vol_pts_unwind: 0.0075
  stress_bid_ask_multiplier: 3.0
  commission_bps_notional: 0
  arbitrage_check: enforce    # enforce | warn

options:
  tenor_years: 5
  strike_mode: atm_spot       # atm_spot | atm_forward | pct_spot | pct_forward | delta_target
  strike_param: null          # e.g. 1.10 for pct modes, 0.50 for delta_target
  delta_definition: bsm       # bsm | smile
  option_type: vanilla        # vanilla | basket (v2)
  sizing_mode: notional_match
  ladder_mode: rolling_ladder
  buildup_months: 12
  hold_months_before_roll: 12
  resize_on_roll: target_pct_nav
  counterparty: {enabled: false, default_intensity: 0.005, recovery: 0.4}

exposure:
  policy: static              # static | rebalance_bands | restrike_on_move | spot_topup | futures_overlay
  target: {type: pct_of_A_exposure, value: 1.0}
  band_pp: 0.10
  moneyness_band: [0.75, 1.40]
  futures_overlay: {enabled: false, initial_margin_pct: 0.10}
  comparison_lenses: [equal_capital, equal_delta, equal_vol]

dry_powder:
  tiers:
    - {index_drawdown: -0.20, deploy_pct_of_deployable: 0.33}
    - {index_drawdown: -0.30, deploy_pct_of_deployable: 0.50}
    - {index_drawdown: -0.40, deploy_pct_of_deployable: 1.00}
  instrument: spot_index
  liquidity_reserve: {floor_pct_nav: 0.03, capital_call_months: 12}
  exit_rule: {enabled: false, convert_over_months: 12}

illiquids:
  - {name: HedgeFunds, weight: 0.33, type: open_ended, expected_return: 0.06, vol: 0.07, rho_equity: 0.6, smoothing_phi: 0.2, liquidity: {frequency_months: 3, notice_days: 90, gate: 0.25}}
  - {name: PrivateEquity, weight: 0.34, type: closed_end, expected_return: 0.10, vol: 0.22, rho_equity: 0.7, smoothing_phi: 0.5, unfunded_commitments: 100_000_000, ta_model: {rc: 0.25, b: 2.5, life_years: 10, age_years: 4}}
  - {name: Infrastructure, weight: 0.33, type: closed_end, expected_return: 0.08, vol: 0.12, rho_equity: 0.5, smoothing_phi: 0.5, unfunded_commitments: 50_000_000, ta_model: {rc: 0.25, b: 2.0, life_years: 12, age_years: 5}}

costs: {equity_bps: 5, equity_bps_stressed: 40, stamp_duty_bps: 0}
spending: {pct_nav_pa: 0.0}
```

The config must be validated by the pydantic schema (ranges, weights summing to 1, PSD correlation, consistent tenors); invalid configs fail fast with clear messages.

**Illustrative inception check** (bullet mode, pro-rata, σ = 20%, r = 4%, q = 1.5%, no costs): A holds equities 875m, illiquids 375m, loan 250m. B (notional match 875m) pays premium ≈ 188.0m (21.49% of notional), holds illiquids 375m and cash ≈ 437.0m, with dollar-delta ≈ 562m (56% of NAV) vs. A's 875m (87.5%). Delta-matched B needs notional ≈ 1,362m, premium ≈ 292.7m, cash ≈ 332.3m. The tool must reproduce these figures from the same inputs. Note that the inception delta of the 5y spot-ATM call is ≈ 0.64, not 0.50. A 50-delta 5y call at these inputs is struck at ≈ 120% of spot (test 27).

---

## 10. User interface and reporting

Streamlit app, sober institutional design (no gimmicks), USD m with thousands separators, all charts exportable.

Tabs:
1. **Assumptions** — editable inputs, load/save YAML, market-data CSV upload, assumptions changelog with timestamp.
2. **Inception** — balance sheets, put–call parity decomposition, carry comparison, margin-trigger distance (d*).
3. **Simulation** — NAV fan charts (5/25/50/75/95), terminal distribution, NAV_B − NAV_A distribution.
4. **Risk** — drawdowns, VaR/CVaR, margin-call and liquidation statistics, utilisation paths, ruin probability.
5. **Liquidity & dry powder** — cash paths, deployment events, liquidity coverage, capital calls.
6. **Option book & exposure** — tranche table (strike, moneyness, residual maturity, vol used and its source, delta, value), aggregate Greeks over time, roll costs, MTM at mid vs. bid, delta and value grid, participation profile, effective-exposure comparison with A, static breakeven chart (§6.6), sizing/strike mode comparison table.
7. **Sensitivity** — tornado, heatmaps, breakeven solver.
8. **Stress tests** — stylised templates and historical replay; side-by-side path charts with event markers.
9. **Audit** — single-path ledger viewer and Excel export; validation report link.

One-click **investment-committee report** (HTML and PDF): executive summary, methodology summary, key inputs, results with confidence intervals, stress tests, sensitivity, limitations, and a model-risk disclaimer.

---

## 11. Limitations to document explicitly (`docs/LIMITATIONS.md`)

Black–Scholes-with-skew is a pricing approximation, not market quotes (validate inception premiums against dealer indications); long-dated option liquidity and OTC bid/ask can be much wider in stress; counterparty and CSA terms; Lombard terms are bank-specific and LTVs can be changed at the bank's discretion; appraisal-based illiquid marks; monthly stepping understates margin risk; all-USD simplification (FX risk versus the family office's CHF reference currency deliberately ignored); tax treatment (dividend withholding, stamp duty, treatment of option gains) not modelled beyond simple costs; model risk in jump and vol dynamics; MC sampling error; no behavioural or governance constraints.

---

## 12. Workflow rules for you (Claude Code)

1. **Plan first.** Before writing code, produce `docs/PLAN.md`: your understanding of the problem, the module plan, the test plan, and a numbered list of ambiguities with your proposed default resolution. **Stop and wait for my confirmation.** At minimum confirm with me:
   - allocation of borrowed funds (pro-rata vs. equity-only);
   - interpretation of the monthly ladder (default `rolling_ladder`, M = H = 12);
   - option sizing mode and exposure target (notional vs. delta vs. a fraction such as 50% of A's exposure);
   - strike mode, and whether the target delta is BSM or smile-consistent;
   - exposure maintenance policy;
   - option price source(s) available (dealer quotes, surface files);
   - underlying index type (price / total / net total return) and held portfolio vs. option index;
   - Lombard term-sheet items (SOFR convention, floor, spread tiers, fees, limit);
   - whether illiquid marks used for margin are reported or true.
2. Create `CLAUDE.md` capturing conventions (§3), the P vs. Q rule, the accounting identity and "never loosen tests".
3. Build in this order, with tests green at each stage before moving on: conventions & config schema → pricing → market models → instruments → strategies & engine (with ledger and identity assertion) → independent recalculation → analytics → sensitivity → reporting → UI.
4. After each stage, run the full test suite and summarise results. Never mark a stage complete with failing or skipped tests.
5. Self-review before finishing: re-derive every formula in `METHODOLOGY.md` against the code line by line, check units and compounding at every interface, and list any discrepancy found and fixed.
6. Do not invent market data. Any default number is labelled PLACEHOLDER in config, UI and reports.

## 13. Acceptance criteria

- `pytest` passes fully; validation report generated; `ruff` and `mypy` clean.
- Independent recalculation matches the engine to 1e-8 on deterministic scenarios.
- Accounting identity holds on every path and step of a 10,000-path run.
- The illustrative inception figures in §9 are reproduced.
- `streamlit run app/streamlit_app.py` launches and every tab works on the default config.
- `METHODOLOGY.md`, `ASSUMPTIONS.md`, `LIMITATIONS.md` and `USER_GUIDE.md` are complete and consistent with the code.
