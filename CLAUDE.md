# CLAUDE.md — project conventions (binding)

Investment decision-support tool. **The live question** (`README.md`): 1 bn in SPX with a
250 m Lombard loan (lending value 75 %, margin call at 90 % of it, SOFR + 75 bp, interest capitalised) — **Keep the loan**, or **Rotate into
calls**: week by week sell SPX, buy 5-year ATM calls on SPXFP sized by the model delta, repay the loan; at every
expiry the replacement closes the gap to Keep's exposure (SPX + the calls' live dollar delta), and every month-end the
exposure is brought back inside ±10 % of Keep's (calls sold at the mark less 1 vol point / calls bought from the T-bills, then from SPX sold for them);
the earlier rules (same dollar delta, same index units, worthless calls lapsing, SPX sold delta-for-delta, unfundable
replacements cut) stay in the sidebar. Historical data only
(Sept 1997 → today), every trading day as a start date, no Monte Carlo. Engine `src/fosim/analytics/leverage_stress.py`,
app `app/historical_app.py` (two pages), formulas `docs/METHODOLOGY.md` §9, choices `docs/ASSUMPTIONS.md` 19p.
The earlier Monte Carlo simulator (strategies A/B/C/D, `SPEC.md`, `docs/PLAN.md`, `app/streamlit_app.py`) is kept
for reference and its tests still run.

## Standard of care
- Correctness first. Speed, UI and breadth come after.
- **Never weaken, skip or loosen a test to make it pass.** Find the bug. If the test is wrong, stop
  and explain why before touching it.
- No silent assumptions: any choice not fixed by `SPEC.md` goes in `docs/ASSUMPTIONS.md` and, where
  reasonable, becomes a config parameter.
- Every formula lives in `docs/METHODOLOGY.md` with notation, units, compounding, day count, and a
  pointer to the implementing function and the test that checks it.
- Never invent market data. Default numbers are PLACEHOLDERs and labelled as such.

## P vs. Q
- **P (real-world)** drives simulated paths (`fosim.market.*`): user expected returns, realised vol,
  jumps, VRP. Functions are named `simulate_*`.
- **Q (risk-neutral)** prices options (`fosim.pricing.*`): zero rate + funding spread, pricing
  dividend curve, implied vol surface. Functions are named `price_*` / `bsm_*`.
- Never pass a realised (P) vol or dividend into a pricer; never pass an implied (Q) vol into a
  path generator except through the explicit VRP coupling `σ_real = IV_short × (1 − vrp)`.

## Conventions (SPEC §3)
- Single currency USD, float64, stored in USD, displayed in USD m. No FX anywhere.
- Time in years. `dt ∈ {1/12, 1/52, 1/252}`. Strategy events on month-end steps only.
- Calendar days per step = 365 / steps_per_year.
- Money-market accrual (loan, cash): simple interest, ACT/360 default (ACT/365 optional).
- Option pricing: continuous compounding, ACT/365 year fractions, continuous dividend yield.
- Expected-return conversion: arithmetic μ_a → m = ln(1+μ_a); geometric g → m = ln(1+g) + σ²/2.
  Log-mean per year = m − σ²/2 in both cases.
- Continuous dividend yield q_c = ln(1 + q_annual). Per-step dividend cash on u units:
  `u · S_{t+dt} · (exp(q_c dt) − 1) · (1 − WHT)`.
- Annual flow rates to per-step: `1 − (1 − rate)^{dt}`.

## Accounting identity (SPEC §5.5 step 12)
`NAV_{t+dt} − NAV_t = Σ P&L components + external flows` to within 1e-6 USD per path per step,
asserted inside the engine on every step; failure raises `AccountingIdentityError` with full
diagnostic context. The order of operations in `fosim/engine/simulator.py` follows §5.5 exactly.

## No look-ahead
Any decision rule at step k may read only market state at indices ≤ k. `MarketPaths` is immutable.

## Tooling
- `.venv/bin/python -m pytest` — full suite (must be green before any stage is declared done).
- `.venv/bin/ruff check . && .venv/bin/mypy` — must be clean for `src/fosim` and `validation/`.
- `.venv/bin/python -m fosim.reporting.validation_report` — regenerates `reports/validation_report.html`.
- `.venv/bin/streamlit run app/historical_app.py` — the decision app (two pages: "Historical simulation", "Call premium history"); `app/streamlit_app.py` and `app/strategy_replay_app.py` are the earlier Monte Carlo and strategy-replay apps.
- `.venv/bin/python scripts/export_trajectories.py [--grid daily] [--sample W]` — every start's trajectory to a long CSV under `reports/` (generated, not committed).
- Page tests use Streamlit's AppTest on the weekly grid (`at.session_state["s_grid"] = "every week"`); the bottom tables are cross-checked against direct `simulate()` runs — keep that check when the layout changes.
- The engine module is imported once by the running server: restart it after changing `leverage_stress.py` (page files reload on save).
