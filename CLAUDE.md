# CLAUDE.md — project conventions (binding)

Decision-support simulator for a Swiss single-family office: levered cash equity portfolio
(Strategy A) vs. unlevered 5-year call replacement with dry powder (Strategy B), plus an
unlevered benchmark (C) and A-with-buffer (D). Full spec: `SPEC.md`. Plan: `docs/PLAN.md`.

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
- `.venv/bin/streamlit run app/streamlit_app.py` — UI.
