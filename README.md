# Leverage vs. long-dated call replacement simulator (USD only)

Decision-support simulator for a Swiss single-family office comparing
**Strategy A** (levered cash equity portfolio on a Lombard facility) with
**Strategy B** (repay the loan, replace the equity sleeve with a monthly ladder of 5-year ATM index
calls and hold the unspent premium as dry powder), plus **C** (unlevered benchmark) and **D**
(A with a cash buffer), on identical simulated market paths.

* Spec: `SPEC.md` · Plan and ambiguity resolutions: `docs/PLAN.md` · Conventions: `CLAUDE.md`
* Methodology (every formula with implementation/test pointers): `docs/METHODOLOGY.md`
* Assumptions: `docs/ASSUMPTIONS.md` · Limitations: `docs/LIMITATIONS.md` · User guide: `docs/USER_GUIDE.md`

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                                  # validation suite (spec §7 tests 1–40)
.venv/bin/python -m fosim.reporting.validation_report       # reports/validation_report.html
.venv/bin/streamlit run app/streamlit_app.py                # UI
.venv/bin/ruff check . && .venv/bin/mypy                    # lint / strict typing
```

**Every default number in `config/default.yaml` is a PLACEHOLDER** — replace with current market
data, dealer quotes and the bank's Lombard term sheet before any decision.
