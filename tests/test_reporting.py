"""Reporting: Excel single-path audit export reconciles (ledger cash flows vs cash account, identity
residual column ≈ 0), the IC HTML report builds, the validation-report generator parses docstrings."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tests.conftest import make_cfg

from fosim.reporting.excel_export import balance_sheet_frame, export_path_audit
from fosim.reporting.html_report import build_ic_report
from fosim.reporting.validation_report import _purposes
from fosim.runner import run_config


def test_excel_audit_export(raw: dict[str, Any], tmp_path: Path) -> None:
    cfg = make_cfg(raw, **{"run.n_paths": 6, "run.horizon_years": 2, "spending.pct_nav_pa": 0.01})
    out = run_config(cfg, ledger_paths=[2])
    f = export_path_audit(out.results, out.paths, 2, tmp_path / "audit.xlsx")
    assert f.exists()
    sheets = pd.ExcelFile(f).sheet_names
    assert {"README", "A_balance_sheet", "A_ledger", "B_balance_sheet", "B_ledger", "B_tranches", "C_balance_sheet"} <= set(sheets)
    for name in ("A", "B", "C"):
        bs = balance_sheet_frame(out.results[name], out.paths, 2)
        assert np.nanmax(np.abs(bs["identity_residual"].to_numpy())) < 1e-4
        # ledger cash rows reconcile with the cash account step by step (excluding step 0 set-up rows)
        led = out.results[name].ledger.to_frame()
        cash_rows = led[(led.path == 2) & (led.account == "cash")]
        flows = cash_rows.groupby("step")["amount_usd"].sum()
        cash = bs["cash"].to_numpy()
        for k in range(1, out.paths.n_steps + 1):
            assert abs((cash[k] - cash[k - 1]) - float(flows.get(k, 0.0))) < 1e-3, (name, k)


def test_ic_report_builds(raw: dict[str, Any], tmp_path: Path) -> None:
    cfg = make_cfg(raw, **{"run.n_paths": 50, "run.horizon_years": 2})
    out = run_config(cfg)
    f = build_ic_report(out, tmp_path / "ic.html")
    txt = f.read_text(encoding="utf-8")
    assert "Executive summary" in txt and "PLACEHOLDER" in txt and "disclaimer" in txt.lower()
    assert "Put–call parity" in txt


def test_validation_report_purposes() -> None:
    from fosim.config import PROJECT_ROOT

    p = _purposes(PROJECT_ROOT / "tests")
    assert any(k.endswith("::test_01_hull_reference") for k in p)
    assert any("test 21" in v[0].lower() or "Spec test 21" in v[0] for v in p.values())
