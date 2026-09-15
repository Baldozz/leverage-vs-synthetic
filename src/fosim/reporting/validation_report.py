"""Generate ``reports/validation_report.html``: every test, its purpose (from the module and test
docstrings) and pass/fail, from one command:  ``python -m fosim.reporting.validation_report``."""

from __future__ import annotations

import ast
import datetime as dt
import html
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from fosim.config import PROJECT_ROOT


def _purposes(tests_dir: Path) -> dict[str, tuple[str, str]]:
    """Map 'module::test' -> (module purpose, test purpose) from docstrings."""
    out: dict[str, tuple[str, str]] = {}
    for f in sorted(tests_dir.glob("test_*.py")):
        tree = ast.parse(f.read_text())
        mod_doc = (ast.get_docstring(tree) or "").strip()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                doc = (ast.get_docstring(node) or "").strip()
                if not doc:
                    # first comment-like line inside the function is not accessible; use the name
                    doc = node.name.replace("_", " ")
                out[f"{f.stem}::{node.name}"] = (mod_doc, doc)
    return out


def run(out_html: Path | None = None, junit: Path | None = None, pytest_args: list[str] | None = None) -> Path:
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    out_html = out_html or reports / "validation_report.html"
    junit = junit or reports / "pytest_junit.xml"
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}", *(pytest_args or [])]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    tree = ET.parse(junit)
    purposes = _purposes(PROJECT_ROOT / "tests")
    rows = []
    n_pass = n_fail = n_skip = 0
    for tc in tree.iter("testcase"):
        cls = tc.get("classname", "")
        mod = cls.split(".")[-1]
        name = tc.get("name", "")
        base = name.split("[")[0]
        status = "PASS"
        detail = ""
        if tc.find("failure") is not None or tc.find("error") is not None:
            status = "FAIL"
            node = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
            detail = (node.get("message") or "")[:300] if node is not None else ""
            n_fail += 1
        elif tc.find("skipped") is not None:
            status = "SKIP"
            n_skip += 1
        else:
            n_pass += 1
        mod_doc, test_doc = purposes.get(f"{mod}::{base}", ("", base.replace("_", " ")))
        rows.append((mod, name, test_doc, status, float(tc.get("time", "0")), detail, mod_doc))
    rows.sort(key=lambda r: (r[0], r[1]))
    css = "body{font-family:Helvetica,Arial,sans-serif;margin:24px;color:#222}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:6px;font-size:13px;vertical-align:top}th{background:#f2f2f2;text-align:left}.PASS{color:#137333;font-weight:600}.FAIL{color:#b00020;font-weight:700}.SKIP{color:#8a6d00}.mod{background:#fafafa;font-weight:600}"
    parts = [
        f"<html><head><meta charset='utf-8'><title>Validation report</title><style>{css}</style></head><body>",
        "<h1>Validation report — leverage vs. long-dated call replacement simulator</h1>",
        f"<p>Generated {dt.datetime.now():%Y-%m-%d %H:%M} · pytest exit code {proc.returncode} · "
        f"<b class='PASS'>{n_pass} passed</b> · <b class='FAIL'>{n_fail} failed</b> · <b class='SKIP'>{n_skip} skipped</b></p>",
        "<p>Each row is one pytest test; the purpose column is the test's docstring (or its name), the module purpose lists the spec test numbers covered.</p>",
        "<table><tr><th>Module</th><th>Test</th><th>Purpose</th><th>Status</th><th>Time (s)</th><th>Detail</th></tr>",
    ]
    last_mod = None
    for mod, name, purpose, status, t, detail, mod_doc in rows:
        if mod != last_mod:
            parts.append(f"<tr class='mod'><td colspan='6'>{html.escape(mod)} — {html.escape(mod_doc.splitlines()[0] if mod_doc else '')}</td></tr>")
            last_mod = mod
        parts.append(f"<tr><td>{html.escape(mod)}</td><td>{html.escape(name)}</td><td>{html.escape(purpose)}</td><td class='{status}'>{status}</td><td>{t:.2f}</td><td>{html.escape(detail)}</td></tr>")
    parts.append("</table>")
    parts.append(f"<h2>pytest output</h2><pre>{html.escape(proc.stdout[-6000:])}</pre></body></html>")
    out_html.write_text("\n".join(parts), encoding="utf-8")
    return out_html


if __name__ == "__main__":
    p = run(pytest_args=sys.argv[1:])
    print(f"written {p}")
