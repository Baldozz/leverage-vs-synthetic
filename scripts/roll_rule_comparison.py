"""Compare the roll rules on the weekly grid: reports/roll_rule_comparison.md.

    .venv/bin/python scripts/roll_rule_comparison.py            # ≈ 10 minutes: four runs of every weekly start since 1997, held to today

Five rotations against the same loan portfolio — Keep's exposure (the default: a monthly check, 1 vol point off the mark on calls sold
early, calls bought below the band from the T-bills then from SPX sold for them), Keep's exposure with no haircut, Keep's exposure with a
quarterly check and the T-bills only below the band, the same dollar delta, the same index units — with the headline tables of the investor recap (value today,
the market bottoms) side by side, settled starts (every call expired at least once) apart from the unsettled ones, and the mechanics of
the target rule (band trades, unwind cost, exposure ratio, the ladder). Default setup otherwise (1 bn SPX, 250 m loan, SOFR + 75 bp,
75 % / 0 % / 90 % lending values, 5-year calls, 52 weekly steps, 15 % withholding).
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fosim.analytics.leverage_stress import (  # noqa: E402
    DAILY_FILE,
    _load,
    corrections,
    rolling_paths,
)

M = 1e6
RULES: dict[str, dict[str, object]] = {
    "Keep's exposure (default)": {"roll": "target"},
    "Keep's exposure (no haircut)": {"roll": "target", "unwind_haircut": 0.0},
    "Keep's exposure (quarterly, T-bills only below the band)": {"roll": "target", "rebalance": "quarterly", "below": "calls"},
    "Same dollar delta": {"roll": "delta"},
    "Same index units": {"roll": "units"},
}
PCT = [("lowest", 0.0), ("5th pct", 0.05), ("25th pct", 0.25), ("median", 0.5), ("75th pct", 0.75), ("95th pct", 0.95), ("highest", 1.0)]


def q(s: pd.Series, p: float) -> float:
    return float(s.min()) if p == 0.0 else float(s.max()) if p == 1.0 else float(s.quantile(p))


def md_table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|", *("| " + " | ".join(r) + " |" for r in rows)])


def main() -> None:
    d = _load(str(DAILY_FILE))
    last = pd.Timestamp(d.date.iloc[-1])
    starts = pd.date_range("1997-09-09", last - pd.DateOffset(days=7), freq="W-FRI")
    bottoms = corrections(0.20, on="price")
    marks = tuple(pd.Timestamp(b) for b in bottoms["trough"])
    settled_cut = last - pd.DateOffset(months=60) - pd.DateOffset(days=7 * 51)   # the last start whose 52nd tranche has expired
    t0 = time.perf_counter()
    runs: dict[str, tuple[pd.DataFrame, dict[str, pd.DataFrame]]] = {}
    for name, kw in RULES.items():
        print(f"{name}: {len(starts):,} starts …", flush=True)
        runs[name] = rolling_paths(starts, last, columns=("nav_A", "nav_B", "headroom_A", "dry_powder_B"), sample="M", mark_days=marks,
                                   progress=lambda i, n: print(f"  {i:,}/{n:,}", end="\r", flush=True), **kw)
        print()
    rows0 = runs[next(iter(RULES))][0]
    years = rows0["years"]
    held = years >= 1.0
    settled = pd.Series(rows0.index <= settled_cut, index=rows0.index)
    groups = {"Settled starts (every call expired at least once)": settled & held, "Unsettled starts (calls still open on the end day)": ~settled & held}

    out: list[str] = [
        "# Roll rules compared — every weekly start since 1997, held to today",
        "",
        f"Generated {date.today():%d %B %Y} by `scripts/roll_rule_comparison.py`. {len(starts):,} weekly starts, {starts[0]:%d %b %Y} → {starts[-1]:%d %b %Y}, "
        f"each held to {last:%d %b %Y}; the same loan portfolio (*Keep the loan*) on every row, the rotation under five roll rules. Default setup otherwise "
        "(1 bn SPX, 250 m Lombard loan at SOFR + 75 bp capitalised, lending values 75 % SPX / 0 % calls / 90 % T-bills, the bank calling at 90 % of the lending value, 5-year ATM calls on SPXFP, 52 weekly steps, 15 % withholding). "
        f"Settled starts: on or before {settled_cut:%d %b %Y}, so every call has expired at least once; unsettled: later starts, whose NAV today is partly a model mark. "
        "The annualised blocks use the starts held at least a year. USD m unless stated.",
        "",
        "**The rules.** *Keep's exposure* (default): at every expiry the replacement closes the gap between Keep's SPX value and what the rotation's holdings were bought "
        "to carry (its SPX plus each surviving call's slot, grown with the index), so each replacement restores its own slot; paid from the payoff and the T-bills, then by "
        "selling SPX; every month-end the live exposure (SPX + the calls' dollar delta) is brought back inside "
        "±10 % of Keep's — calls sold, most in the money first, at the model mark less 1 vol point (or at the mark: *no haircut*), calls bought from the T-bills and, once they are gone, "
        "from SPX sold for them, s = rest ÷ (δ/c − 1) (*quarterly, T-bills only*: the check every quarter and nothing sold to fund it — the default before 25 September). "
        "*Same dollar delta*: an in-the-money call is replaced by an ATM call with the same dollar delta (about twice the units; the option units double at every "
        "in-the-money expiry). *Same index units*: replaced on the same index units. Under the last two a worthless call is replaced on the same units, SPX is sold "
        "delta-for-delta and no band applies.",
        "",
    ]
    names = list(RULES)
    for gname, mask in groups.items():
        idx = rows0.index[mask]
        if len(idx) == 0:
            continue
        out += [f"## {gname}: value today ({len(idx):,} starts held at least a year)", ""]
        nav_a = rows0.loc[idx, "NAV A end"] / M
        ann_a = (1.0 + rows0.loc[idx, "A return"]) ** (1.0 / years.loc[idx]) - 1.0
        hdr = ["Final NAV (m)", "Keep the loan", *names]
        rows = [[lab, f"{q(nav_a, p):,.0f}", *(f"{q(runs[n][0].loc[idx, 'NAV B end'] / M, p):,.0f}" for n in names)] for lab, p in PCT]
        out += [md_table(hdr, rows), ""]
        hdr = ["Annualised return", "Keep the loan", *names]
        rows = [[lab, f"{q(ann_a, p):.1%}", *(f"{q((1.0 + runs[n][0].loc[idx, 'B return']) ** (1.0 / years.loc[idx]) - 1.0, p):.1%}" for n in names)] for lab, p in PCT]
        ahead = ["rotation ahead on (same start)", "", *(f"{(runs[n][0].loc[idx, 'NAV B end'] > nav_a * M).mean():.0%}" for n in names)]
        med_d = ["median Δ annualised, Rotate − Keep", "", *(f"{((1.0 + runs[n][0].loc[idx, 'B return']) ** (1.0 / years.loc[idx]) - 1.0 - ann_a).median() * 100:+.1f} pp" for n in names)]
        worst = ["worst Δ final NAV (m), same start", "", *(f"{(runs[n][0].loc[idx, 'NAV B end'] / M - nav_a).min():+,.0f}" for n in names)]
        best = ["best Δ final NAV (m), same start", "", *(f"{(runs[n][0].loc[idx, 'NAV B end'] / M - nav_a).max():+,.0f}" for n in names)]
        out += [md_table(hdr, [*rows, ahead, med_d, worst, best]), ""]
        dp = ["dry powder today, median (m)", f"{(rows0.loc[idx, 'A: headroom end'] / M).median():,.0f}", *(f"{(runs[n][0].loc[idx, 'B: dry powder end'] / M).median():,.0f}" for n in names)]
        calls = ["calls / NAV today, median", "", *(f"{(runs[n][0].loc[idx, 'end: calls B'] / runs[n][0].loc[idx, 'NAV B end']).median():.0%}" for n in names)]
        expo = ["exposure ÷ Keep's today, median", "1.00", *(f"{(runs[n][0].loc[idx, 'end: exposure B'] / runs[n][0].loc[idx, 'end: exposure A']).median():.2f}" for n in names)]
        cut = ["starts with a cut replacement", "", *(f"{int((runs[n][0].loc[idx, 'B: cut rolls'] > 0).sum()):,}" for n in names)]
        mc = ["margin calls (Keep)", f"{int(rows0.loc[idx, 'A: margin call'].sum()):,}", *("" for _ in names)]
        out += [md_table(["Today", "Keep the loan", *names], [dp, calls, expo, cut, mc]), ""]

    out += ["## At each market bottom (every start running that day)", "",
            "Lowest and median NAV of each portfolio on the bottom day across the starts running that day (the same starts on every column), "
            "the share of those starts where the rotation has more dry powder than Keep's room before a margin call, and where its NAV is higher. "
            "Starts after the settled cut run only into the 2022 bottom.", ""]
    hdr = ["Bottom", "starts", "Keep: lowest / median NAV", *(f"{n}: lowest / median NAV" for n in names), *(f"{n}: more dry powder / higher NAV" for n in names)]
    rows = []
    for label, b in zip(bottoms.index, marks, strict=True):
        pa = runs[names[0]][1]["nav_A"].loc[b].dropna()
        cells = [f"{label} ({b:%d %b %Y})", f"{len(pa):,}", f"{pa.min() / M:,.0f} / {pa.median() / M:,.0f}"]
        for n in names:
            pb = runs[n][1]["nav_B"].loc[b].dropna()
            cells.append(f"{pb.min() / M:,.0f} / {pb.median() / M:,.0f}")
        for n in names:
            ra, db = runs[n][1]["headroom_A"].loc[b].dropna(), runs[n][1]["dry_powder_B"].loc[b].dropna()
            pb = runs[n][1]["nav_B"].loc[b].dropna()
            cells.append(f"{(db > ra).mean():.0%} / {(pb > pa).mean():.0%}")
        rows.append(cells)
    out += [md_table(hdr, rows), ""]

    out += ["## The target rule's mechanics", ""]
    hdr = ["Per start (median, or total)", *names[:3]]
    mech = []
    for lab, col, fmt, how in [("rolls (expiries with a decision)", "rolls", "{:,.0f}", "median"), ("rolls skipped (gap ≤ 0)", "B: rolls skipped", "{:,.0f}", "median"),
                               ("rebalances up (calls sold)", "B: rebalances up", "{:,.0f}", "median"), ("rebalances down (calls bought)", "B: rebalances down", "{:,.0f}", "median"),
                               ("partial rebalances (T-bills short)", "B: partial rebalances", "{:,.0f}", "median"),
                               ("calls sold at rebalances (m)", "B: calls sold at rebalances", "{:,.0f}", "median_m"), ("calls bought at rebalances (m)", "B: calls bought at rebalances", "{:,.0f}", "median_m"),
                               ("SPX sold at rebalances (m)", "B: SPX sold at rebalances", "{:,.0f}", "median_m"),
                               ("unwind cost (m), median start", "B: unwind cost", "{:,.1f}", "median_m"), ("unwind cost (m), all starts", "B: unwind cost", "{:,.0f}", "sum_m"),
                               ("cut replacements", "B: cut rolls", "{:,.0f}", "sum")]:
        vals = []
        for n in names[:3]:
            s = runs[n][0][col]
            v = s.median() if how == "median" else s.median() / M if how == "median_m" else s.sum() / M if how == "sum_m" else s.sum()
            vals.append(fmt.format(v))
        mech.append([lab, *vals])
    ratio_rows = []
    for n in names[:3]:
        r = runs[n][0]["end: exposure B"] / runs[n][0]["end: exposure A"]
        ratio_rows.append(f"{r.quantile(0.05):.2f} / {r.median():.2f} / {r.quantile(0.95):.2f}")
    mech.append(["exposure ÷ Keep's today: 5th / median / 95th pct", *ratio_rows])
    out += [md_table(hdr, mech), "",
            "Each replacement restores its own slot; the band's purchases in a long fall take over the slots of the calls that then expire (skipped rolls), and its sales "
            "remove whole tranches over time, so the weekly ladder thins to a handful of calls. Between monthly checks the live exposure drifts with the delta; on the "
            "check days it is inside the band unless nothing was left to fund it (partial).", "",
            f"Run time {time.perf_counter() - t0:.0f} s."]
    target = ROOT / "reports" / "roll_rule_comparison.md"
    target.write_text("\n".join(out) + "\n")
    print(f"→ {target}")


if __name__ == "__main__":
    main()
