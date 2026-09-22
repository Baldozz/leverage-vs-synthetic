"""Acceptance (SPEC §13): the Streamlit app launches and every tab renders on the default config,
before and after a simulation run, without raising (Streamlit AppTest headless harness)."""

from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.mark.slow
def test_app_renders_all_tabs_and_runs() -> None:
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    # nine tabs present
    assert len(at.tabs) == 9
    # reduce paths for the test and run the simulation
    at.sidebar.number_input[0].set_value(200).run()
    at.sidebar.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Done" in s.value for s in at.sidebar.success)
    # every tab rendered content after the run (no exception anywhere)
    assert len(at.tabs) == 9
    assert at.dataframe  # summary tables exist


@pytest.mark.slow
def bt_end_for(tenor: float) -> date:
    """Last strike date whose maturity still falls inside the data, as the app should show it."""
    import sys

    sys.path.insert(0, str(APP.parents[1] / "src"))
    from fosim.analytics.call_vs_cash import strike_date_range

    return strike_date_range(tenor)[1].date()


def test_all_starts_page_weekly_grid() -> None:
    """Page 1: every week from 1997 to the last start whose calls have expired, held to today — two charts, the percentile table, the reading."""
    at = AppTest.from_file(str(APP.parent / "historical_app.py"), default_timeout=600)
    at.session_state["s_grid"] = "every week"
    at.switch_page("views/all_starts.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("plotly_chart")) == 3 and len(at.dataframe) == 1 and len(at.table) == 5   # two fans + the NAV-by-start-date chart; the corrections dataframe; tables: distribution, one per bottom (4)
    assert any("1,202 lines" in c.value for c in at.caption)   # empty selection = every start year
    pct = lambda cell: float(str(cell).rstrip("%")) / 100  # noqa: E731
    pts = ["lowest", "5th percentile", "25th percentile", "median", "75th percentile", "95th percentile", "highest"]
    cols_ = [" ", "Keep the loan", "Rotate into calls", "Δ Rotate into calls − Keep the loan"]
    stats = at.table[0].value   # tab 'Distribution': section 'Annualised return' (shaded section row, empty cells); Δ = Rotate − Keep of the row
    assert list(stats.columns) == cols_ and list(stats.iloc[:, 0]) == ["Annualised return to the end day", *pts, "rotation ahead on"]
    assert list(stats.iloc[0, 1:]) == ["", "", ""]
    assert 0.10 < pct(stats.iloc[4, 1]) < 0.20 and 0.10 < pct(stats.iloc[4, 2]) < 0.20   # median annualised return ≈ 15 % for both
    for i in range(1, 8):   # Δ in percentage points = Rotate − Keep of the row
        assert stats.iloc[i, 3].endswith(" pp") and abs(float(stats.iloc[i, 3][:-3]) - (pct(stats.iloc[i, 2]) - pct(stats.iloc[i, 1])) * 100) <= 0.11
    ra_ = [pct(c) for c in stats.iloc[1:8, 1]]
    assert ra_ == sorted(ra_) and stats.iloc[8, 3].endswith(" of 1,202 starts")   # 'ahead on' = the same start on both sides
    import sys
    sys.path.insert(0, str(APP.parents[1] / "src"))
    import pandas as pd

    from fosim.analytics.leverage_stress import rolling_paths, simulate
    m = 1e6
    pm = lambda x: f"{round(float(x) / m) + 0.0:+,.0f} m"  # noqa: E731
    # NAV on the end day by start date: Keep, Rotate and the net (Rotate − Keep) on every weekly start, the first point checked against a direct simulation
    import json
    by_start = json.loads(at.get("plotly_chart")[2].proto.spec)["data"]
    assert [t["name"] for t in by_start] == ["Keep the loan", "Rotate into calls", "rotation ahead", "rotation behind", "net: Rotate into calls − Keep the loan"]
    ka, kb, kn = by_start[0], by_start[1], by_start[4]
    assert len(ka["x"]) == len(kb["x"]) == len(kn["x"]) == 1202 and ka["x"][0].startswith("1997-09-12") and ka["x"][-1].startswith("2020-09-18")
    p0, _ = simulate("1997-09-12", "2026-09-14")
    assert ka["y"][0] == pytest.approx(p0["nav_A"].iloc[-1] / m, rel=1e-9) and kb["y"][0] == pytest.approx(p0["nav_B"].iloc[-1] / m, rel=1e-9)
    assert all(n_ == pytest.approx(b_ - a_, abs=1e-9) for a_, b_, n_ in zip(ka["y"], kb["y"], kn["y"], strict=True)) and min(kn["y"]) < 0 < max(kn["y"])
    assert by_start[2]["y"] == [max(v, 0.0) for v in kn["y"]] and by_start[3]["y"] == [min(v, 0.0) for v in kn["y"]]   # the shaded areas split the net at zero
    lay = json.loads(at.get("plotly_chart")[2].proto.spec)["layout"]
    assert lay["legend"]["y"] < 0 and lay["yaxis"]["range"][1] == pytest.approx(max(max(ka["y"]), max(kb["y"])) * 1.05)   # legend under the chart; the NAV axis follows the highest value in the zoom window
    # the profiles beside the fans: the lowest, median and highest final NAV are labelled on the right axis
    fan_a = json.loads(at.get("plotly_chart")[0].proto.spec)["layout"]["yaxis2"]
    assert [t.split(" ")[0] for t in fan_a["ticktext"]] == ["min", "median", "max"] and fan_a["tickvals"][0] == pytest.approx(min(ka["y"])) and fan_a["tickvals"][2] == pytest.approx(max(ka["y"]))
    assert fan_a["tickvals"][1] == pytest.approx(float(pd.Series(ka["y"]).median()))
    # the zoom slider narrows both the fans and the chart by start date
    at.slider(key="r_zoom").set_value((2012, 2016)).run()
    assert not at.exception, [e.value for e in at.exception]
    lay_z = json.loads(at.get("plotly_chart")[2].proto.spec)["layout"]
    assert lay_z["xaxis2"]["range"][0].startswith("2012-01-01") and lay_z["xaxis2"]["range"][1].startswith("2016-12-31")
    zoomed = [y_ for t in (ka, kb) for x_, y_ in zip(t["x"], t["y"], strict=True) if "2012-01-01" <= x_[:10] <= "2016-12-31"]   # the NAV axis follows the highest value of either portfolio inside the window
    assert lay_z["yaxis"]["range"][1] == pytest.approx(max(zoomed) * 1.05) and max(zoomed) < max(kb["y"])
    at.slider(key="r_zoom").set_value((1997, 2026)).run()
    assert not at.exception, [e.value for e in at.exception]
    bottoms = at.dataframe[0].value
    assert list(bottoms["Correction"]) == ["2000–2002", "2007–2009", "2020", "2022"] and [str(d) for d in bottoms["Bottom"]] == ["2002-10-09", "2009-03-09", "2020-03-23", "2022-10-12"]
    assert [round(v) for v in bottoms["SPX at the bottom"]] == [777, 677, 2237, 3577] and [str(d) for d in bottoms["Previous peak"]] == ["2000-03-24", "2007-10-09", "2020-02-19", "2022-01-03"]
    assert list(bottoms.columns[-2:]) == ["Lowest NAV, Keep the loan", "Lowest NAV, Rotate into calls"] and bottoms.iloc[:, -2:].notna().all().all()
    gfc = at.table[2].value   # one table per bottom, in order: 2002, 2009, 2020, 2022; the GFC bottom's worst levered trajectory started on the dot-com peak day
    detail = ["started on", "NAV", "SPX held, market value", "calls held, market value", "cash", "loan", "lending value of the holdings", "dry powder = lending value − loan + cash",
              "interest paid since the start", "premiums paid − payoffs received since the start", "dividends received since the start (net of 15% withholding, reinvested in the SPX)"]
    assert list(gfc.columns) == cols_
    # one block (the lowest NAV that day, both strategies read on it), then the count row
    assert list(gfc.iloc[:12, 0]) == ["Worst trajectory: the lowest NAV that day", *detail] and gfc.iloc[12, 0] == "rotation has more dry powder on" and len(gfc) == 13
    assert list(gfc.iloc[0, 1:]) == ["", "", ""] and gfc.iloc[12, 3].endswith(" of 600 starts running that day")
    # the worst trajectory: the lowest NAV that day in either portfolio (Keep, the 24 Mar 2000 start), both strategies read on that start, Δ on every row
    assert gfc.iloc[1, 1] == "24 Mar 2000, 2755 days before the peak, SPX 1,527 (-2.4% vs the peak)" and gfc.iloc[1, 2] == "same start (the lowest NAV that day: Keep the loan)" and gfc.iloc[1, 3] == ""
    assert gfc.iloc[2, 1].startswith("142 m (-81% from the start)")
    assert gfc.iloc[8, 1].startswith("381 − 366 = 15 m (LTV 96%")   # the loan is deducted: 381 − 366 = 15
    # the cells are the daily simulation's values on 9 Mar 2009 for the 24 Mar 2000 start: recompute it directly
    pa, _ = simulate("2000-03-24", "2026-09-14")
    da = pa.loc["2009-03-09"]
    assert gfc.iloc[2, 1] == f"{da['nav_A'] / m:,.0f} m ({da['nav_A'] / m / 750 - 1:+.0%} from the start)" and gfc.iloc[2, 2] == f"{da['nav_B'] / m:,.0f} m ({da['nav_B'] / m / 750 - 1:+.0%} from the start)" and gfc.iloc[2, 3] == pm(da["nav_B"] - da["nav_A"])
    assert gfc.iloc[3, 1] == f"{da['E_A'] / m:,.0f} m" and gfc.iloc[3, 2] == f"{da['E_B'] / m:,.0f} m" and gfc.iloc[3, 3] == pm(da["E_B"] - da["E_A"])
    # calls bought by that day = the 52 build tranches + the one replacement of March 2006 (the only tranche of the March-2000 start that expired in the money)
    assert int(da["n_bought"]) == 53 and gfc.iloc[4, 1] == "—" and gfc.iloc[4, 3] == pm(da["call_val"])
    assert gfc.iloc[4, 2] == f"{da['call_val'] / m:,.0f} m (1 call alive of the 53 bought, on a notional of {da['call_notional'] / m:,.0f} m of index)"
    assert gfc.iloc[5, 1] == "—" and gfc.iloc[5, 2] == f"{da['cash_B'] / m:,.0f} m" and gfc.iloc[5, 3] == pm(da["cash_B"])
    assert gfc.iloc[6, 1] == f"{da['loan'] / m:,.0f} m" and gfc.iloc[6, 2] == "0 m" and gfc.iloc[6, 3] == pm(-da["loan"])   # the rotation of March 2000 has no loan since 2001
    assert gfc.iloc[7, 1] == f"{0.75 * da['E_A'] / m:,.0f} m (75% of the SPX)" and gfc.iloc[7, 2] == f"{0.75 * da['E_B'] / m:,.0f} m (75% of the SPX + 0% of the calls)" and gfc.iloc[7, 3] == pm(0.75 * (da["E_B"] - da["E_A"]))
    assert gfc.iloc[8, 1].endswith(f"= {da['headroom_A'] / m:,.0f} m (LTV {da['ltv_A']:.0%}: a further 4% SPX fall to a margin call)") and gfc.iloc[8, 2].endswith(f"= {da['dry_powder_B'] / m:,.0f} m")
    assert gfc.iloc[8, 3] == pm(da["dry_powder_B"] - da["headroom_A"]) and da["dry_powder_B"] > da["headroom_A"] and da["nav_B"] > da["nav_A"]
    # cost since the start on two rows: the interest on each loan, and the premiums less the payoffs of the calls
    assert gfc.iloc[9, 1] == f"{da['interest_cum_A'] / m:,.0f} m" and gfc.iloc[9, 2] == f"{da['interest_cum_B'] / m:,.0f} m (during the build)" and gfc.iloc[9, 3] == pm(da["interest_cum_B"] - da["interest_cum_A"])
    assert da["interest_cum_A"] == pytest.approx(pa.loc[:"2009-03-09", "interest_A"].sum())
    assert da["interest_cum_B"] == pytest.approx(pa["interest_B"].sum()) and 5e6 < da["interest_cum_B"] < 15e6   # ≈ 9 m: 250 m at ≈ 7 % repaid over a year
    net_b = da["premiums_cum_B"] - da["payoffs_cum_B"]
    assert gfc.iloc[10, 1] == "—" and gfc.iloc[10, 2] == f"{da['premiums_cum_B'] / m:,.0f} m − {da['payoffs_cum_B'] / m:,.0f} m = {net_b / m:,.0f} m" and gfc.iloc[10, 3] == pm(net_b)
    # dividends since the start, both portfolios: the rotation holds fewer SPX units so it received less; both totals reconcile with the SPX total return (engine test)
    assert gfc.iloc[11, 1] == f"{da['div_cum_A'] / m:,.0f} m" and gfc.iloc[11, 2] == f"{da['div_cum_B'] / m:,.0f} m" and gfc.iloc[11, 3] == pm(da["div_cum_B"] - da["div_cum_A"])
    assert 0.0 < da["div_cum_B"] < da["div_cum_A"] and da["div_cum_A"] == pytest.approx(pa.loc[:"2009-03-09", "div_A"].sum())
    assert int(da["n_calls"]) == 1 and da["loan_B"] == 0.0
    # the picks and the count over every weekly start running on 9 Mar 2009 (the page's grid): recompute all from the engine on the same starts
    bottom = pd.Timestamp("2009-03-09")
    # run past the bottom: `rolling_paths` skips a start whose build is not complete 30 days before the end date, and the page runs every start to today
    _, paths = rolling_paths(pd.date_range("1997-09-09", bottom, freq="W-FRI"), "2010-06-30", columns=("nav_A", "nav_B", "headroom_A", "dry_powder_B"), sample="M", mark_days=(bottom,))
    na, nb, ra, db_ = (paths[c].loc[bottom] for c in ("nav_A", "nav_B", "headroom_A", "dry_powder_B"))
    assert len(na) == 600 and na.notna().all() and "600 starts running that day" in " ".join(mk.value for mk in at.markdown)
    assert na.min() == pytest.approx(da["nav_A"]) and na.idxmin() == pd.Timestamp("2000-03-24")   # the worst trajectory is the lowest NAV that day over all 600 starts
    # the corrections dataframe: the lowest NAV of each portfolio on the bottom day, over the same 600 starts (the GFC row)
    assert bottoms.iloc[1, -2] == pytest.approx(na.min() / m) and bottoms.iloc[1, -1] == pytest.approx(nb.min() / m) and bottoms.iloc[1, -2] == pytest.approx(da["nav_A"] / m)
    assert gfc.iloc[12, 3] == f"{int((db_ > ra).sum()):,} of 600 starts running that day"   # the count row, recomputed from the engine on the same 600 starts
    assert 560 <= int((db_ > ra).sum()) <= 600 and nb.min() > na.min()   # at the GFC bottom the rotation has more dry powder on (almost) every start
    assert at.table[1].value.iloc[1, 1].startswith("24 Mar 2000, on the peak day, SPX 1,527 (+0.0% vs the peak)")   # the 2002 bottom: the worst start is the peak day too
    assert at.table[1].value.iloc[1, 2] == "same start (the lowest NAV that day: Keep the loan)"
    # the 2020 bottom came five weeks after the peak: the worst start had done only a few of its 52 weekly steps, and the cell says so (not "of the 52 bought")
    covid = at.table[3].value
    w20 = pd.Timestamp(covid.iloc[1, 1].split(",")[0])
    p20, _ = simulate(w20, "2026-09-14")
    d20 = p20.loc["2020-03-23"]
    assert 1 <= int(d20["n_bought"]) < 52 and int(d20["n_calls"]) == int(d20["n_bought"]) and d20["loan_B"] > 0.5e6
    assert covid.iloc[4, 2] == f"{d20['call_val'] / m:,.0f} m ({int(d20['n_calls'])} calls alive of the {int(d20['n_bought'])} bought so far: {int(d20['n_bought'])} of the 52 weekly steps done, on a notional of {d20['call_notional'] / m:,.0f} m of index)"
    assert covid.iloc[6, 2] == f"{d20['loan_B'] / m:,.0f} m (rotation still under way)"
    assert any(mk.value.startswith("**2022 bottom — 12 Oct 2022**") for mk in at.markdown) and len(at.table[4].value) == 13   # the 2022 correction has its table too
    assert any("Same start, same end" in m.value for m in at.markdown)
    at.multiselect(key="r_years").set_value([2008, 2009]).run()   # a few start years only
    assert not at.exception, [e.value for e in at.exception]
    assert any("stopped buying calls after a correction" in c.value and "105 lines" in c.value for c in at.caption) and "105 selected starts" in " ".join(h.value for h in at.subheader)
    assert any(m.value.startswith("- Of the 105 selected starts, the rotated portfolio lost **all** its calls on **0%** (0)") for m in at.markdown)   # the closing lines follow the selection: no 2008–09 start lost a call
    at.multiselect(key="r_years").set_value([]).run()
    gfc = next(o for o in at.selectbox(key="r_end").options if "2007–2009 bottom" in o)   # held until the GFC bottom: every start that completed its rotation a month before 9 Mar 2009
    at.selectbox(key="r_end").set_value(gfc).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("each held to 09 Mar 2009" in m.value for m in at.markdown) and "On 09 Mar 2009" in " ".join(h.value for h in at.subheader)
    stats = at.table[0].value
    assert pct(stats.iloc[4, 1]) < 0 and pct(stats.iloc[1, 1]) < -0.15   # at the GFC bottom most starts are under water
    assert len(at.table) == 3 and len(at.dataframe) == 1   # distribution; only the 2002 and 2009 bottoms fall inside runs ending 9 Mar 2009
    assert at.table[2].value.iloc[8, 1].startswith("381 − 366 = 15 m (LTV 96%")   # the worst levered trajectory at the bottom, same as when held to today   # at the GFC bottom the median levered start is well above half its lending value
    assert any("lost **all** its calls" in m.value for m in at.markdown) and any("Margin calls keeping the loan: **0** of the" in m.value for m in at.markdown)
    # exclude the rotations that stopped buying calls: fewer starts everywhere below the fans, the worst trajectory no longer has a worthless expiry
    at.selectbox(key="r_end").set_value(at.selectbox(key="r_end").options[0]).run()
    at.checkbox(key="r_excl").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    n_kept = int(at.table[0].value.iloc[8, 3].split(" of ")[1].split(" starts")[0].replace(",", ""))
    assert 600 < n_kept < 1202 and any("stopped buying calls excluded" in h.value for h in at.subheader) and any("are excluded (both portfolios)" in m.value for m in at.markdown)
    assert any(f"{n_kept:,} lines" in c.value and "0 of the" in c.value for c in at.caption)   # the fans drop the same starts in both portfolios: no orange line left
    assert pct(at.table[0].value.iloc[1, 2]) > 0.067   # the worst rotation is no longer the Sep-2000 start (+6.7 %)
    assert any(f"; {n:,} starts running that day" in m.value for m in at.markdown for n in range(1, 600))   # the 2009 tab counts fewer starts too


def test_premium_history_page_unchanged() -> None:
    """Page 2 (call-vs-cash backtest): its own sidebar (the tenor and the withholding tax shared with page 1), no tabs, chart + tables at default widgets;
    the shared tenor drives every label; the page's inputs survive a visit to page 1."""
    at = AppTest.from_file(str(APP.parent / "historical_app.py"), default_timeout=300)
    at.switch_page("views/premium_history.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 0 and len(at.columns) == 0   # the inputs are in the sidebar, not in columns at the top
    assert [w.label for w in at.sidebar.number_input] == ["Premium / investment (USD m)", "Fixed premium (% of notional)", "Dividend withholding tax (%)"]
    assert [w.label for w in at.sidebar.selectbox] == ["Call tenor (years)"] and [w.label for w in at.sidebar.radio] == ["Premium paid", "Invested in"]
    assert [w.label for w in at.sidebar.date_input] == ["First strike date", "Last strike date"]
    assert len(at.dataframe) == 2  # summary table + return-distribution percentiles
    assert all(len(d.value) <= 12 for d in at.dataframe)  # both fit the fixed height set in the app (12 rows)
    assert not at.warning  # 5-year tenor: no fallback warning
    assert at.date_input(key="bt_end").value == bt_end_for(5.0)
    mode = lambda: at.sidebar.radio[0]  # noqa: E731  (the premium-mode radio carries the tenor in its options, so it has no key)
    for t in (7.0, 10.0):  # tenor-specific columns exist: no fallback warning
        at.selectbox(key="s_tenor").set_value(int(t)).run()
        assert not at.exception, [e.value for e in at.exception]
        assert not at.warning
        assert any(f"{t:g}-year ATM call" in m.value for m in at.markdown)
        assert any(f"{t:g}y Treasury of the day" in o for o in mode().options)  # premium-mode option follows the tenor
        assert at.date_input(key="bt_end").value == bt_end_for(t)  # last strike date follows the tenor
    mode().set_value("Fixed % of notional").run()
    at.selectbox(key="s_tenor").set_value(5).run()
    assert mode().value == "Fixed % of notional"  # premium mode survives a tenor change
    assert any("fixed at" in m.value for m in at.markdown)
    # the page's own inputs survive a visit to page 1 (whose sidebar does not draw them), and the shared tenor is the same widget on both pages
    at.number_input(key="bt_prem_usd").set_value(20.0).run()
    at.selectbox(key="s_tenor").set_value(7).run()
    assert any("20 m in SPXFP" in m.value for m in at.markdown)
    at.session_state["s_grid"] = "every week"
    at.switch_page("views/all_starts.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="s_tenor").value == 7 and not any(w.key == "bt_prem_usd" for w in at.sidebar.number_input)   # tenor shared; page 2's inputs not on page 1
    assert any("7-year ATM calls on SPXFP" in c.value for c in at.caption)
    at.switch_page("views/premium_history.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.number_input(key="bt_prem_usd").value == 20.0 and at.selectbox(key="s_tenor").value == 7 and mode().value == "Fixed % of notional"
    assert any("7-year ATM call on SPXFP vs. 20 m in SPXFP" in m.value for m in at.markdown)


@pytest.mark.slow
def test_strategy_replay_app_runs_1997_replay() -> None:
    """The supporting strategy-replay app renders its three tabs, runs the 1997→2026 replay with default widgets and shows results."""
    at = AppTest.from_file(str(APP.parent / "strategy_replay_app.py"), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 3
    assert any("Done" in s.value for s in at.success)
    assert at.dataframe
