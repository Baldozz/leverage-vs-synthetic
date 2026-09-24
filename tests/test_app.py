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
    # nothing runs until the button is pressed: an info message, no chart, the button enabled
    assert len(at.get("plotly_chart")) == 0 and len(at.table) == 0 and any("press **Launch simulation**" in i.value for i in at.info) and not at.button(key="r_launch").proto.disabled
    # the sidebar defaults: Keep's exposure with the quarterly band; the band widgets enabled, the surplus radio greyed out
    assert at.radio(key="s_roll").value == "Keep's exposure" and at.radio(key="s_rebalance").value == "every quarter" and at.number_input(key="s_band").value == 10.0 and at.number_input(key="s_haircut").value == 1.0
    assert at.radio(key="s_below").value == "ATM calls from the T-bills" and list(at.radio(key="s_below").options) == ["ATM calls from the T-bills", "SPX from the T-bills", "ATM calls, SPX sold for them when the T-bills run out"] and not at.radio(key="s_rebalance").proto.disabled and at.radio(key="s_surplus").proto.disabled and any("Not used under Keep's exposure" in c.value for c in at.caption)
    # this block runs the same dollar delta rule (the values below were established on it); the default rule is launched further down
    at.radio(key="s_roll").set_value("the same dollar delta").run()
    assert at.radio(key="s_rebalance").proto.disabled and at.number_input(key="s_band").proto.disabled and not at.radio(key="s_surplus").proto.disabled
    at.button(key="r_launch").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.button(key="r_launch").proto.disabled and not at.info   # launched: greyed out until something changes
    assert len(at.get("plotly_chart")) == 5 and len(at.dataframe) == 1 and len(at.table) == 5   # two fans + NAV and dry powder by start date + the exposure ratio; the corrections dataframe; tables: distribution, one per bottom (4)
    assert any("1,513 lines" in c.value for c in at.caption)   # every weekly start from 12 Sep 1997 to 4 Sep 2026 (a week before the end day)
    assert any("**50 of the 1,513 starts (after 26 Sep 2025) were still rotating on 14 Sep 2026**" in c.value for c in at.caption)   # a loan left on the end day: the 52nd step of a 26 Sep 2025 start falls on the last day
    assert any("**311 starts (after 22 Sep 2020) have calls that have not yet expired**" in c.value for c in at.caption)   # the starts beyond the "all calls expired" cut
    pct = lambda cell: float(str(cell).rstrip("%")) / 100  # noqa: E731
    pts = ["lowest", "5th percentile", "25th percentile", "median", "75th percentile", "95th percentile", "highest"]
    cols_ = [" ", "Keep the loan", "Rotate into calls", "Δ Rotate into calls − Keep the loan"]
    stats = at.table[0].value   # tab 'Distribution': section 'Annualised return' (shaded section row, empty cells); Δ = Rotate − Keep of the row
    assert list(stats.columns) == cols_ and list(stats.iloc[:, 0]) == ["Annualised return to the end day (1,462 starts held at least a year)", *pts, "rotation ahead on"]
    assert list(stats.iloc[0, 1:]) == ["", "", ""]
    assert 0.10 < pct(stats.iloc[4, 1]) < 0.20 and 0.10 < pct(stats.iloc[4, 2]) < 0.20   # median annualised return ≈ 15 % for both
    for i in range(1, 8):   # Δ in percentage points = Rotate − Keep of the row
        assert stats.iloc[i, 3].endswith(" pp") and abs(float(stats.iloc[i, 3][:-3]) - (pct(stats.iloc[i, 2]) - pct(stats.iloc[i, 1])) * 100) <= 0.11
    ra_ = [pct(c) for c in stats.iloc[1:8, 1]]
    assert ra_ == sorted(ra_) and stats.iloc[8, 3].endswith(" of 1,462 starts")   # 'ahead on' = the same start on both sides; the 51 starts held less than a year are left out of the annualised block
    import sys
    sys.path.insert(0, str(APP.parents[1] / "src"))
    import pandas as pd

    from fosim.analytics.leverage_stress import rolling_paths, simulate
    m = 1e6
    pm = lambda x: f"{round(float(x) / m) + 0.0:+,.0f} m"  # noqa: E731
    # NAV on the end day by start date: Keep, Rotate and the net (Rotate − Keep) on every weekly start, the first point checked against a direct simulation
    import json
    by_start = json.loads(at.get("plotly_chart")[2].proto.spec)["data"]
    assert [t["name"] for t in by_start] == ["Keep the loan", "Rotate into calls", "rotation ahead", "rotation behind", "net: Rotate into calls − Keep the loan", "lowest Keep the loan", "lowest Rotate into calls"]
    ka, kb, kn = by_start[0], by_start[1], by_start[4]
    assert len(ka["x"]) == len(kb["x"]) == len(kn["x"]) == 1513 and ka["x"][0].startswith("1997-09-12") and ka["x"][-1].startswith("2026-09-04")
    # the lowest NAV of each portfolio (whole window): a marker on the argmin of its own trace, labelled with the value in its colour, the start date and the other portfolio's value
    i_a, i_b = min(range(1513), key=lambda j: ka["y"][j]), min(range(1513), key=lambda j: kb["y"][j])
    assert by_start[5]["x"] == [ka["x"][i_a]] and by_start[5]["y"][0] == pytest.approx(ka["y"][i_a]) and by_start[6]["x"] == [kb["x"][i_b]] and by_start[6]["y"][0] == pytest.approx(kb["y"][i_b])
    texts = [a["text"] for a in json.loads(at.get("plotly_chart")[2].proto.spec)["layout"]["annotations"]]
    assert f"<span style='color:#c00000'><b>{ka['y'][i_a]:,.0f} m</b></span> · {pd.Timestamp(ka['x'][i_a]):%d %b %Y} · <span style='color:#0b2a6f'>{kb['y'][i_a]:,.0f} m</span>" in texts
    assert f"<span style='color:#0b2a6f'><b>{kb['y'][i_b]:,.0f} m</b></span> · {pd.Timestamp(kb['x'][i_b]):%d %b %Y} · <span style='color:#c00000'>{ka['y'][i_b]:,.0f} m</span>" in texts
    assert {t for t in texts if t.startswith(("peak", "bottom"))} == {"peak Mar 2000", "bottom Oct 2002", "peak Oct 2007", "bottom Mar 2009", "peak Feb 2020", "bottom Mar 2020", "peak Jan 2022", "bottom Oct 2022"}
    p0, _ = simulate("1997-09-12", "2026-09-14", roll="delta")   # the sidebar's roll rule
    assert ka["y"][0] == pytest.approx(p0["nav_A"].iloc[-1] / m, rel=1e-9) and kb["y"][0] == pytest.approx(p0["nav_B"].iloc[-1] / m, rel=1e-9)
    assert all(n_ == pytest.approx(b_ - a_, abs=1e-9) for a_, b_, n_ in zip(ka["y"], kb["y"], kn["y"], strict=True)) and min(kn["y"]) < 0 < max(kn["y"])
    assert by_start[2]["y"] == [max(v, 0.0) for v in kn["y"]] and by_start[3]["y"] == [min(v, 0.0) for v in kn["y"]]   # the shaded areas split the net at zero
    lay = json.loads(at.get("plotly_chart")[2].proto.spec)["layout"]
    assert lay["legend"]["y"] < 0 and lay["yaxis"]["range"][1] == pytest.approx(max(max(ka["y"]), max(kb["y"])) * 1.05)   # legend under the chart; the NAV axis follows the highest value in the zoom window
    # dry powder on the end day by start date: Keep's room before a margin call and its further fall to a call, Rotate's dry powder, the net — the first start checked against the same simulation
    dp = json.loads(at.get("plotly_chart")[3].proto.spec)["data"]
    assert [t["name"] for t in dp] == ["Keep the loan: room before a margin call", "Rotate into calls: dry powder", "Keep the loan: LTV (right axis; yellow far from a margin call, orange close to it)",
                                       "rotation has more", "rotation has less", "net dry powder: Rotate into calls − Keep the loan", "lowest Keep the loan", "lowest Rotate into calls"]
    e0 = p0.iloc[-1]
    assert dp[0]["y"][0] == pytest.approx(e0["headroom_A"] / m, rel=1e-9) and dp[1]["y"][0] == pytest.approx(e0["dry_powder_B"] / m, rel=1e-9)
    assert dp[2]["y"][0] == pytest.approx(e0["ltv_A"] * 100, rel=1e-9) and dp[2]["marker"]["color"] == dp[2]["y"] and all(0 < v < 100 for v in dp[2]["y"]) and len(dp[2]["x"]) == 1513   # LTV dots, coloured by their own value
    assert all(n_ == pytest.approx(b_ - a_, abs=1e-9) for a_, b_, n_ in zip(dp[0]["y"], dp[1]["y"], dp[5]["y"], strict=True))
    j_a = min(range(1513), key=lambda j: dp[0]["y"][j])
    assert dp[6]["x"] == [dp[0]["x"][j_a]] and dp[6]["y"][0] == pytest.approx(dp[0]["y"][j_a])   # the lowest room before a margin call, marked
    # the profiles beside the fans: the lowest, median and highest final NAV are labelled on the right axis
    fan_a = json.loads(at.get("plotly_chart")[0].proto.spec)["layout"]["yaxis2"]
    assert [t.split(" ")[0] for t in fan_a["ticktext"]] == ["min", "median", "max"] and fan_a["tickvals"][0] == pytest.approx(min(ka["y"])) and fan_a["tickvals"][2] == pytest.approx(max(ka["y"]))
    assert fan_a["tickvals"][1] == pytest.approx(float(pd.Series(ka["y"]).median()))
    # the zoom slider narrows both the fans and the chart by start date
    at.slider(key="r_zoom_1997_2026").set_value((2012, 2016)).run()   # the slider is keyed on its range (the launched start year and end day)
    assert not at.exception, [e.value for e in at.exception]
    lay_z = json.loads(at.get("plotly_chart")[2].proto.spec)["layout"]
    assert lay_z["xaxis2"]["range"][0].startswith("2012-01-01") and lay_z["xaxis2"]["range"][1].startswith("2017-01-30")   # 30 days past the window's end, so the last peak / bottom stays in view
    zoomed = [y_ for t in (ka, kb) for x_, y_ in zip(t["x"], t["y"], strict=True) if "2012-01-01" <= x_[:10] <= "2016-12-31"]   # the NAV axis follows the highest value of either portfolio inside the window
    assert lay_z["yaxis"]["range"][1] == pytest.approx(max(zoomed) * 1.05) and max(zoomed) < max(kb["y"])
    at.slider(key="r_zoom_1997_2026").set_value((1997, 2026)).run()
    assert not at.exception, [e.value for e in at.exception]
    bottoms = at.dataframe[0].value
    assert list(bottoms["Correction"]) == ["2000–2002", "2007–2009", "2020", "2022"] and [str(d) for d in bottoms["Bottom"]] == ["2002-10-09", "2009-03-09", "2020-03-23", "2022-10-12"]
    assert [round(v) for v in bottoms["SPX at the bottom"]] == [777, 677, 2237, 3577] and [str(d) for d in bottoms["Previous peak"]] == ["2000-03-24", "2007-10-09", "2020-02-19", "2022-01-03"]
    assert list(bottoms.columns[-2:]) == ["Lowest NAV, Keep the loan", "Lowest NAV, Rotate into calls"] and bottoms.iloc[:, -2:].notna().all().all()
    gfc = at.table[2].value   # one table per bottom, in order: 2002, 2009, 2020, 2022; the GFC bottom's worst levered trajectory started on the dot-com peak day
    detail = ["started on", "NAV", "SPX held, market value", "T-bills (cash)", "calls held, market value", "loan", "lending value of the holdings", "dry powder = lending value − loan", "interest cumulated"]
    assert list(gfc.columns) == cols_
    # one block (the lowest NAV that day, both strategies read on it), then the count row
    assert list(gfc.iloc[:10, 0]) == ["Worst trajectory: the lowest NAV that day", *detail] and gfc.iloc[10, 0] == "rotation has more dry powder on" and len(gfc) == 11
    assert list(gfc.iloc[0, 1:]) == ["", "", ""] and gfc.iloc[10, 3].endswith(" of 600 starts running that day")
    # the worst trajectory: the lowest NAV that day in either portfolio (Keep, the 24 Mar 2000 start), both strategies read on that start, Δ on every row
    assert gfc.iloc[1, 1] == "24 Mar 2000, 2755 days before the peak, SPX 1,527 (-2.4% vs the peak)" and gfc.iloc[1, 2] == "same start (the lowest NAV that day: Keep the loan)" and gfc.iloc[1, 3] == ""
    assert gfc.iloc[2, 1].startswith("142 m (-81% from the start)")
    assert gfc.iloc[8, 1].startswith("381 − 366 = 15 m (LTV 96%")   # the loan is deducted: 381 − 366 = 15
    # the cells are the daily simulation's values on 9 Mar 2009 for the 24 Mar 2000 start: recompute it directly
    pa, _ = simulate("2000-03-24", "2026-09-14", roll="delta")
    da = pa.loc["2009-03-09"]
    assert gfc.iloc[2, 1] == f"{da['nav_A'] / m:,.0f} m ({da['nav_A'] / m / 750 - 1:+.0%} from the start)" and gfc.iloc[2, 2] == f"{da['nav_B'] / m:,.0f} m ({da['nav_B'] / m / 750 - 1:+.0%} from the start)" and gfc.iloc[2, 3] == pm(da["nav_B"] - da["nav_A"])
    assert gfc.iloc[3, 1] == f"{da['E_A'] / m:,.0f} m" and gfc.iloc[3, 2] == f"{da['E_B'] / m:,.0f} m" and gfc.iloc[3, 3] == pm(da["E_B"] - da["E_A"])
    assert gfc.iloc[4, 1] == "—" and gfc.iloc[4, 2] == f"{da['cash_B'] / m:,.0f} m" and gfc.iloc[4, 3] == pm(da["cash_B"])   # T-bills: the payoffs kept in cash (none here: the 2005 expiries were worthless)
    # calls bought by that day = the 52 build tranches + their 52 replacements of 2005–06 (51 expired worthless and were replaced on the same units, one in the money on the same dollar delta)
    assert int(da["n_bought"]) == 104 and int(da["n_calls"]) == 52 and gfc.iloc[5, 1] == "—" and gfc.iloc[5, 3] == pm(da["call_val"])
    assert gfc.iloc[5, 2] == f"{da['call_val'] / m:,.0f} m (52 calls alive of the 104 bought, on a notional of {da['call_notional'] / m:,.0f} m of index)"
    assert gfc.iloc[6, 1] == f"{da['loan'] / m:,.0f} m" and gfc.iloc[6, 2] == "0 m" and gfc.iloc[6, 3] == pm(-da["loan"])   # the rotation of March 2000 has no loan since 2001
    assert da["nav_B"] > da["nav_A"] and da["dry_powder_B"] > da["headroom_A"] and 200e6 < da["nav_B"] < 300e6   # ≈ 245 m against 142 m: the 2005 replacements, paid with SPX, kept the convexity
    lv_b = 0.75 * da["E_B"] + 0.0 * da["call_val"] + 0.9 * da["cash_B"]   # the lending value of what the rotation holds: SPX 75 %, calls 0 %, T-bills 90 % (sidebar defaults)
    assert gfc.iloc[7, 1] == f"{0.75 * da['E_A'] / m:,.0f} m (75% of the SPX)" and gfc.iloc[7, 2] == f"{lv_b / m:,.0f} m (75% of the SPX + 0% of the calls + 90% of the T-bills)" and gfc.iloc[7, 3] == pm(lv_b - 0.75 * da["E_A"])
    assert gfc.iloc[8, 1].endswith(f"= {da['headroom_A'] / m:,.0f} m (LTV {da['ltv_A']:.0%}: a further 4% SPX fall to a margin call)") and gfc.iloc[8, 2] == f"{lv_b / m:,.0f} − 0 = {da['dry_powder_B'] / m:,.0f} m"
    assert gfc.iloc[8, 3] == pm(da["dry_powder_B"] - da["headroom_A"]) and da["dry_powder_B"] == pytest.approx(lv_b - da["loan_B"]) and da["dry_powder_B"] > da["headroom_A"] and da["nav_B"] > da["nav_A"]
    # the interest cumulated on each loan since the start (Rotate: only during the build)
    assert gfc.iloc[9, 1] == f"{da['interest_cum_A'] / m:,.0f} m" and gfc.iloc[9, 2] == f"{da['interest_cum_B'] / m:,.0f} m (during the build)" and gfc.iloc[9, 3] == pm(da["interest_cum_B"] - da["interest_cum_A"])
    assert da["interest_cum_A"] == pytest.approx(pa.loc[:"2009-03-09", "interest_A"].sum())
    assert da["interest_cum_B"] == pytest.approx(pa["interest_B"].sum()) and 5e6 < da["interest_cum_B"] < 15e6   # ≈ 9 m: 250 m at ≈ 7 % repaid over a year
    assert int(da["n_calls"]) == 52 and da["loan_B"] == 0.0
    # the picks and the count over every weekly start running on 9 Mar 2009 (the page's grid): recompute all from the engine on the same starts
    bottom = pd.Timestamp("2009-03-09")
    # run past the bottom: `rolling_paths` skips a start whose build is not complete 30 days before the end date, and the page runs every start to today
    _, paths = rolling_paths(pd.date_range("1997-09-09", bottom, freq="W-FRI"), "2010-06-30", columns=("nav_A", "nav_B", "headroom_A", "dry_powder_B"), sample="M", mark_days=(bottom,), roll="delta")
    na, nb, ra, db_ = (paths[c].loc[bottom] for c in ("nav_A", "nav_B", "headroom_A", "dry_powder_B"))
    assert len(na) == 600 and na.notna().all() and "600 starts running that day" in " ".join(mk.value for mk in at.markdown)
    assert na.min() == pytest.approx(da["nav_A"]) and na.idxmin() == pd.Timestamp("2000-03-24")   # the worst trajectory is the lowest NAV that day over all 600 starts
    # the corrections dataframe: the lowest NAV of each portfolio on the bottom day, over the same 600 starts (the GFC row)
    assert bottoms.iloc[1, -2] == pytest.approx(na.min() / m) and bottoms.iloc[1, -1] == pytest.approx(nb.min() / m) and bottoms.iloc[1, -2] == pytest.approx(da["nav_A"] / m)
    assert gfc.iloc[10, 3] == f"{int((db_ > ra).sum()):,} of 600 starts running that day"   # the count row, recomputed from the engine on the same 600 starts
    assert 560 <= int((db_ > ra).sum()) <= 600 and nb.min() > na.min()   # at the GFC bottom the rotation has more dry powder on (almost) every start
    assert at.table[1].value.iloc[1, 1].startswith("24 Mar 2000, on the peak day, SPX 1,527 (+0.0% vs the peak)")   # the 2002 bottom: the worst start is the peak day too
    assert at.table[1].value.iloc[1, 2] == "same start (the lowest NAV that day: Keep the loan)"
    # the 2020 bottom came five weeks after the peak: the worst start had done only a few of its 52 weekly steps, and the cell says so (not "of the 52 bought")
    covid = at.table[3].value
    w20 = pd.Timestamp(covid.iloc[1, 1].split(",")[0])
    p20, _ = simulate(w20, "2026-09-14", roll="delta")
    d20 = p20.loc["2020-03-23"]
    assert 1 <= int(d20["n_bought"]) < 52 and int(d20["n_calls"]) == int(d20["n_bought"]) and d20["loan_B"] > 0.5e6
    assert covid.iloc[5, 2] == f"{d20['call_val'] / m:,.0f} m ({int(d20['n_calls'])} calls alive of the {int(d20['n_bought'])} bought so far: {int(d20['n_bought'])} of the 52 weekly steps done, on a notional of {d20['call_notional'] / m:,.0f} m of index)"
    assert covid.iloc[6, 2] == f"{d20['loan_B'] / m:,.0f} m (rotation still under way)"
    assert any(mk.value.startswith("**2022 bottom — 12 Oct 2022**") for mk in at.markdown) and len(at.table[4].value) == 11   # the 2022 correction has its table too
    assert any("Same start, same end" in m.value for m in at.markdown)
    assert at.selectbox(key="r_years").value == 1997 and at.selectbox(key="r_years").options[:2] == ["1997", "1998"]   # the first year = every start
    at.selectbox(key="r_years").set_value(2009).run()   # every start from 2009 onward: the 923 Fridays from 2 Jan 2009 to 4 Sep 2026 — once launched
    assert not at.exception, [e.value for e in at.exception]
    assert not at.button(key="r_launch").proto.disabled and any("The parameters have changed" in c.value for c in at.caption) and "1,513 selected starts" in " ".join(h.value for h in at.subheader)
    at.button(key="r_launch").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("stopped buying calls after a correction" in c.value and "923 lines" in c.value for c in at.caption) and "923 selected starts" in " ".join(h.value for h in at.subheader)
    assert any(m.value.startswith("- Of the 923 selected starts, the rotated portfolio lost **all** its calls on **0%** (0)") for m in at.markdown)   # the closing lines follow the selection: no start from 2009 on lost every call
    assert at.slider(key="r_zoom_2009_2026").value == (2009, 2026) and at.slider(key="r_zoom_2009_2026").min == 2009   # the zoom window follows the launched start year and end day
    at.selectbox(key="r_years").set_value(1997).run()
    at.button(key="r_launch").click().run()
    gfc = next(o for o in at.selectbox(key="r_end").options if "2007–2009 bottom" in o)   # held until the GFC bottom: every start that completed its rotation a month before 9 Mar 2009
    at.selectbox(key="r_end").set_value(gfc).run()
    at.button(key="r_launch").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("each held to 09 Mar 2009" in m.value for m in at.markdown) and "On 09 Mar 2009" in " ".join(h.value for h in at.subheader)
    stats = at.table[0].value
    assert pct(stats.iloc[4, 1]) < 0 and pct(stats.iloc[1, 1]) < -0.15   # at the GFC bottom most starts are under water
    assert len(at.table) == 2 and len(at.dataframe) == 1 and len(at.dataframe[0].value) == 1   # distribution + the one tab of the chosen bottom; the corrections table lists that bottom only
    assert list(at.dataframe[0].value["Correction"]) == ["2007–2009"] and "At the 2007–2009 bottom" in " ".join(h.value for h in at.subheader)
    assert at.table[1].value.iloc[8, 1].startswith("381 − 366 = 15 m (LTV 96%")   # the worst levered trajectory at the bottom, same as when held to today
    assert any("lost **all** its calls" in m.value for m in at.markdown) and any("Margin calls keeping the loan: **0** of the" in m.value for m in at.markdown)
    # under the default rules nothing lapses: no orange line
    at.selectbox(key="r_end").set_value(at.selectbox(key="r_end").options[0]).run()
    at.button(key="r_launch").click().run()
    assert any("0 of the 1,513 starts had at least one call expire worthless" in c.value for c in at.caption)
    assert at.radio(key="s_roll").value == "the same dollar delta" and at.radio(key="s_worthless").value == "replaced, SPX sold to pay it"   # as set above; the worthless default
    at.radio(key="s_roll").set_value("the same index units").run()
    assert not at.button(key="r_launch").proto.disabled   # a rule change is a new run
    at.radio(key="s_roll").set_value("the same dollar delta").run()
    assert at.button(key="r_launch").proto.disabled
    # no band trades under the dollar-delta rule: no band caption, the ratio chart without band edges
    assert not any("Band trades across" in c.value for c in at.caption)
    ratio_fig = json.loads(at.get("plotly_chart")[4].proto.spec)
    assert [t["name"] for t in ratio_fig["data"]] == ["95th percentile", "5th percentile", "median across the starts alive"] and not any("band edge" in a["text"] for a in ratio_fig["layout"].get("annotations", []))
    # the default rule, Keep's exposure with the quarterly band: launched on the same grid — the header counts, the ratio chart against the engine on the first start
    at.radio(key="s_roll").set_value("Keep's exposure").run()
    assert not at.button(key="r_launch").proto.disabled
    at.button(key="r_launch").click().run()
    assert not at.exception, [e.value for e in at.exception]
    band_line = next(c.value for c in at.caption if "Band trades across the 1,513 starts" in c.value)
    assert "up** (calls sold, most in the money first, at the mark less 1 vol point)" in band_line and "down** (ATM calls bought from the T-bills)" in band_line
    ratio_fig = json.loads(at.get("plotly_chart")[4].proto.spec)
    assert {a["text"] for a in ratio_fig["layout"]["annotations"]} >= {"band edge 1.10", "band edge 0.90", "Keep's exposure"}
    med = ratio_fig["data"][2]
    rows_t, paths_t = rolling_paths(pd.date_range("1997-09-12", "1997-10-31", freq="W-FRI"), "2026-09-14", columns=("exposure_A", "exposure_B"), sample="M")   # the first eight weekly starts, default rule
    ratio_t = paths_t["exposure_B"] / paths_t["exposure_A"]
    alive = ratio_t.loc["1997-09-30"].dropna()   # on the first month-end the 12, 19 and 26 Sep 1997 starts are alive: the chart's median is theirs
    assert med["x"][0].startswith("1997-09-30") and len(alive) == 3 and abs(med["y"][0] - float(alive.median())) < 1e-9
    assert (abs(alive - 1.0) < 0.05).all() and 0.6 < min(med["y"]) < 0.8 and 1.1 < max(med["y"]) < 1.3   # 2002: the early starts sit at 0.7× with no T-bills to buy with (Limitations 17); rallies run to the top of the band
    n_up = int(band_line.split("**")[1].split(" up")[0].replace(",", ""))
    assert n_up >= int(rows_t["B: rebalances up"].sum()) > 0   # the header counts every selected start; the first eight are part of it
    assert any("an expiring call is replaced to close the gap to Keep's exposure, the exposure checked every quarter and kept within ±10% of Keep's" in c.value for c in at.caption)
    at.radio(key="s_roll").set_value("the same dollar delta").run()
    at.button(key="r_launch").click().run()
    assert not at.exception, [e.value for e in at.exception]
    # the lapse rule: switch the sidebar to "not replaced" and relaunch — the rotations that let a call lapse are back (orange lines and profile), the 1997–2001 starts lose every call
    at.radio(key="s_worthless").set_value("not replaced").run()
    at.button(key="r_launch").click().run()
    assert not at.exception, [e.value for e in at.exception]
    lapse_line = next(c.value for c in at.caption if "had at least one call expire worthless and not replaced" in c.value)
    n_lapsed = int(lapse_line.split(" of the 1,513 starts had")[0].split(": ")[-1].replace(",", ""))
    assert 300 < n_lapsed < 800 and not any(h.value.endswith("excluded") for h in at.subheader) and len(at.checkbox) == 0   # no exclusion checkbox any more
    assert any("1,513 selected starts" in h.value for h in at.subheader) and any(m.value.startswith("- Of the 1,513 selected starts, the rotated portfolio lost **all** its calls on **") and "on **0%**" not in m.value for m in at.markdown)
    # the Launch simulation button and the sidebar: greyed out while the sidebar equals the launched setup; a changed widget enables it but the page keeps the launched setup until it is pressed
    launch = lambda: at.button(key="r_launch")  # noqa: E731
    assert launch().proto.disabled and not any("The parameters have changed" in c.value for c in at.caption)
    at.number_input(key="s_loan").set_value(300.0).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not launch().proto.disabled and any("The parameters have changed" in c.value for c in at.caption)
    assert any("each leaving from 750 m" in c.value for c in at.caption)   # still the launched 250 m loan
    launch().click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert launch().proto.disabled and any("each leaving from 700 m" in c.value for c in at.caption)   # 1,000 − 300: the new setup ran, and the button is greyed out again at once
    at.number_input(key="s_loan").set_value(250.0).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not launch().proto.disabled and any("each leaving from 700 m" in c.value for c in at.caption)   # back to 250 in the widget, but not launched: the page stays on 300


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
    at.button(key="r_launch").click().run()   # page 1 runs only when launched
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
