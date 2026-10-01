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
    # the sidebar defaults: Keep's exposure with the monthly band, calls bought below it from the T-bills then from SPX sold for them; the band widgets enabled, the surplus radio greyed out
    assert at.radio(key="s_roll").value == "Keep's exposure" and at.radio(key="s_rebalance").value == "every month" and at.number_input(key="s_band").value == 10.0 and at.number_input(key="s_haircut").value == 1.0
    assert at.radio(key="s_below").value == "ATM calls, SPX sold for them when the T-bills run out" and list(at.radio(key="s_below").options) == ["ATM calls from the T-bills", "SPX from the T-bills", "ATM calls, SPX sold for them when the T-bills run out"] and not at.radio(key="s_rebalance").proto.disabled and at.radio(key="s_surplus").proto.disabled and any("Not used under Keep's exposure" in c.value for c in at.caption)
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
    dp_lay = json.loads(at.get("plotly_chart")[3].proto.spec)["layout"]
    j_l = max(range(1513), key=lambda j: dp[2]["y"][j])
    assert any(a["text"] == f"<span style='color:#c9a000'><b>LTV {dp[2]['y'][j_l]:.0f} %</b></span> · {pd.Timestamp(dp[2]['x'][j_l]):%d %b %Y}" for a in dp_lay["annotations"])   # the highest LTV labelled in yellow
    assert any(a["text"] == "margin call (LTV 90%)" for a in dp_lay["annotations"]) and dp[2]["marker"]["cmax"] == 90.0 and dp[2]["marker"]["colorbar"]["orientation"] == "h"   # the call level at 90 %, the colour bar at the bottom
    for i_ in (2, 3, 4):   # the three charts by date share fixed margins and the full width, so their plot areas line up
        lay_i = json.loads(at.get("plotly_chart")[i_].proto.spec)["layout"]
        assert lay_i["margin"]["l"] == 80 and lay_i["margin"]["r"] == 80 and lay_i["yaxis"]["automargin"] is False and lay_i["xaxis"].get("domain", [0.0, 1.0]) == [0.0, 1.0]
    assert dp_lay["xaxis"]["domain"] == [0.0, 1.0] and dp_lay["xaxis2"]["domain"] == [0.0, 1.0]   # the secondary LTV axis no longer takes 6 % of the width
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
    detail = ["started on", "NAV", "SPX held, market value", "T-bills (cash)", "calls held, market value", "loan", "lending value of the holdings", "dry powder = 90% × lending value − loan", "interest cumulated"]
    assert list(gfc.columns) == cols_
    # one block (the lowest NAV that day, both strategies read on it), then the count row
    assert list(gfc.iloc[:10, 0]) == ["Worst trajectory: the lowest NAV that day", *detail] and gfc.iloc[10, 0] == "rotation has more dry powder on" and len(gfc) == 11
    assert list(gfc.iloc[0, 1:]) == ["", "", ""] and gfc.iloc[10, 3].endswith(" of 600 starts running that day")
    # the worst trajectory: the lowest NAV that day in either portfolio (Keep, the 24 Mar 2000 start), both strategies read on that start, Δ on every row
    assert gfc.iloc[1, 1] == "24 Mar 2000, 2755 days before the peak, SPX 1,527 (-2.4% vs the peak)" and gfc.iloc[1, 2] == "same start (the lowest NAV that day: Keep the loan)" and gfc.iloc[1, 3] == ""
    assert gfc.iloc[2, 1].startswith("142 m (-81% from the start)")
    # the cells are the daily simulation's values on 9 Mar 2009 for the 24 Mar 2000 start: recompute it directly
    pa, _ = simulate("2000-03-24", "2026-09-14", roll="delta")
    da = pa.loc["2009-03-09"]
    assert gfc.iloc[8, 1].startswith(f"{0.9 * 0.75 * da['E_A'] / m:,.0f} − {da['loan'] / m:,.0f} = {da['headroom_A'] / m:,.0f} m (LTV {da['ltv_A']:.0%}") and da["headroom_A"] < 0 < 0.75 * da["E_A"] - da["loan"]   # 343 − 366 = −23: in margin call at the 90 % level (not at 100 %: 381 − 366 = 15)
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
    assert gfc.iloc[8, 1].endswith(f"= {da['headroom_A'] / m:,.0f} m (LTV {da['ltv_A']:.0%}: above the 90% call level, in margin call)") and gfc.iloc[8, 2] == f"{0.9 * lv_b / m:,.0f} − 0 = {da['dry_powder_B'] / m:,.0f} m"
    assert gfc.iloc[8, 3] == pm(da["dry_powder_B"] - da["headroom_A"]) and da["dry_powder_B"] == pytest.approx(0.9 * lv_b - da["loan_B"]) and da["dry_powder_B"] > da["headroom_A"] and da["nav_B"] > da["nav_A"]
    assert 0.9 < da["ltv_A"] < 1.0   # LTV 96 %: called at the 90 % level, not at 100 %
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
    assert at.table[1].value.iloc[8, 1].startswith(f"{0.9 * 0.75 * da['E_A'] / m:,.0f} − {da['loan'] / m:,.0f} = {da['headroom_A'] / m:,.0f} m (LTV {da['ltv_A']:.0%}")   # the worst levered trajectory at the bottom, same as when held to today
    from fosim.analytics.leverage_stress import rolling_starts
    rs_b = rolling_starts(pd.date_range("1997-09-09", bottom - pd.DateOffset(days=7), freq="W-FRI"), 0.0, until=bottom, roll="delta")   # the page's starts held to the bottom: the margin calls at the 90 % level
    n_calls_b = int(rs_b["A: margin call"].sum())
    assert n_calls_b >= 1 and any("lost **all** its calls" in m.value for m in at.markdown) and any(f"Margin calls keeping the loan: **{n_calls_b:,}** of the" in m.value for m in at.markdown)
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
    # the default rule, Keep's exposure with the monthly band: launched on the same grid — the header counts, the ratio chart against the engine on the first start
    at.radio(key="s_roll").set_value("Keep's exposure").run()
    assert not at.button(key="r_launch").proto.disabled
    at.button(key="r_launch").click().run()
    assert not at.exception, [e.value for e in at.exception]
    band_line = next(c.value for c in at.caption if "Band trades across the 1,513 starts" in c.value)
    assert "up** (calls sold, most in the money first, at the mark less 1 vol point)" in band_line and "down** (ATM calls bought from the T-bills, then from SPX sold for them — " in band_line
    assert " m of SPX sold in all), **" in band_line and "partial** (the T-bills and the SPX ran out)" in band_line
    ratio_fig = json.loads(at.get("plotly_chart")[4].proto.spec)
    assert {a["text"] for a in ratio_fig["layout"]["annotations"]} >= {"band edge 1.10", "band edge 0.90", "Keep's exposure"}
    med = ratio_fig["data"][2]
    rows_t, paths_t = rolling_paths(pd.date_range("1997-09-12", "1997-10-31", freq="W-FRI"), "2026-09-14", columns=("exposure_A", "exposure_B"), sample="M")   # the first eight weekly starts, default rule
    ratio_t = paths_t["exposure_B"] / paths_t["exposure_A"]
    alive = ratio_t.loc["1997-09-30"].dropna()   # on the first month-end the 12, 19 and 26 Sep 1997 starts are alive: the chart's median is theirs
    assert med["x"][0].startswith("1997-09-30") and len(alive) == 3 and abs(med["y"][0] - float(alive.median())) < 1e-9
    assert (abs(alive - 1.0) < 0.05).all() and 0.7 < min(med["y"]) < 0.95 and 1.05 < max(med["y"]) < 1.3   # between month-ends the median drifts with the delta; SPX sold for calls keeps the early starts near the band in 2001–02
    n_up = int(band_line.split("**")[1].split(" up")[0].replace(",", ""))
    assert n_up >= int(rows_t["B: rebalances up"].sum()) > 0   # the header counts every selected start; the first eight are part of it
    assert any("an expiring call is replaced to close the gap to Keep's exposure, the exposure checked every month and kept within ±10% of Keep's" in c.value for c in at.caption)
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
    the shared tenor drives every label; the page's inputs survive a visit to page 1 (whose weekly launch takes 5–6 minutes, as in its own test)."""
    at = AppTest.from_file(str(APP.parent / "historical_app.py"), default_timeout=600)
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
    at.button(key="r_launch").click().run()   # page 1 runs only when launched (every weekly start since 1997: ~0.2 s each, hence the 600 s cap above)
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


def test_five_scenarios_page() -> None:
    """Page 3 (the investor's five scenarios): its own sidebar, the five named starts by default, the overview table and one chart + table per start,
    the cells recomputed from the engine, the margin-call flag appearing exactly when the engine flags one, the log scale and the extra start."""
    import json
    import sys

    sys.path.insert(0, str(APP.parents[1] / "src"))
    import pandas as pd

    from fosim.analytics.five_scenarios import LABELS, simulate_scenarios, summary
    m = 1e6
    names = [f"{sid} {lab}" for sid, lab in LABELS.items()]
    at = AppTest.from_file(str(APP.parent / "historical_app.py"), default_timeout=600)
    at.switch_page("views/five_scenarios.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [w.label for w in at.sidebar.number_input] == ["Capital (USD m)", "Loan (% of the capital)", "Exposure (% of the capital)", "Premium (% of the capital)", "Steps (1 = all on the start day)",
                                                          "Flat rate (%)", "Spread over the base rate (bp)", "Premium (% of notional)",
                                                          "Tax on the profit at expiry (%)", "Dividend withholding tax (%)", "Lending value of the SPX (%)", "Lending value of the calls (%)", "Margin call at LTV (%)"]
    assert [w.label for w in at.sidebar.radio] == ["Sized by", "Built", "Interest at", "Premium", "Scenario 2: the loan is", "Scenario 3 at a slot's expiry, refill from:", "Scenario 4 at a slot's expiry, the loan is"] and [c.label for c in at.sidebar.checkbox] == []
    assert at.radio(key="f_repay4").value == "repaid from the call proceeds, only the surplus in new calls"
    # the scenarios' rules visible in the sidebar in scenario order (user, 2026-10-01), not inside Advanced
    assert any(md.value == "**The scenarios' rules**" for md in at.sidebar.markdown) and len(at.sidebar.expander) == 1 and len(at.sidebar.expander[0].radio) == 0
    assert [w.label for w in at.sidebar.selectbox] == ["Call tenor (years)"] and at.number_input(key="f_capital").value == 650.0 and at.number_input(key="f_prem").value == 14.5 and at.number_input(key="f_tax").value == 24.0
    assert at.number_input(key="f_spread").proto.disabled and not at.number_input(key="f_rate").proto.disabled   # the flat rate is the default; the spread only with the base rate
    # the sleeve sized by premium (25 % of the capital, the note) and built in 52 weekly slots by default; the exposure input greyed out, scenario 3's refill radio enabled
    assert at.radio(key="f_sizing").value.startswith("premium") and at.number_input(key="f_call_pct").value == 25.0 and at.number_input(key="f_exp_pct").proto.disabled and not at.number_input(key="f_call_pct").proto.disabled
    assert at.radio(key="f_build_unit").value == "weekly" and at.number_input(key="f_build_steps").value == 52 and at.number_input(key="f_build_steps").proto.max == 104 and not at.radio(key="f_rebalance").proto.disabled
    assert at.number_input(key="f_exp_pct").value == 86.0 and at.radio(key="f_rebalance").value.startswith("the whole portfolio")
    starts = ["data start — 09 Sep 1997, SPX 934", "dot-com peak — 24 Mar 2000, SPX 1,527", "dot-com bottom — 09 Oct 2002, SPX 777", "GFC peak — 09 Oct 2007, SPX 1,565", "GFC bottom — 09 Mar 2009, SPX 677"]
    assert at.multiselect(key="f_starts").value == starts and len(at.get("plotly_chart")) == 5 and len(at.table) == 6
    assert [h.value for h in at.subheader] == ["Start 09 Sep 1997 — data start, SPX 934", "Start 24 Mar 2000 — dot-com peak, SPX 1,527", "Start 09 Oct 2002 — dot-com bottom, SPX 777",
                                               "Start 09 Oct 2007 — GFC peak, SPX 1,565", "Start 09 Mar 2009 — GFC bottom, SPX 677"]
    # short text: one setup line at the top, the explanations in the notes at the foot (no caption under the charts)
    assert at.caption[1].value == ("Capital 650 m · loan 162.5 m at 5.50% · call sleeve 25% of the capital in premium, built in 52 weekly slots · 5-year ATM calls on SPXFP at 14.5% of notional · "
                                   "tax 24% of the profit at expiry · every start held to 14 Sep 2026.")
    notes = [c.value for c in at.caption if c.value.startswith("**")]   # the notes at the foot: seven bullet-list blocks
    assert len(at.caption) == 12 and any(md.value == "##### Notes" for md in at.markdown) and [n.split("\n")[0] for n in notes] == [
        "**Setup** (the sidebar; Assumptions 40)", "**The five scenarios**", "**At a slot's expiry** (payoff in cash, tax where due)", "**Departures from the investor's note**", "**Reading the charts**", "**Reading the tables**", "**Caveats**"]
    assert "\n- **Risk**, on each scenario's own daily NAV returns" in notes[5] and "× √252" in notes[5]
    assert notes[3].count("\n- ") == 4 and "- The sleeve is built over 52 weekly slots" in notes[3] and "- Scenario 5 is taxed like 3 and 4" in notes[3] and "- Scenario 3 refills an expired slot from the whole portfolio" in notes[3] and "- Exposure is counted at the model's delta" in notes[3]
    assert "- **Capital**: 650 m; every start held to 14 Sep 2026" in notes[0] and "- **Loan**: 162.5 m (25% of the capital) at 5.50%" in notes[0]
    assert "- **Sleeve**: 25% of the capital in premium = 162.5 m, notional 1,121 m; the note counts 560 m of delta at 50 %, the model's delta at the first start is 44% (492 m)." in notes[0]
    assert "- **Build**: 52 weekly slots (first start: 09 Sep 1997 → 01 Sep 1998)" in notes[0] and "- **Tax**: 24% of a call's profit (payoff − premium, when positive) at expiry, scenarios 3 and 4 and 5;" in notes[0]
    assert notes[1].count("\n- **") == 5 and notes[2].count("\n- **") == 3 and notes[6].endswith("- For illustrative purposes only.")
    # the overview: one row per start, one column per scenario, the cells recomputed from the engine
    ov = at.table[0].value
    assert list(ov.columns) == names and list(ov.index) == ["data start, 09 Sep 1997", "dot-com peak, 24 Mar 2000", "dot-com bottom, 09 Oct 2002", "GFC peak, 09 Oct 2007", "GFC bottom, 09 Mar 2009"]
    runs = {d: simulate_scenarios(d, "2026-09-14") for d in ("1997-09-09", "2000-03-24", "2002-10-09", "2007-10-09", "2009-03-09")}
    for i, d in enumerate(runs):
        sm = summary(*runs[d])
        for sid, name in zip(LABELS, names, strict=True):
            assert ov.iloc[i][name].startswith(f"{sm.loc['NAV today', sid] / m:,.0f} m · {sm.loc['annualised return', sid]:+.1%}")
    assert ov.iloc[0][names[4]].endswith("· ✕ Sep 2003") and ov.iloc[1][names[4]].endswith("· ✕ Mar 2011") and not any("✕" in ov.iloc[i][names[4]] for i in (2, 3, 4))
    assert not any("⚠" in v for v in ov.to_numpy().ravel())   # no margin call at the 90 % level on the defaults
    # the GFC-bottom start's table against the engine: short cells, numbers and dates
    paths, infos = runs["2009-03-09"]
    sm = summary(paths, infos)
    t = at.table[5].value
    assert list(t.columns) == names and list(t.index) == ["NAV today", "lowest NAV", "max drawdown", "volatility", "1-day VaR 95 / 99", "1-day CVaR 95 / 99", "Sharpe ratio",
                                                         "exposure today", "margin call", "loan repaid", "calls expired", "premiums paid", "payoffs received", "tax paid", "interest paid", "holdings today"]
    for sid, name in zip(LABELS, names, strict=True):
        assert t.loc["NAV today", name] == f"{sm.loc['NAV today', sid] / m:,.0f} m · {sm.loc['annualised return', sid]:+.1%} a year"
        assert t.loc["lowest NAV", name] == f"{sm.loc['lowest NAV', sid] / m:,.0f} m, {sm.loc['lowest NAV on', sid]:%b %Y}" and t.loc["exposure today", name] == f"{sm.loc['exposure today', sid] / m:,.0f} m"
        # the risk rows (daily NAV returns) against the engine
        assert t.loc["max drawdown", name] == f"{sm.loc['max drawdown', sid]:.0%}, {sm.loc['max drawdown on', sid]:%b %Y}" and t.loc["volatility", name] == f"{sm.loc['volatility', sid]:.1%} a year"
        assert t.loc["1-day VaR 95 / 99", name] == f"{sm.loc['VaR 95 %', sid]:.1%} / {sm.loc['VaR 99 %', sid]:.1%}" and t.loc["1-day CVaR 95 / 99", name] == f"{sm.loc['CVaR 95 %', sid]:.1%} / {sm.loc['CVaR 99 %', sid]:.1%}"
        assert t.loc["Sharpe ratio", name] == f"{sm.loc['Sharpe ratio', sid]:.2f}"
    nav1 = paths["1"]["nav"].to_numpy()   # the volatility recomputed from the path, not read back from summary
    assert sm.loc["volatility", "1"] == pytest.approx((nav1[1:] / nav1[:-1] - 1.0).std(ddof=1) * 252 ** 0.5, rel=1e-12)
    assert t.loc["margin call", names[0]] == "—" and t.loc["margin call", names[1]] == f"none, LTV up to {infos['2'].max_ltv:.0%}" and t.loc["margin call", names[3]] == f"none, LTV up to {infos['4'].max_ltv:.0%}"
    assert t.loc["loan repaid", names[1]] == "10 Mar 2014" and t.loc["loan repaid", names[3]] == "02 Mar 2015" and t.loc["loan repaid", names[0]] == "—"   # 4's loan: slot by slot, gone at the 52nd expiry
    assert t.loc["calls expired", names[2]] == "156 (0 worthless)" and t.loc["calls expired", names[0]] == "—"
    assert t.loc["tax paid", names[2]] == f"{sm.loc['tax paid', '3'] / m:,.0f} m" and t.loc["tax paid", names[4]] == f"{sm.loc['tax paid', '5'] / m:,.0f} m" and t.loc["interest paid", names[1]] == f"{sm.loc['interest paid', '2'] / m:,.0f} m"
    assert t.loc["holdings today", names[2]] == f"SPX {sm.loc['SPX today', '3'] / m:,.0f} m · calls {sm.loc['calls today', '3'] / m:,.0f} m (52)" and t.loc["holdings today", names[4]].startswith("SPX 0 m · calls ")
    assert at.table[1].value.loc["calls expired", names[4]] == "52 (52 worthless) · ✕ Sep 2003"   # the 1997 start: every slot of the all-in-calls portfolio lapsed
    sm97 = summary(*runs["1997-09-09"])   # its risk rows read to the wipe-out day: a full drawdown, finite numbers, not "—"
    assert at.table[1].value.loc["max drawdown", names[4]] == f"-100%, {sm97.loc['max drawdown on', '5']:%b %Y}" and sm97.loc["returns counted", "5"] < sm97.loc["returns counted", "1"]
    assert at.table[1].value.loc["Sharpe ratio", names[4]] == f"{sm97.loc['Sharpe ratio', '5']:.2f}" and at.table[1].value.loc["volatility", names[4]] != "—"
    # the dot-com-peak chart: five NAV lines, five exposure lines, the two LTV lines, the expiry markers (one per slot), the loan repayment, the end labels, the wipe-out label, the build band; no margin-call label
    p2000, _ = runs["2000-03-24"]
    fig = json.loads(at.get("plotly_chart")[1].proto.spec)
    assert [tr["name"] for tr in fig["data"]] == [names[0], f"{names[0]}: exposure", names[1], f"{names[1]}: exposure", "2: loan repaid", f"{names[1]}: LTV", names[2], f"{names[2]}: exposure", "3: expiry",
                                                  names[3], f"{names[3]}: exposure", "4: expiry", f"{names[3]}: LTV", names[4], f"{names[4]}: exposure", "5: expiry"]
    for sid, j in (("1", 0), ("2", 2), ("3", 6), ("4", 9), ("5", 13)):
        assert fig["data"][j]["y"][-1] == pytest.approx(p2000[sid]["nav"].iloc[-1] / m, rel=1e-9) and len(fig["data"][j]["x"]) == len(p2000[sid])
        assert fig["data"][j + 1]["y"][-1] == pytest.approx(p2000[sid]["exposure"].iloc[-1] / m, rel=1e-9) and fig["data"][j + 1]["yaxis"] == "y2"
    assert fig["data"][4]["x"] == ["2005-03-24T00:00:00"] and len(fig["data"][8]["x"]) == 260 and fig["data"][8]["customdata"][0][6] == 1 and fig["data"][5]["y"][0] == pytest.approx(p2000["2"]["ltv"].iloc[0] * 100)
    assert fig["data"][5]["yaxis"] == "y3" and fig["data"][8]["marker"]["size"] == 5
    texts = [a["text"] for a in fig["layout"]["annotations"]]
    assert "<b>5 wiped out</b> 16 Mar 2011" in texts and f"<span style='color:#2a78d6'>●</span> 1: {p2000['1']['nav'].iloc[-1] / m:,.0f} m" in texts and "margin call (LTV 90%)" in texts and "build to Mar 2001" in texts
    assert any(sh["x1"] == "2001-03-16T00:00:00" and sh["x0"] == "2000-03-24T00:00:00" for sh in fig["layout"]["shapes"])   # the build band
    assert not any("margin call</b>" in tx for tx in texts) and {tx for tx in texts if tx.startswith(("peak", "bottom"))} == {"peak Mar 2000", "bottom Oct 2002", "peak Oct 2007", "bottom Mar 2009", "peak Feb 2020", "bottom Mar 2020", "peak Jan 2022", "bottom Oct 2022"}   # the start is the peak day, so its line is drawn
    for i in range(5):   # fixed margins on the five charts, the NAV and exposure axes in log scale by default
        lay = json.loads(at.get("plotly_chart")[i].proto.spec)["layout"]
        assert lay["margin"]["l"] == 80 and lay["margin"]["r"] == 120 and lay["yaxis"]["automargin"] is False and lay["yaxis"]["type"] == "log" and lay["yaxis2"]["type"] == "log"
    # the margin-call flag: with the bank calling at 50 % of the lending value the levered long from the 2000 peak is called (LTV up to 59 %); the label, the marker and the overview cell follow the engine
    at.number_input(key="f_margin").set_value(50.0).run()
    assert not at.exception, [e.value for e in at.exception]
    q, qi = simulate_scenarios("2000-03-24", "2026-09-14", margin_call=0.5)
    first = qi["2"].margin_call_first
    assert first is not None and qi["4"].margin_call_first is not None
    fig = json.loads(at.get("plotly_chart")[1].proto.spec)
    texts = [a["text"] for a in fig["layout"]["annotations"]]
    assert f"<b>⚠ 2 margin call</b> {first:%d %b %Y} — LTV {float(q['2']['ltv'].loc[first]):.0%}" in texts and "margin call (LTV 50%)" in texts
    flag = next(tr for tr in fig["data"] if tr["name"] == "2: margin call")
    assert flag["x"] == [first.strftime("%Y-%m-%dT%H:%M:%S")] and flag["marker"]["symbol"] == "x" and flag["marker"]["color"] == "#c00000"
    ov = at.table[0].value
    assert ov.iloc[1][names[1]].endswith(f"· ⚠ {first:%b %Y}")
    assert at.table[2].value.loc["margin call", names[1]] == f"⚠ {first:%d %b %Y}, LTV {float(q['2']['ltv'].loc[first]):.0%}, {qi['2'].days_in_margin_call} days"
    at.number_input(key="f_margin").set_value(90.0).run()
    # linear scale and an extra start
    at.checkbox(key="f_log").uncheck().run()
    assert not at.exception, [e.value for e in at.exception]
    assert all(json.loads(at.get("plotly_chart")[i].proto.spec)["layout"]["yaxis"].get("type") != "log" and json.loads(at.get("plotly_chart")[i].proto.spec)["layout"]["yaxis"]["rangemode"] == "tozero" for i in range(5))
    at.checkbox(key="f_extra").check().run()
    at.date_input(key="f_extra_date").set_value(pd.Timestamp("2020-03-23").date()).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("plotly_chart")) == 6 and at.subheader[5].value == "Start 23 Mar 2020 — chosen start, SPX 2,237" and len(at.table) == 7 and list(at.table[0].value.index)[5] == "chosen start, 23 Mar 2020"
    # the other pages' sidebars are untouched: page 2's labels as before, the shared tenor the same widget
    at.selectbox(key="s_tenor").set_value(7).run()
    at.switch_page("views/premium_history.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [w.label for w in at.sidebar.number_input] == ["Premium / investment (USD m)", "Fixed premium (% of notional)", "Dividend withholding tax (%)"] and at.selectbox(key="s_tenor").value == 7
    at.switch_page("views/five_scenarios.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="s_tenor").value == 7 and any("7-year ATM calls on SPXFP at" in c.value for c in at.caption) and at.number_input(key="f_margin").value == 90.0


def test_call_share_sweep_page() -> None:
    """Page 4: scenario 3 alone with its call share swept from the five named starts — the sidebar (its own keys, page 3's defaults, the swept range),
    the frontier and the measures cross-checked against direct runs (0 % = scenario 1, every other point = scenario 3 at that share), the table,
    the range and the sizing switches, the month-end mode behind its Launch button (not launched: twenty minutes), page 3 untouched."""
    import json

    from fosim.analytics.five_scenarios import simulate_scenarios, summary

    at = AppTest.from_file(str(APP.parent / "historical_app.py"), default_timeout=600)
    at.session_state["c_prem_to"] = 20.0   # 0, 5, 10, 15, 20 %: five shares × five starts keep the test short
    at.switch_page("views/call_share_sweep.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    # the sidebar: scenario 3's inputs only (no loan, no lending values, no scenario 2 / 4 rules), the swept range, the tenor and WHT shared
    assert [w.label for w in at.sidebar.radio] == ["Sized by", "Built", "Premium", "Scenario 3 at a slot's expiry, refill from:"] and [w.label for w in at.sidebar.selectbox] == ["Call tenor (years)"]
    assert [w.label for w in at.sidebar.number_input] == ["Capital (USD m)", "from", "to", "step", "from", "to", "step", "Steps (1 = all on the start day)", "Premium (% of notional)", "Tax on the profit at expiry (%)", "Dividend withholding tax (%)"]
    assert at.number_input(key="c_capital").value == 650.0 and at.number_input(key="c_prem").value == 14.5 and at.number_input(key="c_tax").value == 24.0 and at.number_input(key="c_build_steps").value == 52
    assert (at.number_input(key="c_prem_from").value, at.number_input(key="c_prem_to").value, at.number_input(key="c_prem_step").value) == (5.0, 20.0, 5.0)
    assert (at.number_input(key="c_exp_from").value, at.number_input(key="c_exp_to").value, at.number_input(key="c_exp_step").value) == (10.0, 100.0, 10.0)
    assert all(at.number_input(key=f"c_exp_{k}").proto.disabled for k in ("from", "to", "step")) and not any(at.number_input(key=f"c_prem_{k}").proto.disabled for k in ("from", "to", "step"))
    assert at.radio(key="c_sizing").value.startswith("premium") and not at.radio(key="c_rebalance").proto.disabled and len(at.sidebar.expander) == 1
    starts = ["data start — 09 Sep 1997, SPX 934", "dot-com peak — 24 Mar 2000, SPX 1,527", "dot-com bottom — 09 Oct 2002, SPX 777", "GFC peak — 09 Oct 2007, SPX 1,565", "GFC bottom — 09 Mar 2009, SPX 677"]
    assert at.multiselect(key="c_starts").value == starts and at.radio(key="c_mode").value == "the five named starts" and len(at.get("plotly_chart")) == 2 and len(at.table) == 1
    assert at.caption[1].value == ("Capital 650 m · scenario 3 (SPX + calls) alone, the sleeve sized by premium, swept 0 % then 5 → 20 % step 5 of the capital · built in 52 weekly slots · "
                                   "5-year ATM calls on SPXFP at 14.5% of notional · tax 24% of the profit at expiry · every start held to 14 Sep 2026.")
    notes = [c.value for c in at.caption if c.value.startswith("**")]
    assert [n.split("\n")[0] for n in notes] == ["**Setup** (the sidebar; Assumptions 41)", "**Reading the charts**", "**Reading the tables**", "**Caveats**"] and "- **0 %** buys no call at all" in notes[0]
    # the frontier against the engine: the GFC-bottom curve's points are scenario 3 at 0, 5, 10, 15, 20 %; the 0 % point and the diamond are scenario 1
    fig = json.loads(at.get("plotly_chart")[0].proto.spec)
    traces = {tr["name"]: tr for tr in fig["data"]}
    gfc = traces["GFC bottom"]
    assert gfc["text"] == ["0 %", "5 %", "10 %", "15 %", "20 %"] and gfc["mode"] == "lines+markers+text" and "◇ 0 % = long the market (scenario 1)" in traces
    paths, infos = simulate_scenarios("2009-03-09", "2026-09-14", call_frac=0.10, scenarios=("1", "3"))
    sm = summary(paths, infos)
    assert gfc["x"][2] == pytest.approx(sm.loc["volatility", "3"] * 100, rel=1e-9) and gfc["y"][2] == pytest.approx(sm.loc["annualised return", "3"] * 100, rel=1e-9)
    assert gfc["x"][0] == pytest.approx(sm.loc["volatility", "1"] * 100, rel=1e-9) and gfc["y"][0] == pytest.approx(sm.loc["annualised return", "1"] * 100, rel=1e-9)
    assert traces["GFC bottom: long"]["x"] == [gfc["x"][0]] and traces["GFC bottom: long"]["marker"]["symbol"] == "diamond"
    assert gfc["customdata"][2][0] == pytest.approx(sm.loc["Sharpe ratio", "3"], rel=1e-9) and gfc["customdata"][2][1] == pytest.approx(sm.loc["max drawdown", "3"], rel=1e-9)
    # the measures against the share: six panels, the GFC-bottom line of the first = the annualised return
    fig2 = json.loads(at.get("plotly_chart")[1].proto.spec)
    assert [a["text"] for a in fig2["layout"]["annotations"][:6]] == ["annualised return", "annualised volatility", "max drawdown", "1-day VaR 99 %", "Sharpe ratio", "1-day CVaR 99 %"]
    first_panel = next(tr for tr in fig2["data"] if tr["name"] == "GFC bottom")
    assert first_panel["x"] == [0.0, 5.0, 10.0, 15.0, 20.0] and first_panel["y"][2] == pytest.approx(sm.loc["annualised return", "3"] * 100, rel=1e-9)
    # the table: one row block per start, the columns the shares
    t = at.table[0].value
    assert list(t.columns) == ["0 %", "5 %", "10 %", "15 %", "20 %"] and list(t.index)[:4] == ["data start · return", "data start · vol", "data start · Sharpe", "data start · max DD"] and len(t) == 20
    assert t.loc["GFC bottom · return", "10 %"] == f"{sm.loc['annualised return', '3']:+.1%}" and t.loc["GFC bottom · Sharpe", "10 %"] == f"{sm.loc['Sharpe ratio', '3']:.2f}" and t.loc["GFC bottom · return", "0 %"] == f"{sm.loc['annualised return', '1']:+.1%}"
    # the range is what was asked: a step of 10 from 5 to 20 gives 0, 5, 15
    at.number_input(key="c_prem_step").set_value(10.0).run()
    assert not at.exception, [e.value for e in at.exception]
    gfc = {tr["name"]: tr for tr in json.loads(at.get("plotly_chart")[0].proto.spec)["data"]}["GFC bottom"]
    assert gfc["text"] == ["0 %", "5 %", "15 %"] and list(at.table[0].value.columns) == ["0 %", "5 %", "15 %"]
    # exposure sizing: the exposure share is swept, scenario 3's refill fixed on the whole portfolio, the 50 % point = scenario 3 at that exposure
    at.radio(key="c_sizing").set_value("exposure: % of the capital in SPX-equivalent (delta × notional)").run()   # the exposure triplet enabled by this run
    assert not at.exception, [e.value for e in at.exception]
    at.number_input(key="c_exp_from").set_value(50.0)
    at.number_input(key="c_exp_step").set_value(50.0).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio(key="c_rebalance").proto.disabled and at.radio(key="c_rebalance").value.startswith("the whole portfolio") and all(at.number_input(key=f"c_prem_{k}").proto.disabled for k in ("from", "to", "step"))
    gfc = {tr["name"]: tr for tr in json.loads(at.get("plotly_chart")[0].proto.spec)["data"]}["GFC bottom"]
    smx = summary(*simulate_scenarios("2009-03-09", "2026-09-14", sizing="exposure", exposure_frac=0.5, scenarios=("3",)))
    assert gfc["text"] == ["0 %", "50 %", "100 %"] and gfc["x"][1] == pytest.approx(smx.loc["volatility", "3"] * 100, rel=1e-9) and gfc["y"][1] == pytest.approx(smx.loc["annualised return", "3"] * 100, rel=1e-9)
    assert "sized by exposure, swept 0 % then 50 → 100 % step 50 of the capital" in at.caption[1].value
    # every month-end start: behind the Launch button (not pressed here: twenty minutes); the named sections still shown
    at.radio(key="c_mode").set_value("every month-end start").run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.button(key="c_launch").proto.disabled and any("Press **Launch**" in i.value for i in at.info) and len(at.get("plotly_chart")) == 2 and len(at.table) == 1
    # page 3 untouched by page 4's settings: its own keys and defaults, the tenor shared
    at.selectbox(key="s_tenor").set_value(7)
    at.switch_page("views/five_scenarios.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio(key="f_sizing").value.startswith("premium") and at.number_input(key="f_call_pct").value == 25.0 and at.number_input(key="f_exp_pct").value == 86.0 and at.selectbox(key="s_tenor").value == 7
    assert [w.label for w in at.sidebar.radio] == ["Sized by", "Built", "Interest at", "Premium", "Scenario 2: the loan is", "Scenario 3 at a slot's expiry, refill from:", "Scenario 4 at a slot's expiry, the loan is"]
