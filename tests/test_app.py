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
    assert len(at.get("plotly_chart")) == 2 and len(at.dataframe) == 6   # two fans (with the final-NAV profile); statistics, market bottoms, one table per bottom (4)
    assert any("1,202 lines" in c.value for c in at.caption)   # empty selection = every start year
    num = lambda cell: float(str(cell).split(" m")[0].replace(",", ""))  # noqa: E731
    pts = ["lowest", "5th percentile", "25th percentile", "median", "75th percentile", "95th percentile", "highest"]
    cols_ = [" ", "Keep the loan", "Rotate into calls", "Δ Rotate into calls − Keep the loan"]
    stats = at.dataframe[0].value   # one section row 'Final NAV' (shaded, empty cells) then the statistics; Δ = the paired difference on the same start
    assert list(stats.columns) == cols_ and list(stats.iloc[:, 0]) == ["Final NAV", *pts, "mean", "rotation ahead on"]
    assert list(stats.iloc[0, 1:]) == ["", "", ""] and num(stats.iloc[1, 1]) > 750 and num(stats.iloc[7, 2]) > num(stats.iloc[4, 2])
    ka = [num(c) for c in stats.iloc[1:8, 1]]
    assert ka == sorted(ka)
    for i in range(1, 9):   # Δ = the Rotate cell − the Keep cell of the row
        assert abs(num(stats.iloc[i, 3]) - (num(stats.iloc[i, 2]) - num(stats.iloc[i, 1]))) <= 1
    assert stats.iloc[9, 3].endswith(" of 1,202 starts")
    bottoms = at.dataframe[1].value
    assert list(bottoms["Correction"]) == ["2000–2002", "2007–2009", "2020", "2022"] and [str(d) for d in bottoms["Bottom"]] == ["2002-10-09", "2009-03-09", "2020-03-23", "2022-10-12"]
    assert [round(v) for v in bottoms["SPX at the bottom"]] == [777, 677, 2237, 3577] and [str(d) for d in bottoms["Previous peak"]] == ["2000-03-24", "2007-10-09", "2020-02-19", "2022-01-03"]
    gfc = at.dataframe[3].value   # one table per bottom, in order: 2002, 2009, 2020, 2022; the GFC bottom's worst levered trajectory started on the dot-com peak day
    detail = ["started on", "NAV", "SPX held, market value", "calls held, market value", "cash", "loan", "lending value of the holdings", "dry powder = lending value − loan + cash", "cost since the start"]
    assert list(gfc.columns) == cols_
    assert list(gfc.iloc[:, 0]) == ["NAV", *pts, "rotation ahead on", "Worst trajectory", *detail, "Dry powder", *pts, "rotation ahead on"]
    assert list(gfc.iloc[0, 1:]) == ["", "", ""] and list(gfc.iloc[9, 1:]) == ["", "", ""] and list(gfc.iloc[19, 1:]) == ["", "", ""]   # the three section rows
    assert gfc.iloc[1, 1].startswith("142 m (-81% from the start)") and gfc.iloc[1, 2].startswith("261 m (")
    # the worst trajectory: the lowest NAV that day in either portfolio (Keep, the 24 Mar 2000 start), both strategies read on that start, Δ on every row
    assert gfc.iloc[10, 1] == "24 Mar 2000, 2755 days before the peak, SPX 1,527 (-2.4% vs the peak)" and gfc.iloc[10, 2] == "same start (the lowest NAV that day: Keep the loan)" and gfc.iloc[10, 3] == ""
    nav_a = [num(c) for c in gfc.iloc[1:8, 1]]
    assert nav_a == sorted(nav_a) and nav_a[0] < 150 < nav_a[1]   # lowest < 5th < 25th < median < 75th < 95th < highest
    assert gfc.iloc[17, 1].startswith("381 − 366 = 15 m (LTV 96%") and gfc.iloc[23, 2] == "303 m"   # the loan is deducted: 381 − 366 = 15; the median rotated start had 303 m
    # the cells are the daily simulation's values on 9 Mar 2009 for the 24 Mar 2000 start: recompute it directly
    import sys
    sys.path.insert(0, str(APP.parents[1] / "src"))
    from fosim.analytics.leverage_stress import rolling_paths, simulate
    pa, _ = simulate("2000-03-24", "2026-09-14")
    da = pa.loc["2009-03-09"]
    m = 1e6
    pm = lambda x: f"{round(float(x) / m) + 0.0:+,.0f} m"  # noqa: E731
    assert gfc.iloc[11, 1] == f"{da['nav_A'] / m:,.0f} m ({da['nav_A'] / m / 750 - 1:+.0%} from the start)" and gfc.iloc[11, 2] == f"{da['nav_B'] / m:,.0f} m ({da['nav_B'] / m / 750 - 1:+.0%} from the start)" and gfc.iloc[11, 3] == pm(da["nav_B"] - da["nav_A"])
    assert gfc.iloc[12, 1] == f"{da['E_A'] / m:,.0f} m" and gfc.iloc[12, 2] == f"{da['E_B'] / m:,.0f} m" and gfc.iloc[12, 3] == pm(da["E_B"] - da["E_A"])
    assert gfc.iloc[13, 1] == "—" and gfc.iloc[13, 2] == f"{da['call_val'] / m:,.0f} m (1 call alive of the 52 bought, on a notional of {da['call_notional'] / m:,.0f} m of index)" and gfc.iloc[13, 3] == pm(da["call_val"])
    assert gfc.iloc[14, 1] == "—" and gfc.iloc[14, 2] == f"{da['cash_B'] / m:,.0f} m" and gfc.iloc[14, 3] == pm(da["cash_B"])
    assert gfc.iloc[15, 1] == f"{da['loan'] / m:,.0f} m" and gfc.iloc[15, 2] == "0 m" and gfc.iloc[15, 3] == pm(-da["loan"])   # the rotation of March 2000 has no loan since 2001
    assert gfc.iloc[16, 1] == f"{0.75 * da['E_A'] / m:,.0f} m (75% of the SPX)" and gfc.iloc[16, 2] == f"{0.75 * da['E_B'] / m:,.0f} m (75% of the SPX + 0% of the calls)" and gfc.iloc[16, 3] == pm(0.75 * (da["E_B"] - da["E_A"]))
    assert gfc.iloc[17, 1].endswith(f"= {da['headroom_A'] / m:,.0f} m (LTV {da['ltv_A']:.0%}: a further 4% SPX fall to a margin call)") and gfc.iloc[17, 2].endswith(f"= {da['dry_powder_B'] / m:,.0f} m")
    assert gfc.iloc[17, 3] == pm(da["dry_powder_B"] - da["headroom_A"]) and da["dry_powder_B"] > da["headroom_A"] and da["nav_B"] > da["nav_A"]
    assert gfc.iloc[18, 1] == f"interest {da['interest_cum_A'] / m:,.0f} m"
    assert gfc.iloc[18, 2] == f"premiums {da['premiums_cum_B'] / m:,.0f} m − payoffs {da['payoffs_cum_B'] / m:,.0f} m = {(da['premiums_cum_B'] - da['payoffs_cum_B']) / m:,.0f} m"
    assert gfc.iloc[18, 3] == pm(da["premiums_cum_B"] - da["payoffs_cum_B"] - da["interest_cum_A"]) and da["interest_cum_A"] == pytest.approx(pa.loc[:"2009-03-09", "interest_A"].sum())
    assert int(da["n_calls"]) == 1 and da["loan_B"] == 0.0
    # NAV and Dry powder sections: Keep and Rotate are each column's own distribution over every weekly start running on 9 Mar 2009 (the page's grid);
    # Δ = Rotate − Keep of the row; 'rotation ahead on' counts the starts where B > A on the same start. Recompute all from the engine on the same starts.
    import pandas as pd
    bottom = pd.Timestamp("2009-03-09")
    # run past the bottom: `rolling_paths` skips a start whose build is not complete 30 days before the end date, and the page runs every start to today
    _, paths = rolling_paths(pd.date_range("1997-09-09", bottom, freq="W-FRI"), "2010-06-30", columns=("nav_A", "nav_B", "headroom_A", "dry_powder_B"), sample="M", mark_days=(bottom,))
    na, nb, ra, db_ = (paths[c].loc[bottom] for c in ("nav_A", "nav_B", "headroom_A", "dry_powder_B"))
    assert len(na) == 600 and na.notna().all() and "600 starts running that day" in " ".join(mk.value for mk in at.markdown)
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)
    points = lambda s_: [s_.min(), *[s_.quantile(q) for q in qs], s_.max()]  # noqa: E731
    assert [gfc.iloc[i, 2] for i in range(2, 7)] == [f"{nb.quantile(q) / m:,.0f} m" for q in qs] and [gfc.iloc[i, 1] for i in range(2, 7)] == [f"{na.quantile(q) / m:,.0f} m" for q in qs]
    assert gfc.iloc[7, 1].startswith(f"{na.max() / m:,.0f} m (") and gfc.iloc[7, 2].startswith(f"{nb.max() / m:,.0f} m (") and gfc.iloc[1, 2].startswith(f"{nb.min() / m:,.0f} m (")
    assert [gfc.iloc[i, 3] for i in range(1, 8)] == [pm(y - x) for x, y in zip(points(na), points(nb), strict=True)] and gfc.iloc[8, 3] == f"{int((nb > na).sum()):,} of 600 starts"
    assert [gfc.iloc[i, 1] for i in range(20, 27)] == [f"{x / m:,.0f} m" for x in points(ra)] and [gfc.iloc[i, 2] for i in range(20, 27)] == [f"{x / m:,.0f} m" for x in points(db_)]
    assert [gfc.iloc[i, 3] for i in range(20, 27)] == [pm(y - x) for x, y in zip(points(ra), points(db_), strict=True)] and gfc.iloc[27, 3] == f"{int((db_ > ra).sum()):,} of 600 starts"
    assert 590 <= int((nb > na).sum()) <= 600 and gfc.iloc[1, 3] == pm(nb.min() - na.min())   # at the GFC bottom the rotation is ahead on (almost) every start
    assert at.dataframe[2].value.iloc[10, 1].startswith("24 Mar 2000, on the peak day, SPX 1,527 (+0.0% vs the peak)")   # the 2002 bottom: the worst start is the peak day too
    assert at.dataframe[2].value.iloc[10, 2] == "same start (the lowest NAV that day: Keep the loan)"
    assert any(mk.value.startswith("**2022 bottom — 12 Oct 2022**") for mk in at.markdown) and len(at.dataframe[5].value) == 28   # the 2022 correction has its table too
    assert any("Same start, same end" in m.value for m in at.markdown)
    at.multiselect(key="r_years").set_value([2008, 2009]).run()   # a few start years only
    assert not at.exception, [e.value for e in at.exception]
    assert any("stopped buying calls after a correction" in c.value and "105 lines" in c.value for c in at.caption) and "105 selected starts" in " ".join(h.value for h in at.subheader)
    at.multiselect(key="r_years").set_value([]).run()
    gfc = next(o for o in at.selectbox(key="r_end").options if "2007–2009 bottom" in o)   # held until the GFC bottom: every start that completed its rotation a month before 9 Mar 2009
    at.selectbox(key="r_end").set_value(gfc).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("each held to 09 Mar 2009" in m.value for m in at.markdown) and "Final NAV on 09 Mar 2009" in " ".join(h.value for h in at.subheader)
    stats = at.dataframe[0].value
    assert num(stats.iloc[1, 1]) < 300 and num(stats.iloc[4, 1]) < 750   # at the GFC bottom most starts are under water
    assert len(at.dataframe) == 4   # statistics, bottoms; only the 2002 and 2009 bottoms fall inside runs ending 9 Mar 2009
    assert at.dataframe[3].value.iloc[17, 1].startswith("381 − 366 = 15 m (LTV 96%")   # the worst levered trajectory at the bottom, same as when held to today   # at the GFC bottom the median levered start is well above half its lending value
    assert any("lost **all** its calls" in m.value for m in at.markdown) and any("Margin calls (Keep the loan): **0**" in m.value for m in at.markdown)


def test_premium_history_page_unchanged() -> None:
    """Page 2 (call-vs-cash backtest): no tabs, chart + tables rendered at default widgets; its own tenor drives every label."""
    at = AppTest.from_file(str(APP.parent / "historical_app.py"), default_timeout=300)
    at.switch_page("views/premium_history.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 0
    assert len(at.dataframe) == 2  # summary table + return-distribution percentiles
    assert all(len(d.value) <= 12 for d in at.dataframe)  # both fit the fixed height set in the app (12 rows)
    assert not at.warning  # 5-year tenor: no fallback warning
    assert at.date_input(key="bt_end").value == bt_end_for(5.0)
    for t in (7.0, 10.0):  # tenor-specific columns exist: no fallback warning
        at.selectbox(key="bt_tenor").set_value(int(t)).run()
        assert not at.exception, [e.value for e in at.exception]
        assert not at.warning
        assert any(f"{t:g}-year ATM call" in m.value for m in at.markdown)
        assert any(f"{t:g}y Treasury of the day" in o for o in at.radio[0].options)  # premium-mode option follows the tenor
        assert at.date_input(key="bt_end").value == bt_end_for(t)  # last strike date follows the tenor
    at.radio[0].set_value("Fixed % of notional").run()
    at.selectbox(key="bt_tenor").set_value(5).run()
    assert at.radio[0].value == "Fixed % of notional"  # premium mode survives a tenor change
    assert any("fixed at" in m.value for m in at.markdown)


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
