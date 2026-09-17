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
    stats = at.dataframe[0].value
    assert list(stats.index) == ["5th percentile", "25th percentile", "median", "75th percentile", "95th percentile", "mean", "standard deviation", "lowest (worst start)", "highest (best start)"]
    assert (stats.iloc[:, 0] > 0).all() and stats.loc["lowest (worst start)"].iloc[0] > 750 and stats.loc["highest (best start)"].iloc[1] > stats.loc["median"].iloc[1]
    bottoms = at.dataframe[1].value
    assert list(bottoms["Correction"]) == ["2000–2002", "2007–2009", "2020", "2022"] and [str(d) for d in bottoms["Bottom"]] == ["2002-10-09", "2009-03-09", "2020-03-23", "2022-10-12"]
    assert [round(v) for v in bottoms["SPX at the bottom"]] == [777, 677, 2237, 3577] and [str(d) for d in bottoms["Previous peak"]] == ["2000-03-24", "2007-10-09", "2020-02-19", "2022-01-03"]
    gfc = at.dataframe[3].value   # one table per bottom, in order: 2002, 2009, 2020, 2022; the GFC bottom's worst levered trajectory started on the dot-com peak day
    assert list(gfc.columns) == ["Keep the loan", "Rotate into calls", "Δ Rotate into calls − Keep the loan"]
    assert list(gfc.index) == ["Lowest NAV that day", "NAV that day: 5th percentile", "NAV that day: 25th percentile", "NAV that day: median", "That trajectory started on",
                               "… SPX held, market value", "… calls held, market value", "… cash", "… loan", "… lending value of the holdings", "… dry powder = lending value − loan + cash",
                               "Dry powder that day: 5th percentile", "Dry powder that day: 25th percentile", "Dry powder that day: median"]
    assert gfc.iloc[0, 0].startswith("142 m (-81% from the start)") and gfc.iloc[4, 0] == "24 Mar 2000, 2755 days before the peak, SPX 1,527 (-2.4% vs the peak)"
    assert gfc.iloc[4, 1].startswith("16 May 2008, 220 days after the peak, SPX 1,425 (-8.9% vs the peak)")
    nav_a = [float(v.split(" m")[0].replace(",", "")) for v in gfc.iloc[0:4, 0]]
    assert nav_a == sorted(nav_a) and nav_a[0] < 150 < nav_a[1]   # worst < 5th < 25th < median
    assert list(gfc.iloc[[6, 8, 9], 2]) == ["", "", ""] and gfc.iloc[4, 2] == "" and gfc.iloc[10, 2].startswith("+")   # no delta on calls, loan, lending value; a delta on the dry powder
    assert gfc.iloc[10, 0].startswith("381 − 366 = 15 m (LTV 96%") and gfc.iloc[13, 1] == "303 m"   # the loan is deducted: 381 − 366 = 15; the median rotated start had 303 m
    # the cells are the daily simulation's values on 9 Mar 2009: recompute the two trajectories directly
    import sys
    sys.path.insert(0, str(APP.parents[1] / "src"))
    from fosim.analytics.leverage_stress import simulate
    pa, _ = simulate("2000-03-24", "2026-09-14")
    pb, _ = simulate("2008-05-16", "2026-09-14")
    da, db = pa.loc["2009-03-09"], pb.loc["2009-03-09"]
    assert gfc.iloc[5, 0] == f"{da['E_A'] / 1e6:,.0f} m" and gfc.iloc[8, 0] == f"{da['loan'] / 1e6:,.0f} m" and gfc.iloc[10, 0].endswith(f"= {da['headroom_A'] / 1e6:,.0f} m (LTV {da['ltv_A']:.0%}: a further 4% SPX fall to a margin call)")
    assert gfc.iloc[5, 1] == f"{db['E_B'] / 1e6:,.0f} m" and gfc.iloc[6, 1] == f"{db['call_val'] / 1e6:,.0f} m ({int(db['n_calls'])} calls alive of the 52 bought, on a notional of {db['call_notional'] / 1e6:,.0f} m of index)" and gfc.iloc[7, 1] == f"{db['cash_B'] / 1e6:,.0f} m"
    assert gfc.iloc[8, 1].startswith(f"{db['loan_B'] / 1e6:,.0f} m (rotation still under way)") and gfc.iloc[10, 1].endswith(f"= {db['dry_powder_B'] / 1e6:,.0f} m")
    assert gfc.iloc[9, 1] == f"{0.75 * db['E_B'] / 1e6:,.0f} m (75% of the SPX + 0% of the calls)" and gfc.iloc[10, 2] == f"{(db['dry_powder_B'] - da['headroom_A']) / 1e6:+,.0f} m"
    assert 40 < db["n_calls"] <= 52 and db["call_val"] < 0.3 * db["E_B"]   # mid-build: 43 of the 52 tranches bought; at market value (53 m) a fraction of the SPX held (252 m)
    assert at.dataframe[2].value.iloc[4, 0].startswith("24 Mar 2000, on the peak day, SPX 1,527 (+0.0% vs the peak)")
    assert at.dataframe[2].value.iloc[4, 1] == "01 Sep 2000, 161 days after the peak, SPX 1,521 (-0.4% vs the peak)"   # the retest of the March-2000 high and any(m.value.startswith("**2007–2009 bottom — 09 Mar 2009**") for m in at.markdown)
    assert any(m.value.startswith("**2022 bottom — 12 Oct 2022**") for m in at.markdown) and len(at.dataframe[5].value) == 14   # the 2022 correction has its table too
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
    assert stats.loc["lowest (worst start)"].iloc[0] < 300 and stats.loc["median"].iloc[0] < 750   # at the GFC bottom most starts are under water
    assert len(at.dataframe) == 4   # only the 2002 and 2009 bottoms fall inside runs ending 9 Mar 2009
    assert at.dataframe[3].value.iloc[10, 0].startswith("381 − 366 = 15 m (LTV 96%")   # the worst levered trajectory at the bottom, same as when held to today   # at the GFC bottom the median levered start is well above half its lending value
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
