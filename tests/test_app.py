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


def test_rotation_page_renders_one_start_date() -> None:
    """Page 1: both portfolios from the first data day to today — five metrics, three charts, the roll tables."""
    at = AppTest.from_file(str(APP.parent / "historical_app.py"), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.metric) == 5 and at.metric[0].label == "SPX sold" and at.metric[3].label == "Loan repaid"
    assert len(at.get("plotly_chart")) == 3
    assert len(at.dataframe) == 2  # rolls by year, every roll
    assert "52 weekly steps" in str(at.metric[4].delta) and at.metric[3].value.endswith(" m") and float(at.metric[3].value[:-2].replace(",", "")) > 250  # loan repaid = 250 m + interest during the build
    assert any("the loan is gone from" in m.value for m in at.markdown) and not any("**a margin call**" in m.value for m in at.markdown)  # bold = a call happened
    at.sidebar.number_input(key="s_weeks").set_value(1).run()  # one shot: one build step, the whole 250 m repaid on day one
    assert not at.exception, [e.value for e in at.exception]
    assert "1 weekly step" in str(at.metric[4].delta) and at.metric[3].value == "250 m"


def test_all_starts_page_weekly_grid() -> None:
    """Page 2: every week from 1997 to the last start whose calls have expired, held to today — two charts, the percentile table, the reading."""
    at = AppTest.from_file(str(APP.parent / "historical_app.py"), default_timeout=600)
    at.session_state["s_grid"] = "every week"
    at.switch_page("views/all_starts.py")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("plotly_chart")) == 2 and len(at.dataframe) == 1   # final value by start date, its distribution; percentile table
    dist = at.dataframe[0].value
    assert list(dist.index)[:3] == ["5th percentile", "25th percentile", "50th percentile"] and (dist.iloc[:, 0] > 750).all() and (dist.iloc[:, 1] > 750).all()   # every start ends above the 750 m it began with
    assert any("Same start, same end" in m.value for m in at.markdown)
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
