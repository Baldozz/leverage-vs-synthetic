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


def test_backtest_app_renders_without_tabs() -> None:
    """The primary app (call-vs-cash backtest) is a single page: no tabs, chart + tables rendered at default widgets."""
    at = AppTest.from_file(str(APP.parent / "historical_app.py"), default_timeout=300)
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
