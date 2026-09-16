"""Acceptance (SPEC §13): the Streamlit app launches and every tab renders on the default config,
before and after a simulation run, without raising (Streamlit AppTest headless harness)."""

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
def test_backtest_app_renders_without_tabs() -> None:
    """The primary app (call-vs-cash backtest) is a single page: no tabs, chart + tables rendered at default widgets."""
    at = AppTest.from_file(str(APP.parent / "historical_app.py"), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 0
    assert len(at.dataframe) == 2  # summary table + return-distribution percentiles
    assert not at.warning  # 5-year tenor: no fallback warning
    at.number_input(key="bt_tenor").set_value(7.0).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("7-year" in w.value for w in at.warning)  # falls back to 5y columns until ust_7y/iv_7y exist
    assert any("7-year ATM call" in m.value for m in at.markdown)


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
