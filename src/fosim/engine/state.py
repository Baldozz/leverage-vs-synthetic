"""Portfolio state, P&L components and results containers for the step engine (SPEC §5.5)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fosim.engine.ledger import EventCounters, Ledger
from fosim.instruments.option_ladder import OptionBook

F64 = NDArray[np.float64]

COMPONENTS: tuple[str, ...] = (
    "equity_mtm",  # held equity portfolio market move
    "spot_mtm",  # deployed spot index (B) market move
    "option_mtm",  # option book mark-to-market change incl. settlement
    "illiquid_mtm",  # illiquid NAV change net of calls/distributions (official marks)
    "dividends",  # cash dividends net of WHT
    "loan_interest",  # negative
    "cash_interest",
    "fees",  # commitment / arrangement fees (negative)
    "transaction_costs",  # equity trading costs, commissions, stamp duty (negative)
    "slippage",  # forced-sale slippage (negative)
    "option_bid_ask",  # ask−mid on purchases, bid−mid on sales (negative)
    "futures_pnl",  # futures overlay variation margin
    "swap_mtm",  # swap MTM change (floating_swapped loan)
    "counterparty_loss",  # OTC counterparty default losses (negative)
    "spending",  # external outflow (negative)
)
COMP_IDX = {c: i for i, c in enumerate(COMPONENTS)}


@dataclass
class PortfolioState:
    """Vectorised state of one strategy over all paths. Amounts in USD, units in index units."""

    P: int
    n_idx: int
    cash: F64 = field(init=False)
    loan: F64 = field(init=False)
    loan_rate: F64 = field(init=False)  # contractual rate fixed at the last reset
    held_units: F64 = field(init=False)  # held equity portfolio units (level H)
    spot_units: F64 = field(init=False)  # (P, n_idx) deployed spot index units
    spot_cost_basis: F64 = field(init=False)  # USD paid for deployed spot (dry powder ROI)
    futures_units: F64 = field(init=False)  # (P, n_idx)
    swap_mtm: F64 = field(init=False)
    book: OptionBook | None = None
    # marks carried from the previous step (for P&L components)
    held_mv: F64 = field(init=False)
    spot_mv: F64 = field(init=False)
    option_mv: F64 = field(init=False)
    option_mv_bid: F64 = field(init=False)
    illiquid_mv: F64 = field(init=False)
    nav: F64 = field(init=False)
    alive: NDArray[np.bool_] = field(init=False)
    # margin bookkeeping
    grace_clock: NDArray[np.int64] = field(init=False)
    # dry powder
    dp_tier_fired: NDArray[np.bool_] = field(init=False)  # (P, n_tiers)
    dp_prev_peak_dd: F64 = field(init=False)
    exit_units_per_month: F64 = field(init=False)  # (P, n_idx)
    exit_months_left: NDArray[np.int64] = field(init=False)
    ladder_pending: NDArray[np.bool_] = field(init=False)  # (P, n_ladder)
    held_units0: float = 0.0  # for spot_selldown build-up
    # greeks of the last mark (B)
    dollar_delta: F64 = field(init=False)
    dollar_delta_smile: F64 = field(init=False)
    dollar_gamma: F64 = field(init=False)
    vega_1pt: F64 = field(init=False)
    theta_year: F64 = field(init=False)
    rho_100bp: F64 = field(init=False)
    dq_100bp: F64 = field(init=False)
    sigma_mid: F64 | None = None
    residual_T: F64 | None = None

    def __post_init__(self) -> None:
        P, n = self.P, self.n_idx
        self.cash = np.zeros(P)
        self.loan = np.zeros(P)
        self.loan_rate = np.zeros(P)
        self.held_units = np.zeros(P)
        self.spot_units = np.zeros((P, n))
        self.spot_cost_basis = np.zeros(P)
        self.futures_units = np.zeros((P, n))
        self.swap_mtm = np.zeros(P)
        self.held_mv = np.zeros(P)
        self.spot_mv = np.zeros(P)
        self.option_mv = np.zeros(P)
        self.option_mv_bid = np.zeros(P)
        self.illiquid_mv = np.zeros(P)
        self.nav = np.zeros(P)
        self.alive = np.ones(P, dtype=bool)
        self.grace_clock = np.zeros(P, dtype=np.int64)
        self.dp_tier_fired = np.zeros((P, 0), dtype=bool)
        self.dp_prev_peak_dd = np.zeros(P)
        self.exit_units_per_month = np.zeros((P, n))
        self.exit_months_left = np.zeros(P, dtype=np.int64)
        self.ladder_pending = np.zeros((P, 0), dtype=bool)
        self.dollar_delta = np.zeros(P)
        self.dollar_delta_smile = np.zeros(P)
        self.dollar_gamma = np.zeros(P)
        self.vega_1pt = np.zeros(P)
        self.theta_year = np.zeros(P)
        self.rho_100bp = np.zeros(P)
        self.dq_100bp = np.zeros(P)

    def total_nav(self) -> F64:
        out: F64 = self.cash - self.loan + self.held_mv + self.spot_mv + self.option_mv + self.illiquid_mv + self.swap_mtm
        return out


@dataclass
class StepRecorder:
    """Time series recorded for every path (P, N+1) and P&L components (P, N+1, n_comp)."""

    P: int
    N: int
    series: dict[str, F64] = field(default_factory=dict)
    components: F64 = field(init=False)

    SERIES: tuple[str, ...] = (
        "nav", "nav_true", "nav_bid", "cash", "loan", "held_mv", "spot_mv", "option_mv", "illiquid_mv", "illiquid_true",
        "utilisation", "lending_value", "loan_rate", "cash_rate", "option_rate_5y", "dollar_delta", "dollar_delta_smile",
        "dollar_gamma", "vega_1pt", "theta_year", "rho_100bp", "dq_100bp", "n_tranches", "futures_notional", "swap_mtm",
        "effective_leverage", "equity_exposure", "spot_cost_basis",
    )

    def __post_init__(self) -> None:
        for s in self.SERIES:
            self.series[s] = np.zeros((self.P, self.N + 1))
        self.components = np.zeros((self.P, self.N + 1, len(COMPONENTS)))

    def record(self, k: int, **kw: F64) -> None:
        for name, arr in kw.items():
            self.series[name][:, k] = arr


@dataclass
class StrategyResult:
    name: str
    recorder: StepRecorder
    events: EventCounters
    ledger: Ledger
    transition_cost: float
    inception: dict[str, float]
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def nav(self) -> F64:
        return self.recorder.series["nav"]

    def series(self, name: str) -> F64:
        return self.recorder.series[name]

    def components_total_by_path(self) -> pd.DataFrame:
        """Horizon totals over steps 1..N (step 0 holds inception costs already inside NAV₀)."""
        tot = self.recorder.components[:, 1:, :].sum(axis=1)
        return pd.DataFrame(tot, columns=list(COMPONENTS))

    def components_mean_by_step(self) -> pd.DataFrame:
        return pd.DataFrame(self.recorder.components.mean(axis=0), columns=list(COMPONENTS))
