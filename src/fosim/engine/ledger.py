"""Double-entry style ledger and event log (SPEC §5.1 events, §5.6).

Every cash flow on the *audited* paths (``ledger_paths``) is a row (step, path, strategy, account,
amount, description). Rows are only stored for the selected paths to keep memory bounded; the
aggregate P&L components for every path live in ``StrategyResult.components``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

F64 = NDArray[np.float64]


@dataclass
class Ledger:
    strategy: str
    ledger_paths: list[int]
    rows: list[tuple[int, int, str, str, float, str]] = field(default_factory=list)
    tranche_snapshots: list[dict[str, object]] = field(default_factory=list)

    def post(self, step: int, account: str, amount: F64, description: str, mask: NDArray[np.bool_] | None = None) -> None:
        """Record ``amount[p]`` for each audited path p (skipping zeros and masked-out paths)."""
        for p in self.ledger_paths:
            if mask is not None and not mask[p]:
                continue
            a = float(amount[p])
            if a != 0.0 and np.isfinite(a):
                self.rows.append((step, p, self.strategy, account, a, description))

    def post_scalar(self, step: int, path: int, account: str, amount: float, description: str) -> None:
        if path in self.ledger_paths and amount != 0.0:
            self.rows.append((step, path, self.strategy, account, amount, description))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["step", "path", "strategy", "account", "amount_usd", "description"])

    def tranche_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.tranche_snapshots)


@dataclass
class EventCounters:
    """Per-path event statistics (all arrays of shape (P,))."""

    margin_warnings: F64
    margin_calls: F64
    forced_liquidations: F64
    cure_sales_volume: F64
    forced_sale_volume: F64
    slippage_loss: F64
    liquidity_shortfalls: F64
    ruin: NDArray[np.bool_]
    ruin_step: NDArray[np.int64]
    min_headroom: F64  # min over time of (1 − utilisation)
    dry_powder_deployments: F64
    dry_powder_deployed_usd: F64
    cash_constrained_purchases: F64
    option_purchases: F64
    option_sales: F64
    roll_cost_usd: F64  # bid/ask + commissions paid on option trades
    futures_margin_calls: F64
    counterparty_losses: F64

    @staticmethod
    def zeros(P: int) -> EventCounters:
        def z() -> F64:
            return np.zeros(P)

        return EventCounters(
            z(), z(), z(), z(), z(), z(), z(), np.zeros(P, dtype=bool), np.full(P, -1, dtype=np.int64), np.full(P, np.inf),
            z(), z(), z(), z(), z(), z(), z(), z(),
        )
