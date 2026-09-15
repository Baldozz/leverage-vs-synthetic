"""Strategy C — unlevered cash benchmark: A's illiquid amount, the rest in the held equity portfolio, no loan."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fosim.strategies.base import inception_balance_cfg

if TYPE_CHECKING:
    from fosim.engine.simulator import Simulator, StrategyContext


class UnleveredStrategy:
    name = "C"
    uses_margin = False
    has_option_book = False

    def each_step(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        return None

    def initialise(self, sim: Simulator, ctx: StrategyContext) -> None:
        cfg = sim.cfg
        bal = inception_balance_cfg(cfg)
        st = ctx.state
        st.held_units[:] = bal["equity_C"] / sim.paths.held[:, 0]
        st.held_mv[:] = bal["equity_C"]
        st.cash[:] = bal["cash_C"]
        st.loan[:] = bal["loan_C"]
        ctx.inception.update({"loan": bal["loan_C"], "equity": bal["equity_C"], "cash": bal["cash_C"], "illiquid": bal["illiquid"]})

    def liquidity_waterfall(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        st = ctx.state
        short = st.cash < -1e-9
        if short.any():
            sim.sell_held(ctx, k, np.where(short, -st.cash, 0.0), stressed=sim.stress_flag(k), reason="liquidity sale")
        sim.record_shortfall(ctx, k)

    def month_end(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        sim.spending(ctx, k)
