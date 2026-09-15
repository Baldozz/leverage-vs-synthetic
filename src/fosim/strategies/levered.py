"""Strategy A — levered cash portfolio (status quo) and Strategy D — A with a loan-funded cash buffer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fosim.strategies.base import inception_balance_cfg

if TYPE_CHECKING:
    from fosim.engine.simulator import Simulator, StrategyContext


class LeveredStrategy:
    name = "A"
    uses_margin = True
    has_option_book = False

    def __init__(self, cash_buffer_pct_nav: float = 0.0, name: str = "A") -> None:
        self.buffer = cash_buffer_pct_nav
        self.name = name

    def initialise(self, sim: Simulator, ctx: StrategyContext) -> None:
        cfg = sim.cfg
        bal = inception_balance_cfg(cfg)
        st = ctx.state
        buffer = self.buffer * bal["nav0"]
        st.loan[:] = bal["loan"] + buffer
        st.cash[:] = buffer + bal["cash_A"]
        st.held_units[:] = bal["equity_A"] / sim.paths.held[:, 0]
        st.held_mv[:] = bal["equity_A"]
        fee = sim.loan_terms.arrangement_fee(float(st.loan[0]))
        if fee > 0:
            st.loan += fee  # arrangement fee financed on the facility
            ctx.add_component(0, "fees", np.full(st.P, -fee))
            ctx.ledger.post(0, "loan", np.full(st.P, fee), "arrangement fee financed on the facility")
        ctx.inception.update({"loan": float(st.loan[0]), "equity": bal["equity_A"], "cash": buffer, "illiquid": bal["illiquid"]})

    def each_step(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        return None

    def liquidity_waterfall(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        st = ctx.state
        short = st.cash < 0
        if not short.any():
            return
        need = -np.minimum(st.cash, 0.0)
        # (1) draw on the facility within headroom (limit and lending value at the call threshold)
        lv = sim.lending_value_now(ctx, k)
        head = np.maximum(np.minimum(sim.cfg.loan_terms.facility_limit - st.loan, sim.cfg.leverage.thresholds.call * lv - st.loan), 0.0)
        draw = np.where(short, np.minimum(need, head), 0.0)
        st.loan += draw
        st.cash += draw
        ctx.ledger.post(k, "loan", draw, "facility draw for liquidity shortfall", short)
        ctx.ledger.post(k, "cash", draw, "facility draw for liquidity shortfall", short)
        # (2) sell held equity at normal cost
        still = st.cash < -1e-9
        if still.any():
            sim.sell_held(ctx, k, np.where(still, -st.cash, 0.0), stressed=sim.stress_flag(k), reason="liquidity sale")
        sim.record_shortfall(ctx, k)

    def month_end(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        st = ctx.state
        if sim.cfg.leverage.rebalancing == "maintain_target_leverage":
            target = sim.cfg.leverage.ratio_of_nav * st.nav + self.buffer * st.nav
            delta = np.where(st.alive, target - st.loan, 0.0)
            up = delta > 1.0
            if up.any():
                amt = np.where(up, delta, 0.0)
                st.loan += amt
                st.cash += amt
                ctx.ledger.post(k, "loan", amt, "rebalance: draw to target leverage", up)
                ctx.ledger.post(k, "cash", amt, "rebalance: draw to target leverage", up)
                sim.buy_held(ctx, k, amt, stressed=sim.stress_flag(k), reason="rebalance buy")
            dn = delta < -1.0
            if dn.any():
                amt = np.where(dn, -delta, 0.0)
                sim.sell_held(ctx, k, amt, stressed=sim.stress_flag(k), reason="rebalance sale")
                rep = np.minimum(amt, np.maximum(st.cash, 0.0))
                st.loan -= rep
                st.cash -= rep
                ctx.ledger.post(k, "loan", -rep, "rebalance: repay to target leverage", dn)
                ctx.ledger.post(k, "cash", -rep, "rebalance: repay to target leverage", dn)
        sim.spending(ctx, k)
