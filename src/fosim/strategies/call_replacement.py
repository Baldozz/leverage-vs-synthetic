"""Strategy B — unlevered call-option equity replacement with dry powder (SPEC §2, §5.2, §5.3, §5.7).

Transition at t0 from A's balance sheet: sell equities to repay the loan, sell the remaining
equities (both at the configured trading cost), hold the illiquids unchanged, then start the
ladder. Month-end order: (1) ladder sales/purchases, (2) exit rule, (3) dry-powder deployment,
(4) exposure policy, (5) spending.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from fosim.engine.conventions import per_step_rate
from fosim.instruments.option_ladder import LadderSchedule, quote_new_tranche
from fosim.strategies.base import inception_balance_cfg

if TYPE_CHECKING:
    from fosim.engine.simulator import Simulator, StrategyContext

F64 = NDArray[np.float64]


class CallReplacementStrategy:
    name = "B"
    uses_margin = False
    has_option_book = True

    def __init__(self, name: str = "B") -> None:
        self.name = name

    # ------------------------------------------------------------------ inception
    def initialise(self, sim: Simulator, ctx: StrategyContext) -> None:
        cfg = sim.cfg
        bal = inception_balance_cfg(cfg)
        st = ctx.state
        P = st.P
        if bal["custom"]:
            # balance sheet entered directly: no transition trades
            st.cash[:] = bal["cash_B"]
            st.loan[:] = bal["loan_B"]
            st.held_units[:] = bal["equity_B"] / sim.paths.held[:, 0]
            st.held_mv[:] = bal["equity_B"]
        else:
            # start from A's balance sheet, then transition
            st.loan[:] = bal["loan"]
            st.held_units[:] = bal["equity_A"] / sim.paths.held[:, 0]
            st.held_mv[:] = bal["equity_A"]
            sell_to_repay = np.full(P, bal["loan"])
            sim.sell_held(ctx, 0, sell_to_repay, stressed=False, reason="transition: sell equities to repay loan", cost_only_component=True)
            st.loan[:] = 0.0
            st.cash -= bal["loan"]
            ctx.ledger.post(0, "loan", np.full(P, -bal["loan"]), "transition: loan repaid")
            ctx.ledger.post(0, "cash", np.full(P, -bal["loan"]), "transition: loan repaid")
            if cfg.options.buildup_holding == "cash" or cfg.options.ladder_mode == "bullet":
                sim.sell_held(ctx, 0, st.held_mv.copy(), stressed=False, reason="transition: sell remaining equities", cost_only_component=True)
        st.held_units0 = float(st.held_units[0])
        self.schedule = LadderSchedule.from_config(cfg.options)
        st.ladder_pending = np.zeros((P, self.schedule.n_slots), dtype=bool)
        st.dp_tier_fired = np.zeros((P, len(cfg.dry_powder.tiers)), dtype=bool)
        self.E_A0 = bal["reference_equity_exposure"]
        self.sleeve0 = bal["equity_sleeve_B"]
        self.nav0 = bal["nav0"]
        self.uses_margin = bool(bal["loan_B"] > 0)
        ctx.inception.update({"loan": bal["loan_B"], "equity": bal["equity_B"], "illiquid": bal["illiquid"], "equity_sleeve": bal["equity_sleeve_B"], "A_equity_exposure": bal["reference_equity_exposure"]})

    # ------------------------------------------------------------------ fixed-notional schedule (weekly / monthly)
    def each_step(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        opt = sim.cfg.options
        if opt.ladder_mode == "fixed_notional_schedule" and opt.purchase_frequency == "weekly":
            self._scheduled_purchase(sim, ctx, k)

    def _scheduled_purchase(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        """Buy ``notional_per_purchase`` of fresh calls (per the strike mode) while the active book is below target."""
        opt = sim.cfg.options
        st = ctx.state
        assert st.book is not None
        per = opt.notional_pct_nav * st.nav if opt.notional_pct_nav is not None else np.full(st.P, float(opt.notional_per_purchase or 0.0))
        if opt.target_total_notional is not None:
            # only the scheduled tranches (origin 0) count toward the target; dry-powder / rebalance tranches sit on top
            active = (st.book.notional * st.book.active * (st.book.origin == 0)).sum(axis=1)
            notional = np.minimum(np.maximum(opt.target_total_notional - active, 0.0), per)
        else:
            notional = per  # no cap: keep adding every purchase step
        want = (notional > 1.0) & st.alive
        if opt.iv_pause_threshold is not None:
            atm = sim.vol_sources[0].parametric.atm_vol(opt.tenor_years, sim.paths.iv_short[:, k])
            want &= atm <= opt.iv_pause_threshold
        if not want.any():
            return
        quotes = [quote_new_tranche(opt, sim.vol_sources[i], sim.snapshot(k), i) for i in range(sim.n_idx)]
        sim.buy_extra_tranche(ctx, k, want, notional, quotes, origin=0, reason=f"scheduled purchase ({opt.purchase_frequency})")

    # ------------------------------------------------------------------ liquidity
    def liquidity_waterfall(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        st = ctx.state
        short = st.cash < -1e-9
        if not short.any():
            return
        # (1) deployed spot index
        need = np.where(short, -st.cash, 0.0)
        sim.sell_spot(ctx, k, need, stressed=sim.stress_flag(k), reason="liquidity sale of deployed spot")
        short = st.cash < -1e-9
        if short.any() and st.held_units.any():
            sim.sell_held(ctx, k, np.where(short, -st.cash, 0.0), stressed=sim.stress_flag(k), reason="liquidity sale of held equities")
        short = st.cash < -1e-9
        if short.any():
            # (2) options at bid, pro rata across tranches
            need = np.where(short, -st.cash, 0.0)
            has_book = st.option_mv_bid > 1e-9
            frac = np.where(has_book, np.minimum(need / np.where(has_book, st.option_mv_bid, 1.0), 1.0), 0.0)
            sim.sell_option_fraction(ctx, k, frac, reason="liquidity sale of options at bid")
        short = st.cash < -1e-9
        if short.any() and np.any(st.futures_units != 0):
            sim.close_futures(ctx, k, short, reason="liquidity: close futures overlay")
        sim.record_shortfall(ctx, k)

    # ------------------------------------------------------------------ month end
    def month_end(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        month = int(sim.grid.month_index[k])
        self._ladder(sim, ctx, k, month)
        self._exit_rule(sim, ctx, k)
        self._dry_powder(sim, ctx, k)
        self._exposure_policy(sim, ctx, k)
        sim.spending(ctx, k)

    # --- sizing ------------------------------------------------------------------------------
    def total_target_notional(self, sim: Simulator, ctx: StrategyContext, k: int, quotes: list, at_roll: bool) -> F64:  # type: ignore[type-arg]
        """Total book target notional (USD) per path under ``options.sizing_mode`` at this purchase."""
        cfg = sim.cfg
        opt = cfg.options
        st = ctx.state
        w = sim.paths.book_weights
        # book-level delta and premium per unit notional, weighted across indices
        delta = sum(w[i] * quotes[i].delta for i in range(len(quotes)))
        prem = sum(w[i] * quotes[i].price_ask / sim.paths.S[:, k, i] for i in range(len(quotes)))
        scale = st.nav / self.nav0 if (opt.sizing_basis == "pct_nav" or (at_roll and opt.resize_on_roll == "target_pct_nav")) else np.ones(st.P)
        E = self.E_A0 * scale
        mode = opt.sizing_mode
        if mode == "notional_match":
            out = E
        elif mode == "delta_match":
            out = E / delta
        elif mode == "beta_adjusted_delta_match":
            out = E * sim._beta_held / delta
        elif mode == "delta_fraction_of_A":
            assert opt.sizing_param is not None
            out = opt.sizing_param * E / delta
        elif mode == "premium_budget":
            assert opt.sizing_param is not None
            out = opt.sizing_param * (self.nav0 * scale) / prem
        elif mode == "cash_reserve_target":
            assert opt.sizing_param is not None
            budget = np.maximum(self.sleeve0 * scale - opt.sizing_param * self.nav0 * scale, 0.0)
            out = budget / prem
        else:
            raise ValueError(mode)
        return np.asarray(out, dtype=np.float64)

    def _ladder(self, sim: Simulator, ctx: StrategyContext, k: int, month: int) -> None:
        cfg = sim.cfg
        opt = cfg.options
        st = ctx.state
        assert st.book is not None
        if opt.ladder_mode == "fixed_notional_schedule":
            if opt.purchase_frequency == "monthly":
                self._scheduled_purchase(sim, ctx, k)
            return
        sch = self.schedule
        M = sch.n_slots
        quotes = [quote_new_tranche(opt, sim.vol_sources[i], sim.snapshot(k), i) for i in range(sim.n_idx)]
        # IV pause rule
        paused = np.zeros(st.P, dtype=bool)
        if opt.iv_pause_threshold is not None:
            atm5 = sim.vol_sources[0].parametric.atm_vol(opt.tenor_years, sim.paths.iv_short[:, k])
            paused = atm5 > opt.iv_pause_threshold
        for j in range(M):
            scheduled = sch.is_purchase_month(j, month)
            if sch.is_roll_sale_month(j, month):
                act = st.book.active[:, j]
                if act.any():
                    sim.sell_slot(ctx, k, j, act, reason=f"ladder roll: sell slot {j} at bid")
            want = scheduled | st.ladder_pending[:, j]
            want &= ~st.book.active[:, j] & st.alive
            if not want.any():
                continue
            # spot sell-down build-up: sell 1/M of the initial held position for this tranche's funding
            if opt.buildup_holding == "spot_selldown" and month < sch.buildup_months and st.held_units.any():
                units = np.where(want, min(st.held_units0 / M, 1e300), 0.0)
                units = np.minimum(units, st.held_units)
                sim.sell_held(ctx, k, units * sim.paths.held[:, k], stressed=sim.stress_flag(k), reason="build-up: sell held equities 1/M")
            at_roll = month >= sch.buildup_months if sch.mode != "bullet" else month > 0
            if at_roll and opt.resize_on_roll == "keep_notional":
                target_total = np.where(want, st.book.notional[:, j] * M, 0.0)
                target_total = np.where(target_total > 0, target_total, self.total_target_notional(sim, ctx, k, quotes, at_roll))
            else:
                target_total = self.total_target_notional(sim, ctx, k, quotes, at_roll)
            per_tranche = target_total / M
            buy = want & ~paused
            st.ladder_pending[:, j] = want & paused
            if buy.any():
                sim.buy_tranche(ctx, k, j, buy, per_tranche, quotes, origin=0, reason=f"ladder purchase slot {j}")

    def _exit_rule(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        cfg = sim.cfg.dry_powder
        st = ctx.state
        if not cfg.exit_rule.enabled:
            return
        dd = sim.book_dd[:, k]
        deployed = st.spot_units.sum(axis=1) > 0
        start = deployed & (dd >= -1e-12) & (st.exit_months_left == 0)
        if start.any():
            K = cfg.exit_rule.convert_over_months
            st.exit_units_per_month[start] = st.spot_units[start] / K
            st.exit_months_left[start] = K
        run = st.exit_months_left > 0
        if run.any():
            units = np.minimum(st.exit_units_per_month, st.spot_units) * run[:, None]
            proceeds = sim.sell_spot_units(ctx, k, units, stressed=sim.stress_flag(k), reason="exit rule: convert deployed spot")
            st.exit_months_left[run] -= 1
            # buy a call tranche with the proceeds (extra slot, ATM)
            quotes = [quote_new_tranche(sim.cfg.options, sim.vol_sources[i], sim.snapshot(k), i, force_atm=True) for i in range(sim.n_idx)]
            prem = sum(sim.paths.book_weights[i] * quotes[i].price_ask / sim.paths.S[:, k, i] for i in range(sim.n_idx))
            notional = np.where(run, proceeds / prem, 0.0)
            sim.buy_extra_tranche(ctx, k, run & (notional > 0), notional, quotes, origin=4, reason="exit rule: buy call tranche")

    def _dry_powder(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        cfg = sim.cfg.dry_powder
        st = ctx.state
        if not cfg.enabled or not cfg.tiers:
            return
        dd = sim.book_dd[:, k]
        iv = sim.paths.iv_short[:, k]
        # re-arm all tiers after a new peak
        st.dp_tier_fired[dd >= -1e-12, :] = False
        reserve = self.liquidity_reserve(sim, ctx, k)
        for ti, tier in enumerate(cfg.tiers):
            hit = np.ones(st.P, dtype=bool)
            if tier.index_drawdown is not None:
                hit &= dd <= tier.index_drawdown
            if tier.iv_short_above is not None:
                hit &= iv >= tier.iv_short_above
            fire = hit & ~st.dp_tier_fired[:, ti] & st.alive
            if not fire.any():
                continue
            st.dp_tier_fired[fire, ti] = True
            deployable = np.maximum(st.cash - reserve, 0.0)
            amt = np.where(fire, deployable * tier.deploy_pct_of_deployable, 0.0)
            go = fire & (amt > 1.0)
            if not go.any():
                continue
            ctx.events.dry_powder_deployments[go] += 1
            ctx.events.dry_powder_deployed_usd[go] += amt[go]
            if cfg.instrument == "spot_index":
                sim.buy_spot(ctx, k, amt, stressed=sim.stress_flag(k), reason=f"dry powder tier {ti + 1}: buy spot index")
            else:
                quotes = [quote_new_tranche(sim.cfg.options, sim.vol_sources[i], sim.snapshot(k), i, force_atm=True) for i in range(sim.n_idx)]
                prem = sum(sim.paths.book_weights[i] * quotes[i].price_ask / sim.paths.S[:, k, i] for i in range(sim.n_idx))
                sim.buy_extra_tranche(ctx, k, go, amt / prem, quotes, origin=2, reason=f"dry powder tier {ti + 1}: buy call tranche")

    def liquidity_reserve(self, sim: Simulator, ctx: StrategyContext, k: int) -> F64:
        """max(floor % NAV, projected capital calls over N months × stress multiplier, planned spending over N months)."""
        cfg = sim.cfg
        lr = cfg.dry_powder.liquidity_reserve
        st = ctx.state
        floor = lr.floor_pct_nav * st.nav
        n_steps = round(lr.capital_call_months / 12 * sim.grid.steps_per_year)
        calls = np.zeros(st.P)
        for j, ill in enumerate(cfg.illiquids):
            if ill.type == "closed_end" and ill.ta_model is not None:
                rc = per_step_rate(ill.ta_model.rc, sim.grid.dt)
                calls += sim.paths.illiquids.unfunded[:, k, j] * (1 - (1 - rc) ** n_steps)
        calls *= lr.capital_call_stress_multiplier
        spend = (cfg.spending.pct_nav_pa * st.nav + cfg.spending.fixed_usd_pa) * lr.capital_call_months / 12
        return np.asarray(np.maximum.reduce([floor, calls, spend]), dtype=np.float64)

    def _exposure_policy(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        cfg = sim.cfg
        ex = cfg.exposure
        st = ctx.state
        if ex.policy == "static":
            return
        assert st.book is not None
        sim.mark_options(ctx, k)  # refresh deltas after ladder / dry-powder trades
        current = sim.equity_dollar_delta(ctx, k)
        target = self.exposure_target(sim, ctx, k)
        gap = np.where(st.alive, target - current, 0.0)
        if ex.policy == "rebalance_bands":
            reserve = self.liquidity_reserve(sim, ctx, k)
            low = st.alive & (current < target * (1 - ex.band_pp))
            high = st.alive & (current > target * (1 + ex.band_pp))
            if low.any():
                quotes = [quote_new_tranche(cfg.options, sim.vol_sources[i], sim.snapshot(k), i) for i in range(sim.n_idx)]
                delta = sum(sim.paths.book_weights[i] * quotes[i].delta for i in range(sim.n_idx))
                prem = sum(sim.paths.book_weights[i] * quotes[i].price_ask / sim.paths.S[:, k, i] for i in range(sim.n_idx))
                notional = np.where(low, gap / delta, 0.0)
                cap = np.maximum(st.cash - reserve, 0.0) / prem
                binding = low & (notional > cap)
                ctx.events.cash_constrained_purchases[binding] += 1
                notional = np.minimum(notional, cap)
                sim.buy_extra_tranche(ctx, k, low & (notional > 0), notional, quotes, origin=1, reason="rebalance_bands: buy tranche")
            if high.any():
                opt_delta = st.dollar_delta if cfg.options.delta_definition == "bsm" else st.dollar_delta_smile
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac = np.where(high & (opt_delta > 0), np.clip((current - target) / np.maximum(opt_delta, 1e-300), 0.0, 1.0), 0.0)
                sim.sell_option_fraction(ctx, k, frac, reason="rebalance_bands: sell tranches at bid")
        elif ex.policy == "restrike_on_move":
            S = np.take_along_axis(sim.snapshot(k).S, st.book.index_id, axis=1)
            m = S / st.book.strike
            out = st.book.active & ((m < ex.moneyness_band[0]) | (m > ex.moneyness_band[1]))
            if out.any():
                quotes = [quote_new_tranche(cfg.options, sim.vol_sources[i], sim.snapshot(k), i) for i in range(sim.n_idx)]
                for j in range(st.book.n_slots):
                    mask = out[:, j]
                    if not mask.any():
                        continue
                    notional = np.where(mask, st.book.notional[:, j], 0.0)
                    sim.sell_slot(ctx, k, j, mask, reason=f"restrike: sell slot {j} at bid")
                    sim.buy_tranche(ctx, k, j, mask, notional, quotes, origin=3, reason=f"restrike: buy slot {j} ATM")
        elif ex.policy == "spot_topup":
            reserve = self.liquidity_reserve(sim, ctx, k)
            buy = gap > 1.0
            if buy.any():
                amt = np.minimum(np.where(buy, gap, 0.0), np.maximum(st.cash - reserve, 0.0))
                sim.buy_spot(ctx, k, amt, stressed=sim.stress_flag(k), reason="spot_topup: buy spot index")
            sell = (gap < -1.0) & (st.spot_mv > 0)
            if sell.any():
                sim.sell_spot(ctx, k, np.where(sell, np.minimum(-gap, st.spot_mv), 0.0), stressed=sim.stress_flag(k), reason="spot_topup: sell spot index")
        elif ex.policy == "futures_overlay":
            # set futures so that total delta = target (both directions); notional split by book weights
            S = sim.paths.S[:, k, :]
            new_units = (target - (current - (st.futures_units * S).sum(axis=1)))[:, None] * sim.paths.book_weights[None, :] / S
            change = new_units - st.futures_units
            cost = np.abs(change * S).sum(axis=1) * sim.cost_bps(sim.stress_flag(k))
            st.futures_units[:] = new_units
            st.cash -= cost
            ctx.add_component(k, "transaction_costs", -cost)
            ctx.ledger.post(k, "cash", -cost, "futures overlay: trading cost")

    def exposure_target(self, sim: Simulator, ctx: StrategyContext, k: int) -> F64:
        ex = sim.cfg.exposure.target
        st = ctx.state
        if ex.type == "pct_of_A_exposure":
            scale = st.nav / self.nav0 if sim.cfg.options.sizing_basis == "pct_nav" else np.ones(st.P)
            return np.asarray(ex.value * self.E_A0 * scale, dtype=np.float64)
        if ex.type == "usd":
            return np.full(st.P, ex.value)
        return np.asarray(ex.value * st.nav, dtype=np.float64)
