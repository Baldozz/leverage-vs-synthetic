"""Step engine (SPEC §5.5) — runs every strategy on the same immutable ``MarketPaths``.

Order of operations within each step k → k+1 (implemented literally in ``_step``):
 1. read state at k;  2. market at k+1 (from MarketPaths);
 3. accrue interest on start-of-step cash and loan at rates fixed at k (pay or capitalise);
 4. credit dividends (net of WHT) on start-of-step holdings;
 5. illiquid NAV update, capital calls and distributions;
 6. option expiries settled to cash;  7. mark all positions at k+1;
 8. liquidity check and strategy waterfall;  9. margin check and cure/liquidation (A, D);
10. month-end actions;  11. record;  12. accounting identity assertion (1e-6 USD).

P vs Q: all P-measure dynamics come from ``MarketPaths``; every option price here is Q-measure
through ``VolSource`` and the zero curve. Long options never enter a lending value (asserted).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fosim.config.schema import SimConfig
from fosim.engine.conventions import TimeGrid, dividend_cash
from fosim.engine.ledger import EventCounters, Ledger
from fosim.engine.state import COMP_IDX, PortfolioState, StepRecorder, StrategyResult
from fosim.instruments.cash import cash_interest
from fosim.instruments.lombard_loan import (
    LoanTerms,
    cure_sale_size,
    lending_value,
    ltv_multiplier,
    utilisation,
)
from fosim.instruments.option_ladder import MarketSnapshot, NewTrancheQuote, OptionBook, mark_book
from fosim.market.equity import running_drawdown
from fosim.market.generator import spawn_streams
from fosim.market.paths import MarketPaths
from fosim.pricing.black_scholes import bsm_price
from fosim.pricing.vol_surface import VolSource
from fosim.strategies.base import Strategy, held_beta_to_book
from fosim.strategies.call_replacement import CallReplacementStrategy
from fosim.strategies.levered import LeveredStrategy
from fosim.strategies.unlevered import UnleveredStrategy

F64 = NDArray[np.float64]
B_ = NDArray[np.bool_]

IDENTITY_TOL = 1e-6


class AccountingIdentityError(AssertionError):
    pass


@dataclass
class StrategyContext:
    strategy: Strategy
    state: PortfolioState
    recorder: StepRecorder
    events: EventCounters
    ledger: Ledger
    inception: dict[str, float] = field(default_factory=dict)
    transition_cost: float = 0.0

    def add_component(self, k: int, name: str, amount: F64) -> None:
        self.recorder.components[:, k, COMP_IDX[name]] += amount


class Simulator:
    def __init__(self, cfg: SimConfig, paths: MarketPaths, ledger_paths: list[int] | None = None, base_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.paths = paths
        self.grid: TimeGrid = paths.grid
        self.P = paths.n_paths
        self.N = paths.n_steps
        self.n_idx = paths.n_indices
        self.ledger_paths = ledger_paths if ledger_paths is not None else [0]
        self.loan_terms = LoanTerms(cfg.loan_terms, cfg.leverage)
        self.book_dd = paths.book_drawdown()
        ti = cfg.dry_powder.trigger_index or ("SPX" if "SPX" in cfg.index_names else None)
        self.trigger_dd = running_drawdown(paths.S[:, :, cfg.index_names.index(ti)]) if ti is not None else self.book_dd
        self.held_dd = np.minimum(paths.held / np.maximum.accumulate(paths.held, axis=1) - 1.0, 0.0)
        # Q-measure inputs
        r0_5y = float(self._zero_rate_fn(0)(cfg.options.tenor_years)[0]) + cfg.pricing.option_funding_spread
        self.vol_sources: list[VolSource] = [
            VolSource.build(
                cfg.implied_vol, cfg.pricing, ic.name, ic.spot, r0_5y,
                r0_5y if ic.underlying_type == "excess_return" else self._q_curve(i)(cfg.options.tenor_years),
                float(paths.iv_short[0, 0]), base_dir,
            )
            for i, ic in enumerate(cfg.equity_indices)
        ]
        self._q_fns: list[Callable[[float], float]] = [self._q_curve(i) for i in range(self.n_idx)]
        self._q_equals_r: tuple[bool, ...] = tuple(ic.underlying_type == "excess_return" for ic in cfg.equity_indices)
        self._beta_held = held_beta_to_book(cfg.held_equity_portfolio.betas, cfg.index_names, paths.book_weights, cfg.held_equity_portfolio.beta_to_option_book)
        self._cp_rng = spawn_streams(cfg.run.seed)["counterparty"] if cfg.options.counterparty.enabled else None
        self.timings: dict[str, float] = {}
        self._settled: dict[str, F64] = {}

    # ------------------------------------------------------------------ Q-measure helpers
    def _q_curve(self, i: int) -> Callable[[float], float]:
        ic = self.cfg.equity_indices[i]
        tenors = np.asarray(ic.pricing_dividend_curve.tenors, dtype=np.float64)
        qs = np.asarray(ic.pricing_dividend_curve.q, dtype=np.float64)
        if ic.underlying_type == "total_return":
            return lambda T: 0.0
        return lambda T: float(np.interp(T, tenors, qs))

    def _zero_rate_fn(self, k: int) -> Callable[[float], F64]:
        """Discount curve at step k: the observed option-rate series (flat across tenors) when supplied, else the rate model."""
        opt = self.paths.series.get("option_rate")
        if opt is not None:
            return lambda tau: np.broadcast_to(opt[:, k], (self.P,)).astype(np.float64)
        return lambda tau: self.paths.zero_rate(k, tau)

    def loan_base(self, k: int) -> F64:
        s = self.paths.series.get("loan_base")
        return s[:, k] if s is not None else self.paths.r_short[:, k]

    def cash_yield(self, k: int) -> F64:
        s = self.paths.series.get("cash_yield")
        return s[:, k] - self.cfg.rates.cash.spread if s is not None else self.paths.r_short[:, k] - self.cfg.rates.cash.spread

    def snapshot(self, k: int) -> MarketSnapshot:
        stressed = self.paths.iv_short[:, k] > self.cfg.pricing.stress_iv_short_above
        return MarketSnapshot(
            k=k, t=float(self.grid.t[k]), month=int(self.grid.month_index[k]), S=self.paths.S[:, k, :],
            iv_short=self.paths.iv_short[:, k], zero_rate=self._zero_rate_fn(k),
            q_curve=self._q_fns, funding_spread=self.cfg.pricing.option_funding_spread, stressed=stressed, q_equals_r=self._q_equals_r,
        )

    def stress_flag(self, k: int) -> B_:
        out: B_ = self.paths.iv_short[:, k] > self.cfg.pricing.stress_iv_short_above
        return out

    def cost_bps(self, stressed: B_ | bool) -> F64:
        c = self.cfg.costs
        base = (c.equity_bps + c.stamp_duty_bps) * 1e-4
        stress = (c.equity_bps_stressed + c.stamp_duty_bps) * 1e-4
        out: F64 = np.where(np.asarray(stressed), stress, base) * np.ones(self.P)
        return out

    # ------------------------------------------------------------------ equity trades
    def sell_held(self, ctx: StrategyContext, k: int, amount: F64, stressed: B_ | bool, reason: str, cost_only_component: bool = False) -> None:
        st = ctx.state
        H = self.paths.held[:, k]
        mv = st.held_units * H
        amt = np.minimum(np.maximum(amount, 0.0), mv)
        units = np.where(H > 0, amt / H, 0.0)
        cost = amt * self.cost_bps(stressed)
        st.held_units -= units
        st.held_mv = st.held_units * H
        st.cash += amt - cost
        ctx.add_component(k, "transaction_costs", -cost)
        if cost_only_component:
            ctx.transition_cost += float(cost.mean())
        ctx.ledger.post(k, "cash", amt, reason)
        ctx.ledger.post(k, "cash", -cost, reason + " (trading cost)")

    def buy_held(self, ctx: StrategyContext, k: int, amount: F64, stressed: B_ | bool, reason: str) -> None:
        st = ctx.state
        H = self.paths.held[:, k]
        amt = np.maximum(amount, 0.0)
        cost = amt * self.cost_bps(stressed)
        st.held_units += amt / H
        st.held_mv = st.held_units * H
        st.cash -= amt + cost
        ctx.add_component(k, "transaction_costs", -cost)
        ctx.ledger.post(k, "cash", -amt, reason)
        ctx.ledger.post(k, "cash", -cost, reason + " (trading cost)")

    def buy_spot(self, ctx: StrategyContext, k: int, amount: F64, stressed: B_ | bool, reason: str) -> None:
        st = ctx.state
        S = self.paths.S[:, k, :]
        amt = np.maximum(amount, 0.0)
        cost = amt * self.cost_bps(stressed)
        st.spot_units += amt[:, None] * self.paths.book_weights[None, :] / S
        st.spot_mv = (st.spot_units * S).sum(axis=1)
        st.spot_cost_basis += amt
        st.cash -= amt + cost
        ctx.add_component(k, "transaction_costs", -cost)
        ctx.ledger.post(k, "cash", -amt, reason)
        ctx.ledger.post(k, "cash", -cost, reason + " (trading cost)")

    def sell_spot(self, ctx: StrategyContext, k: int, amount: F64, stressed: B_ | bool, reason: str) -> F64:
        st = ctx.state
        S = self.paths.S[:, k, :]
        mv = (st.spot_units * S).sum(axis=1)
        amt = np.minimum(np.maximum(amount, 0.0), mv)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(mv > 0, amt / np.maximum(mv, 1e-300), 0.0)
        units = st.spot_units * frac[:, None]
        return self.sell_spot_units(ctx, k, units, stressed, reason)

    def sell_spot_units(self, ctx: StrategyContext, k: int, units: F64, stressed: B_ | bool, reason: str) -> F64:
        st = ctx.state
        S = self.paths.S[:, k, :]
        units = np.minimum(np.maximum(units, 0.0), st.spot_units)
        amt = (units * S).sum(axis=1)
        cost = amt * self.cost_bps(stressed)
        mv_before = (st.spot_units * S).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            basis_frac = np.where(mv_before > 0, amt / np.maximum(mv_before, 1e-300), 0.0)
        st.spot_cost_basis *= 1.0 - basis_frac
        st.spot_units -= units
        st.spot_mv = (st.spot_units * S).sum(axis=1)
        st.cash += amt - cost
        ctx.add_component(k, "transaction_costs", -cost)
        ctx.ledger.post(k, "cash", amt, reason)
        ctx.ledger.post(k, "cash", -cost, reason + " (trading cost)")
        return np.asarray(amt - cost, dtype=np.float64)

    # ------------------------------------------------------------------ option trades
    def _book(self, ctx: StrategyContext) -> OptionBook:
        assert ctx.state.book is not None
        return ctx.state.book

    def buy_tranche(self, ctx: StrategyContext, k: int, slot: int, mask: B_, notional: F64, quotes: list[NewTrancheQuote], origin: int, reason: str) -> None:
        """Buy a fresh tranche into ``slot`` for paths in ``mask``: notional split across indices by book weights."""
        st = ctx.state
        book = self._book(ctx)
        w = self.paths.book_weights
        # per-index tranches share the slot only when n_idx == 1; with several indices the slot holds index 0
        # and sibling slots (slot + n_ladder*i) hold the others — handled by the slot layout below
        cash_avail = np.maximum(st.cash, 0.0)
        prem_per_notional = sum(w[i] * (quotes[i].price_ask / self.paths.S[:, k, i]) for i in range(self.n_idx))
        prem_per_notional = prem_per_notional * (1 + self.cfg.pricing.commission_bps_notional * 1e-4)
        max_notional = np.where(prem_per_notional > 0, cash_avail / np.maximum(prem_per_notional, 1e-300), 0.0)
        constrained = mask & (notional > max_notional + 1e-9)
        ctx.events.cash_constrained_purchases[constrained] += 1
        n_use = np.where(mask, np.minimum(np.maximum(notional, 0.0), max_notional), 0.0)
        # a purchase cut to less than 1 % of the requested notional by the cash constraint is skipped
        # (flagged above) rather than executed as a dust tranche that would occupy a slot
        n_use = np.where(n_use < 0.01 * np.maximum(notional, 0.0), 0.0, n_use)
        for i in range(self.n_idx):
            j = self._slot_for(slot, i)
            q = quotes[i]
            S = self.paths.S[:, k, i]
            units = np.where(mask, n_use * w[i] / S, 0.0)
            premium = units * q.price_ask
            commission = n_use * w[i] * self.cfg.pricing.commission_bps_notional * 1e-4
            mid_value = units * q.price_mid
            m = mask & (units > 0)
            book.units[m, j] = units[m]
            book.strike[m, j] = q.strike[m]
            book.expiry_step[m, j] = k + round(q.T * self.grid.steps_per_year)
            book.purchase_step[m, j] = k
            book.index_id[m, j] = i
            book.k_ref[m, j] = q.k_ref[m]
            book.cost[m, j] = premium[m]
            book.notional[m, j] = (n_use * w[i])[m]
            book.vol_at_purchase[m, j] = q.sigma_ask[m]
            book.source[m, j] = q.source
            book.active[m, j] = True
            book.origin[m, j] = origin
            st.cash -= premium + commission
            st.option_mv += mid_value
            ctx.add_component(k, "option_bid_ask", -(premium - mid_value))
            ctx.add_component(k, "transaction_costs", -commission)
            ctx.events.roll_cost_usd += (premium - mid_value) + commission
            ctx.events.option_purchases[m] += 1
            ctx.ledger.post(k, "cash", -premium, f"{reason} [{self.cfg.index_names[i]}] premium at ask", m)
            ctx.ledger.post(k, "memo:bid_ask_cost", -(premium - mid_value), f"{reason} bid/ask cost (ask − mid, non-cash memo)", m)
            ctx.ledger.post(k, "cash", -commission, f"{reason} commission", m & (commission > 0))

    def _slot_for(self, slot: int, index: int) -> int:
        book_slots = self._n_logical_slots
        return slot + index * book_slots

    def buy_extra_tranche(self, ctx: StrategyContext, k: int, mask: B_, notional: F64, quotes: list[NewTrancheQuote], origin: int, reason: str) -> None:
        book = self._book(ctx)
        free = book.free_extra_slot(mask)
        no_slot = mask & (free < 0)
        if no_slot.any():
            raise RuntimeError("option book has no free extra slot; increase options.extra_slots")
        for j in np.unique(free[mask]):
            m = mask & (free == j)
            self.buy_tranche(ctx, k, int(j), m, notional, quotes, origin, reason)

    def sell_slot(self, ctx: StrategyContext, k: int, slot: int, mask: B_, reason: str) -> None:
        """Sell the whole tranche in ``slot`` (all index siblings) at bid."""
        st = ctx.state
        book = self._book(ctx)
        snap = self.snapshot(k)
        for i in range(self.n_idx):
            j = self._slot_for(slot, i)
            m = mask & book.active[:, j]
            if not m.any():
                continue
            T = (book.expiry_step[:, j] - k) / self.grid.steps_per_year
            S = snap.S[:, i]
            # tranche lifetimes in a slot follow the deterministic schedule: one residual tenor per slot
            T_slot = float(T[m][0])
            assert np.all(np.abs(T[m] - T_slot) < 1e-12), "mixed residual tenors within one slot"
            Tm = np.full(int(m.sum()), max(T_slot, 0.0))
            r = snap.zero_rate(Tm[0])[m] + snap.funding_spread
            q: F64 | float = r if self._q_equals_r[i] else self._q_fns[i](Tm[0])
            K = book.strike[m, j]
            F = S[m] * np.exp((r - q) * Tm)
            k_use = np.log(K / F) if self.cfg.implied_vol.skew_mode == "sticky_moneyness" else book.k_ref[m, j]
            sig, _ = self.vol_sources[i].vol(k_use, float(Tm[0]), snap.iv_short[m], snap.month)
            half = self.vol_sources[i].half_spread(float(Tm[0]), None, snap.iv_short[m], "unwind")
            mid = bsm_price(S[m], K, r, q, sig, Tm, "call")
            bid = bsm_price(S[m], K, r, q, np.maximum(sig - half, 1e-8), Tm, "call")
            units = book.units[m, j]
            proceeds = units * bid
            mid_value = units * mid
            st.cash[m] += proceeds
            st.option_mv[m] -= mid_value
            comp = np.zeros(self.P)
            comp[m] = proceeds - mid_value
            ctx.add_component(k, "option_bid_ask", comp)
            ctx.events.roll_cost_usd[m] += mid_value - proceeds
            ctx.events.option_sales[m] += 1
            pr = np.zeros(self.P)
            pr[m] = proceeds
            ctx.ledger.post(k, "cash", pr, f"{reason} [{self.cfg.index_names[i]}] proceeds at bid", m)
            ctx.ledger.post(k, "memo:bid_ask_cost", comp, f"{reason} bid/ask cost (bid − mid, non-cash memo)", m)
            book.clear(np.flatnonzero(m), j)

    def sell_option_fraction(self, ctx: StrategyContext, k: int, frac: F64, reason: str) -> None:
        """Sell a fraction of every active tranche at bid (pro rata), per path."""
        st = ctx.state
        book = self._book(ctx)
        snap = self.snapshot(k)
        f = np.clip(frac, 0.0, 1.0)
        anym = f > 0
        if not anym.any():
            return
        mark = mark_book(book, self.vol_sources, snap, self.grid.steps_per_year, self.cfg.implied_vol.skew_mode, self.cfg.pricing)
        # bid prices per slot
        S = np.take_along_axis(snap.S, book.index_id, axis=1)
        T = mark.residual_T
        r = np.zeros_like(T)
        q = np.zeros_like(T)
        for j in range(book.n_slots):
            for tv in np.unique(np.round(T[:, j], 10)):
                mm = np.round(T[:, j], 10) == tv
                r[mm, j] = snap.zero_rate(float(tv))[mm] + snap.funding_spread
            for i in range(self.n_idx):
                mi = book.index_id[:, j] == i
                for tv in np.unique(np.round(T[mi, j], 10)):
                    mm = mi & (np.round(T[:, j], 10) == tv)
                    q[mm, j] = self._q_fns[i](float(tv))
                if self._q_equals_r[i]:
                    q[mi, j] = r[mi, j]
        half = np.zeros_like(T)
        for i in range(self.n_idx):
            hs = self.vol_sources[i].half_spread(0.0, None, snap.iv_short, "unwind")
            half[book.index_id == i] = np.broadcast_to(hs[:, None], half.shape)[book.index_id == i]
        bid = bsm_price(S, np.where(book.active, book.strike, S), r, q, np.maximum(mark.sigma_mid - half, 1e-8), T, "call")
        units_sold = book.units * book.active * f[:, None]
        proceeds = (units_sold * bid).sum(axis=1)
        mid_value = (units_sold * mark.greeks.price).sum(axis=1)
        st.cash += proceeds
        st.option_mv -= mid_value
        ctx.add_component(k, "option_bid_ask", proceeds - mid_value)
        ctx.events.roll_cost_usd += mid_value - proceeds
        ctx.events.option_sales[anym] += 1
        ctx.ledger.post(k, "cash", proceeds, reason, anym)
        ctx.ledger.post(k, "memo:bid_ask_cost", proceeds - mid_value, reason + " bid/ask cost (bid − mid, non-cash memo)", anym)
        book.units -= units_sold
        book.cost *= 1.0 - f[:, None]
        book.notional *= 1.0 - f[:, None]
        gone = book.active & (book.units <= 1e-12)
        book.active[gone] = False
        book.units[gone] = 0.0

    def close_futures(self, ctx: StrategyContext, k: int, mask: B_, reason: str) -> None:
        st = ctx.state
        S = self.paths.S[:, k, :]
        cost = (np.abs(st.futures_units * S).sum(axis=1) * self.cost_bps(self.stress_flag(k))) * mask
        st.futures_units[mask] = 0.0
        st.cash -= cost
        ctx.add_component(k, "transaction_costs", -cost)
        ctx.ledger.post(k, "cash", -cost, reason, mask)

    def mark_options(self, ctx: StrategyContext, k: int) -> None:
        st = ctx.state
        if st.book is None:
            return
        snap = self.snapshot(k)
        mark = mark_book(st.book, self.vol_sources, snap, self.grid.steps_per_year, self.cfg.implied_vol.skew_mode, self.cfg.pricing)
        st.option_mv = mark.value_mid
        st.option_mv_bid = mark.value_bid
        st.dollar_delta = mark.dollar_delta
        st.dollar_delta_smile = mark.dollar_delta_smile
        st.dollar_gamma = mark.dollar_gamma
        st.vega_1pt = mark.vega_1pt
        st.theta_year = mark.theta_year
        st.rho_100bp = mark.rho_100bp
        st.dq_100bp = mark.dq_100bp
        st.sigma_mid = mark.sigma_mid
        st.residual_T = mark.residual_T

    def equity_dollar_delta(self, ctx: StrategyContext, k: int) -> F64:
        """Total equity dollar delta: options (chosen definition) + deployed spot + held + futures."""
        st = ctx.state
        S = self.paths.S[:, k, :]
        opt = st.dollar_delta if self.cfg.options.delta_definition == "bsm" else st.dollar_delta_smile
        out: F64 = opt + (st.spot_units * S).sum(axis=1) + st.held_mv * self._beta_held + (st.futures_units * S).sum(axis=1)
        return out

    # ------------------------------------------------------------------ margin
    def lending_value_now(self, ctx: StrategyContext, k: int) -> F64:
        st = ctx.state
        h = ltv_multiplier(self.cfg.leverage, self.paths.iv_short[:, k], self.held_dd[:, k])
        # long options never enter the lending value: only held/spot equities, illiquids and cash count
        return lending_value(self.cfg.leverage, st.held_mv + st.spot_mv, st.illiquid_mv, st.cash, h)

    def spending(self, ctx: StrategyContext, k: int) -> None:
        sp = self.cfg.spending
        if k == 0 or (sp.pct_nav_pa == 0 and sp.fixed_usd_pa == 0):
            return
        st = ctx.state
        amt = (sp.pct_nav_pa * np.maximum(st.nav, 0.0) + sp.fixed_usd_pa) / 12.0 * st.alive
        st.cash -= amt
        ctx.add_component(k, "spending", -amt)
        ctx.ledger.post(k, "cash", -amt, "family-office spending")

    def record_shortfall(self, ctx: StrategyContext, k: int) -> None:
        st = ctx.state
        short = (st.cash < -1e-6) & st.alive
        ctx.events.liquidity_shortfalls[short] += 1

    # ------------------------------------------------------------------ run
    def run(self, strategies: list[Strategy] | None = None, progress: Callable[[int, int], None] | None = None) -> dict[str, StrategyResult]:
        cfg = self.cfg
        if strategies is None:
            strategies = [LeveredStrategy(name="A"), CallReplacementStrategy(name="B"), UnleveredStrategy()]
            if cfg.leverage.strategy_d.enabled:
                strategies.append(LeveredStrategy(cfg.leverage.strategy_d.cash_buffer_pct_nav, name="D"))
        n_ladder = self._ladder_slots()
        extra = cfg.options.extra_slots
        if cfg.options.ladder_mode == "fixed_notional_schedule":
            steps_per_purchase = 1 if cfg.options.purchase_frequency == "weekly" else self.grid.steps_per_year // 12
            # upper bound on concurrently alive tranches: one purchase per purchase step over the option's life
            # (partial, cash-constrained purchases can exceed target / notional_per_purchase)
            max_alive = int(np.ceil(cfg.options.tenor_years * self.grid.steps_per_year / max(steps_per_purchase, 1))) + 2
            if cfg.dry_powder.enabled and cfg.dry_powder.instrument == "call_tranches":
                max_alive += 12 * len(cfg.dry_powder.tiers) * int(np.ceil(cfg.options.tenor_years))  # room for episode purchases living a full tenor
            extra = max(extra, max_alive)
        self._n_logical_slots = n_ladder + extra
        ctxs: list[StrategyContext] = []
        for s in strategies:
            st = PortfolioState(self.P, self.n_idx)
            if s.has_option_book:
                st.book = OptionBook(self.P, self._n_logical_slots * self.n_idx, n_ladder, n_logical=self._n_logical_slots)
                # extra-slot search only over the first index block; siblings follow via _slot_for
                st.book.n_ladder = n_ladder
            ctx = StrategyContext(s, st, StepRecorder(self.P, self.N), EventCounters.zeros(self.P), Ledger(s.name, self.ledger_paths))
            ctxs.append(ctx)
        t0 = time.time()
        for ctx in ctxs:
            self._initialise(ctx)
        for k in range(self.N):
            for ctx in ctxs:
                self._step(ctx, k)
            if progress is not None:
                progress(k + 1, self.N)
        self.timings["run_seconds"] = time.time() - t0
        results: dict[str, StrategyResult] = {}
        for ctx in ctxs:
            results[ctx.strategy.name] = StrategyResult(
                name=ctx.strategy.name, recorder=ctx.recorder, events=ctx.events, ledger=ctx.ledger,
                transition_cost=ctx.transition_cost, inception=dict(ctx.inception), meta={"uses_margin": ctx.strategy.uses_margin},
            )
        return results

    def _ladder_slots(self) -> int:
        from fosim.instruments.option_ladder import LadderSchedule

        return LadderSchedule.from_config(self.cfg.options).n_slots

    def _initialise(self, ctx: StrategyContext) -> None:
        st = ctx.state
        st.illiquid_mv = self._illiquid_official(0)
        ctx.strategy.initialise(self, ctx)
        if ctx.strategy.uses_margin and (st.cash < 0).any():
            draw = -np.minimum(st.cash, 0.0)
            st.loan += draw
            st.cash += draw
            ctx.ledger.post(0, "loan", draw, "inception: facility draw", draw > 0)
            ctx.ledger.post(0, "cash", draw, "inception: facility draw", draw > 0)
        st.loan_rate = self.loan_terms.rate_at(self.loan_base(0), st.loan, 0.0)
        st.held_mv = st.held_units * self.paths.held[:, 0]
        st.swap_mtm = self.loan_terms.swap_mtm(st.loan, self._zero_rate_fn(0), 0.0)
        st.nav = st.total_nav()
        # t0 is treated as a month-end for strategy set-up (the ladder starts at month 0)
        ctx.strategy.each_step(self, ctx, 0)
        ctx.strategy.month_end(self, ctx, 0)
        self.mark_options(ctx, 0)
        st.held_mv = st.held_units * self.paths.held[:, 0]
        st.spot_mv = (st.spot_units * self.paths.S[:, 0, :]).sum(axis=1)
        st.nav = st.total_nav()
        self._record(ctx, 0)
        if ctx.strategy.uses_margin:
            lv0 = self.lending_value_now(ctx, 0)
            ctx.recorder.series["utilisation"][:, 0] = utilisation(st.loan, lv0)
            ctx.recorder.series["lending_value"][:, 0] = lv0
        ctx.inception["nav0"] = float(st.nav.mean())
        ctx.inception["cash0"] = float(st.cash.mean())
        ctx.inception["option_premium0"] = float((st.book.cost.sum(axis=1)).mean()) if st.book is not None else 0.0
        ctx.inception["dollar_delta0"] = float(self.equity_dollar_delta(ctx, 0).mean())

    def _illiquid_official(self, k: int) -> F64:
        il = self.paths.illiquids
        arr = il.nav_reported if self.cfg.illiquids_marking.margin_uses == "reported" else il.nav_true
        out: F64 = arr[:, k, :].sum(axis=1)
        return out

    def _step(self, ctx: StrategyContext, k: int) -> None:
        st = ctx.state
        k1 = k + 1
        grid = self.grid
        nav_start = st.nav.copy()
        alive = st.alive
        # 3. interest on start-of-step balances at rates fixed at k
        loan_i = self.loan_terms.interest(st.loan, st.loan_rate, grid.days) * alive
        lev_cfg = self.cfg.leverage
        if lev_cfg.interest == "pay_cash":
            st.cash -= loan_i
            ctx.ledger.post(k1, "cash", -loan_i, "loan interest paid")
        elif lev_cfg.capitalisation_frequency == "step":
            st.loan += loan_i
            ctx.ledger.post(k1, "loan", loan_i, "loan interest capitalised")
        else:
            # monthly roll: accrue on the balance fixed at the last roll, add to the principal at month-end
            st.accrued_interest += loan_i
            ctx.ledger.post(k1, "accrued_interest", loan_i, "loan interest accrued (added to the loan at the monthly roll)")
            if grid.is_month_end[k1]:
                ctx.ledger.post(k1, "loan", st.accrued_interest, "monthly roll: accrued interest capitalised into the loan")
                st.loan += st.accrued_interest
                st.accrued_interest[:] = 0.0
        ctx.add_component(k1, "loan_interest", -loan_i)
        cash_rate = self.cash_yield(k)
        ci = cash_interest(st.cash, cash_rate + self.cfg.rates.cash.spread, self.cfg.rates.cash.spread, grid.days, self.cfg.leverage.day_count, st.loan_rate) * alive
        st.cash += ci
        ctx.add_component(k1, "cash_interest", ci)
        ctx.ledger.post(k1, "cash", ci, "cash interest")
        if ctx.strategy.uses_margin and self.cfg.loan_terms.commitment_fee_bps > 0:
            fee = self.loan_terms.commitment_fee(st.loan, grid.days) * alive
            st.cash -= fee
            ctx.add_component(k1, "fees", -fee)
            ctx.ledger.post(k1, "cash", -fee, "commitment fee on undrawn facility")
        # 4. dividends on start-of-step holdings
        div = np.zeros(self.P)
        if st.held_units.any():
            div += self._held_dividends(st.held_units, k1)
        for i, p in enumerate(self.paths.index_params):
            if st.spot_units[:, i].any():
                qc = p.cash_dividend_q_c * self._div_stress_mult(i, k)
                div += np.asarray(dividend_cash(st.spot_units[:, i], self.paths.S[:, k1, i], qc, grid.dt, p.wht))
        div *= alive
        st.cash += div
        ctx.add_component(k1, "dividends", div)
        ctx.ledger.post(k1, "cash", div, "dividends net of withholding tax")
        # 5. illiquids: official NAV update, calls and distributions
        il = self.paths.illiquids
        C = il.contributions[:, k1, :].sum(axis=1) * alive
        D = il.distributions[:, k1, :].sum(axis=1) * alive
        new_ill = np.where(alive, self._illiquid_official(k1), st.illiquid_mv)
        ctx.add_component(k1, "illiquid_mtm", new_ill - st.illiquid_mv - C + D)
        st.illiquid_mv = new_ill
        st.cash += D - C
        ctx.ledger.post(k1, "cash", -C, "capital calls (contributions to closed-end funds)")
        ctx.ledger.post(k1, "cash", D, "distributions from closed-end funds")
        # 6. option expiries settled to cash
        option_mv_before = st.option_mv.copy()
        if st.book is not None:
            self._settle_expiries(ctx, k1)
        # 7. mark to market at k+1
        old_held, old_spot = st.held_mv.copy(), st.spot_mv.copy()
        st.held_mv = st.held_units * self.paths.held[:, k1]
        st.spot_mv = (st.spot_units * self.paths.S[:, k1, :]).sum(axis=1)
        ctx.add_component(k1, "equity_mtm", st.held_mv - old_held)
        ctx.add_component(k1, "spot_mtm", st.spot_mv - old_spot)
        self.mark_options(ctx, k1)
        settled = self._settled.get(ctx.strategy.name, np.zeros(self.P)) if st.book is not None else np.zeros(self.P)
        ctx.add_component(k1, "option_mtm", st.option_mv - option_mv_before + settled)
        # futures variation margin
        if np.any(st.futures_units != 0):
            vm = (st.futures_units * (self.paths.S[:, k1, :] - self.paths.S[:, k, :])).sum(axis=1)
            st.cash += vm
            ctx.add_component(k1, "futures_pnl", vm)
            ctx.ledger.post(k1, "cash", vm, "futures overlay variation margin")
            im = self.cfg.exposure.futures_overlay.initial_margin_pct * np.abs(st.futures_units * self.paths.S[:, k1, :]).sum(axis=1)
            ctx.events.futures_margin_calls[(st.cash < im) & alive] += 1
        # swap MTM
        if self.cfg.loan_terms.rate_type == "floating_swapped":
            new_swap = self.loan_terms.swap_mtm(st.loan, self._zero_rate_fn(k1), float(grid.t[k1]))
            ctx.add_component(k1, "swap_mtm", new_swap - st.swap_mtm)
            st.swap_mtm = new_swap
        # counterparty default (options)
        if self._cp_rng is not None and st.book is not None:
            cp = self.cfg.options.counterparty
            u = self._cp_rng.random(self.P)
            default = (u < 1 - np.exp(-cp.default_intensity * grid.dt)) & alive
            if default.any():
                loss = st.option_mv * (1 - cp.recovery) * default
                st.cash -= loss  # novation cost to re-establish the book with a new dealer
                ctx.add_component(k1, "counterparty_loss", -loss)
                ctx.events.counterparty_losses[default] += 1
                ctx.ledger.post(k1, "cash", -loss, "counterparty default: novation cost", default)
        st.nav = st.total_nav()
        # 8. liquidity check
        if (st.cash < -1e-9).any():
            ctx.strategy.liquidity_waterfall(self, ctx, k1)
            st.nav = st.total_nav()
        # 9. margin
        if ctx.strategy.uses_margin:
            self._margin(ctx, k1)
            st.nav = st.total_nav()
        # ruin check
        self._ruin(ctx, k1)
        # every-step strategy actions (weekly option purchases): re-mark and refresh NAV if anything traded
        purchases_before = ctx.events.option_purchases.sum()
        ctx.strategy.each_step(self, ctx, k1)
        if ctx.events.option_purchases.sum() != purchases_before:
            self.mark_options(ctx, k1)
            st.nav = st.total_nav()
        # 10. month-end actions
        if grid.is_month_end[k1]:
            if self.loan_terms.is_reset_step(int(grid.month_index[k1]), True):
                st.loan_rate = self.loan_terms.rate_at(self.loan_base(k1), st.loan, float(grid.t[k1]))
            ctx.strategy.month_end(self, ctx, k1)
            if st.book is not None:
                self.mark_options(ctx, k1)
            st.nav = st.total_nav()
        # 11. record
        self._record(ctx, k1, cash_rate=cash_rate)
        # 12. accounting identity
        comp_sum = ctx.recorder.components[:, k1, :].sum(axis=1)
        diff = st.nav - nav_start - comp_sum
        bad = np.abs(diff) > IDENTITY_TOL * np.maximum(1.0, np.abs(nav_start) / 1e9)
        if bad.any():
            pi = int(np.flatnonzero(bad)[0])
            comps = {name: float(ctx.recorder.components[pi, k1, ci]) for ci, name in enumerate(COMP_IDX)}
            raise AccountingIdentityError(
                f"strategy {ctx.strategy.name} path {pi} step {k1}: NAV {nav_start[pi]:.6f} -> {st.nav[pi]:.6f}, "
                f"components sum {comp_sum[pi]:.6f}, residual {diff[pi]:.3e}; components={comps}; "
                f"cash={st.cash[pi]:.4f} loan={st.loan[pi]:.4f} held={st.held_mv[pi]:.4f} spot={st.spot_mv[pi]:.4f} "
                f"options={st.option_mv[pi]:.4f} illiquid={st.illiquid_mv[pi]:.4f}"
            )

    # per-strategy scratch for settled payoffs within a step (populated in __init__)

    def _settle_expiries(self, ctx: StrategyContext, k1: int) -> None:
        st = ctx.state
        book = st.book
        assert book is not None
        exp = book.active & (book.expiry_step == k1)
        settled = np.zeros(self.P)
        if exp.any():
            S = np.take_along_axis(self.paths.S[:, k1, :], book.index_id, axis=1)
            pay = np.where(exp, book.units * np.maximum(S - book.strike, 0.0), 0.0).sum(axis=1)
            st.cash += pay
            settled = pay
            ctx.ledger.post(k1, "cash", pay, "option expiry: cash-settled payoff", pay > 0)
            for j in range(book.n_slots):
                m = exp[:, j]
                if m.any():
                    book.clear(np.flatnonzero(m), j)
        self._settled[ctx.strategy.name] = settled

    def _held_dividends(self, units: F64, k1: int) -> F64:
        """Dividends on the held portfolio: sleeve-weighted realised dividend yields of the price-return indices."""
        cfg = self.cfg
        out = np.zeros(self.P)
        H = self.paths.held[:, k1]
        dv = self.paths.series.get("div_yield")
        for i, p in enumerate(self.paths.index_params):
            w = cfg.equity_indices[i].weight_in_equity_sleeve
            if w == 0.0:
                continue
            base_q = dv[:, k1 - 1] if (dv is not None and p.underlying_type == "price_return") else p.cash_dividend_q_c
            qc = base_q * self._div_stress_mult(i, k1 - 1)
            out += np.asarray(dividend_cash(units * w, H, qc, self.grid.dt, p.wht))
        return out

    def _div_stress_mult(self, i: int, k: int) -> F64:
        cut = self.cfg.equity_indices[i].dividend_stress_cut
        if cut == 0.0:
            return np.ones(self.P)
        out: F64 = np.where(self.book_dd[:, k] <= self.cfg.equity_indices[i].dividend_stress_trigger, 1.0 - cut, 1.0)
        return out

    def _margin(self, ctx: StrategyContext, k1: int) -> None:
        cfg = self.cfg.leverage
        th = cfg.thresholds
        st = ctx.state
        ev = ctx.events
        lv = self.lending_value_now(ctx, k1)
        u = utilisation(st.loan + st.accrued_interest, lv)
        alive = st.alive & (st.loan > 0)
        ev.min_headroom = np.minimum(ev.min_headroom, np.where(alive, 1.0 - u, np.inf))
        ctx.recorder.series["utilisation"][:, k1] = u
        ctx.recorder.series["lending_value"][:, k1] = lv
        warn = alive & (u >= th.warn) & (u < th.call)
        ev.margin_warnings[warn] += 1
        call = alive & (u >= th.call)
        if not call.any():
            st.grace_clock[:] = 0
            return
        ev.margin_calls[call] += 1
        ctx.ledger.post(k1, "event", np.where(call, u, 0.0), "MARGIN CALL (utilisation)", call)
        # cure 1: cash. Repay R so that (L − R)/(LV − ℓ_cash R) = u*  → R = (L − u* LV)/(1 − u* ℓ_cash)
        lc = cfg.ltv_base.cash
        R = np.where(call, np.maximum((st.loan - th.target_after_cure * lv) / (1.0 - th.target_after_cure * lc), 0.0), 0.0)
        R = np.minimum(R, np.maximum(st.cash, 0.0))
        st.loan -= R
        st.cash -= R
        ctx.ledger.post(k1, "loan", -R, "margin cure: repay from cash", R > 0)
        ctx.ledger.post(k1, "cash", -R, "margin cure: repay from cash", R > 0)
        lv = self.lending_value_now(ctx, k1)
        u = utilisation(st.loan + st.accrued_interest, lv)
        still = alive & (u >= th.call)
        closeout = alive & (u >= th.closeout)
        grace_steps = int(np.floor(cfg.cure_grace_days / self.grid.days + 1e-9))
        st.grace_clock = np.where(still, st.grace_clock + 1, 0)
        expired = still & (st.grace_clock > grace_steps)
        bank = closeout | expired
        investor = still & ~bank & (grace_steps == 0)
        base_s = cfg.slippage_bps * 1e-4
        for mask, s, label in ((investor, base_s, "cure sale"), (bank, base_s * cfg.slippage_stress_multiplier, "FORCED LIQUIDATION")):
            if not mask.any():
                continue
            ltv_e = cfg.ltv_base.equity_index * ltv_multiplier(cfg, self.paths.iv_short[:, k1], self.held_dd[:, k1])
            x = cure_sale_size(st.loan, lv, th.target_after_cure, ltv_e, s)
            x = np.where(mask, np.minimum(x, st.held_mv + st.spot_mv), 0.0)
            # sell held first, then deployed spot
            H = self.paths.held[:, k1]
            x_held = np.minimum(x, st.held_mv)
            st.held_units -= np.where(H > 0, x_held / H, 0.0)
            st.held_mv = st.held_units * H
            x_spot = x - x_held
            if x_spot.any():
                S = self.paths.S[:, k1, :]
                with np.errstate(divide="ignore", invalid="ignore"):
                    fr = np.where(st.spot_mv > 0, x_spot / np.maximum(st.spot_mv, 1e-300), 0.0)
                st.spot_units *= 1.0 - fr[:, None]
                st.spot_mv = (st.spot_units * S).sum(axis=1)
            proceeds = x * (1.0 - s)
            st.loan -= proceeds
            ctx.add_component(k1, "slippage", -(x * s))
            ev.slippage_loss += x * s
            if label == "cure sale":
                ev.cure_sales_volume += x
            else:
                ev.forced_liquidations[mask] += 1
                ev.forced_sale_volume += x
            ctx.ledger.post(k1, "loan", -proceeds, f"margin {label}: sale proceeds repay loan", mask)
            ctx.ledger.post(k1, "memo:slippage", -(x * s), f"margin {label}: slippage loss on assets sold (non-cash memo)", mask)
            ctx.ledger.post(k1, "memo:asset_sale", -x, f"margin {label}: assets sold at market value (non-cash memo)", mask)
            lv = self.lending_value_now(ctx, k1)
            u = utilisation(st.loan + st.accrued_interest, lv)
            st.grace_clock[mask] = 0
        ctx.recorder.series["utilisation"][:, k1] = u
        ctx.recorder.series["lending_value"][:, k1] = lv
        # a loan left uncovered after selling everything: cash cannot cover -> handled by ruin check

    def _ruin(self, ctx: StrategyContext, k1: int) -> None:
        st = ctx.state
        nav = st.total_nav()
        new_ruin = st.alive & (nav <= 0.0)
        if not new_ruin.any():
            return
        ev = ctx.events
        ev.ruin[new_ruin] = True
        ev.ruin_step[new_ruin] = k1
        # freeze: liquidate everything at the current marks, net against the loan; positions to zero
        st.cash[new_ruin] = nav[new_ruin]
        st.loan[new_ruin] = 0.0
        st.accrued_interest[new_ruin] = 0.0
        st.held_units[new_ruin] = 0.0
        st.held_mv[new_ruin] = 0.0
        st.spot_units[new_ruin] = 0.0
        st.spot_mv[new_ruin] = 0.0
        st.option_mv[new_ruin] = 0.0
        st.option_mv_bid[new_ruin] = 0.0
        st.illiquid_mv[new_ruin] = 0.0
        st.swap_mtm[new_ruin] = 0.0
        st.futures_units[new_ruin] = 0.0
        if st.book is not None:
            st.book.active[new_ruin, :] = False
            st.book.units[new_ruin, :] = 0.0
        st.alive[new_ruin] = False
        ctx.ledger.post(k1, "event", np.where(new_ruin, nav, 0.0), "RUIN: NAV <= 0, path frozen", new_ruin)

    def _record(self, ctx: StrategyContext, k: int, cash_rate: F64 | None = None) -> None:
        st = ctx.state
        r = ctx.recorder
        il = self.paths.illiquids
        ill_true = il.nav_true[:, k, :].sum(axis=1) * st.alive
        nav_true = st.nav - st.illiquid_mv + ill_true
        S = self.paths.S[:, k, :]
        opt_dd = st.dollar_delta if self.cfg.options.delta_definition == "bsm" else st.dollar_delta_smile
        exposure = opt_dd + st.spot_mv + st.held_mv * self._beta_held + (st.futures_units * S).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff_lev = np.where(st.nav > 0, (st.held_mv + st.spot_mv) / np.maximum(st.nav, 1e-300), np.nan)
        r.record(
            k, nav=st.nav, nav_true=nav_true, nav_bid=st.nav - st.option_mv + st.option_mv_bid, cash=st.cash, loan=st.loan, accrued_interest=st.accrued_interest,
            held_mv=st.held_mv, spot_mv=st.spot_mv, option_mv=st.option_mv, illiquid_mv=st.illiquid_mv, illiquid_true=ill_true,
            loan_rate=st.loan_rate, cash_rate=(cash_rate if cash_rate is not None else self.cash_yield(k)),
            option_rate_5y=self._zero_rate_fn(k)(self.cfg.options.tenor_years) + self.cfg.pricing.option_funding_spread,
            dollar_delta=st.dollar_delta, dollar_delta_smile=st.dollar_delta_smile, dollar_gamma=st.dollar_gamma, vega_1pt=st.vega_1pt,
            theta_year=st.theta_year, rho_100bp=st.rho_100bp, dq_100bp=st.dq_100bp,
            n_tranches=(st.book.active.sum(axis=1).astype(np.float64) if st.book is not None else np.zeros(self.P)),
            futures_notional=(st.futures_units * S).sum(axis=1), swap_mtm=st.swap_mtm, effective_leverage=eff_lev, equity_exposure=exposure,
            spot_cost_basis=st.spot_cost_basis,
        )
        if st.book is not None and self.ledger_paths:
            self._snapshot_tranches(ctx, k)

    def _snapshot_tranches(self, ctx: StrategyContext, k: int) -> None:
        st = ctx.state
        book = st.book
        assert book is not None
        from fosim.instruments.option_ladder import SOURCE_NAMES

        S = self.paths.S[:, k, :]
        for p in self.ledger_paths:
            for j in np.flatnonzero(book.active[p]):
                i = int(book.index_id[p, j])
                T = float(st.residual_T[p, j]) if st.residual_T is not None else float((book.expiry_step[p, j] - k) / self.grid.steps_per_year)
                sig = float(st.sigma_mid[p, j]) if st.sigma_mid is not None else np.nan
                ctx.ledger.tranche_snapshots.append(
                    {
                        "step": k, "path": p, "strategy": ctx.strategy.name, "slot": int(j), "index": self.cfg.index_names[i],
                        "origin": int(book.origin[p, j]), "units": float(book.units[p, j]), "strike": float(book.strike[p, j]),
                        "spot": float(S[p, i]), "moneyness_S_over_K": float(S[p, i] / book.strike[p, j]), "residual_years": T,
                        "vol_mid": sig, "vol_at_purchase_ask": float(book.vol_at_purchase[p, j]), "source": SOURCE_NAMES[int(book.source[p, j])],
                        "cost_usd": float(book.cost[p, j]), "notional_usd": float(book.notional[p, j]),
                        "purchase_step": int(book.purchase_step[p, j]), "expiry_step": int(book.expiry_step[p, j]),
                    }
                )
