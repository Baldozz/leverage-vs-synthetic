"""Vectorised option book and ladder schedule (SPEC §5.2).

Tranches live in fixed slots: arrays of shape (P, n_slots). Slot j of the ladder is (re)purchased on
a deterministic month schedule; extra slots serve exposure policies and dry-powder call purchases.
All pricing is Q-measure BSM with σ from the ``VolSource`` at the tranche's *current* log-moneyness
(sticky moneyness) or its purchase log-moneyness (sticky strike), r = zero rate for the residual
maturity (+ funding spread) and q = pricing dividend curve at the residual maturity.
Purchases at ask (σ + ½ spread_new), marks at mid, sales at bid (σ − ½ spread_unwind).
Long options carry no margin: this module never computes a lending value (asserted by the engine).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from fosim.config.schema import OptionsConfig, PricingConfig
from fosim.pricing.black_scholes import Greeks, bsm_greeks, bsm_price, forward_price
from fosim.pricing.vol_surface import SourceLabel, VolSource

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
B_ = NDArray[np.bool_]

SOURCE_CODES: dict[SourceLabel, int] = {
    "parametric": 0,
    "dealer_quote": 1,
    "parametric_quote_calibrated": 2,
    "surface_file": 3,
    "scenario_override": 4,
}
SOURCE_NAMES = {v: k for k, v in SOURCE_CODES.items()}


@dataclass
class OptionBook:
    """State of all tranches for one strategy, vectorised over paths."""

    n_paths: int
    n_slots: int
    n_ladder: int
    n_logical: int = 0  # slots per index block (extra-slot search is confined to the first block); 0 → n_slots
    units: F64 = field(init=False)
    strike: F64 = field(init=False)
    expiry_step: I64 = field(init=False)
    purchase_step: I64 = field(init=False)
    index_id: I64 = field(init=False)
    k_ref: F64 = field(init=False)  # log-moneyness at purchase (sticky strike)
    cost: F64 = field(init=False)  # premium paid (USD) — cost basis
    notional: F64 = field(init=False)
    vol_at_purchase: F64 = field(init=False)
    source: I64 = field(init=False)
    active: B_ = field(init=False)
    origin: I64 = field(init=False)  # 0 ladder, 1 rebalance, 2 dry powder, 3 restrike, 4 exit-rule

    def __post_init__(self) -> None:
        P, n = self.n_paths, self.n_slots
        self.units = np.zeros((P, n))
        self.strike = np.ones((P, n))
        self.expiry_step = np.full((P, n), -1, dtype=np.int64)
        self.purchase_step = np.full((P, n), -1, dtype=np.int64)
        self.index_id = np.zeros((P, n), dtype=np.int64)
        self.k_ref = np.zeros((P, n))
        self.cost = np.zeros((P, n))
        self.notional = np.zeros((P, n))
        self.vol_at_purchase = np.zeros((P, n))
        self.source = np.zeros((P, n), dtype=np.int64)
        self.active = np.zeros((P, n), dtype=bool)
        self.origin = np.zeros((P, n), dtype=np.int64)

    # ----------------------------------------------------------------------- helpers
    def free_extra_slot(self, mask: B_) -> I64:
        """For each path in ``mask`` return an inactive extra slot index (−1 if none)."""
        out = np.full(self.n_paths, -1, dtype=np.int64)
        stop = self.n_logical if self.n_logical > 0 else self.n_slots
        extra_free = ~self.active[:, self.n_ladder : stop]
        has = extra_free.any(axis=1)
        first = np.argmax(extra_free, axis=1) + self.n_ladder
        out[mask & has] = first[mask & has]
        return out

    def clear(self, paths: NDArray[np.intp] | B_, slot: int | I64) -> None:
        self.active[paths, slot] = False
        self.units[paths, slot] = 0.0
        self.cost[paths, slot] = 0.0
        self.notional[paths, slot] = 0.0
        self.expiry_step[paths, slot] = -1


@dataclass(frozen=True)
class MarketSnapshot:
    """Per-path market state at a step needed to price the book."""

    k: int
    t: float
    month: int
    S: F64  # (P, n_idx)
    iv_short: F64  # (P,)
    zero_rate: Callable[[float], F64]  # tenor -> (P,)
    q_curve: list[Callable[[float], float]]  # per index: tenor -> q
    funding_spread: float
    stressed: B_  # (P,) bid/ask stress flag
    q_equals_r: tuple[bool, ...] = ()  # per index: excess-return underlying priced with q = r (flat forward)


@dataclass(frozen=True)
class BookMark:
    value_mid: F64  # (P,)
    value_bid: F64
    greeks: Greeks  # per slot (P, n_slots), zero where inactive
    dollar_delta: F64  # (P,) BSM definition
    dollar_delta_smile: F64
    vega_1pt: F64
    theta_year: F64
    rho_100bp: F64
    dq_100bp: F64
    dollar_gamma: F64
    sigma_mid: F64  # (P, n_slots)
    residual_T: F64  # (P, n_slots)


def _slot_inputs(book: OptionBook, snap: MarketSnapshot, steps_per_year: int) -> tuple[F64, F64, F64, F64, F64]:
    """(S, K, r, q, T) arrays (P, n_slots) for active slots (T = 0, benign values elsewhere)."""
    P, n = book.units.shape
    T = np.where(book.active, (book.expiry_step - snap.k) / steps_per_year, 0.0).astype(np.float64)
    T = np.maximum(T, 0.0)
    S = np.take_along_axis(snap.S, book.index_id, axis=1)
    K = np.where(book.active, book.strike, S)
    r = np.zeros((P, n))
    q = np.zeros((P, n))
    # rates and dividends depend on (path, residual tenor); residual tenor is slot-specific but shared
    # across paths for ladder slots (deterministic schedule); handle generally per unique tenor
    for j in range(n):
        if not book.active[:, j].any():
            continue
        Tj = T[:, j]
        uniq = np.unique(np.round(Tj, 10))
        for tv in uniq:
            m = np.round(Tj, 10) == tv
            r[m, j] = snap.zero_rate(float(tv))[m] + snap.funding_spread
        for i in np.unique(book.index_id[:, j]):
            mi = book.index_id[:, j] == i
            for tv in np.unique(np.round(Tj[mi], 10)):
                mm = mi & (np.round(Tj, 10) == tv)
                q[mm, j] = snap.q_curve[int(i)](float(tv))
            if snap.q_equals_r and snap.q_equals_r[int(i)]:
                q[mi, j] = r[mi, j]
    return S, K, r, q, T


def slot_vols(
    book: OptionBook, vs: list[VolSource], snap: MarketSnapshot, S: F64, K: F64, r: F64, q: F64, T: F64, sticky: str
) -> tuple[F64, F64]:
    """(σ_mid, ψ(T)) per slot from each index's vol source."""
    P, n = S.shape
    sig = np.zeros((P, n))
    psi = np.zeros((P, n))
    F = forward_price(S, r, q, T)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_now = np.where(T > 0, np.log(K / F), 0.0)
    k_use = k_now if sticky == "sticky_moneyness" else book.k_ref
    for j in range(n):
        if not book.active[:, j].any():
            continue
        for i in np.unique(book.index_id[:, j]):
            mi = (book.index_id[:, j] == i) & book.active[:, j]
            if not mi.any():
                continue
            for tv in np.unique(np.round(T[mi, j], 10)):
                mm = mi & (np.round(T[:, j], 10) == tv)
                if tv <= 0:
                    continue
                s, _src = vs[int(i)].vol(k_use[mm, j], float(tv), snap.iv_short[mm], snap.month)
                sig[mm, j] = s
                psi[mm, j] = vs[int(i)].parametric.psi_T(float(tv))
    return sig, psi


def mark_book(
    book: OptionBook, vs: list[VolSource], snap: MarketSnapshot, steps_per_year: int, sticky: str, pricing: PricingConfig
) -> BookMark:
    S, K, r, q, T = _slot_inputs(book, snap, steps_per_year)
    sig, psi = slot_vols(book, vs, snap, S, K, r, q, T, sticky)
    g = bsm_greeks(S, K, r, q, sig, T, "call")
    act = book.active.astype(np.float64)
    u = book.units * act
    half = np.zeros_like(sig)
    for i in range(len(vs)):
        mi = book.index_id == i
        if mi.any():
            hs = vs[i].half_spread(0.0, None, snap.iv_short, "unwind")  # (P,) stress-aware base
            half[mi] = np.broadcast_to(hs[:, None], half.shape)[mi]
    price_bid = bsm_price(S, K, r, q, np.maximum(sig - half, 1e-8), T, "call")
    value_mid = (g.price * u).sum(axis=1)
    value_bid = (price_bid * u).sum(axis=1)
    dd = (g.delta * u * S).sum(axis=1)
    dd_smile = ((g.delta + g.vega * (-psi / S)) * u * S).sum(axis=1) if sticky == "sticky_moneyness" else dd
    return BookMark(
        value_mid=value_mid, value_bid=value_bid, greeks=g, dollar_delta=dd, dollar_delta_smile=dd_smile,
        vega_1pt=(g.vega * u).sum(axis=1) * 0.01, theta_year=(g.theta * u).sum(axis=1),
        rho_100bp=(g.rho * u).sum(axis=1) * 0.01, dq_100bp=(g.dq * u).sum(axis=1) * 0.01,
        dollar_gamma=(g.gamma * u * S * S).sum(axis=1) * 0.01, sigma_mid=sig, residual_T=T,
    )


# ----------------------------------------------------------------------------- new-tranche pricing


@dataclass(frozen=True)
class NewTrancheQuote:
    strike: F64  # (P,)
    sigma_mid: F64
    sigma_ask: F64
    price_ask: F64  # per unit of index
    price_mid: F64
    delta: F64  # per chosen definition
    delta_bsm: F64
    k_ref: F64
    source: int
    r: F64
    q: F64 | float
    T: float


def solve_delta_strike_vectorised(
    S: F64, r: F64, q: F64 | float, T: float, target: float, vol_fn: Callable[[F64], F64], psi_T: float, definition: str, n_iter: int = 80
) -> F64:
    """Bisection in K (delta is monotone decreasing in K) with σ(K) re-evaluated each iteration."""
    lo, hi = 0.2 * S, 8.0 * S

    def delta_at(K: F64) -> F64:
        s = vol_fn(K)
        g = bsm_greeks(S, K, r, q, s, T, "call")
        d = g.delta
        if definition == "smile":
            d = d + g.vega * (-psi_T / S)
        return np.asarray(d, dtype=np.float64)

    d_lo, d_hi = delta_at(lo), delta_at(hi)
    if np.any(d_lo < target) or np.any(d_hi > target):
        raise ValueError(f"delta target {target} not bracketed on [0.2 S, 8 S]")
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        d = delta_at(mid)
        hi = np.where(d < target, mid, hi)
        lo = np.where(d >= target, mid, lo)
    return 0.5 * (lo + hi)


def quote_new_tranche(
    opt: OptionsConfig, vs: VolSource, snap: MarketSnapshot, index: int, T: float | None = None, force_atm: bool = False
) -> NewTrancheQuote:
    """Strike per ``options.strike_mode`` and ask/mid prices for a fresh tranche on ``index``."""
    T0 = opt.tenor_years if T is None else T
    S = snap.S[:, index]
    r = snap.zero_rate(T0) + snap.funding_spread
    q: F64 | float = r if (snap.q_equals_r and snap.q_equals_r[index]) else snap.q_curve[index](T0)
    F = forward_price(S, r, q, T0)
    psi = float(vs.parametric.psi_T(T0))
    mode = "atm_spot" if force_atm else opt.strike_mode
    if mode == "atm_spot":
        K = S.copy()
    elif mode == "atm_forward":
        K = F.copy()
    elif mode == "pct_spot":
        assert opt.strike_param is not None
        K = S * opt.strike_param
    elif mode == "pct_forward":
        assert opt.strike_param is not None
        K = F * opt.strike_param
    elif mode == "delta_target":
        assert opt.strike_param is not None

        def vol_fn(Kx: F64) -> F64:
            s, _ = vs.vol(np.log(Kx / F), T0, snap.iv_short, snap.month)
            return s

        K = solve_delta_strike_vectorised(S, r, q, T0, opt.strike_param, vol_fn, psi, opt.delta_definition)
    else:
        raise ValueError(mode)
    k_ref = np.log(K / F)
    prem_fixed = vs.premium_override(T0, snap.month)
    if prem_fixed is not None:
        # fixed premium in % of notional: back out the vol of each (K/S, r, q) combination at today's rate
        from fosim.pricing.implied_vol import implied_vol

        sig_paid = np.empty_like(S)
        r_arr = np.broadcast_to(np.asarray(r, dtype=np.float64), S.shape)
        q_arr = np.broadcast_to(np.asarray(q, dtype=np.float64), S.shape)
        keys = np.round(np.stack([K / S, r_arr, q_arr], axis=1), 10)
        for row in np.unique(keys, axis=0):
            m = np.all(keys == row, axis=1)
            sig_paid[m] = implied_vol(prem_fixed * 100.0, 100.0, 100.0 * row[0], row[1], row[2], T0, "call")
        # the book is marked at the surface vol; the gap between the price paid and the mark is booked as a cost
        sig, _ = vs.vol(k_ref, T0, snap.iv_short, snap.month)
        g_mid = bsm_greeks(S, K, r, q, sig, T0, "call")
        d_bsm = np.asarray(g_mid.delta, dtype=np.float64)
        d_use = d_bsm + g_mid.vega * (-psi / S) if opt.delta_definition == "smile" else d_bsm
        return NewTrancheQuote(
            strike=K, sigma_mid=np.asarray(sig), sigma_ask=np.asarray(sig_paid), price_ask=np.asarray(prem_fixed * S), price_mid=np.asarray(g_mid.price),
            delta=np.asarray(d_use), delta_bsm=d_bsm, k_ref=k_ref, source=SOURCE_CODES["scenario_override"], r=np.asarray(r), q=q, T=T0,
        )
    if np.allclose(k_ref, k_ref[0]):
        # source lookup is scalar in k for dealer quotes: evaluate per unique k when all paths share it
        sig, src = vs.vol(float(k_ref[0]), T0, snap.iv_short, snap.month)
    else:
        sig, src = vs.vol(k_ref, T0, snap.iv_short, snap.month)
    half = vs.half_spread(T0, float(k_ref[0]) if np.allclose(k_ref, k_ref[0]) else None, snap.iv_short, "new")
    sig_ask = sig + half
    g_mid = bsm_greeks(S, K, r, q, sig, T0, "call")
    p_ask = bsm_price(S, K, r, q, sig_ask, T0, "call")
    d_bsm = np.asarray(g_mid.delta, dtype=np.float64)
    d_use = d_bsm + g_mid.vega * (-psi / S) if opt.delta_definition == "smile" else d_bsm
    return NewTrancheQuote(
        strike=K, sigma_mid=np.asarray(sig), sigma_ask=np.asarray(sig_ask), price_ask=np.asarray(p_ask),
        price_mid=np.asarray(g_mid.price), delta=np.asarray(d_use), delta_bsm=d_bsm, k_ref=k_ref,
        source=SOURCE_CODES[src], r=np.asarray(r), q=q, T=T0,
    )


# ----------------------------------------------------------------------------- ladder schedule


@dataclass(frozen=True)
class LadderSchedule:
    """Deterministic purchase months per ladder slot (mode-dependent), used by the engine and test 23."""

    mode: Literal["bullet", "monthly_buildup_hold_to_expiry", "rolling_ladder", "fixed_notional_schedule"]
    n_slots: int
    buildup_months: int
    hold_months: int
    tenor_months: int
    roll_residual_months: int

    @staticmethod
    def from_config(opt: OptionsConfig) -> LadderSchedule:
        tenor_m = round(opt.tenor_years * 12)
        if opt.ladder_mode == "fixed_notional_schedule":
            # the whole book lives in the extra-slot pool; n_slots here is only the ladder block (kept minimal)
            return LadderSchedule("fixed_notional_schedule", 1, 1, tenor_m, tenor_m, 0)
        if opt.ladder_mode == "bullet":
            return LadderSchedule("bullet", 1, 1, tenor_m, tenor_m, round(opt.bullet_roll_residual_years * 12))
        if opt.ladder_mode == "monthly_buildup_hold_to_expiry":
            return LadderSchedule("monthly_buildup_hold_to_expiry", opt.buildup_months, opt.buildup_months, tenor_m, tenor_m, 0)
        return LadderSchedule("rolling_ladder", opt.buildup_months, opt.buildup_months, opt.hold_months_before_roll, tenor_m, 0)

    def is_purchase_month(self, slot: int, month: int) -> bool:
        """Scheduled (re)purchase of ``slot`` at ``month`` (fresh tranche), independent of path state."""
        if month < slot or self.mode == "fixed_notional_schedule":
            return False
        if self.mode == "bullet":
            # buy at 0, then whenever the previous tranche is rolled: every (tenor − residual) months
            period = self.tenor_months - self.roll_residual_months
            return month % max(period, 1) == 0
        if self.mode == "monthly_buildup_hold_to_expiry":
            return (month - slot) % self.tenor_months == 0
        return (month - slot) % self.hold_months == 0

    def is_roll_sale_month(self, slot: int, month: int) -> bool:
        """The existing tranche in ``slot`` is sold at bid at ``month`` before the fresh purchase."""
        if month <= slot or self.mode == "fixed_notional_schedule":
            return False
        if self.mode == "bullet":
            period = self.tenor_months - self.roll_residual_months
            return self.roll_residual_months > 0 and month % max(period, 1) == 0
        if self.mode == "monthly_buildup_hold_to_expiry":
            return False  # tranches expire, never sold early
        return (month - slot) % self.hold_months == 0

    def hand_built_table(self, n_months: int) -> list[dict[str, int]]:
        rows = []
        for m in range(n_months + 1):
            for j in range(self.n_slots):
                if self.is_purchase_month(j, m):
                    rows.append({"month": m, "slot": j, "expiry_month": m + self.tenor_months, "sold_before": int(self.is_roll_sale_month(j, m))})
        return rows
