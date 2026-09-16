"""Immutable ``MarketPaths`` container (SPEC §4): strategy-independent simulated market state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fosim.engine.conventions import TimeGrid
from fosim.market.equity import IndexParams
from fosim.market.illiquids import IlliquidPaths
from fosim.market.rates import RateModel

F64 = NDArray[np.float64]


def _freeze(a: F64) -> F64:
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class MarketPaths:
    """All arrays are read-only; index axis 1 is time (0..n_steps).

    S:            (P, N+1, n_idx) option-index price levels
    held:         (P, N+1) held equity portfolio level (100 at t0)
    iv_short:     (P, N+1) short-dated ATM implied vol
    r_short:      (P, N+1) short rate (SOFR proxy)
    jump_counts:  (P, N) common jump counts in step k
    log_jumps:    (P, N, n_idx) log jump per index in step k
    illiquids:    precomputed illiquid NAVs and flows
    """

    grid: TimeGrid
    index_names: list[str]
    index_params: list[IndexParams]
    S: F64
    held: F64
    iv_short: F64
    r_short: F64
    jump_counts: F64
    log_jumps: F64
    illiquids: IlliquidPaths
    rate_model: RateModel
    book_weights: F64  # option-book weights per index
    seed: int
    correlation_repaired: bool = False
    meta: dict[str, object] = field(default_factory=dict)
    # optional observed series (P, N+1), decimals: "loan_base", "cash_yield", "option_rate", "div_yield" (continuous).
    # When present the engine uses them instead of the single short rate / configured dividend yield.
    series: dict[str, F64] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("S", "held", "iv_short", "r_short", "jump_counts", "log_jumps"):
            _freeze(getattr(self, name))
        for name in ("nav_true", "nav_reported", "contributions", "distributions", "unfunded"):
            _freeze(getattr(self.illiquids, name))

    @property
    def n_paths(self) -> int:
        return int(self.S.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.S.shape[1] - 1)

    @property
    def n_indices(self) -> int:
        return int(self.S.shape[2])

    def zero_rate(self, k: int, tenor: float | F64) -> F64:
        """Zero rate y_t(τ) per path at step k (uses r_short[:, k] for stochastic models)."""
        return self.rate_model.zero_rate(self.r_short[:, k], tenor)

    def book_index_level(self) -> F64:
        """Option-book-weighted geometric index level (P, N+1), normalised to 100 at t0."""
        w = self.book_weights
        lvl = np.exp(np.sum(np.log(self.S / self.S[:, :1, :]) * w, axis=2)) * 100.0
        out: F64 = np.asarray(lvl, dtype=np.float64)
        return out

    def book_drawdown(self) -> F64:
        from fosim.market.equity import running_drawdown

        return running_drawdown(self.book_index_level())

    def slice_paths(self, idx: NDArray[np.intp]) -> MarketPaths:
        """Sub-select paths (e.g. for a single-path audit)."""
        il = self.illiquids
        return MarketPaths(
            grid=self.grid, index_names=self.index_names, index_params=self.index_params,
            S=self.S[idx].copy(), held=self.held[idx].copy(), iv_short=self.iv_short[idx].copy(),
            r_short=self.r_short[idx].copy(), jump_counts=self.jump_counts[idx].copy(), log_jumps=self.log_jumps[idx].copy(),
            illiquids=IlliquidPaths(
                names=il.names, nav_true=il.nav_true[idx].copy(), nav_reported=il.nav_reported[idx].copy(),
                contributions=il.contributions[idx].copy(), distributions=il.distributions[idx].copy(),
                unfunded=il.unfunded[idx].copy(), true_return=il.true_return[idx].copy(), reported_return=il.reported_return[idx].copy(),
            ),
            rate_model=self.rate_model, book_weights=self.book_weights, seed=self.seed,
            correlation_repaired=self.correlation_repaired, meta=dict(self.meta), series={k: v[idx].copy() for k, v in self.series.items()},
        )
