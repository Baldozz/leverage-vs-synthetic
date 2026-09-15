"""Strategy interface. The simulator owns the step loop (§5.5); strategies implement the hooks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from fosim.engine.simulator import Simulator, StrategyContext

F64 = NDArray[np.float64]


class Strategy(Protocol):
    name: str
    uses_margin: bool
    has_option_book: bool

    def initialise(self, sim: Simulator, ctx: StrategyContext) -> None:
        """Set the inception balance sheet at k = 0 (including transition trades and fees)."""
        ...

    def liquidity_waterfall(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        """Raise cash for paths with cash < 0 at step k (§5.5 step 8)."""
        ...

    def month_end(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        """Scheduled actions at a month-end step k (§5.5 step 10)."""
        ...

    def each_step(self, sim: Simulator, ctx: StrategyContext, k: int) -> None:
        """Actions at every step (used for weekly option purchases); runs before month-end actions."""
        ...


def inception_balance_cfg(cfg: object) -> dict[str, float]:
    """Inception balance sheet from a SimConfig: the custom block when enabled, else the derived rule."""
    from fosim.config.schema import SimConfig

    assert isinstance(cfg, SimConfig)
    ci = cfg.custom_inception
    if ci.enabled:
        ref = ci.reference_equity_exposure if ci.reference_equity_exposure is not None else ci.A.equities
        return {
            "nav0": ci.A.equities + ci.A.cash + ci.illiquids - ci.A.loan, "loan": ci.A.loan, "equity_A": ci.A.equities, "cash_A": ci.A.cash,
            "illiquid": ci.illiquids, "equity_sleeve_B": ci.B.equities + ci.B.cash, "cash_B": ci.B.cash, "equity_B": ci.B.equities, "loan_B": ci.B.loan,
            "equity_C": ci.C.equities, "cash_C": ci.C.cash, "loan_C": ci.C.loan, "reference_equity_exposure": ref, "custom": 1.0,
        }
    b = inception_balance(cfg.portfolio.nav0, cfg.portfolio.weights.equities, cfg.portfolio.weights.illiquids, cfg.leverage.ratio_of_nav, cfg.leverage.allocation_of_borrowed_funds)
    b.update({"cash_A": 0.0, "cash_B": b["equity_sleeve_B"], "equity_B": 0.0, "loan_B": 0.0, "equity_C": b["equity_sleeve_B"], "cash_C": 0.0, "loan_C": 0.0, "reference_equity_exposure": b["equity_A"], "custom": 0.0})
    return b


def inception_balance(cfg_nav0: float, w_eq: float, w_ill: float, lev_ratio: float, allocation: str) -> dict[str, float]:
    """Inception balance sheet of A (SPEC §2): pro_rata or equity_only allocation of borrowed funds."""
    loan = lev_ratio * cfg_nav0
    if allocation == "pro_rata":
        gross = cfg_nav0 + loan
        equity, illiquid = gross * w_eq, gross * w_ill
    else:
        illiquid = cfg_nav0 * w_ill
        equity = cfg_nav0 * w_eq + loan
    return {"nav0": cfg_nav0, "loan": loan, "equity_A": equity, "illiquid": illiquid, "equity_sleeve_B": cfg_nav0 - illiquid}


def held_beta_to_book(betas: dict[str, float], index_names: list[str], book_weights: Sequence[float] | NDArray[np.float64], override: float | None) -> float:
    """β of the held equity book to the option-book index: the explicit override, else Σ β_i w_i."""
    if override is not None:
        return float(override)
    return float(sum(betas[n] * float(book_weights[i]) for i, n in enumerate(index_names)))
