"""Lombard loan and margin mechanics (SPEC §4.3 loan flexibility, §5.1).

Loan rate at a reset date: ``max(base, floor) + spread(tier)``; base = short rate (SOFR proxy);
tiers keyed on facility utilisation (loan / facility limit) or loan size. Interest accrues on the
start-of-step balance with simple interest on the configured day count; paid in cash or capitalised.
Commitment fee accrues on the undrawn limit; arrangement fee is charged once at inception.

Margin: lending value LV = Σ ℓ_i(t) MV_i with ℓ_i(t) = ℓ_i^base · h(state) where h is the step
function of IV_s and index drawdown in ``ltv_stress_schedule`` (the smallest applicable multiplier).
Utilisation u = L/LV. Cure sale size to restore u* selling a basket of average LTV ℓ at slippage s:
    x = (L − u* LV) / ((1 − s) − u* ℓ),   valid when the denominator > 0.
Closed-form trigger for constant LTVs and no accrual: d* = 1 − (L − ℓ_I I)/(ℓ_E E₀).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fosim.config.schema import LeverageConfig, LoanTermsConfig
from fosim.engine.conventions import accrual_factor

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class LoanTerms:
    cfg: LoanTermsConfig
    lev: LeverageConfig

    def spread(self, loan: F64) -> F64:
        """Tiered spread by facility utilisation (loan / limit) or loan size."""
        basis = loan / self.cfg.facility_limit if self.cfg.tier_basis == "facility_utilisation" else loan
        out = np.full_like(loan, self.cfg.spread_tiers[-1].spread, dtype=np.float64)
        # tiers ascending in utilisation_below; the first tier whose bound strictly exceeds the basis applies
        for tier in reversed(self.cfg.spread_tiers):
            out = np.where(basis < tier.utilisation_below, tier.spread, out)
        return out

    def floating_rate(self, base: ArrayLike, loan: F64) -> F64:
        b = np.asarray(base, dtype=np.float64)
        if self.cfg.base_floor is not None:
            b = np.maximum(b, self.cfg.base_floor)
        out: F64 = b + self.spread(loan)
        return out

    def rate_at(self, base: ArrayLike, loan: F64, t_years: float) -> F64:
        """Contractual rate at time t: fixed for ``fixed_years`` when rate_type is fixed/swapped, else floating."""
        if self.cfg.rate_type in ("fixed", "floating_swapped") and self.cfg.fixed_years is not None and t_years < self.cfg.fixed_years - 1e-12:
            assert self.cfg.fixed_rate is not None
            return np.full_like(loan, self.cfg.fixed_rate + float(np.mean(self.spread(loan))), dtype=np.float64) if self.cfg.rate_type == "fixed" else self.cfg.fixed_rate + self.spread(loan)
        return self.floating_rate(base, loan)

    def is_reset_step(self, month_index: int, is_month_end: bool) -> bool:
        return is_month_end and (month_index % self.cfg.reset_months == 0)

    def interest(self, loan: F64, rate: F64, days: float) -> F64:
        out: F64 = loan * rate * accrual_factor(days, self.lev.day_count)
        return out

    def commitment_fee(self, loan: F64, days: float) -> F64:
        undrawn = np.maximum(self.cfg.facility_limit - loan, 0.0)
        out: F64 = undrawn * self.cfg.commitment_fee_bps * 1e-4 * accrual_factor(days, self.lev.day_count)
        return out

    def arrangement_fee(self, loan0: float) -> float:
        return loan0 * self.cfg.arrangement_fee_bps * 1e-4

    def swap_mtm(self, loan: F64, zero_rate_fn: Callable[[float], F64], t_years: float) -> F64:
        """Pay-fixed / receive-float swap MTM on the loan notional (floating_swapped), annual periods.

        Value ≈ L · [(1 − DF(T_rem)) − fixed · Σ_i DF(τ_i)·Δ_i]  (float leg PV minus fixed leg PV).
        """
        if self.cfg.rate_type != "floating_swapped" or self.cfg.fixed_years is None or self.cfg.fixed_rate is None:
            return np.zeros_like(loan)
        rem = self.cfg.fixed_years - t_years
        if rem <= 1e-9:
            return np.zeros_like(loan)
        n_full = int(np.floor(rem + 1e-9))
        taus = [rem - n_full + i for i in range(0, n_full + 1)] if rem - n_full > 1e-9 else [float(i) for i in range(1, n_full + 1)]
        taus = [t for t in taus if t > 1e-9]
        dfs = [np.exp(-zero_rate_fn(t) * t) for t in taus]
        prev = 0.0
        annuity = np.zeros_like(loan)
        for t, df in zip(taus, dfs, strict=True):
            annuity = annuity + df * (t - prev)
            prev = t
        out: F64 = loan * ((1.0 - dfs[-1]) - self.cfg.fixed_rate * annuity)
        return out


# ------------------------------------------------------------------------------ margin


def ltv_multiplier(lev: LeverageConfig, iv_short: F64, index_drawdown: F64) -> F64:
    """h(state): the smallest multiplier among the stress rules whose trigger is breached (1 if none)."""
    h = np.ones_like(iv_short)
    for rule in lev.ltv_stress_schedule:
        hit = np.zeros_like(iv_short, dtype=bool)
        if rule.iv_short_above is not None:
            hit |= iv_short > rule.iv_short_above
        if rule.index_drawdown_below is not None:
            hit |= index_drawdown < rule.index_drawdown_below
        h = np.where(hit, np.minimum(h, rule.multiplier), h)
    return h


def lending_value(lev: LeverageConfig, equity_mv: F64, illiquid_mv: F64, cash: F64, h: F64) -> F64:
    """LV = h · (ℓ_E · E + ℓ_I · I + ℓ_cash · max(cash, 0))."""
    lb = lev.ltv_base
    out: F64 = h * (lb.equity_index * equity_mv + lb.illiquid * illiquid_mv + lb.cash * np.maximum(cash, 0.0))
    return out


def utilisation(loan: F64, lv: F64) -> F64:
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(lv > 0, loan / np.maximum(lv, 1e-300), np.where(loan > 0, np.inf, 0.0))
    return np.asarray(u, dtype=np.float64)


def cure_sale_size(loan: F64, lv: F64, u_target: float, ltv_basket: F64, slippage: float) -> F64:
    """x = (L − u* LV) / ((1 − s) − u* ℓ); asserts a positive denominator. Returns max(x, 0)."""
    denom = (1.0 - slippage) - u_target * ltv_basket
    if np.any(denom <= 0):
        raise ValueError("cure sale formula invalid: (1 − s) − u*·ℓ must be positive")
    x = (loan - u_target * lv) / denom
    return np.maximum(x, 0.0)


def margin_trigger_decline(loan: float, equity0: float, illiquid: float, ltv_e: float, ltv_i: float) -> float:
    """d* = 1 − (L − ℓ_I I)/(ℓ_E E₀): equity decline at which u reaches 1 (constant LTVs, no accrual)."""
    return 1.0 - (loan - ltv_i * illiquid) / (ltv_e * equity0)
