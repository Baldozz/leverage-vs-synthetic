"""Interest-rate models (SPEC §4.3): flat, Vasicek (exact discretisation, closed-form bonds), curve.

One USD rate model drives three derived rates (with their own spreads):
  * loan base rate  = short rate r_t (SOFR proxy), floor and spread applied in the loan instrument;
  * cash yield      = r_t − rates.cash.spread;
  * option rate(T)  = zero rate y_t(T) (+ pricing.option_funding_spread).

Vasicek: dr = a(b − r)dt + σ_r dW,
    r_{t+dt} = r_t e^{−a dt} + b(1 − e^{−a dt}) + σ_r √((1 − e^{−2a dt})/(2a)) Z
    P(τ) = exp(A(τ) − B(τ) r),  B = (1 − e^{−aτ})/a,
    A = (B − τ)(a²b − σ_r²/2)/a² − σ_r² B²/(4a);  y(τ) = −ln P/τ.
The Vasicek zero curve is the P-measure curve; ``term_premium`` (annualised, applied linearly from
0 at τ = 0 to full at τ ≥ 1y) is an explicit P→Q wedge, default 0 (assumption P = Q).

Curve mode: user zero or par curve (annual coupons for par), bootstrapped to discount factors and
interpolated linearly in log DF. Deterministic scenario shifts: parallel, steepener, flattener,
buckets. Negative rates are permitted everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from fosim.config.schema import RatesConfig, ScenarioShift

F64 = NDArray[np.float64]


class RateModel(Protocol):
    """Zero rate y(τ) for continuous compounding, given the short-rate state ``r_short`` (per path)."""

    def zero_rate(self, r_short: ArrayLike, tenor: ArrayLike) -> F64: ...

    def short_rate_path(self, r0: float, n_paths: int, n_steps: int, dt: float, z: F64 | None) -> F64: ...

    @property
    def stochastic(self) -> bool: ...


def apply_shift(tenor: F64, zero: F64, shift: ScenarioShift) -> F64:
    """Deterministic curve shifts in bp: parallel; steepener (+bp at 10y, 0 at 0y); flattener (−); buckets."""
    t = np.asarray(tenor, dtype=np.float64)
    z = np.asarray(zero, dtype=np.float64)
    bp = shift.bp * 1e-4
    if shift.type == "none":
        return z
    if shift.type == "parallel":
        return z + bp
    if shift.type == "steepener":
        return z + bp * np.clip(t / 10.0, 0.0, 1.0)
    if shift.type == "flattener":
        return z - bp * np.clip(t / 10.0, 0.0, 1.0)
    if shift.type == "buckets":
        if not shift.buckets:
            raise ValueError("scenario_shift.type=buckets requires a buckets mapping {tenor: bp}")
        ks = np.array(sorted(shift.buckets), dtype=np.float64)
        vs = np.array([shift.buckets[k] for k in sorted(shift.buckets)], dtype=np.float64) * 1e-4
        return z + np.interp(t, ks, vs)
    raise ValueError(shift.type)


@dataclass(frozen=True)
class FlatRates:
    """Deterministic constant or term-structure curve; short rate = zero rate at the shortest tenor."""

    tenors: F64
    zeros: F64
    shift: ScenarioShift

    @property
    def stochastic(self) -> bool:
        return False

    @property
    def r0(self) -> float:
        return float(self.zero_rate(0.0, self.tenors[0]))

    def zero_rate(self, r_short: ArrayLike, tenor: ArrayLike) -> F64:
        t = np.asarray(tenor, dtype=np.float64)
        r = np.asarray(r_short, dtype=np.float64)
        if len(self.tenors) == 1:
            base = np.full(np.broadcast(t, r).shape, self.zeros[0])
        else:
            base = np.broadcast_to(np.interp(t, self.tenors, self.zeros), np.broadcast(t, r).shape).copy()
        return apply_shift(np.broadcast_to(t, base.shape), base, self.shift)

    def short_rate_path(self, r0: float, n_paths: int, n_steps: int, dt: float, z: F64 | None) -> F64:
        return np.full((n_paths, n_steps + 1), self.r0, dtype=np.float64)


@dataclass(frozen=True)
class VasicekRates:
    a: float
    b: float
    sigma: float
    term_premium: float
    shift: ScenarioShift

    @property
    def stochastic(self) -> bool:
        return True

    def B(self, tau: F64) -> F64:
        out: F64 = (1.0 - np.exp(-self.a * tau)) / self.a
        return out

    def A(self, tau: F64) -> F64:
        B = self.B(tau)
        a, b, s = self.a, self.b, self.sigma
        out: F64 = (B - tau) * (a * a * b - 0.5 * s * s) / (a * a) - s * s * B * B / (4.0 * a)
        return out

    def bond_price(self, r: ArrayLike, tau: ArrayLike) -> F64:
        r_ = np.asarray(r, dtype=np.float64)
        t_ = np.asarray(tau, dtype=np.float64)
        out: F64 = np.exp(self.A(t_) - self.B(t_) * r_)
        return out

    def zero_rate(self, r_short: ArrayLike, tenor: ArrayLike) -> F64:
        r_ = np.asarray(r_short, dtype=np.float64)
        t_ = np.asarray(tenor, dtype=np.float64)
        r_, t_ = np.broadcast_arrays(r_, t_)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(t_ > 0, -(self.A(t_) - self.B(t_) * r_) / np.maximum(t_, 1e-300), r_)
        y = y + self.term_premium * np.clip(t_, 0.0, 1.0)
        return apply_shift(t_, np.asarray(y, dtype=np.float64), self.shift)

    def short_rate_path(self, r0: float, n_paths: int, n_steps: int, dt: float, z: F64 | None) -> F64:
        if z is None:
            raise ValueError("Vasicek needs rate shocks z of shape (n_paths, n_steps)")
        r = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        r[:, 0] = r0
        ea = np.exp(-self.a * dt)
        sd = self.sigma * np.sqrt((1.0 - np.exp(-2.0 * self.a * dt)) / (2.0 * self.a))
        for k in range(n_steps):
            r[:, k + 1] = r[:, k] * ea + self.b * (1.0 - ea) + sd * z[:, k]
        return r

    def stationary_mean_var(self) -> tuple[float, float]:
        return self.b, self.sigma**2 / (2.0 * self.a)

    def conditional_mean_var(self, r0: float, t: float) -> tuple[float, float]:
        ea = np.exp(-self.a * t)
        return float(r0 * ea + self.b * (1 - ea)), float(self.sigma**2 * (1 - np.exp(-2 * self.a * t)) / (2 * self.a))


@dataclass(frozen=True)
class CurveRates:
    """Deterministic curve from discount factors on pillars; log-linear DF interpolation."""

    tenors: F64  # pillars (> 0), ascending
    log_df: F64
    shift: ScenarioShift

    @property
    def stochastic(self) -> bool:
        return False

    @property
    def r0(self) -> float:
        return float(-self.log_df[0] / self.tenors[0])

    def zero_rate(self, r_short: ArrayLike, tenor: ArrayLike) -> F64:
        t = np.asarray(tenor, dtype=np.float64)
        r = np.asarray(r_short, dtype=np.float64)
        shape = np.broadcast(t, r).shape
        t_b = np.broadcast_to(t, shape)
        # log DF linear in tenor between pillars; log DF(0) = 0; flat zero-rate extrapolation beyond the last pillar
        xs = np.concatenate([[0.0], self.tenors])
        ys = np.concatenate([[0.0], self.log_df])
        ldf = np.interp(np.minimum(t_b, self.tenors[-1]), xs, ys)
        last_zero = -self.log_df[-1] / self.tenors[-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(t_b > 0, -ldf / np.maximum(t_b, 1e-300), self.r0)
        y = np.where(t_b > self.tenors[-1], last_zero, y)
        return apply_shift(t_b, np.asarray(y, dtype=np.float64), self.shift)

    def short_rate_path(self, r0: float, n_paths: int, n_steps: int, dt: float, z: F64 | None) -> F64:
        return np.full((n_paths, n_steps + 1), self.r0, dtype=np.float64)

    def discount_factor(self, tenor: ArrayLike) -> F64:
        t = np.asarray(tenor, dtype=np.float64)
        out: F64 = np.exp(-self.zero_rate(0.0, t) * t)
        return out


def bootstrap_par_curve(tenors: F64, par_rates: F64) -> F64:
    """Bootstrap annual-coupon par rates to log discount factors on the pillars.

    Pillars must be whole years (1, 2, ..., n) or a subset; log DF is interpolated linearly between
    known pillars for intermediate coupon dates (iterative solve). Returns log DF at the pillars.
    """
    tenors = np.asarray(tenors, dtype=np.float64)
    par = np.asarray(par_rates, dtype=np.float64)
    if np.any(tenors <= 0) or np.any(np.diff(tenors) <= 0):
        raise ValueError("par curve tenors must be positive and strictly increasing")
    log_df: list[float] = []
    known_t: list[float] = [0.0]
    known_l: list[float] = [0.0]
    for T, c in zip(tenors, par, strict=True):
        n = round(float(T))
        if abs(T - n) > 1e-9:
            raise ValueError("par bootstrap requires whole-year pillar tenors (annual coupons)")
        coupon_dates = np.arange(1, n, dtype=np.float64)  # 1..n-1

        def solve(lT: float, T: float = T, c: float = c, coupon_dates: F64 = coupon_dates) -> float:
            xs = np.array([*known_t, T])
            ys = np.array([*known_l, lT])
            df_c = np.exp(np.interp(coupon_dates, xs, ys))
            return float(c * df_c.sum() + (1 + c) * np.exp(lT) - 1.0)

        from scipy.optimize import brentq

        lT_sol: float = brentq(solve, -5.0, 1.0, xtol=1e-15, rtol=1e-15, maxiter=500)
        log_df.append(lT_sol)
        known_t.append(float(T))
        known_l.append(lT_sol)
    return np.asarray(log_df, dtype=np.float64)


def par_rate_from_curve(curve: CurveRates, T: int) -> float:
    """Annual-coupon par rate implied by the curve (used to check the bootstrap)."""
    dfs = curve.discount_factor(np.arange(1, T + 1, dtype=np.float64))
    return float((1.0 - dfs[-1]) / dfs.sum())


def build_rate_model(cfg: RatesConfig, base_dir: Path | None = None) -> tuple[RateModel, float]:
    """Return (model, r0)."""
    if cfg.model == "flat":
        if cfg.flat_curve is not None:
            m = FlatRates(np.asarray(cfg.flat_curve.tenors, dtype=np.float64), np.asarray(cfg.flat_curve.zero, dtype=np.float64), cfg.scenario_shift)
        else:
            m = FlatRates(np.array([1.0]), np.array([cfg.usd.r0]), cfg.scenario_shift)
        return m, m.r0
    if cfg.model == "vasicek":
        v = VasicekRates(cfg.usd.a, cfg.usd.b, cfg.usd.sigma, cfg.usd.term_premium, cfg.scenario_shift)
        return v, cfg.usd.r0
    if cfg.model == "curve":
        assert cfg.curves_file is not None
        p = Path(cfg.curves_file)
        if base_dir is not None and not p.is_absolute():
            p = base_dir / p
        df = pd.read_csv(p)
        cols = {c.lower().strip(): c for c in df.columns}
        if "tenor" not in cols or "rate" not in cols:
            raise ValueError("curve CSV needs columns tenor, rate (decimal)")
        t = df[cols["tenor"]].to_numpy(dtype=np.float64)
        r = df[cols["rate"]].to_numpy(dtype=np.float64)
        order = np.argsort(t)
        t, r = t[order], r[order]
        if cfg.curve_type == "par":
            ldf = bootstrap_par_curve(t, r)
        else:
            ldf = -r * t
        c = CurveRates(t, ldf, cfg.scenario_shift)
        return c, c.r0
    raise ValueError(cfg.model)
