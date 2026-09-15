"""Implied-volatility surface and option price sources (SPEC §4.2, §4.2b).

Parametric surface (default):
    IV_ATM(T, t) = θ_T · (IV_s,t / θ_s)^{β(T)}            (θ_T, β(T) linear in T on the user grid)
    σ(K, T, t)   = max(IV_ATM(T, t) + ψ(T)·k, floor),  k = ln(K/F),  ψ(T) = ψ_1y / √T
    sticky_moneyness: k re-evaluated at the current forward; sticky_strike: k frozen at purchase.
Bid/ask: σ_ask = σ_mid + ½·spread, σ_bid = σ_mid − ½·spread (vol points; separate spread for new
trades and unwinds; multiplied by a stress factor when IV_s exceeds a threshold).

Precedence of sources (highest first): dealer_quotes → vol_surface_file → parametric;
scenario_overrides replace the ATM level for the months they cover (the skew is still applied);
historical_iv_file feeds IV_s (and optional long-dated levels) through the market module.

Grid surface (CSV, columns ``tenor,moneyness,iv`` with moneyness = K/F, or ``tenor,delta,iv``):
linear interpolation in total variance w = σ²T along tenor and monotone (PCHIP) interpolation
across log-moneyness; calendar- and butterfly-arbitrage checks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import ndtr

from fosim.config.schema import DealerQuote, ImpliedVolConfig, PricingConfig, ScenarioOverride
from fosim.pricing.black_scholes import bsm_d1_d2, bsm_price, forward_price
from fosim.pricing.implied_vol import implied_vol

F64 = NDArray[np.float64]
SourceLabel = Literal["parametric", "dealer_quote", "parametric_quote_calibrated", "surface_file", "scenario_override"]


class ArbitrageViolation(ValueError):
    """Raised when a user vol surface has calendar or butterfly arbitrage and enforcement is on."""


def _interp(x: ArrayLike, xp: list[float] | F64, fp: list[float] | F64) -> F64:
    """Linear interpolation with flat extrapolation."""
    out: F64 = np.interp(np.asarray(x, dtype=np.float64), np.asarray(xp, dtype=np.float64), np.asarray(fp, dtype=np.float64))
    return out


# ------------------------------------------------------------------------------ parametric surface


@dataclass(frozen=True)
class ParametricSurface:
    """The §4.2 model. All methods vectorise over ``k``, ``T`` and ``iv_short``."""

    tenors: F64
    theta: F64
    beta: F64
    theta_short: float
    skew_1y: float
    skew_mode: Literal["sticky_moneyness", "sticky_strike"]
    vol_floor: float
    level_shift: tuple[F64, F64] | None = None  # (tenors, additive shifts) from dealer-quote calibration

    @staticmethod
    def from_config(cfg: ImpliedVolConfig) -> ParametricSurface:
        ts = cfg.term_structure
        return ParametricSurface(
            tenors=np.asarray(ts.tenors, dtype=np.float64),
            theta=np.asarray(ts.theta, dtype=np.float64),
            beta=np.asarray(ts.beta, dtype=np.float64),
            theta_short=cfg.short.theta,
            skew_1y=cfg.skew_1y,
            skew_mode=cfg.skew_mode,
            vol_floor=cfg.vol_floor,
        )

    def theta_T(self, T: ArrayLike) -> F64:
        return _interp(T, self.tenors, self.theta)

    def beta_T(self, T: ArrayLike) -> F64:
        return _interp(T, self.tenors, self.beta)

    def psi_T(self, T: ArrayLike) -> F64:
        """Skew slope ψ(T) = ψ_1y / √T in vol per unit log-moneyness."""
        T_ = np.asarray(T, dtype=np.float64)
        with np.errstate(divide="ignore"):
            out: F64 = np.where(T_ > 0, self.skew_1y / np.sqrt(np.maximum(T_, 1e-300)), 0.0)
        return out

    def atm_vol(self, T: ArrayLike, iv_short: ArrayLike) -> F64:
        """IV_ATM(T, t) = θ_T (IV_s/θ_s)^{β(T)} (+ calibration shift)."""
        T_ = np.asarray(T, dtype=np.float64)
        ivs = np.asarray(iv_short, dtype=np.float64)
        level = self.theta_T(T_) * (ivs / self.theta_short) ** self.beta_T(T_)
        if self.level_shift is not None:
            level = level + _interp(T_, self.level_shift[0], self.level_shift[1])
        out: F64 = np.maximum(level, self.vol_floor)
        return out

    def vol(self, k: ArrayLike, T: ArrayLike, iv_short: ArrayLike) -> F64:
        """σ(k, T, t) with k = ln(K/F) (caller supplies the sticky-rule-consistent k)."""
        out: F64 = np.maximum(self.atm_vol(T, iv_short) + self.psi_T(T) * np.asarray(k, dtype=np.float64), self.vol_floor)
        return out

    def with_level_shift(self, tenors: F64, shifts: F64) -> ParametricSurface:
        return ParametricSurface(
            tenors=self.tenors, theta=self.theta, beta=self.beta, theta_short=self.theta_short,
            skew_1y=self.skew_1y, skew_mode=self.skew_mode, vol_floor=self.vol_floor,
            level_shift=(np.asarray(tenors, dtype=np.float64), np.asarray(shifts, dtype=np.float64)),
        )


# ------------------------------------------------------------------------------ grid surface


@dataclass(frozen=True)
class GridSurface:
    """User vol surface on a tenor × log-moneyness grid (moneyness = K/F)."""

    tenors: F64  # sorted, ascending
    log_moneyness: F64  # sorted, ascending
    iv: F64  # shape (n_tenor, n_k)
    calendar_violations: list[tuple[float, float, float]] = field(default_factory=list)  # (k, T1, T2)
    butterfly_violations: list[tuple[float, float]] = field(default_factory=list)  # (T, k)

    @staticmethod
    def from_csv(path: str | Path, r: float = 0.0, q: float = 0.0) -> GridSurface:
        df = pd.read_csv(path)
        cols = {c.lower().strip(): c for c in df.columns}
        if "tenor" not in cols or "iv" not in cols:
            raise ValueError("vol surface CSV needs columns tenor, iv and either moneyness (K/F) or delta")
        df = df.rename(columns={cols["tenor"]: "tenor", cols["iv"]: "iv"})
        if "moneyness" in cols:
            df = df.rename(columns={cols["moneyness"]: "moneyness"})
            df["k"] = np.log(df["moneyness"].astype(float))
        elif "delta" in cols:
            df = df.rename(columns={cols["delta"]: "delta"})
            # convert call delta (in (0,1)) to forward moneyness under BSM with the quoted iv:
            # delta = e^{-qT} N(d1) -> d1 = N^{-1}(delta e^{qT}); k = ln(K/F) = -(d1 σ√T) + σ²T/2
            from scipy.special import ndtri

            T = df["tenor"].astype(float).to_numpy()
            s = df["iv"].astype(float).to_numpy()
            d = df["delta"].astype(float).to_numpy()
            d1 = ndtri(np.clip(d * np.exp(q * T), 1e-12, 1 - 1e-12))
            df["k"] = -d1 * s * np.sqrt(T) + 0.5 * s * s * T
        else:
            raise ValueError("vol surface CSV needs a moneyness (K/F) or delta column")
        piv = df.pivot_table(index="tenor", columns="k", values="iv", aggfunc="mean")
        if piv.isna().any().any():
            raise ValueError("vol surface grid must be rectangular (every tenor × strike filled)")
        tenors = piv.index.to_numpy(dtype=np.float64)
        ks = piv.columns.to_numpy(dtype=np.float64)
        iv = piv.to_numpy(dtype=np.float64)
        if np.any(iv <= 0):
            raise ValueError("vol surface implied vols must be positive")
        surf = GridSurface(tenors=tenors, log_moneyness=ks, iv=iv)
        return surf.with_checks()

    def with_checks(self) -> GridSurface:
        cal: list[tuple[float, float, float]] = []
        w = self.iv**2 * self.tenors[:, None]
        for j, k in enumerate(self.log_moneyness):
            for i in range(len(self.tenors) - 1):
                if w[i + 1, j] < w[i, j] - 1e-12:
                    cal.append((float(k), float(self.tenors[i]), float(self.tenors[i + 1])))
        bfly: list[tuple[float, float]] = []
        # butterfly: normalised call price c(K) = N(d1) - K N(d2) with F = 1, r = q = 0 must be convex in K
        for i, T in enumerate(self.tenors):
            interp = self._strike_interp(i)
            kk = np.linspace(self.log_moneyness[0], self.log_moneyness[-1], 201)
            K = np.exp(kk)
            s = interp(kk)
            sT = s * np.sqrt(T)
            d1 = (-kk + 0.5 * sT * sT) / sT
            d2 = d1 - sT
            c = ndtr(d1) - K * ndtr(d2)
            dK = np.diff(K)
            second = (c[2:] - c[1:-1]) / dK[1:] - (c[1:-1] - c[:-2]) / dK[:-1]
            second = second / (0.5 * (dK[1:] + dK[:-1]))
            bad = np.flatnonzero(second < -1e-8)
            for b in bad:
                bfly.append((float(T), float(kk[b + 1])))
        return GridSurface(self.tenors, self.log_moneyness, self.iv, cal, bfly)

    @property
    def has_arbitrage(self) -> bool:
        return bool(self.calendar_violations or self.butterfly_violations)

    def _strike_interp(self, i: int) -> PchipInterpolator:
        if len(self.log_moneyness) < 2:
            const = float(self.iv[i, 0])
            return PchipInterpolator(np.array([-10.0, 10.0]), np.array([const, const]))
        return PchipInterpolator(self.log_moneyness, self.iv[i], extrapolate=True)

    def vol(self, k: ArrayLike, T: ArrayLike) -> F64:
        """Total-variance-linear in tenor, PCHIP across log-moneyness, flat extrapolation in tenor."""
        k0 = np.asarray(k, dtype=np.float64)
        T0 = np.asarray(T, dtype=np.float64)
        scalar = k0.ndim == 0 and T0.ndim == 0
        k_, T_ = np.broadcast_arrays(np.atleast_1d(k0), np.atleast_1d(T0))
        kc = np.clip(k_, self.log_moneyness[0], self.log_moneyness[-1])
        # vols per tenor row at kc, then interpolate w = σ²T along T
        rows = np.stack([self._strike_interp(i)(kc) for i in range(len(self.tenors))], axis=0)
        w_rows = rows**2 * self.tenors.reshape((-1,) + (1,) * kc.ndim)
        Tc = np.clip(T_, self.tenors[0], self.tenors[-1])
        idx = np.searchsorted(self.tenors, Tc, side="right") - 1
        idx = np.clip(idx, 0, len(self.tenors) - 2)
        t0, t1 = self.tenors[idx], self.tenors[idx + 1]
        wgt = np.where(t1 > t0, (Tc - t0) / (t1 - t0), 0.0)
        w0 = np.take_along_axis(w_rows, idx[None, ...], axis=0)[0]
        w1 = np.take_along_axis(w_rows, (idx + 1)[None, ...], axis=0)[0]
        w = w0 + wgt * (w1 - w0)
        with np.errstate(divide="ignore", invalid="ignore"):
            # sigma = sqrt(w/Tc): flat extrapolation of sigma outside the tenor grid
            out: F64 = np.where(T_ > 0, np.sqrt(np.maximum(w, 0.0) / np.maximum(Tc, 1e-300)), rows[0])
        if scalar:
            return np.asarray(out[0], dtype=np.float64)
        return out


# ------------------------------------------------------------------------------ dealer quotes


@dataclass(frozen=True)
class QuoteReconciliation:
    index: str
    tenor: float
    strike_pct_spot: float
    quote_mid_pct: float
    quote_mid_iv: float
    quote_half_spread_vol: float | None
    model_iv_before: float
    model_premium_before_pct: float
    shift_applied: float


def backout_quote_iv(qt: DealerQuote, S: float, r: float, q: float) -> tuple[float, float, float | None]:
    """Return (mid_iv, mid_premium_pct, half_spread_vol) for a dealer quote.

    Uses the dealer's own spot/rate/dividend where supplied, else the model's inputs.
    """
    S_ = qt.spot if qt.spot is not None else S
    r_ = qt.rate if qt.rate is not None else r
    q_ = qt.dividend if qt.dividend is not None else q
    K = qt.strike_pct_spot * S_
    if qt.bid_iv is not None or qt.ask_iv is not None:
        ivs = [v for v in (qt.bid_iv, qt.ask_iv) if v is not None]
        mid_iv = float(np.mean(ivs))
        half = (qt.ask_iv - qt.bid_iv) / 2 if (qt.bid_iv is not None and qt.ask_iv is not None) else None
        mid_pct = float(bsm_price(S_, K, r_, q_, mid_iv, qt.tenor)) / S_
        return mid_iv, mid_pct, half
    pcts = [v for v in (qt.bid_pct, qt.ask_pct) if v is not None]
    prem = [p * S_ for p in pcts]
    ivs2 = [implied_vol(p, S_, K, r_, q_, qt.tenor, "call") for p in prem]
    mid_iv = float(np.mean(ivs2))
    half2 = (ivs2[1] - ivs2[0]) / 2 if len(ivs2) == 2 else None
    return mid_iv, float(np.mean(pcts)), half2


# ------------------------------------------------------------------------------ composite source


@dataclass
class VolSource:
    """Composite option-vol source for one index implementing the §4.2b precedence.

    ``vol(k, T, iv_short, month)`` returns (σ_mid array, source label). ``k`` is ln(K/F) under the
    caller's sticky rule. Overrides and quotes are looked up by tenor (±``tenor_tol`` years) and
    strike (±``k_tol`` in log-moneyness).
    """

    parametric: ParametricSurface
    pricing: PricingConfig
    grid: GridSurface | None = None
    quotes: dict[tuple[float, float], tuple[float, float | None]] = field(default_factory=dict)  # (tenor, k) -> (iv, half)
    reconciliation: list[QuoteReconciliation] = field(default_factory=list)
    overrides: list[ScenarioOverride] = field(default_factory=list)
    tenor_tol: float = 1.0 / 24.0
    k_tol: float = 0.01

    @staticmethod
    def build(
        iv_cfg: ImpliedVolConfig,
        pricing: PricingConfig,
        index_name: str,
        S0: float,
        r0: float,
        q0: float,
        iv_short0: float,
        base_dir: Path | None = None,
    ) -> VolSource:
        para = ParametricSurface.from_config(iv_cfg)
        grid: GridSurface | None = None
        if pricing.vol_surface_file is not None and pricing.source in ("vol_surface_file", "dealer_quotes"):
            p = Path(pricing.vol_surface_file)
            if base_dir is not None and not p.is_absolute():
                p = base_dir / p
            grid = GridSurface.from_csv(p, r0, q0)
            if grid.has_arbitrage:
                msg = (
                    f"vol surface {p} has arbitrage: calendar={grid.calendar_violations[:5]} "
                    f"butterfly={grid.butterfly_violations[:5]}"
                )
                if pricing.arbitrage_check == "enforce":
                    raise ArbitrageViolation(msg + " — set pricing.arbitrage_check=warn to override")
                import warnings

                warnings.warn(msg, stacklevel=2)
        quotes: dict[tuple[float, float], tuple[float, float | None]] = {}
        recon: list[QuoteReconciliation] = []
        overrides: list[ScenarioOverride] = []
        if pricing.source in ("scenario_overrides", "dealer_quotes", "parametric", "vol_surface_file"):
            for o in pricing.scenario_overrides:
                if o.index is not None and o.index != index_name:
                    continue
                if o.premium_pct is not None:
                    # convert a premium in % of notional into the implied vol of a call struck at strike_pct_spot
                    # (scale-free: uses S = 100; the inception rate/dividend of the option tenor are used for every tenor)
                    T_o = o.tenor if o.tenor is not None else 5.0
                    iv = implied_vol(o.premium_pct * 100.0, 100.0, 100.0 * o.strike_pct_spot, r0, q0, T_o, "call")
                    o = o.model_copy(update={"iv": iv, "premium_pct": None})
                overrides.append(o)
        if pricing.dealer_quotes and pricing.source in ("dealer_quotes",):
            F0 = float(forward_price(S0, r0, q0, 1.0))  # placeholder, per-quote forward computed below
            shifts_t: list[float] = []
            shifts_v: list[float] = []
            for qt in [x for x in pricing.dealer_quotes if x.index == index_name]:
                mid_iv, mid_pct, half = backout_quote_iv(qt, S0, r0, q0)
                F0 = float(forward_price(S0, r0, q0, qt.tenor))
                k_q = float(np.log(qt.strike_pct_spot * S0 / F0))
                model_iv = float(para.vol(k_q, qt.tenor, iv_short0))
                model_pct = float(bsm_price(S0, qt.strike_pct_spot * S0, r0, q0, model_iv, qt.tenor)) / S0
                quotes[(qt.tenor, k_q)] = (mid_iv, half)
                shifts_t.append(qt.tenor)
                shifts_v.append(mid_iv - model_iv)
                recon.append(
                    QuoteReconciliation(
                        index=index_name, tenor=qt.tenor, strike_pct_spot=qt.strike_pct_spot,
                        quote_mid_pct=mid_pct, quote_mid_iv=mid_iv, quote_half_spread_vol=half,
                        model_iv_before=model_iv, model_premium_before_pct=model_pct, shift_applied=mid_iv - model_iv,
                    )
                )
            if shifts_t:
                order = np.argsort(shifts_t)
                # average shifts at duplicate tenors
                st = np.asarray(shifts_t)[order]
                sv = np.asarray(shifts_v)[order]
                ut = np.unique(st)
                uv = np.array([sv[st == t].mean() for t in ut])
                para = para.with_level_shift(ut, uv)
        return VolSource(parametric=para, pricing=pricing, grid=grid, quotes=quotes, reconciliation=recon, overrides=overrides)

    # --- lookup ----------------------------------------------------------------------------
    def _override(self, T: float, month: int) -> ScenarioOverride | None:
        for o in self.overrides:
            if o.iv is None:
                continue
            if month < o.month_from or (o.month_to is not None and month > o.month_to):
                continue
            if o.tenor is not None and abs(o.tenor - T) > self.tenor_tol:
                continue
            return o
        return None

    def _quote(self, T: float, k: float) -> tuple[float, float | None] | None:
        for (tq, kq), v in self.quotes.items():
            if abs(tq - T) <= self.tenor_tol and abs(kq - k) <= self.k_tol:
                return v
        return None

    def vol(self, k: ArrayLike, T: float, iv_short: ArrayLike, month: int = 0) -> tuple[F64, SourceLabel]:
        """Mid vol for log-moneyness ``k`` (array over paths) at residual tenor ``T`` and month."""
        k_ = np.asarray(k, dtype=np.float64)
        ivs = np.asarray(iv_short, dtype=np.float64)
        ov = self._override(T, month)
        if ov is not None and ov.iv is not None:
            out: F64 = np.maximum(ov.iv + self.parametric.psi_T(T) * k_, self.parametric.vol_floor)
            return np.broadcast_to(out, np.broadcast(k_, ivs).shape).copy(), "scenario_override"
        if self.quotes and k_.ndim == 0:
            qv = self._quote(T, float(k_))
            if qv is not None:
                return np.broadcast_to(np.float64(qv[0]), ivs.shape).copy(), "dealer_quote"
        if self.grid is not None and self.pricing.source == "vol_surface_file":
            # grid gives the inception surface; scale by the parametric factor (IV_s/θ_s)^β to move it in time
            base = self.grid.vol(k_, T)
            factor = (ivs / self.parametric.theta_short) ** self.parametric.beta_T(T)
            out2: F64 = np.maximum(base * factor, self.parametric.vol_floor)
            return np.broadcast_to(out2, np.broadcast(k_, ivs).shape).copy(), "surface_file"
        out3 = self.parametric.vol(k_, T, ivs)
        label: SourceLabel = "parametric_quote_calibrated" if self.parametric.level_shift is not None else "parametric"
        return np.broadcast_to(out3, np.broadcast(k_, ivs).shape).copy(), label

    def half_spread(self, T: float, k: float | None, iv_short: ArrayLike, trade: Literal["new", "unwind"]) -> F64:
        """Half bid/ask in vol points, stress-multiplied when IV_s exceeds the configured level."""
        ivs = np.asarray(iv_short, dtype=np.float64)
        if k is not None and self.quotes:
            qv = self._quote(T, k)
            if qv is not None and qv[1] is not None:
                return np.broadcast_to(np.float64(qv[1]), ivs.shape).copy()
        base = self.pricing.bid_ask_vol_pts_new if trade == "new" else self.pricing.bid_ask_vol_pts_unwind
        mult = np.where(ivs > self.pricing.stress_iv_short_above, self.pricing.stress_bid_ask_multiplier, 1.0)
        out: F64 = 0.5 * base * mult * np.ones_like(ivs)
        return out


# ------------------------------------------------------------------------------ strike solvers


def solve_delta_target_strike(
    S: float,
    r: float,
    q: float,
    T: float,
    target_delta: float,
    vol_of_strike: Callable[[float], float],
    delta_definition: Literal["bsm", "smile"] = "bsm",
    psi_T: float = 0.0,
    K_lo_rel: float = 0.3,
    K_hi_rel: float = 5.0,
) -> float:
    """Solve K such that the call delta equals ``target_delta`` with σ(K) re-evaluated inside brentq."""
    from fosim.pricing.black_scholes import bsm_greeks

    def delta_at(K: float) -> float:
        s = vol_of_strike(K)
        g = bsm_greeks(S, K, r, q, s, T, "call")
        d = float(g.delta)
        if delta_definition == "smile":
            d += float(g.vega) * (-psi_T / S)
        return d

    lo, hi = K_lo_rel * S, K_hi_rel * S
    f_lo, f_hi = delta_at(lo) - target_delta, delta_at(hi) - target_delta
    if not (f_lo > 0 > f_hi):
        raise ValueError(
            f"delta_target={target_delta} not bracketed on K∈[{lo:.4g},{hi:.4g}]: delta(lo)={f_lo + target_delta:.4f}, delta(hi)={f_hi + target_delta:.4f}"
        )
    K: float = brentq(lambda K: delta_at(K) - target_delta, lo, hi, xtol=1e-12, rtol=1e-14, maxiter=200)
    return K


def d1_only(S: ArrayLike, K: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike, T: ArrayLike) -> F64:
    return bsm_d1_d2(S, K, r, q, sigma, T)[0]
