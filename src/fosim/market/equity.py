"""Equity index and held-portfolio simulation under the real-world measure P (SPEC §4.1, §4.1b).

GBM (exact):        ln S_{t+dt} = ln S_t + (m − q_c^{px} − σ²/2) dt + σ √dt Z
Merton:             + Σ_{jumps in step} J,  J ~ N(μ_J, σ_J²), common clock N ~ Poisson(λ dt),
                    drift compensator −λ κ dt, κ = exp(μ_J + σ_J²/2) − 1.
Coupled stoch-vol:  σ_t = IV_short,t (1 − vrp) at the start of the step.

``q_c^{px}`` is the part of the total return that leaves the price index:
  price_return: q_c;  total_return / excess_return: 0;  net_total_return: q_c − ln(1 + q (1 − WHT)) (the WHT leakage).
An ``excess_return`` index (e.g. rolled futures) has a flat forward: it is priced with q = r (Black-76) and pays no dividends.

Held portfolio (Strategy A's actual equity book):
    Δln H = Σ_i β_i Δln S_i + (α − σ_ε²/2) dt + σ_ε √dt ε,
so β = 1, σ_ε = 0, α = 0 reproduces the option index exactly (test 38).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fosim.config.schema import EquityIndexConfig, EquityModelConfig, HeldPortfolioConfig
from fosim.engine.conventions import continuous_dividend_yield, continuous_drift

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class IndexParams:
    name: str
    spot: float
    m: float  # continuous total-return drift
    q_c: float  # continuous realised dividend yield
    q_c_px: float  # dividend leakage from the price index (see module docstring)
    sigma: float
    jump_mean: float
    jump_vol: float
    wht: float
    underlying_type: str

    @property
    def cash_dividend_q_c(self) -> float:
        """Continuous yield of dividends paid in cash to a holder of the index (0 for TR indices)."""
        return self.q_c if self.underlying_type == "price_return" else 0.0

    @property
    def kappa(self) -> float:
        return float(np.exp(self.jump_mean + 0.5 * self.jump_vol**2) - 1.0)


def index_params(cfg: EquityIndexConfig) -> IndexParams:
    m = continuous_drift(cfg.expected_return, cfg.return_type, cfg.realized_vol)
    q_c = continuous_dividend_yield(cfg.dividend_yield)
    if cfg.underlying_type == "price_return":
        q_px = q_c
    elif cfg.underlying_type in ("total_return", "excess_return"):
        q_px = 0.0  # TR: dividends reinvested; ER (rolled futures): expected_return is the excess return itself
    else:  # net_total_return: index reinvests dividends net of WHT; the WHT leakage leaves the index
        q_px = q_c - float(np.log1p(cfg.dividend_yield * (1.0 - cfg.dividend_wht)))
    return IndexParams(
        name=cfg.name, spot=cfg.spot, m=m, q_c=q_c, q_c_px=q_px, sigma=cfg.realized_vol,
        jump_mean=cfg.jump_mean, jump_vol=cfg.jump_vol, wht=cfg.dividend_wht, underlying_type=cfg.underlying_type,
    )


def simulate_indices(
    params: list[IndexParams],
    model: EquityModelConfig,
    dt: float,
    z: F64,  # (P, N, n_idx) correlated standard normals
    jump_counts: F64 | None,  # (P, N) common Poisson counts, or None
    jump_z: F64 | None,  # (P, N, n_idx) standard normals for jump sizes
    iv_short: F64 | None = None,  # (P, N+1) for coupled_stochvol
) -> tuple[F64, F64]:
    """Return (S with shape (P, N+1, n_idx), log_jumps (P, N, n_idx))."""
    P, N, n = z.shape
    S = np.empty((P, N + 1, n), dtype=np.float64)
    log_jumps = np.zeros((P, N, n), dtype=np.float64)
    use_jumps = model.type == "merton" or (model.type == "coupled_stochvol" and model.jumps_enabled)
    use_jumps = use_jumps and model.jumps_enabled and model.jump_lambda > 0
    if use_jumps and (jump_counts is None or jump_z is None):
        raise ValueError("jump draws required when jumps are enabled")
    for i, p in enumerate(params):
        S[:, 0, i] = p.spot
        if model.type == "coupled_stochvol":
            if iv_short is None:
                raise ValueError("coupled_stochvol needs iv_short paths")
            sig = iv_short[:, :N] * (1.0 - model.vrp_ratio)  # start-of-step vol
        else:
            sig = np.full((P, N), p.sigma)
        drift = (p.m - p.q_c_px - 0.5 * sig * sig) * dt
        incr = drift + sig * np.sqrt(dt) * z[:, :, i]
        if use_jumps:
            assert jump_counts is not None and jump_z is not None
            nj = jump_counts
            lj = nj * p.jump_mean + p.jump_vol * np.sqrt(nj) * jump_z[:, :, i]
            comp = model.jump_lambda * p.kappa * dt
            incr = incr + lj - comp
            log_jumps[:, :, i] = lj
        S[:, 1:, i] = p.spot * np.exp(np.cumsum(incr, axis=1))
    return S, log_jumps


def simulate_held_portfolio(
    S: F64, held: HeldPortfolioConfig, names: list[str], dt: float, eps: F64, h0: float = 100.0
) -> F64:
    """Held-portfolio level H (P, N+1): Σ β_i Δln S_i + (α − σ_ε²/2) dt + σ_ε √dt ε."""
    P, N1, _ = S.shape
    dlnS = np.diff(np.log(S), axis=1)  # (P, N, n)
    beta = np.array([held.betas[nm] for nm in names], dtype=np.float64)
    te = held.tracking_error_vol
    incr = dlnS @ beta + (held.expected_alpha - 0.5 * te * te) * dt + te * np.sqrt(dt) * eps
    H = np.empty((P, N1), dtype=np.float64)
    H[:, 0] = h0
    H[:, 1:] = h0 * np.exp(np.cumsum(incr, axis=1))
    return H


def running_drawdown(level: F64) -> F64:
    """Drawdown from the running peak along axis 1 (uses only past information)."""
    peak = np.maximum.accumulate(level, axis=1)
    out: F64 = level / peak - 1.0
    return out
