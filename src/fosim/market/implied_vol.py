"""Short-dated implied vol as a log-OU process (SPEC §4.2), P-measure dynamics.

x = ln IV_s:  x_{t+dt} = x_t e^{−κ dt} + ln θ (1 − e^{−κ dt}) + η √((1 − e^{−2κ dt})/(2κ)) Z_IV
              + j_IV · N_t (additive jump per equity jump event), then floor/cap on IV_s.
Stationary distribution of x (no jumps): mean ln θ, variance η²/(2κ) (test 11).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fosim.config.schema import ShortIVConfig

F64 = NDArray[np.float64]


def simulate_iv_short(cfg: ShortIVConfig, dt: float, z: F64, jump_counts: F64 | None) -> F64:
    """Return IV_s paths of shape (P, N+1)."""
    P, N = z.shape
    x = np.empty((P, N + 1), dtype=np.float64)
    iv0 = cfg.iv0 if cfg.iv0 is not None else cfg.theta
    x[:, 0] = np.log(iv0)
    ek = np.exp(-cfg.kappa * dt)
    sd = cfg.eta * np.sqrt((1.0 - np.exp(-2.0 * cfg.kappa * dt)) / (2.0 * cfg.kappa))
    lt = np.log(cfg.theta)
    lo, hi = np.log(cfg.floor), np.log(cfg.cap)
    for k in range(N):
        xn = x[:, k] * ek + lt * (1.0 - ek) + sd * z[:, k]
        if jump_counts is not None and cfg.jump_add != 0.0:
            xn = xn + cfg.jump_add * jump_counts[:, k]
        x[:, k + 1] = np.clip(xn, lo, hi)
    return np.exp(x)


def stationary_moments(cfg: ShortIVConfig) -> tuple[float, float]:
    """(mean, variance) of ln IV_s in the stationary regime without jumps or clipping."""
    return float(np.log(cfg.theta)), float(cfg.eta**2 / (2.0 * cfg.kappa))
