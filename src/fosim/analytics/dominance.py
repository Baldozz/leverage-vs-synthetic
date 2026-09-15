"""Empirical first- and second-order stochastic dominance tests (SPEC §6.1)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


def _ecdf(x: F64, grid: F64) -> F64:
    xs = np.sort(x)
    return np.searchsorted(xs, grid, side="right") / len(xs)


def stochastic_dominance(x: F64, y: F64, tol: float = 1e-9) -> dict[str, object]:
    """Does X dominate Y? FSD: F_X ≤ F_Y everywhere; SSD: ∫F_X ≤ ∫F_Y everywhere (on the pooled grid).

    Returns the verdicts and the maximum violation (how far the dominated CDF crosses over).
    """
    grid = np.sort(np.concatenate([x, y]))
    Fx, Fy = _ecdf(x, grid), _ecdf(y, grid)
    d = Fx - Fy
    fsd = bool(np.all(d <= tol))
    fsd_rev = bool(np.all(-d <= tol))
    dx = np.diff(grid, prepend=grid[0])
    Ix, Iy = np.cumsum(Fx * dx), np.cumsum(Fy * dx)
    d2 = Ix - Iy
    ssd = bool(np.all(d2 <= tol * max(1.0, float(np.abs(Iy).max()))))
    ssd_rev = bool(np.all(-d2 <= tol * max(1.0, float(np.abs(Ix).max()))))
    return {
        "x_fsd_y": fsd, "y_fsd_x": fsd_rev, "x_ssd_y": ssd, "y_ssd_x": ssd_rev,
        "max_fsd_violation": float(np.max(d)), "max_ssd_violation": float(np.max(d2)),
        "crossing_points": int(np.sum(np.diff(np.sign(d[np.abs(d) > tol])) != 0)) if np.any(np.abs(d) > tol) else 0,
    }
