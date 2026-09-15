"""Correlation-matrix validation, Higham (2002) nearest correlation matrix, Cholesky (SPEC §4.1)."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidatedCorrelation:
    names: list[str]
    matrix: F64  # the (possibly repaired) PSD correlation matrix
    cholesky: F64  # lower-triangular L with L Lᵀ = matrix
    repaired: bool
    frobenius_distance: float
    min_eigenvalue_input: float


def validate_correlation(names: list[str], matrix: F64, tol: float = 1e-10) -> ValidatedCorrelation:
    """Check symmetry / unit diagonal / |ρ| ≤ 1; repair to the nearest PSD correlation matrix if needed.

    A repair emits a ``UserWarning`` (loud) and logs the Frobenius distance.
    """
    m = np.asarray(matrix, dtype=np.float64)
    n = len(names)
    if m.shape != (n, n):
        raise ValueError(f"correlation matrix must be {n}x{n}")
    if not np.allclose(m, m.T, atol=tol):
        raise ValueError("correlation matrix is not symmetric")
    if not np.allclose(np.diag(m), 1.0, atol=tol):
        raise ValueError("correlation matrix diagonal must be 1")
    if np.any(np.abs(m) > 1 + tol):
        raise ValueError("correlation entries must lie in [-1, 1]")
    eig = np.linalg.eigvalsh(m)
    min_eig = float(eig.min())
    repaired = False
    dist = 0.0
    out = m.copy()
    if min_eig < -1e-12:
        out = nearest_correlation_matrix(m)
        dist = float(np.linalg.norm(out - m, "fro"))
        repaired = True
        msg = (
            f"Correlation matrix is not positive semi-definite (min eigenvalue {min_eig:.3e}); "
            f"replaced by the nearest correlation matrix (Higham 2002), Frobenius distance {dist:.4e}"
        )
        warnings.warn(msg, stacklevel=2)
        log.warning(msg)
    # Cholesky on the PSD matrix with a tiny diagonal regularisation for exact-singular cases
    chol = _safe_cholesky(out)
    return ValidatedCorrelation(list(names), out, chol, repaired, dist, min_eig)


def _safe_cholesky(m: F64) -> F64:
    jitter = 0.0
    for _ in range(8):
        try:
            L: F64 = np.linalg.cholesky(m + jitter * np.eye(len(m)))
            if jitter > 0:
                # renormalise so that L Lᵀ has a unit diagonal again
                d = np.sqrt(np.sum(L * L, axis=1))
                L = L / d[:, None]
            return L
        except np.linalg.LinAlgError:
            jitter = 1e-14 if jitter == 0 else jitter * 100
    raise np.linalg.LinAlgError("Cholesky factorisation failed even after regularisation")


def nearest_correlation_matrix(a: F64, tol: float = 1e-12, max_iter: int = 1000) -> F64:
    """Higham (2002) alternating projections with Dykstra's correction."""
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    y = a.copy()
    ds = np.zeros_like(a)
    for _ in range(max_iter):
        r = y - ds
        # projection onto PSD cone
        w, v = np.linalg.eigh(r)
        x_new = (v * np.maximum(w, 0.0)) @ v.T
        x_new = 0.5 * (x_new + x_new.T)
        ds = x_new - r
        # projection onto unit-diagonal set
        y_new = x_new.copy()
        y_new[np.diag_indices(n)] = 1.0
        converged = (
            np.linalg.norm(y_new - y, "fro") / max(np.linalg.norm(y, "fro"), 1e-300) < tol
            and np.linalg.norm(x_new - y_new, "fro") < 1e-10
        )
        y = y_new
        if converged:
            break
    out = 0.5 * (y + y.T)
    out[np.diag_indices(n)] = 1.0
    # final tiny eigenvalue clip to guarantee PSD to machine precision
    w, v = np.linalg.eigh(out)
    if w.min() < 0:
        out = (v * np.maximum(w, 0.0)) @ v.T
        d = np.sqrt(np.diag(out))
        out = out / np.outer(d, d)
    return np.asarray(out, dtype=np.float64)
