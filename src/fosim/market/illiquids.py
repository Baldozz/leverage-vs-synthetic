"""Illiquid assets (SPEC §4.5): true returns, Geltner smoothing, Takahashi–Alexander cash flows.

True (economic) return per step, lognormal with optional beta to equity jump events:
    ln(1 + G_t) = (m − σ²/2) dt + σ √dt Z + β_J · J_t − λ (e^{β_J μ_J + β_J² σ_J²/2} − 1) dt
where J_t is the option-book-weighted log jump of the equity indices in the step.

Reported NAV (appraisal smoothing, Geltner AR(1)). Let I_true = Π(1 + G) be the fund's unit growth
index (flows never enter it). At reporting dates every ``reporting_months`` months:
    r_true = I_true / I_true,last − 1,   r_obs = (1 − φ) r_true + φ r_obs,prev,   I_obs ← I_obs (1 + r_obs).
The reported mark is NAV_rep,t = NAV_true,t · I_obs,last / I_true,t: it is held flat between reports
(growth since the last report is stripped) and jumps at report dates; cash flows enter both marks
exactly, so a fully distributed fund reports zero. Long-run mean of r_obs equals that of r_true
(test 12); its vol is lower.

Takahashi–Alexander (closed-end): C_t = RC_step U_t (× crisis mult), U_{t+1} = U_t − C_t,
RD_t = min((age/L)^B, 1) converted to per step (× crisis mult), D_t = RD_t NAV_t,
NAV_{t+1} = NAV_t (1 + G_t) + C_t − D_t.   Contributions come from the investor's cash; distributions go to cash.
Illiquids are identical across strategies and are precomputed once per path (they are never sold).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fosim.config.schema import IlliquidConfig
from fosim.engine.conventions import continuous_drift, per_step_rate

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class IlliquidPaths:
    """All arrays (P, N+1, n_ill); flows at index k occur during step k (index 0 = 0)."""

    names: list[str]
    nav_true: F64
    nav_reported: F64
    contributions: F64  # cash paid INTO the fund (positive = outflow from investor cash)
    distributions: F64  # cash received FROM the fund
    unfunded: F64
    true_return: F64  # simple return G_t applied in step k
    reported_return: F64  # r_obs applied in step k (0 between reporting dates)


def simulate_true_returns(
    cfgs: list[IlliquidConfig], dt: float, z: F64, book_log_jump: F64 | None, jump_lambda: float, jump_mean: float, jump_vol: float
) -> F64:
    """Simple true returns G (P, N, n_ill)."""
    P, N, n = z.shape
    G = np.empty((P, N, n), dtype=np.float64)
    for i, c in enumerate(cfgs):
        m = continuous_drift(c.expected_return, c.return_type, c.vol)
        incr = (m - 0.5 * c.vol**2) * dt + c.vol * np.sqrt(dt) * z[:, :, i]
        if book_log_jump is not None and c.jump_beta != 0.0 and jump_lambda > 0:
            bj = c.jump_beta
            comp = jump_lambda * (np.exp(bj * jump_mean + 0.5 * bj * bj * jump_vol**2) - 1.0) * dt
            incr = incr + bj * book_log_jump - comp
        G[:, :, i] = np.expm1(incr)
    return G


def simulate_illiquids(
    cfgs: list[IlliquidConfig],
    nav0: F64,  # (n_ill,) inception NAV per asset
    G: F64,  # (P, N, n_ill) true simple returns per step
    steps_per_year: int,
    is_month_end: NDArray[np.bool_],
    month_index: NDArray[np.int64],
    index_drawdown: F64 | None,  # (P, N+1) option-book index drawdown at each time (for crisis rules)
) -> IlliquidPaths:
    P, N, n = G.shape
    dt = 1.0 / steps_per_year
    nav_t = np.zeros((P, N + 1, n))
    nav_r = np.zeros((P, N + 1, n))
    contr = np.zeros((P, N + 1, n))
    dist = np.zeros((P, N + 1, n))
    unf = np.zeros((P, N + 1, n))
    rrep = np.zeros((P, N + 1, n))
    for i, c in enumerate(cfgs):
        nav_t[:, 0, i] = nav0[i]
        nav_r[:, 0, i] = nav0[i]
        unf[:, 0, i] = c.unfunded_commitments
        # unit growth indices: I_true compounds every step; I_obs is updated at reporting dates from the
        # Geltner-smoothed return of I_true since the last report. Flows never enter these indices.
        i_true = np.ones(P)
        i_true_last = np.ones(P)
        i_obs_last = np.ones(P)
        r_obs_prev = np.zeros(P)
        rc_step = per_step_rate(c.ta_model.rc, dt) if c.ta_model is not None else 0.0
        for k in range(N):
            g = G[:, k, i]
            i_true = i_true * (1.0 + g)
            nav_after_growth = nav_t[:, k, i] * (1.0 + g)
            C = np.zeros(P)
            D = np.zeros(P)
            U = unf[:, k, i].copy()
            if c.type == "closed_end" and c.ta_model is not None:
                ta = c.ta_model
                age = ta.age_years + (k + 1) * dt
                rd_annual = min((age / ta.life_years) ** ta.b, 1.0)
                rd_step = per_step_rate(rd_annual, dt)
                cm = np.ones(P)
                dm = np.ones(P)
                if ta.crisis is not None and index_drawdown is not None:
                    in_crisis = index_drawdown[:, k] <= ta.crisis.drawdown_trigger  # start-of-step information
                    cm = np.where(in_crisis, ta.crisis.contribution_multiplier, 1.0)
                    dm = np.where(in_crisis, ta.crisis.distribution_multiplier, 1.0)
                if c.new_commitment_pacing_pa > 0:
                    U = U + c.new_commitment_pacing_pa * nav0[i] * dt
                C = np.minimum(rc_step * cm, 1.0) * U
                U = U - C
                D = np.minimum(rd_step * dm, 1.0) * nav_after_growth
            nav_t[:, k + 1, i] = nav_after_growth + C - D
            contr[:, k + 1, i] = C
            dist[:, k + 1, i] = D
            unf[:, k + 1, i] = U
            report_now = bool(is_month_end[k + 1]) and (month_index[k + 1] % c.reporting_months == 0)
            if report_now:
                r_true_period = i_true / i_true_last - 1.0
                r_obs = (1.0 - c.smoothing_phi) * r_true_period + c.smoothing_phi * r_obs_prev
                i_obs_last = i_obs_last * (1.0 + r_obs)
                i_true_last = i_true.copy()
                rrep[:, k + 1, i] = r_obs
                r_obs_prev = r_obs
            # reported mark = true NAV with the growth since the last report stripped and the smoothed
            # index applied: NAV_rep = NAV_true · I_obs,last / I_true  (flat between reports, jumps at reports)
            nav_r[:, k + 1, i] = nav_t[:, k + 1, i] * i_obs_last / i_true
    return IlliquidPaths(
        names=[c.name for c in cfgs], nav_true=nav_t, nav_reported=nav_r, contributions=contr,
        distributions=dist, unfunded=unf, true_return=np.concatenate([np.zeros((P, 1, n)), G], axis=1), reported_return=rrep,
    )


def geltner_smooth(r_true: F64, phi: float) -> F64:
    """Pure AR(1) smoothing of a return series along axis −1 (used by tests)."""
    out = np.empty_like(r_true)
    prev = np.zeros(r_true.shape[:-1])
    for k in range(r_true.shape[-1]):
        prev = (1 - phi) * r_true[..., k] + phi * prev
        out[..., k] = prev
    return out
