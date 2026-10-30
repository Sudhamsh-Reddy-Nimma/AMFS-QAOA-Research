"""Vectorized (row-wise) mirrors of the certification rules in `stopping_rules.py`.

Purpose
-------
Phase 7 needs millions of rule evaluations (many replications x many candidate pools).  Calling
`evaluate_confirmation_sample` one sample at a time is too slow, so this module re-implements the
same *formulas* on a matrix `Y` of shape (R, n): each row is one independent confirmation sample.

Nothing here is a new statistical method.  Every function is a line-for-line mirror of the scalar
implementation in `stopping_rules.py`, and `codes/tests/test_phase7_batch_equivalence.py` asserts that
the two agree to 1e-9 on random data for every rule.  If you change `stopping_rules.py`, that test must
keep passing.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.stats import t as student_t


def _hoeffding(Y: np.ndarray, alpha: float, rw: float) -> np.ndarray:
    n = Y.shape[1]
    return np.minimum(1.0, Y.mean(axis=1) + rw * math.sqrt(math.log(1.0 / max(alpha, 1e-15)) / (2.0 * n)))


def _eb(Y: np.ndarray, alpha: float, rw: float) -> np.ndarray:
    n = Y.shape[1]
    if n < 2:
        return np.ones(Y.shape[0])
    var = Y.var(axis=1, ddof=1)
    lt = math.log(2.0 / max(alpha, 1e-15))
    rad = np.sqrt(2.0 * var * lt / n) + 7.0 * rw * lt / (3.0 * (n - 1))
    return np.minimum(1.0, Y.mean(axis=1) + rad)


def _intersection(Y: np.ndarray, alpha: float, rw: float) -> np.ndarray:
    return np.minimum(_hoeffding(Y, 0.5 * alpha, rw), _eb(Y, 0.5 * alpha, rw))


def _time_uniform(
    Y: np.ndarray,
    alpha: float,
    loss_min: float,
    loss_max: float = 1.0,
    n_horizon: Optional[int] = None,
) -> np.ndarray:
    """Row-wise running-minimum predictable-plug-in EB confidence sequence (see stopping_rules.py)."""
    R, t = Y.shape
    rw = max(1e-9, float(loss_max - loss_min))
    log_inv = math.log(1.0 / max(alpha, 1e-15))
    z = np.clip((Y - loss_min) / rw, 0.0, 1.0)
    idx = np.arange(1.0, t + 1.0)[None, :]
    csum = np.cumsum(z, axis=1)
    mu_curr = csum / idx
    mu0 = min(1.0, max(0.0, (0.5 - loss_min) / rw))
    mu_prev = np.empty_like(z)
    mu_prev[:, 0] = mu0
    if t > 1:
        mu_prev[:, 1:] = mu_curr[:, :-1]
    sq = (z - mu_curr) ** 2
    prev_sq = np.zeros_like(z)
    if t > 1:
        prev_sq[:, 1:] = np.cumsum(sq[:, :-1], axis=1)
    sigma2_prev = (0.25 + prev_sq) / idx
    eff = idx * np.log(idx + 1.0) if n_horizon is None else float(n_horizon)
    lam = np.minimum(0.5, np.sqrt((2.0 * log_inv) / np.maximum(1e-12, eff * sigma2_prev)))
    psi = (-np.log(1.0 - lam) - lam) / 4.0
    v = 4.0 * (z - mu_prev) ** 2
    ucb_z = (log_inv + np.cumsum(v * psi, axis=1) + np.cumsum(lam * z, axis=1)) / np.cumsum(lam, axis=1)
    best = ucb_z.min(axis=1)
    return np.minimum(1.0, np.minimum(float(loss_max), float(loss_min) + rw * best))


def _student_t(Y: np.ndarray, alpha: float) -> np.ndarray:
    n = Y.shape[1]
    if n < 5:
        return np.ones(Y.shape[0])
    var = Y.var(axis=1, ddof=1)
    se = np.sqrt(var / n)
    crit = float(student_t.ppf(1.0 - alpha, df=n - 1))
    out = np.minimum(1.0, Y.mean(axis=1) + crit * se)
    return np.where(var < 1e-10, 1.0, out)


def _eb_under_range(Y: np.ndarray, alpha: float) -> np.ndarray:
    n = Y.shape[1]
    var = Y.var(axis=1, ddof=1)
    lt = math.log(2.0 / max(alpha, 1e-15))
    rad = 0.45 * np.sqrt(2.0 * var * lt / n) + (7.0 * 0.15 * lt) / (3.0 * max(1, n - 1))
    return np.minimum(1.0, Y.mean(axis=1) + rad)


# Rules that do NOT use the spectral support (worst-case range [0, 1])
RULES_R1 = {
    "hoeffding_valid", "empirical_bernstein_valid", "intersection_eb_hoeff_valid", "time_uniform_eb_cs_valid",
}
# Rules that use the spectral support [loss_min, 1]
RULES_SPECTRAL = {
    "hoeffding_spectral_valid", "eb_spectral_valid", "intersection_spectral_valid", "time_uniform_eb_cs_spectral_valid",
}
# Heuristic / deliberately invalid diagnostics (no finite-sample guarantee)
RULES_DIAGNOSTIC = {"student_t_heuristic", "eb_under_range_r05_invalid"}


def ucb_matrix(
    Y: np.ndarray,
    rule: str,
    alpha: float,
    loss_min: float = 0.0,
    n_horizon: Optional[int] = None,
) -> np.ndarray:
    """Upper confidence bound for every row of `Y` under `rule` (all support ends at 1.0)."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be 2-D (replications x samples)")
    if rule in RULES_R1:
        lo, rw = 0.0, 1.0
    elif rule in RULES_SPECTRAL:
        lo, rw = float(loss_min), 1.0 - float(loss_min)
    else:
        lo, rw = 0.0, 1.0

    if rule in ("hoeffding_valid", "hoeffding_spectral_valid"):
        return _hoeffding(Y, alpha, rw)
    if rule in ("empirical_bernstein_valid", "eb_spectral_valid"):
        return _eb(Y, alpha, rw)
    if rule in ("intersection_eb_hoeff_valid", "intersection_spectral_valid"):
        return _intersection(Y, alpha, rw)
    if rule in ("time_uniform_eb_cs_valid", "time_uniform_eb_cs_spectral_valid"):
        return _time_uniform(Y, alpha, lo, 1.0, n_horizon)
    if rule == "student_t_heuristic":
        return _student_t(Y, alpha)
    if rule == "eb_under_range_r05_invalid":
        return _eb_under_range(Y, alpha)
    raise ValueError(f"Unknown rule {rule}")


def certify_matrix(
    Y: np.ndarray,
    tau: float,
    rule: str,
    alpha: float,
    loss_min: float = 0.0,
    n_horizon: Optional[int] = None,
) -> np.ndarray:
    """Boolean certification decision (UCB <= tau) for each row."""
    return ucb_matrix(Y, rule, alpha, loss_min, n_horizon) <= tau
