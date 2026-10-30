"""Mathematically Verified Target-Certification & Spectrally-Tightened Confidence-Bound Module.

Implements:
1. Exact non-asymptotic bounds on both worst-case support [0, 1] (R = 1.0) AND online-known
   spectrally-tightened support [1 - U_spec(G)/W, 1] of width R_spec(G) = U_spec(G)/W <= 1.0:
   - `hoeffding_ucb_valid` / `hoeffding_spectral_valid` (Hoeffding, JASA 1963, Theorem 2)
   - `empirical_bernstein_ucb_valid` / `eb_spectral_valid` (Maurer & Pontil, COLT 2009, Theorem 4)
   - `intersection_eb_hoeff_ucb_valid` / `intersection_spectral_valid` (Boole split alpha_k / 2)
   - `time_uniform_eb_cs_valid` / `time_uniform_eb_cs_spectral_valid` (Waudby-Smith & Ramdas,
     JRSSB 86(1):1-27, 2024, Proposition 2 / Eqs. 14-15)
2. Fixes the sequential horizon bug in `time_uniform_eb_cs_valid`:
   - Uses a strictly G_{i-1}-measurable (predictable) tuning sequence lambda_i that NEVER changes
     retroactively as sample size grows: either horizon-free `i * log(i + 1)` scaling when
     `n_horizon is None`, or a single F_{k-1}-measurable `fixed_horizon` set before sampling starts.
3. Explicitly separates diagnostic/heuristic rules (`student_t_heuristic`, `eb_under_range_r05_invalid`,
   `naive_insample_normal`) that lack finite-sample guarantees and are used in calibration stress tests.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.stats import t as student_t


@dataclass
class CertificationOutcome:
    certified: bool
    status: str  # 'certified' | 'not_certified' | 'inconclusive_budget_exhausted' | 'invalid_assumption'
    ucb_loss: float
    empirical_mean_loss: float
    empirical_var_loss: float
    empirical_se_loss: float
    confirmation_shots_used: int
    attempt_index: int
    alpha_spent: float
    rule_name: str
    is_non_asymptotic_guarantee: bool
    achieved_tolerance: Optional[float] = None  # max(0, ucb_loss - tau_checked)
    loss_range_used: float = 1.0


def compute_spectral_loss_support(
    spectral_upper_bound: Optional[float],
    total_weight: Optional[float],
) -> Tuple[float, float, float]:
    """Return online-computable deterministic support [loss_min, loss_max] and width R_spec.

    By Mohar & Poljak (Czechoslovak Math. J. 40(2):343-352, 1990), for any graph G=(V,E,w)
    with w_e >= 0 and total weight W > 0, every cut satisfies 0 <= C(z) <= U_spec(G) <= W.
    Hence every single-shot normalized loss Y_s = 1 - C(z_s)/W satisfies:
        1 - U_spec(G)/W <= Y_s <= 1.0,
    with deterministic support width R_spec(G) = U_spec(G) / W in (0, 1].
    """
    if (
        spectral_upper_bound is None
        or total_weight is None
        or total_weight <= 0.0
        or spectral_upper_bound <= 0.0
    ):
        return 0.0, 1.0, 1.0
    r_spec = float(min(1.0, max(1e-6, spectral_upper_bound / total_weight)))
    loss_min = float(max(0.0, 1.0 - r_spec))
    return loss_min, 1.0, r_spec


def allocate_alpha_k(
    alpha_total: float,
    attempt_idx: int,
    max_attempts: Optional[int] = 1,
    schedule: str = "fixed_bonferroni",
) -> float:
    """Compute valid per-attempt significance level alpha_k such that sum_k alpha_k <= alpha_total."""
    if alpha_total <= 0.0 or alpha_total >= 1.0:
        raise ValueError(f"alpha_total must lie in (0, 1), got {alpha_total}")
    if attempt_idx < 1:
        raise ValueError(f"attempt_idx must be >= 1, got {attempt_idx}")

    if schedule == "fixed_bonferroni":
        if max_attempts is None or max_attempts < 1:
            raise ValueError("fixed_bonferroni requires finite max_attempts >= 1")
        if attempt_idx > max_attempts:
            return 0.0
        return float(alpha_total) / float(max_attempts)
    elif schedule == "harmonic_spending":
        # Convergent schedule: sum_{k=1}^infty alpha / (k*(k+1)) = alpha
        if max_attempts is not None and attempt_idx > max_attempts:
            return 0.0
        return float(alpha_total) / float(attempt_idx * (attempt_idx + 1))
    raise ValueError(f"Unknown alpha-spending schedule: {schedule}")


def hoeffding_ucb_valid(
    samples: np.ndarray,
    alpha_k: float,
    loss_range: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Exact one-sided Hoeffding (JASA 1963, Theorem 2) UCB on support of width `loss_range`."""
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    if n < 1 or alpha_k <= 0.0:
        return 1.0, 1.0, 0.25, 0.5
    mean_val = float(np.mean(arr))
    var_val = float(np.var(arr, ddof=1)) if n > 1 else 0.25 * (loss_range ** 2)
    se_val = math.sqrt(var_val / n)
    rad = float(loss_range) * math.sqrt(math.log(1.0 / max(alpha_k, 1e-15)) / (2.0 * n))
    return min(1.0, mean_val + rad), mean_val, var_val, se_val


def empirical_bernstein_ucb_valid(
    samples: np.ndarray,
    alpha_k: float,
    loss_range: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Exact one-sided Maurer-Pontil (COLT 2009, Theorem 4) Empirical-Bernstein UCB on support of width `loss_range`."""
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    if n < 2 or alpha_k <= 0.0:
        return 1.0, (float(np.mean(arr)) if n == 1 else 1.0), 0.25 * (loss_range ** 2), 0.5 * loss_range
    mean_val = float(np.mean(arr))
    var_val = float(np.var(arr, ddof=1))
    se_val = math.sqrt(var_val / n)
    log_term = math.log(2.0 / max(alpha_k, 1e-15))
    rad = math.sqrt(2.0 * var_val * log_term / n) + (7.0 * float(loss_range) * log_term) / (3.0 * (n - 1))
    return min(1.0, mean_val + rad), mean_val, var_val, se_val


def intersection_eb_hoeff_ucb_valid(
    samples: np.ndarray,
    alpha_k: float,
    loss_range: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Valid intersection of Hoeffding and Empirical-Bernstein UCBs using split level alpha_k / 2."""
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    if n < 2 or alpha_k <= 0.0:
        return hoeffding_ucb_valid(arr, alpha_k, loss_range=loss_range)
    half_alpha = 0.5 * alpha_k
    u_h, mean_v, var_v, se_v = hoeffding_ucb_valid(arr, half_alpha, loss_range=loss_range)
    u_eb, _, _, _ = empirical_bernstein_ucb_valid(arr, half_alpha, loss_range=loss_range)
    return min(u_h, u_eb), mean_v, var_v, se_v


def time_uniform_eb_cs_valid(
    samples: np.ndarray,
    alpha_k: float,
    n_horizon: Optional[int] = None,
    loss_min: float = 0.0,
    loss_max: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Exact time-uniform predictable-plugin Empirical-Bernstein CS (Waudby-Smith & Ramdas, JRSSB 2024, Prop. 2).

    Mathematical Guarantees:
    1. Prefix Invariance & Predictability:
       - If `n_horizon is None` (default), uses the horizon-free predictable step-index scaling
         `eff_scale = i * log(i + 1)` (Waudby-Smith & Ramdas 2024, Eq. 15), so lambda_i is strictly
         G_{i-1}-measurable and invariant to any future sample size t > i.
       - If `n_horizon` is provided, it must be a fixed, predeclared F_{k-1}-measurable integer
         constant across all sequential evaluations of the stream.
    2. Spectrally-Tightened Bounded Support [loss_min, loss_max]:
       - Given known support [a, b] = [loss_min, loss_max] of width R = b - a > 0, transforms
         Z_i = clip((Y_i - a) / R, 0, 1) in [0, 1], runs the exact supermartingale on Z_1..Z_t,
         and maps the running-minimum upper confidence bound back via U_t = a + R * U_{Z,t}.
    """
    arr = np.asarray(samples, dtype=float)
    t = len(arr)
    r_width = max(1e-9, float(loss_max - loss_min))
    if t < 1 or alpha_k <= 0.0:
        return 1.0, 1.0, 0.25 * (r_width ** 2), 0.5 * r_width

    if n_horizon is not None and int(n_horizon) < 1:
        raise ValueError(f"n_horizon must be None or >= 1, got {n_horizon}")

    log_inv_alpha = math.log(1.0 / max(alpha_k, 1e-15))
    z = np.clip((arr - loss_min) / r_width, 0.0, 1.0)
    idx = np.arange(1.0, t + 1.0)
    csum_z = np.cumsum(z)
    mu_curr = csum_z / idx
    mu0 = min(1.0, max(0.0, (0.5 - loss_min) / r_width))
    mu_prev = np.empty(t, dtype=float)
    mu_prev[0] = mu0
    if t > 1:
        mu_prev[1:] = mu_curr[:-1]
    sq_dev_curr = (z - mu_curr) ** 2
    prev_sq_dev = np.empty(t, dtype=float)
    prev_sq_dev[0] = 0.0
    if t > 1:
        prev_sq_dev[1:] = np.cumsum(sq_dev_curr[:-1])
    sigma2_prev = (0.25 + prev_sq_dev) / idx
    if n_horizon is None:
        eff_scale = idx * np.log(idx + 1.0)
    else:
        eff_scale = float(n_horizon)
    lam = np.minimum(0.5, np.sqrt((2.0 * log_inv_alpha) / np.maximum(1e-12, eff_scale * sigma2_prev)))
    psi = (-np.log(1.0 - lam) - lam) / 4.0
    v = 4.0 * ((z - mu_prev) ** 2)
    ucb_z_seq = (log_inv_alpha + np.cumsum(v * psi) + np.cumsum(lam * z)) / np.cumsum(lam)
    best_ucb_z = float(np.min(ucb_z_seq))

    best_ucb_y = min(float(loss_max), float(loss_min) + r_width * best_ucb_z)
    mean_val = float(np.mean(arr))
    var_val = float(np.var(arr, ddof=1)) if t > 1 else 0.25 * (r_width ** 2)
    se_val = math.sqrt(var_val / t)
    return min(1.0, best_ucb_y), mean_val, var_val, se_val


def student_t_ucb_heuristic(
    samples: np.ndarray,
    alpha_k: float,
) -> Tuple[float, float, float, float]:
    """Asymptotic/heuristic Student-t UCB (explicitly non-guaranteed for skewed/small-n distributions).

    Refuses degenerate zero-variance samples (returns 1.0 if var_val < 1e-10) so constant
    prefixes on point-mass distributions do not trigger instant false certification.
    """
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    if n < 5 or alpha_k <= 0.0:
        return 1.0, (float(np.mean(arr)) if n > 0 else 1.0), 0.25, 0.5
    mean_val = float(np.mean(arr))
    var_val = float(np.var(arr, ddof=1))
    if var_val < 1e-10:
        return 1.0, mean_val, 0.0, 0.0
    se_val = math.sqrt(var_val / n)
    t_crit = float(student_t.ppf(1.0 - alpha_k, df=n - 1))
    ucb = mean_val + t_crit * se_val
    return min(1.0, ucb), mean_val, var_val, se_val


def theoretical_sufficient_shots_hoeffding(
    delta: float,
    alpha_k: float,
    power_target: float = 0.80,
    loss_range: float = 1.0,
) -> int:
    """Sufficient upper-bound sample size for Hoeffding certification to achieve power >= 1 - beta."""
    if delta <= 0.0:
        return 10**9
    beta = max(1e-12, 1.0 - power_target)
    num = (loss_range ** 2) * ((math.sqrt(math.log(1.0 / alpha_k)) + math.sqrt(math.log(1.0 / beta))) ** 2)
    return int(math.ceil(num / (2.0 * (delta ** 2))))


def theoretical_sufficient_shots_eb(
    delta: float,
    sigma2: float,
    alpha_k: float,
    power_target: float = 0.80,
    loss_range: float = 1.0,
) -> int:
    """Sufficient upper-bound sample size for Empirical-Bernstein certification to achieve power >= 1 - beta."""
    if delta <= 0.0:
        return 10**9
    beta = max(1e-12, 1.0 - power_target)
    log_a = math.log(2.0 / max(alpha_k, 1e-15))
    log_b = math.log(2.0 / beta)
    for n in range(10, 50000, 5):
        eb_rad = math.sqrt(2.0 * sigma2 * log_a / n) + (7.0 * loss_range * log_a) / (3.0 * (n - 1))
        mean_tail = math.sqrt(2.0 * sigma2 * log_b / n) + (2.0 * loss_range * log_b) / (3.0 * n)
        if eb_rad + mean_tail <= delta:
            return n
    return 50000


def evaluate_confirmation_sample(
    samples: np.ndarray,
    tau_threshold: float,
    alpha_k: float,
    attempt_idx: int = 1,
    rule: str = "intersection_eb_hoeff_valid",
    loss_range: float = 1.0,
    loss_min: Optional[float] = None,
    loss_max: float = 1.0,
    n_horizon: Optional[int] = None,
) -> CertificationOutcome:
    """Evaluate a confirmation sample against threshold `tau_threshold` at level `alpha_k`.

    Supports both worst-case [0, 1] rules (`*_valid`) and online spectrally-tightened rules
    (`*_spectral_valid` where `loss_range = U_spec(G)/W` and `loss_min = 1 - U_spec(G)/W`),
    plus diagnostic tight/invalid baselines (`student_t_heuristic`, `eb_under_range_r05_invalid`).
    """
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    eff_min = float(max(0.0, loss_max - loss_range)) if loss_min is None else float(loss_min)
    eff_range = float(loss_max - eff_min) if loss_min is not None else float(loss_range)

    if n < 2 or alpha_k <= 0.0:
        return CertificationOutcome(
            False, "inconclusive_budget_exhausted", 1.0, 1.0, 0.25, 0.5, n, attempt_idx, alpha_k, rule, True, None, eff_range
        )
    if np.any(arr < -1e-9) or np.any(arr > 1.0 + 1e-9):
        return CertificationOutcome(
            False, "invalid_assumption", 1.0, float(np.mean(arr)), 0.25, 0.5, n, attempt_idx, alpha_k, rule, False, None, eff_range
        )
    # Verify spectral support assumption when a tighter support [eff_min, loss_max] is used
    if np.any(arr < eff_min - 1e-7) or np.any(arr > loss_max + 1e-7):
        return CertificationOutcome(
            False, "invalid_assumption", 1.0, float(np.mean(arr)), 0.25, 0.5, n, attempt_idx, alpha_k, rule, False, None, eff_range
        )

    is_guaranteed = True
    if rule in ("hoeffding_valid", "hoeffding_spectral_valid"):
        ucb, m_v, v_v, se_v = hoeffding_ucb_valid(arr, alpha_k, loss_range=eff_range)
    elif rule in ("empirical_bernstein_valid", "split_eb_confirmation", "eb_spectral_valid"):
        ucb, m_v, v_v, se_v = empirical_bernstein_ucb_valid(arr, alpha_k, loss_range=eff_range)
    elif rule in ("intersection_eb_hoeff_valid", "intersection_spectral_valid"):
        ucb, m_v, v_v, se_v = intersection_eb_hoeff_ucb_valid(arr, alpha_k, loss_range=eff_range)
    elif rule in ("time_uniform_eb_cs_valid", "time_uniform_eb_cs", "time_uniform_eb_cs_spectral_valid"):
        ucb, m_v, v_v, se_v = time_uniform_eb_cs_valid(
            arr, alpha_k, n_horizon=n_horizon, loss_min=eff_min, loss_max=loss_max
        )
    elif rule == "eb_under_range_r05_invalid":
        # Deliberately tight/under-scaled bound (scales EB variance & range by 0.45) to verify calibration sensitivity
        _, m_v, v_v, se_v = empirical_bernstein_ucb_valid(arr, alpha_k, loss_range=0.45)
        log_t = math.log(2.0 / max(alpha_k, 1e-15))
        rad_tight = 0.45 * math.sqrt(2.0 * v_v * log_t / n) + (7.0 * 0.15 * log_t) / (3.0 * max(1, n - 1))
        ucb = min(1.0, m_v + rad_tight)
        is_guaranteed = False
        eff_range = 0.45
    elif rule in ("student_t_heuristic", "split_student_t_holm"):
        ucb, m_v, v_v, se_v = student_t_ucb_heuristic(arr, alpha_k)
        is_guaranteed = False
    else:
        raise ValueError(f"Unknown certification rule: {rule}")

    certified = bool(ucb <= tau_threshold)
    status = "certified" if certified else "not_certified"
    ach_tol = max(0.0, float(ucb - tau_threshold))
    return CertificationOutcome(
        certified=certified,
        status=status,
        ucb_loss=ucb,
        empirical_mean_loss=m_v,
        empirical_var_loss=v_v,
        empirical_se_loss=se_v,
        confirmation_shots_used=n,
        attempt_index=attempt_idx,
        alpha_spent=alpha_k,
        rule_name=rule,
        is_non_asymptotic_guarantee=is_guaranteed,
        achieved_tolerance=ach_tol,
        loss_range_used=eff_range,
    )


def certify_candidate_split_sample(
    oracle,
    theta: np.ndarray,
    target_loss: Optional[float],
    alpha_total: float = 0.05,
    attempt_idx: int = 1,
    max_attempts: int = 2,
    conf_shots: int = 150,
    rule: str = "intersection_eb_hoeff_valid",
    alpha_schedule: str = "fixed_bonferroni",
    indifference_delta: float = 0.0,
    use_spectral_support: bool = False,
) -> CertificationOutcome:
    """Draw a fresh confirmation batch from `oracle.confirmation_loss_samples` and test target attainment."""
    if target_loss is None:
        return CertificationOutcome(
            False, "not_certified", 1.0, 1.0, 0.25, 0.5, 0, attempt_idx, 0.0, rule, True, None, 1.0
        )
    alpha_k = allocate_alpha_k(alpha_total, attempt_idx, max_attempts=max_attempts, schedule=alpha_schedule)
    if alpha_k <= 0.0:
        return CertificationOutcome(
            False, "not_certified", 1.0, 1.0, 0.25, 0.5, 0, attempt_idx, 0.0, rule, True, None, 1.0
        )

    use_spec = use_spectral_support or ("spectral" in rule)
    if use_spec and hasattr(oracle, "engine"):
        spec_ub = getattr(oracle.engine, "spectral_upper_bound", None)
        tot_w = getattr(oracle.engine, "total_weight", None)
        loss_min, loss_max, loss_range = compute_spectral_loss_support(spec_ub, tot_w)
    else:
        loss_min, loss_max, loss_range = 0.0, 1.0, 1.0

    tau_checked = float(target_loss) + float(indifference_delta)
    avail = oracle.remaining_shots()

    if rule in ("time_uniform_eb_cs_valid", "time_uniform_eb_cs", "time_uniform_eb_cs_spectral_valid"):
        batch_step = 25
        if avail < batch_step:
            return CertificationOutcome(
                False, "inconclusive_budget_exhausted", 1.0, 1.0, 0.25, 0.5, 0, attempt_idx, alpha_k, rule, True, None, loss_range
            )
        all_samples: List[float] = []
        used = 0
        max_s = min(conf_shots, avail)
        # Predeclare F_{k-1}-measurable fixed horizon ONCE before sequential loop begins (or None for horizon-free):
        fixed_horizon = max(max_s, 50)
        while used + batch_step <= max_s and oracle.remaining_shots() >= batch_step:
            b = oracle.confirmation_loss_samples(theta, batch_step)
            if len(b) == 0:
                break
            all_samples.extend(float(x) for x in b)
            used += len(b)
            out = evaluate_confirmation_sample(
                np.asarray(all_samples),
                tau_checked,
                alpha_k,
                attempt_idx=attempt_idx,
                rule=rule,
                loss_range=loss_range,
                loss_min=loss_min,
                loss_max=loss_max,
                n_horizon=fixed_horizon,
            )
            if out.certified:
                return out
        if len(all_samples) < 2:
            return CertificationOutcome(
                False, "inconclusive_budget_exhausted", 1.0, 1.0, 0.25, 0.5, used, attempt_idx, alpha_k, rule, True, None, loss_range
            )
        return evaluate_confirmation_sample(
            np.asarray(all_samples),
            tau_checked,
            alpha_k,
            attempt_idx=attempt_idx,
            rule=rule,
            loss_range=loss_range,
            loss_min=loss_min,
            loss_max=loss_max,
            n_horizon=fixed_horizon,
        )

    if avail < min(20, conf_shots):
        return CertificationOutcome(
            False, "inconclusive_budget_exhausted", 1.0, 1.0, 0.25, 0.5, 0, attempt_idx, alpha_k, rule, True, None, loss_range
        )
    s_draw = min(conf_shots, avail)
    samples = oracle.confirmation_loss_samples(theta, s_draw)
    if len(samples) < 5:
        return CertificationOutcome(
            False, "inconclusive_budget_exhausted", 1.0, 1.0, 0.25, 0.5, len(samples), attempt_idx, alpha_k, rule, True, None, loss_range
        )
    return evaluate_confirmation_sample(
        samples,
        tau_checked,
        alpha_k,
        attempt_idx=attempt_idx,
        rule=rule,
        loss_range=loss_range,
        loss_min=loss_min,
        loss_max=loss_max,
    )
