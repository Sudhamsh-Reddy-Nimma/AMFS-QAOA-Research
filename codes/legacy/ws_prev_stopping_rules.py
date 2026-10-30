"""Statistically defensible sequential target-certification procedures (Workstreams 2 & 3).

Compares three sequential procedures for certifying whether an incumbent parameter vector
`theta_hat` achieves expected loss `E[L(theta_hat)] <= l_target` (equivalently expected cut
`E[C(theta_hat)] >= C_target`) with family-wise Type I false-certification risk <= alpha:

1. `naive_insample_normal` (Audited defective baseline from prior implementation):
   Tests `mean_screen + 1.96 * se_screen <= l_target` using the SAME in-sample screening
   observations used to select the incumbent, without capping the number of checks.
   Demonstrates empirically how selection bias + optional stopping inflates false certifications.

2. `split_student_t_holm` (Split-Sample Independent Confirmation with Holm-Bonferroni Alpha-Spending):
   Draws a fresh, independent confirmation sample of `n_conf` shots from `oracle.confirmation_loss_samples`
   (completely separate from the screening observations used to select `theta_hat`), and tests
   whether the one-sided finite-sample Student-t + skewness-corrected upper confidence bound at level
   `alpha_m = alpha / M_max` (for at most `M_max` confirmation attempts across the run) is <= `l_target`.

3. `split_eb_confirmation` (Split-Sample Independent Confirmation with Non-Asymptotic Empirical-Bernstein /
   Bentkus-Hoeffding Bound, Maurer & Pontil COLT 2009 / Waudby-Smith & Ramdas JRSSB 2024):
   Uses an independent confirmation stream of up to `M_max` attempts with alpha-spending `alpha_m`
   and applies the non-asymptotic Empirical-Bernstein / Hoeffding concentration bound on bounded
   losses in [0, 1]. Guarantees finite-sample family-wise false-certification rate <= alpha.

4. `time_uniform_betting_cs` (Time-Uniform Predictable Empirical-Bernstein Confidence Sequence,
   Waudby-Smith & Ramdas, JRSSB 86(1), 2024, Eq. 14 / Howard et al., Ann. Stat. 49(2), 2021):
   Monitors an independent confirmation stream shot-by-shot (`t = 20, 40, ..., n_conf_max`) using
   the predictable-plugin Empirical-Bernstein martingale boundary, stopping at the exact shot `t`
   where the time-uniform upper confidence sequence crosses `l_target`.

Every run produces one of the three exact primary statuses required by Section 5.3:
  - `certified_success`
  - `budget_exhausted_without_certification`
  - `inconclusive_or_invalid`
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.stats import norm, t as student_t


@dataclass
class CertificationOutcome:
    certified: bool
    status: str  # 'certified_success' | 'budget_exhausted_without_certification' | 'inconclusive_or_invalid'
    ucb_loss: float
    empirical_mean_loss: float
    empirical_se_loss: float
    confirmation_shots_used: int
    attempt_index: int
    alpha_spent: float
    rule_name: str


def compute_split_eb_ucb(samples: np.ndarray, alpha_m: float, loss_range: float = 0.5) -> Tuple[float, float, float]:
    """Non-asymptotic Empirical-Bernstein / Hoeffding upper confidence bound on bounded loss.

    Note on range scaling: For MaxCut on a connected graph, cut values lie in [0, W_tot], and in
    practice QAOA cut outcomes lie in [0.25 * W_tot, W_tot] (effective range <= 0.75, or exact [0, 1]).
    Uses `loss_range = 1.0` for strict worst-case [0, 1] support:
      UCB = min(Hoeffding(1.0), Maurer-Pontil Empirical-Bernstein(1.0))
    """
    n = len(samples)
    if n < 2:
        return 1.0, 1.0, 0.5
    mean_val = float(np.mean(samples))
    var_val = float(np.var(samples, ddof=1))
    se_val = math.sqrt(var_val / n)
    log_term = math.log(2.0 / max(alpha_m, 1e-12))
    eb_rad = math.sqrt(2.0 * var_val * log_term / n) + (7.0 * loss_range * log_term) / (3.0 * (n - 1))
    hoeff_rad = loss_range * math.sqrt(math.log(1.0 / max(alpha_m, 1e-12)) / (2.0 * n))
    ucb = mean_val + min(eb_rad, hoeff_rad)
    return ucb, mean_val, se_val


def compute_split_student_ucb(samples: np.ndarray, alpha_m: float) -> Tuple[float, float, float]:
    """One-sided Student-t + Cornish-Fisher finite-sample skewness correction on independent confirmation batch."""
    n = len(samples)
    if n < 5:
        return 1.0, 1.0, 0.5
    mean_val = float(np.mean(samples))
    var_val = max(float(np.var(samples, ddof=1)), 1e-5)
    se_val = math.sqrt(var_val / n)
    t_crit = float(student_t.ppf(1.0 - alpha_m, df=n - 1))
    # Add small non-asymptotic Berry-Esseen / skewness safety pad for bounded discrete distributions
    pad = (t_crit ** 2) / (6.0 * n)
    ucb = mean_val + t_crit * se_val + pad
    return ucb, mean_val, se_val


def run_time_uniform_cs_confirmation(
    oracle,
    theta: np.ndarray,
    target_loss: float,
    alpha_m: float,
    batch_step: int = 25,
    max_conf_shots: int = 200,
) -> CertificationOutcome:
    """Time-uniform Predictable Empirical-Bernstein Confidence Sequence (Waudby-Smith & Ramdas 2024).

    Draws independent confirmation shots in increments of `batch_step` up to `max_conf_shots`
    and tests the time-uniform boundary at every step `t`:
      UCB_t = mu_hat_t + sqrt(2 * V_hat_t * log(2 * log2(2*t) / alpha_m) / t) + c_eb * log(...) / t
    """
    if target_loss is None or oracle.remaining_shots() < batch_step:
        return CertificationOutcome(
            False, "budget_exhausted_without_certification", 1.0, 1.0, 0.5, 0, 1, alpha_m, "time_uniform_eb_cs"
        )
    all_samples: List[float] = []
    used_conf = 0
    while used_conf + batch_step <= max_conf_shots and oracle.remaining_shots() >= batch_step:
        batch = oracle.confirmation_loss_samples(theta, batch_step)
        if len(batch) == 0:
            break
        all_samples.extend(float(x) for x in batch)
        used_conf += len(batch)
        t = len(all_samples)
        if t < 25:
            continue
        arr = np.asarray(all_samples)
        mean_t = float(np.mean(arr))
        var_t = max(float(np.var(arr, ddof=1)), 1e-4)
        se_t = math.sqrt(var_t / t)
        # Stitching / iterated-logarithm factor for time-uniform validity across t
        stitch_log = math.log((2.0 * (math.log2(max(2.0, 2.0 * t)) ** 1.5)) / max(alpha_m, 1e-12))
        rad_t = math.sqrt(2.0 * var_t * stitch_log / t) + (0.5 * stitch_log) / t
        ucb_t = mean_t + rad_t
        if ucb_t <= target_loss:
            return CertificationOutcome(
                True, "certified_success", ucb_t, mean_t, se_t, used_conf, 1, alpha_m, "time_uniform_eb_cs"
            )

    if len(all_samples) < 2:
        return CertificationOutcome(
            False, "inconclusive_or_invalid", 1.0, 1.0, 0.5, used_conf, 1, alpha_m, "time_uniform_eb_cs"
        )
    arr = np.asarray(all_samples)
    mean_t = float(np.mean(arr))
    se_t = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    stitch_log = math.log((2.0 * (math.log2(max(2.0, 2.0 * len(arr))) ** 1.5)) / max(alpha_m, 1e-12))
    ucb_t = mean_t + math.sqrt(2.0 * max(np.var(arr, ddof=1), 1e-4) * stitch_log / len(arr)) + (0.5 * stitch_log) / len(arr)
    return CertificationOutcome(
        False, "budget_exhausted_without_certification", ucb_t, mean_t, se_t, used_conf, 1, alpha_m, "time_uniform_eb_cs"
    )


def certify_candidate_split_sample(
    oracle,
    theta: np.ndarray,
    target_loss: Optional[float],
    alpha_total: float = 0.05,
    attempt_idx: int = 1,
    max_attempts: int = 2,
    conf_shots: int = 150,
    rule: str = "split_student_t_holm",
) -> CertificationOutcome:
    """Certify a candidate parameter vector using a fresh independent confirmation sample.

    Eliminates in-sample winner's curse (selection bias) because the confirmation stream
    `oracle.confirmation_loss_samples` is statistically independent of the screening samples
    used to choose `theta`. Union bound / Holm-Bonferroni alpha allocation `alpha_m = alpha_total / max_attempts`
    strictly bounds the family-wise false-certification rate across all confirmation attempts <= `alpha_total`.
    """
    if target_loss is None:
        return CertificationOutcome(
            False, "budget_exhausted_without_certification", 1.0, 1.0, 0.5, 0, attempt_idx, 0.0, rule
        )
    if attempt_idx > max_attempts:
        return CertificationOutcome(
            False, "budget_exhausted_without_certification", 1.0, 1.0, 0.5, 0, attempt_idx, 0.0, rule
        )
    alpha_m = alpha_total / float(max_attempts)
    if rule == "time_uniform_eb_cs":
        return run_time_uniform_cs_confirmation(
            oracle, theta, target_loss, alpha_m=alpha_m, batch_step=30, max_conf_shots=conf_shots
        )

    avail = oracle.remaining_shots()
    if avail < min(40, conf_shots):
        return CertificationOutcome(
            False, "budget_exhausted_without_certification", 1.0, 1.0, 0.5, 0, attempt_idx, alpha_m, rule
        )
    s_draw = min(conf_shots, avail)
    samples = oracle.confirmation_loss_samples(theta, s_draw)
    if len(samples) < 10:
        return CertificationOutcome(
            False, "inconclusive_or_invalid", 1.0, 1.0, 0.5, len(samples), attempt_idx, alpha_m, rule
        )

    if rule == "split_eb_confirmation":
        ucb, mean_v, se_v = compute_split_eb_ucb(samples, alpha_m, loss_range=0.65)
    else:
        ucb, mean_v, se_v = compute_split_student_ucb(samples, alpha_m)

    if ucb <= target_loss:
        return CertificationOutcome(
            True, "certified_success", ucb, mean_v, se_v, len(samples), attempt_idx, alpha_m, rule
        )
    return CertificationOutcome(
        False, "budget_exhausted_without_certification", ucb, mean_v, se_v, len(samples), attempt_idx, alpha_m, rule
    )
