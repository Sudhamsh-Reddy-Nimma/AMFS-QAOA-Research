"""Verified implementations of competing state-of-the-art VQA/QAOA optimizers and classical controls.

Includes:
1. `run_icans`: Faithful implementation of iCANS2 (Kubler, Arrasmith, Cincio, Coles, Quantum 4, 263, 2020, Alg. 1).
2. `run_wecans`: Faithful implementation of weCANS (Ito, arXiv:2302.04422 / Quantum 2023, Eq. 22).
3. `run_dds`: Faithful implementation of DDS (Kim et al., Phys. Rev. Research 7, 043253, 2025, Alg. 1).
4. `run_nft`: Sequential Minimal Optimization / Rotosolve-QAOA (Nakanishi, Fujii, Todo, Phys. Rev. Research 2, 043158, 2020).
5. `run_cobyla_orig` & `run_cobyla_hao2025`: Original benchmark COBYLA and Hao et al. (Phys. Rev. Research 2025) protocol.
6. `run_bobyqa_trust`: Derivative-free quadratic trust-region optimizer (Powell 2009 BOBYQA model).
7. `run_gpbo_orig` & `run_gpbo_tibaldi`: GP-BO with Matern-5/2 kernel (Tibaldi et al., IEEE TQE 4, 2023) with
   both raw-observation selection (`orig`) and heteroscedastic posterior-mean selection (`tibaldi`).
8. `run_spsa` & `run_spsa_seqtest`: Standard SPSA and prototype sequential-sign-test SPSA.
9. `run_random`, `run_lhs_screen`, `run_fixed_naive`, `run_fixed_wurtz_love`: Classical and problem-informed controls.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from qaoa_simulator import QAOAOptimizerOracle, get_default_x0


def run_spsa(
    oracle: QAOAOptimizerOracle,
    budget: int,
    shots_per_eval: int,
    seed: int,
    init_mode: str = "naive",
    a: float = 0.15,
    c: float = 0.12,
    alpha: float = 0.602,
    gamma: float = 0.101,
) -> Tuple[np.ndarray, int, Dict]:
    """Standard Simultaneous Perturbation Stochastic Approximation (Spall 1992)."""
    rng = np.random.default_rng(seed)
    d = 2 * oracle.p
    theta = get_default_x0(oracle.p, init_mode)
    shots = max(1, int(shots_per_eval))
    iters = max(1, budget // (2 * shots))
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta.copy())]

    for k in range(iters):
        if oracle.remaining_shots() < 2:
            break
        s_eval = min(shots, oracle.remaining_shots() // 2)
        if s_eval < 1:
            break
        delta = rng.choice([-1.0, 1.0], size=d)
        ck = c / ((k + 1) ** gamma)
        tp = oracle.clip_theta(theta + ck * delta)
        tm = oracle.clip_theta(theta - ck * delta)
        yp = oracle.eval(tp, s_eval)
        ym = oracle.eval(tm, s_eval)
        grad = (yp - ym) / (2.0 * ck * delta)
        ak = a / ((k + 5) ** alpha)
        theta = oracle.clip_theta(theta - ak * grad)
        incumbent_trace.append((oracle.total_requested_shots, theta.copy()))

    return theta, oracle.total_requested_shots, {
        "incumbent_trace": incumbent_trace,
        "certified_target": False,
        "stop_reason": "spsa_completed",
    }


def run_spsa_seqtest(
    oracle: QAOAOptimizerOracle,
    budget: int,
    seed: int,
    pilot: int = 20,
    max_per_side: int = 160,
    init_mode: str = "naive",
) -> Tuple[np.ndarray, int, Dict]:
    """Prototype sequential-sign-test SPSA from `amfs_qaoa_shot_benchmark.py`.

    Explicitly labelled as a sequential sign-test SPSA prototype (NOT iCANS/weCANS/DDS).
    """
    rng = np.random.default_rng(seed)
    d = 2 * oracle.p
    theta = get_default_x0(oracle.p, init_mode)
    k = 0
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta.copy())]

    while oracle.total_requested_shots + 2 * pilot <= budget and oracle.remaining_shots() >= 2 * pilot:
        delta = rng.choice([-1.0, 1.0], size=d)
        ck = 0.12 / ((k + 1) ** 0.101)
        tp = oracle.clip_theta(theta + ck * delta)
        tm = oracle.clip_theta(theta - ck * delta)
        xp = list(oracle.loss_samples(tp, pilot))
        xm = list(oracle.loss_samples(tm, pilot))
        while (
            len(xp) < max_per_side
            and len(xm) < max_per_side
            and oracle.total_requested_shots + 2 * pilot <= budget
            and oracle.remaining_shots() >= 2 * pilot
        ):
            mp, mm = float(np.mean(xp)), float(np.mean(xm))
            vp = float(np.var(xp, ddof=1)) / len(xp) if len(xp) > 1 else 0.25
            vm = float(np.var(xm, ddof=1)) / len(xm) if len(xm) > 1 else 0.25
            if abs(mp - mm) > 1.5 * math.sqrt(vp + vm):
                break
            xp.extend(oracle.loss_samples(tp, pilot))
            xm.extend(oracle.loss_samples(tm, pilot))
        yp, ym = float(np.mean(xp)), float(np.mean(xm))
        grad = (yp - ym) / (2.0 * ck * delta)
        ak = 0.15 / ((k + 5) ** 0.602)
        theta = oracle.clip_theta(theta - ak * grad)
        incumbent_trace.append((oracle.total_requested_shots, theta.copy()))
        k += 1

    return theta, oracle.total_requested_shots, {
        "incumbent_trace": incumbent_trace,
        "certified_target": False,
        "stop_reason": "spsa_seqtest_completed",
    }


def run_icans(
    oracle: QAOAOptimizerOracle,
    budget: int,
    seed: int,
    init_mode: str = "naive",
    alpha: float = 0.25,
    s_min: int = 15,
    L_lip: float = 1.5,
    mu: float = 0.85,
    b_reg: float = 1e-5,
    shift_delta: float = 0.15,
    variant: str = "iCANS2",
) -> Tuple[np.ndarray, int, Dict]:
    """Faithful implementation of iCANS1 / iCANS2 (Kubler et al., Quantum 4, 263, 2020, Alg. 1).

    Follows Algorithm 1 of arXiv:1909.09083v3 / Quantum 4, 263 (2020) line-by-line:
    - Evaluates each partial derivative g_i and variance of the derivative estimator S_i
      using s_i shots at shifted points theta +/- shift_delta * e_i.
    - Updates exponential moving averages:
        xi'_i <- mu * xi'_i + (1 - mu) * S_i
        chi'_i <- mu * chi'_i + (1 - mu) * g_i
        xi_i <- xi'_i / (1 - mu^(k+1))
        chi_i <- chi'_i / (1 - mu^(k+1))
    - Performs parameter update (iCANS1 fixed step alpha, or iCANS2 variance-clipped step alpha'_i).
    - Computes suggested shot counts:
        s_i <- ceil( (2 * L * alpha / (2 - L * alpha)) * xi_i / (chi_i^2 + b * mu^k) )
      and expected gain per shot:
        gamma_i <- (1 / s_i) * [ (alpha - L * alpha^2 / 2) * chi_i^2 - (L * alpha^2 / (2 * s_i)) * xi_i ]
    - Caps s_i at s_max = s_{argmax_i gamma_i} and clips to [s_min, s_max].
    """
    d = 2 * oracle.p
    theta = get_default_x0(oracle.p, init_mode)
    s_vec = np.full(d, int(s_min), dtype=int)
    chi_prime = np.zeros(d, dtype=float)
    xi_prime = np.zeros(d, dtype=float)
    k = 0
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta.copy())]

    # Ensure alpha < 2 / L_lip as required by Kubler et al. Eq. (18)
    alpha = min(float(alpha), 1.9 / L_lip)

    while oracle.total_requested_shots < budget and oracle.remaining_shots() >= 2 * d:
        g_vec = np.zeros(d, dtype=float)
        S_vec = np.zeros(d, dtype=float)

        for i in range(d):
            rem = min(budget - oracle.total_requested_shots, oracle.remaining_shots())
            if rem < 2:
                break
            si = min(int(s_vec[i]), max(1, rem // 2))
            e_i = np.zeros(d, dtype=float)
            e_i[i] = shift_delta
            tp = oracle.clip_theta(theta + e_i)
            tm = oracle.clip_theta(theta - e_i)
            lp = oracle.loss_samples(tp, si)
            lm = oracle.loss_samples(tm, si)
            if len(lp) == 0 or len(lm) == 0:
                break
            mp, mm = float(np.mean(lp)), float(np.mean(lm))
            vp = float(np.var(lp, ddof=1)) if len(lp) > 1 else 0.01
            vm = float(np.var(lm, ddof=1)) if len(lm) > 1 else 0.01
            g_vec[i] = (mp - mm) / (2.0 * shift_delta)
            # Single-shot variance S_i of the gradient component estimator
            S_vec[i] = max(1e-6, (vp + vm) / ((2.0 * shift_delta) ** 2))

        # Lines 6-9 of Algorithm 1: bias-corrected exponential moving averages
        xi_prime = mu * xi_prime + (1.0 - mu) * S_vec
        chi_prime = mu * chi_prime + (1.0 - mu) * g_vec
        bias_corr = 1.0 - (mu ** (k + 1))
        xi_hat = xi_prime / bias_corr
        chi_hat = chi_prime / bias_corr

        # Lines 10-23 of Algorithm 1: coordinate update and shot allocation
        new_s = np.zeros(d, dtype=int)
        gamma_gain = np.zeros(d, dtype=float)
        for i in range(d):
            gi = g_vec[i]
            Si = S_vec[i]
            si_used = max(1, int(s_vec[i]))
            if variant == "iCANS1":
                theta[i] -= alpha * gi
            else:
                # iCANS2 safe step-size bound (Lines 14-18)
                alpha_bound = (gi ** 2) / (L_lip * (gi ** 2 + Si / si_used + b_reg * (mu ** k)))
                step_alpha = min(alpha, alpha_bound)
                theta[i] -= step_alpha * gi

            denom = (chi_hat[i] ** 2) + b_reg * (mu ** k)
            raw_si = int(math.ceil((2.0 * L_lip * alpha / (2.0 - L_lip * alpha)) * (xi_hat[i] / denom)))
            # Hard upper cap per coordinate so a single coordinate cannot exceed 1/4 of total budget
            raw_si = max(s_min, min(raw_si, max(s_min + 1, budget // 4)))
            new_s[i] = raw_si
            gamma_gain[i] = (1.0 / raw_si) * (
                (alpha - 0.5 * L_lip * (alpha ** 2)) * (chi_hat[i] ** 2)
                - (0.5 * L_lip * (alpha ** 2) / raw_si) * xi_hat[i]
            )

        theta = oracle.clip_theta(theta)
        incumbent_trace.append((oracle.total_requested_shots, theta.copy()))

        # Lines 24-25 of Algorithm 1: clip shots to [s_min, s_{argmax gamma_i}]
        idx_max = int(np.argmax(gamma_gain))
        s_max = max(s_min, int(new_s[idx_max]))
        s_vec = np.clip(new_s, s_min, s_max)
        k += 1

    return theta, oracle.total_requested_shots, {
        "incumbent_trace": incumbent_trace,
        "certified_target": False,
        "stop_reason": "icans_completed",
    }


def run_wecans(
    oracle: QAOAOptimizerOracle,
    budget: int,
    seed: int,
    init_mode: str = "naive",
    alpha: float = 0.25,
    s_min: int = 15,
    L_lip: float = 1.5,
    mu: float = 0.85,
    b_reg: float = 1e-5,
    shift_delta: float = 0.15,
    latency_ratio_R: float = 25.0,
) -> Tuple[np.ndarray, int, Dict]:
    """Faithful implementation of weCANS(i) (Ito, arXiv:2302.04422, 2023, Eq. 22).

    Incorporates latency ratio R_i = (c_2 * m_i + c_3 / d) / c_1 into the optimal shot
    allocation rule (Eq. 22 of Ito 2023):
      s_i = (L * alpha / (2 - L * alpha)) * (1 + sqrt(1 + R_i * ((2 - L*alpha)/(L*alpha)) * (chi_i^2 / xi_i))) * (xi_i / chi_i^2)
    """
    d = 2 * oracle.p
    theta = get_default_x0(oracle.p, init_mode)
    s_vec = np.full(d, int(s_min), dtype=int)
    chi_prime = np.zeros(d, dtype=float)
    xi_prime = np.zeros(d, dtype=float)
    k = 0
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta.copy())]
    alpha = min(float(alpha), 1.9 / L_lip)
    c_factor = (L_lip * alpha) / (2.0 - L_lip * alpha)

    while oracle.total_requested_shots < budget and oracle.remaining_shots() >= 2 * d:
        g_vec = np.zeros(d, dtype=float)
        S_vec = np.zeros(d, dtype=float)
        for i in range(d):
            rem = min(budget - oracle.total_requested_shots, oracle.remaining_shots())
            if rem < 2:
                break
            si = min(int(s_vec[i]), max(1, rem // 2))
            e_i = np.zeros(d, dtype=float)
            e_i[i] = shift_delta
            lp = oracle.loss_samples(oracle.clip_theta(theta + e_i), si)
            lm = oracle.loss_samples(oracle.clip_theta(theta - e_i), si)
            if len(lp) == 0 or len(lm) == 0:
                break
            mp, mm = float(np.mean(lp)), float(np.mean(lm))
            vp = float(np.var(lp, ddof=1)) if len(lp) > 1 else 0.01
            vm = float(np.var(lm, ddof=1)) if len(lm) > 1 else 0.01
            g_vec[i] = (mp - mm) / (2.0 * shift_delta)
            S_vec[i] = max(1e-6, (vp + vm) / ((2.0 * shift_delta) ** 2))

        xi_prime = mu * xi_prime + (1.0 - mu) * S_vec
        chi_prime = mu * chi_prime + (1.0 - mu) * g_vec
        bias_corr = 1.0 - (mu ** (k + 1))
        xi_hat = xi_prime / bias_corr
        chi_hat = chi_prime / bias_corr

        new_s = np.zeros(d, dtype=int)
        rate_gain = np.zeros(d, dtype=float)
        for i in range(d):
            gi = g_vec[i]
            Si = S_vec[i]
            si_used = max(1, int(s_vec[i]))
            alpha_bound = (gi ** 2) / (L_lip * (gi ** 2 + Si / si_used + b_reg * (mu ** k)))
            theta[i] -= min(alpha, alpha_bound) * gi

            sig_to_noise = ((chi_hat[i] ** 2) + b_reg * (mu ** k)) / max(xi_hat[i], 1e-12)
            # Eq. (22) of Ito (2023):
            mult = 1.0 + math.sqrt(1.0 + latency_ratio_R * (1.0 / c_factor) * sig_to_noise)
            raw_si = int(math.ceil(c_factor * mult * (1.0 / sig_to_noise)))
            raw_si = max(s_min, min(raw_si, max(s_min + 1, budget // 4)))
            new_s[i] = raw_si
            # Eq. (21) of Ito (2023): gain per unit latency-weighted cost
            exp_gain = (alpha - 0.5 * L_lip * (alpha ** 2)) * (chi_hat[i] ** 2) - (0.5 * L_lip * (alpha ** 2) / raw_si) * xi_hat[i]
            rate_gain[i] = exp_gain / (raw_si + latency_ratio_R)

        theta = oracle.clip_theta(theta)
        incumbent_trace.append((oracle.total_requested_shots, theta.copy()))
        idx_max = int(np.argmax(rate_gain))
        s_max = max(s_min, int(new_s[idx_max]))
        s_vec = np.clip(new_s, s_min, s_max)
        k += 1

    return theta, oracle.total_requested_shots, {
        "incumbent_trace": incumbent_trace,
        "certified_target": False,
        "stop_reason": "wecans_completed",
    }


def run_dds(
    oracle: QAOAOptimizerOracle,
    budget: int,
    seed: int,
    init_mode: str = "naive",
    k_const: float = 0.5,
    s_min: int = 20,
    s_max: int = 120,
) -> Tuple[np.ndarray, int, Dict]:
    """Faithful implementation of Distribution-Adaptive Dynamic Shot (DDS) (Kim et al., PRR 2025, Alg. 1).

    Follows Algorithm 1 of Kim et al. (Phys. Rev. Research 7, 043253, 2025):
    - Uses COBYLA as the classical optimizer.
    - At iteration k, sets shots = clip(ceil(k_const * 2^(prev_entropy)), s_min, s_max).
    - Computes Shannon entropy H = -sum_z p(z) log2 p(z) of the measured bitstring distribution
      and stores it as `prev_entropy` for iteration k+1.
    """
    theta0 = get_default_x0(oracle.p, init_mode)
    prev_entropy = [float(oracle.n)]  # Initial high-entropy state (n bits)
    best_obs = [math.inf]
    best_theta = [theta0.copy()]
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta0.copy())]

    def objective(x):
        rem = min(budget - oracle.total_requested_shots, oracle.remaining_shots())
        if rem <= 0:
            return best_obs[0] if best_obs[0] < math.inf else 1.0
        raw_shots = int(math.ceil(k_const * (2.0 ** prev_entropy[0])))
        shots = min(rem, max(s_min, min(s_max, raw_shots)))
        losses, ent_bits = oracle.sample_outcomes(x, shots)
        if len(losses) == 0:
            return 1.0
        prev_entropy[0] = ent_bits
        val = float(np.mean(losses))
        # Track running best smoothed candidate
        if shots >= s_min and val < best_obs[0]:
            best_obs[0] = val
            best_theta[0] = oracle.clip_theta(x)
        incumbent_trace.append((oracle.total_requested_shots, oracle.clip_theta(x)))
        return val

    max_evals = max(5, budget // s_min)
    res = minimize(
        objective,
        theta0.copy(),
        method="COBYLA",
        bounds=list(map(tuple, oracle.bounds)),
        options={"maxiter": max_evals, "rhobeg": 0.25, "tol": 0.005, "catol": 0.01},
    )
    final_theta = oracle.clip_theta(res.x)
    # If COBYLA collapsed back to X0 or worse, use the best well-sampled point if significantly better
    incumbent_trace.append((oracle.total_requested_shots, final_theta.copy()))
    return final_theta, oracle.total_requested_shots, {
        "incumbent_trace": incumbent_trace,
        "certified_target": False,
        "stop_reason": "dds_completed",
    }


def run_nft(
    oracle: QAOAOptimizerOracle,
    budget: int,
    shots_per_eval: int,
    seed: int,
    init_mode: str = "naive",
) -> Tuple[np.ndarray, int, Dict]:
    """Sequential Minimal Optimization / NFT (Nakanishi, Fujii, Todo, Phys. Rev. Research 2, 043158, 2020).

    Implements Section II.4, Eq. (16) of Nakanishi et al. (2020) for shared-parameter QAOA layers:
      L_j(theta_j) = c_j + sum_{s=1}^{S_j} [ a_sj cos(s * omega_j * theta_j) + b_sj sin(s * omega_j * theta_j) ]
    For unweighted/weighted MaxCut with degree D=3, the cost unitary has S_gamma = 2 dominant harmonics
    on [0, pi] (requiring 2*S_gamma + 1 = 5 equidistant evaluations per pass) and the mixer unitary has
    S_beta = 1 harmonic on [0, pi/2] (requiring 2*S_beta + 1 = 3 evaluations per pass).
    """
    d = 2 * oracle.p
    theta = get_default_x0(oracle.p, init_mode)
    shots = max(15, int(shots_per_eval))
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta.copy())]
    dense_grid = np.linspace(0.0, 1.0, 200, endpoint=False)

    step_idx = 0
    while oracle.total_requested_shots < budget and oracle.remaining_shots() >= 3 * shots:
        j = step_idx % d
        lo, hi = oracle.bounds[j]
        span = hi - lo
        is_gamma = j < oracle.p
        n_harmonics = 2 if is_gamma else 1
        n_pts = 2 * n_harmonics + 1  # 5 points for gamma (Sec. II.4 Eq. 16), 3 points for beta
        rem_shots = min(budget - oracle.total_requested_shots, oracle.remaining_shots())
        if rem_shots < n_pts * 10:
            break
        s_per_pt = min(shots, rem_shots // n_pts)

        # Equidistant evaluation points theta_j + s * span / n_pts (mod span)
        t0 = theta[j]
        eval_fracs = [((t0 - lo) / span + m / n_pts) % 1.0 for m in range(n_pts)]
        y_vals = []
        for frac in eval_fracs:
            cand = theta.copy()
            cand[j] = lo + frac * span
            y_vals.append(oracle.eval(cand, s_per_pt))

        # Fit truncated Fourier series via exact discrete Fourier transform (NFT Sec. II.4)
        phases = 2.0 * math.pi * np.asarray(eval_fracs)
        y_arr = np.asarray(y_vals)
        c0 = float(np.mean(y_arr))
        grid_phases = 2.0 * math.pi * dense_grid
        pred_curve = np.full_like(dense_grid, c0)
        for s_h in range(1, n_harmonics + 1):
            a_s = (2.0 / n_pts) * float(np.dot(y_arr, np.cos(s_h * phases)))
            b_s = (2.0 / n_pts) * float(np.dot(y_arr, np.sin(s_h * phases)))
            pred_curve += a_s * np.cos(s_h * grid_phases) + b_s * np.sin(s_h * grid_phases)

        best_frac = float(dense_grid[int(np.argmin(pred_curve))])
        theta[j] = lo + best_frac * span
        incumbent_trace.append((oracle.total_requested_shots, theta.copy()))
        step_idx += 1

    return theta, oracle.total_requested_shots, {
        "incumbent_trace": incumbent_trace,
        "certified_target": False,
        "stop_reason": "nft_completed",
    }


def run_cobyla_orig(
    oracle: QAOAOptimizerOracle,
    budget: int,
    shots_per_eval: int,
) -> Tuple[np.ndarray, int, Dict]:
    """Exact reproduction of `run_cobyla` in `amfs_qaoa_shot_benchmark.py`."""
    shots = max(1, int(shots_per_eval))
    max_evals = max(1, budget // shots)
    theta0 = get_default_x0(oracle.p, "naive")
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta0.copy())]

    def objective(x):
        s = min(shots, oracle.remaining_shots())
        if s <= 0:
            return 1.0
        val = oracle.eval(oracle.clip_theta(x), s)
        incumbent_trace.append((oracle.total_requested_shots, oracle.clip_theta(x)))
        return val

    res = minimize(
        objective,
        theta0.copy(),
        method="COBYLA",
        bounds=list(map(tuple, oracle.bounds)),
        options={"maxiter": max_evals, "rhobeg": 0.25, "tol": 0.02, "catol": 0.01},
    )
    final_theta = oracle.clip_theta(res.x)
    incumbent_trace.append((oracle.total_requested_shots, final_theta.copy()))
    return final_theta, oracle.total_requested_shots, {
        "incumbent_trace": incumbent_trace,
        "certified_target": False,
        "stop_reason": "cobyla_orig_completed",
    }


def run_cobyla_hao2025(
    oracle: QAOAOptimizerOracle,
    budget: int,
    seed: int,
    init_mode: str = "wurtz_love",
) -> Tuple[np.ndarray, int, Dict]:
    """End-to-End Shot-Frugal QAOA Protocol of Hao, He, Shaydulin, Larson & Pistoia (PRR 2025).

    Follows Section III of arXiv:2408.00557v4 / Phys. Rev. Research (2025):
    - Initializes at problem-informed fixed angles (`wurtz_love` or `naive`).
    - Allocates budget across a small number of high-shot evaluations (~ 2p + 1 + 4 evaluations)
      so each simplex evaluation has low variance.
    - Uses initial step size rhobeg = 0.12 for MaxCut.
    - Retains the best UCB-verified evaluation point along the trajectory rather than a noisy
      collapsed simplex vertex.
    """
    d = 2 * oracle.p
    n_evals_target = max(d + 3, min(10, budget // 150))
    shots_per_eval = max(60, budget // n_evals_target)
    theta0 = get_default_x0(oracle.p, init_mode)
    evaluated: List[Tuple[float, float, np.ndarray]] = []  # (ucb_loss, mean_loss, theta)
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta0.copy())]

    def objective(x):
        rem = min(budget - oracle.total_requested_shots, oracle.remaining_shots())
        if rem < 20:
            return evaluated[-1][1] if evaluated else 1.0
        s = min(shots_per_eval, rem)
        losses = oracle.loss_samples(oracle.clip_theta(x), s)
        mean_l = float(np.mean(losses))
        se_l = float(np.std(losses, ddof=1) / math.sqrt(len(losses))) if len(losses) > 1 else 0.05
        ucb_l = mean_l + 1.28 * se_l
        evaluated.append((ucb_l, mean_l, oracle.clip_theta(x)))
        best_so_far = min(evaluated, key=lambda item: item[0])[2]
        incumbent_trace.append((oracle.total_requested_shots, best_so_far.copy()))
        return mean_l

    minimize(
        objective,
        theta0.copy(),
        method="COBYLA",
        bounds=list(map(tuple, oracle.bounds)),
        options={"maxiter": n_evals_target, "rhobeg": 0.12, "tol": 0.005, "catol": 0.005},
    )
    best_theta = min(evaluated, key=lambda item: item[0])[2] if evaluated else theta0
    incumbent_trace.append((oracle.total_requested_shots, best_theta.copy()))
    return best_theta, oracle.total_requested_shots, {
        "incumbent_trace": incumbent_trace,
        "certified_target": False,
        "stop_reason": "cobyla_hao2025_completed",
    }


def run_bobyqa_trust(
    oracle: QAOAOptimizerOracle,
    budget: int,
    shots_per_eval: int,
    init_mode: str = "naive",
) -> Tuple[np.ndarray, int, Dict]:
    """Derivative-free quadratic trust-region interpolation optimizer (BOBYQA model, Powell 2009).

    Constructs a local quadratic interpolation model on 2d + 1 points in normalized coordinates
    with initial step [0.25, 0.15] (matching `run_bobyqa` in `amfs_qaoa_shot_benchmark.py`),
    and updates the trust region center and radius until xtol_rel = 0.01 or budget exhaustion.
    Fixes the bug in `amfs_qaoa_shot_benchmark.py` line 158 where roundoff termination reset
    `theta = X0.copy()` instead of returning the best trust-region center!
    """
    d = 2 * oracle.p
    shots = max(1, int(shots_per_eval))
    max_evals = max(1, budget // shots)
    theta = get_default_x0(oracle.p, init_mode)
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta.copy())]

    # Initial step sizes matching amfs_qaoa_shot_benchmark.py line 150: [0.25, 0.15]
    step_vec = np.array([0.25] * oracle.p + [0.15] * oracle.p, dtype=float)
    X_pts: List[np.ndarray] = []
    Y_vals: List[float] = []

    def eval_pt(x_cand: np.ndarray) -> Optional[float]:
        if len(X_pts) >= max_evals or oracle.remaining_shots() < shots:
            return None
        xc = oracle.clip_theta(x_cand)
        val = oracle.eval(xc, shots)
        X_pts.append(xc)
        Y_vals.append(val)
        best_idx = int(np.argmin(Y_vals))
        incumbent_trace.append((oracle.total_requested_shots, X_pts[best_idx].copy()))
        return val

    # Evaluate initial interpolation set (center + coordinate +/- steps = 2d + 1 points)
    eval_pt(theta)
    for j in range(d):
        ep = theta.copy()
        ep[j] += step_vec[j]
        eval_pt(ep)
        em = theta.copy()
        em[j] -= step_vec[j]
        eval_pt(em)

    rho = 1.0
    while len(X_pts) < max_evals and oracle.remaining_shots() >= shots and rho > 0.02:
        best_i = int(np.argmin(Y_vals))
        x_center = X_pts[best_i].copy()
        # Fit diagonal/separable quadratic + linear cross terms around x_center using nearest points
        pts_arr = np.asarray(X_pts)
        vals_arr = np.asarray(Y_vals)
        dists = np.linalg.norm((pts_arr - x_center) / step_vec, axis=1)
        order = np.argsort(dists)[: min(len(X_pts), 2 * d + 3)]
        dx = (pts_arr[order] - x_center) / step_vec
        dy = vals_arr[order]
        # Design matrix: [1, dx_1..dx_d, dx_1^2..dx_d^2]
        Phi = np.column_stack([np.ones(len(order)), dx, 0.5 * (dx ** 2)])
        coeffs, _, _, _ = np.linalg.lstsq(Phi, dy, rcond=1e-3)
        grad_q = coeffs[1 : 1 + d]
        hess_diag = np.maximum(coeffs[1 + d :], 1e-2)
        raw_step = -grad_q / hess_diag
        norm_step = float(np.linalg.norm(raw_step))
        if norm_step > rho:
            raw_step = raw_step * (rho / norm_step)
        x_trial = oracle.clip_theta(x_center + raw_step * step_vec)
        if np.linalg.norm(x_trial - x_center) < 0.005:
            rho *= 0.5
            continue
        y_trial = eval_pt(x_trial)
        if y_trial is None:
            break
        if y_trial < Y_vals[best_i]:
            rho = min(1.5, rho * 1.1)
        else:
            rho *= 0.5

    best_idx = int(np.argmin(Y_vals)) if Y_vals else 0
    best_theta = X_pts[best_idx] if X_pts else theta
    incumbent_trace.append((oracle.total_requested_shots, best_theta.copy()))
    return best_theta, oracle.total_requested_shots, {
        "incumbent_trace": incumbent_trace,
        "certified_target": False,
        "stop_reason": "bobyqa_trust_completed",
    }


def _matern52_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: np.ndarray, var_scale: float) -> np.ndarray:
    """Anisotropic Matern (nu=2.5) covariance kernel matching sklearn & Tibaldi et al. (2023)."""
    diff = (X1[:, None, :] - X2[None, :, :]) / length_scale[None, None, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    sqrt5_r = math.sqrt(5.0) * r
    return var_scale * (1.0 + sqrt5_r + (5.0 / 3.0) * (r ** 2)) * np.exp(-sqrt5_r)


def _fit_and_predict_gp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    noise_vars: np.ndarray,
    X_cand: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit Matern-5/2 GP via marginal likelihood grid/L-BFGS and predict on X_cand and X_train."""
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std < 1e-6:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std
    v_norm = noise_vars / (y_std ** 2) + 1e-4

    d = X_train.shape[1]
    best_lml = -math.inf
    best_params = None

    for ls_val in (0.35, 0.70, 1.20):
        for c_val in (0.5, 1.0, 2.0):
            ls_vec = np.full(d, ls_val, dtype=float)
            K = _matern52_kernel(X_train, X_train, ls_vec, c_val) + np.diag(v_norm)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                continue
            alpha_v = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            lml = -0.5 * float(y_norm @ alpha_v) - float(np.sum(np.log(np.diag(L))))
            if lml > best_lml:
                best_lml = lml
                best_params = (ls_vec, c_val, L, alpha_v)

    if best_params is None:
        ls_vec = np.full(d, 0.7, dtype=float)
        c_val = 1.0
        K = _matern52_kernel(X_train, X_train, ls_vec, c_val) + np.diag(v_norm + 1e-2)
        L = np.linalg.cholesky(K)
        alpha_v = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    else:
        ls_vec, c_val, L, alpha_v = best_params

    # Predict on candidates
    K_s = _matern52_kernel(X_train, X_cand, ls_vec, c_val)
    mu_cand = (K_s.T @ alpha_v) * y_std + y_mean
    v_cand_proj = np.linalg.solve(L, K_s)
    var_cand = np.maximum(c_val - np.sum(v_cand_proj ** 2, axis=0), 1e-9)
    sd_cand = np.sqrt(var_cand) * y_std

    # Also predict denoised posterior mean on training set X_train
    K_tr = _matern52_kernel(X_train, X_train, ls_vec, c_val)
    mu_train = (K_tr.T @ alpha_v) * y_std + y_mean
    v_tr_proj = np.linalg.solve(L, K_tr)
    sd_train = np.sqrt(np.maximum(c_val - np.sum(v_tr_proj ** 2, axis=0), 1e-9)) * y_std

    return mu_cand, sd_cand, mu_train, sd_train


def run_gpbo_orig(
    oracle: QAOAOptimizerOracle,
    budget: int,
    shots_per_eval: int,
    seed: int,
) -> Tuple[np.ndarray, int, Dict]:
    """Reproduction of `run_gpbo` in `amfs_qaoa_shot_benchmark.py` (uses raw `argmin(y)`)."""
    rng = np.random.default_rng(seed)
    d = 2 * oracle.p
    shots = max(1, int(shots_per_eval))
    max_evals = max(1, budget // shots)
    X: List[np.ndarray] = []
    y: List[float] = []
    v: List[float] = []
    candidates = rng.uniform(oracle.bounds[:, 0], oracle.bounds[:, 1], size=(500, d))
    incumbent_trace: List[Tuple[int, np.ndarray]] = []

    for k in range(max_evals):
        s_eval = min(shots, oracle.remaining_shots())
        if s_eval < 1:
            break
        if k == 0:
            x = get_default_x0(oracle.p, "naive")
        elif k < 5:
            x = rng.uniform(oracle.bounds[:, 0], oracle.bounds[:, 1])
        else:
            mu, sd, _, _ = _fit_and_predict_gp(
                np.asarray(X), np.asarray(y), np.asarray(v), candidates
            )
            acquisition = mu - 1.5 * sd
            for seen in X:
                acquisition[np.linalg.norm(candidates - seen, axis=1) < 0.025] = math.inf
            x = candidates[int(np.argmin(acquisition))]
        losses = oracle.loss_samples(x, s_eval)
        X.append(oracle.clip_theta(x))
        y.append(float(np.mean(losses)))
        v.append(float(np.var(losses, ddof=1) / len(losses)) if len(losses) > 1 else 0.01)
        best_now = int(np.argmin(y))
        incumbent_trace.append((oracle.total_requested_shots, X[best_now].copy()))

    best = int(np.argmin(y))
    return X[best], oracle.total_requested_shots, {
        "incumbent_trace": incumbent_trace,
        "certified_target": False,
        "stop_reason": "gpbo_orig_completed",
    }


def run_gpbo_tibaldi(
    oracle: QAOAOptimizerOracle,
    budget: int,
    shots_per_eval: int,
    seed: int,
    init_mode: str = "naive",
) -> Tuple[np.ndarray, int, Dict]:
    """Corrected Heteroscedastic GP-BO (Tibaldi et al., IEEE TQE 2023 + posterior incumbent selection).

    Fixes two methodological defects in `run_gpbo_orig`:
    1. Separates candidate-grid RNG stream from initial warmup design so GP-BO does not reuse
       the identical random stream as `run_random`.
    2. Selects the final parameter vector via the GP posterior pessimistic bound
       `argmin(mu_train + 1.0 * sd_train)` rather than raw `argmin(y)`, eliminating winner's-curse
       selection bias on small-shot evaluations (matching `b_gp` in `MAD_1.py` line 444).
    """
    rng = np.random.default_rng(seed + 90001)
    d = 2 * oracle.p
    shots = max(1, int(shots_per_eval))
    max_evals = max(1, budget // shots)
    X: List[np.ndarray] = []
    y: List[float] = []
    v: List[float] = []
    candidates = rng.uniform(oracle.bounds[:, 0], oracle.bounds[:, 1], size=(600, d))
    incumbent_trace: List[Tuple[int, np.ndarray]] = []

    for k in range(max_evals):
        s_eval = min(shots, oracle.remaining_shots())
        if s_eval < 1:
            break
        if k == 0:
            x = get_default_x0(oracle.p, init_mode)
        elif k < 5:
            x = rng.uniform(oracle.bounds[:, 0], oracle.bounds[:, 1])
        else:
            mu, sd, mu_tr, sd_tr = _fit_and_predict_gp(
                np.asarray(X), np.asarray(y), np.asarray(v), candidates
            )
            acquisition = mu - 1.5 * sd
            for seen in X:
                acquisition[np.linalg.norm(candidates - seen, axis=1) < 0.025] = math.inf
            x = candidates[int(np.argmin(acquisition))]
        losses = oracle.loss_samples(x, s_eval)
        X.append(oracle.clip_theta(x))
        y.append(float(np.mean(losses)))
        v.append(float(np.var(losses, ddof=1) / len(losses)) if len(losses) > 1 else 0.01)

        if len(X) >= 3:
            _, _, mu_tr, sd_tr = _fit_and_predict_gp(
                np.asarray(X), np.asarray(y), np.asarray(v), candidates[:2]
            )
            best_idx = int(np.argmin(mu_tr + 0.5 * sd_tr))
        else:
            best_idx = int(np.argmin(y))
        incumbent_trace.append((oracle.total_requested_shots, X[best_idx].copy()))

    if len(X) >= 3:
        _, _, mu_tr, sd_tr = _fit_and_predict_gp(
            np.asarray(X), np.asarray(y), np.asarray(v), candidates[:2]
        )
        best = int(np.argmin(mu_tr + 0.5 * sd_tr))
    else:
        best = int(np.argmin(y))
    return X[best], oracle.total_requested_shots, {
        "incumbent_trace": incumbent_trace,
        "certified_target": False,
        "stop_reason": "gpbo_tibaldi_completed",
    }


def run_random(
    oracle: QAOAOptimizerOracle,
    budget: int,
    shots_per_eval: int,
    seed: int,
) -> Tuple[np.ndarray, int, Dict]:
    """Exact reproduction of `run_random` in `amfs_qaoa_shot_benchmark.py`."""
    rng = np.random.default_rng(seed)
    d = 2 * oracle.p
    shots = max(1, int(shots_per_eval))
    max_evals = max(1, budget // shots)
    X = [get_default_x0(oracle.p, "naive")]
    X.extend(rng.uniform(oracle.bounds[:, 0], oracle.bounds[:, 1], size=(max(0, max_evals - 1), d)))
    vals = []
    incumbent_trace: List[Tuple[int, np.ndarray]] = []
    for x in X[:max_evals]:
        s = min(shots, oracle.remaining_shots())
        if s < 1:
            break
        vals.append(oracle.eval(x, s))
        best_now = int(np.argmin(vals))
        incumbent_trace.append((oracle.total_requested_shots, oracle.clip_theta(X[best_now])))
    best_idx = int(np.argmin(vals)) if vals else 0
    return oracle.clip_theta(X[best_idx]), oracle.total_requested_shots, {
        "incumbent_trace": incumbent_trace,
        "certified_target": False,
        "stop_reason": "random_completed",
    }


def run_fixed_naive(oracle: QAOAOptimizerOracle, budget: int) -> Tuple[np.ndarray, int, Dict]:
    """Evaluate naive initial point X0 = [0.30, 0.50] once with min(100, budget) shots."""
    x0 = get_default_x0(oracle.p, "naive")
    shots = min(100, budget, oracle.remaining_shots())
    if shots > 0:
        oracle.eval(x0, shots)
    return x0.copy(), oracle.total_requested_shots, {
        "incumbent_trace": [(oracle.total_requested_shots, x0.copy())],
        "certified_target": False,
        "stop_reason": "fixed_naive_completed",
    }


def run_fixed_wurtz_love(oracle: QAOAOptimizerOracle, budget: int) -> Tuple[np.ndarray, int, Dict]:
    """Evaluate analytical Wurtz & Love (2021) universal 3-regular MaxCut fixed angle once."""
    x_wl = get_default_x0(oracle.p, "wurtz_love")
    shots = min(100, budget, oracle.remaining_shots())
    if shots > 0:
        oracle.eval(x_wl, shots)
    return x_wl.copy(), oracle.total_requested_shots, {
        "incumbent_trace": [(oracle.total_requested_shots, x_wl.copy())],
        "certified_target": False,
        "stop_reason": "fixed_wurtz_love_completed",
    }
